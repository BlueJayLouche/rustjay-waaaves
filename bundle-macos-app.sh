#!/bin/bash
# General-purpose macOS app bundler (ad-hoc signed for local use)
#
# Usage:
#   ./bundle-macos-app.sh [options]
#
# Required:
#   -n, --name <name>             App name (e.g. "My App")
#   -b, --binary <path>           Path to the compiled binary
#   -i, --bundle-id <id>          Bundle identifier (e.g. "com.example.myapp")
#
# Optional:
#   -v, --version <version>       App version string (default: 0.1.0)
#   -o, --output <dir>            Output directory for the .app bundle (default: ./dist)
#   -m, --min-macos <version>     Minimum macOS version (default: 11.0)
#       --build-cmd <cmd>         Command to run before bundling (e.g. "cargo build --release")
#   -I, --icon <path>             Path to icon source image (png or jpg)
#   -a, --assets <dir>            Directory to copy into Contents/Resources
#   -f, --framework <path>        Path to a .framework to bundle (repeatable)
#   -d, --dylib <path>            Path to a .dylib to bundle (repeatable)
#       --camera <desc>           Request camera access; value is the usage description
#                                 shown in the macOS permission prompt
#       --microphone <desc>       Request microphone access; value is the usage description
#                                 shown in the macOS permission prompt
#       --entitlements <path>     Path to a .entitlements file to pass to codesign
#                                 (required for sandboxed apps; optional for ad-hoc)
#       --no-sign                 Skip code signing entirely
#       --help                    Show this help message
#
# Examples:
#   # Rust app with NDI and Syphon
#   ./bundle-macos-app.sh \
#     --name "My App" \
#     --binary target/release/my-app \
#     --bundle-id com.example.myapp \
#     --build-cmd "cargo build --release" \
#     --framework ../syphon-rs/syphon-lib/Syphon.framework \
#     --dylib "/Library/NDI SDK for Apple/lib/macOS/libndi.dylib" \
#     --assets assets
#
#   # Swift/C++ app, no extra frameworks
#   ./bundle-macos-app.sh \
#     --name "Other App" \
#     --binary build/OtherApp \
#     --bundle-id com.example.otherapp \
#     --version 1.2.0 \
#     --icon assets/icon.png

set -e

# ── Colours ──────────────────────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# ── Defaults ─────────────────────────────────────────────────────────────────

APP_VERSION="0.1.0"
OUTPUT_DIR="./dist"
MIN_MACOS="11.0"
BUILD_CMD=""
ICON_SOURCE=""
ASSETS_DIR=""
SIGN=true
FRAMEWORKS=()
DYLIBS=()
CAMERA_DESC=""
MICROPHONE_DESC=""
ENTITLEMENTS=""

# ── Argument parsing ──────────────────────────────────────────────────────────

usage() {
    sed -n '/^# Usage:/,/^[^#]/{ /^#/{ s/^# \{0,1\}//; p } }' "$0"
    exit 0
}

die() {
    echo -e "${RED}Error: $1${NC}" >&2
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)        usage ;;
        -n|--name)        APP_NAME="$2";       shift 2 ;;
        -b|--binary)      BINARY_PATH="$2";    shift 2 ;;
        -i|--bundle-id)   BUNDLE_ID="$2";      shift 2 ;;
        -v|--version)     APP_VERSION="$2";    shift 2 ;;
        -o|--output)      OUTPUT_DIR="$2";     shift 2 ;;
        -m|--min-macos)   MIN_MACOS="$2";      shift 2 ;;
           --build-cmd)   BUILD_CMD="$2";      shift 2 ;;
        -I|--icon)        ICON_SOURCE="$2";    shift 2 ;;
        -a|--assets)      ASSETS_DIR="$2";     shift 2 ;;
        -f|--framework)   FRAMEWORKS+=("$2");  shift 2 ;;
        -d|--dylib)       DYLIBS+=("$2");      shift 2 ;;
           --camera)      CAMERA_DESC="$2";    shift 2 ;;
           --microphone)  MICROPHONE_DESC="$2"; shift 2 ;;
           --entitlements) ENTITLEMENTS="$2";  shift 2 ;;
           --no-sign)     SIGN=false;          shift ;;
        *) die "Unknown option: $1. Run with --help for usage." ;;
    esac
done

# ── Validation ────────────────────────────────────────────────────────────────

[[ -z "$APP_NAME" ]]   && die "--name is required."
[[ -z "$BINARY_PATH" ]] && die "--binary is required."
[[ -z "$BUNDLE_ID" ]]  && die "--bundle-id is required."

# Validate frameworks and dylibs exist up front so we fail before building
for fw in "${FRAMEWORKS[@]}"; do
    [[ -d "$fw" ]] || die "Framework not found: $fw"
done
for lib in "${DYLIBS[@]}"; do
    [[ -f "$lib" ]] || die "Dylib not found: $lib"
done
if [[ -n "$ICON_SOURCE" ]]; then
    [[ -f "$ICON_SOURCE" ]] || die "Icon source not found: $ICON_SOURCE"
fi
if [[ -n "$ASSETS_DIR" ]]; then
    [[ -d "$ASSETS_DIR" ]] || die "Assets directory not found: $ASSETS_DIR"
fi
if [[ -n "$ENTITLEMENTS" ]]; then
    [[ -f "$ENTITLEMENTS" ]] || die "Entitlements file not found: $ENTITLEMENTS"
fi

# Derive a filesystem-safe name (replace spaces with hyphens)
APP_NAME_SAFE="${APP_NAME// /-}"
APP_DIR="${OUTPUT_DIR}/${APP_NAME_SAFE}.app"
BINARY_NAME="$(basename "$BINARY_PATH")"

# ── Helpers ───────────────────────────────────────────────────────────────────

step()    { echo -e "\n${CYAN}▶ $1${NC}"; }
ok()      { echo -e "${GREEN}  ✓ $1${NC}"; }
warn()    { echo -e "${YELLOW}  ⚠ $1${NC}"; }
fail()    { echo -e "${RED}  ✗ $1${NC}"; }

create_icns() {
    local source_image="$1"
    local output_name="$2"
    local iconset_dir=".tmp_icon.iconset"

    rm -rf "$iconset_dir"
    mkdir -p "$iconset_dir"

    local temp_png=".tmp_icon_source.png"
    if [[ "$source_image" == *.png ]] || [[ "$source_image" == *.PNG ]]; then
        temp_png="$source_image"
    else
        sips -s format png "$source_image" --out "$temp_png" 2>/dev/null
    fi

    sips -z 16   16   "$temp_png" --out "$iconset_dir/icon_16x16.png"      2>/dev/null
    sips -z 32   32   "$temp_png" --out "$iconset_dir/icon_16x16@2x.png"   2>/dev/null
    sips -z 32   32   "$temp_png" --out "$iconset_dir/icon_32x32.png"      2>/dev/null
    sips -z 64   64   "$temp_png" --out "$iconset_dir/icon_32x32@2x.png"   2>/dev/null
    sips -z 128  128  "$temp_png" --out "$iconset_dir/icon_128x128.png"    2>/dev/null
    sips -z 256  256  "$temp_png" --out "$iconset_dir/icon_128x128@2x.png" 2>/dev/null
    sips -z 256  256  "$temp_png" --out "$iconset_dir/icon_256x256.png"    2>/dev/null
    sips -z 512  512  "$temp_png" --out "$iconset_dir/icon_256x256@2x.png" 2>/dev/null
    sips -z 512  512  "$temp_png" --out "$iconset_dir/icon_512x512.png"    2>/dev/null
    sips -z 1024 1024 "$temp_png" --out "$iconset_dir/icon_512x512@2x.png" 2>/dev/null

    iconutil -c icns "$iconset_dir" -o "${output_name}.icns"
    local result=$?

    rm -rf "$iconset_dir"
    [[ "$temp_png" != "$source_image" ]] && rm -f "$temp_png"

    return $result
}

# ── Main build ────────────────────────────────────────────────────────────────

echo -e "${GREEN}"
echo "╔══════════════════════════════════════════╗"
echo "║       macOS App Bundler                  ║"
echo "╚══════════════════════════════════════════╝"
echo -e "${NC}"
echo "  App name   : $APP_NAME"
echo "  Bundle ID  : $BUNDLE_ID"
echo "  Version    : $APP_VERSION"
echo "  Binary     : $BINARY_PATH"
echo "  Output     : $APP_DIR"
echo "  Min macOS  : $MIN_MACOS"
[[ ${#FRAMEWORKS[@]} -gt 0 ]] && echo "  Frameworks : ${FRAMEWORKS[*]}"
[[ ${#DYLIBS[@]} -gt 0 ]]     && echo "  Dylibs     : ${DYLIBS[*]}"
[[ -n "$CAMERA_DESC" ]]       && echo "  Camera     : $CAMERA_DESC"
[[ -n "$MICROPHONE_DESC" ]]   && echo "  Microphone : $MICROPHONE_DESC"
[[ -n "$ENTITLEMENTS" ]]      && echo "  Entitlements: $ENTITLEMENTS"

# ── Step 1: Optional build command ───────────────────────────────────────────

if [[ -n "$BUILD_CMD" ]]; then
    step "Running build command: $BUILD_CMD"
    eval "$BUILD_CMD"
    ok "Build complete"
fi

# Re-check binary exists after building (may not have existed before build-cmd)
[[ -f "$BINARY_PATH" ]] || die "Binary not found after build: $BINARY_PATH"

# ── Step 2: Create bundle structure ──────────────────────────────────────────

step "Creating bundle structure"
rm -rf "$APP_DIR"
mkdir -p "$APP_DIR/Contents/MacOS"
mkdir -p "$APP_DIR/Contents/Resources"
mkdir -p "$APP_DIR/Contents/Frameworks"
ok "Created $APP_DIR"

# ── Step 3: Copy binary ───────────────────────────────────────────────────────

step "Copying binary"
cp "$BINARY_PATH" "$APP_DIR/Contents/MacOS/$BINARY_NAME"
chmod +x "$APP_DIR/Contents/MacOS/$BINARY_NAME"
ok "Copied $BINARY_NAME"

# ── Step 4: Copy assets ───────────────────────────────────────────────────────

if [[ -n "$ASSETS_DIR" ]]; then
    step "Copying assets"
    cp -r "$ASSETS_DIR" "$APP_DIR/Contents/Resources/"
    ok "Copied $ASSETS_DIR → Contents/Resources/"
fi

# ── Step 5: Process icon ──────────────────────────────────────────────────────

ICON_NAME=""
if [[ -n "$ICON_SOURCE" ]]; then
    step "Processing icon"
    if create_icns "$ICON_SOURCE" "$APP_NAME_SAFE"; then
        mv "${APP_NAME_SAFE}.icns" "$APP_DIR/Contents/Resources/"
        ICON_NAME="$APP_NAME_SAFE"
        ok "Created ${APP_NAME_SAFE}.icns"
    else
        warn "Failed to create .icns — continuing without icon"
    fi
fi

# ── Step 6: Write Info.plist ──────────────────────────────────────────────────

step "Writing Info.plist"

PLIST_ICON_ENTRY=""
if [[ -n "$ICON_NAME" ]]; then
    PLIST_ICON_ENTRY="    <key>CFBundleIconFile</key>
    <string>${ICON_NAME}</string>"
fi

PLIST_PERMISSIONS=""
if [[ -n "$CAMERA_DESC" ]]; then
    PLIST_PERMISSIONS+="    <key>NSCameraUsageDescription</key>
    <string>${CAMERA_DESC}</string>
"
fi
if [[ -n "$MICROPHONE_DESC" ]]; then
    PLIST_PERMISSIONS+="    <key>NSMicrophoneUsageDescription</key>
    <string>${MICROPHONE_DESC}</string>
"
fi

cat > "$APP_DIR/Contents/Info.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleExecutable</key>
    <string>${BINARY_NAME}</string>
    <key>CFBundleIdentifier</key>
    <string>${BUNDLE_ID}</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>${APP_NAME}</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>${APP_VERSION}</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>LSMinimumSystemVersion</key>
    <string>${MIN_MACOS}</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSRequiresAquaSystemAppearance</key>
    <false/>
${PLIST_PERMISSIONS}${PLIST_ICON_ENTRY}
</dict>
</plist>
EOF
ok "Info.plist written"

# ── Step 7: Bundle frameworks ─────────────────────────────────────────────────

if [[ ${#FRAMEWORKS[@]} -gt 0 ]]; then
    step "Bundling frameworks"
    for fw in "${FRAMEWORKS[@]}"; do
        fw_name="$(basename "$fw")"
        cp -R "$fw" "$APP_DIR/Contents/Frameworks/"
        ok "Copied $fw_name"
    done
fi

# ── Step 8: Bundle dylibs, fix install names and rpaths ──────────────────────

RPATH_ADDED=false

if [[ ${#DYLIBS[@]} -gt 0 ]]; then
    step "Bundling dylibs"

    # Register @rpath on the binary once, only if it isn't already present
    # (some toolchains e.g. Cargo add it at link time)
    if ! otool -l "$APP_DIR/Contents/MacOS/$BINARY_NAME" \
            | grep -q "@executable_path/../Frameworks"; then
        install_name_tool -add_rpath "@executable_path/../Frameworks" \
            "$APP_DIR/Contents/MacOS/$BINARY_NAME"
        RPATH_ADDED=true
    else
        ok "Binary already has @executable_path/../Frameworks rpath — skipping"
    fi

    for lib in "${DYLIBS[@]}"; do
        lib_name="$(basename "$lib")"
        cp "$lib" "$APP_DIR/Contents/Frameworks/"

        # Fix the dylib's own install name
        install_name_tool -id "@rpath/${lib_name}" \
            "$APP_DIR/Contents/Frameworks/${lib_name}"

        # Rewrite the reference baked into the binary
        install_name_tool -change "$lib" "@rpath/${lib_name}" \
            "$APP_DIR/Contents/MacOS/$BINARY_NAME"

        ok "Bundled $lib_name with @rpath"
    done

    # Verify before signing
    echo ""
    echo "  Library references in binary after patching:"
    otool -L "$APP_DIR/Contents/MacOS/$BINARY_NAME" | sed 's/^/    /'
fi

# ── Step 9: Sign (innermost to outermost) ─────────────────────────────────────

if $SIGN; then
    step "Signing (innermost → outermost)"

    for fw in "${FRAMEWORKS[@]}"; do
        fw_name="$(basename "$fw")"
        codesign --force --sign - "$APP_DIR/Contents/Frameworks/$fw_name"
        ok "Signed $fw_name"
    done

    for lib in "${DYLIBS[@]}"; do
        lib_name="$(basename "$lib")"
        codesign --force --sign - "$APP_DIR/Contents/Frameworks/$lib_name"
        ok "Signed $lib_name"
    done

    # Pass entitlements to the top-level bundle sign if provided
    CODESIGN_EXTRA_ARGS=()
    if [[ -n "$ENTITLEMENTS" ]]; then
        CODESIGN_EXTRA_ARGS+=(--entitlements "$ENTITLEMENTS")
    fi
    codesign --force --sign - "${CODESIGN_EXTRA_ARGS[@]}" "$APP_DIR"
    ok "Signed $APP_NAME_SAFE.app"
else
    warn "Skipping code signing (--no-sign)"
fi

# ── Summary ───────────────────────────────────────────────────────────────────

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════╗"
echo -e "║       Build complete ✓                   ║"
echo -e "╚══════════════════════════════════════════╝${NC}"
echo ""
echo "  Bundle: $APP_DIR"
echo ""

if [[ -n "$ICON_NAME" ]]; then
    ok "Icon: ${ICON_NAME}.icns"
else
    warn "Icon: not included"
fi

if [[ ${#FRAMEWORKS[@]} -gt 0 ]]; then
    for fw in "${FRAMEWORKS[@]}"; do
        ok "Framework: $(basename "$fw")"
    done
else
    warn "Frameworks: none"
fi

if [[ ${#DYLIBS[@]} -gt 0 ]]; then
    for lib in "${DYLIBS[@]}"; do
        ok "Dylib: $(basename "$lib") (bundled with @rpath)"
    done
else
    warn "Dylibs: none"
fi

$SIGN && ok "Signing: ad-hoc" || warn "Signing: skipped"
[[ -n "$ENTITLEMENTS" ]] && ok "Entitlements: $ENTITLEMENTS"

if [[ -n "$CAMERA_DESC" ]] || [[ -n "$MICROPHONE_DESC" ]]; then
    echo ""
    echo "  Permission prompts:"
    [[ -n "$CAMERA_DESC" ]]     && ok "Camera access declared"
    [[ -n "$MICROPHONE_DESC" ]] && ok "Microphone access declared"
fi

echo ""
echo "To run:"
echo "  open \"$APP_DIR\""
echo ""
echo "To share:"
echo "  zip -r \"${APP_NAME_SAFE}.zip\" \"$APP_DIR\""
echo "  (Recipients may need to right-click → Open on first launch)"
echo ""
