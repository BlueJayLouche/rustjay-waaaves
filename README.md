# RustJay Waaaves

A high-performance VJ (Visual Jockey) application written in Rust, ported from the OpenFrameworks-based "BLUEJAY_WAAAVES" project.

## Disclaimer

**This is NOT an official port of Gravity Waaaves.** The original creator, **Andrei Jay**, will not provide any support whatsoever for this software.

This source code is distributed AS-IS, there is no guaranteed continuing support.

## Special Thanks to Andrei Jay

This project would not exist without the incredible work of **Andrei Jay**, who created the original VSEJET GRAVITY_WAAAVES and generously open-sourced his work. His contributions to the video synthesis community through open-source projects, educational resources, and creative tools have enabled countless artists and developers to explore the world of analog-style video feedback and synthesis.

**Please support Andrei:**
- Patreon: https://www.patreon.com/c/andrei_jay
- Ko-fi: https://ko-fi.com/andreijay
- Website: https://videosynthecosphere.com
- Alternative Website: https://andreijaycreativecoding.com

You can download WPDSK and GW-DSK from his website - **please go support him!**

---

## Overview

RustJay Waaaves is a real-time video effects processor designed for live visual performance. It uses a modular shader pipeline with three processing blocks, providing extensive parameter control via an ImGui-based interface.

## Architecture

### Dual-Window Design

The application uses a dual-window architecture inspired by the original OpenFrameworks version:

- **Output Window**: wgpu-based rendering of the final visual output
- **Control Window**: ImGui-based interface for real-time parameter manipulation

### Shader Pipeline

The rendering pipeline consists of three main blocks:

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Block 1   │───▶│   Block 2   │───▶│   Block 3   │──▶ Output
│  (Channels) │    │  (Feedback) │    │   (Final)   │
└──────┬──────┘    └──────┬──────┘    └─────────────┘
       │                  │
       ▼                  ▼
   FB1 Delay          FB2 Delay
   (1-120 frames)     (1-120 frames)
```

#### Block 1: Channel Mixing
- Two input channels (CH1, CH2) with independent transforms
- Geometric effects: kaleidoscope, rotation, displacement, mirroring
- Color effects: HSB adjustment, posterization, inversion
- Filters: blur, sharpen with adjustable radii
- Feedback loop (FB1) with keying and temporal filtering

#### Block 2: Secondary Processing
- Processes secondary input or Block 1 output
- Same geometric and color effects as Block 1
- Independent feedback loop (FB2)

#### Block 3: Final Mixing
- Re-processes both Block 1 and Block 2 outputs
- Colorization with 5-band color mapping
- Matrix mixer for RGB channel manipulation
- Final compositing with keying

## Features

### Visual Effects

- **Geometric Transformations**
  - 2D displacement (X, Y, Z/scale)
  - Rotation with aspect ratio preservation modes
  - Kaleidoscope with adjustable segments and rotation
  - Horizontal/vertical mirror and flip
  - Shear transformation matrix

- **Color Processing**
  - HSB color space adjustments
  - Posterization with invert option
  - Color inversion (RGB, HSB channels)
  - Solarization effect
  - 5-band colorization

- **Filters**
  - Box blur with radius control
  - Sharpen with boost compensation
  - Temporal filtering for feedback smoothing

- **Compositing**
  - Multiple blend modes: lerp, add, difference, multiply, dodge
  - Chroma keying with adjustable threshold and softness
  - Overflow modes: clamp, wrap, fold
  - Matrix mixer for RGB channel routing

### Modulation

- 16 macro banks with assignable parameters
- LFO (Low Frequency Oscillator) per macro
  - Waveforms: sine, triangle, saw, square, random
  - Tempo sync to BPM
  - Adjustable rate and amplitude
- Audio reactivity via FFT analysis
- OSC control support
- **MIDI Learn/Mapping**
  - Map any parameter to MIDI CC, Note, or Pitch Bend
  - MIDI Learn mode for easy assignment
  - 14-bit CC support for high-resolution control
  - Range scaling (MIDI 0-127 → parameter min-max)
  - Persistent mappings saved to `midi_mappings.toml`
  - Multi-device support with hot-plugging

### Inputs

- **Webcam capture** - Direct camera input with device selection
- **NDI Input** - Network Device Interface for receiving video over IP
  - Automatic source discovery on local network
  - Low-latency frame receiving with background threading
  - Supports BGRA/BGRX formats with automatic conversion
  - Configurable bandwidth (highest quality by default)
- **Syphon Input** (macOS only) - Zero-copy GPU texture sharing from other applications
  - Automatic server discovery
  - UUID-based connection to avoid name conflicts
  - Native BGRA format, no conversion overhead
- Spout input (Windows only, planned)
- Video file playback (planned)

### Outputs

- **Full-screen wgpu output** - Hardware-accelerated rendering
- **NDI Output** - Broadcast video over the network as an NDI source
  - Async triple-buffered GPU readback for zero frame drops
  - Dedicated send thread for low-latency streaming
  - Configurable source name for easy discovery
  - BGRA/BGRX format support with alpha channel option
- **Syphon Output** (macOS only) - Share output as a Syphon server for other apps
  - Zero-copy GPU-to-GPU via IOSurface
  - Visible to Resolume, VDMX, MadMapper, and any Syphon-enabled app
- Video recording (planned)

## Building

### Prerequisites

- Rust 1.75+ (latest stable recommended)
- OpenGL 3.3 compatible GPU (or Metal via wgpu)
- macOS, Windows, or Linux
- Syphon.framework (macOS, bundled via `syphon-rs` — see below)

### Dependencies

Key dependencies:
- `wgpu 25.0` - Cross-platform GPU acceleration
- `winit 0.30` - Windowing
- `imgui 0.12` + `imgui-wgpu 0.25` - GUI framework
- `glam` - Fast linear algebra
- `cpal` - Audio I/O
- `midir` - Cross-platform MIDI input
- `serde` + `toml` - Configuration
- `grafton-ndi 0.11` - NDI input/output support

### Syphon Setup (macOS Only)

Syphon support is enabled by default. The build system finds the framework automatically at `../syphon-rs/syphon-lib/Syphon.framework` (the sibling `syphon-rs` repo).

If your layout differs, set `SYPHON_FRAMEWORK_DIR` before building:

```bash
SYPHON_FRAMEWORK_DIR=/path/to/syphon-rs/syphon-lib cargo build --release
```

To disable Syphon entirely:
```bash
cargo build --release --no-default-features --features "webcam ndi"
```

### Build Commands

```bash
# Debug build
cargo build

# Release build (optimized)
cargo build --release

# Run with logging
RUST_LOG=info cargo run
```

## Configuration

Configuration is stored in `config.toml`:

```toml
[output_window]
width = 1280
height = 720
fps = 60

[control_window]
width = 1920
height = 1080
fps = 30

[pipeline]
internal_width = 1280
internal_height = 720
max_delay_frames = 120

[control]
midi_enabled = true  # Set to false if running alongside a DAW
```

The file is automatically created with defaults on first run.

### MIDI Configuration

MIDI mappings are stored separately in `midi_mappings.toml`:

```toml
[[mappings]]
param_id = "block1.fb1_mix_amount"
device = "Akai APC40"
message_type = "CC"
channel = 1
controller = 16
min_value = 0.0
max_value = 1.0
```

## Usage

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `ESC` | Exit application |
| `F1` | Toggle fullscreen (output) |
| `F2` | Show/hide control window |
| `Space` | Clear feedback buffers |
| `1-9` | Select macro bank |

### Using NDI

#### NDI Input
1. Go to the **Inputs** tab
2. For Input 1 or Input 2, select "NDI" from the dropdown
3. Click "Refresh NDI Sources" to scan the network
4. Select your NDI source from the list
5. The source will automatically connect and start streaming

#### NDI Output
1. Go to the **Settings** tab
2. Enable "NDI Output" checkbox
3. Set your preferred NDI source name (default: "RustJay Waaaves")
4. The output will be broadcast to the network immediately
5. Use NDI Studio Monitor, OBS, or any NDI-enabled app to view it

### Parameter Control

Parameters are organized in tabs:
- **Block 1**: Primary channel mixing and feedback
- **Block 2**: Secondary processing
- **Block 3**: Final colorization and mixing
- **Macros**: LFO and modulation setup
- **Inputs**: Video/audio input configuration
- **Settings**: Output and recording options
- **MIDI**: MIDI mapping and device management

### Macro Assignment

1. Select a macro bank (0-15)
2. Click on a parameter while holding the macro's modifier key
3. Adjust the macro amount slider

### MIDI Mapping

#### Using MIDI Learn

1. Go to the **MIDI** tab
2. Click **"Enable MIDI Learn Mode"**
3. Click any parameter in the GUI (it will highlight green)
4. Move a knob/fader on your MIDI controller
5. The mapping is automatically created!

#### Manual Mapping

1. In the MIDI tab, expand **"Quick Map"**
2. Select a parameter from the dropdown
3. Choose a MIDI CC number
4. Click **"Create Mapping"**

#### Managing Mappings

- View all active mappings in the **"Active Mappings"** section
- Click **"X"** next to a mapping to delete it
- Click **"Clear All Mappings"** to remove all mappings
- Click **"Save Mappings"** to persist to `midi_mappings.toml`

#### Supported MIDI Messages

- **Control Change (CC)**: Standard 0-127 values
- **Note On/Off**: Velocity used as value (0-127)
- **Pitch Bend**: 14-bit resolution (-8192 to 8191)
- **Channel Aftertouch**: Pressure value (0-127)

## Performance Optimization

The implementation includes several performance optimizations:

- **Early-exit shaders**: Skip expensive operations when parameters are zero
- **Triple-buffered NDI output**: Async GPU→CPU readback with concurrent buffer processing
- **Dedicated send thread**: Non-blocking NDI frame transmission
- **Framebuffer pooling**: Reuse GPU memory allocations
- **Efficient uniforms**: Cache uniform locations, batch updates
- **Shader branch reduction**: Use mix() instead of conditionals where possible

### Benchmarks

Typical performance on modern hardware:
- 1920x1080 @ 60fps: <2ms GPU time per frame
- 4K @ 60fps: <5ms GPU time per frame

## Shader Porting Notes

The GL3 shaders were ported to WGSL with the following considerations:

1. **Uniform buffers**: Organized by functional groups, matching Rust struct layout
2. **Vec3 alignment**: Custom Vec3 type with 16-byte alignment to match WGSL
3. **Branch elimination**: Replaced if-statements with mix() operations
4. **Early exits**: Added threshold checks to skip expensive operations
5. **Precision**: Maintained float32 for compatibility with original

## Development

### Project Structure

```
src/
├── config/        # Configuration management (TOML)
├── core/          # Core types and shared state
├── engine/        # wgpu rendering engine
│   ├── pipelines/ # Shader pipelines (Block1, Block2, Block3)
│   ├── texture.rs # Texture utilities
│   └── mod.rs     # Main engine with dual-window support
├── gui/           # ImGui interface
├── input/         # Video input handling
│   ├── ndi.rs     # NDI input receiver
│   ├── webcam.rs  # Webcam capture
│   └── ...
├── output/        # Video output handling
│   ├── ndi_sender.rs   # NDI output sender
│   └── ndi_async.rs    # Async NDI output processor
├── params/        # Parameter structures
└── utils/         # Helper utilities
```

### Adding New Parameters

1. Add field to appropriate params struct in `src/params/mod.rs`
2. Add uniform to shader in WGSL pipeline files
3. Add UI control in `src/gui/mod.rs`
4. Add conversion in pipeline's `update_params` method

### Shader Development

Shaders use WGSL (WebGPU Shading Language). When modifying:

1. Maintain compatibility with wgpu
2. Test on both Metal (macOS) and Vulkan (Windows/Linux)
3. Profile with GPU analysis tools
4. Document any precision changes

## License

MIT License - See LICENSE file for details

## Credits

Ported from the OpenFrameworks "BLUEJAY_WAAAVES" project.
Original shaders and design concept by Andrei Jay.

## Roadmap

- [x] Basic wgpu rendering pipeline
- [x] ImGui control window
- [x] Dual-window architecture
- [x] MIDI learn/mapping system
- [ ] SPIR-V shader compilation for validation
- [x] NDI input/output
- [ ] Video file player
- [x] Audio analysis and reactivity
- [ ] TouchOSC integration
- [x] Preset system
- [x] Syphon input/output (macOS)

## Troubleshooting

### Black screen on startup
- Check GPU supports wgpu (Metal on macOS, Vulkan on Windows/Linux)
- Verify shaders compiled successfully (check logs)
- Try windowed mode first

### Low frame rate
- Reduce internal resolution
- Disable temporal filtering
- Lower delay buffer size

### Input not working
- Check device permissions (camera/mic)
- Verify NDI source is active and visible on the network
- Try different input resolution
- For NDI: Ensure firewall allows NDI traffic (ports 5960-5969 TCP/UDP)

### GUI not responding
- Check winit event loop is running
- Verify imgui context initialization

### MIDI not working / Application crashes with DAW

If you're running a DAW (Digital Audio Workstation) or other MIDI applications:

1. **Disable MIDI in config**: Edit `config.toml` and set:
   ```toml
   [control]
   midi_enabled = false
   ```

2. **Close other MIDI applications**: Some DAWs exclusively lock MIDI devices

3. **Check MIDI device permissions**: On macOS, check System Preferences → Security & Privacy

4. **Verify device connection**: The MIDI tab shows connected devices

5. **Check logs**: Run with `RUST_LOG=info cargo run` to see MIDI initialization messages

### NDI Issues

**NDI source not found:**
- Ensure NDI Runtime is installed (download from ndi.video)
- Check that source and receiver are on the same network/subnet
- Try using NDI Studio Monitor to verify source visibility

**NDI output not visible to other apps:**
- Check firewall settings - NDI uses ports 5960-5969
- Verify the NDI source name is unique
- Some NDI receivers require a few seconds to discover new sources

**Low frame rate with NDI output:**
- The async triple-buffered implementation minimizes overhead
- If dropping frames, check network bandwidth (1080p60 ~250Mbps)
- Consider lowering resolution or frame rate in Settings

### Syphon Issues

**"Library not loaded: Syphon.framework" at runtime:**
1. Verify the framework exists: `ls ../syphon-rs/syphon-lib/Syphon.framework`
2. Ensure you cloned `syphon-rs` as a sibling to this repo
3. If your layout differs: `SYPHON_FRAMEWORK_DIR=/path/to/syphon-rs/syphon-lib cargo build --release`

**Syphon server not appearing in other apps:**
- Check macOS Local Network permissions for the app
- Syphon uses Bonjour — ensure mDNS is not blocked by firewall
- Try SyphonInject or SyphonVirtualScreen to verify Syphon is working system-wide

**No Syphon servers appearing in the input list:**
- Click "Refresh" in the Inputs tab
- Ensure the sending application has Syphon output enabled
- Both apps must be running on the same machine

## Support

If you like this work, please consider supporting the original creator Andrei Jay through the links above.

For issues with this Rust port, please file a GitHub issue.
