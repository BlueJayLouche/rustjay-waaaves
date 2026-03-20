# RustJay Waaaves - Agent Documentation

## Overview

This document provides technical details for AI agents and developers working on the RustJay Waaaves project.

### Quick Reference

- **[SIMPLE_ENGINE.md](./SIMPLE_ENGINE.md)** - Complete technical reference for the simplified feedback engine:
  - Architecture overview and dual-window system
  - Feedback loop design and tap tempo implementation
  - Texture format handling and shader pipeline
  - Common issues & troubleshooting guide
  - Extension guide for adding new features
  - **Use this as the primary reference for understanding the rendering system**

### Current Status (February 2025)

- ✅ GUI with comprehensive controls for all shader blocks
- ✅ Webcam input device selection and capture with auto-start
- ✅ **Block 1: Complete 3-stage modular architecture**
  - Stage 1: Input sampling with geometric transforms (CH1, CH2, FB1 positions)
  - Stage 2: Effects processing (HSB, blur, sharpen, filters)
  - Stage 3: Mixing with keying, FB1 transforms, and delay
- ✅ **FB1 Delay**: Ring buffer (0-120 frames / 2 seconds)
- ✅ **FB1 Geometric Transforms**: Scale, rotate, displace, kaleidoscope, mirrors
- ✅ LFO tabs with per-parameter modulation, tap tempo, and live visualization
- ✅ OpenFrameworks preset compatibility (import gwSaveStateXXX.json files)
- ✅ Input persistence between runs (saved to config.toml)
- ✅ **OSC Address Expose**: Hover tooltips showing OSC addresses for all parameters
- ✅ **UI Scale**: Discrete presets (100%, 150%, 200%, 250%, 300%) with runtime application
- ✅ **Global Shortcuts**: Shift+T (tap tempo), Shift+F (fullscreen toggle)
- ⚠️ Known Issue: OBS Virtual Camera not supported on macOS (use NDI instead)
- 🔄 In Progress: Block 2 and Block 3 parity with OF version

## Architecture

### Dual-Window System

The application uses winit's multi-window support to create two independent windows:

1. **Output Window** (`output_window`): wgpu-based rendering surface
   - Runs the shader pipeline (Block1 → Block2 → Block3)
   - Renders to a surface for display
   - Can be fullscreened for output

2. **Control Window** (`control_window`): ImGui interface
   - Separate wgpu device/surface (can use LowPower preference)
   - Renders ImGui UI for parameter control
   - Communicates via `SharedState` (Arc<Mutex<>>)

```rust
// In engine/mod.rs
struct App {
    output_window: Option<Arc<Window>>,
    output_engine: Option<WgpuEngine>,
    control_window: Option<Arc<Window>>,
    control_gui: Option<ControlGui>,
    control_renderer: Option<ControlWindowRenderer>,
}
```

### Shared State Communication

Both windows communicate through thread-safe shared state:

```rust
pub struct SharedState {
    pub block1: Block1Params,
    pub block2: Block2Params,
    pub block3: Block3Params,
    pub lfo_banks: Vec<LfoBank>,
    pub audio: AudioState,
    pub frame_count: u64,
    pub clear_feedback: bool,
    pub output_size: (u32, u32),
    pub internal_size: (u32, u32),
    pub is_recording: bool,
}
```

## Shader Pipeline

### Uniform Buffer Layout

WGSL structs must match Rust struct layouts exactly:

1. **Alignment**: Use `#[repr(C, align(16))]` for structs
2. **Vec3**: Custom 16-byte aligned type for WGSL `vec3<f32>`
3. **Padding**: Explicit padding fields where WGSL has them

Example:
```rust
#[repr(C, align(16))]
#[derive(Copy, Clone)]
pub struct Vec3([f32; 3]);  // 12 bytes, 16-byte aligned

#[repr(C, align(16))]
pub struct Block1Uniforms {
    pub width: f32,
    pub height: f32,
    pub inv_width: f32,
    pub inv_height: f32,
    pub ch1_hsb_attenuate: Vec3,  // Not Vec4!
    // ...
}
```

### Pipeline Flow

```
Input Textures
     │
     ▼
┌─────────────┐
│  Block 1    │ ← FB1 feedback
│  Pipeline   │ ← Uniforms from SharedState
└──────┬──────┘
       │ block1_texture
       ▼
┌─────────────┐
│  Block 2    │ ← FB2 feedback
│  Pipeline   │
└──────┬──────┘
       │ block2_texture
       ▼
┌─────────────┐
│  Block 3    │ ← Final mix
│  Pipeline   │ → Surface
└─────────────┘
```

## ImGui Integration

### Architecture

The control window uses `imgui-wgpu` for rendering:

```
Control Window
     │
     ▼
ImGuiRenderer (imgui_renderer.rs)
├── wgpu Surface (LowPower adapter)
├── imgui::Context
└── imgui_wgpu::Renderer
     │
     ▼
ControlGui::build_ui() ──▶ SharedState
```

### Rendering Flow

1. Control window gets `RedrawRequested` event
2. `ImGuiRenderer::render_frame()` called with UI builder closure
3. Closure calls `ControlGui::build_ui(ui)`
4. UI reads/writes `SharedState` via `Arc<Mutex<>>`
5. ImGui renders to wgpu surface

### Event Handling

Input events are passed to ImGui via `ImGuiRenderer::handle_event()`:

```rust
// In window_event handler
if let Some(ref mut renderer) = self.imgui_renderer {
    renderer.handle_event(&event, control_window);
}
```

### UI Structure

- **Block 1 Tab**: Channel 1, Channel 2, FB1
- **Block 2 Tab**: Block 2 Input, FB2
- **Block 3 Tab**: Block 1 Re-process, Block 2 Re-process, Matrix Mixer
- **Macros Tab**: LFO banks (0-15), parameter assignments
- **Inputs Tab**: Input 1/2 source selection, audio analysis
- **Settings Tab**: Display info, OSC/MIDI status

### Implementation Notes

- `render_frame()` takes a closure to avoid borrow issues with imgui context
- Display size updated each frame from window size
- Dark theme with colored section headers matches OF version
- The control window should not repaint continuously when idle. Current behavior is:
  - Fast redraw bursts while the user is actively interacting with the UI
  - Slow idle refresh when nothing is happening
  - This avoids starving the output window on the shared event loop / shared device

## Memory Layout Reference

### Vec3/WGSL vec3 Alignment

WGSL `vec3<f32>` has:
- Size: 12 bytes
- Alignment: 16 bytes (same as vec4)

Rust equivalent:
```rust
#[repr(C, align(16))]
struct Vec3([f32; 3]);  // 16 bytes total (12 + 4 padding)
```

### Block2Uniforms Layout

| Field | Offset | Size | WGSL Type |
|-------|--------|------|-----------|
| width | 0 | 4 | f32 |
| height | 4 | 4 | f32 |
| inv_width | 8 | 4 | f32 |
| inv_height | 12 | 4 | f32 |
| input_aspect | 16 | 4 | f32 |
| ... | ... | ... | ... |
| input_hsb_attenuate | 48 | 16 | vec3 |
| ... | ... | ... | ... |
| fb2_shear_matrix | 192 | 16 | vec4 |
| ... | ... | ... | ... |
| _padding | 336 | 112 | array |

Total: **448 bytes** (must match WGSL)

## Common Issues

### "Buffer size mismatch" Error

**Symptom**: `Buffer is bound with size X where the shader expects Y`

**Cause**: Rust struct size doesn't match WGSL struct size

**Fix**:
1. Check struct alignment: `#[repr(C, align(16))]`
2. Verify Vec3 fields use custom Vec3, not Vec4
3. Verify shear_matrix uses Vec4, not [f32; 4]
4. Check padding matches WGSL exactly

### Surface Format Mismatch

**Symptom**: `Render pipeline targets are incompatible with render pass`

**Cause**: Block3Pipeline uses wrong format for output surface

**Fix**: Pass surface_format from engine to Block3Pipeline::new()

```rust
let surface_format = surface_caps.formats[0];  // Bgra8UnormSrgb on macOS
let block3_pipeline = Block3Pipeline::new(&device, width, height, surface_format);
```

## Recent Fixes

### Output Window Occlusion Stall / Control Window Starvation

**Problem**: The app could become sluggish or appear to freeze when:
- The output window was hidden behind other windows
- The control window repainted too aggressively on the shared event loop

This showed up as very low effective output cadence in logs, for example `Frame 0` to `Frame 60`
taking several seconds instead of about one second at 60 FPS.

**Root Cause**:
1. The output window render path continued to drive the swapchain even while the output window was occluded
2. The control window shared the same device/queue and event loop, so continuous ImGui redraws could starve output rendering
3. Multiple startup discovery tasks and device scans made the slowdown easier to trigger during initialization

**Fix**:
1. **Occlusion handling for output window**
   - `WindowEvent::Occluded(bool)` is tracked for the output window
   - When occluded, output redraw requests are suspended
   - When visible again, redraw resumes immediately
   - This avoids hitting surface acquisition / presentation while the output window is hidden

2. **Event-driven control window redraw**
   - The control window no longer repaints continuously while idle
   - UI input marks the control window as "active" for a short burst of responsive redraws
   - Outside that burst, the control window falls back to a slow idle refresh
   - This keeps sliders and tabs usable without continuously stealing frame time from the output window

3. **Asynchronous discovery / reduced main-thread polling**
   - Webcam/audio/NDI/Syphon discovery was moved off the UI constructor / UI callbacks
   - MIDI hot-plug scanning now backs off instead of retrying aggressively on the main loop

**Where to look in code**:
- `src/engine/mod.rs`
  - Output occlusion tracking and redraw gating
  - Control window redraw scheduling (`control_needs_redraw`, active burst, idle refresh)
- `src/engine/imgui_renderer.rs`
  - Control surface presentation configuration
- `src/gui/mod.rs`
  - Async device/source discovery for UI lists
- `src/midi/input.rs`
  - MIDI scan backoff

**Current expected behavior**:
- Output window hidden: app should remain responsive, and output rendering should resume when the window becomes visible
- Control window idle: lower repaint rate
- Control window interaction: noticeably faster repaint rate during active editing

**Important tradeoff**:
- The control window is intentionally not updated at full rate while idle
- If the UI feels too sluggish during interaction, tune the "active burst" duration or idle refresh interval in `src/engine/mod.rs`

### Output Mode Radio Buttons Not Working

**Problem**: Output mode radio buttons in Settings tab were not responding to clicks.

**Cause**: Logic error where `mode` was re-initialized from shared state every frame, causing the change detection to fail.

**Fix**: Simplified the logic to capture current mode, render radio buttons, then compare after all buttons are rendered:
```rust
let mut selected_mode = /* get from shared state */;
let old_mode = selected_mode;
ui.radio_button("Block 1##out", &mut selected_mode, 0);
ui.radio_button("Block 2##out", &mut selected_mode, 1);
ui.radio_button("Block 3##out", &mut selected_mode, 2);
if selected_mode != old_mode {
    // Update shared state
}
```

### Block 1 Always Purple / Block 2 Always Black

**Problem**: Block 1 output was always purple, Block 2 was always black, even though Preview Input 1/2 showed camera correctly.

**Cause**: Debug code in Block 1 shader was forcing purple output:
```rust
// DEBUG: Force purple output to verify shader is executing
return vec4<f32>(0.8, 0.0, 0.8, 1.0);
```

Block 2 was black due to incorrect texture bindings (separate issue with input selection).

**Fix**: Removed debug return statement, restored proper output:
```rust
return final_color;
```

### Shader Performance Optimizations

**Problem**: Shaders were performing expensive HSB conversions even when no HSB processing was enabled.

**Solution**: Added early exit optimization in Block 1 and Block 2 shaders:
```rust
// Skip HSB conversion if no HSB operations needed
let needs_hsb = hsb_attenuate.x != 1.0 || hsb_attenuate.y != 1.0 || hsb_attenuate.z != 1.0 ||
                get_switch(switches, 4u) || get_switch(switches, 5u) || get_switch(switches, 6u) ||
                get_switch(switches, 8u); // solarize

var ch_rgb = ch_color.rgb;
if (needs_hsb) {
    var ch_hsb = rgb2hsb(ch_color.rgb);
    // ... HSB processing ...
    ch_rgb = hsb2rgb(ch_hsb);
}
```

### Global Tap Tempo with LFO Phase Reset

**Problem**: Tap tempo button existed but didn't reset LFO phases, and BPM wasn't being passed to the engine.

**Solution**: 
1. Added `bpm` field to `SharedState`
2. Modified `handle_tap_tempo()` to reset all LFO phases:
```rust
// Reset all LFO phases on every tap (global sync)
if let Ok(mut state) = self.shared_state.lock() {
    for lfo in &mut state.lfo_banks {
        lfo.phase = 0.0;
    }
}
```
3. Added auto-reset after 2 seconds of inactivity
4. Updated engine to use BPM from SharedState instead of hardcoded 120.0

### Modular Shader Architecture - IMPLEMENTED

**Status**: ✅ Implemented for Block 1 - Not bound by OpenFrameworks limitations

**Problem**: Current monolithic shaders are hard to debug and maintain
- Block 1 shader: ~1000 lines
- When output is black: which of 20 operations caused it?
- Can't visualize intermediate stages
- Always runs all operations even when not needed

**Solution**: Three-stage modular architecture per block - NOW IMPLEMENTED

#### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                         BLOCK 1                              │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ STAGE 1: INPUT SAMPLING                                 │ │
│  │  - Sample Input 1, Input 2, FB textures                │ │
│  │  - Apply coordinate transforms                        │ │
│  │  - Output: Transformed samples                        │ │
│  └────────────────────────────────────────────────────────┘ │
│                              ↓                               │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ STAGE 2: EFFECT PROCESSING (Optional)                   │ │
│  │  - HSB adjustments, blur, filters                      │ │
│  │  - SKIPPED if no effects enabled                      │ │
│  │  - Output: Processed colors                           │ │
│  └────────────────────────────────────────────────────────┘ │
│                              ↓                               │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ STAGE 3: MIXING & FEEDBACK                              │ │
│  │  - Mix Channel 1 + Channel 2 + Feedback               │ │
│  │  - Output: Final result                               │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

#### Benefits

1. **Debuggable** - Can view output of each stage independently
2. **Performant** - Skip Stage 2 entirely when no effects enabled
3. **Maintainable** - ~150 lines per stage vs ~1000 lines monolithic
4. **Testable** - Unit test each stage independently
5. **Better than OF** - Not limited by OpenFrameworks architecture

#### Implementation Plan

**Phase 1**: Stage 1 (Input Sampling) - DONE (shader written)
- Transform UVs and sample textures
- Support Input 1, Input 2, Feedback
- Can visualize sampled inputs directly

**Phase 2**: Stage 2 (Effects) - IN PROGRESS
- HSB processing with early exit
- Blur/sharpen filters
- Posterize, solarize
- Only runs if effects enabled

**Phase 3**: Stage 3 (Mixing)
- Combine all inputs
- Blend modes: Lerp, Add, Diff, Mult, Dodge
- Keying support
- Feedback mixing

**Phase 4**: Integration
- Chain stages with ping-pong buffers
- Debug visualization UI
- Performance profiling

See `ARCHITECTURE_PLAN.md` for full details.

### Linked Parameter Sliders

**Problem**: Multiple parameter sliders were linked - changing one would affect others (e.g., Block1 CH1 X Displace affected CH2 and FB1 X Displace).

**Cause**: ImGui uses widget labels as unique IDs. Multiple sliders with identical labels like "X Displace" shared the same ID and interfered with each other.

**Fix**: Added unique `##identifier` suffixes to all sliders:
- Block 1 Channel 1: `##ch1` suffix (X Displace##ch1, Y Displace##ch1, etc.)
- Block 1 Channel 2: `##ch2adj` suffix (X Displace##ch2adj, etc.)
- Block 1 FB1: `##fb1` suffix
- Block 2 Input: `##b2in` suffix
- Block 2 FB2: `##fb2` suffix
- Block 3 Block 1 Re-process: `##b3b1` suffix
- Block 3 Block 2 Re-process: `##b3b2` suffix

The `##` syntax creates an ID that doesn't appear in the visible label.

### Webcam Input Not Starting

**Problem**: Selecting "Webcam" as input type didn't start the camera.

**Cause**: The logic required BOTH `InputType::Webcam` AND a selected device index >= 0. When only the radio button was clicked (without selecting from the dropdown), `selected_webcam1 = -1`, so no StartWebcam request was sent.

**Fix**: Added auto-selection logic that picks the first available webcam device when Webcam type is selected but no device is chosen:
```rust
if type_changed && self.input1_type == InputType::Webcam 
    && self.selected_webcam1 < 0 && !self.webcam_devices.is_empty() {
    self.selected_webcam1 = 0;
}
```

### Debug Grid Colors

The shader pipelines show colored debug grids when no input signal is detected:
- **Block 1**: Blue grid (shows when no input textures available)
- **Block 2**: Green grid (shows when input AND feedback are both black)
- **Block 3**: Red grid (shows when no signal at final output)

**Note**: Block 2 typically shows Block 1's output (blue grid) instead of its own green grid because:
1. Block 2 defaults to using Block 1 as input (`block2_input_select = 0`)
2. Block 1's blue grid has brightness > 0.01, so Block 2 passes it through
3. The green grid only appears when input is completely black AND feedback is black

To see the green grid, set Block 2 to use Input 1 or Input 2 (not Block 1), and ensure that input has no camera data.

### Debug Logging

Comprehensive logging was added to diagnose input/camera issues:
- `[GUI]` logs for input change requests from the UI
- `[INPUT]` logs for request processing in the engine
- `[WEBCAM]` logs for camera capture thread activity
- Frame upload logging: `Uploading input X frame to GPU: WxH (N bytes)`

Enable with: `RUST_LOG=info cargo run`

## Building

### Without Webcam Support (No libclang required)

If you encounter build errors related to `v4l2-sys-mit` or `libclang`, you can build without webcam support:

```bash
cargo build --no-default-features
```

This disables the webcam input feature but keeps all other functionality (NDI, video files, shaders, etc.).

### With Webcam Support

Webcam support requires `libclang` for the `nokhwa` camera library. Install it using the provided script:

```bash
./scripts/install_dependencies.sh
```

Then build normally:

```bash
cargo build --release
```

## Development Tips

### Adding New Parameters

1. Add to params struct in `src/params/mod.rs`
2. Add to uniform struct in pipeline file
3. Add to WGSL shader struct
4. Add to `update_params()` conversion
5. Add to GUI in `src/gui/mod.rs`

### Testing Shader Changes

1. Edit WGSL in pipeline file (inline string)
2. `cargo run` to test
3. Check for validation errors in console

### Debugging Uniforms

Add size check in update_params:
```rust
println!("Uniforms size: {}", std::mem::size_of::<Block1Uniforms>());
```

## Dependencies

Key crate versions:
- `wgpu = "25.0"` - Updated for imgui-wgpu compatibility
- `winit = "0.30"`
- `imgui = "0.12"`
- `imgui-wgpu = "0.25"`
- `glam = "0.29"`

### wgpu 25 API Changes

Notable API changes from wgpu 22 to 25:
- `Instance::new()` takes `&InstanceDescriptor` instead of value
- `request_adapter()` returns `Result` instead of `Option`
- `request_device()` no longer takes `trace_path` parameter
- `DeviceDescriptor` has new `trace` field
- `entry_point` is now `Option<&str>` in pipeline descriptors

## Platform Notes

### macOS
- Uses Metal backend via wgpu
- Surface format is typically `Bgra8UnormSrgb`
- Control window shares device with output window

### Multi-Monitor Considerations

When dragging the output window to an external monitor:

1. **Surface Format Changes**: Different displays may prefer different formats
   - The `resize()` function detects format changes and recreates the Block3 pipeline
   - This ensures correct rendering on the new display

2. **Shared Device**: Both windows use the same wgpu device/queue
   - This is necessary to avoid Metal "device lost" errors
   - The adapter is selected based on the output window's initial surface

3. **Known Limitations**:
   - If external monitor uses a different GPU, the shared device may not be optimal
   - On macOS with dual-GPU systems, the adapter choice is made at startup

### Windows
- Uses Vulkan or DX12 backend
- May need different surface format handling

### Linux
- Uses Vulkan backend
- May require additional GPU drivers

## Future Enhancements

Planned architecture improvements:

1. **Compute Shaders**: For feedback delay buffers
2. **Async Compute**: Parallel pipeline stages
3. **HDR Output**: 16-bit float framebuffers
4. **Multi-GPU**: Split blocks across GPUs
5. **WebAssembly**: wasm32 target with WebGPU

## Block 1 Implementation Comparison: OF vs Rust

### Overview

The Rust implementation of Block 1 uses a **3-stage modular architecture** that differs significantly from the monolithic shader approach in the OpenFrameworks (OF) version. This section documents the differences for testing and verification.

### Architecture Comparison

| Aspect | OpenFrameworks Version | Rust Version |
|--------|------------------------|--------------|
| **Shader Structure** | Single monolithic shader (~960 lines) | 3 separate stages (Stage 1, 2, 3) |
| **Processing Flow** | All operations in one pass | Sequential ping-pong buffers |
| **Debuggability** | Hard to debug intermediate steps | Can view each stage output |
| **Performance** | Always runs all operations | Skips Stage 2 if no effects |
| **FB1 Delay** | Ring buffer (120 frames, 1-119 range) | Ring buffer (120 frames, 0-120 range) |
| **FB1 Transforms** | Applied in main shader | Applied in Stage 3 mixing |

### Detailed Differences

#### 1. **Feedback (FB1) Texture Binding**

**OF Version:**
```glsl
// FB1 texture passed as tex0
uniform sampler2D tex0; // fb2 for now (comment in OF code)
// In C++: shader.setUniformTexture("fb1Tex", *fbTex, 0);
```

**Rust Version:**
```rust
// Feedback texture from BlockResources
wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(self.resources.get_feedback_view()) }
// Delay buffer (optional)
wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(self.resources.get_delay_view(...)) }
```

**Key Difference:** Both have delay buffers. OF always uses delayed feedback (default 1 frame). Rust can select immediate (0) or delayed (1-120) in shader.

#### 2. **FB1 Geometric Transforms**

**OF Version (in shader):**
```glsl
vec2 fb1Coords = texCoordVarying * vec2(width, height);
// Mirrors, flips
if(fb1HMirror==1) { fb1Coords.x = abs(width - fb1Coords.x); }
if(fb1VMirror==1) { fb1Coords.y = abs(height - fb1Coords.y); }
// Kaleidoscope
fb1Coords = kaleidoscope(fb1Coords, fb1KaleidoscopeAmount, fb1KaleidoscopeSlice);
// Scale/Rotate
fb1Coords += fb1XYDisplace;
fb1Coords -= center;
fb1Coords *= fb1ZDisplace;
fb1Coords += center;
fb1Coords = rotate(fb1Coords, fb1Rotate, fb1RotateMode);
fb1Coords = shear(fb1Coords, fb1ShearMatrix);
```

**Rust Version (Stage 3 shader):**
```wgsl
fn sample_fb1(texcoord: vec2<f32>) -> vec3<f32> {
    var uv = texcoord;
    uv = apply_fb1_kaleidoscope(uv);
    uv = apply_fb1_mirrors(uv);
    uv = transform_fb1_uv(uv);  // Scale, rotate, displace
    
    if (uniforms.fb1_delay_time > 0) {
        return textureSample(delay_tex, delay_sampler, uv).rgb;
    } else {
        return textureSample(feedback_tex, feedback_sampler, uv).rgb;
    }
}
```

**Key Difference:** Rust applies transforms BEFORE sampling feedback; OF applied transforms to coordinates then sampled.

#### 3. **Channel Processing Order**

**OF Version:**
```glsl
// CH1 processed (blur/sharpen, HSB, etc.)
vec4 ch1Color = blurAndSharpen(ch1Tex, ...);
// CH2 processed similarly
vec4 ch2Color = blurAndSharpen(ch2Tex, ...);
// FB1 processed
vec4 fb1Color = blurAndSharpen(tex0, ...);  // tex0 is feedback
// Then mixed:
vec4 outColor = mixnKeyVideo(ch1Color, ch2Color, ...);
outColor = mixnKeyVideo(outColor, fb1Color, ...);
```

**Rust Version:**
```wgsl
// Stage 1: Sample inputs with transforms (CH1, CH2, FB positions)
// Stage 2: Apply effects to CH1 (blur, HSB, etc.)
// Stage 3: 
let input1 = textureSample(stage2_tex, ...);  // Processed CH1
let input2 = textureSample(input2_tex, ...);  // Processed CH2
let feedback = sample_fb1(texcoord);          // FB with transforms
// Mix CH1 + CH2
result = mix_colors(input1, masked_input2, ...);
// Mix with FB1
result = mix_colors(result, processed_feedback, ...);
```

**Key Difference:** 
- OF processes CH1, CH2, and FB1 effects in parallel then mixes
- Rust processes CH1 through Stage 1→2, CH2 separately, then mixes in Stage 3
- OF applies blur/sharpen to feedback; Rust currently does NOT apply filters to feedback (only color adjustments)

#### 4. **Delay Implementation**

**OF Version:**
```cpp
// DelayBuffer class with 120 frame ring buffer
class DelayBuffer {
    static constexpr int MAX_FRAMES = 120;
    std::array<ofFbo, MAX_FRAMES> frames;
    int writeIndex = 0;
    
    ofTexture& getFrame(int delay) {
        // delay: 0 = most recent, 1 = 1 frame ago, etc.
        int readIndex = (writeIndex - 1 - delay + MAX_FRAMES) % MAX_FRAMES;
        return frames[readIndex].getTexture();
    }
};

// Usage in PipelineManager:
fb1DelayTime = 1; // Default: 1 frame delay (range: 1-119)
ofTexture& fb1Tex = fb1Delay.getFrame(fb1DelayTime);
block1.setFeedbackTexture(fb1Tex);  // Delayed frame passed as feedback
// ... render block1 ...
fb1Delay.pushFrame(block1.getOutput());  // Store output for next frames
```

**Rust Version:**
```rust
// Ring buffer of 120 frames
pub delay_buffers: Vec<Texture>,
pub delay_write_index: usize,

// Get delayed frame for reading:
pub fn get_delay_view(&self, delay_time: usize) -> &wgpu::TextureView {
    let delay_frames = delay_time.min(self.max_delay_frames).max(1);
    let read_index = if self.delay_write_index >= delay_frames {
        self.delay_write_index - delay_frames
    } else {
        self.max_delay_frames - (delay_frames - self.delay_write_index)
    };
    &self.delay_buffers[read_index].view
}

// In shader (Stage 3):
if (uniforms.fb1_delay_time > 0) {
    // Sample from delay buffer
    feedback = textureSample(delay_tex, delay_sampler, transformed_uv).rgb;
} else {
    // Sample from immediate feedback (previous frame output)
    feedback = textureSample(feedback_tex, feedback_sampler, transformed_uv).rgb;
}
```

**Key Differences:**
- **OF**: Delay is applied CPU-side; shader always receives delayed texture. Range 1-119 frames.
- **Rust**: Both immediate and delayed textures passed to shader; selection happens in shader. Range 0-120 frames (0 = immediate).
- **Both**: Use 120-frame ring buffer, support ~2 seconds of delay at 60fps.

#### 5. **FB1 Filters (Blur/Sharpen)**

**OF Version:**
```glsl
// blurAndSharpen function applied to feedback texture
vec4 fb1Color = blurAndSharpen(tex0, fb1Coords/vec2(width,height), 
    fb1SharpenAmount, fb1SharpenRadius, fb1FiltersBoost,
    fb1BlurRadius, fb1BlurAmount);
```

**Rust Version:**
```wgsl
// Blur function (separate from sharpen)
fn fb1_blur(uv: vec2<f32>, amount: f32, radius: f32) -> vec3<f32> {
    // 9-sample box blur with weighted samples
}

// Sharpen function with boost
fn fb1_sharpen(uv: vec2<f32>, amount: f32, radius: f32, color: vec3<f32>) -> vec3<f32> {
    // Laplacian sharpening with filters_boost
}

// In sample_fb1():
var color = textureSample(...);
if (uniforms.fb1_blur_amount > 0.001) {
    color = fb1_blur(uv, uniforms.fb1_blur_amount, uniforms.fb1_blur_radius);
}
if (uniforms.fb1_sharpen_amount > 0.001) {
    color = fb1_sharpen(uv, uniforms.fb1_sharpen_amount, uniforms.fb1_sharpen_radius, color);
}
```

#### 6. **Temporal Filters**

**OF Version:**
```glsl
vec4 temporalFilter1Color = texture(fb1TemporalFilter, texCoordVarying);
// Applied to output after mixing
outColor = mix(outColor, temporalFilter1Color, fb1TemporalFilter1Amount);
```

**Rust Version:** Parameters exist but not yet implemented (requires additional texture binding for temporal filter frame)

#### 7. **FB1 Color Processing**

| Feature | OF | Rust |
|---------|-----|------|
| HSB Offset | ✅ | ✅ |
| HSB Attenuate | ✅ | ✅ |
| HSB Powmap | ✅ | ✅ |
| Hue Shaper | ✅ | ✅ |
| Hue Invert | ✅ | ✅ |
| Saturation Invert | ✅ | ✅ |
| Bright Invert | ✅ | ✅ |
| RGB Posterize | ✅ | ✅ |
| HSB Posterize | ✅ | ✅ |

**OF Hue Shaper:**
```glsl
float hueShaper(float inHue, float shaper) {
    inHue = fract(abs(inHue + shaper * sin(inHue * 0.3184713)));
    return inHue;
}
```

**Rust Implementation:** Matches OF exactly

#### 8. **Keying Implementation**

**OF Version:**
```glsl
// Full mixing with keying in mixnKeyVideo function
vec4 mixnKeyVideo(vec4 fg, vec4 bg, float mixAmount, int mixType, 
    float keyThreshold, float keySoft, vec3 keyValue, int keyOrder, 
    int mixOverflow, vec4 lumaKeyColor, int lumaKeyOn) {
    
    // Key order: 1 = swap fg/bg (key first, then mix)
    if(keyOrder==1){
        vec4 dummy=fg; fg=bg; bg=dummy;
    }
    
    // Perform mix...
    
    // Apply keying after mixing
    float chromaDistance=distance(keyValue,fg.rgb);
    if(chromaDistance < keyThreshold){
        // Mix in bg based on keySoft and distance
        outColor=mix(bg,outColor,keySoft*abs(1.0-(chromaDistance-keyThreshold)));
    }
}

// Key parameters:
// - Mode: 0=lumakey, 1=chromakey (always active, no "off")
// - Key Value: -1.0 to 1.0 (for both luma and chroma)
// - Key Threshold: -1.0 to 1.0 (default 1.0 = essentially off)
// - Key Soft: -1.0 to 1.0
```

**Rust Version (Updated):**
```wgsl
// OF-style integrated keying
fn mixn_key_video(fg: vec3<f32>, bg: vec3<f32>, amount: f32, mix_type: i32, 
                 overflow: i32, key_order: i32) -> vec3<f32> {
    // Key order: 1 = swap fg/bg
    if (key_order == 1) { swap fg and bg }
    
    // Perform mix
    mixed = apply_mix(fg, bg, amount, mix_type);
    
    // Apply keying after mixing (OF style)
    let key_mix = calculate_key_mix(fg);
    if (key_mix > 0.0) {
        mixed = mix(mixed, bg, key_mix);
    }
    return mixed;
}

fn calculate_key_mix(fg: vec3<f32>) -> f32 {
    // Mode 0 = Lumakey, 1 = Chromakey
    if (uniforms.key_mode == 0) {
        chroma_distance = abs(luma(fg) - luma(key_value));
    } else {
        chroma_distance = distance(fg, key_value);
    }
    
    // OF-style soft keying
    if (chroma_distance < key_threshold) {
        return key_soft * abs(1.0 - (chroma_distance - key_threshold));
    }
    return 0.0;
}
```

**Key Changes (Feb 2025):**
- Key mode now matches OF: 0=lumakey, 1=chromakey (no "off" state)
- Key value range: -1.0 to 1.0 (was 0.0 to 1.0)
- Key threshold/soft range: -1.0 to 1.0 (was 0.0 to 1.0)
- Keying integrated into mix operation (not pre-applied as mask)
- Default threshold=1.0 means keying is effectively off until adjusted

### Testing Checklist

When testing Block 1, verify these behaviors match between OF and Rust:

#### FB1 Geometric Transforms
- [ ] **X/Y Displace**: Feedback shifts horizontally/vertically
- [ ] **Z Displace (Scale)**: Feedback zooms in/out from center
- [ ] **Rotate**: Feedback rotates around center
- [ ] **Kaleidoscope**: Creates symmetrical patterns
- [ ] **H/V Mirror**: Creates mirrored reflections
- [ ] **Shear Matrix**: Skews the feedback (OF has this, Rust recently added)

#### FB1 Color Adjustments
- [ ] **HSB Offset**: Shifts hue/saturation/brightness
- [ ] **HSB Attenuate**: Multiplies HSB values
- [ ] **HSB Powmap**: Power curve on HSB (OF has this, Rust recently added)
- [ ] **Posterize**: Reduces color levels
- [ ] **Hue Shaper**: Special hue shaping function

#### FB1 Filters
- [ ] **Blur**: Feedback appears blurred
- [ ] **Sharpen**: Feedback appears sharpened
- [ ] **Temporal Filter**: Blends with previous frames (OF only)

#### Delay (New in Rust)
- [ ] **Delay Time**: 0 = immediate, 1-120 = frames of delay
- [ ] **Delay + Mix**: Delayed feedback mixes correctly with live input

#### Mixing
- [ ] **Mix Amount**: 0 = no FB1, 1.0 = full FB1
- [ ] **Mix Types**: Lerp, Add, Diff, Mult, Dodge
- [ ] **Key Order**: Key→Mix vs Mix→Key produces different results

### Known Differences (Expected)

1. **Performance**: Rust skips Stage 2 when no effects enabled; OF always runs blur/sharpen
2. **Shader Complexity**: Rust stages are ~150 lines each; OF shader is ~960 lines
3. **Debug Views**: Rust can show Stage1/Stage2/Stage3 separately; OF cannot
4. **Delay**: Rust has it; OF does not
5. **Temporal Filters**: OF has them; Rust does not yet

### Known Issues to Test

1. **FB1 Rotate Direction**: Verify rotation direction matches (CW vs CCW)
2. **Scale Center**: Verify zoom happens from center, not corner
3. **Mirror Edge Cases**: Test mirrors at exactly 0.5 coordinates
4. **Delay Wrap-around**: Test delay buffer doesn't glitch at wrap point
5. **Overflow Modes**: Test wrap/clamp/mirror for geometric overflow

## New Features (February 2025)

### Preset System

A full preset management system has been implemented following the original OF app's design:

```rust
// Preset data structure (src/params/preset.rs)
pub struct PresetData {
    pub block1: Block1Params,
    pub block2: Block2Params,
    pub block3: Block3Params,
    pub block1_modulations: HashMap<String, ParamModulationData>,
    pub block2_modulations: HashMap<String, ParamModulationData>,
    pub block3_modulations: HashMap<String, ParamModulationData>,
    pub tempo: PresetTempoData,
}
```

**Features:**
- **Bank Management**: Presets are organized in banks (folders under `presets/`)
- **Save/Load**: Full parameter state including audio modulations
- **Auto-naming**: Presets saved as `{name}.json` in current bank
- **Default Bank**: Created automatically if no banks exist

**GUI Integration:**
- Bank selector dropdown in preset section
- Preset name input with Save button
- Load preset dropdown with Load/Delete buttons
- Status messages for user feedback

### Audio Reactivity System

Audio modulation can now be applied to parameters with full tracking:

```rust
// In SharedState (src/core/mod.rs)
pub struct SharedState {
    // ... other fields ...
    pub block1_modulations: HashMap<String, ParamModulationData>,
    pub block2_modulations: HashMap<String, ParamModulationData>,
    pub block3_modulations: HashMap<String, ParamModulationData>,
}
```

**GUI Features:**
1. **Parameter Selector**: Dropdown of all available parameters per block
2. **FFT Band Selector**: Choose which frequency band (0-7) to use
3. **Modulation Amount**: Slider for modulation depth
4. **Apply Button**: Adds the modulation to the active list
5. **Active Modulations List**: Shows all applied modulations with:
   - Parameter name
   - Modulation type (Audio/BPM) and settings
   - Remove button for each

**Usage:**
1. Open Audio Reactivity panel (button at bottom of Block tabs)
2. Select a parameter from the dropdown
3. Enable "Audio Mod" checkbox
4. Choose FFT band (e.g., Bass for kick drum response)
5. Set modulation amount
6. Click "Apply Modulation"
7. See applied modulation in the list below

### Block1 Blue Grid Fix

The blue debug grid that was interfering with camera input has been removed:

**Previous Issue:**
- Shader showed blue grid when output brightness < 0.01
- This was overriding valid camera textures with low brightness

**Fix:**
- Removed the debug grid code from block1 shader
- Camera input now displays normally without interference

## Pop-Out Tabs (February 2025)

The control surface now supports modular pop-out tabs, allowing you to arrange the interface for your workflow.

### Features

- **Right-click any main tab** to open context menu
- **Pop Out** - Opens tab in floating ImGui window
- **Dock** - Returns tab to main window
- **Visual indicator** (⧉) shows which tabs are floating
- **Position and size persisted** to `layout.toml`

### Supported Tabs

Main tabs that can be popped out:
- Block 1, Block 2, Block 3
- Macros
- Inputs  
- Settings

### Layout Persistence

Window layout is automatically saved to `layout.toml`:
```toml
[popped_tabs.Block1]
pos_x = 100.0
pos_y = 150.0
width = 500.0
height = 600.0
collapsed = false
```

Manual controls in **Settings → Window Layout**:
- **Save Layout** - Manually save current layout
- **Reset Layout** - Dock all floating tabs

### Usage Tips

1. Pop out tabs you use frequently (e.g., keep Block 1 and Macros visible)
2. Arrange on second monitor for live performance
3. Resize windows to show only needed controls
4. Layout restores automatically on app restart

## OSC Address Expose (February 2025)

The application now supports OSC (Open Sound Control) address expose via hover tooltips. This feature helps users identify the OSC address for each parameter when integrating with external controllers.

### Enabling OSC Address Display

1. Go to **Settings** tab
2. Check **"Show OSC Addresses on Hover"**
3. Hover over any parameter to see its OSC address

### OSC Address Scheme

```
/block1/ch1/{param}        # Channel 1 parameters
/block1/ch2/{param}        # Channel 2 parameters
/block1/fb1/{param}        # FB1 feedback parameters
/block2/input/{param}      # Block 2 input parameters
/block2/fb2/{param}        # FB2 feedback parameters
/block3/b1/{param}         # Block 1 re-process parameters
/block3/b2/{param}         # Block 2 re-process parameters
/block3/matrix/{param}     # Matrix mixer parameters
/block3/final/{param}      # Final mix parameters
/global/bpm                # Global tempo BPM
/global/tap_tempo          # Tap tempo trigger
```

### Example OSC Addresses

**Block 1 - Channel 1:**
- `/block1/ch1/x_displace` - X displacement
- `/block1/ch1/y_displace` - Y displacement
- `/block1/ch1/rotate` - Rotation
- `/block1/ch1/blur_amount` - Blur amount

**Block 1 - Channel 2:**
- `/block1/ch2/mix_amount` - Mix amount
- `/block1/ch2/key_threshold` - Key threshold
- `/block1/ch2/x_displace` - X displacement

**Block 1 - FB1:**
- `/block1/fb1/mix_amount` - Feedback mix
- `/block1/fb1/x_displace` - X displacement
- `/block1/fb1/delay_time` - Delay time (frames)
- `/block1/fb1/hsb_offset` - HSB offset

**Block 2:**
- `/block2/input/x_displace` - Input X displacement
- `/block2/fb2/mix_amount` - FB2 mix amount
- `/block2/fb2/blur_amount` - FB2 blur

**Global:**
- `/global/bpm` - Current BPM (20-300)
- `/global/tap_tempo` - Tap tempo (trigger)

### Implementation Notes

- OSC addresses are shown in tooltips when hovering over parameter controls
- Current parameter values are displayed alongside addresses
- The feature is controlled by `AppConfig::show_osc_addresses`
- Tooltips use ImGui's `is_item_hovered()` for detection
- Address scheme follows logical hierarchy: `/block{num}/{section}/{param}`

## GUI Structure (Updated February 2025)

The GUI has been completely reorganized to match the reference OF app:

### Main Tabs
- **Block 1** (Orange) - Channel mixing and FB1 feedback
  - Sub-tabs: CH1 Adjust, CH2 Mix/Key, CH2 Adjust, FB1 Parameters, LFO
- **Block 2** (Green) - Secondary processing and FB2
  - Sub-tabs: Input Adjust, FB2 Parameters, LFO
- **Block 3** (Pink) - Final mixing and re-processing
  - Sub-tabs: Block 1 Re-Pro, Block 2 Re-Pro, Matrix Mixer, Final Mix, LFO
- **Macros** (Purple) - 16 macro banks with LFO controls
- **Inputs** (Cyan) - Input source configuration
- **Settings** (Gray) - Output mode and display info

### Collapsible Sections
Each panel uses ImGui collapsing headers for elegant organization:
```rust
if ui.collapsing_header("Section Name", imgui::TreeNodeFlags::empty()) {
    ui.indent();
    // Controls here
    ui.unindent();
}
```

### Webcam Input Flow
1. Select "Webcam" from Input Type dropdown
2. Device selection dropdown appears with available cameras
3. Auto-selects first device when switching to Webcam type
4. Click "Start Webcam" to activate
5. "Refresh Device List" button to re-scan devices

### LFO Tabs (New February 2025)

Each block now has an LFO tab for per-parameter modulation:

**Block 1 LFO Tab:**
- Channel 1: X/Y/Z Displace, Rotate, Kaleidoscope, Blur, Sharpen
- Channel 2: X/Y Displace, Rotate, Mix Amount
- FB1: X/Y Displace, Rotate, Mix Amount

**Block 2 LFO Tab:**
- Input: X/Y Displace, Rotate, Blur
- FB2: X/Y Displace, Rotate, Mix Amount

**Block 3 LFO Tab:**
- Block 1 Re-process: X/Y Displace, Rotate
- Block 2 Re-process: X/Y Displace, Rotate
- Final Mix: Mix Amount

**LFO Controls per Parameter:**
- **Enable**: Toggle LFO on/off for this parameter
- **Amplitude**: How much the LFO modulates the parameter (0-1)
- **Tempo Sync**: When enabled, rate is controlled by beat division
- **Rate/Beat Division**: Free rate (Hz) or tempo-synced division (1/16 to 8 beats)
- **Waveform**: Sine, Triangle, Ramp, Saw, Square

**Global Tempo Control:**
- **BPM**: Manual BPM entry (20-300)
- **TAP TEMPO**: Click repeatedly to set tempo (flashes on beat)
- **PLAY/PAUSE**: Start/stop LFO modulation
- **SYNC**: Enable/disable tempo sync globally

### OpenFrameworks Preset Compatibility (New February 2025)

The app can now import presets from the original BLUEJAY_WAAAVES OpenFrameworks application:

**Import Process:**
1. Click "Import OF Dir..." button in the preset section
2. The app looks for OF presets in `~/Developer/of_v0.12.0_osx_release/apps/vj/BLUEJAY_WAAAVES/bin/data/saveStates`
3. All `gwSaveStateXXX.json` files are converted to the Rust format
4. Imported presets are saved to the current bank with cleaned-up names

**Mapping:**
- OF array-based parameter groups are mapped to named Rust struct fields
- CH1 Adjust → `ch1_x_displace`, `ch1_y_displace`, etc.
- FB1 Geo → `fb1_x_displace`, `fb1_y_displace`, etc.
- Discrete values (booleans, enums) are extracted from separate arrays
- LFO data is currently not imported (placeholder for future)

**Example OF to Rust Mapping:**
```rust
// OF format: ch1Adjust[16] array
"ch1Adjust": [0.5, 0.0, 0.0, 0.0, ...]

// Rust format: Named fields
Block1Params {
    ch1_x_displace: 0.5,
    ch1_y_displace: 0.0,
    ch1_z_displace: 0.0,
    ch1_rotate: 0.0,
    ...
}
```

## Block 1 Implementation Lessons Learned

This section documents the critical bugs fixed and patterns established during Block 1 implementation. Use this as a guide for implementing Block 2 and Block 3.

### Critical Bug #1: Shared Uniform Buffer Overwrite

**Impact:** CH1 showed CH2's parameters - all per-channel processing broken  
**Root Cause:** Both channels wrote to the same uniform buffer before rendering

```rust
// WRONG - Single buffer
queue.write_buffer(&uniforms, 0, &ch1_bytes);  // Write CH1
queue.write_buffer(&uniforms, 0, &ch2_bytes);  // Overwrites CH1!
render_ch1();  // Uses CH2 data
render_ch2();  // Uses CH2 data
```

**Fix:** Separate buffers for each channel
```rust
// CORRECT - Separate buffers
queue.write_buffer(&uniforms_ch1, 0, &ch1_bytes);
queue.write_buffer(&uniforms_ch2, 0, &ch2_bytes);
render_ch1();  // Binds uniforms_ch1
render_ch2();  // Binds uniforms_ch2
```

**Key Insight:** GPU command encoding is deferred. `write_buffer` happens immediately on CPU, but render passes execute later on GPU. The second write overwrites the first before either render executes.

### Critical Bug #2: Uniform Buffer Layout Mismatch

**Impact:** FB1 transforms/color effects didn't work  
**Root Cause:** Rust `Vec3` size (16 bytes) ≠ WGSL `vec3` size (12 bytes)

**The Problem:**
```rust
// Rust Vec3 has size 16 (12 data + 4 padding)
#[repr(C, align(16))]
struct Vec3([f32; 3]);  // 16 bytes

// WGSL vec3 has size 12
fb1_hsb_offset: vec3<f32>,  // 12 bytes
```

This 4-byte difference causes offset drift for all subsequent fields.

**Fix:** Use `[f32; 3]` with explicit padding
```rust
_pad_before_hsb: [f32; 2],      // Align to 16-byte boundary
fb1_hsb_offset: [f32; 3],       // 12 bytes
_pad_after_hsb_offset: f32,     // Pad to next 16-byte boundary
fb1_hsb_attenuate: [f32; 3],    // 12 bytes
```

**Verification:** Always check offsets with:
```rust
fn offset_of<T, F>(f: fn(&T) -> &F) -> usize {
    let uninit = std::mem::MaybeUninit::<T>::uninit();
    let base_ptr = uninit.as_ptr();
    let field_ptr = f(unsafe { &*base_ptr });
    (field_ptr as *const F as usize) - (base_ptr as usize)
}
```

### Critical Bug #3: Keying Always Active

**Impact:** Could never get pure CH1/CH2 blend - keying always interfered  
**Root Cause:** Keying function ran unconditionally

**Fix:** Only apply keying when threshold < 1.0
```wgsl
if (params.threshold >= 0.999) {
    return fg;  // No keying
}
// Otherwise apply keying
```

### Critical Bug #4: LFO Waveform Not Synced

**Impact:** LFO always used sine wave regardless of GUI selection  
**Root Cause:** GUI-local state not synced to shared state

**Fix:** Sync all LFO parameters when changed
```rust
if new_waveform != lfo.waveform {
    lfo.waveform = new_waveform;
    // Sync to shared state
    if let Ok(mut state) = self.shared_state.lock() {
        state.lfo_banks[bank_index].waveform = new_waveform;
    }
}
```

### Implementation Checklist for Block 2/3

**Per-Channel Processing (Stage 1 & 2 pattern):**
- [ ] Create separate uniform buffers for each channel
- [ ] Create separate bind groups for each channel
- [ ] Update struct initialization to populate both buffers
- [ ] Run layout verification script

**Mixing & Feedback (Stage 3 pattern):**
- [ ] Single uniform buffer (no per-channel needed)
- [ ] Match WGSL layout for vec3 fields (use `[f32; 3]` + padding)
- [ ] Implement key_order logic correctly
- [ ] Add threshold check to disable keying at 1.0

**LFO Integration:**
- [ ] Sync waveform from GUI to shared state
- [ ] Sync rate, tempo_sync, division to shared state

### Reference Documents

- `BLOCK1_IMPLEMENTATION_GUIDE.md` - Complete implementation guide
- `BLOCK1_STATUS.md` - Current status and testing checklist
- `src/engine/blocks/block1.rs` - Full implementation with comments

## Troubleshooting

### Grey Screen Output

**Symptom:** When webcam is started, output shows solid grey instead of camera image.

**Possible Causes:**
1. **Texture format mismatch** - Webcam may be providing YUYV instead of RGB
2. **Shader processing issue** - HSB conversion in blur_and_sharpen may have precision issues
3. **Texture coordinate issues** - UV mapping may be sampling outside valid texture area
4. **Uniform buffer misalignment** - Incorrect parameter values being passed to shader

**Debugging Steps:**
1. Check logs for frame upload messages:
   ```
   Input 1 frame: 640x480, non-zero pixels: X, avg brightness: Y
   ```
2. Enable shader debug output (uncomment line 789 in block1.rs):
   ```rust
   return vec4<f32>(uv.x, uv.y, 0.0, 1.0);  // Shows UV gradient
   ```
3. Check if blue screen appears (indicates black input):
   - If blue: texture sampling issue
   - If grey: shader processing issue

**Known Code Locations:**
- Webcam capture: `src/input/webcam.rs`
- Texture upload: `src/input/texture_input.rs` lines 104-170
- Shader sampling: `src/engine/pipelines/block1.rs` lines 520-572
- HSB conversion: `src/engine/pipelines/block1.rs` lines 430-444

## Simple Feedback Engine Reference

The simplified single-block feedback engine (`--simple` flag) is a minimal implementation that demonstrates core concepts used in the full build.

### Key Files

| File | Purpose |
|------|---------|
| `src/simple_main.rs` | Application entry point with dual-window setup |
| `src/engine/simple_engine.rs` | Core rendering engine with ping-pong feedback |
| `src/engine/simple_feedback.rs` | Single WGSL shader with hue shift and transforms |
| `src/engine/lfo_tempo.rs` | Tempo-synced LFO bank implementation |

### Architecture Highlights

1. **Dual-Window System**: Output window (shader) + Control window (ImGui) sharing one device
2. **Ping-Pong Feedback**: Two textures alternating read/write each frame
3. **Tap Tempo**: 4+ taps calculates BPM, each tap resets LFO phase
4. **Unified Texture Format**: All textures use surface format (Bgra8UnormSrgb) for compatibility

### Lessons for Main Build

The simple engine demonstrates solutions to several issues in the main build:

**Texture Format Consistency**:
```rust
// All textures must use the same format as the surface
let surface_format = config.format; // Bgra8UnormSrgb on macOS
let texture = Texture::create_render_target_with_format(device, width, height, label, surface_format);
```

**Device Sharing on macOS**:
```rust
// Cannot create multiple devices on Metal - must share
let control_renderer = ImGuiRenderer::new(instance, engine.device(), engine.queue(), window);
```

**Feedback UV Transformations**:
```rust
// Apply rotate/zoom/translate to feedback UVs for kaleidoscope effects
let feedback_uv = transform_uv(texcoord, rotate_lfo, zoom_lfo, ...);
let feedback_color = textureSample(feedback_tex, sampler, feedback_uv);
```

### Complete Documentation

See **[SIMPLE_ENGINE.md](./SIMPLE_ENGINE.md)** for:
- Detailed architecture diagrams
- Shader pipeline walkthrough
- Common issues and solutions
- Extension guide for adding features
- Performance considerations
- Debugging techniques

## References

- Original OF project: `~/Developer/of_v0.12.0_osx_release/apps/vj/BLUEJAY_WAAAVES/`
- WGSL Spec: https://www.w3.org/TR/WGSL/
- wgpu Docs: https://docs.rs/wgpu/
- imgui-rs Docs: https://docs.rs/imgui/
