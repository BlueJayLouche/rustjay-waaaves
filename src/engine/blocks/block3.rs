//! # Modular Block 3
//!
//! Final output block with 2-stage architecture:
//! 1. Stage 1a/1b: Block 1 & Block 2 Re-processing (transforms, filters, colorize, dither)
//! 2. Stage 2: Matrix Mixer + Final Mix with keying
//!
//! Unlike Block 1/2, there's no feedback loop - it's purely feed-forward.

use crate::engine::blocks::StageVertex;
use crate::engine::texture::Texture;
use crate::params::Block3Params;


/// Vec3 type matching WGSL's vec3<f32> - 16-byte aligned but only 12 bytes of data
#[repr(C, align(16))]
#[derive(Debug, Copy, Clone, bytemuck::AnyBitPattern)]
pub struct Vec3([f32; 3]);

impl Vec3 {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self([x, y, z])
    }
}

impl From<glam::Vec3> for Vec3 {
    fn from(v: glam::Vec3) -> Self {
        Self([v.x, v.y, v.z])
    }
}

/// Stage 1 uniforms for Block re-processing (used for both Block 1 and Block 2)
#[repr(C, align(16))]
#[derive(Debug, Copy, Clone, bytemuck::Zeroable)]
struct Stage1Uniforms {
    // Resolution (0-16)
    width: f32,
    height: f32,
    inv_width: f32,
    inv_height: f32,
    
    // Geometric transforms (16-64)
    xy_displace: [f32; 2],  // 16
    z_displace: f32,        // 24
    rotate: f32,            // 28
    shear_matrix: [f32; 4], // 32 (vec4)
    kaleidoscope: f32,      // 48
    kaleidoscope_slice: f32,// 52
    _pad1: f32,             // 56
    _pad2: f32,             // 60
    
    // Switches packed as floats (64-80)
    h_mirror: f32,          // 64
    v_mirror: f32,          // 68
    h_flip: f32,            // 72
    v_flip: f32,            // 76
    geo_overflow: i32,      // 80
    rotate_mode: i32,       // 84
    _pad3: f32,             // 88
    _pad4: f32,             // 92
    
    // Filters (96-128)
    blur_amount: f32,       // 96
    blur_radius: f32,       // 100
    sharpen_amount: f32,    // 104
    sharpen_radius: f32,    // 108
    filters_boost: f32,     // 112
    _pad5: f32,             // 116
    _pad6: f32,             // 120
    _pad7: f32,             // 124
    
    // Colorize (128-232)
    colorize_switch: f32,      // 128
    colorize_mode: i32,        // 132
    // After i32 at 132, need padding to reach 16-byte boundary at 144
    // colorize_mode ends at 136 (132+4), next 16-byte boundary is 144
    // So we need 144 - 136 = 8 bytes = 2 floats of padding
    _pad_pre_band1b: [f32; 2], // 136-144 (8 bytes)
    colorize_band1: [f32; 3],   // 144-156 (12 bytes, 16-byte aligned)
    _pad8: f32,                 // 156-160
    colorize_band2: [f32; 3],   // 160-172
    _pad9: f32,                 // 172-176
    colorize_band3: [f32; 3],   // 176-188
    _pad10: f32,                // 188-192
    colorize_band4: [f32; 3],   // 192-204
    _pad11: f32,                // 204-208
    colorize_band5: [f32; 3],   // 208-220
    _pad12: f32,                // 220-224
    
    // Dither (224-240)
    dither_amount: f32,     // 224
    dither_switch: f32,     // 228
    dither_type: i32,       // 232
    _pad13: f32,            // 236-240
}

/// Stage 2 uniforms for Matrix Mix + Final Mix
#[repr(C, align(16))]
#[derive(Debug, Copy, Clone, bytemuck::Zeroable)]
struct Stage2Uniforms {
    // Matrix mixer (0-64)
    matrix_mix_type: i32,   // 0
    matrix_mix_overflow: i32, // 4
    _pad0: f32,             // 8
    _pad1: f32,             // 12
    bg_into_fg_red: [f32; 3], // 16
    _pad2: f32,             // 28
    bg_into_fg_green: [f32; 3], // 32
    _pad3: f32,             // 44
    bg_into_fg_blue: [f32; 3], // 48
    _pad4: f32,             // 60
    
    // Final mix (64-112)
    final_mix_amount: f32,  // 64
    _pad5: f32,             // 68
    _pad6: f32,             // 72
    _pad7: f32,             // 76
    final_key_value: [f32; 3], // 80
    final_key_threshold: f32, // 92
    final_key_soft: f32,    // 96
    final_mix_type: i32,    // 100
    final_mix_overflow: i32,// 104
    final_key_order: i32,   // 108
    _pad8: f32,             // 112
    
    // Final dither (112-128)
    final_dither_amount: f32,  // 112
    final_dither_switch: f32,  // 116
    final_dither_type: i32,    // 120
    _pad9: f32,                // 124
}

/// Modular Block 3 implementation
pub struct ModularBlock3 {
    /// Ping-pong buffers for Block 1 re-processing
    block1_buffer_a: Texture,
    block1_buffer_b: Texture,
    block1_current: usize,
    
    /// Ping-pong buffers for Block 2 re-processing  
    block2_buffer_a: Texture,
    block2_buffer_b: Texture,
    block2_current: usize,
    
    /// Stage 1: Re-processing pipeline (shared for Block 1 and Block 2)
    stage1_pipeline: wgpu::RenderPipeline,
    stage1_bind_group_layout: wgpu::BindGroupLayout,
    stage1_uniforms_block1: wgpu::Buffer,
    stage1_uniforms_block2: wgpu::Buffer,
    
    /// Stage 2: Matrix Mix + Final Mix pipeline
    stage2_pipeline: wgpu::RenderPipeline,
    stage2_bind_group_layout: wgpu::BindGroupLayout,
    stage2_uniforms: wgpu::Buffer,
    
    /// Vertex buffer
    vertex_buffer: wgpu::Buffer,
    
    /// Dimensions
    width: u32,
    height: u32,
    
    /// Cached sampler (avoids per-frame allocation)
    cached_sampler: wgpu::Sampler,
    
    /// Cached bind groups (avoid per-frame allocation)
    cached_block1_bind_group: Option<wgpu::BindGroup>,
    cached_block2_bind_group: Option<wgpu::BindGroup>,
    cached_stage2_bind_group: Option<wgpu::BindGroup>,
    
    /// Cached texture view IDs to detect when bind groups need recreation
    last_block1_input_id: usize,
    last_block2_input_id: usize,
}

impl ModularBlock3 {
    /// Create new modular Block 3
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, width: u32, height: u32) -> Self {
        // Create ping-pong buffers for Block 1 re-processing
        let block1_buffer_a = Texture::create_render_target_with_format(
            device, width, height, "Block3 B1 Buffer A", wgpu::TextureFormat::Bgra8Unorm,
        );
        let block1_buffer_b = Texture::create_render_target_with_format(
            device, width, height, "Block3 B1 Buffer B", wgpu::TextureFormat::Bgra8Unorm,
        );
        block1_buffer_a.clear_to_black(queue);
        block1_buffer_b.clear_to_black(queue);
        
        // Create ping-pong buffers for Block 2 re-processing
        let block2_buffer_a = Texture::create_render_target_with_format(
            device, width, height, "Block3 B2 Buffer A", wgpu::TextureFormat::Bgra8Unorm,
        );
        let block2_buffer_b = Texture::create_render_target_with_format(
            device, width, height, "Block3 B2 Buffer B", wgpu::TextureFormat::Bgra8Unorm,
        );
        block2_buffer_a.clear_to_black(queue);
        block2_buffer_b.clear_to_black(queue);
        
        // Create vertex buffer
        let vertex_buffer = StageVertex::create_quad_buffer(device, queue);
        
        // Create Stage 1 (re-processing)
        let (stage1_pipeline, stage1_bind_group_layout, stage1_uniforms_block1, stage1_uniforms_block2) =
            Self::create_stage1(device, queue, width, height);
        
        // Create Stage 2 (matrix mix + final mix)
        let (stage2_pipeline, stage2_bind_group_layout, stage2_uniforms) =
            Self::create_stage2(device, queue);
        
        // Create cached sampler
        let cached_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        
        Self {
            block1_buffer_a,
            block1_buffer_b,
            block1_current: 0,
            block2_buffer_a,
            block2_buffer_b,
            block2_current: 0,
            stage1_pipeline,
            stage1_bind_group_layout,
            stage1_uniforms_block1,
            stage1_uniforms_block2,
            stage2_pipeline,
            stage2_bind_group_layout,
            stage2_uniforms,
            vertex_buffer,
            width,
            height,
            cached_sampler,
            cached_block1_bind_group: None,
            cached_block2_bind_group: None,
            cached_stage2_bind_group: None,
            last_block1_input_id: 0,
            last_block2_input_id: 0,
        }
    }
    
    /// Create Stage 1: Re-processing pipeline for Block 1 and Block 2
    fn create_stage1(
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        width: u32,
        height: u32,
    ) -> (wgpu::RenderPipeline, wgpu::BindGroupLayout, wgpu::Buffer, wgpu::Buffer) {
        let shader_code = r#"
            struct VertexInput {
                @location(0) position: vec2<f32>,
                @location(1) texcoord: vec2<f32>,
            };
            
            struct VertexOutput {
                @builtin(position) position: vec4<f32>,
                @location(0) texcoord: vec2<f32>,
            };
            
            @vertex
            fn vs_main(input: VertexInput) -> VertexOutput {
                var output: VertexOutput;
                output.position = vec4<f32>(input.position, 0.0, 1.0);
                output.texcoord = input.texcoord;
                return output;
            }
            
            struct Uniforms {
                width: f32,
                height: f32,
                inv_width: f32,
                inv_height: f32,
                xy_displace: vec2<f32>,
                z_displace: f32,
                rotate: f32,
                shear_matrix: vec4<f32>,
                kaleidoscope: f32,
                kaleidoscope_slice: f32,
                _pad1: f32,
                _pad2: f32,
                h_mirror: f32,
                v_mirror: f32,
                h_flip: f32,
                v_flip: f32,
                geo_overflow: i32,
                rotate_mode: i32,
                _pad3: f32,
                _pad4: f32,
                blur_amount: f32,
                blur_radius: f32,
                sharpen_amount: f32,
                sharpen_radius: f32,
                filters_boost: f32,
                _pad5: f32,
                _pad6: f32,
                _pad7: f32,
                colorize_switch: f32,      // 128
                colorize_mode: i32,        // 132
                _pad_pre_band1b: vec2<f32>, // 136-144 (padding to align vec3)
                colorize_band1: vec3<f32>, // 144-156
                _pad8: f32,                // 156-160
                colorize_band2: vec3<f32>, // 160-172
                _pad9: f32,                // 172-176
                colorize_band3: vec3<f32>, // 176-188
                _pad10: f32,               // 188-192
                colorize_band4: vec3<f32>, // 192-204
                _pad11: f32,               // 204-208
                colorize_band5: vec3<f32>, // 208-220
                _pad12: f32,               // 220-224
                dither_amount: f32,        // 224
                dither_switch: f32,        // 228
                dither_type: i32,          // 232
                _pad13: f32,
            };
            
            @group(0) @binding(0)
            var<uniform> uniforms: Uniforms;
            @group(0) @binding(1)
            var input_tex: texture_2d<f32>;
            @group(0) @binding(2)
            var input_sampler: sampler;
            
            const PI: f32 = 3.1415926535;
            const TWO_PI: f32 = 6.2831855;
            
            // RGB to HSB conversion
            fn rgb2hsb(c: vec3<f32>) -> vec3<f32> {
                let K = vec4<f32>(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
                let p = mix(vec4<f32>(c.bg, K.wz), vec4<f32>(c.gb, K.xy), step(c.b, c.g));
                let q = mix(vec4<f32>(p.xyw, c.r), vec4<f32>(c.r, p.yzx), step(p.x, c.r));
                let d = q.x - min(q.w, q.y);
                let e = 1.0e-10;
                return vec3<f32>(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
            }
            
            // HSB to RGB conversion
            fn hsb2rgb(c: vec3<f32>) -> vec3<f32> {
                let K = vec4<f32>(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
                let p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
                return c.z * mix(K.xxx, clamp(p - K.xxx, vec3<f32>(0.0), vec3<f32>(1.0)), c.y);
            }
            
            fn wrap(v: f32) -> f32 {
                return fract(abs(v));
            }
            
            fn foldover(v: f32) -> f32 {
                var result = v;
                if (result < 0.0) { result = abs(result); }
                if (result > 1.0) { result = 1.0 - fract(result); }
                if (result < 0.0) { result = abs(result); }
                return result;
            }
            
            fn rotate_uv(uv: vec2<f32>, angle: f32, mode: i32) -> vec2<f32> {
                let center = vec2<f32>(0.5, 0.5);
                var centered = uv - center;
                let theta = radians(angle);
                
                if (mode == 0) {
                    // Mode 0: Non-preserving (original OF style)
                    let spiral = (abs(uv.x + uv.y) / 2.0) * 0.5;
                    centered = centered + vec2<f32>(spiral);
                }
                
                let cos_t = cos(theta);
                let sin_t = sin(theta);
                let rot_x = centered.x * cos_t - centered.y * sin_t;
                let rot_y = centered.x * sin_t + centered.y * cos_t;
                
                return vec2<f32>(rot_x, rot_y) + center;
            }
            
            fn kaleidoscope_uv(uv: vec2<f32>, segment: f32, slice: f32) -> vec2<f32> {
                if (segment <= 0.0 || slice <= 0.0) {
                    return uv;
                }
                
                var result = rotate_uv(uv, slice, 1);
                var centered = result - vec2<f32>(0.5);
                
                let radius = length(centered);
                var angle = atan2(centered.y, centered.x);
                
                let segment_angle = TWO_PI * slice;
                let slice_index = floor(angle / segment_angle);
                let slice_angle = angle - slice_index * segment_angle;
                let mirrored_angle = abs(slice_angle - segment_angle * 0.5);
                
                angle = mix(angle, slice_index * segment_angle + mirrored_angle, segment);
                centered = radius * vec2<f32>(cos(angle), sin(angle));
                
                result = centered + vec2<f32>(0.5);
                return rotate_uv(result, -slice, 0);
            }
            
            fn apply_geo_transforms(uv: vec2<f32>) -> vec2<f32> {
                var result = uv;
                
                // Horizontal flip
                if (uniforms.h_flip > 0.5) {
                    result.x = 1.0 - result.x;
                }
                
                // Vertical flip
                if (uniforms.v_flip > 0.5) {
                    result.y = 1.0 - result.y;
                }
                
                // Horizontal mirror
                if (uniforms.h_mirror > 0.5) {
                    if (result.x > 0.5) {
                        result.x = abs(1.0 - result.x);
                    }
                }
                
                // Vertical mirror
                if (uniforms.v_mirror > 0.5) {
                    if (result.y > 0.5) {
                        result.y = abs(1.0 - result.y);
                    }
                }
                
                return result;
            }
            
            fn apply_shear(uv: vec2<f32>) -> vec2<f32> {
                // Early exit if shear is identity
                let shear = uniforms.shear_matrix;
                if (shear.x == 1.0 && shear.y == 0.0 && shear.z == 0.0 && shear.w == 1.0) {
                    return uv;
                }
                
                var centered = uv - vec2<f32>(0.5);
                centered = vec2<f32>(
                    shear.x * centered.x + shear.y * centered.y,
                    shear.z * centered.x + shear.w * centered.y
                );
                return centered + vec2<f32>(0.5);
            }
            
            fn blur_and_sharpen(uv: vec2<f32>, color: vec3<f32>) -> vec3<f32> {
                var result = color;
                let tex_size = vec2<f32>(uniforms.width, uniforms.height);
                
                // Blur
                if (uniforms.blur_amount > 0.001) {
                    let blur_size = vec2<f32>(uniforms.blur_radius) / tex_size;
                    var blur_color = color;
                    
                    // 8-sample box blur
                    blur_color += textureSample(input_tex, input_sampler, uv + blur_size * vec2<f32>(1.0, 1.0)).rgb;
                    blur_color += textureSample(input_tex, input_sampler, uv + blur_size * vec2<f32>(0.0, 1.0)).rgb;
                    blur_color += textureSample(input_tex, input_sampler, uv + blur_size * vec2<f32>(-1.0, 1.0)).rgb;
                    blur_color += textureSample(input_tex, input_sampler, uv + blur_size * vec2<f32>(-1.0, 0.0)).rgb;
                    blur_color += textureSample(input_tex, input_sampler, uv + blur_size * vec2<f32>(-1.0, -1.0)).rgb;
                    blur_color += textureSample(input_tex, input_sampler, uv + blur_size * vec2<f32>(0.0, -1.0)).rgb;
                    blur_color += textureSample(input_tex, input_sampler, uv + blur_size * vec2<f32>(1.0, -1.0)).rgb;
                    blur_color += textureSample(input_tex, input_sampler, uv + blur_size * vec2<f32>(1.0, 0.0)).rgb;
                    blur_color *= 0.125;
                    
                    result = mix(color, blur_color, uniforms.blur_amount);
                }
                
                // Sharpen (applied after blur)
                if (uniforms.sharpen_amount > 0.001) {
                    let sharpen_size = vec2<f32>(uniforms.sharpen_radius * 2.0) / tex_size;
                    let lum_weights = vec3<f32>(0.299, 0.587, 0.114);
                    
                    // Sample neighbors for sharpening
                    var bright_sharpen = 0.0;
                    bright_sharpen += dot(textureSample(input_tex, input_sampler, uv + sharpen_size * vec2<f32>(1.0, 0.0)).rgb, lum_weights);
                    bright_sharpen += dot(textureSample(input_tex, input_sampler, uv + sharpen_size * vec2<f32>(-1.0, 0.0)).rgb, lum_weights);
                    bright_sharpen += dot(textureSample(input_tex, input_sampler, uv + sharpen_size * vec2<f32>(0.0, 1.0)).rgb, lum_weights);
                    bright_sharpen += dot(textureSample(input_tex, input_sampler, uv + sharpen_size * vec2<f32>(0.0, -1.0)).rgb, lum_weights);
                    bright_sharpen += dot(textureSample(input_tex, input_sampler, uv + sharpen_size * vec2<f32>(1.0, 1.0)).rgb, lum_weights);
                    bright_sharpen += dot(textureSample(input_tex, input_sampler, uv + sharpen_size * vec2<f32>(-1.0, 1.0)).rgb, lum_weights);
                    bright_sharpen += dot(textureSample(input_tex, input_sampler, uv + sharpen_size * vec2<f32>(1.0, -1.0)).rgb, lum_weights);
                    bright_sharpen += dot(textureSample(input_tex, input_sampler, uv + sharpen_size * vec2<f32>(-1.0, -1.0)).rgb, lum_weights);
                    bright_sharpen *= 0.125;
                    
                    var hsb = rgb2hsb(result);
                    hsb.z -= uniforms.sharpen_amount * bright_sharpen;
                    let boost_factor = 1.0 + uniforms.sharpen_amount + uniforms.filters_boost;
                    hsb.z *= boost_factor;
                    result = hsb2rgb(hsb);
                }
                
                return result;
            }
            
            fn apply_colorize(color: vec3<f32>) -> vec3<f32> {
                if (uniforms.colorize_switch < 0.5) {
                    return color;
                }
                
                let hsb = rgb2hsb(color);
                let brightness = hsb.z;
                
                // Pre-calculate HSB and RGB modes for each band
                let hsb1 = uniforms.colorize_band1 + vec3<f32>(0.0, 0.0, brightness);
                let hsb2 = uniforms.colorize_band2 + vec3<f32>(0.0, 0.0, brightness);
                let hsb3 = uniforms.colorize_band3 + vec3<f32>(0.0, 0.0, brightness);
                let hsb4 = uniforms.colorize_band4 + vec3<f32>(0.0, 0.0, brightness);
                let hsb5 = uniforms.colorize_band5 + vec3<f32>(0.0, 0.0, brightness);
                
                let rgb_hsb1 = hsb2rgb(hsb1);
                let rgb_hsb2 = hsb2rgb(hsb2);
                let rgb_hsb3 = hsb2rgb(hsb3);
                let rgb_hsb4 = hsb2rgb(hsb4);
                let rgb_hsb5 = hsb2rgb(hsb5);
                
                let rgb1 = uniforms.colorize_band1 + color;
                let rgb2 = uniforms.colorize_band2 + color;
                let rgb3 = uniforms.colorize_band3 + color;
                let rgb4 = uniforms.colorize_band4 + color;
                let rgb5 = uniforms.colorize_band5 + color;
                
                // Select mode (0=HSB, 1=RGB)
                let mode_mix = f32(uniforms.colorize_mode);
                let col1 = mix(rgb_hsb1, rgb1, mode_mix);
                let col2 = mix(rgb_hsb2, rgb2, mode_mix);
                let col3 = mix(rgb_hsb3, rgb3, mode_mix);
                let col4 = mix(rgb_hsb4, rgb4, mode_mix);
                let col5 = mix(rgb_hsb5, rgb5, mode_mix);
                
                // Band mixing
                let band_mix1 = clamp(brightness * 4.0, 0.0, 1.0);
                let band_mix2 = clamp((brightness - 0.25) * 4.0, 0.0, 1.0);
                let band_mix3 = clamp((brightness - 0.5) * 4.0, 0.0, 1.0);
                let band_mix4 = clamp((brightness - 0.75) * 4.0, 0.0, 1.0);
                
                var result = mix(
                    mix(
                        mix(col1, col2, band_mix1),
                        mix(col2, col3, band_mix2),
                        step(0.25, brightness)
                    ),
                    mix(col3, col4, band_mix3),
                    step(0.5, brightness)
                );
                result = mix(result, mix(col4, col5, band_mix4), step(0.75, brightness));
                
                return result;
            }
            
            // === DITHER ALGORITHMS ===
            // Comprehensive collection from classic to glitchy
            
            // --- Helper: Hash/Noise functions ---
            fn hash12(p: vec2<f32>) -> f32 {
                let p3 = fract(vec3<f32>(p.xyx) * 0.1031);
                let p4 = p3 + dot(p3, p3.yzx + 33.33);
                return fract((p4.x + p4.y) * p4.z);
            }
            
            fn hash13(p: vec3<f32>) -> f32 {
                let p3 = fract(p * 0.1031);
                let p4 = p3 + dot(p3, p3.zxy + 33.33);
                return fract((p4.x + p4.y) * p4.z);
            }
            
            // --- 1. ORDERED DITHERS (Bayer) ---
            fn bayer4x4(coord: vec2<f32>) -> f32 {
                let x = i32(coord.x) & 3;
                let y = i32(coord.y) & 3;
                let matrix = array<i32, 16>(
                    0, 8, 2, 10,
                    12, 4, 14, 6,
                    3, 11, 1, 9,
                    15, 7, 13, 5
                );
                return f32(matrix[y * 4 + x]) / 16.0;
            }
            
            fn bayer8x8(coord: vec2<f32>) -> f32 {
                let x = i32(coord.x) & 7;
                let y = i32(coord.y) & 7;
                let matrix = array<i32, 64>(
                    0, 32, 8, 40, 2, 34, 10, 42,
                    48, 16, 56, 24, 50, 18, 58, 26,
                    12, 44, 4, 36, 14, 46, 6, 38,
                    60, 28, 52, 20, 62, 30, 54, 22,
                    3, 35, 11, 43, 1, 33, 9, 41,
                    51, 19, 59, 27, 49, 17, 57, 25,
                    15, 47, 7, 39, 13, 45, 5, 37,
                    63, 31, 55, 23, 61, 29, 53, 21
                );
                return f32(matrix[y * 8 + x]) / 64.0;
            }
            
            // --- 2. NOISE DITHERS ---
            fn blue_noise(coord: vec2<f32>) -> f32 {
                // Blue noise approximation - more high-frequency content
                let x = i32(coord.x);
                let y = i32(coord.y);
                var n = x * 374761 + y * 668265;
                n = (n << 13) ^ n;
                n = n * (n * n * 15731 + 789221) + 1376312589;
                return f32(n & 0x7fffffff) / f32(0x7fffffff);
            }
            
            fn white_noise(coord: vec2<f32>) -> f32 {
                return fract(sin(dot(coord, vec2<f32>(12.9898, 78.233))) * 43758.5453);
            }
            
            fn triangular_noise(coord: vec2<f32>) -> f32 {
                let n1 = white_noise(coord);
                let n2 = white_noise(coord + vec2<f32>(1.0));
                return (n1 + n2) * 0.5;
            }
            
            fn ign_noise(coord: vec2<f32>) -> f32 {
                // Interleaved Gradient Noise - good for animation
                return fract(52.9829189 * fract(0.06711056 * coord.x + 0.00583715 * coord.y));
            }
            
            // --- 3. GLITCH / ARTIFACT DITHERS ---
            
            fn scanline_dither(in_color: f32, coord: vec2<f32>, palette: f32) -> f32 {
                // Horizontal scanlines like CRT
                let y = i32(coord.y);
                let pattern = f32(y % 4);
                let threshold = pattern / 4.0;
                let scaled = in_color * palette;
                let dithered = scaled + threshold - 0.5;
                return clamp(floor(dithered) / palette, 0.0, 1.0);
            }
            
            fn checkerboard_dither(in_color: f32, coord: vec2<f32>, palette: f32) -> f32 {
                // 2x2 checker pattern
                let x = i32(coord.x) & 1;
                let y = i32(coord.y) & 1;
                let threshold = f32(x ^ y);
                let scaled = in_color * palette;
                let dithered = scaled + threshold - 0.5;
                return clamp(floor(dithered) / palette, 0.0, 1.0);
            }
            
            fn stripe_dither(in_color: f32, coord: vec2<f32>, palette: f32) -> f32 {
                // Vertical stripes - corrupted signal look
                let x = i32(coord.x);
                let threshold = f32(x % 8) / 8.0;
                let scaled = in_color * palette;
                let dithered = scaled + threshold - 0.5;
                return clamp(floor(dithered) / palette, 0.0, 1.0);
            }
            
            fn bitcrush_dither(in_color: f32, coord: vec2<f32>, bits: f32) -> f32 {
                // Digital bit-depth reduction with noise
                let step_size = 1.0 / bits;
                let noise = white_noise(coord);
                let dithered = in_color + (noise - 0.5) * step_size;
                return floor(dithered * bits) / bits;
            }
            
            fn threshold_dither(in_color: f32, coord: vec2<f32>) -> f32 {
                // Extreme 1-bit with noise variation
                let noise = white_noise(coord);
                let threshold = 0.5 + (noise - 0.5) * 0.15;
                return select(0.0, 1.0, in_color > threshold);
            }
            
            fn pixelsort_dither(in_color: f32, coord: vec2<f32>, palette: f32) -> f32 {
                // Simulates pixel sorting glitch
                let y = i32(coord.y);
                let pattern = f32(y % 2);
                let step_size = 1.0 / palette;
                let offset = select(0.0, step_size, pattern > 0.5);
                let dithered = in_color + offset;
                return clamp(floor(dithered * palette) / palette, 0.0, 1.0);
            }
            
            fn atkinson_dither(in_color: f32, coord: vec2<f32>, palette: f32) -> f32 {
                // Simulates error diffusion
                let step_size = 1.0 / palette;
                let x = i32(coord.x);
                let y = i32(coord.y);
                let noise = fract(sin(f32(x * 7 + y * 13)) * 43758.5453);
                let threshold = noise;
                let scaled = in_color * palette;
                let dithered = scaled + threshold - 0.5;
                return clamp(floor(dithered) / palette, 0.0, 1.0);
            }
            
            fn glitch_channel_dither(in_color: f32, coord: vec2<f32>, palette: f32, channel: i32) -> f32 {
                // RGB channel offset - creates color fringing
                let offset = vec2<f32>(f32(channel) * 3.0, f32(channel) * 7.0);
                let threshold = white_noise(coord + offset);
                let scaled = in_color * palette;
                let dithered = scaled + threshold - 0.5;
                return clamp(floor(dithered) / palette, 0.0, 1.0);
            }
            
            // --- Main Dither Router ---
            fn dither_channel(in_color: f32, coord: vec2<f32>, palette: f32, dither_type: i32, channel: i32) -> f32 {
                let step_size = 1.0 / palette;
                
                switch(dither_type) {
                    // Ordered (0-1) - Standard Bayer ordered dither
                    case 0: {
                        let threshold = bayer4x4(coord);
                        // Map input to palette range, add threshold, then quantize
                        let scaled = in_color * palette;
                        let dithered = scaled + threshold - 0.5;
                        return clamp(floor(dithered) / palette, 0.0, 1.0);
                    }
                    case 1: {
                        let threshold = bayer8x8(coord);
                        let scaled = in_color * palette;
                        let dithered = scaled + threshold - 0.5;
                        return clamp(floor(dithered) / palette, 0.0, 1.0);
                    }
                    // Noise (2-4) - Use noise as threshold
                    case 2: {
                        let threshold = blue_noise(coord);
                        let scaled = in_color * palette;
                        let dithered = scaled + threshold - 0.5;
                        return clamp(floor(dithered) / palette, 0.0, 1.0);
                    }
                    case 3: {
                        let threshold = white_noise(coord);
                        let scaled = in_color * palette;
                        let dithered = scaled + threshold - 0.5;
                        return clamp(floor(dithered) / palette, 0.0, 1.0);
                    }
                    case 4: {
                        let threshold = ign_noise(coord);
                        let scaled = in_color * palette;
                        let dithered = scaled + threshold - 0.5;
                        return clamp(floor(dithered) / palette, 0.0, 1.0);
                    }
                    // Glitch/Artifact (5-12)
                    case 5: { return scanline_dither(in_color, coord, palette); }
                    case 6: { return checkerboard_dither(in_color, coord, palette); }
                    case 7: { return stripe_dither(in_color, coord, palette); }
                    case 8: { return bitcrush_dither(in_color, coord, palette); }
                    case 9: { return threshold_dither(in_color, coord); }
                    case 10: { return pixelsort_dither(in_color, coord, palette); }
                    case 11: { return atkinson_dither(in_color, coord, palette); }
                    case 12: { return glitch_channel_dither(in_color, coord, palette, channel); }
                    default: { return in_color; }
                }
            }
            
            fn apply_dither(color: vec3<f32>, coord: vec2<f32>) -> vec3<f32> {
                if (uniforms.dither_switch < 0.5) {
                    return color;
                }
                let dither_type = i32(uniforms.dither_type);
                let palette = uniforms.dither_amount;
                return vec3<f32>(
                    dither_channel(color.r, coord, palette, dither_type, 0),
                    dither_channel(color.g, coord, palette, dither_type, 1),
                    dither_channel(color.b, coord, palette, dither_type, 2)
                );
            }
            
            @fragment
            fn fs_main(@location(0) texcoord: vec2<f32>) -> @location(0) vec4<f32> {
                // Start with original UV
                var uv = texcoord;
                
                // Apply geometric transforms
                uv = apply_geo_transforms(uv);
                uv = kaleidoscope_uv(uv, uniforms.kaleidoscope, uniforms.kaleidoscope_slice);
                
                // Apply displace (only if non-zero)
                if (uniforms.xy_displace.x != 0.0 || uniforms.xy_displace.y != 0.0) {
                    uv = uv + uniforms.xy_displace;
                }
                
                // Apply scale (Z displace) - multiply like OF, only if not 1.0
                if (uniforms.z_displace != 1.0) {
                    let center = vec2<f32>(0.5);
                    uv = (uv - center) * uniforms.z_displace + center;
                }
                
                // Apply rotation (only if non-zero)
                if (uniforms.rotate != 0.0) {
                    uv = rotate_uv(uv, uniforms.rotate, uniforms.rotate_mode);
                }
                
                // Apply shear
                uv = apply_shear(uv);
                
                // Sample input
                var color = textureSample(input_tex, input_sampler, uv).rgb;
                
                // Handle overflow modes
                let out_of_bounds = uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0;
                if (uniforms.geo_overflow == 0 && out_of_bounds) {
                    color = vec3<f32>(0.0);
                }
                
                // Apply blur and sharpen
                color = blur_and_sharpen(uv, color);
                
                // Apply colorize
                color = apply_colorize(color);
                
                // Apply dither (use pixel coordinate for dither pattern)
                let pixel_coord = uv * vec2<f32>(uniforms.width, uniforms.height);
                color = apply_dither(color, pixel_coord);
                
                return vec4<f32>(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
            }
        "#;
        
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Block3 Stage1 Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_code.into()),
        });
        
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Block3 Stage1 BGL"),
            entries: &[
                // Uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Input texture
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Block3 Stage1 Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Block3 Stage1 Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[StageVertex::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Bgra8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        
        // Create separate uniform buffers for Block 1 and Block 2
        let uniforms_block1 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Block3 Stage1 Uniforms B1"),
            size: std::mem::size_of::<Stage1Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let uniforms_block2 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Block3 Stage1 Uniforms B2"),
            size: std::mem::size_of::<Stage1Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        (pipeline, bind_group_layout, uniforms_block1, uniforms_block2)
    }
    
    /// Create Stage 2: Matrix Mix + Final Mix pipeline
    fn create_stage2(
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
    ) -> (wgpu::RenderPipeline, wgpu::BindGroupLayout, wgpu::Buffer) {
        let shader_code = r#"
            struct VertexInput {
                @location(0) position: vec2<f32>,
                @location(1) texcoord: vec2<f32>,
            };
            
            struct VertexOutput {
                @builtin(position) position: vec4<f32>,
                @location(0) texcoord: vec2<f32>,
            };
            
            @vertex
            fn vs_main(input: VertexInput) -> VertexOutput {
                var output: VertexOutput;
                output.position = vec4<f32>(input.position, 0.0, 1.0);
                output.texcoord = input.texcoord;
                return output;
            }
            
            struct Uniforms {
                matrix_mix_type: i32,
                matrix_mix_overflow: i32,
                _pad0: f32,
                _pad1: f32,
                bg_into_fg_red: vec3<f32>,
                _pad2: f32,
                bg_into_fg_green: vec3<f32>,
                _pad3: f32,
                bg_into_fg_blue: vec3<f32>,
                _pad4: f32,
                final_mix_amount: f32,
                _pad5: f32,
                _pad6: f32,
                _pad7: f32,
                final_key_value: vec3<f32>,
                final_key_threshold: f32,
                final_key_soft: f32,
                final_mix_type: i32,
                final_mix_overflow: i32,
                final_key_order: i32,
                _pad8: f32,
                // Final dither (add at end to not break alignment)
                final_dither_amount: f32,
                final_dither_switch: f32,
                final_dither_type: i32,
                _pad9: f32,
            };
            
            @group(0) @binding(0)
            var<uniform> uniforms: Uniforms;
            @group(0) @binding(1)
            var block1_tex: texture_2d<f32>;
            @group(0) @binding(2)
            var block1_sampler: sampler;
            @group(0) @binding(3)
            var block2_tex: texture_2d<f32>;
            @group(0) @binding(4)
            var block2_sampler: sampler;
            
            fn wrap(v: f32) -> f32 {
                return fract(abs(v));
            }
            
            fn foldover(v: f32) -> f32 {
                var result = v;
                if (result < 0.0) { result = abs(result); }
                if (result > 1.0) { result = 1.0 - fract(result); }
                if (result < 0.0) { result = abs(result); }
                return result;
            }
            
            fn matrix_mix(fg: vec3<f32>, bg: vec3<f32>) -> vec3<f32> {
                var out_color = vec3<f32>(0.0);
                
                let fgR = vec3<f32>(fg.r, fg.r, fg.r);
                let fgG = vec3<f32>(fg.g, fg.g, fg.g);
                let fgB = vec3<f32>(fg.b, fg.b, fg.b);
                
                let scale = vec3<f32>(0.33, 0.33, 0.33);
                
                // lerp
                if (uniforms.matrix_mix_type == 0) {
                    out_color.r = dot(mix(fgR, bg, uniforms.bg_into_fg_red), scale);
                    out_color.g = dot(mix(fgG, bg, uniforms.bg_into_fg_green), scale);
                    out_color.b = dot(mix(fgB, bg, uniforms.bg_into_fg_blue), scale);
                }
                // add
                else if (uniforms.matrix_mix_type == 1) {
                    out_color.r = dot(fgR + uniforms.bg_into_fg_red * bg, scale);
                    out_color.g = dot(fgG + uniforms.bg_into_fg_green * bg, scale);
                    out_color.b = dot(fgB + uniforms.bg_into_fg_blue * bg, scale);
                }
                // diff
                else if (uniforms.matrix_mix_type == 2) {
                    out_color.r = dot(abs(fgR - uniforms.bg_into_fg_red * bg), scale);
                    out_color.g = dot(abs(fgG - uniforms.bg_into_fg_green * bg), scale);
                    out_color.b = dot(abs(fgB - uniforms.bg_into_fg_blue * bg), scale);
                }
                // mult
                else if (uniforms.matrix_mix_type == 3) {
                    out_color.r = dot(mix(fgR, bg * fgR, uniforms.bg_into_fg_red), scale);
                    out_color.g = dot(mix(fgG, bg * fgG, uniforms.bg_into_fg_green), scale);
                    out_color.b = dot(mix(fgB, bg * fgB, uniforms.bg_into_fg_blue), scale);
                }
                // dodge
                else if (uniforms.matrix_mix_type == 4) {
                    out_color.r = dot(mix(fgR, bg / (1.00001 - fgR), uniforms.bg_into_fg_red), scale);
                    out_color.g = dot(mix(fgG, bg / (1.00001 - fgG), uniforms.bg_into_fg_green), scale);
                    out_color.b = dot(mix(fgB, bg / (1.00001 - fgB), uniforms.bg_into_fg_blue), scale);
                }
                
                // overflow handling
                if (uniforms.matrix_mix_overflow == 0) {
                    out_color = clamp(out_color, vec3<f32>(0.0), vec3<f32>(1.0));
                } else if (uniforms.matrix_mix_overflow == 1) {
                    out_color = vec3<f32>(wrap(out_color.x), wrap(out_color.y), wrap(out_color.z));
                } else if (uniforms.matrix_mix_overflow == 2) {
                    out_color = vec3<f32>(foldover(out_color.x), foldover(out_color.y), foldover(out_color.z));
                }
                
                return out_color;
            }
            
            fn final_mix(fg: vec4<f32>, bg: vec4<f32>) -> vec4<f32> {
                var out_color = fg;
                
                // Mix based on type
                if (uniforms.final_mix_type == 0) { // lerp
                    out_color = mix(fg, bg, uniforms.final_mix_amount);
                } else if (uniforms.final_mix_type == 1) { // add/sub
                    out_color = vec4<f32>(fg.rgb + uniforms.final_mix_amount * bg.rgb, 1.0);
                } else if (uniforms.final_mix_type == 2) { // diff
                    out_color = vec4<f32>(abs(fg.rgb - uniforms.final_mix_amount * bg.rgb), 1.0);
                } else if (uniforms.final_mix_type == 3) { // mult
                    out_color = vec4<f32>(mix(fg.rgb, fg.rgb * bg.rgb, uniforms.final_mix_amount), 1.0);
                } else if (uniforms.final_mix_type == 4) { // dodge
                    out_color = vec4<f32>(mix(fg.rgb, fg.rgb / (1.00001 - bg.rgb), uniforms.final_mix_amount), 1.0);
                }
                
                // overflow handling
                if (uniforms.final_mix_overflow == 0) {
                    out_color = vec4<f32>(clamp(out_color.rgb, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
                } else if (uniforms.final_mix_overflow == 1) {
                    out_color = vec4<f32>(wrap(out_color.r), wrap(out_color.g), wrap(out_color.b), 1.0);
                } else if (uniforms.final_mix_overflow == 2) {
                    out_color = vec4<f32>(foldover(out_color.r), foldover(out_color.g), foldover(out_color.b), 1.0);
                }
                
                // Keying
                let chroma_dist = distance(uniforms.final_key_value, fg.rgb);
                if (chroma_dist < uniforms.final_key_threshold) {
                    let mix_factor = uniforms.final_key_soft * abs(1.0 - (chroma_dist - uniforms.final_key_threshold));
                    out_color = mix(bg, out_color, mix_factor);
                }
                
                return out_color;
            }
            
            // === DITHER FUNCTIONS FOR FINAL OUTPUT ===
            fn hash12(p: vec2<f32>) -> f32 {
                let p3 = fract(vec3<f32>(p.xyx) * 0.1031);
                let p4 = p3 + dot(p3, p3.yzx + 33.33);
                return fract((p4.x + p4.y) * p4.z);
            }
            
            fn bayer4x4_final(coord: vec2<f32>) -> f32 {
                let x = i32(coord.x) & 3;
                let y = i32(coord.y) & 3;
                let matrix = array<i32, 16>(
                    0, 8, 2, 10,
                    12, 4, 14, 6,
                    3, 11, 1, 9,
                    15, 7, 13, 5
                );
                return f32(matrix[y * 4 + x]) / 16.0;
            }
            
            fn bayer8x8_final(coord: vec2<f32>) -> f32 {
                let x = i32(coord.x) & 7;
                let y = i32(coord.y) & 7;
                let matrix = array<i32, 64>(
                    0, 32, 8, 40, 2, 34, 10, 42,
                    48, 16, 56, 24, 50, 18, 58, 26,
                    12, 44, 4, 36, 14, 46, 6, 38,
                    60, 28, 52, 20, 62, 30, 54, 22,
                    3, 35, 11, 43, 1, 33, 9, 41,
                    51, 19, 59, 27, 49, 17, 57, 25,
                    15, 47, 7, 39, 13, 45, 5, 37,
                    63, 31, 55, 23, 61, 29, 53, 21
                );
                return f32(matrix[y * 8 + x]) / 64.0;
            }
            
            fn blue_noise_final(coord: vec2<f32>) -> f32 {
                let x = i32(coord.x);
                let y = i32(coord.y);
                var n = x * 374761 + y * 668265;
                n = (n << 13) ^ n;
                n = n * (n * n * 15731 + 789221) + 1376312589;
                return f32(n & 0x7fffffff) / f32(0x7fffffff);
            }
            
            fn white_noise_final(coord: vec2<f32>) -> f32 {
                return fract(sin(dot(coord, vec2<f32>(12.9898, 78.233))) * 43758.5453);
            }
            
            fn ign_noise_final(coord: vec2<f32>) -> f32 {
                return fract(52.9829189 * fract(0.06711056 * coord.x + 0.00583715 * coord.y));
            }
            
            fn dither_channel_final(in_color: f32, coord: vec2<f32>, palette: f32, dither_type: i32, channel: i32) -> f32 {
                var threshold: f32 = 0.5;
                
                switch(dither_type) {
                    // Ordered (0-1)
                    case 0: { 
                        threshold = bayer4x4_final(coord);
                        let scaled = in_color * palette;
                        let dithered = scaled + threshold - 0.5;
                        return clamp(floor(dithered) / palette, 0.0, 1.0);
                    }
                    case 1: { 
                        threshold = bayer8x8_final(coord);
                        let scaled = in_color * palette;
                        let dithered = scaled + threshold - 0.5;
                        return clamp(floor(dithered) / palette, 0.0, 1.0);
                    }
                    // Noise (2-4)
                    case 2: { 
                        threshold = blue_noise_final(coord);
                        let scaled = in_color * palette;
                        let dithered = scaled + threshold - 0.5;
                        return clamp(floor(dithered) / palette, 0.0, 1.0);
                    }
                    case 3: { 
                        threshold = white_noise_final(coord);
                        let scaled = in_color * palette;
                        let dithered = scaled + threshold - 0.5;
                        return clamp(floor(dithered) / palette, 0.0, 1.0);
                    }
                    case 4: { 
                        threshold = ign_noise_final(coord);
                        let scaled = in_color * palette;
                        let dithered = scaled + threshold - 0.5;
                        return clamp(floor(dithered) / palette, 0.0, 1.0);
                    }
                    // Glitch (5-12)
                    case 5: { // Scanlines
                        let y = i32(coord.y);
                        let pattern = f32(y % 4);
                        threshold = pattern / 4.0;
                        let scaled = in_color * palette;
                        let dithered = scaled + threshold - 0.5;
                        return clamp(floor(dithered) / palette, 0.0, 1.0);
                    }
                    case 6: { // Checkerboard
                        let x = i32(coord.x) & 1;
                        let y = i32(coord.y) & 1;
                        threshold = f32(x ^ y);
                        let scaled = in_color * palette;
                        let dithered = scaled + threshold - 0.5;
                        return clamp(floor(dithered) / palette, 0.0, 1.0);
                    }
                    case 7: { // Stripes
                        let x = i32(coord.x);
                        threshold = f32(x % 8) / 8.0;
                        let scaled = in_color * palette;
                        let dithered = scaled + threshold - 0.5;
                        return clamp(floor(dithered) / palette, 0.0, 1.0);
                    }
                    case 8: { // Bit Crush
                        let step_size = 1.0 / palette;
                        let noise = white_noise_final(coord);
                        return floor((in_color + (noise - 0.5) * step_size) * palette) / palette;
                    }
                    case 9: { // 1-Bit Threshold
                        let noise = white_noise_final(coord);
                        let t = 0.5 + (noise - 0.5) * 0.15;
                        return select(0.0, 1.0, in_color > t);
                    }
                    case 10: { // Pixel Sort
                        let step_size = 1.0 / palette;
                        let y = i32(coord.y);
                        let offset = select(0.0, step_size * 0.5, f32(y % 2) > 0.5);
                        return clamp(round((in_color + offset) * palette) / palette, 0.0, 1.0);
                    }
                    case 11: { // Atkinson
                        let step_size = 1.0 / palette;
                        let noise = fract(sin(f32(i32(coord.x) * 7 + i32(coord.y) * 13)) * 43758.5453);
                        let dithered = in_color + (noise - 0.5) * step_size * 1.2;
                        return clamp(floor(dithered * palette + 0.4) / palette, 0.0, 1.0);
                    }
                    case 12: { // RGB Split
                        let offset = vec2<f32>(f32(channel) * 3.0, f32(channel) * 7.0);
                        threshold = white_noise_final(coord + offset);
                        let scaled = in_color * palette;
                        let dithered = scaled + threshold - 0.5;
                        return clamp(floor(dithered) / palette, 0.0, 1.0);
                    }
                    default: { return in_color; }
                }
            }
            
            fn apply_final_dither(color: vec3<f32>, coord: vec2<f32>) -> vec3<f32> {
                if (uniforms.final_dither_switch < 0.5) {
                    return color;
                }
                let dither_type = i32(uniforms.final_dither_type);
                let palette = uniforms.final_dither_amount;
                return vec3<f32>(
                    dither_channel_final(color.r, coord, palette, dither_type, 0),
                    dither_channel_final(color.g, coord, palette, dither_type, 1),
                    dither_channel_final(color.b, coord, palette, dither_type, 2)
                );
            }
            
            @fragment
            fn fs_main(@location(0) texcoord: vec2<f32>) -> @location(0) vec4<f32> {
                let block1_color = textureSample(block1_tex, block1_sampler, texcoord);
                let block2_color = textureSample(block2_tex, block2_sampler, texcoord);
                
                // Determine foreground/background based on final_key_order
                var fg = block1_color.rgb;
                var bg = block2_color.rgb;
                
                if (uniforms.final_key_order == 1) {
                    fg = block2_color.rgb;
                    bg = block1_color.rgb;
                }
                
                // Matrix Mixer
                let mixed = matrix_mix(fg, bg);
                
                // Final Mix with keying
                let final_color = final_mix(vec4<f32>(mixed, 1.0), vec4<f32>(bg, 1.0));
                
                // Apply final dither to output
                let pixel_coord = texcoord * vec2<f32>(textureDimensions(block1_tex));
                let dithered_color = apply_final_dither(final_color.rgb, pixel_coord);
                
                return vec4<f32>(dithered_color, 1.0);
            }
        "#;
        
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Block3 Stage2 Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_code.into()),
        });
        
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Block3 Stage2 BGL"),
            entries: &[
                // Uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Block 1 texture
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // Block 2 texture
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Block3 Stage2 Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        
        // Determine output format - use Bgra8Unorm (macOS native) for intermediate; surface blit handles sRGB
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Block3 Stage2 Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[StageVertex::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Bgra8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Block3 Stage2 Uniforms"),
            size: std::mem::size_of::<Stage2Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        (pipeline, bind_group_layout, uniforms)
    }
    
    /// Render Block 3
    pub fn render(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        block1_input_view: &wgpu::TextureView,
        block2_input_view: &wgpu::TextureView,
        params: &Block3Params,
    ) -> &wgpu::TextureView {
        // Update uniform buffers first (these don't borrow self mutably)
        self.update_stage1_uniforms(queue, params);
        self.update_stage2_uniforms(queue, params);
        
        // Stage 1a: Process Block 1
        let block1_output_view = &self.block1_buffer_a.view;
        let block1_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Block3 Stage1 B1 Bind Group"),
            layout: &self.stage1_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.stage1_uniforms_block1.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(block1_input_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.cached_sampler),
                },
            ],
        });
        
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Block3 Stage1 B1 Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: block1_output_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            
            render_pass.set_pipeline(&self.stage1_pipeline);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_bind_group(0, &block1_bind_group, &[]);
            render_pass.draw(0..6, 0..1);
        }
        
        // Stage 1b: Process Block 2
        let block2_output_view = &self.block2_buffer_a.view;
        let block2_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Block3 Stage1 B2 Bind Group"),
            layout: &self.stage1_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.stage1_uniforms_block2.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(block2_input_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.cached_sampler),
                },
            ],
        });
        
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Block3 Stage1 B2 Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: block2_output_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            
            render_pass.set_pipeline(&self.stage1_pipeline);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_bind_group(0, &block2_bind_group, &[]);
            render_pass.draw(0..6, 0..1);
        }
        
        // Stage 2: Matrix Mix + Final Mix
        let final_output_view = &self.block1_buffer_b.view;
        let stage2_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Block3 Stage2 Bind Group"),
            layout: &self.stage2_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.stage2_uniforms.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(block1_output_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.cached_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(block2_output_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&self.cached_sampler),
                },
            ],
        });
        
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Block3 Stage2 Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: final_output_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            
            render_pass.set_pipeline(&self.stage2_pipeline);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_bind_group(0, &stage2_bind_group, &[]);
            render_pass.draw(0..6, 0..1);
        }
        
        final_output_view
    }
    
    /// Update Stage 1 uniform buffers
    fn update_stage1_uniforms(&self, queue: &wgpu::Queue, params: &Block3Params) {
        // Block 1 uniforms
        let block1_uniforms = Stage1Uniforms {
            width: self.width as f32,
            height: self.height as f32,
            inv_width: 1.0 / self.width as f32,
            inv_height: 1.0 / self.height as f32,
            xy_displace: [params.block1_x_displace, params.block1_y_displace],
            z_displace: params.block1_z_displace,
            rotate: params.block1_rotate,
            shear_matrix: params.block1_shear_matrix.to_array(),
            kaleidoscope: params.block1_kaleidoscope_amount,
            kaleidoscope_slice: params.block1_kaleidoscope_slice,
            _pad1: 0.0,
            _pad2: 0.0,
            h_mirror: if params.block1_h_mirror { 1.0 } else { 0.0 },
            v_mirror: if params.block1_v_mirror { 1.0 } else { 0.0 },
            h_flip: if params.block1_h_flip { 1.0 } else { 0.0 },
            v_flip: if params.block1_v_flip { 1.0 } else { 0.0 },
            geo_overflow: params.block1_geo_overflow,
            rotate_mode: params.block1_rotate_mode,
            _pad3: 0.0,
            _pad4: 0.0,
            blur_amount: params.block1_blur_amount,
            blur_radius: params.block1_blur_radius,
            sharpen_amount: params.block1_sharpen_amount,
            sharpen_radius: params.block1_sharpen_radius,
            filters_boost: params.block1_filters_boost,
            _pad5: 0.0,
            _pad6: 0.0,
            _pad7: 0.0,
            colorize_switch: if params.block1_colorize_switch { 1.0 } else { 0.0 },
            colorize_mode: params.block1_colorize_hsb_rgb,
            _pad_pre_band1b: [0.0, 0.0],  // Padding to align colorize_band1 to 16 bytes
            colorize_band1: [params.block1_colorize_band1.x, params.block1_colorize_band1.y, params.block1_colorize_band1.z],
            _pad8: 0.0,
            colorize_band2: [params.block1_colorize_band2.x, params.block1_colorize_band2.y, params.block1_colorize_band2.z],
            _pad9: 0.0,
            colorize_band3: [params.block1_colorize_band3.x, params.block1_colorize_band3.y, params.block1_colorize_band3.z],
            _pad10: 0.0,
            colorize_band4: [params.block1_colorize_band4.x, params.block1_colorize_band4.y, params.block1_colorize_band4.z],
            _pad11: 0.0,
            colorize_band5: [params.block1_colorize_band5.x, params.block1_colorize_band5.y, params.block1_colorize_band5.z],
            _pad12: 0.0,
            dither_amount: params.block1_dither,
            dither_switch: if params.block1_dither_switch { 1.0 } else { 0.0 },
            dither_type: params.block1_dither_type,
            _pad13: 0.0,
        };
        
        queue.write_buffer(
            &self.stage1_uniforms_block1,
            0,
            unsafe { std::slice::from_raw_parts(&block1_uniforms as *const _ as *const u8, std::mem::size_of::<Stage1Uniforms>()) },
        );
        
        // Block 2 uniforms
        let block2_uniforms = Stage1Uniforms {
            width: self.width as f32,
            height: self.height as f32,
            inv_width: 1.0 / self.width as f32,
            inv_height: 1.0 / self.height as f32,
            xy_displace: [params.block2_x_displace, params.block2_y_displace],
            z_displace: params.block2_z_displace,
            rotate: params.block2_rotate,
            shear_matrix: params.block2_shear_matrix.to_array(),
            kaleidoscope: params.block2_kaleidoscope_amount,
            kaleidoscope_slice: params.block2_kaleidoscope_slice,
            _pad1: 0.0,
            _pad2: 0.0,
            h_mirror: if params.block2_h_mirror { 1.0 } else { 0.0 },
            v_mirror: if params.block2_v_mirror { 1.0 } else { 0.0 },
            h_flip: if params.block2_h_flip { 1.0 } else { 0.0 },
            v_flip: if params.block2_v_flip { 1.0 } else { 0.0 },
            geo_overflow: params.block2_geo_overflow,
            rotate_mode: params.block2_rotate_mode,
            _pad3: 0.0,
            _pad4: 0.0,
            blur_amount: params.block2_blur_amount,
            blur_radius: params.block2_blur_radius,
            sharpen_amount: params.block2_sharpen_amount,
            sharpen_radius: params.block2_sharpen_radius,
            filters_boost: params.block2_filters_boost,
            _pad5: 0.0,
            _pad6: 0.0,
            _pad7: 0.0,
            colorize_switch: if params.block2_colorize_switch { 1.0 } else { 0.0 },
            colorize_mode: params.block2_colorize_hsb_rgb,
            _pad_pre_band1b: [0.0, 0.0],  // Padding to align colorize_band1 to 16 bytes
            colorize_band1: [params.block2_colorize_band1.x, params.block2_colorize_band1.y, params.block2_colorize_band1.z],
            _pad8: 0.0,
            colorize_band2: [params.block2_colorize_band2.x, params.block2_colorize_band2.y, params.block2_colorize_band2.z],
            _pad9: 0.0,
            colorize_band3: [params.block2_colorize_band3.x, params.block2_colorize_band3.y, params.block2_colorize_band3.z],
            _pad10: 0.0,
            colorize_band4: [params.block2_colorize_band4.x, params.block2_colorize_band4.y, params.block2_colorize_band4.z],
            _pad11: 0.0,
            colorize_band5: [params.block2_colorize_band5.x, params.block2_colorize_band5.y, params.block2_colorize_band5.z],
            _pad12: 0.0,
            dither_amount: params.block2_dither,
            dither_switch: if params.block2_dither_switch { 1.0 } else { 0.0 },
            dither_type: params.block2_dither_type,
            _pad13: 0.0,
        };
        
        queue.write_buffer(
            &self.stage1_uniforms_block2,
            0,
            unsafe { std::slice::from_raw_parts(&block2_uniforms as *const _ as *const u8, std::mem::size_of::<Stage1Uniforms>()) },
        );
    }
    
    /// Update Stage 2 uniform buffer
    fn update_stage2_uniforms(&self, queue: &wgpu::Queue, params: &Block3Params) {
        let uniforms = Stage2Uniforms {
            matrix_mix_type: params.matrix_mix_type,
            matrix_mix_overflow: params.matrix_mix_overflow,
            _pad0: 0.0,
            _pad1: 0.0,
            bg_into_fg_red: [params.bg_rgb_into_fg_red.x, params.bg_rgb_into_fg_red.y, params.bg_rgb_into_fg_red.z],
            _pad2: 0.0,
            bg_into_fg_green: [params.bg_rgb_into_fg_green.x, params.bg_rgb_into_fg_green.y, params.bg_rgb_into_fg_green.z],
            _pad3: 0.0,
            bg_into_fg_blue: [params.bg_rgb_into_fg_blue.x, params.bg_rgb_into_fg_blue.y, params.bg_rgb_into_fg_blue.z],
            _pad4: 0.0,
            final_mix_amount: params.final_mix_amount,
            _pad5: 0.0,
            _pad6: 0.0,
            _pad7: 0.0,
            final_key_value: [params.final_key_value.x, params.final_key_value.y, params.final_key_value.z],
            final_key_threshold: params.final_key_threshold,
            final_key_soft: params.final_key_soft,
            final_mix_type: params.final_mix_type,
            final_mix_overflow: params.final_mix_overflow,
            final_key_order: params.final_key_order,
            _pad8: 0.0,
            final_dither_amount: params.final_dither,
            final_dither_switch: if params.final_dither_switch { 1.0 } else { 0.0 },
            final_dither_type: params.final_dither_type,
            _pad9: 0.0,
        };
        
        queue.write_buffer(
            &self.stage2_uniforms,
            0,
            unsafe { std::slice::from_raw_parts(&uniforms as *const _ as *const u8, std::mem::size_of::<Stage2Uniforms>()) },
        );
    }
    
    /// Get the output texture view (Stage 2 renders to block1_buffer_b)
    pub fn get_output_view(&self) -> &wgpu::TextureView {
        &self.block1_buffer_b.view
    }
    
    /// Get output texture for copying
    pub fn get_output_texture(&self) -> &wgpu::Texture {
        &self.block1_buffer_b.texture
    }
    
    /// Resize for new dimensions
    pub fn resize(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        
        // Recreate buffers
        self.block1_buffer_a = Texture::create_render_target_with_format(
            device, width, height, "Block3 B1 Buffer A", wgpu::TextureFormat::Bgra8Unorm,
        );
        self.block1_buffer_b = Texture::create_render_target_with_format(
            device, width, height, "Block3 B1 Buffer B", wgpu::TextureFormat::Bgra8Unorm,
        );
        self.block1_buffer_a.clear_to_black(queue);
        self.block1_buffer_b.clear_to_black(queue);
        
        self.block2_buffer_a = Texture::create_render_target_with_format(
            device, width, height, "Block3 B2 Buffer A", wgpu::TextureFormat::Bgra8Unorm,
        );
        self.block2_buffer_b = Texture::create_render_target_with_format(
            device, width, height, "Block3 B2 Buffer B", wgpu::TextureFormat::Bgra8Unorm,
        );
        self.block2_buffer_a.clear_to_black(queue);
        self.block2_buffer_b.clear_to_black(queue);
    }
}
