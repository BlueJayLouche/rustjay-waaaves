//! # Modular Block 2
//!
//! Three-stage architecture for Block 2 processing:
//! 1. Stage 1: Input sampling with geometric transforms
//! 2. Stage 2: Effects processing (HSB, blur, etc.)
//! 3. Stage 3: Mixing with FB2 feedback
//!
//! Unlike Block 1, Block 2 processes a single selectable input
//! (Block 1 output, Input 1, or Input 2) with FB2 feedback.

use crate::engine::blocks::{BlockResources, StageVertex};
use crate::params::Block2Params;


/// Vec3 type for uniforms - 16-byte aligned but only 12 bytes data
#[repr(C, align(16))]
#[derive(Debug, Copy, Clone, bytemuck::Zeroable)]
pub struct Vec3([f32; 3]);

impl Vec3 {
    pub const ZERO: Self = Self([0.0, 0.0, 0.0]);
    pub const ONE: Self = Self([1.0, 1.0, 1.0]);
}

impl From<glam::Vec3> for Vec3 {
    fn from(v: glam::Vec3) -> Self {
        Self([v.x, v.y, v.z])
    }
}

/// Stage 1 uniforms - Input sampling with transforms
#[repr(C, align(16))]
#[derive(Debug, Copy, Clone, bytemuck::Zeroable)]
struct Stage1Uniforms {
    // Resolution
    width: f32,
    height: f32,
    inv_width: f32,
    inv_height: f32,
    
    // Input transforms
    input_scale: f32,
    input_rotate: f32,
    input_x_displace: f32,
    input_y_displace: f32,
    input_z_displace: f32,
    
    // Switches (packed as bits)
    h_mirror: f32,
    v_mirror: f32,
    h_flip: f32,
    v_flip: f32,
    
    // Kaleidoscope
    kaleidoscope_amount: f32,
    kaleidoscope_slice: f32,
    
    // Input selection (0=block1, 1=input1, 2=input2)
    input_select: i32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

/// Stage 2 uniforms - Effects processing
/// Layout matches WGSL exactly - DO NOT REORDER
#[repr(C, align(16))]
#[derive(Debug, Copy, Clone, bytemuck::Zeroable)]
struct Stage2Uniforms {
    // Resolution (offset 0-16)
    width: f32,
    height: f32,
    inv_width: f32,
    inv_height: f32,
    
    // HSB attenuate - vec3 is 12 bytes, then _pad0 (offset 16-32)
    hsb_attenuate: [f32; 3],  // 12 bytes
    _pad0: f32,               // 4 bytes padding
    
    // Filters (offset 32-52)
    blur_amount: f32,
    blur_radius: f32,
    sharpen_amount: f32,
    sharpen_radius: f32,
    filters_boost: f32,
    
    // Posterize (offset 52-64) - NO padding here, matches WGSL
    posterize: f32,
    posterize_inv: f32,
    posterize_switch: i32,
    // Note: WGSL packs solarize immediately after posterize_switch
    
    // Solarize + Inverts (offset 64-84)
    solarize: f32,
    hue_invert: f32,
    sat_invert: f32,
    bright_invert: f32,
    rgb_invert: f32,
    
    // Overflow mode (offset 84-96, aligned to 16)
    geo_overflow: i32,
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
}

/// Stage 3 uniforms - Mixing and FB2
/// Layout matches WGSL exactly - DO NOT REORDER without updating WGSL
/// 
/// NOTE: Vec3 fields need 16-byte alignment. After key_order (i32) at offset 40,
/// we need 12 bytes of padding to reach offset 52 for the next vec3.
/// But WGSL puts _pad1 at 44 and _pad2 at 48, with fb2_hsb_offset at 48.
/// 
/// SOLUTION: Reorder fields to put all scalars together and vec3s at 16-byte boundaries:
/// - Scalars from offset 32-64 (key_soft, key_mode, key_order, etc.)
/// - First vec3 at offset 64 (fb2_hsb_offset)
#[repr(C, align(16))]
#[derive(Debug, Copy, Clone, bytemuck::Zeroable)]
struct Stage3Uniforms {
    // Mix params (offset 0-16)
    mix_amount: f32,          // 0
    mix_type: i32,            // 4
    mix_overflow: i32,        // 8
    _pad0: f32,               // 12
    
    // Input keying - first part (offset 16-32)
    key_value: [f32; 3],      // 16 (vec3, 12 bytes)
    key_threshold: f32,       // 28
    key_soft: f32,            // 32
    
    // Scalars that don't need special alignment (offset 32-64)
    key_mode: i32,            // 36
    key_order: i32,           // 40
    fb2_hue_shaper: f32,      // 44
    fb2_posterize: f32,       // 48
    fb2_posterize_switch: i32,// 52
    _pad1: f32,               // 56
    _pad2: f32,               // 60
    
    // FB2 Color adjustments - all vec3 at 16-byte aligned offsets (64, 80, 96)
    fb2_hsb_offset: [f32; 3],     // 64 (12 bytes)
    _pad3: f32,                   // 76
    fb2_hsb_attenuate: [f32; 3],  // 80 (12 bytes)
    _pad4: f32,                   // 92
    fb2_hsb_powmap: [f32; 3],     // 96 (12 bytes)
    _pad5: f32,                   // 108
    
    // FB2 Inverts (offset 112-128)
    fb2_hue_invert: f32,           // 112
    fb2_saturation_invert: f32,    // 116
    fb2_bright_invert: f32,        // 120
    fb2_rgb_invert: f32,           // 124
    
    // FB2 Geometric (offset 128-168)
    fb2_x_displace: f32,           // 128
    fb2_y_displace: f32,           // 132
    fb2_z_displace: f32,           // 136
    fb2_rotate: f32,               // 140
    fb2_kaleidoscope_amount: f32,  // 144
    fb2_kaleidoscope_slice: f32,   // 148
    fb2_h_mirror: f32,             // 152
    fb2_v_mirror: f32,             // 156
    fb2_h_flip: f32,               // 160
    fb2_v_flip: f32,               // 164
    
    // FB2 Filters (offset 168-180)
    fb2_blur_amount: f32,          // 168
    fb2_blur_radius: f32,          // 172
    fb2_sharpen_amount: f32,       // 176
    fb2_sharpen_radius: f32,       // 180
    
    // Shear matrix must be 16-byte aligned (offset 192)
    fb2_filters_boost: f32,        // 184
    _pad8: f32,                    // 188 (padding to align shear to 192)
    fb2_shear_matrix: [f32; 4],    // 192-207 (16 bytes, 16-byte aligned)
    
    // Delay and misc (offset 208-224)
    fb2_delay_time: i32,           // 208
    fb2_rotate_mode: i32,          // 212
    fb2_geo_overflow: i32,         // 216
    _pad10: i32,                   // 220
    
    // End padding - wgpu requires struct size to be 16-byte aligned
    _pad11: f32,                   // 224
    _pad12: f32,                   // 228
    _pad13: f32,                   // 232
    _pad14: f32,                   // 236
    _pad15: f32,                   // 240
    _pad16: f32,                   // 244
    _pad17: f32,                   // 248
    _pad18: f32,                   // 252
    _pad19: f32,                   // 256
    _pad20: f32,                   // 260
    _pad21: f32,                   // 264
    _pad22: f32,                   // 268
}

/// Modular Block 2 implementation
pub struct ModularBlock2 {
    pub resources: BlockResources,
    
    // Stage 1: Input sampling
    stage1_pipeline: wgpu::RenderPipeline,
    stage1_bind_group_layout: wgpu::BindGroupLayout,
    stage1_uniforms: wgpu::Buffer,
    
    // Stage 2: Effects
    stage2_pipeline: wgpu::RenderPipeline,
    stage2_bind_group_layout: wgpu::BindGroupLayout,
    stage2_uniforms: wgpu::Buffer,
    
    // Stage 3: Mixing & FB2
    stage3_pipeline: wgpu::RenderPipeline,
    stage3_bind_group_layout: wgpu::BindGroupLayout,
    stage3_uniforms: wgpu::Buffer,
    
    // Vertex buffer (shared across stages)
    vertex_buffer: wgpu::Buffer,
    
    // Dimensions
    width: u32,
    height: u32,
    
    // Cached sampler (avoids per-frame allocation)
    cached_sampler: wgpu::Sampler,
}

impl ModularBlock2 {
    /// Create new modular Block 2
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, width: u32, height: u32) -> Self {
        let resources = BlockResources::new(device, queue, width, height, "Block2");
        
        // Create vertex buffer
        let vertex_buffer = StageVertex::create_quad_buffer(device, queue);
        
        // Create Stage 1
        let (stage1_pipeline, stage1_bind_group_layout, stage1_uniforms) = 
            Self::create_stage1(device, queue, width, height);
        
        // Create Stage 2
        let (stage2_pipeline, stage2_bind_group_layout, stage2_uniforms) = 
            Self::create_stage2(device, queue, width, height);
        
        // Create Stage 3
        let (stage3_pipeline, stage3_bind_group_layout, stage3_uniforms) = 
            Self::create_stage3(device, queue, width, height);
        
        // Create cached sampler
        let cached_sampler = create_default_sampler(device);
        
        Self {
            resources,
            stage1_pipeline,
            stage1_bind_group_layout,
            stage1_uniforms,
            stage2_pipeline,
            stage2_bind_group_layout,
            stage2_uniforms,
            stage3_pipeline,
            stage3_bind_group_layout,
            stage3_uniforms,
            vertex_buffer,
            width,
            height,
            cached_sampler,
        }
    }
    
    /// Create Stage 1: Input sampling pipeline
    fn create_stage1(
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        width: u32,
        height: u32,
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
                width: f32,
                height: f32,
                inv_width: f32,
                inv_height: f32,
                input_scale: f32,
                input_rotate: f32,
                input_x_displace: f32,
                input_y_displace: f32,
                input_z_displace: f32,
                h_mirror: f32,
                v_mirror: f32,
                h_flip: f32,
                v_flip: f32,
                kaleidoscope_amount: f32,
                kaleidoscope_slice: f32,
                input_select: i32,
            };
            
            @group(0) @binding(0)
            var<uniform> uniforms: Uniforms;
            
            @group(0) @binding(1)
            var block1_tex: texture_2d<f32>;
            @group(0) @binding(2)
            var block1_sampler: sampler;
            
            @group(0) @binding(3)
            var input1_tex: texture_2d<f32>;
            @group(0) @binding(4)
            var input1_sampler: sampler;
            
            @group(0) @binding(5)
            var input2_tex: texture_2d<f32>;
            @group(0) @binding(6)
            var input2_sampler: sampler;
            
            fn transform_uv(uv: vec2<f32>) -> vec2<f32> {
                var result = uv - vec2<f32>(0.5);
                
                // Scale (Z displace) - apply BEFORE rotate to match Stage 3 order
                result = result / uniforms.input_z_displace;
                
                // Rotate
                let angle = uniforms.input_rotate * 0.0174533; // degrees to radians
                let cos_a = cos(angle);
                let sin_a = sin(angle);
                let rot_x = result.x * cos_a - result.y * sin_a;
                let rot_y = result.x * sin_a + result.y * cos_a;
                result = vec2<f32>(rot_x, rot_y);
                
                // Displace
                result = result + vec2<f32>(uniforms.input_x_displace, uniforms.input_y_displace);
                
                return result + vec2<f32>(0.5);
            }
            
            fn apply_kaleidoscope(uv: vec2<f32>) -> vec2<f32> {
                if (uniforms.kaleidoscope_amount <= 0.0 || uniforms.kaleidoscope_slice <= 0.0) {
                    return uv;
                }
                
                var centered = uv - vec2<f32>(0.5);
                let radius = length(centered);
                var angle = atan2(centered.y, centered.x);
                
                let slice_size = 6.28318530718 * uniforms.kaleidoscope_slice;
                let slice_index = floor(angle / slice_size);
                let slice_angle = angle - slice_index * slice_size;
                let mirrored_angle = abs(slice_angle - slice_size * 0.5);
                
                angle = mix(angle, mirrored_angle, uniforms.kaleidoscope_amount);
                centered = radius * vec2<f32>(cos(angle), sin(angle));
                
                return centered + vec2<f32>(0.5);
            }
            
            fn apply_mirrors(uv: vec2<f32>) -> vec2<f32> {
                var result = uv;
                if (uniforms.h_mirror > 0.5) {
                    result.x = abs(result.x * 2.0 - 1.0) * 0.5 + 0.25;
                }
                if (uniforms.v_mirror > 0.5) {
                    result.y = abs(result.y * 2.0 - 1.0) * 0.5 + 0.25;
                }
                if (uniforms.h_flip > 0.5) {
                    result.x = 1.0 - result.x;
                }
                if (uniforms.v_flip > 0.5) {
                    result.y = 1.0 - result.y;
                }
                return result;
            }
            
            @fragment
            fn fs_main(@location(0) texcoord: vec2<f32>) -> @location(0) vec4<f32> {
                // Transform UV
                var uv = transform_uv(texcoord);
                uv = apply_kaleidoscope(uv);
                uv = apply_mirrors(uv);
                
                // Select input based on input_select
                var color: vec4<f32>;
                if (uniforms.input_select == 0) {
                    color = textureSample(block1_tex, block1_sampler, uv);
                } else if (uniforms.input_select == 1) {
                    color = textureSample(input1_tex, input1_sampler, uv);
                } else {
                    color = textureSample(input2_tex, input2_sampler, uv);
                }
                
                return color;
            }
        "#;
        
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Block2 Stage 1 Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_code.into()),
        });
        
        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Block2 Stage 1 Bind Group Layout"),
            entries: &[
                // Uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Block1 texture
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
                // Input1 texture
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
                // Input2 texture
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Block2 Stage 1 Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Block2 Stage 1 Pipeline"),
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
        
        // Create uniform buffer
        let uniforms = Stage1Uniforms {
            width: width as f32,
            height: height as f32,
            inv_width: 1.0 / width as f32,
            inv_height: 1.0 / height as f32,
            input_scale: 1.0,
            input_rotate: 0.0,
            input_x_displace: 0.0,
            input_y_displace: 0.0,
            input_z_displace: 1.0,
            h_mirror: 0.0,
            v_mirror: 0.0,
            h_flip: 0.0,
            v_flip: 0.0,
            kaleidoscope_amount: 0.0,
            kaleidoscope_slice: 0.0,
            input_select: 0,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        };
        
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Block2 Stage 1 Uniforms"),
            size: std::mem::size_of::<Stage1Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        (pipeline, bind_group_layout, uniform_buffer)
    }
    
    /// Create Stage 2: Effects pipeline
    fn create_stage2(
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        width: u32,
        height: u32,
    ) -> (wgpu::RenderPipeline, wgpu::BindGroupLayout, wgpu::Buffer) {
        // Full Stage 2 shader with effects
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
                hsb_attenuate: vec3<f32>,
                _pad0: f32,
                blur_amount: f32,
                blur_radius: f32,
                sharpen_amount: f32,
                sharpen_radius: f32,
                filters_boost: f32,
                posterize: f32,
                posterize_inv: f32,
                posterize_switch: i32,
                solarize: f32,
                hue_invert: f32,
                sat_invert: f32,
                bright_invert: f32,
                rgb_invert: f32,
                geo_overflow: i32,
                _pad1: f32,
                _pad2: f32,
                _pad3: f32,
            };
            
            @group(0) @binding(0)
            var<uniform> uniforms: Uniforms;
            @group(0) @binding(1)
            var input_tex: texture_2d<f32>;
            @group(0) @binding(2)
            var input_sampler: sampler;
            
            // RGB to HSB conversion
            fn rgb_to_hsb(c: vec3<f32>) -> vec3<f32> {
                let max_val = max(max(c.r, c.g), c.b);
                let min_val = min(min(c.r, c.g), c.b);
                let delta = max_val - min_val;
                
                let b = max_val;
                var s = 0.0;
                if (max_val > 0.0) {
                    s = delta / max_val;
                }
                
                var h = 0.0;
                if (delta > 0.0) {
                    if (max_val == c.r) {
                        h = (c.g - c.b) / delta;
                        if (h < 0.0) { h = h + 6.0; }
                    } else if (max_val == c.g) {
                        h = (c.b - c.r) / delta + 2.0;
                    } else {
                        h = (c.r - c.g) / delta + 4.0;
                    }
                    h = h / 6.0;
                }
                
                return vec3<f32>(h, s, b);
            }
            
            // HSB to RGB conversion
            fn hsb_to_rgb(c: vec3<f32>) -> vec3<f32> {
                let h = c.x * 6.0;
                let s = c.y;
                let b = c.z;
                
                let i = floor(h);
                let f = h - i;
                let p = b * (1.0 - s);
                let q = b * (1.0 - s * f);
                let t = b * (1.0 - s * (1.0 - f));
                
                if (i < 1.0) { return vec3<f32>(b, t, p); }
                else if (i < 2.0) { return vec3<f32>(q, b, p); }
                else if (i < 3.0) { return vec3<f32>(p, b, t); }
                else if (i < 4.0) { return vec3<f32>(p, q, b); }
                else if (i < 5.0) { return vec3<f32>(t, p, b); }
                else { return vec3<f32>(b, p, q); }
            }
            
            // Apply blur - optimized 9-tap blur
            fn apply_blur(uv: vec2<f32>, amount: f32, radius: f32) -> vec3<f32> {
                if (amount <= 0.001 || radius <= 0.001) {
                    return textureSample(input_tex, input_sampler, uv).rgb;
                }
                
                let texel_size = vec2<f32>(uniforms.inv_width, uniforms.inv_height);
                let blur_radius = radius * 15.0; // Increased scale for comparable blur
                
                // 9-tap box blur (3x3 kernel) - much faster than 81 samples
                var result = vec3<f32>(0.0);
                
                // Center sample (weight 4)
                result = result + textureSample(input_tex, input_sampler, uv).rgb * 4.0;
                
                // Edge samples (weight 2 each)
                result = result + textureSample(input_tex, input_sampler, uv + vec2<f32>( blur_radius, 0.0) * texel_size).rgb * 2.0;
                result = result + textureSample(input_tex, input_sampler, uv + vec2<f32>(-blur_radius, 0.0) * texel_size).rgb * 2.0;
                result = result + textureSample(input_tex, input_sampler, uv + vec2<f32>(0.0,  blur_radius) * texel_size).rgb * 2.0;
                result = result + textureSample(input_tex, input_sampler, uv + vec2<f32>(0.0, -blur_radius) * texel_size).rgb * 2.0;
                
                // Corner samples (weight 1 each)
                result = result + textureSample(input_tex, input_sampler, uv + vec2<f32>( blur_radius,  blur_radius) * texel_size).rgb;
                result = result + textureSample(input_tex, input_sampler, uv + vec2<f32>(-blur_radius,  blur_radius) * texel_size).rgb;
                result = result + textureSample(input_tex, input_sampler, uv + vec2<f32>( blur_radius, -blur_radius) * texel_size).rgb;
                result = result + textureSample(input_tex, input_sampler, uv + vec2<f32>(-blur_radius, -blur_radius) * texel_size).rgb;
                
                // Normalize (total weight = 16)
                result = result / 16.0;
                
                let original = textureSample(input_tex, input_sampler, uv).rgb;
                return mix(original, result, amount);
            }
            
            // Apply sharpen
            fn apply_sharpen(uv: vec2<f32>, amount: f32, radius: f32, color: vec3<f32>) -> vec3<f32> {
                if (amount <= 0.001 || radius <= 0.001) {
                    return color;
                }
                
                let texel_size = vec2<f32>(uniforms.inv_width, uniforms.inv_height);
                let sharpen_radius = radius * 2.0;
                
                let left = textureSample(input_tex, input_sampler, uv + vec2<f32>(-sharpen_radius, 0.0) * texel_size).rgb;
                let right = textureSample(input_tex, input_sampler, uv + vec2<f32>(sharpen_radius, 0.0) * texel_size).rgb;
                let up = textureSample(input_tex, input_sampler, uv + vec2<f32>(0.0, -sharpen_radius) * texel_size).rgb;
                let down = textureSample(input_tex, input_sampler, uv + vec2<f32>(0.0, sharpen_radius) * texel_size).rgb;
                
                let laplacian = color * 4.0 - left - right - up - down;
                let sharpened = color + laplacian * amount * (1.0 + uniforms.filters_boost);
                
                return clamp(sharpened, vec3<f32>(0.0), vec3<f32>(1.0));
            }
            
            // Apply posterize
            fn apply_posterize(color: vec3<f32>, levels: f32, levels_inv: f32, switch_mode: i32) -> vec3<f32> {
                if (levels <= 1.0) { return color; }
                
                if (switch_mode == 0) {
                    // RGB posterize
                    return floor(color * levels) * levels_inv;
                } else {
                    // HSB posterize
                    var hsb = rgb_to_hsb(color);
                    hsb = floor(hsb * levels) * levels_inv;
                    return hsb_to_rgb(hsb);
                }
            }
            
            // Apply solarize
            fn apply_solarize(color: vec3<f32>, amount: f32) -> vec3<f32> {
                if (amount <= 0.001) { return color; }
                var result = color;
                if (result.r > 0.5) { result.r = 1.0 - result.r; }
                if (result.g > 0.5) { result.g = 1.0 - result.g; }
                if (result.b > 0.5) { result.b = 1.0 - result.b; }
                return mix(color, result, amount);
            }
            
            // Apply inverts
            fn apply_inverts(color: vec3<f32>) -> vec3<f32> {
                var result = color;
                
                // RGB invert
                if (uniforms.rgb_invert > 0.5) {
                    result = vec3<f32>(1.0) - result;
                }
                
                // HSB-based inverts
                if (uniforms.hue_invert > 0.5 || uniforms.sat_invert > 0.5 || uniforms.bright_invert > 0.5) {
                    var hsb = rgb_to_hsb(result);
                    
                    if (uniforms.hue_invert > 0.5) {
                        hsb.x = fract(hsb.x + 0.5);
                    }
                    if (uniforms.sat_invert > 0.5) {
                        hsb.y = 1.0 - hsb.y;
                    }
                    if (uniforms.bright_invert > 0.5) {
                        hsb.z = 1.0 - hsb.z;
                    }
                    
                    result = hsb_to_rgb(hsb);
                }
                
                return result;
            }
            
            // Handle overflow modes
            fn handle_overflow(color: vec3<f32>) -> vec3<f32> {
                if (uniforms.geo_overflow == 0) {
                    // Clamp
                    return clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));
                } else if (uniforms.geo_overflow == 1) {
                    // Wrap
                    return fract(color);
                } else {
                    // Mirror
                    return abs(fract(color + vec3<f32>(1.0)) - vec3<f32>(0.5)) * 2.0;
                }
            }
            
            @fragment
            fn fs_main(@location(0) texcoord: vec2<f32>) -> @location(0) vec4<f32> {
                var color = textureSample(input_tex, input_sampler, texcoord).rgb;
                
                // Check if any HSB processing needed
                let needs_hsb = uniforms.hsb_attenuate.x != 1.0 || 
                               uniforms.hsb_attenuate.y != 1.0 || 
                               uniforms.hsb_attenuate.z != 1.0 ||
                               uniforms.hue_invert > 0.5 ||
                               uniforms.sat_invert > 0.5 ||
                               uniforms.bright_invert > 0.5 ||
                               uniforms.posterize_switch == 1 ||
                               uniforms.solarize > 0.001;
                
                if (needs_hsb) {
                    var hsb = rgb_to_hsb(color);
                    
                    // Apply HSB attenuate
                    hsb = hsb * uniforms.hsb_attenuate;
                    
                    hsb.x = fract(hsb.x); // Wrap hue
                    color = hsb_to_rgb(hsb);
                }
                
                // Apply blur
                color = apply_blur(texcoord, uniforms.blur_amount, uniforms.blur_radius);
                
                // Apply sharpen
                color = apply_sharpen(texcoord, uniforms.sharpen_amount, uniforms.sharpen_radius, color);
                
                // Apply solarize
                color = apply_solarize(color, uniforms.solarize);
                
                // Apply inverts
                color = apply_inverts(color);
                
                // Apply posterize
                color = apply_posterize(color, uniforms.posterize, uniforms.posterize_inv, uniforms.posterize_switch);
                
                // Handle overflow
                color = handle_overflow(color);
                
                return vec4<f32>(color, 1.0);
            }
        "#;
        
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Block2 Stage 2 Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_code.into()),
        });
        
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Block2 Stage 2 Bind Group Layout"),
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
            label: Some("Block2 Stage 2 Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Block2 Stage 2 Pipeline"),
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
        
        // Create dummy uniform buffer (will be replaced with real one)
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Block2 Stage 2 Uniforms"),
            size: std::mem::size_of::<Stage2Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        (pipeline, bind_group_layout, uniform_buffer)
    }
    
    /// Create Stage 3: Mixing & FB2 pipeline
    fn create_stage3(
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        width: u32,
        height: u32,
    ) -> (wgpu::RenderPipeline, wgpu::BindGroupLayout, wgpu::Buffer) {
        // Full Stage 3 shader with mixing, keying, and FB2 transforms
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
                // Mix params (offset 0-16)
                mix_amount: f32,
                mix_type: i32,
                mix_overflow: i32,
                _pad0: f32,
                
                // Input keying - first part (offset 16-32)
                key_value: vec3<f32>,
                key_threshold: f32,
                key_soft: f32,
                
                // Scalars that don't need special alignment (offset 32-64)
                key_mode: i32,
                key_order: i32,
                fb2_hue_shaper: f32,
                fb2_posterize: f32,
                fb2_posterize_switch: i32,
                _pad1: f32,
                _pad2: f32,
                
                // FB2 Color adjustments - all vec3 at 16-byte aligned offsets (64, 80, 96)
                fb2_hsb_offset: vec3<f32>,
                _pad3: f32,
                fb2_hsb_attenuate: vec3<f32>,
                _pad4: f32,
                fb2_hsb_powmap: vec3<f32>,
                _pad5: f32,
                
                // FB2 Inverts (offset 112-128)
                fb2_hue_invert: f32,
                fb2_saturation_invert: f32,
                fb2_bright_invert: f32,
                fb2_rgb_invert: f32,
                
                // FB2 Geometric (offset 128-168)
                fb2_x_displace: f32,
                fb2_y_displace: f32,
                fb2_z_displace: f32,
                fb2_rotate: f32,
                fb2_kaleidoscope_amount: f32,
                fb2_kaleidoscope_slice: f32,
                fb2_h_mirror: f32,
                fb2_v_mirror: f32,
                fb2_h_flip: f32,
                fb2_v_flip: f32,
                
                // FB2 Filters (offset 168-180)
                fb2_blur_amount: f32,
                fb2_blur_radius: f32,
                fb2_sharpen_amount: f32,
                fb2_sharpen_radius: f32,
                
                // Shear matrix must be 16-byte aligned (offset 192)
                fb2_filters_boost: f32,        // 184
                _pad8: f32,                    // 188 (padding to align shear to 192)
                fb2_shear_matrix: vec4<f32>,   // 192-207
                
                // Delay and misc (offset 208-224)
                fb2_delay_time: i32,
                fb2_rotate_mode: i32,
                fb2_geo_overflow: i32,
                _pad10: i32,
                
                // End padding - wgpu requires struct size to be 16-byte aligned
                // and adds implicit padding. We need explicit padding to match.
                _pad11: f32,
                _pad12: f32,
                _pad13: f32,
                _pad14: f32,
                _pad15: f32,
                _pad16: f32,
                _pad17: f32,
                _pad18: f32,
                _pad19: f32,
                _pad20: f32,
                _pad21: f32,
                _pad22: f32,
            };
            
            @group(0) @binding(0)
            var<uniform> uniforms: Uniforms;
            
            @group(0) @binding(1)
            var input_tex: texture_2d<f32>;
            @group(0) @binding(2)
            var input_sampler: sampler;
            
            @group(0) @binding(3)
            var fb2_tex: texture_2d<f32>;
            @group(0) @binding(4)
            var fb2_sampler: sampler;
            
            @group(0) @binding(5)
            var delay_tex: texture_2d<f32>;
            @group(0) @binding(6)
            var delay_sampler: sampler;
            
            // RGB to HSB conversion
            fn rgb_to_hsb(c: vec3<f32>) -> vec3<f32> {
                let max_val = max(max(c.r, c.g), c.b);
                let min_val = min(min(c.r, c.g), c.b);
                let delta = max_val - min_val;
                
                let b = max_val;
                var s = 0.0;
                if (max_val > 0.0) {
                    s = delta / max_val;
                }
                
                var h = 0.0;
                if (delta > 0.0) {
                    if (max_val == c.r) {
                        h = (c.g - c.b) / delta;
                        if (h < 0.0) { h = h + 6.0; }
                    } else if (max_val == c.g) {
                        h = (c.b - c.r) / delta + 2.0;
                    } else {
                        h = (c.r - c.g) / delta + 4.0;
                    }
                    h = h / 6.0;
                }
                
                return vec3<f32>(h, s, b);
            }
            
            // HSB to RGB conversion
            fn hsb_to_rgb(c: vec3<f32>) -> vec3<f32> {
                let h = c.x * 6.0;
                let s = c.y;
                let b = c.z;
                
                let i = floor(h);
                let f = h - i;
                let p = b * (1.0 - s);
                let q = b * (1.0 - s * f);
                let t = b * (1.0 - s * (1.0 - f));
                
                if (i < 1.0) { return vec3<f32>(b, t, p); }
                else if (i < 2.0) { return vec3<f32>(q, b, p); }
                else if (i < 3.0) { return vec3<f32>(p, b, t); }
                else if (i < 4.0) { return vec3<f32>(p, q, b); }
                else if (i < 5.0) { return vec3<f32>(t, p, b); }
                else { return vec3<f32>(b, p, q); }
            }
            
            // Apply overflow modes
            fn apply_overflow(color: vec3<f32>, mode: i32) -> vec3<f32> {
                switch(mode) {
                    case 0: { return clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)); }
                    case 1: { return fract(color); }
                    case 2: { return abs(fract(color * 0.5 + 0.5) * 2.0 - 1.0); }
                    default: { return color; }
                }
            }
            
            // Mix types: 0=lerp, 1=add, 2=diff, 3=mult, 4=dodge
            // Note: mix_amount represents how much of 'b' to blend into 'a'
            // amount=0.0: 100% a, 0% b
            // amount=1.0: 0% a, 100% b
            fn mix_colors(a: vec3<f32>, b: vec3<f32>, amount: f32, mix_type: i32, overflow: i32) -> vec3<f32> {
                var result: vec3<f32>;
                
                switch(mix_type) {
                    case 0: { result = mix(a, b, amount); }
                    case 1: { result = a + b * amount; }
                    case 2: { result = abs(a - b) * amount + a * (1.0 - amount); }
                    case 3: { result = mix(a, a * b, amount); }
                    case 4: { result = mix(a, a / (1.0 - b + 0.001), amount); }
                    default: { result = mix(a, b, amount); }
                }
                
                return apply_overflow(result, overflow);
            }
            
            // Calculate key mix amount
            fn calculate_key_mix(color: vec3<f32>) -> f32 {
                if (uniforms.key_threshold >= 0.999) {
                    return 0.0;
                }
                
                var dist: f32;
                if (uniforms.key_mode == 0) {
                    // Luma key
                    let color_luma = dot(color, vec3<f32>(0.299, 0.587, 0.114));
                    let key_luma = dot(uniforms.key_value, vec3<f32>(0.299, 0.587, 0.114));
                    dist = abs(color_luma - key_luma);
                } else {
                    // Chroma key
                    dist = distance(color, uniforms.key_value);
                }
                
                let soft = max(uniforms.key_soft, 0.001);
                
                if (dist < uniforms.key_threshold) {
                    let edge_distance = uniforms.key_threshold - dist;
                    return min(edge_distance / soft, 1.0);
                }
                
                return 0.0;
            }
            
            // Mix with keying
            fn mix_with_key(fg: vec3<f32>, bg: vec3<f32>, amount: f32, mix_type: i32, overflow: i32, key_order: i32) -> vec3<f32> {
                let keying_enabled = uniforms.key_threshold < 0.999;
                
                if (!keying_enabled) {
                    return mix_colors(fg, bg, amount, mix_type, overflow);
                }
                
                if (key_order == 0) {
                    // Key first, then mix
                    let key_amount = calculate_key_mix(fg);
                    let keyed_fg = mix(fg, bg, key_amount);
                    return mix_colors(keyed_fg, bg, amount, mix_type, overflow);
                } else {
                    // Mix first, then key
                    let mixed = mix_colors(fg, bg, amount, mix_type, overflow);
                    let key_amount = calculate_key_mix(mixed);
                    return mix(mixed, bg, key_amount);
                }
            }
            
            // FB2 UV transforms
            fn transform_fb2_uv(uv: vec2<f32>) -> vec2<f32> {
                var result = uv - vec2<f32>(0.5);
                
                // Scale
                if (uniforms.fb2_z_displace > 0.001) {
                    result = result / uniforms.fb2_z_displace;
                }
                
                // Rotate
                if (abs(uniforms.fb2_rotate) > 0.001) {
                    let angle = radians(uniforms.fb2_rotate);
                    let cos_r = cos(angle);
                    let sin_r = sin(angle);
                    let rot_x = result.x * cos_r - result.y * sin_r;
                    let rot_y = result.x * sin_r + result.y * cos_r;
                    result = vec2<f32>(rot_x, rot_y);
                }
                
                // Apply shear
                let shear = uniforms.fb2_shear_matrix;
                result = vec2<f32>(
                    result.x * shear.x + result.y * shear.y,
                    result.x * shear.z + result.y * shear.w
                );
                
                // Displace
                result = result + vec2<f32>(0.5) + vec2<f32>(uniforms.fb2_x_displace, uniforms.fb2_y_displace);
                
                return result;
            }
            
            // Apply kaleidoscope
            fn apply_kaleidoscope(uv: vec2<f32>) -> vec2<f32> {
                let amount = uniforms.fb2_kaleidoscope_amount;
                let slice = uniforms.fb2_kaleidoscope_slice;
                
                if (amount <= 0.001 || slice <= 0.001) {
                    return uv;
                }
                
                var centered = uv - vec2<f32>(0.5);
                let radius = length(centered);
                var angle = atan2(centered.y, centered.x);
                
                let slice_size = 6.28318530718 * slice;
                let slice_index = floor(angle / slice_size);
                let slice_angle = angle - slice_index * slice_size;
                let mirrored_angle = abs(slice_angle - slice_size * 0.5);
                
                let kaleido_angle = mix(angle, slice_index * slice_size + mirrored_angle, amount);
                let kaleido_uv = vec2<f32>(cos(kaleido_angle) * radius, sin(kaleido_angle) * radius) + vec2<f32>(0.5);
                
                return mix(uv, kaleido_uv, amount);
            }
            
            // Apply mirrors and flips
            fn apply_mirrors(uv: vec2<f32>) -> vec2<f32> {
                var result = uv;
                
                if (uniforms.fb2_h_mirror > 0.5) {
                    result.x = abs(result.x - 0.5) + 0.5;
                    result.x = 1.0 - result.x;
                }
                
                if (uniforms.fb2_v_mirror > 0.5) {
                    result.y = abs(result.y - 0.5) + 0.5;
                    result.y = 1.0 - result.y;
                }
                
                if (uniforms.fb2_h_flip > 0.5) {
                    result.x = 1.0 - result.x;
                }
                
                if (uniforms.fb2_v_flip > 0.5) {
                    result.y = 1.0 - result.y;
                }
                
                return result;
            }
            
            // Apply blur to FB2
            fn apply_blur(uv: vec2<f32>, amount: f32, radius: f32) -> vec3<f32> {
                if (amount <= 0.001 || radius <= 0.001) {
                    return textureSample(fb2_tex, fb2_sampler, uv).rgb;
                }
                
                let texel_size = vec2<f32>(1.0 / 640.0, 1.0 / 480.0);
                let blur_radius = radius * 5.0;
                
                var sum = vec3<f32>(0.0);
                var total_weight = 0.0;
                
                for (var x: i32 = -1; x <= 1; x = x + 1) {
                    for (var y: i32 = -1; y <= 1; y = y + 1) {
                        let offset = vec2<f32>(f32(x), f32(y)) * texel_size * blur_radius;
                        let weight = 1.0 - (abs(f32(x)) + abs(f32(y))) * 0.25;
                        sum = sum + textureSample(fb2_tex, fb2_sampler, uv + offset).rgb * weight;
                        total_weight = total_weight + weight;
                    }
                }
                
                let blurred = sum / total_weight;
                let original = textureSample(fb2_tex, fb2_sampler, uv).rgb;
                return mix(original, blurred, amount);
            }
            
            // Apply sharpen to FB2
            fn apply_sharpen(uv: vec2<f32>, amount: f32, radius: f32, color: vec3<f32>) -> vec3<f32> {
                if (amount <= 0.001 || radius <= 0.001) {
                    return color;
                }
                
                let texel_size = vec2<f32>(1.0 / 640.0, 1.0 / 480.0);
                let sharpen_radius = radius * 2.0;
                
                let left = textureSample(fb2_tex, fb2_sampler, uv + vec2<f32>(-sharpen_radius, 0.0) * texel_size).rgb;
                let right = textureSample(fb2_tex, fb2_sampler, uv + vec2<f32>(sharpen_radius, 0.0) * texel_size).rgb;
                let up = textureSample(fb2_tex, fb2_sampler, uv + vec2<f32>(0.0, -sharpen_radius) * texel_size).rgb;
                let down = textureSample(fb2_tex, fb2_sampler, uv + vec2<f32>(0.0, sharpen_radius) * texel_size).rgb;
                
                let laplacian = (left + right + up + down) * 0.25 - color;
                let boosted_amount = amount * (1.0 + uniforms.fb2_filters_boost);
                
                return clamp(color + laplacian * boosted_amount, vec3<f32>(0.0), vec3<f32>(1.0));
            }
            
            // Apply FB2 color adjustments
            fn apply_fb2_color(color: vec3<f32>) -> vec3<f32> {
                var result = color;
                
                // Check if any HSB processing needed
                let needs_hsb = length(uniforms.fb2_hsb_offset) > 0.001 ||
                               length(uniforms.fb2_hsb_attenuate - vec3<f32>(1.0)) > 0.001 ||
                               length(uniforms.fb2_hsb_powmap - vec3<f32>(1.0)) > 0.001 ||
                               uniforms.fb2_hue_shaper > 0.001 ||
                               uniforms.fb2_hue_invert > 0.5 ||
                               uniforms.fb2_saturation_invert > 0.5 ||
                               uniforms.fb2_bright_invert > 0.5;
                
                if (needs_hsb) {
                    var hsb = rgb_to_hsb(result);
                    
                    // Hue shaper
                    if (uniforms.fb2_hue_shaper > 0.001) {
                        hsb.x = fract(abs(hsb.x + uniforms.fb2_hue_shaper * sin(hsb.x * 0.3184713)));
                    }
                    
                    // Powmap
                    if (length(uniforms.fb2_hsb_powmap - vec3<f32>(1.0)) > 0.001) {
                        hsb = pow(abs(hsb), uniforms.fb2_hsb_powmap);
                    }
                    
                    // Attenuate
                    hsb = hsb * uniforms.fb2_hsb_attenuate;
                    
                    // Offset
                    hsb.x = fract(hsb.x + uniforms.fb2_hsb_offset.x);
                    hsb.y = clamp(hsb.y + uniforms.fb2_hsb_offset.y, 0.0, 1.0);
                    hsb.z = clamp(hsb.z + uniforms.fb2_hsb_offset.z, 0.0, 1.0);
                    
                    // Inverts
                    if (uniforms.fb2_hue_invert > 0.5) { hsb.x = 1.0 - hsb.x; }
                    if (uniforms.fb2_saturation_invert > 0.5) { hsb.y = 1.0 - hsb.y; }
                    if (uniforms.fb2_bright_invert > 0.5) { hsb.z = 1.0 - hsb.z; }
                    
                    hsb.x = fract(hsb.x);
                    hsb.y = clamp(hsb.y, 0.0, 1.0);
                    hsb.z = clamp(hsb.z, 0.0, 1.0);
                    
                    result = hsb_to_rgb(hsb);
                }
                
                // Posterize
                if (uniforms.fb2_posterize > 1.0) {
                    let steps = uniforms.fb2_posterize;
                    if (uniforms.fb2_posterize_switch > 0) {
                        var hsb = rgb_to_hsb(result);
                        hsb = floor(hsb * steps) / steps;
                        result = hsb_to_rgb(hsb);
                    } else {
                        result = floor(result * steps) / steps;
                    }
                }
                
                // RGB invert
                if (uniforms.fb2_rgb_invert > 0.5) {
                    result = vec3<f32>(1.0) - result;
                }
                
                return result;
            }
            
            // Sample FB2 with all transforms and filters
            fn sample_fb2(texcoord: vec2<f32>) -> vec3<f32> {
                var uv = texcoord;
                
                // Apply geometric transforms
                uv = apply_kaleidoscope(uv);
                uv = apply_mirrors(uv);
                uv = transform_fb2_uv(uv);
                
                // Sample from appropriate source
                var color: vec3<f32>;
                if (uniforms.fb2_delay_time > 0) {
                    color = textureSample(delay_tex, delay_sampler, uv).rgb;
                } else {
                    color = textureSample(fb2_tex, fb2_sampler, uv).rgb;
                }
                
                // Apply blur
                if (uniforms.fb2_blur_amount > 0.001) {
                    color = apply_blur(uv, uniforms.fb2_blur_amount, uniforms.fb2_blur_radius);
                }
                
                // Apply sharpen
                if (uniforms.fb2_sharpen_amount > 0.001) {
                    color = apply_sharpen(uv, uniforms.fb2_sharpen_amount, uniforms.fb2_sharpen_radius, color);
                }
                
                // Apply color adjustments
                color = apply_fb2_color(color);
                
                return color;
            }
            
            @fragment
            fn fs_main(@location(0) texcoord: vec2<f32>) -> @location(0) vec4<f32> {
                // Sample input
                let input_color = textureSample(input_tex, input_sampler, texcoord).rgb;
                
                // Sample FB2 with transforms
                let fb2_color = sample_fb2(texcoord);
                
                // Mix with keying
                var result = input_color;
                if (uniforms.mix_amount > 0.001) {
                    result = mix_with_key(input_color, fb2_color, uniforms.mix_amount, 
                                         uniforms.mix_type, uniforms.mix_overflow, uniforms.key_order);
                }
                
                return vec4<f32>(result, 1.0);
            }
        "#;
        
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Block2 Stage 3 Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_code.into()),
        });
        
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Block2 Stage 3 Bind Group Layout"),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // FB2 texture
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
                // Delay texture
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Block2 Stage 3 Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Block2 Stage 3 Pipeline"),
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
        
        // Create uniform buffer
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Block2 Stage 3 Uniforms"),
            size: std::mem::size_of::<Stage3Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        (pipeline, bind_group_layout, uniform_buffer)
    }
}

impl ModularBlock2 {
    /// Render Block 2 with all three stages
    pub fn render(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        block1_view: &wgpu::TextureView,
        input1_view: &wgpu::TextureView,
        input2_view: &wgpu::TextureView,
        output_view: &wgpu::TextureView,
        params: &Block2Params,
    ) {
        // Stage 1: Input sampling - writes to input_view (ping-pong buffer B)
        self.render_stage1(encoder, device, queue, block1_view, input1_view, input2_view, params);
        
        // After Stage 1, the result is in buffer B (input_view).
        // Swap so buffer B becomes output_view for Stage 2 to read from.
        self.resources.swap();
        
        // Stage 2: Effects (if any effects are enabled)
        // Reads from output_view (now buffer B), writes to input_view (buffer A)
        if self.has_effects_enabled(params) {
            self.render_stage2(encoder, device, queue, params);
            // After Stage 2, result is in buffer A. Swap for Stage 3.
            self.resources.swap();
        }
        
        // Stage 3: Mixing with FB2
        // Reads from output_view (buffer with Stage 1/2 result)
        self.render_stage3(encoder, device, queue, output_view, params);
        
        // Note: swap() is intentionally NOT called here.
        // Stage 3 renders to the external output_view, not to the ping-pong buffers.
        // The current ping-pong state is preserved for next frame's Stage 1.
    }
    
    /// Render Stage 1: Input sampling with transforms
    fn render_stage1(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        block1_view: &wgpu::TextureView,
        input1_view: &wgpu::TextureView,
        input2_view: &wgpu::TextureView,
        params: &Block2Params,
    ) {
        // Update uniforms
        self.update_stage1_uniforms(queue, params);
        
        // Create bind group using cached sampler
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Block2 Stage 1 Bind Group"),
            layout: &self.stage1_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.stage1_uniforms.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(block1_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.cached_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(input1_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&self.cached_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(input2_view),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::Sampler(&self.cached_sampler),
                },
            ],
        });
        
        // Render to the input view of the ping-pong buffer
        let target_view = self.resources.get_input_view();
        
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Block2 Stage 1 Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target_view,
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
            render_pass.set_bind_group(0, &bind_group, &[]);
            render_pass.draw(0..6, 0..1);
        }
    }
    
    /// Check if any effects are enabled for Stage 2
    fn has_effects_enabled(&self, params: &Block2Params) -> bool {
        // Check if any effect parameter is non-default
        params.block2_input_blur_amount > 0.001
            || params.block2_input_sharpen_amount > 0.001
            || params.block2_input_hsb_attenuate.x != 1.0
            || params.block2_input_hsb_attenuate.y != 1.0
            || params.block2_input_hsb_attenuate.z != 1.0
            || params.block2_input_posterize > 1.0
            || params.block2_input_solarize
            || params.block2_input_hue_invert
            || params.block2_input_saturation_invert
            || params.block2_input_bright_invert
            || params.block2_input_rgb_invert
    }
    
    /// Render Stage 2: Effects processing
    fn render_stage2(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        params: &Block2Params,
    ) {
        // Update uniforms
        self.update_stage2_uniforms(queue, params);
        
        // Create bind group using cached sampler
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Block2 Stage 2 Bind Group"),
            layout: &self.stage2_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.stage2_uniforms.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(self.resources.get_output_view()),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.cached_sampler),
                },
            ],
        });
        
        // Render to the other buffer (ping-pong)
        let target_view = self.resources.get_input_view();
        
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Block2 Stage 2 Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target_view,
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
            render_pass.set_bind_group(0, &bind_group, &[]);
            render_pass.draw(0..6, 0..1);
        }
        
        // Note: Buffer swap is handled by the caller (render())
    }
    
    /// Update Stage 2 uniforms from params
    fn update_stage2_uniforms(&self, queue: &wgpu::Queue, params: &Block2Params) {
        let hsb_attenuate = params.block2_input_hsb_attenuate;
        let uniforms = Stage2Uniforms {
            width: self.width as f32,
            height: self.height as f32,
            inv_width: 1.0 / self.width as f32,
            inv_height: 1.0 / self.height as f32,
            hsb_attenuate: [hsb_attenuate.x, hsb_attenuate.y, hsb_attenuate.z],
            _pad0: 0.0,
            blur_amount: params.block2_input_blur_amount,
            blur_radius: params.block2_input_blur_radius,
            sharpen_amount: params.block2_input_sharpen_amount,
            sharpen_radius: params.block2_input_sharpen_radius,
            filters_boost: params.block2_input_filters_boost,
            posterize: params.block2_input_posterize,
            posterize_inv: 1.0 / params.block2_input_posterize.max(1.0),
            posterize_switch: if params.block2_input_posterize_switch { 1 } else { 0 },
            solarize: if params.block2_input_solarize { 1.0 } else { 0.0 },
            hue_invert: if params.block2_input_hue_invert { 1.0 } else { 0.0 },
            sat_invert: if params.block2_input_saturation_invert { 1.0 } else { 0.0 },
            bright_invert: if params.block2_input_bright_invert { 1.0 } else { 0.0 },
            rgb_invert: if params.block2_input_rgb_invert { 1.0 } else { 0.0 },
            geo_overflow: params.block2_input_geo_overflow,
            _pad1: 0.0,
            _pad2: 0.0,
            _pad3: 0.0,
        };
        
        let uniform_bytes = unsafe {
            std::slice::from_raw_parts(
                &uniforms as *const _ as *const u8,
                std::mem::size_of::<Stage2Uniforms>()
            )
        };
        queue.write_buffer(&self.stage2_uniforms, 0, uniform_bytes);
    }
    
    /// Render Stage 3: Mixing with FB2
    fn render_stage3(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        output_view: &wgpu::TextureView,
        params: &Block2Params,
    ) {
        // Update uniforms
        self.update_stage3_uniforms(queue, params);
        
        // Lazily allocate delay buffers when delay is active
        let delay_frames = params.fb2_delay_time as usize;
        if delay_frames > 0 {
            self.resources.ensure_delay_capacity(device, queue, delay_frames, "Block2");
        }
        // When delay_time > 0, both textures use the delayed frame (shader only uses delay_tex)
        // When delay_time == 0, fb2_tex uses immediate feedback, delay_tex is unused
        let (fb2_view, delay_view) = if delay_frames > 0 {
            let delayed = self.resources.get_delay_view(delay_frames);
            (delayed, delayed)
        } else {
            (self.resources.get_feedback_view(), self.resources.get_delay_view(1))
        };
        
        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Block2 Stage 3 Bind Group"),
            layout: &self.stage3_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.stage3_uniforms.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(self.resources.get_output_view()),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.cached_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(fb2_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&self.cached_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(delay_view),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::Sampler(&self.cached_sampler),
                },
            ],
        });
        
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Block2 Stage 3 Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: output_view,
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
            
            render_pass.set_pipeline(&self.stage3_pipeline);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_bind_group(0, &bind_group, &[]);
            render_pass.draw(0..6, 0..1);
        }
    }
    
    /// Update Stage 1 uniforms from params
    fn update_stage1_uniforms(&self, queue: &wgpu::Queue, params: &Block2Params) {
        let uniforms = Stage1Uniforms {
            width: self.width as f32,
            height: self.height as f32,
            inv_width: 1.0 / self.width as f32,
            inv_height: 1.0 / self.height as f32,
            input_scale: 1.0,  // TODO: Add to params
            input_rotate: params.block2_input_rotate,
            input_x_displace: params.block2_input_x_displace,
            input_y_displace: params.block2_input_y_displace,
            input_z_displace: params.block2_input_z_displace,
            h_mirror: if params.block2_input_h_mirror { 1.0 } else { 0.0 },
            v_mirror: if params.block2_input_v_mirror { 1.0 } else { 0.0 },
            h_flip: if params.block2_input_h_flip { 1.0 } else { 0.0 },
            v_flip: if params.block2_input_v_flip { 1.0 } else { 0.0 },
            kaleidoscope_amount: params.block2_input_kaleidoscope_amount,
            kaleidoscope_slice: params.block2_input_kaleidoscope_slice,
            input_select: params.block2_input_select,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        };
        
        let uniform_bytes = unsafe {
            std::slice::from_raw_parts(
                &uniforms as *const _ as *const u8,
                std::mem::size_of::<Stage1Uniforms>()
            )
        };
        queue.write_buffer(&self.stage1_uniforms, 0, uniform_bytes);
    }
    
    /// Update Stage 3 uniforms from params
    fn update_stage3_uniforms(&self, queue: &wgpu::Queue, params: &Block2Params) {
        let uniforms = Stage3Uniforms {
            // Mix params
            mix_amount: params.fb2_mix_amount,
            mix_type: params.fb2_mix_type,
            mix_overflow: params.fb2_mix_overflow,
            _pad0: 0.0,
            
            // Input keying
            key_value: [params.fb2_key_value.x, params.fb2_key_value.y, params.fb2_key_value.z],
            key_threshold: params.fb2_key_threshold,
            key_soft: params.fb2_key_soft,
            
            // Scalars (moved before vec3s to avoid alignment issues)
            key_mode: params.fb2_key_mode,
            key_order: params.fb2_key_order,
            fb2_hue_shaper: params.fb2_hue_shaper,
            fb2_posterize: params.fb2_posterize,
            fb2_posterize_switch: if params.fb2_posterize_switch { 1 } else { 0 },
            _pad1: 0.0,
            _pad2: 0.0,
            
            // FB2 Color - vec3s at 16-byte aligned offsets
            fb2_hsb_offset: params.fb2_hsb_offset.to_array(),
            _pad3: 0.0,
            fb2_hsb_attenuate: params.fb2_hsb_attenuate.to_array(),
            _pad4: 0.0,
            fb2_hsb_powmap: params.fb2_hsb_powmap.to_array(),
            _pad5: 0.0,
            
            // FB2 Inverts
            fb2_hue_invert: if params.fb2_hue_invert { 1.0 } else { 0.0 },
            fb2_saturation_invert: if params.fb2_saturation_invert { 1.0 } else { 0.0 },
            fb2_bright_invert: if params.fb2_bright_invert { 1.0 } else { 0.0 },
            fb2_rgb_invert: if params.fb2_rgb_invert { 1.0 } else { 0.0 },
            
            // FB2 Geometric
            fb2_x_displace: params.fb2_x_displace,
            fb2_y_displace: params.fb2_y_displace,
            fb2_z_displace: params.fb2_z_displace,
            fb2_rotate: params.fb2_rotate,
            fb2_kaleidoscope_amount: params.fb2_kaleidoscope_amount,
            fb2_kaleidoscope_slice: params.fb2_kaleidoscope_slice,
            fb2_h_mirror: if params.fb2_h_mirror { 1.0 } else { 0.0 },
            fb2_v_mirror: if params.fb2_v_mirror { 1.0 } else { 0.0 },
            fb2_h_flip: if params.fb2_h_flip { 1.0 } else { 0.0 },
            fb2_v_flip: if params.fb2_v_flip { 1.0 } else { 0.0 },
            
            // FB2 Filters
            fb2_blur_amount: params.fb2_blur_amount,
            fb2_blur_radius: params.fb2_blur_radius,
            fb2_sharpen_amount: params.fb2_sharpen_amount,
            fb2_sharpen_radius: params.fb2_sharpen_radius,
            fb2_filters_boost: params.fb2_filters_boost,
            _pad8: 0.0,
            
            // FB2 Shear
            fb2_shear_matrix: params.fb2_shear_matrix.to_array(),
            
            // Delay
            fb2_delay_time: params.fb2_delay_time,
            fb2_rotate_mode: params.fb2_rotate_mode,
            fb2_geo_overflow: params.fb2_geo_overflow,
            _pad10: 0,
            _pad11: 0.0,
            _pad12: 0.0,
            _pad13: 0.0,
            _pad14: 0.0,
            _pad15: 0.0,
            _pad16: 0.0,
            _pad17: 0.0,
            _pad18: 0.0,
            _pad19: 0.0,
            _pad20: 0.0,
            _pad21: 0.0,
            _pad22: 0.0,
        };
        
        let uniform_bytes = unsafe {
            std::slice::from_raw_parts(
                &uniforms as *const _ as *const u8,
                std::mem::size_of::<Stage3Uniforms>()
            )
        };
        queue.write_buffer(&self.stage3_uniforms, 0, uniform_bytes);
    }
    
    /// Update params and render (convenience method)
    pub fn update_params(&self, queue: &wgpu::Queue, params: &Block2Params) {
        self.update_stage1_uniforms(queue, params);
        self.update_stage3_uniforms(queue, params);
    }

    /// Clear all internal buffers (feedback + ping-pong) to black.
    /// Call on input switch to avoid stale frames bleeding through the feedback path.
    pub fn clear_all(&self, queue: &wgpu::Queue) {
        self.resources.clear_all(queue);
    }
}

/// Helper to create a default sampler
fn create_default_sampler(device: &wgpu::Device) -> wgpu::Sampler {
    device.create_sampler(&wgpu::SamplerDescriptor {
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    })
}
