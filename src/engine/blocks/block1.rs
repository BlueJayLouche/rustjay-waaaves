//! # Modular Block 1
//!
//! Simplified 3-stage Block 1 implementation:
//! 1. Input Sampling - Sample Input 1, Input 2, FB with transforms
//! 2. Effects - HSB, blur (optional, skip if not needed)
//! 3. Mixing - Combine inputs

use std::cell::RefCell;
use crate::engine::blocks::{BlockResources, StageVertex};
use crate::params::Block1Params;

/// 16-byte aligned Vec3 for WGSL compatibility
#[repr(C, align(16))]
#[derive(Debug, Copy, Clone, bytemuck::AnyBitPattern)]
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

impl Vec3 {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self([x, y, z])
    }
    
    pub fn x(&self) -> f32 { self.0[0] }
    pub fn y(&self) -> f32 { self.0[1] }
    pub fn z(&self) -> f32 { self.0[2] }
}

/// Simplified Block 1 using modular stages
pub struct ModularBlock1 {
    /// GPU resources (ping-pong buffers + feedback)
    pub resources: BlockResources,
    
    /// Stage 1: Input sampling pipeline
    stage1_pipeline: wgpu::RenderPipeline,
    stage1_bind_group_layout: wgpu::BindGroupLayout,
    stage1_uniforms_ch1: wgpu::Buffer,
    stage1_uniforms_ch2: wgpu::Buffer,
    
    /// Stage 2: Effects pipeline (optional)
    stage2_pipeline: Option<wgpu::RenderPipeline>,
    stage2_bind_group_layout: wgpu::BindGroupLayout,
    stage2_uniforms_ch1: wgpu::Buffer,
    stage2_uniforms_ch2: wgpu::Buffer,
    
    /// Stage 3: Mixing pipeline
    stage3_pipeline: wgpu::RenderPipeline,
    stage3_bind_group_layout: wgpu::BindGroupLayout,
    stage3_uniforms: wgpu::Buffer,
    
    /// Vertex buffer for full-screen quad
    vertex_buffer: wgpu::Buffer,
    
    /// Dimensions
    width: u32,
    height: u32,
    
    /// Bind group cache to avoid recreating every frame
    /// Key: (uniforms_ptr, input1_ptr, input2_ptr, feedback_ptr) -> BindGroup
    /// CRITICAL: Must include uniforms_ptr since CH1/CH2 have different transforms
    stage1_bind_group_cache: RefCell<std::collections::HashMap<(u64, u64, u64, u64), wgpu::BindGroup>>,
    
    /// Stage 2 bind group cache
    /// Key: (uniforms_ptr, input_ptr) -> BindGroup
    /// CRITICAL: Must include uniforms_ptr since CH1/CH2 have different transforms
    stage2_bind_group_cache: RefCell<std::collections::HashMap<(u64, u64), wgpu::BindGroup>>,
    
    /// Cached samplers to avoid creating every frame
    sampler_cache: RefCell<Option<wgpu::Sampler>>,

}

/// Stage 1 uniforms (must match shader)
#[repr(C, align(16))]
#[derive(Debug, Copy, Clone, bytemuck::Zeroable)]
struct Stage1Uniforms {
    // Resolution
    width: f32,
    height: f32,
    
    // Transform for Input 1
    in1_scale: f32,
    in1_rotate: f32,
    in1_x_displace: f32,
    in1_y_displace: f32,
    
    // Transform for Input 2
    in2_scale: f32,
    in2_rotate: f32,
    in2_x_displace: f32,
    in2_y_displace: f32,
    
    // Transform for Feedback
    fb_scale: f32,
    fb_rotate: f32,
    fb_x_displace: f32,
    fb_y_displace: f32,
    
    // Which inputs to sample (1.0 = yes, 0.0 = no)
    sample_input2: f32,
    sample_feedback: f32,
    
    // Geometric transforms (kaleidoscope, mirrors, flips)
    kaleidoscope_amount: f32,
    kaleidoscope_slice: f32,
    h_mirror: f32,
    v_mirror: f32,
    h_flip: f32,
    v_flip: f32,
}

/// Stage 2 uniforms - Effects parameters
#[repr(C, align(16))]
#[derive(Debug, Copy, Clone, bytemuck::Zeroable)]
struct Stage2Uniforms {
    // HSB adjustments (hue, saturation, brightness)
    hsb_h: f32,
    hsb_s: f32,
    hsb_b: f32,
    filters_boost: f32,
    
    // Blur parameters
    blur_amount: f32,
    blur_radius: f32,
    _pad1: f32,
    _pad2: f32,
    
    // Sharpen parameters
    sharpen_amount: f32,
    sharpen_radius: f32,
    _pad3: f32,
    _pad4: f32,
    
    // Resolution for effects
    width: f32,
    height: f32,
    inv_width: f32,
    inv_height: f32,
    
    // Invert switches (1.0 = on, 0.0 = off)
    hue_invert: f32,
    saturation_invert: f32,
    bright_invert: f32,
    rgb_invert: f32,
    
    // Solarize
    solarize: f32,
    _pad5: f32,
    _pad6: f32,
    _pad7: f32,
    
    // Posterize
    posterize: f32,
    posterize_switch: f32,
    _pad8: f32,
    _pad9: f32,
}

/// Stage 3 uniforms - Mixing with keying and feedback
#[repr(C, align(16))]
#[derive(Debug, Copy, Clone, bytemuck::Zeroable)]
struct Stage3Uniforms {
    // Input 2 mix
    input2_amount: f32,
    input2_mix_type: i32,      // 0=lerp, 1=add, 2=diff, 3=mult, 4=dodge
    input2_mix_overflow: i32,  // 0=wrap, 1=clamp, 2=mirror
    _pad0: f32,
    
    // Feedback mix
    feedback_amount: f32,
    feedback_mix_type: i32,
    feedback_mix_overflow: i32,
    _pad1: f32,
    
    // CH2 Keying parameters
    ch2_key_value_r: f32,
    ch2_key_value_g: f32,
    ch2_key_value_b: f32,
    ch2_key_threshold: f32,
    ch2_key_soft: f32,
    ch2_key_mode: i32,        // 0=off, 1=luma, 2=chroma
    ch2_key_order: i32,       // 0=key then mix, 1=mix then key
    
    // FB1 Keying parameters
    fb1_key_value_r: f32,
    fb1_key_value_g: f32,
    fb1_key_value_b: f32,
    fb1_key_threshold: f32,
    fb1_key_soft: f32,
    fb1_key_mode: i32,
    fb1_key_order: i32,
    
    // Padding to align vec3 fields to 16-byte boundary (offset 96)
    _pad_before_hsb: [f32; 2],  // 8 bytes: offsets 88-96
    
    // FB1 Color adjustments ([f32; 3] for 12-byte size to match WGSL vec3)
    fb1_hsb_offset: [f32; 3],     // Offset 96, size 12 (ends at 108)
    _pad_after_hsb_offset: f32,   // Offset 108-112 (padding to align next vec3)
    fb1_hsb_attenuate: [f32; 3],  // Offset 112, size 12 (ends at 124)
    _pad_after_hsb_attenuate: f32,// Offset 124-128 (padding to align next vec3)
    fb1_hsb_powmap: [f32; 3],     // Offset 128, size 12 (ends at 140)
    fb1_hue_shaper: f32,      // Hue shaping amount
    fb1_posterize: f32,
    fb1_posterize_switch: f32, // 0=RGB posterize, 1=HSB posterize
    
    // FB1 Color inverts
    fb1_hue_invert: f32,
    fb1_saturation_invert: f32,
    fb1_bright_invert: f32,
    _pad2: f32,
    
    // FB1 Geometric transforms
    fb1_x_displace: f32,
    fb1_y_displace: f32,
    fb1_z_displace: f32,
    fb1_rotate: f32,
    fb1_kaleidoscope_amount: f32,
    fb1_kaleidoscope_slice: f32,
    fb1_h_mirror: f32,
    fb1_v_mirror: f32,
    
    // FB1 Filters
    fb1_blur_amount: f32,
    fb1_blur_radius: f32,
    fb1_sharpen_amount: f32,
    fb1_sharpen_radius: f32,
    fb1_filters_boost: f32,
    _pad3: f32,
    _pad4: f32,
    
    // FB1 Temporal filters
    fb1_temporal_filter1_amount: f32,
    fb1_temporal_filter1_resonance: f32,
    fb1_temporal_filter2_amount: f32,
    fb1_temporal_filter2_resonance: f32,
    
    // Delay time (0 = no delay, 1+ = number of frames delay)
    fb1_delay_time: i32,
    _pad5: i32,
    _pad6: i32,
    _pad7: i32,
}

impl ModularBlock1 {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, width: u32, height: u32) -> Self {
        let resources = BlockResources::new(device, queue, width, height, "Block1");
        let vertex_buffer = StageVertex::create_quad_buffer(device, queue);
        
        // Create stage 1 pipeline (input sampling)
        let (stage1_pipeline, stage1_bind_group_layout, stage1_uniforms_ch1, stage1_uniforms_ch2) = 
            Self::create_stage1(device, queue, width, height);
        
        // Create stage 2 pipeline (effects) - separate buffers for CH1 and CH2
        let (stage2_pipeline, stage2_bind_group_layout, stage2_uniforms_ch1, stage2_uniforms_ch2) = 
            Self::create_stage2(device, width, height);
        
        // Create stage 3 pipeline (mixing)
        let (stage3_pipeline, stage3_bind_group_layout, stage3_uniforms) = 
            Self::create_stage3(device);
        
        Self {
            resources,
            stage1_pipeline,
            stage1_bind_group_layout,
            stage1_uniforms_ch1,
            stage1_uniforms_ch2,
            stage2_pipeline: Some(stage2_pipeline),
            stage2_bind_group_layout,
            stage2_uniforms_ch1,
            stage2_uniforms_ch2,
            stage3_pipeline,
            stage3_bind_group_layout,
            stage3_uniforms,
            vertex_buffer,
            width,
            height,
            // Initialize bind group caches
            stage1_bind_group_cache: RefCell::new(std::collections::HashMap::new()),
            stage2_bind_group_cache: RefCell::new(std::collections::HashMap::new()),
            sampler_cache: RefCell::new(None),
        }
    }
    
    /// Create Stage 1: Input Sampling pipeline
    /// Returns separate uniform buffers for CH1 and CH2 to prevent parameter overwriting
    fn create_stage1(device: &wgpu::Device, _queue: &wgpu::Queue, width: u32, height: u32) -> (wgpu::RenderPipeline, wgpu::BindGroupLayout, wgpu::Buffer, wgpu::Buffer) {
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
                in1_scale: f32,
                in1_rotate: f32,
                in1_x_displace: f32,
                in1_y_displace: f32,
                in2_scale: f32,
                in2_rotate: f32,
                in2_x_displace: f32,
                in2_y_displace: f32,
                fb_scale: f32,
                fb_rotate: f32,
                fb_x_displace: f32,
                fb_y_displace: f32,
                sample_input2: f32,
                sample_feedback: f32,
                kaleidoscope_amount: f32,
                kaleidoscope_slice: f32,
                h_mirror: f32,
                v_mirror: f32,
                h_flip: f32,
                v_flip: f32,
            };
            
            @group(0) @binding(0)
            var<uniform> uniforms: Uniforms;
            
            @group(0) @binding(1)
            var input1_tex: texture_2d<f32>;
            @group(0) @binding(2)
            var input1_sampler: sampler;
            
            @group(0) @binding(3)
            var input2_tex: texture_2d<f32>;
            @group(0) @binding(4)
            var input2_sampler: sampler;
            
            @group(0) @binding(5)
            var fb_tex: texture_2d<f32>;
            @group(0) @binding(6)
            var fb_sampler: sampler;
            
            fn transform_uv(uv: vec2<f32>, scale: f32, rotate: f32, displace: vec2<f32>) -> vec2<f32> {
                var result = uv - vec2<f32>(0.5);
                // Scale first, then rotate (matches Stage 3 order)
                result = result / scale;
                let cos_r = cos(rotate);
                let sin_r = sin(rotate);
                let rot_x = result.x * cos_r - result.y * sin_r;
                let rot_y = result.x * sin_r + result.y * cos_r;
                result = vec2<f32>(rot_x, rot_y);
                return result + vec2<f32>(0.5) + displace;
            }
            
            // Apply kaleidoscope effect to UV
            fn apply_kaleidoscope(uv: vec2<f32>, amount: f32, slice: f32) -> vec2<f32> {
                if (amount <= 0.001 || slice <= 0.001) {
                    return uv;
                }
                
                // Center coordinates
                var centered = uv - vec2<f32>(0.5);
                
                // Convert to polar
                let radius = length(centered);
                var angle = atan2(centered.y, centered.x);
                
                // Apply kaleidoscope slicing
                let slice_size = 6.28318530718 * slice; // 2PI * slice
                let slice_index = floor(angle / slice_size);
                let slice_angle = angle - slice_index * slice_size;
                
                // Mirror within slice
                let mirrored_angle = abs(slice_angle - slice_size * 0.5);
                
                // Combine with original based on amount
                let kaleido_angle = mix(angle, slice_index * slice_size + mirrored_angle, amount);
                
                // Convert back to cartesian
                let kaleido_uv = vec2<f32>(cos(kaleido_angle) * radius, sin(kaleido_angle) * radius) + vec2<f32>(0.5);
                return mix(uv, kaleido_uv, amount);
            }
            
            // Apply geometric transforms (mirrors, flips)
            fn apply_geo_transforms(uv: vec2<f32>, h_mirror: f32, v_mirror: f32, h_flip: f32, v_flip: f32) -> vec2<f32> {
                var result = uv;
                
                // Horizontal mirror (mirror around center x=0.5)
                if (h_mirror > 0.5) {
                    result.x = abs(result.x - 0.5) + 0.5;
                    result.x = 1.0 - result.x; // Flip after mirror to match OF behavior
                }
                
                // Vertical mirror (mirror around center y=0.5)
                if (v_mirror > 0.5) {
                    result.y = abs(result.y - 0.5) + 0.5;
                    result.y = 1.0 - result.y; // Flip after mirror to match OF behavior
                }
                
                // Horizontal flip
                if (h_flip > 0.5) {
                    result.x = 1.0 - result.x;
                }
                
                // Vertical flip
                if (v_flip > 0.5) {
                    result.y = 1.0 - result.y;
                }
                
                return result;
            }
            
            @fragment
            fn fs_main(@location(0) texcoord: vec2<f32>) -> @location(0) vec4<f32> {
                // Apply kaleidoscope to base UV
                var uv = apply_kaleidoscope(texcoord, uniforms.kaleidoscope_amount, uniforms.kaleidoscope_slice);
                
                // Apply geometric transforms
                uv = apply_geo_transforms(uv, uniforms.h_mirror, uniforms.v_mirror, uniforms.h_flip, uniforms.v_flip);
                
                // Always sample Input 1
                let uv1 = transform_uv(uv, uniforms.in1_scale, uniforms.in1_rotate, 
                                       vec2<f32>(uniforms.in1_x_displace, uniforms.in1_y_displace));
                let c1 = textureSample(input1_tex, input1_sampler, uv1);
                
                // Input 2 (if enabled)
                var c2 = vec4<f32>(0.0);
                if (uniforms.sample_input2 > 0.5) {
                    let uv2 = transform_uv(texcoord, uniforms.in2_scale, uniforms.in2_rotate,
                                           vec2<f32>(uniforms.in2_x_displace, uniforms.in2_y_displace));
                    c2 = textureSample(input2_tex, input2_sampler, uv2);
                }
                
                // Feedback (if enabled)
                var fb = vec4<f32>(0.0);
                if (uniforms.sample_feedback > 0.5) {
                    let uvfb = transform_uv(texcoord, uniforms.fb_scale, uniforms.fb_rotate,
                                            vec2<f32>(uniforms.fb_x_displace, uniforms.fb_y_displace));
                    fb = textureSample(fb_tex, fb_sampler, uvfb);
                }
                
                // Output full color for the active input(s)
                // Each channel should show full RGB color, not packed luminance
                if (uniforms.sample_input2 > 0.5 && uniforms.sample_feedback > 0.5) {
                    // All three inputs - blend them (simple additive for now)
                    return c1 + c2 + fb;
                } else if (uniforms.sample_input2 > 0.5) {
                    // Input 1 + Input 2
                    return c1 + c2;
                } else if (uniforms.sample_feedback > 0.5) {
                    // Input 1 + Feedback
                    return c1 + fb;
                } else {
                    // Only Input 1
                    return c1;
                }
            }
        "#;
        
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Block1 Stage1 Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_code.into()),
        });
        
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Block1 Stage1 BGL"),
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
                // Input 1
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
                // Input 2
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
                // Feedback
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
            label: Some("Block1 Stage1 Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Block1 Stage1 Pipeline"),
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
                    blend: Some(wgpu::BlendState::REPLACE),
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
        
        // Create separate uniform buffers for CH1 and CH2
        // This prevents CH2 parameters from overwriting CH1 parameters
        let uniforms_ch1 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Block1 Stage1 Uniforms CH1"),
            size: std::mem::size_of::<Stage1Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let uniforms_ch2 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Block1 Stage1 Uniforms CH2"),
            size: std::mem::size_of::<Stage1Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        (pipeline, bind_group_layout, uniforms_ch1, uniforms_ch2)
    }
    
    /// Create Stage 2: Effects pipeline (HSB, blur, kaleidoscope)
    /// Returns separate uniform buffers for CH1 and CH2
    fn create_stage2(device: &wgpu::Device, width: u32, height: u32) -> (wgpu::RenderPipeline, wgpu::BindGroupLayout, wgpu::Buffer, wgpu::Buffer) {
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
                hsb_h: f32,
                hsb_s: f32,
                hsb_b: f32,
                filters_boost: f32,
                blur_amount: f32,
                blur_radius: f32,
                _pad1: f32,
                _pad2: f32,
                sharpen_amount: f32,
                sharpen_radius: f32,
                _pad3: f32,
                _pad4: f32,
                width: f32,
                height: f32,
                inv_width: f32,
                inv_height: f32,
                hue_invert: f32,
                saturation_invert: f32,
                bright_invert: f32,
                rgb_invert: f32,
                solarize: f32,
                _pad5: f32,
                _pad6: f32,
                _pad7: f32,
                posterize: f32,
                posterize_switch: f32,
                _pad8: f32,
                _pad9: f32,
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
                
                // Brightness
                let b = max_val;
                
                // Saturation
                var s = 0.0;
                if (max_val > 0.0) {
                    s = delta / max_val;
                }
                
                // Hue
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
                
                var r = 0.0;
                var g = 0.0;
                var bl = 0.0;
                
                if (i == 0.0) { r = b; g = t; bl = p; }
                else if (i == 1.0) { r = q; g = b; bl = p; }
                else if (i == 2.0) { r = p; g = b; bl = t; }
                else if (i == 3.0) { r = p; g = q; bl = b; }
                else if (i == 4.0) { r = t; g = p; bl = b; }
                else { r = b; g = p; bl = q; }
                
                return vec3<f32>(r, g, bl);
            }
            
            // Apply blur effect - optimized 9-tap blur
            fn blur(uv: vec2<f32>, amount: f32, radius: f32) -> vec3<f32> {
                if (amount <= 0.001 || radius <= 0.001) {
                    return textureSample(input_tex, input_sampler, uv).rgb;
                }
                
                let texel_size = vec2<f32>(uniforms.inv_width, uniforms.inv_height);
                let blur_radius = radius * 15.0; // Increased scale for comparable blur with fewer samples
                
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
                
                // Mix between original and blurred
                let original = textureSample(input_tex, input_sampler, uv).rgb;
                return mix(original, result, amount);
            }
            
            // Apply sharpen effect
            fn sharpen(uv: vec2<f32>, amount: f32, radius: f32) -> vec3<f32> {
                if (amount <= 0.001 || radius <= 0.001) {
                    return textureSample(input_tex, input_sampler, uv).rgb;
                }
                
                let texel_size = vec2<f32>(uniforms.inv_width, uniforms.inv_height);
                let sharpen_radius = radius * 2.0; // Sharpen uses smaller radius
                
                // Sample center
                let center = textureSample(input_tex, input_sampler, uv).rgb;
                
                // Sample neighbors (cross pattern)
                let left = textureSample(input_tex, input_sampler, uv + vec2<f32>(-sharpen_radius, 0.0) * texel_size).rgb;
                let right = textureSample(input_tex, input_sampler, uv + vec2<f32>(sharpen_radius, 0.0) * texel_size).rgb;
                let up = textureSample(input_tex, input_sampler, uv + vec2<f32>(0.0, -sharpen_radius) * texel_size).rgb;
                let down = textureSample(input_tex, input_sampler, uv + vec2<f32>(0.0, sharpen_radius) * texel_size).rgb;
                
                // Laplacian filter: center*4 - left - right - up - down
                let laplacian = center * 4.0 - left - right - up - down;
                
                // Apply sharpening
                let sharpened = center + laplacian * amount;
                
                return clamp(sharpened, vec3<f32>(0.0), vec3<f32>(1.0));
            }
            
            // Apply inverts
            fn apply_inverts(color: vec3<f32>, hue_inv: f32, sat_inv: f32, bright_inv: f32, rgb_inv: f32) -> vec3<f32> {
                var result = color;
                
                // RGB invert (simple)
                if (rgb_inv > 0.5) {
                    result = vec3<f32>(1.0) - result;
                }
                
                // HSB-based inverts
                if (hue_inv > 0.5 || sat_inv > 0.5 || bright_inv > 0.5) {
                    var hsb = rgb_to_hsb(result);
                    
                    // Hue invert (shift by 0.5)
                    if (hue_inv > 0.5) {
                        hsb.x = fract(hsb.x + 0.5);
                    }
                    
                    // Saturation invert
                    if (sat_inv > 0.5) {
                        hsb.y = 1.0 - hsb.y;
                    }
                    
                    // Brightness invert
                    if (bright_inv > 0.5) {
                        hsb.z = 1.0 - hsb.z;
                    }
                    
                    result = hsb_to_rgb(hsb);
                }
                
                return result;
            }
            
            // Apply solarize effect
            fn solarize(color: vec3<f32>, amount: f32) -> vec3<f32> {
                if (amount <= 0.001) {
                    return color;
                }
                
                // Solarize inverts colors above threshold
                let threshold = 0.5;
                var result = color;
                
                for (var i: i32 = 0; i < 3; i = i + 1) {
                    if (result[i] > threshold) {
                        result[i] = mix(result[i], 1.0 - result[i], amount);
                    }
                }
                
                return result;
            }
            
            // Apply posterize effect (reduce color levels)
            fn posterize(color: vec3<f32>, levels: f32) -> vec3<f32> {
                if (levels <= 0.001) {
                    return color;
                }
                
                // levels = number of color steps (e.g., 4.0 = 4 levels per channel)
                let steps = max(levels, 2.0);
                return floor(color * steps) / steps;
            }
            
            // Apply filters boost (enhance effect intensity)
            fn apply_boost(color: vec3<f32>, boost: f32) -> vec3<f32> {
                if (boost <= 0.001) {
                    return color;
                }
                
                // Boost enhances contrast by pushing values away from mid-gray
                let mid_gray = vec3<f32>(0.5);
                return mid_gray + (color - mid_gray) * (1.0 + boost);
            }
            
            @fragment
            fn fs_main(@location(0) texcoord: vec2<f32>) -> @location(0) vec4<f32> {
                // Start with original color
                var color = textureSample(input_tex, input_sampler, texcoord).rgb;
                
                // Apply blur
                if (uniforms.blur_amount > 0.001) {
                    color = blur(texcoord, uniforms.blur_amount, uniforms.blur_radius);
                }
                
                // Apply sharpen
                if (uniforms.sharpen_amount > 0.001) {
                    color = sharpen(texcoord, uniforms.sharpen_amount, uniforms.sharpen_radius);
                }
                
                // Apply HSB adjustments
                let hsb_adjust = vec3<f32>(uniforms.hsb_h, uniforms.hsb_s, uniforms.hsb_b);
                if (length(hsb_adjust) > 0.001) {
                    var hsb = rgb_to_hsb(color);
                    
                    // Apply hue shift
                    hsb.x = fract(hsb.x + uniforms.hsb_h);
                    
                    // Apply saturation adjustment
                    hsb.y = clamp(hsb.y * (1.0 + uniforms.hsb_s), 0.0, 1.0);
                    
                    // Apply brightness adjustment
                    hsb.z = clamp(hsb.z * (1.0 + uniforms.hsb_b), 0.0, 1.0);
                    
                    color = hsb_to_rgb(hsb);
                }
                
                // Apply inverts
                color = apply_inverts(color, uniforms.hue_invert, uniforms.saturation_invert, 
                                      uniforms.bright_invert, uniforms.rgb_invert);
                
                // Apply solarize
                color = solarize(color, uniforms.solarize);
                
                // Apply posterize
                if (uniforms.posterize_switch > 0.5) {
                    color = posterize(color, uniforms.posterize);
                }
                
                // Apply filters boost (enhance contrast)
                color = apply_boost(color, uniforms.filters_boost);
                
                return vec4<f32>(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
            }
        "#;
        
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Block1 Stage2 Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_code.into()),
        });
        
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Block1 Stage2 BGL"),
            entries: &[
                // Uniforms (binding 0)
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
                // Input texture (binding 1)
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
                // Sampler (binding 2)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Block1 Stage2 Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Block1 Stage2 Pipeline"),
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
                    blend: Some(wgpu::BlendState::REPLACE),
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
        
        // Create separate uniform buffers for CH1 and CH2
        let uniforms_ch1 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Block1 Stage2 Uniforms CH1"),
            size: std::mem::size_of::<Stage2Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let uniforms_ch2 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Block1 Stage2 Uniforms CH2"),
            size: std::mem::size_of::<Stage2Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        (pipeline, bind_group_layout, uniforms_ch1, uniforms_ch2)
    }
    
    /// Create Stage 3: Mixing pipeline with keying and multiple inputs
    fn create_stage3(device: &wgpu::Device) -> (wgpu::RenderPipeline, wgpu::BindGroupLayout, wgpu::Buffer) {
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
                input2_amount: f32,
                input2_mix_type: i32,
                input2_mix_overflow: i32,
                _pad0: f32,
                feedback_amount: f32,
                feedback_mix_type: i32,
                feedback_mix_overflow: i32,
                _pad1: f32,
                // CH2 keying
                ch2_key_value_r: f32,
                ch2_key_value_g: f32,
                ch2_key_value_b: f32,
                ch2_key_threshold: f32,
                ch2_key_soft: f32,
                ch2_key_mode: i32,
                ch2_key_order: i32,
                // FB1 keying
                fb1_key_value_r: f32,
                fb1_key_value_g: f32,
                fb1_key_value_b: f32,
                fb1_key_threshold: f32,
                fb1_key_soft: f32,
                fb1_key_mode: i32,
                fb1_key_order: i32,
                fb1_hsb_offset: vec3<f32>,
                fb1_hsb_attenuate: vec3<f32>,
                fb1_hsb_powmap: vec3<f32>,
                fb1_hue_shaper: f32,
                fb1_posterize: f32,
                fb1_posterize_switch: f32,
                fb1_hue_invert: f32,
                fb1_saturation_invert: f32,
                fb1_bright_invert: f32,
                _pad2: f32,
                // Geometric transforms must match Rust struct order
                fb1_x_displace: f32,
                fb1_y_displace: f32,
                fb1_z_displace: f32,
                fb1_rotate: f32,
                fb1_kaleidoscope_amount: f32,
                fb1_kaleidoscope_slice: f32,
                fb1_h_mirror: f32,
                fb1_v_mirror: f32,
                // Filters
                fb1_blur_amount: f32,
                fb1_blur_radius: f32,
                fb1_sharpen_amount: f32,
                fb1_sharpen_radius: f32,
                fb1_filters_boost: f32,
                _pad3: f32,
                _pad4: f32,
                // Temporal filters
                fb1_temporal_filter1_amount: f32,
                fb1_temporal_filter1_resonance: f32,
                fb1_temporal_filter2_amount: f32,
                fb1_temporal_filter2_resonance: f32,
                // Delay
                fb1_delay_time: i32,
                _pad5: i32,
                _pad6: i32,
                _pad7: i32,
            };
            
            @group(0) @binding(0)
            var<uniform> uniforms: Uniforms;
            
            // Stage 2 output (processed Input 1)
            @group(0) @binding(1)
            var stage2_tex: texture_2d<f32>;
            @group(0) @binding(2)
            var stage2_sampler: sampler;
            
            // Raw Input 2
            @group(0) @binding(3)
            var input2_tex: texture_2d<f32>;
            @group(0) @binding(4)
            var input2_sampler: sampler;
            
            // Feedback buffer (current frame or delayed)
            @group(0) @binding(5)
            var feedback_tex: texture_2d<f32>;
            @group(0) @binding(6)
            var feedback_sampler: sampler;
            
            // Delay buffer (for feedback delay effect)
            @group(0) @binding(7)
            var delay_tex: texture_2d<f32>;
            @group(0) @binding(8)
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
                
                var r = 0.0;
                var g = 0.0;
                var bl = 0.0;
                
                if (i == 0.0) { r = b; g = t; bl = p; }
                else if (i == 1.0) { r = q; g = b; bl = p; }
                else if (i == 2.0) { r = p; g = b; bl = t; }
                else if (i == 3.0) { r = p; g = q; bl = b; }
                else if (i == 4.0) { r = t; g = p; bl = b; }
                else { r = b; g = p; bl = q; }
                
                return vec3<f32>(r, g, bl);
            }
            
            // Apply overflow/wrap modes
            fn apply_overflow(color: vec3<f32>, mode: i32) -> vec3<f32> {
                switch(mode) {
                    case 0: { // Wrap
                        return fract(color);
                    }
                    case 1: { // Clamp
                        return clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));
                    }
                    case 2: { // Mirror
                        return abs(fract(color * 0.5 + 0.5) * 2.0 - 1.0);
                    }
                    default: {
                        return color;
                    }
                }
            }
            
            // Mix with CH2 keying (for CH1/CH2 mix)
            // mix_type: 0=lerp, 1=add, 2=diff, 3=mult, 4=dodge
            fn mixn_key_video_ch2(fg: vec3<f32>, bg: vec3<f32>, amount: f32, mix_type: i32, 
                             overflow: i32, key_order: i32) -> vec3<f32> {
                
                // Check if CH2 keying is enabled
                let keying_enabled = abs(uniforms.ch2_key_threshold) < 0.999;
                
                var mixed: vec3<f32>;
                
                if (keying_enabled) {
                    if (key_order == 0) {
                        // Key First, Then Mix
                        let key_amount = calculate_key_mix_ch2(fg);
                        let keyed_fg = mix(fg, bg, key_amount);
                        
                        switch(mix_type) {
                            case 0: { mixed = mix(keyed_fg, bg, amount); }
                            case 1: { mixed = keyed_fg + bg * amount; }
                            case 2: { mixed = abs(keyed_fg - bg) * amount + keyed_fg * (1.0 - amount); }
                            case 3: { mixed = mix(keyed_fg, keyed_fg * bg, amount); }
                            case 4: { mixed = mix(keyed_fg, keyed_fg / (1.0 - bg + 0.001), amount); }
                            default: { mixed = mix(keyed_fg, bg, amount); }
                        }
                    } else {
                        // Mix First, Then Key
                        switch(mix_type) {
                            case 0: { mixed = mix(fg, bg, amount); }
                            case 1: { mixed = fg + bg * amount; }
                            case 2: { mixed = abs(fg - bg) * amount + fg * (1.0 - amount); }
                            case 3: { mixed = mix(fg, fg * bg, amount); }
                            case 4: { mixed = mix(fg, fg / (1.0 - bg + 0.001), amount); }
                            default: { mixed = mix(fg, bg, amount); }
                        }
                        
                        let key_amount = calculate_key_mix_ch2(mixed);
                        mixed = mix(mixed, bg, key_amount);
                    }
                } else {
                    // No keying - simple mix
                    switch(mix_type) {
                        case 0: { mixed = mix(fg, bg, amount); }
                        case 1: { mixed = fg + bg * amount; }
                        case 2: { mixed = abs(fg - bg) * amount + fg * (1.0 - amount); }
                        case 3: { mixed = mix(fg, fg * bg, amount); }
                        case 4: { mixed = mix(fg, fg / (1.0 - bg + 0.001), amount); }
                        default: { mixed = mix(fg, bg, amount); }
                    }
                }
                
                mixed = apply_overflow(mixed, overflow);
                return mixed;
            }
            
            // Mix with FB1 keying (for FB1 mix)
            fn mixn_key_video_fb1(fg: vec3<f32>, bg: vec3<f32>, amount: f32, mix_type: i32, 
                             overflow: i32, key_order: i32) -> vec3<f32> {
                
                // Check if FB1 keying is enabled
                let keying_enabled = abs(uniforms.fb1_key_threshold) < 0.999;
                
                var mixed: vec3<f32>;
                
                if (keying_enabled) {
                    if (key_order == 0) {
                        // Key First, Then Mix
                        let key_amount = calculate_key_mix_fb1(fg);
                        let keyed_fg = mix(fg, bg, key_amount);
                        
                        switch(mix_type) {
                            case 0: { mixed = mix(keyed_fg, bg, amount); }
                            case 1: { mixed = keyed_fg + bg * amount; }
                            case 2: { mixed = abs(keyed_fg - bg) * amount + keyed_fg * (1.0 - amount); }
                            case 3: { mixed = mix(keyed_fg, keyed_fg * bg, amount); }
                            case 4: { mixed = mix(keyed_fg, keyed_fg / (1.0 - bg + 0.001), amount); }
                            default: { mixed = mix(keyed_fg, bg, amount); }
                        }
                    } else {
                        // Mix First, Then Key
                        switch(mix_type) {
                            case 0: { mixed = mix(fg, bg, amount); }
                            case 1: { mixed = fg + bg * amount; }
                            case 2: { mixed = abs(fg - bg) * amount + fg * (1.0 - amount); }
                            case 3: { mixed = mix(fg, fg * bg, amount); }
                            case 4: { mixed = mix(fg, fg / (1.0 - bg + 0.001), amount); }
                            default: { mixed = mix(fg, bg, amount); }
                        }
                        
                        let key_amount = calculate_key_mix_fb1(mixed);
                        mixed = mix(mixed, bg, key_amount);
                    }
                } else {
                    // No keying - simple mix
                    switch(mix_type) {
                        case 0: { mixed = mix(fg, bg, amount); }
                        case 1: { mixed = fg + bg * amount; }
                        case 2: { mixed = abs(fg - bg) * amount + fg * (1.0 - amount); }
                        case 3: { mixed = mix(fg, fg * bg, amount); }
                        case 4: { mixed = mix(fg, fg / (1.0 - bg + 0.001), amount); }
                        default: { mixed = mix(fg, bg, amount); }
                    }
                }
                
                mixed = apply_overflow(mixed, overflow);
                return mixed;
            }
            
            // Simple mix without keying (for internal use)
            fn mix_colors(a: vec3<f32>, b: vec3<f32>, amount: f32, mix_type: i32, overflow: i32) -> vec3<f32> {
                var result: vec3<f32>;
                
                switch(mix_type) {
                    case 0: { // Lerp
                        result = mix(a, b, amount);
                    }
                    case 1: { // Add
                        result = a + b * amount;
                    }
                    case 2: { // Difference
                        result = abs(a - b) * amount + a * (1.0 - amount);
                    }
                    case 3: { // Multiply
                        result = mix(a, a * b, amount);
                    }
                    case 4: { // Dodge
                        result = mix(a, a / (1.0 - b + 0.001), amount);
                    }
                    default: {
                        result = mix(a, b, amount);
                    }
                }
                
                return apply_overflow(result, overflow);
            }
            
            // CH2 Keying: calculates how much of the background to show based on key match
            fn calculate_key_mix_ch2(fg: vec3<f32>) -> f32 {
                let threshold_abs = abs(uniforms.ch2_key_threshold);
                if (threshold_abs >= 0.999) {
                    return 0.0;
                }
                
                let key_color = vec3<f32>(uniforms.ch2_key_value_r, uniforms.ch2_key_value_g, uniforms.ch2_key_value_b);
                var dist: f32;
                
                if (uniforms.ch2_key_mode == 0) {
                    let fg_luma = dot(fg, vec3<f32>(0.299, 0.587, 0.114));
                    let key_luma = dot(key_color, vec3<f32>(0.299, 0.587, 0.114));
                    dist = abs(fg_luma - key_luma);
                } else {
                    dist = distance(fg, key_color);
                }
                
                let threshold = threshold_abs;
                let soft = max(abs(uniforms.ch2_key_soft), 0.001);
                
                if (dist < threshold) {
                    let edge_distance = threshold - dist;
                    let key_strength = min(edge_distance / soft, 1.0);
                    return key_strength;
                }
                
                return 0.0;
            }
            
            // FB1 Keying: calculates how much of the background to show based on key match
            fn calculate_key_mix_fb1(fg: vec3<f32>) -> f32 {
                let threshold_abs = abs(uniforms.fb1_key_threshold);
                if (threshold_abs >= 0.999) {
                    return 0.0;
                }
                
                let key_color = vec3<f32>(uniforms.fb1_key_value_r, uniforms.fb1_key_value_g, uniforms.fb1_key_value_b);
                var dist: f32;
                
                if (uniforms.fb1_key_mode == 0) {
                    let fg_luma = dot(fg, vec3<f32>(0.299, 0.587, 0.114));
                    let key_luma = dot(key_color, vec3<f32>(0.299, 0.587, 0.114));
                    dist = abs(fg_luma - key_luma);
                } else {
                    dist = distance(fg, key_color);
                }
                
                let threshold = threshold_abs;
                let soft = max(abs(uniforms.fb1_key_soft), 0.001);
                
                if (dist < threshold) {
                    let edge_distance = threshold - dist;
                    let key_strength = min(edge_distance / soft, 1.0);
                    return key_strength;
                }
                
                return 0.0;
            }
            
            // Hue shaper function (from OF shader)
            fn hue_shaper(hue: f32, shaper: f32) -> f32 {
                // OF: inHue=fract(abs(inHue+shaper*sin(inHue*0.3184713) ));
                // 0.3184713 ≈ 1/(PI) * some scaling
                return fract(abs(hue + shaper * sin(hue * 0.3184713)));
            }
            
            // Apply blur effect to feedback
            fn fb1_blur(uv: vec2<f32>, amount: f32, radius: f32) -> vec3<f32> {
                if (amount < 0.001) {
                    return textureSample(feedback_tex, feedback_sampler, uv).rgb;
                }
                
                let texel_size = vec2<f32>(1.0 / 640.0, 1.0 / 480.0); // Approximate
                let blur_radius = radius * 5.0;
                
                var sum = vec3<f32>(0.0);
                var total_weight = 0.0;
                
                // Simple box blur with 9 samples
                for (var x: i32 = -1; x <= 1; x = x + 1) {
                    for (var y: i32 = -1; y <= 1; y = y + 1) {
                        let offset = vec2<f32>(f32(x), f32(y)) * texel_size * blur_radius;
                        let weight = 1.0 - (abs(f32(x)) + abs(f32(y))) * 0.25;
                        sum = sum + textureSample(feedback_tex, feedback_sampler, uv + offset).rgb * weight;
                        total_weight = total_weight + weight;
                    }
                }
                
                let blurred = sum / total_weight;
                let original = textureSample(feedback_tex, feedback_sampler, uv).rgb;
                return mix(original, blurred, amount);
            }
            
            // Apply sharpen effect to feedback
            fn fb1_sharpen(uv: vec2<f32>, amount: f32, radius: f32, color: vec3<f32>) -> vec3<f32> {
                if (amount < 0.001) {
                    return color;
                }
                
                let texel_size = vec2<f32>(1.0 / 640.0, 1.0 / 480.0);
                let sharpen_radius = radius * 2.0;
                
                // Sample neighbors
                let left = textureSample(feedback_tex, feedback_sampler, uv + vec2<f32>(-sharpen_radius, 0.0) * texel_size).rgb;
                let right = textureSample(feedback_tex, feedback_sampler, uv + vec2<f32>(sharpen_radius, 0.0) * texel_size).rgb;
                let up = textureSample(feedback_tex, feedback_sampler, uv + vec2<f32>(0.0, -sharpen_radius) * texel_size).rgb;
                let down = textureSample(feedback_tex, feedback_sampler, uv + vec2<f32>(0.0, sharpen_radius) * texel_size).rgb;
                
                // Laplacian
                let laplacian = (left + right + up + down) * 0.25 - color;
                
                // Apply sharpening with boost
                let boosted_amount = amount * (1.0 + uniforms.fb1_filters_boost);
                return clamp(color + laplacian * boosted_amount, vec3<f32>(0.0), vec3<f32>(1.0));
            }
            
            // Apply FB1 color adjustments
            fn apply_fb1_color(color: vec3<f32>) -> vec3<f32> {
                var result = color;
                
                // HSB adjustments
                let offset = uniforms.fb1_hsb_offset;
                let atten = uniforms.fb1_hsb_attenuate;
                let powmap = uniforms.fb1_hsb_powmap;
                let hue_shaper_val = uniforms.fb1_hue_shaper;
                
                if (length(offset) > 0.001 || length(atten - vec3<f32>(1.0)) > 0.001 || 
                    length(powmap - vec3<f32>(1.0)) > 0.001 || hue_shaper_val > 0.001) {
                    var hsb = rgb_to_hsb(result);
                    
                    // Apply hue shaper
                    if (hue_shaper_val > 0.001) {
                        hsb.x = hue_shaper(hsb.x, hue_shaper_val);
                    }
                    
                    // Apply powmap (power curve)
                    if (length(powmap - vec3<f32>(1.0)) > 0.001) {
                        hsb = pow(hsb, powmap);
                    }
                    
                    // Apply attenuate (multiply)
                    hsb = hsb * atten;
                    
                    // Apply offset (add)
                    hsb.x = fract(hsb.x + offset.x);
                    hsb.y = clamp(hsb.y + offset.y, 0.0, 1.0);
                    hsb.z = clamp(hsb.z + offset.z, 0.0, 1.0);
                    
                    // Apply inverts
                    if (uniforms.fb1_hue_invert > 0.5) {
                        hsb.x = 1.0 - hsb.x;
                    }
                    if (uniforms.fb1_saturation_invert > 0.5) {
                        hsb.y = 1.0 - hsb.y;
                    }
                    if (uniforms.fb1_bright_invert > 0.5) {
                        hsb.z = 1.0 - hsb.z;
                    }
                    
                    // Wrap hue, clamp sat/bright
                    hsb.x = fract(hsb.x);
                    hsb.y = clamp(hsb.y, 0.0, 1.0);
                    hsb.z = clamp(hsb.z, 0.0, 1.0);
                    
                    result = hsb_to_rgb(hsb);
                }
                
                // Posterize (RGB mode or HSB mode based on posterize_switch)
                if (uniforms.fb1_posterize > 1.0) {
                    let steps = uniforms.fb1_posterize;
                    if (uniforms.fb1_posterize_switch > 0.5) {
                        // HSB posterize
                        var hsb = rgb_to_hsb(result);
                        hsb = floor(hsb * steps) / steps;
                        result = hsb_to_rgb(hsb);
                    } else {
                        // RGB posterize
                        result = floor(result * steps) / steps;
                    }
                }
                
                return result;
            }
            
            // Transform UV for FB1 (scale, rotate, displace)
            fn transform_fb1_uv(uv: vec2<f32>) -> vec2<f32> {
                var result = uv - vec2<f32>(0.5);
                
                // Apply scale (z_displace controls scale)
                let scale = uniforms.fb1_z_displace;
                if (scale > 0.001) {
                    result = result / scale;
                }
                
                // Apply rotation
                let rotate = uniforms.fb1_rotate;
                if (abs(rotate) > 0.001) {
                    let cos_r = cos(rotate);
                    let sin_r = sin(rotate);
                    let rot_x = result.x * cos_r - result.y * sin_r;
                    let rot_y = result.x * sin_r + result.y * cos_r;
                    result = vec2<f32>(rot_x, rot_y);
                }
                
                // Apply displacement
                result = result + vec2<f32>(0.5) + vec2<f32>(uniforms.fb1_x_displace, uniforms.fb1_y_displace);
                
                return result;
            }
            
            // Apply kaleidoscope to UV
            fn apply_fb1_kaleidoscope(uv: vec2<f32>) -> vec2<f32> {
                let amount = uniforms.fb1_kaleidoscope_amount;
                let slice = uniforms.fb1_kaleidoscope_slice;
                
                if (amount <= 0.001 || slice <= 0.001) {
                    return uv;
                }
                
                // Center coordinates
                var centered = uv - vec2<f32>(0.5);
                
                // Convert to polar
                let radius = length(centered);
                var angle = atan2(centered.y, centered.x);
                
                // Apply kaleidoscope slicing
                let slice_size = 6.28318530718 * slice; // 2PI * slice
                let slice_index = floor(angle / slice_size);
                let slice_angle = angle - slice_index * slice_size;
                
                // Mirror within slice
                let mirrored_angle = abs(slice_angle - slice_size * 0.5);
                
                // Combine with original based on amount
                let kaleido_angle = mix(angle, slice_index * slice_size + mirrored_angle, amount);
                
                // Convert back to cartesian
                let kaleido_uv = vec2<f32>(cos(kaleido_angle) * radius, sin(kaleido_angle) * radius) + vec2<f32>(0.5);
                return mix(uv, kaleido_uv, amount);
            }
            
            // Apply mirror transforms
            fn apply_fb1_mirrors(uv: vec2<f32>) -> vec2<f32> {
                var result = uv;
                
                // Horizontal mirror
                if (uniforms.fb1_h_mirror > 0.5) {
                    result.x = abs(result.x - 0.5) + 0.5;
                    result.x = 1.0 - result.x;
                }
                
                // Vertical mirror
                if (uniforms.fb1_v_mirror > 0.5) {
                    result.y = abs(result.y - 0.5) + 0.5;
                    result.y = 1.0 - result.y;
                }
                
                return result;
            }
            
            // Sample FB1 with all transforms and filters applied
            fn sample_fb1(texcoord: vec2<f32>) -> vec3<f32> {
                // Apply geometric transforms
                var uv = texcoord;
                uv = apply_fb1_kaleidoscope(uv);
                uv = apply_fb1_mirrors(uv);
                uv = transform_fb1_uv(uv);
                
                // Sample from appropriate source (delayed or immediate)
                var color: vec3<f32>;
                if (uniforms.fb1_delay_time > 0) {
                    color = textureSample(delay_tex, delay_sampler, uv).rgb;
                } else {
                    color = textureSample(feedback_tex, feedback_sampler, uv).rgb;
                }
                
                // Apply blur (sampling from feedback texture)
                if (uniforms.fb1_blur_amount > 0.001) {
                    // Need to sample with original texcoord for blur kernel
                    color = fb1_blur(uv, uniforms.fb1_blur_amount, uniforms.fb1_blur_radius);
                }
                
                // Apply sharpen
                if (uniforms.fb1_sharpen_amount > 0.001) {
                    color = fb1_sharpen(uv, uniforms.fb1_sharpen_amount, uniforms.fb1_sharpen_radius, color);
                }
                
                return color;
            }
            
            @fragment
            fn fs_main(@location(0) texcoord: vec2<f32>) -> @location(0) vec4<f32> {
                // Sample inputs
                let input1 = textureSample(stage2_tex, stage2_sampler, texcoord).rgb;
                let input2 = textureSample(input2_tex, input2_sampler, texcoord).rgb;
                
                // Sample FB1 with transforms
                let feedback = sample_fb1(texcoord);
                
                // Apply FB1 color adjustments to feedback
                let processed_feedback = apply_fb1_color(feedback);
                
                // OF-style mixing with integrated keying
                // First mix: Input 1 (fg) with Input 2 (bg) using CH2 keying
                var result = input1;
                if (uniforms.input2_amount > 0.001) {
                    result = mixn_key_video_ch2(input1, input2, uniforms.input2_amount,
                                           uniforms.input2_mix_type, uniforms.input2_mix_overflow,
                                           uniforms.ch2_key_order);
                }
                
                // Second mix: Result (fg) with FB1 (bg) using FB1 keying
                if (uniforms.feedback_amount > 0.001) {
                    result = mixn_key_video_fb1(result, processed_feedback, uniforms.feedback_amount,
                                           uniforms.feedback_mix_type, uniforms.feedback_mix_overflow,
                                           uniforms.fb1_key_order);
                }
                
                return vec4<f32>(result, 1.0);
            }
        "#;
        
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Block1 Stage3 Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_code.into()),
        });
        
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Block1 Stage3 BGL"),
            entries: &[
                // Uniforms (binding 0)
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
                // Stage 2 output (processed Input 1) - binding 1
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
                // Sampler for Stage 2 - binding 2
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // Raw Input 2 - binding 3
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
                // Sampler for Input 2 - binding 4
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // Feedback buffer - binding 5
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
                // Sampler for Feedback - binding 6
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // Delay buffer - binding 7
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Sampler for Delay - binding 8
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Block1 Stage3 Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Block1 Stage3 Pipeline"),
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
                    blend: Some(wgpu::BlendState::REPLACE),
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
            label: Some("Block1 Stage3 Uniforms"),
            size: std::mem::size_of::<Stage3Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        (pipeline, bind_group_layout, uniforms)
    }
    
    /// Render Block 1 with all stages
    pub fn render(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input1_view: &wgpu::TextureView,
        input2_view: &wgpu::TextureView,
        params: &Block1Params,
    ) {
        // ============================================
        // CHANNEL 1: Stage 1 (Input Sampling) → Buffer A
        // ============================================
        // CH1 input select: 0=input1, 1=input2
        let ch1_use_input2 = params.ch1_input_select == 1;
        let stage1_ch1_uniforms = Stage1Uniforms {
            width: self.width as f32,
            height: self.height as f32,
            in1_scale: params.ch1_z_displace,
            in1_rotate: params.ch1_rotate,
            in1_x_displace: params.ch1_x_displace,
            in1_y_displace: params.ch1_y_displace,
            in2_scale: 1.0,
            in2_rotate: 0.0,
            in2_x_displace: 0.0,
            in2_y_displace: 0.0,
            fb_scale: 1.0,
            fb_rotate: 0.0,
            fb_x_displace: 0.0,
            fb_y_displace: 0.0,
            sample_input2: 0.0, // Always 0 - we bind the selected input to input1_tex
            sample_feedback: 0.0,
            kaleidoscope_amount: params.ch1_kaleidoscope_amount,
            kaleidoscope_slice: params.ch1_kaleidoscope_slice,
            h_mirror: if params.ch1_h_mirror { 1.0 } else { 0.0 },
            v_mirror: if params.ch1_v_mirror { 1.0 } else { 0.0 },
            h_flip: if params.ch1_h_flip { 1.0 } else { 0.0 },
            v_flip: if params.ch1_v_flip { 1.0 } else { 0.0 },
        };
        self.write_stage1_uniforms(queue, &self.stage1_uniforms_ch1, &stage1_ch1_uniforms);
        
        // Bind the appropriate input based on ch1_input_select
        let ch1_input_view = if ch1_use_input2 { input2_view } else { input1_view };
        // Bind dummy black to input2 since we're not using it
        let dummy_view = self.resources.get_feedback_view(); // Reuse as dummy
        let stage1_ch1_bind_group = self.get_stage1_bind_group(
            device, &self.stage1_uniforms_ch1, ch1_input_view, dummy_view, dummy_view
        );
        
        // Stage 1 CH1: Render to Buffer A
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Block1 Stage1 CH1"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.resources.buffer_a.view,
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
            render_pass.set_bind_group(0, &stage1_ch1_bind_group, &[]);
            render_pass.draw(0..6, 0..1);
        }
        
        // ============================================
        // CHANNEL 1: Stage 2 (Effects) → Buffer B
        // ============================================
        // CRITICAL: Must complete CH1 Stage 2 before CH2 Stage 1, 
        // because CH2 Stage 1 will overwrite buffer_a which Stage 2 needs to read!
        // 
        // OPTIMIZATION: Skip Stage 2 shader if no effects are enabled
        // Just copy buffer_a → buffer_b using hardware texture copy
        if self.should_skip_stage2_ch1(params) {
            // Fast path: hardware texture copy (~10x faster than shader)
            self.copy_texture_fast(
                encoder,
                &self.resources.buffer_a.texture,
                &self.resources.buffer_b.texture,
            );
        } else if let Some(ref stage2_pipeline) = self.stage2_pipeline {
            // Full effects shader path
            let stage2_ch1_uniforms = Stage2Uniforms {
                hsb_h: params.ch1_hsb_attenuate.x,
                hsb_s: params.ch1_hsb_attenuate.y,
                hsb_b: params.ch1_hsb_attenuate.z,
                filters_boost: params.ch1_filters_boost,
                blur_amount: params.ch1_blur_amount,
                blur_radius: params.ch1_blur_radius,
                _pad1: 0.0,
                _pad2: 0.0,
                sharpen_amount: params.ch1_sharpen_amount,
                sharpen_radius: params.ch1_sharpen_radius,
                _pad3: 0.0,
                _pad4: 0.0,
                width: self.width as f32,
                height: self.height as f32,
                inv_width: 1.0 / self.width as f32,
                inv_height: 1.0 / self.height as f32,
                hue_invert: if params.ch1_hue_invert { 1.0 } else { 0.0 },
                saturation_invert: if params.ch1_saturation_invert { 1.0 } else { 0.0 },
                bright_invert: if params.ch1_bright_invert { 1.0 } else { 0.0 },
                rgb_invert: if params.ch1_rgb_invert { 1.0 } else { 0.0 },
                solarize: if params.ch1_solarize { 1.0 } else { 0.0 },
                _pad5: 0.0,
                _pad6: 0.0,
                _pad7: 0.0,
                posterize: params.ch1_posterize,
                posterize_switch: if params.ch1_posterize_switch { 1.0 } else { 0.0 },
                _pad8: 0.0,
                _pad9: 0.0,
            };
            self.write_stage2_uniforms(queue, &self.stage2_uniforms_ch1, &stage2_ch1_uniforms);
            
            let stage2_ch1_bind_group = self.get_stage2_bind_group(
                device, &self.stage2_uniforms_ch1, &self.resources.buffer_a.view
            );
            
            {
                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Block1 Stage2 CH1"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.resources.buffer_b.view,
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
                
                render_pass.set_pipeline(stage2_pipeline);
                render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
                render_pass.set_bind_group(0, &stage2_ch1_bind_group, &[]);
                render_pass.draw(0..6, 0..1);
            }
        }
        
        // ============================================
        // CHANNEL 2: Stage 1 (Input Sampling) → Buffer A
        // NOTE: This MUST happen after CH1 Stage 2 is complete, 
        // because CH2 Stage 1 overwrites buffer_a
        // ============================================
        // Only process CH2 if it's being mixed
        if params.ch2_mix_amount > 0.001 {
            // CH2 input select: 0=input1, 1=input2
            let ch2_use_input2 = params.ch2_input_select == 1;
            let stage1_ch2_uniforms = Stage1Uniforms {
                width: self.width as f32,
                height: self.height as f32,
                in1_scale: params.ch2_z_displace,
                in1_rotate: params.ch2_rotate,
                in1_x_displace: params.ch2_x_displace,
                in1_y_displace: params.ch2_y_displace,
                in2_scale: 1.0,
                in2_rotate: 0.0,
                in2_x_displace: 0.0,
                in2_y_displace: 0.0,
                fb_scale: 1.0,
                fb_rotate: 0.0,
                fb_x_displace: 0.0,
                fb_y_displace: 0.0,
                sample_input2: 0.0, // Always 0 - we bind the selected input to input1_tex
                sample_feedback: 0.0,
                kaleidoscope_amount: params.ch2_kaleidoscope_amount,
                kaleidoscope_slice: params.ch2_kaleidoscope_slice,
                h_mirror: if params.ch2_h_mirror { 1.0 } else { 0.0 },
                v_mirror: if params.ch2_v_mirror { 1.0 } else { 0.0 },
                h_flip: if params.ch2_h_flip { 1.0 } else { 0.0 },
                v_flip: if params.ch2_v_flip { 1.0 } else { 0.0 },
            };
            self.write_stage1_uniforms(queue, &self.stage1_uniforms_ch2, &stage1_ch2_uniforms);
            
            // Bind the appropriate input based on ch2_input_select
            let ch2_input_view = if ch2_use_input2 { input2_view } else { input1_view };
            // Bind dummy black to unused slots
            let dummy_view = self.resources.get_feedback_view();
            let stage1_ch2_bind_group = self.get_stage1_bind_group(
                device, &self.stage1_uniforms_ch2, ch2_input_view, dummy_view, dummy_view
            );
            
            // Stage 1 CH2: Render to Buffer A (temporarily, will be overwritten by Stage 3 output later)
            {
                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Block1 Stage1 CH2"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.resources.buffer_a.view, // Use buffer_a temporarily
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
                render_pass.set_bind_group(0, &stage1_ch2_bind_group, &[]);
                render_pass.draw(0..6, 0..1);
            }
            
            // ============================================
            // CHANNEL 2: Stage 2 (Effects) → CH2 Buffer
            // ============================================
            // OPTIMIZATION: Skip Stage 2 shader if no effects are enabled
            if self.should_skip_stage2_ch2(params) {
                // Fast path: hardware texture copy
                self.copy_texture_fast(
                    encoder,
                    &self.resources.buffer_a.texture,
                    &self.resources.ch2_buffer.texture,
                );
            } else if let Some(ref stage2_pipeline) = self.stage2_pipeline {
                // Full effects shader path
                let stage2_ch2_uniforms = Stage2Uniforms {
                        hsb_h: params.ch2_hsb_attenuate.x,
                        hsb_s: params.ch2_hsb_attenuate.y,
                        hsb_b: params.ch2_hsb_attenuate.z,
                        filters_boost: params.ch2_filters_boost,
                        blur_amount: params.ch2_blur_amount,
                        blur_radius: params.ch2_blur_radius,
                        _pad1: 0.0,
                        _pad2: 0.0,
                        sharpen_amount: params.ch2_sharpen_amount,
                        sharpen_radius: params.ch2_sharpen_radius,
                        _pad3: 0.0,
                        _pad4: 0.0,
                        width: self.width as f32,
                        height: self.height as f32,
                        inv_width: 1.0 / self.width as f32,
                        inv_height: 1.0 / self.height as f32,
                        hue_invert: if params.ch2_hue_invert { 1.0 } else { 0.0 },
                        saturation_invert: if params.ch2_saturation_invert { 1.0 } else { 0.0 },
                        bright_invert: if params.ch2_bright_invert { 1.0 } else { 0.0 },
                        rgb_invert: if params.ch2_rgb_invert { 1.0 } else { 0.0 },
                        solarize: if params.ch2_solarize { 1.0 } else { 0.0 },
                        _pad5: 0.0,
                        _pad6: 0.0,
                        _pad7: 0.0,
                        posterize: params.ch2_posterize,
                        posterize_switch: if params.ch2_posterize_switch { 1.0 } else { 0.0 },
                        _pad8: 0.0,
                        _pad9: 0.0,
                    };
                    self.write_stage2_uniforms(queue, &self.stage2_uniforms_ch2, &stage2_ch2_uniforms);
                    
                    // Read from buffer_a (Stage1 CH2 output), write to ch2_buffer
                    let stage2_ch2_bind_group = self.get_stage2_bind_group(
                        device, &self.stage2_uniforms_ch2, &self.resources.buffer_a.view
                    );
                    
                    {
                        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("Block1 Stage2 CH2"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: &self.resources.ch2_buffer.view,
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
                        
                        render_pass.set_pipeline(stage2_pipeline);
                        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
                        render_pass.set_bind_group(0, &stage2_ch2_bind_group, &[]);
                        render_pass.draw(0..6, 0..1);
                    }
                }
        }
        
        // ============================================
        // Stage 3: Mixing (CH1 + CH2 + FB) → Buffer A
        // ============================================
        let stage3_uniforms = Stage3Uniforms {
                input2_amount: params.ch2_mix_amount,
                input2_mix_type: params.ch2_mix_type,
                input2_mix_overflow: params.ch2_mix_overflow,
                _pad0: 0.0,
                feedback_amount: params.fb1_mix_amount,
                feedback_mix_type: params.fb1_mix_type,
                feedback_mix_overflow: params.fb1_mix_overflow,
                _pad1: 0.0,
                // CH2 keying
                ch2_key_value_r: params.ch2_key_value_red,
                ch2_key_value_g: params.ch2_key_value_green,
                ch2_key_value_b: params.ch2_key_value_blue,
                ch2_key_threshold: params.ch2_key_threshold,
                ch2_key_soft: params.ch2_key_soft,
                ch2_key_mode: params.ch2_key_mode,
                ch2_key_order: params.ch2_key_order,
                // FB1 keying
                fb1_key_value_r: params.fb1_key_value_red,
                fb1_key_value_g: params.fb1_key_value_green,
                fb1_key_value_b: params.fb1_key_value_blue,
                fb1_key_threshold: params.fb1_key_threshold,
                fb1_key_soft: params.fb1_key_soft,
                fb1_key_mode: params.fb1_key_mode,
                fb1_key_order: params.fb1_key_order,
                _pad_before_hsb: [0.0, 0.0],
                fb1_hsb_offset: params.fb1_hsb_offset.to_array(),
                _pad_after_hsb_offset: 0.0,
                fb1_hsb_attenuate: params.fb1_hsb_attenuate.to_array(),
                _pad_after_hsb_attenuate: 0.0,
                fb1_hsb_powmap: params.fb1_hsb_powmap.to_array(),
                fb1_hue_shaper: params.fb1_hue_shaper,
                fb1_posterize: params.fb1_posterize,
                fb1_posterize_switch: if params.fb1_posterize_switch { 1.0 } else { 0.0 },
                fb1_hue_invert: if params.fb1_hue_invert { 1.0 } else { 0.0 },
                fb1_saturation_invert: if params.fb1_saturation_invert { 1.0 } else { 0.0 },
                fb1_bright_invert: if params.fb1_bright_invert { 1.0 } else { 0.0 },
                _pad2: 0.0,
                // Geometric transforms (must match struct order)
                fb1_x_displace: params.fb1_x_displace,
                fb1_y_displace: params.fb1_y_displace,
                fb1_z_displace: params.fb1_z_displace,
                fb1_rotate: params.fb1_rotate,
                fb1_kaleidoscope_amount: params.fb1_kaleidoscope_amount,
                fb1_kaleidoscope_slice: params.fb1_kaleidoscope_slice,
                fb1_h_mirror: if params.fb1_h_mirror { 1.0 } else { 0.0 },
                fb1_v_mirror: if params.fb1_v_mirror { 1.0 } else { 0.0 },
                // Filters
                fb1_blur_amount: params.fb1_blur_amount,
                fb1_blur_radius: params.fb1_blur_radius,
                fb1_sharpen_amount: params.fb1_sharpen_amount,
                fb1_sharpen_radius: params.fb1_sharpen_radius,
                fb1_filters_boost: params.fb1_filters_boost,
                _pad3: 0.0,
                _pad4: 0.0,
                // Temporal filters
                fb1_temporal_filter1_amount: params.fb1_temporal_filter1_amount,
                fb1_temporal_filter1_resonance: params.fb1_temporal_filter1_resonance,
                fb1_temporal_filter2_amount: params.fb1_temporal_filter2_amount,
                fb1_temporal_filter2_resonance: params.fb1_temporal_filter2_resonance,
                // Delay
                fb1_delay_time: params.fb1_delay_time,
                _pad5: 0,
                _pad6: 0,
                _pad7: 0,
        };
        self.write_stage3_uniforms(queue, &stage3_uniforms);
        
        let stage3_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Block1 Stage3 Bind Group"),
            layout: &self.stage3_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.stage3_uniforms.as_entire_binding() },
                // Processed CH1 (from buffer_b)
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.resources.buffer_b.view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.create_sampler(device)) },
                // Processed CH2 (from ch2_buffer)
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.resources.ch2_buffer.view) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&self.create_sampler(device)) },
                // Feedback
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(self.resources.get_feedback_view()) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::Sampler(&self.create_sampler(device)) },
                // Delay buffer (for feedback delay)
                wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(self.resources.get_delay_view(params.fb1_delay_time as usize)) },
                wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::Sampler(&self.create_sampler(device)) },
            ],
        });
        
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Block1 Stage3"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.resources.buffer_a.view,
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
            render_pass.set_bind_group(0, &stage3_bind_group, &[]);
            render_pass.draw(0..6, 0..1);
        }
        
    self.resources.current_output = 0;
    
    // Update delay buffer ring with current output
    self.resources.update_delay_buffer(encoder);
    }
    
    /// Check if Stage 2 effects are effectively disabled for CH1
    /// Returns true if we can skip the effects shader and just copy the texture
    fn should_skip_stage2_ch1(&self, params: &Block1Params) -> bool {
        // Blur and sharpen are the expensive operations
        if params.ch1_blur_amount > 0.001 || params.ch1_sharpen_amount > 0.001 {
            return false;
        }
        
        // Check HSB adjustments (1.0 = no change)
        let hsb_unchanged = (params.ch1_hsb_attenuate.x - 1.0).abs() < 0.001
            && (params.ch1_hsb_attenuate.y - 1.0).abs() < 0.001
            && (params.ch1_hsb_attenuate.z - 1.0).abs() < 0.001;
        
        // Check inverts
        let no_inverts = !params.ch1_hue_invert 
            && !params.ch1_saturation_invert 
            && !params.ch1_bright_invert
            && !params.ch1_rgb_invert;
        
        // Check solarize
        let no_solarize = !params.ch1_solarize;
        
        // Check posterize (levels < 2 means effectively disabled)
        let no_posterize = params.ch1_posterize < 2.0 || !params.ch1_posterize_switch;
        
        hsb_unchanged && no_inverts && no_solarize && no_posterize
    }
    
    /// Check if Stage 2 effects are effectively disabled for CH2
    fn should_skip_stage2_ch2(&self, params: &Block1Params) -> bool {
        // Blur and sharpen are the expensive operations
        if params.ch2_blur_amount > 0.001 || params.ch2_sharpen_amount > 0.001 {
            return false;
        }
        
        // Check HSB adjustments
        let hsb_unchanged = (params.ch2_hsb_attenuate.x - 1.0).abs() < 0.001
            && (params.ch2_hsb_attenuate.y - 1.0).abs() < 0.001
            && (params.ch2_hsb_attenuate.z - 1.0).abs() < 0.001;
        
        let no_inverts = !params.ch2_hue_invert 
            && !params.ch2_saturation_invert 
            && !params.ch2_bright_invert
            && !params.ch2_rgb_invert;
        
        let no_solarize = !params.ch2_solarize;
        let no_posterize = params.ch2_posterize < 2.0 || !params.ch2_posterize_switch;
        
        hsb_unchanged && no_inverts && no_solarize && no_posterize
    }
    
    /// Fast path: copy texture using encoder instead of shader render
    /// Used when Stage 2 effects are disabled
    fn copy_texture_fast(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        source: &wgpu::Texture,
        dest: &wgpu::Texture,
    ) {
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: source,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: dest,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
    }
    
    /// Helper to write Stage 1 uniforms
    fn write_stage1_uniforms(&self, queue: &wgpu::Queue, target_buffer: &wgpu::Buffer, uniforms: &Stage1Uniforms) {
        let uniform_bytes = unsafe {
            std::slice::from_raw_parts(
                uniforms as *const _ as *const u8,
                std::mem::size_of::<Stage1Uniforms>()
            )
        };
        queue.write_buffer(target_buffer, 0, uniform_bytes);
    }
    
    /// Helper to write Stage 2 uniforms
    fn write_stage2_uniforms(&self, queue: &wgpu::Queue, target_buffer: &wgpu::Buffer, uniforms: &Stage2Uniforms) {
        let uniform_bytes = unsafe {
            std::slice::from_raw_parts(
                uniforms as *const _ as *const u8,
                std::mem::size_of::<Stage2Uniforms>()
            )
        };
        queue.write_buffer(target_buffer, 0, uniform_bytes);
    }
    
    /// Helper to write Stage 3 uniforms
    fn write_stage3_uniforms(&self, queue: &wgpu::Queue, uniforms: &Stage3Uniforms) {
        let uniform_bytes = unsafe {
            std::slice::from_raw_parts(
                uniforms as *const _ as *const u8,
                std::mem::size_of::<Stage3Uniforms>()
            )
        };
        queue.write_buffer(&self.stage3_uniforms, 0, uniform_bytes);
    }
    
    /// Get or create cached sampler
    fn get_sampler(&self, device: &wgpu::Device) -> wgpu::Sampler {
        let mut cache = self.sampler_cache.borrow_mut();
        if cache.is_none() {
            *cache = Some(device.create_sampler(&wgpu::SamplerDescriptor {
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            }));
        }
        cache.as_ref().unwrap().clone()
    }
    
    /// Get cached Stage 1 bind group or create if inputs changed
    fn get_stage1_bind_group(
        &self,
        device: &wgpu::Device,
        uniforms: &wgpu::Buffer,
        input1_view: &wgpu::TextureView,
        input2_view: &wgpu::TextureView,
        feedback_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        // Create cache key from texture view pointers AND uniform buffer
        // This is critical: CH1 and CH2 use different uniform buffers for transforms
        let key = (
            uniforms as *const _ as u64,  // Include uniform buffer in key!
            input1_view as *const _ as u64,
            input2_view as *const _ as u64,
            feedback_view as *const _ as u64,
        );
        
        // Check cache
        let mut cache = self.stage1_bind_group_cache.borrow_mut();
        if !cache.contains_key(&key) {
            // Create new bind group
            let sampler = self.get_sampler(device);
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Block1 Stage1 Bind Group"),
                layout: &self.stage1_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: uniforms.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(input1_view) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&sampler) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(input2_view) },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&sampler) },
                    wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(feedback_view) },
                    wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::Sampler(&sampler) },
                ],
            });
            cache.insert(key, bind_group);
            
            // Limit cache size to prevent unbounded growth
            if cache.len() > 16 {
                // Remove oldest entry (simple approach)
                let oldest_key = *cache.keys().next().unwrap();
                cache.remove(&oldest_key);
            }
        }
        
        cache.get(&key).unwrap().clone()
    }
    
    /// Get cached Stage 2 bind group or create if input changed
    fn get_stage2_bind_group(
        &self,
        device: &wgpu::Device,
        uniforms: &wgpu::Buffer,
        input_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        // Include uniform buffer in key - CH1 and CH2 use different uniforms
        let key = (
            uniforms as *const _ as u64,
            input_view as *const _ as u64,
        );
        
        let mut cache = self.stage2_bind_group_cache.borrow_mut();
        if !cache.contains_key(&key) {
            let sampler = self.get_sampler(device);
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Block1 Stage2 Bind Group"),
                layout: &self.stage2_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: uniforms.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(input_view) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&sampler) },
                ],
            });
            cache.insert(key, bind_group);
            
            // Limit cache size
            if cache.len() > 8 {
                let oldest_key = *cache.keys().next().unwrap();
                cache.remove(&oldest_key);
            }
        }
        
        cache.get(&key).unwrap().clone()
    }
    
    /// Get the output view (final result from Stage 3)
    pub fn get_output_view(&self) -> &wgpu::TextureView {
        // Stage 3 output is always in buffer_a
        &self.resources.buffer_a.view
    }
    
    /// Update feedback for next frame
    pub fn update_feedback(&self, encoder: &mut wgpu::CommandEncoder) {
        self.resources.update_feedback(encoder);
    }

    /// Clear all internal buffers (feedback + ping-pong) to black.
    /// Call on input switch to avoid stale frames bleeding through the feedback path.
    pub fn clear_all(&self, queue: &wgpu::Queue) {
        self.resources.clear_all(queue);
    }

    /// Invalidate the bind group caches.
    ///
    /// Required on input switch because the cache keys are raw pointer addresses.
    /// When the old texture is dropped and a new one allocated, the allocator may
    /// reuse the same address — causing a stale cache hit that renders the old frame.
    pub fn invalidate_bind_group_caches(&self) {
        self.stage1_bind_group_cache.borrow_mut().clear();
        self.stage2_bind_group_cache.borrow_mut().clear();
    }
    
    /// Create a sampler
    fn create_sampler(&self, device: &wgpu::Device) -> wgpu::Sampler {
        device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        })
    }
}

// Layout verification - this will fail to compile if the struct changes
#[cfg(test)]
mod layout_tests {
    use super::*;
    
    #[test]
    fn verify_stage3_uniforms_size() {
        assert_eq!(std::mem::size_of::<Stage3Uniforms>(), 272);
    }
}
