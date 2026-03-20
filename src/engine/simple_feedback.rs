//! # Simple Feedback Shader
//! 
//! Minimal single-shader feedback loop:
//! - Webcam input (VGA 640x480 scaled to 1280x720)
//! - Feedback with hue shift (LFO-modulated)
//! - Feedback with rotate + zoom (tempo-synced LFO)
//! - Ping-pong or single texture feedback

use crate::engine::pipelines::COMMON_VERTEX_SHADER;

/// Simple uniform struct with dual-input mixing and keying
#[repr(C, align(16))]
#[derive(Debug, Copy, Clone, bytemuck::AnyBitPattern)]
pub struct SimpleFeedbackUniforms {
    // Resolution (16 bytes)
    pub width: f32,
    pub height: f32,
    pub inv_width: f32,
    pub inv_height: f32,
    
    // LFO values (tempo-synced, 0-1 range) (16 bytes)
    pub hue_lfo: f32,      // Cycles through hue shift
    pub rotate_lfo: f32,   // Cycles through rotation
    pub zoom_lfo: f32,     // Cycles through zoom
    pub _pad1: f32,
    
    // Manual controls (16 bytes)
    pub feedback_amount: f32,  // Mix between input and feedback
    pub hue_amount: f32,       // How much LFO affects hue (0-1)
    pub rotate_amount: f32,    // How much LFO affects rotation (0-1)
    pub zoom_amount: f32,      // How much LFO affects zoom (0-1)
    
    // Additional controls (16 bytes)
    pub rotate_center_x: f32,  // Rotation center (0.5 = center)
    pub rotate_center_y: f32,
    pub zoom_center_x: f32,
    pub zoom_center_y: f32,
    
    // Manual offsets (16 bytes)
    pub manual_rotate: f32,    // Manual rotation (radians)
    pub manual_zoom: f32,      // Manual zoom (1.0 = no zoom)
    pub manual_translate_x: f32,
    pub manual_translate_y: f32,
    
    // Mixing parameters (16 bytes)
    pub mix_amount: f32,       // Mix between input1 and input2 (0-1)
    pub mix_type: i32,         // 0=Lerp, 1=Add, 2=Diff, 3=Mult, 4=Dodge
    pub key_order: i32,        // 0=Key First Then Mix, 1=Mix First Then Key
    pub _pad2: f32,
    
    // Keying parameters (16 bytes)
    pub key_threshold: f32,    // Key threshold (-1 to 1)
    pub key_softness: f32,     // Key softness (0 to 1)
    pub key_invert: f32,       // Invert key (0 or 1)
    pub key_type: i32,         // 0=Lumakey, 1=Chromakey
}

impl Default for SimpleFeedbackUniforms {
    fn default() -> Self {
        Self {
            width: 1280.0,
            height: 720.0,
            inv_width: 1.0 / 1280.0,
            inv_height: 1.0 / 720.0,
            hue_lfo: 0.0,
            rotate_lfo: 0.0,
            zoom_lfo: 0.0,
            _pad1: 0.0,
            feedback_amount: 0.7,
            hue_amount: 0.5,
            rotate_amount: 0.2,
            zoom_amount: 0.1,
            rotate_center_x: 0.5,
            rotate_center_y: 0.5,
            zoom_center_x: 0.5,
            zoom_center_y: 0.5,
            manual_rotate: 0.0,
            manual_zoom: 1.0,
            manual_translate_x: 0.0,
            manual_translate_y: 0.0,
            // Mixing defaults
            mix_amount: 0.0,
            mix_type: 0,
            key_order: 0,
            _pad2: 0.0,
            // Keying defaults
            key_threshold: 0.0,
            key_softness: 0.0,
            key_invert: 0.0,
            key_type: 0,
        }
    }
}

/// Simple feedback pipeline
pub struct SimpleFeedbackPipeline {
    pub pipeline: wgpu::RenderPipeline,
    pub bind_group: wgpu::BindGroup,
    pub uniform_buffer: wgpu::Buffer,
    pub bind_group_layout: wgpu::BindGroupLayout,
    width: u32,
    height: u32,
    
    // Blit pipeline for presenting output
    pub blit_pipeline: wgpu::RenderPipeline,
    pub blit_bind_group_layout: wgpu::BindGroupLayout,
    blit_sampler: wgpu::Sampler,
    surface_format: wgpu::TextureFormat,
}

/// Vertex descriptor for simple engine (position + texcoord)
fn simple_vertex_desc() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: 4 * std::mem::size_of::<f32>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x2, // position
            },
            wgpu::VertexAttribute {
                offset: 2 * std::mem::size_of::<f32>() as wgpu::BufferAddress,
                shader_location: 1,
                format: wgpu::VertexFormat::Float32x2, // texcoord
            },
        ],
    }
}

// Simple blit shader for presenting output
const BLIT_VERTEX_SHADER: &str = r#"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) texcoord: vec2<f32>,
};

@vertex
fn vs_main(@location(0) position: vec2<f32>, @location(1) texcoord: vec2<f32>) -> VertexOutput {
    var output: VertexOutput;
    output.position = vec4<f32>(position, 0.0, 1.0);
    output.texcoord = texcoord;
    return output;
}
"#;

const BLIT_FRAGMENT_SHADER: &str = r#"
@group(0) @binding(0)
var source_tex: texture_2d<f32>;
@group(0) @binding(1)
var source_sampler: sampler;

@fragment
fn fs_main(@location(0) texcoord: vec2<f32>) -> @location(0) vec4<f32> {
    return textureSample(source_tex, source_sampler, texcoord);
}
"#;

impl SimpleFeedbackPipeline {
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        Self::new_with_format(device, width, height, wgpu::TextureFormat::Bgra8Unorm)
    }
    
    pub fn new_with_format(device: &wgpu::Device, width: u32, height: u32, format: wgpu::TextureFormat) -> Self {
        log::info!("[SIMPLE] Creating feedback pipeline: {}x{} with format {:?}", width, height, format);
        
        let shader_code = format!(r#"
{}

@group(0) @binding(0)
var<uniform> uniforms: SimpleFeedbackUniforms;

@group(0) @binding(1)
var input1_tex: texture_2d<f32>;
@group(0) @binding(2)
var input1_sampler: sampler;

@group(0) @binding(3)
var input2_tex: texture_2d<f32>;
@group(0) @binding(4)
var input2_sampler: sampler;

@group(0) @binding(5)
var feedback_tex: texture_2d<f32>;
@group(0) @binding(6)
var feedback_sampler: sampler;

struct SimpleFeedbackUniforms {{
    width: f32,
    height: f32,
    inv_width: f32,
    inv_height: f32,
    
    hue_lfo: f32,
    rotate_lfo: f32,
    zoom_lfo: f32,
    _pad1: f32,
    
    feedback_amount: f32,
    hue_amount: f32,
    rotate_amount: f32,
    zoom_amount: f32,
    
    rotate_center_x: f32,
    rotate_center_y: f32,
    zoom_center_x: f32,
    zoom_center_y: f32,
    
    manual_rotate: f32,
    manual_zoom: f32,
    manual_translate_x: f32,
    manual_translate_y: f32,
    
    // Mixing parameters
    mix_amount: f32,
    mix_type: i32,
    key_order: i32,
    _pad2: f32,
    
    // Keying parameters
    key_threshold: f32,
    key_softness: f32,
    key_invert: f32,
    key_type: i32,
}}

// RGB to HSB conversion
fn rgb2hsb(c: vec3<f32>) -> vec3<f32> {{
    let K = vec4<f32>(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    let p = mix(vec4<f32>(c.bg, K.wz), vec4<f32>(c.gb, K.xy), step(c.b, c.g));
    let q = mix(vec4<f32>(p.xyw, c.r), vec4<f32>(c.r, p.yzx), step(p.x, c.r));
    let d = q.x - min(q.w, q.y);
    let e = 1.0e-10;
    return vec3<f32>(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}}

// HSB to RGB conversion
fn hsb2rgb(c: vec3<f32>) -> vec3<f32> {{
    let K = vec4<f32>(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    let p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, vec3<f32>(0.0), vec3<f32>(1.0)), c.y);
}}

// Apply hue shift based on LFO
fn apply_hue_shift(color: vec3<f32>, lfo_value: f32, amount: f32) -> vec3<f32> {{
    if (amount < 0.001) {{
        return color;
    }}
    
    var hsb = rgb2hsb(color);
    // LFO value 0-1 maps to 0-2PI hue shift
    let hue_shift = lfo_value * 2.0 * 3.14159265 * amount;
    hsb.x = fract(hsb.x + hue_shift);
    return hsb2rgb(hsb);
}}

// Rotate and zoom UV coordinates
fn transform_uv(uv: vec2<f32>, rotate_lfo: f32, zoom_lfo: f32, 
                rotate_amount: f32, zoom_amount: f32,
                manual_rotate: f32, manual_zoom: f32,
                manual_translate: vec2<f32>) -> vec2<f32> {{
    
    // Center UVs
    var result = uv - vec2<f32>(0.5);
    
    // Apply rotation (LFO + manual)
    let total_rotate = rotate_lfo * 2.0 * 3.14159265 * rotate_amount + manual_rotate;
    let cos_r = cos(total_rotate);
    let sin_r = sin(total_rotate);
    let rotated_x = result.x * cos_r - result.y * sin_r;
    let rotated_y = result.x * sin_r + result.y * cos_r;
    result = vec2<f32>(rotated_x, rotated_y);
    
    // Apply zoom (LFO + manual)
    // LFO 0-1 maps to 0.5-1.5 zoom range when amount is 1.0
    let lfo_zoom = 1.0 + (zoom_lfo - 0.5) * zoom_amount;
    let total_zoom = lfo_zoom * manual_zoom;
    result = result / total_zoom;
    
    // Apply translation
    result = result - manual_translate;
    
    // Uncenter
    return result + vec2<f32>(0.5);
}}

// Calculate luminance of a color
fn luminance(c: vec3<f32>) -> f32 {{
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}}

// Calculate key mix amount based on color and keying parameters
fn calculate_key(color: vec3<f32>, threshold: f32, softness: f32, invert: f32, key_type: i32) -> f32 {{
    var key_value: f32;
    
    if (key_type == 0) {{
        // Lumakey - key based on brightness
        key_value = luminance(color);
    }} else {{
        // Chromakey - key based on green channel dominance
        let green_dominance = color.g - (color.r + color.b) * 0.5;
        key_value = green_dominance + 0.5; // Center around 0.5
    }}
    
    // Apply threshold and softness
    let soft = max(softness, 0.001); // Prevent division by zero
    var key_mix = smoothstep(threshold - soft * 0.5, threshold + soft * 0.5, key_value);
    
    // Invert if requested
    if (invert > 0.5) {{
        key_mix = 1.0 - key_mix;
    }}
    
    return key_mix;
}}

// Apply overflow/clamping
fn apply_overflow(color: vec3<f32>, overflow: i32) -> vec3<f32> {{
    switch(overflow) {{
        case 0: {{ return clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)); }} // Clamp
        case 1: {{ return fract(color); }} // Wrap
        case 2: {{ return 1.0 - abs(fract(color) * 2.0 - 1.0); }} // Fold
        default: {{ return clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)); }}
    }}
}}

// Mix two colors with different blend modes
fn mix_colors(fg: vec3<f32>, bg: vec3<f32>, amount: f32, mix_type: i32) -> vec3<f32> {{
    switch(mix_type) {{
        case 0: {{ return mix(fg, bg, amount); }} // Lerp
        case 1: {{ return fg + bg * amount; }} // Add
        case 2: {{ return fg - bg * amount; }} // Diff
        case 3: {{ return fg * (1.0 - amount) + fg * bg * amount; }} // Mult
        case 4: {{ return fg / (1.0 - bg * amount + 0.001); }} // Dodge
        default: {{ return mix(fg, bg, amount); }}
    }}
}}

// Mix and key two video sources
fn mixn_key_video(fg: vec3<f32>, bg: vec3<f32>, amount: f32, mix_type: i32,
                  key_threshold: f32, key_softness: f32, key_invert: f32, key_type: i32,
                  key_order: i32) -> vec3<f32> {{
    var mixed: vec3<f32>;
    
    if (key_order == 0) {{
        // Key First Then Mix: key the foreground, then mix with background
        let key_amount = calculate_key(fg, key_threshold, key_softness, key_invert, key_type);
        let keyed_fg = mix(fg, bg, key_amount);
        mixed = mix_colors(keyed_fg, bg, amount, mix_type);
    }} else {{
        // Mix First Then Key: mix first, then key the result
        mixed = mix_colors(fg, bg, amount, mix_type);
        let key_amount = calculate_key(mixed, key_threshold, key_softness, key_invert, key_type);
        mixed = mix(mixed, bg, key_amount);
    }}
    
    return mixed;
}}

@fragment
fn fs_main(@location(0) texcoord: vec2<f32>) -> @location(0) vec4<f32> {{
    // Sample both inputs (Input 2 is the same camera, unmodified)
    let input1_color = textureSample(input1_tex, input1_sampler, texcoord);
    let input2_color = textureSample(input2_tex, input2_sampler, texcoord);
    
    // Apply hue shift to Input 1 only (before mixing)
    let shifted_input1 = apply_hue_shift(input1_color.rgb, uniforms.hue_lfo, uniforms.hue_amount);
    
    // Mix Input 1 (hue-shifted) with Input 2 (normal)
    let mixed_input = mixn_key_video(
        shifted_input1, 
        input2_color.rgb, 
        uniforms.mix_amount, 
        uniforms.mix_type,
        uniforms.key_threshold,
        uniforms.key_softness,
        uniforms.key_invert,
        uniforms.key_type,
        uniforms.key_order
    );
    
    // Transform UVs for feedback sampling
    let feedback_uv = transform_uv(
        texcoord,
        uniforms.rotate_lfo,
        uniforms.zoom_lfo,
        uniforms.rotate_amount,
        uniforms.zoom_amount,
        uniforms.manual_rotate,
        uniforms.manual_zoom,
        vec2<f32>(uniforms.manual_translate_x, uniforms.manual_translate_y)
    );
    
    // Sample feedback with transformed UVs (no hue shift on feedback)
    let feedback_color = textureSample(feedback_tex, feedback_sampler, feedback_uv);
    
    // Mix input with feedback
    let output_rgb = mix(mixed_input, feedback_color.rgb, uniforms.feedback_amount);
    
    return vec4<f32>(output_rgb, 1.0);
}}
"#, COMMON_VERTEX_SHADER);
        
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Simple Feedback Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_code.into()),
        });
        
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Simple Feedback Bind Group Layout"),
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
                // Input 1 texture
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
                // Input 1 sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // Input 2 texture
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
                // Input 2 sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // Feedback texture
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
                // Feedback sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Simple Feedback Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Simple Feedback Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[simple_vertex_desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Simple Feedback Uniform Buffer"),
            size: std::mem::size_of::<SimpleFeedbackUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Create dummy bind group (will be replaced on first frame)
        let dummy_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Dummy Texture"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let dummy_view = dummy_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let dummy_sampler = device.create_sampler(&wgpu::SamplerDescriptor::default());
        
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Simple Feedback Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&dummy_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&dummy_sampler) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&dummy_view) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&dummy_sampler) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&dummy_view) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::Sampler(&dummy_sampler) },
            ],
        });
        
        // Create blit pipeline for presenting output to surface
        let blit_shader_code = format!("{BLIT_VERTEX_SHADER}{BLIT_FRAGMENT_SHADER}");
        let blit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Blit Shader"),
            source: wgpu::ShaderSource::Wgsl(blit_shader_code.into()),
        });
        
        let blit_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Blit Bind Group Layout"),
            entries: &[
                // Source texture
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Source sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        
        let blit_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Blit Pipeline Layout"),
            bind_group_layouts: &[&blit_bind_group_layout],
            push_constant_ranges: &[],
        });
        
        let blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Blit Pipeline"),
            layout: Some(&blit_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &blit_shader,
                entry_point: Some("vs_main"),
                buffers: &[simple_vertex_desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &blit_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format, // Same format as surface
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        
        let blit_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        
        Self {
            pipeline,
            bind_group,
            uniform_buffer,
            bind_group_layout,
            width,
            height,
            blit_pipeline,
            blit_bind_group_layout,
            blit_sampler,
            surface_format: format,
        }
    }
    
    pub fn update_params(&self, queue: &wgpu::Queue, params: &SimpleFeedbackUniforms) {
        let bytes = unsafe {
            std::slice::from_raw_parts(
                params as *const _ as *const u8,
                std::mem::size_of::<SimpleFeedbackUniforms>()
            )
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytes);
    }
    
    pub fn update_textures(
        &mut self,
        device: &wgpu::Device,
        input1_view: &wgpu::TextureView,
        input2_view: &wgpu::TextureView,
        feedback_view: &wgpu::TextureView,
    ) {
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        
        self.bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Simple Feedback Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.uniform_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(input1_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&sampler) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(input2_view) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&sampler) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(feedback_view) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::Sampler(&sampler) },
            ],
        });
    }
    
    /// Create a bind group for the blit pipeline
    pub fn create_blit_bind_group(&self, device: &wgpu::Device, source_view: &wgpu::TextureView) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Blit Bind Group"),
            layout: &self.blit_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(source_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.blit_sampler) },
            ],
        })
    }
}
