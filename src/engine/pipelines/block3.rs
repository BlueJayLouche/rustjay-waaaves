//! # Block 3 Pipeline
//!
//! Final mixing and output with WGSL shader.

use crate::engine::pipelines::{create_pipeline_layout, create_render_pipeline, COMMON_VERTEX_SHADER};
use crate::params::Block3Params;
use glam::Vec4;

/// Vec3 type matching WGSL's vec3<f32> - 16-byte aligned but only 12 bytes of data
#[repr(C, align(16))]
#[derive(Debug, Copy, Clone, bytemuck::AnyBitPattern)]
pub struct Vec3([f32; 3]);

impl Vec3 {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self([x, y, z])
    }
}

impl From<Vec4> for Vec3 {
    fn from(v: Vec4) -> Self {
        Self([v.x, v.y, v.z])
    }
}

impl From<glam::Vec3> for Vec3 {
    fn from(v: glam::Vec3) -> Self {
        Self([v.x, v.y, v.z])
    }
}

/// Uniforms for Block3 shader
#[repr(C, align(16))]
#[derive(Debug, Copy, Clone, bytemuck::AnyBitPattern)]
pub struct Block3Uniforms {
    pub width: f32,
    pub height: f32,
    pub inv_width: f32,
    pub inv_height: f32,
    
    // Block 1 re-process
    pub block1_xy_displace: [f32; 2],
    pub block1_z_displace: f32,
    pub block1_rotate: f32,
    pub block1_shear_matrix: Vec4,
    pub block1_kaleidoscope: f32,
    pub block1_kaleidoscope_slice: f32,
    pub block1_blur_amount: f32,
    pub block1_blur_radius: f32,
    pub block1_sharpen_amount: f32,
    pub block1_sharpen_radius: f32,
    pub block1_filters_boost: f32,
    pub block1_dither: f32,
    pub block1_switches: u32, // colorize_switch, dither_switch
    pub block1_colorize_mode: i32,
    pub block1_dither_type: i32,
    pub _pad1: f32,
    
    pub block1_colorize_band1: Vec3,
    pub _pad2: f32,
    pub block1_colorize_band2: Vec3,
    pub _pad3: f32,
    pub block1_colorize_band3: Vec3,
    pub _pad4: f32,
    pub block1_colorize_band4: Vec3,
    pub _pad5: f32,
    pub block1_colorize_band5: Vec3,
    pub _pad6: f32,
    
    // Block 2 re-process
    pub block2_xy_displace: [f32; 2],
    pub block2_z_displace: f32,
    pub block2_rotate: f32,
    pub block2_shear_matrix: Vec4,
    pub block2_kaleidoscope: f32,
    pub block2_kaleidoscope_slice: f32,
    pub block2_blur_amount: f32,
    pub block2_blur_radius: f32,
    pub block2_sharpen_amount: f32,
    pub block2_sharpen_radius: f32,
    pub block2_filters_boost: f32,
    pub block2_dither: f32,
    pub block2_switches: u32,
    pub block2_colorize_mode: i32,
    pub block2_dither_type: i32,
    pub _pad7: f32,
    
    pub block2_colorize_band1: Vec3,
    pub _pad8: f32,
    pub block2_colorize_band2: Vec3,
    pub _pad9: f32,
    pub block2_colorize_band3: Vec3,
    pub _pad10: f32,
    pub block2_colorize_band4: Vec3,
    pub _pad11: f32,
    pub block2_colorize_band5: Vec3,
    pub _pad12: f32,
    
    // Matrix mixer
    pub matrix_mix_type: i32,
    pub matrix_mix_overflow: i32,
    pub _pad13: [f32; 2],
    pub bg_into_fg_red: Vec3,
    pub _pad14: f32,
    pub bg_into_fg_green: Vec3,
    pub _pad15: f32,
    pub bg_into_fg_blue: Vec3,
    pub _pad16: f32,
    
    // Final mix
    pub final_mix_amount: f32,
    pub final_key_value: Vec3,
    pub final_key_threshold: f32,
    pub final_key_soft: f32,
    pub final_mix_type: i32,
    pub final_mix_overflow: i32,
    pub final_key_order: i32,
    pub _pad17: f32,
}

impl Default for Block3Uniforms {
    fn default() -> Self {
        Self {
            width: 1280.0,
            height: 720.0,
            inv_width: 1.0 / 1280.0,
            inv_height: 1.0 / 720.0,
            block1_xy_displace: [0.0, 0.0],
            block1_z_displace: 1.0,
            block1_rotate: 0.0,
            block1_shear_matrix: Vec4::new(1.0, 0.0, 0.0, 1.0),
            block1_kaleidoscope: 0.0,
            block1_kaleidoscope_slice: 0.0,
            block1_blur_amount: 0.0,
            block1_blur_radius: 1.0,
            block1_sharpen_amount: 0.0,
            block1_sharpen_radius: 1.0,
            block1_filters_boost: 0.0,
            block1_dither: 16.0,
            block1_switches: 0,
            block1_colorize_mode: 0,
            block1_dither_type: 0,
            _pad1: 0.0,
            block1_colorize_band1: Vec3::new(0.0, 0.0, 0.0),
            _pad2: 0.0,
            block1_colorize_band2: Vec3::new(0.0, 0.0, 0.0),
            _pad3: 0.0,
            block1_colorize_band3: Vec3::new(0.0, 0.0, 0.0),
            _pad4: 0.0,
            block1_colorize_band4: Vec3::new(0.0, 0.0, 0.0),
            _pad5: 0.0,
            block1_colorize_band5: Vec3::new(0.0, 0.0, 0.0),
            _pad6: 0.0,
            block2_xy_displace: [0.0, 0.0],
            block2_z_displace: 1.0,
            block2_rotate: 0.0,
            block2_shear_matrix: Vec4::new(1.0, 0.0, 0.0, 1.0),
            block2_kaleidoscope: 0.0,
            block2_kaleidoscope_slice: 0.0,
            block2_blur_amount: 0.0,
            block2_blur_radius: 1.0,
            block2_sharpen_amount: 0.0,
            block2_sharpen_radius: 1.0,
            block2_filters_boost: 0.0,
            block2_dither: 16.0,
            block2_switches: 0,
            block2_colorize_mode: 0,
            block2_dither_type: 0,
            _pad7: 0.0,
            block2_colorize_band1: Vec3::new(0.0, 0.0, 0.0),
            _pad8: 0.0,
            block2_colorize_band2: Vec3::new(0.0, 0.0, 0.0),
            _pad9: 0.0,
            block2_colorize_band3: Vec3::new(0.0, 0.0, 0.0),
            _pad10: 0.0,
            block2_colorize_band4: Vec3::new(0.0, 0.0, 0.0),
            _pad11: 0.0,
            block2_colorize_band5: Vec3::new(0.0, 0.0, 0.0),
            _pad12: 0.0,
            matrix_mix_type: 0,
            matrix_mix_overflow: 0,
            _pad13: [0.0; 2],
            bg_into_fg_red: Vec3::new(0.0, 0.0, 0.0),
            _pad14: 0.0,
            bg_into_fg_green: Vec3::new(0.0, 0.0, 0.0),
            _pad15: 0.0,
            bg_into_fg_blue: Vec3::new(0.0, 0.0, 0.0),
            _pad16: 0.0,
            final_mix_amount: 0.0,
            final_key_value: Vec3::new(0.0, 0.0, 0.0),
            final_key_threshold: 1.0,
            final_key_soft: 0.0,
            final_mix_type: 0,
            final_mix_overflow: 0,
            final_key_order: 0,
            _pad17: 0.0,
        }
    }
}

/// Block 3 render pipeline
pub struct Block3Pipeline {
    pub pipeline: wgpu::RenderPipeline,
    pub bind_group: wgpu::BindGroup,
    pub uniform_buffer: wgpu::Buffer,
    pub bind_group_layout: wgpu::BindGroupLayout,
    width: u32,
    height: u32,
}

impl Block3Pipeline {
    /// Create new Block3 pipeline
    pub fn new(device: &wgpu::Device, width: u32, height: u32, format: wgpu::TextureFormat) -> Self {
        let shader_code = format!(r#"
{}

@group(0) @binding(0)
var<uniform> uniforms: Block3Uniforms;

@group(0) @binding(1)
var block1_tex: texture_2d<f32>;
@group(0) @binding(2)
var block1_sampler: sampler;
@group(0) @binding(3)
var block2_tex: texture_2d<f32>;
@group(0) @binding(4)
var block2_sampler: sampler;

struct Block3Uniforms {{
    width: f32,
    height: f32,
    inv_width: f32,
    inv_height: f32,
    
    block1_xy_displace: vec2<f32>,
    block1_z_displace: f32,
    block1_rotate: f32,
    block1_shear_matrix: vec4<f32>,
    block1_kaleidoscope: f32,
    block1_kaleidoscope_slice: f32,
    block1_blur_amount: f32,
    block1_blur_radius: f32,
    block1_sharpen_amount: f32,
    block1_sharpen_radius: f32,
    block1_filters_boost: f32,
    block1_dither: f32,
    block1_switches: u32,
    block1_colorize_mode: i32,
    block1_dither_type: i32,
    _pad1: f32,
    
    block1_colorize_band1: vec3<f32>,
    _pad2: f32,
    block1_colorize_band2: vec3<f32>,
    _pad3: f32,
    block1_colorize_band3: vec3<f32>,
    _pad4: f32,
    block1_colorize_band4: vec3<f32>,
    _pad5: f32,
    block1_colorize_band5: vec3<f32>,
    _pad6: f32,
    
    block2_xy_displace: vec2<f32>,
    block2_z_displace: f32,
    block2_rotate: f32,
    block2_shear_matrix: vec4<f32>,
    block2_kaleidoscope: f32,
    block2_kaleidoscope_slice: f32,
    block2_blur_amount: f32,
    block2_blur_radius: f32,
    block2_sharpen_amount: f32,
    block2_sharpen_radius: f32,
    block2_filters_boost: f32,
    block2_dither: f32,
    block2_switches: u32,
    block2_colorize_mode: i32,
    block2_dither_type: i32,
    _pad7: f32,
    
    block2_colorize_band1: vec3<f32>,
    _pad8: f32,
    block2_colorize_band2: vec3<f32>,
    _pad9: f32,
    block2_colorize_band3: vec3<f32>,
    _pad10: f32,
    block2_colorize_band4: vec3<f32>,
    _pad11: f32,
    block2_colorize_band5: vec3<f32>,
    _pad12: f32,
    
    matrix_mix_type: i32,
    matrix_mix_overflow: i32,
    _pad13: vec2<f32>,
    bg_into_fg_red: vec3<f32>,
    _pad14: f32,
    bg_into_fg_green: vec3<f32>,
    _pad15: f32,
    bg_into_fg_blue: vec3<f32>,
    _pad16: f32,
    
    final_mix_amount: f32,
    final_key_value: vec3<f32>,
    final_key_threshold: f32,
    final_key_soft: f32,
    final_mix_type: i32,
    final_mix_overflow: i32,
    final_key_order: i32,
    _pad17: f32,
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

fn wrap01(v: f32) -> f32 {{
    return fract(abs(v));
}}

fn fold01(v: f32) -> f32 {{
    var result = v;
    if (result < 0.0) {{ result = abs(result); }}
    if (result > 1.0) {{ result = 1.0 - fract(result); }}
    if (result < 0.0) {{ result = abs(result); }}
    return result;
}}

// Matrix Mixer
fn matrix_mix(fg: vec3<f32>, bg: vec3<f32>) -> vec3<f32> {{
    var out_color = vec3<f32>(0.0);
    
    let fgR = vec3<f32>(fg.r, fg.r, fg.r);
    let fgG = vec3<f32>(fg.g, fg.g, fg.g);
    let fgB = vec3<f32>(fg.b, fg.b, fg.b);
    
    let scale = vec3<f32>(0.33, 0.33, 0.33);
    
    // lerp
    if (uniforms.matrix_mix_type == 0) {{
        out_color.r = dot(mix(fgR, bg, uniforms.bg_into_fg_red), scale);
        out_color.g = dot(mix(fgG, bg, uniforms.bg_into_fg_green), scale);
        out_color.b = dot(mix(fgB, bg, uniforms.bg_into_fg_blue), scale);
    }}
    // add
    else if (uniforms.matrix_mix_type == 1) {{
        out_color.r = dot(fgR + uniforms.bg_into_fg_red * bg, scale);
        out_color.g = dot(fgG + uniforms.bg_into_fg_green * bg, scale);
        out_color.b = dot(fgB + uniforms.bg_into_fg_blue * bg, scale);
    }}
    // diff
    else if (uniforms.matrix_mix_type == 2) {{
        out_color.r = dot(abs(fgR - uniforms.bg_into_fg_red * bg), scale);
        out_color.g = dot(abs(fgG - uniforms.bg_into_fg_green * bg), scale);
        out_color.b = dot(abs(fgB - uniforms.bg_into_fg_blue * bg), scale);
    }}
    // mult
    else if (uniforms.matrix_mix_type == 3) {{
        out_color.r = dot(mix(fgR, bg * fgR, uniforms.bg_into_fg_red), scale);
        out_color.g = dot(mix(fgG, bg * fgG, uniforms.bg_into_fg_green), scale);
        out_color.b = dot(mix(fgB, bg * fgB, uniforms.bg_into_fg_blue), scale);
    }}
    // dodge
    else if (uniforms.matrix_mix_type == 4) {{
        out_color.r = dot(mix(fgR, bg / (1.00001 - fgR), uniforms.bg_into_fg_red), scale);
        out_color.g = dot(mix(fgG, bg / (1.00001 - fgG), uniforms.bg_into_fg_green), scale);
        out_color.b = dot(mix(fgB, bg / (1.00001 - fgB), uniforms.bg_into_fg_blue), scale);
    }}
    
    // overflow handling
    if (uniforms.matrix_mix_overflow == 0) {{
        out_color = clamp(out_color, vec3<f32>(0.0), vec3<f32>(1.0));
    }} else if (uniforms.matrix_mix_overflow == 1) {{
        out_color = vec3<f32>(wrap01(out_color.x), wrap01(out_color.y), wrap01(out_color.z));
    }} else if (uniforms.matrix_mix_overflow == 2) {{
        out_color = vec3<f32>(fold01(out_color.x), fold01(out_color.y), fold01(out_color.z));
    }}
    
    return out_color;
}}

// Final mix and key
fn final_mix(fg: vec4<f32>, bg: vec4<f32>) -> vec4<f32> {{
    var out_color = fg;
    
    // Mix based on type
    if (uniforms.final_mix_type == 0) {{ // lerp
        out_color = mix(fg, bg, uniforms.final_mix_amount);
    }} else if (uniforms.final_mix_type == 1) {{ // add/sub
        out_color = vec4<f32>(fg.rgb + uniforms.final_mix_amount * bg.rgb, 1.0);
    }} else if (uniforms.final_mix_type == 2) {{ // diff
        out_color = vec4<f32>(abs(fg.rgb - uniforms.final_mix_amount * bg.rgb), 1.0);
    }} else if (uniforms.final_mix_type == 3) {{ // mult
        out_color = vec4<f32>(mix(fg.rgb, fg.rgb * bg.rgb, uniforms.final_mix_amount), 1.0);
    }} else if (uniforms.final_mix_type == 4) {{ // dodge
        out_color = vec4<f32>(mix(fg.rgb, fg.rgb / (1.00001 - bg.rgb), uniforms.final_mix_amount), 1.0);
    }}
    
    // overflow handling
    if (uniforms.final_mix_overflow == 0) {{
        out_color = vec4<f32>(clamp(out_color.rgb, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
    }} else if (uniforms.final_mix_overflow == 1) {{
        out_color = vec4<f32>(wrap01(out_color.r), wrap01(out_color.g), wrap01(out_color.b), 1.0);
    }} else if (uniforms.final_mix_overflow == 2) {{
        out_color = vec4<f32>(fold01(out_color.r), fold01(out_color.g), fold01(out_color.b), 1.0);
    }}
    
    // Keying
    let chroma_dist = distance(uniforms.final_key_value, fg.rgb);
    if (chroma_dist < uniforms.final_key_threshold) {{
        let mix_factor = uniforms.final_key_soft * abs(1.0 - (chroma_dist - uniforms.final_key_threshold));
        out_color = mix(bg, out_color, mix_factor);
    }}
    
    return out_color;
}}

@fragment
fn fs_main(@location(0) texcoord: vec2<f32>) -> @location(0) vec4<f32> {{
    let uv = vec2<f32>(texcoord.x, 1.0 - texcoord.y);
    
    // Sample inputs
    let block1_color = textureSample(block1_tex, block1_sampler, texcoord);
    let block2_color = textureSample(block2_tex, block2_sampler, texcoord);
    
    // Determine foreground/background based on final_key_order
    // final_key_order == 0: Block1 = FG, Block2 = BG (1 -> 2)
    // final_key_order == 1: Block2 = FG, Block1 = BG (2 -> 1)
    var fg = block1_color.rgb;
    var bg = block2_color.rgb;
    
    if (uniforms.final_key_order == 1) {{
        fg = block2_color.rgb;
        bg = block1_color.rgb;
    }}
    
    // Matrix Mixer
    var mixed = matrix_mix(fg, bg);
    
    // Final Mix with keying
    let final_color = final_mix(vec4<f32>(mixed, 1.0), vec4<f32>(bg, 1.0));
    
    return final_color;
}}
"#, COMMON_VERTEX_SHADER);
        
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Block3 Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_code.into()),
        });
        
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Block3 Bind Group Layout"),
            entries: &[
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
        
        let pipeline_layout = create_pipeline_layout(device, &bind_group_layout, "Block3 Pipeline Layout");
        
        let pipeline = create_render_pipeline(
            device,
            &pipeline_layout,
            &shader,
            &shader,
            format, // Output to surface (must match surface format)
            "Block3 Pipeline",
        );
        
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Block3 Uniform Buffer"),
            size: std::mem::size_of::<Block3Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Create dummy bind group
        let dummy_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Dummy Texture"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Bgra8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let dummy_view = dummy_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let dummy_sampler = device.create_sampler(&wgpu::SamplerDescriptor::default());
        
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Block3 Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&dummy_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&dummy_sampler) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&dummy_view) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&dummy_sampler) },
            ],
        });
        
        Self {
            pipeline,
            bind_group,
            uniform_buffer,
            bind_group_layout,
            width,
            height,
        }
    }
    
    /// Update uniform buffer from parameters
    pub fn update_params(&self, queue: &wgpu::Queue, params: &Block3Params) {
        let uniforms = Block3Uniforms {
            width: self.width as f32,
            height: self.height as f32,
            inv_width: 1.0 / self.width as f32,
            inv_height: 1.0 / self.height as f32,
            
            block1_xy_displace: [params.block1_x_displace, params.block1_y_displace],
            block1_z_displace: params.block1_z_displace,
            block1_rotate: params.block1_rotate,
            block1_shear_matrix: params.block1_shear_matrix,
            block1_blur_amount: params.block1_blur_amount,
            block1_sharpen_amount: params.block1_sharpen_amount,
            block1_switches: if params.block1_colorize_switch { 1 } else { 0 },
            block1_colorize_mode: params.block1_colorize_hsb_rgb,
            block1_colorize_band1: params.block1_colorize_band1.into(),
            block1_colorize_band2: params.block1_colorize_band2.into(),
            block1_colorize_band3: params.block1_colorize_band3.into(),
            block1_colorize_band4: params.block1_colorize_band4.into(),
            block1_colorize_band5: params.block1_colorize_band5.into(),
            
            block2_xy_displace: [params.block2_x_displace, params.block2_y_displace],
            block2_z_displace: params.block2_z_displace,
            block2_rotate: params.block2_rotate,
            block2_shear_matrix: params.block2_shear_matrix,
            block2_blur_amount: params.block2_blur_amount,
            block2_sharpen_amount: params.block2_sharpen_amount,
            block2_switches: if params.block2_colorize_switch { 1 } else { 0 },
            block2_colorize_mode: params.block2_colorize_hsb_rgb,
            block2_colorize_band1: params.block2_colorize_band1.into(),
            block2_colorize_band2: params.block2_colorize_band2.into(),
            block2_colorize_band3: params.block2_colorize_band3.into(),
            block2_colorize_band4: params.block2_colorize_band4.into(),
            block2_colorize_band5: params.block2_colorize_band5.into(),
            
            matrix_mix_type: params.matrix_mix_type,
            matrix_mix_overflow: params.matrix_mix_overflow,
            bg_into_fg_red: params.bg_rgb_into_fg_red.into(),
            bg_into_fg_green: params.bg_rgb_into_fg_green.into(),
            bg_into_fg_blue: params.bg_rgb_into_fg_blue.into(),
            
            final_mix_amount: params.final_mix_amount,
            final_key_threshold: params.final_key_threshold,
            final_key_soft: params.final_key_soft,
            final_mix_type: params.final_mix_type,
            final_mix_overflow: params.final_mix_overflow,
            final_key_order: params.final_key_order,
            
            ..Default::default()
        };
        
        let bytes = unsafe {
            std::slice::from_raw_parts(
                &uniforms as *const _ as *const u8,
                std::mem::size_of::<Block3Uniforms>()
            )
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytes);
    }
    
    /// Resize pipeline for new output dimensions
    pub fn resize(&mut self, _device: &wgpu::Device, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        // Pipeline doesn't need recreation, just update uniforms with new resolution
    }
    
    /// Update texture bindings for Block1 and Block2 inputs
    pub fn update_textures(
        &mut self,
        device: &wgpu::Device,
        block1_view: &wgpu::TextureView,
        block2_view: &wgpu::TextureView,
    ) {
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        
        self.bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Block3 Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.uniform_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(block1_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&sampler) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(block2_view) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&sampler) },
            ],
        });
    }
}
