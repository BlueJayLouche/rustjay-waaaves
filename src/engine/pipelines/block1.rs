//! # Block 1 Pipeline
//!
//! Channel mixing and feedback processing with WGSL shader.
//! Ported from BLUEJAY_WAAAVES shader1.frag

use crate::engine::pipelines::{create_pipeline_layout, create_render_pipeline};
use crate::params::Block1Params;
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

/// Uniforms for Block1 shader - matching OF app naming convention
#[repr(C, align(16))]
#[derive(Debug, Copy, Clone, bytemuck::AnyBitPattern)]
pub struct Block1Uniforms {
    // Resolution (16 bytes)
    pub width: f32,
    pub height: f32,
    pub inv_width: f32,
    pub inv_height: f32,
    
    // Input texture dimensions (16 bytes)
    pub ch1_input_width: f32,
    pub ch1_input_height: f32,
    pub ch2_input_width: f32,
    pub ch2_input_height: f32,
    
    // Channel 1 (112 bytes)
    pub ch1_aspect: f32,
    pub ch1_crib_x: f32,
    pub ch1_scale: f32,
    pub ch1_hd_zcrib: f32,
    pub ch1_xy_displace: [f32; 2],
    pub ch1_z_displace: f32,
    pub ch1_rotate: f32,
    pub ch1_hsb_attenuate: Vec3,
    pub ch1_posterize: f32,
    pub ch1_posterize_inv: f32,
    pub ch1_kaleidoscope: f32,
    pub ch1_kaleidoscope_slice: f32,
    pub ch1_blur_amount: f32,
    pub ch1_blur_radius: f32,
    pub ch1_sharpen_amount: f32,
    pub ch1_sharpen_radius: f32,
    pub ch1_filters_boost: f32,
    pub ch1_switches: u32,
    pub ch1_geo_overflow: i32,
    pub ch1_hd_aspect_on: i32,
    pub _pad1: f32,
    
    // Channel 2 mix (48 bytes)
    pub ch2_mix_amount: f32,
    pub ch2_key_value: Vec3,
    pub ch2_key_threshold: f32,
    pub ch2_key_soft: f32,
    pub ch2_mix_type: i32,
    pub ch2_mix_overflow: i32,
    pub ch2_key_order: i32,
    pub ch2_key_mode: i32,
    
    // Channel 2 adjust (112 bytes)
    pub ch2_aspect: f32,
    pub ch2_crib_x: f32,
    pub ch2_scale: f32,
    pub ch2_hd_zcrib: f32,
    pub ch2_xy_displace: [f32; 2],
    pub ch2_z_displace: f32,
    pub ch2_rotate: f32,
    pub ch2_hsb_attenuate: Vec3,
    pub ch2_posterize: f32,
    pub ch2_posterize_inv: f32,
    pub ch2_kaleidoscope: f32,
    pub ch2_kaleidoscope_slice: f32,
    pub ch2_blur_amount: f32,
    pub ch2_blur_radius: f32,
    pub ch2_sharpen_amount: f32,
    pub ch2_sharpen_radius: f32,
    pub ch2_filters_boost: f32,
    pub ch2_switches: u32,
    pub ch2_geo_overflow: i32,
    pub ch2_hd_aspect_on: i32,
    pub _pad3: f32,
    
    // FB1 mix (48 bytes)
    pub fb1_mix_amount: f32,
    pub fb1_key_value: Vec3,
    pub fb1_key_threshold: f32,
    pub fb1_key_soft: f32,
    pub fb1_mix_type: i32,
    pub fb1_mix_overflow: i32,
    pub fb1_key_order: i32,
    pub _pad4: i32,
    
    // FB1 geometry (64 bytes)
    pub fb1_xy_displace: [f32; 2],
    pub fb1_z_displace: f32,
    pub fb1_rotate: f32,
    pub fb1_shear_matrix: Vec4,
    pub fb1_kaleidoscope: f32,
    pub fb1_kaleidoscope_slice: f32,
    
    // FB1 color (64 bytes)
    pub fb1_hsb_offset: Vec3,
    pub fb1_hue_shaper: f32,
    pub fb1_hsb_attenuate: Vec3,
    pub fb1_hsb_powmap: Vec3,
    pub fb1_posterize: f32,
    pub fb1_posterize_inv: f32,
    
    // FB1 filters (48 bytes)
    pub fb1_blur_amount: f32,
    pub fb1_blur_radius: f32,
    pub fb1_sharpen_amount: f32,
    pub fb1_sharpen_radius: f32,
    pub fb1_temporal1_amount: f32,
    pub fb1_temporal1_res: f32,
    pub fb1_temporal2_amount: f32,
    pub fb1_temporal2_res: f32,
    pub fb1_filters_boost: f32,
    pub fb1_switches: u32,
    pub fb1_rotate_mode: i32,
    pub fb1_geo_overflow: i32,
    pub _pad5: f32,
    
    // Input selection (16 bytes)
    pub ch1_input_select: i32,
    pub ch2_input_select: i32,
    pub _pad6: [f32; 2],
}

impl Default for Block1Uniforms {
    fn default() -> Self {
        Self {
            width: 1280.0,
            height: 720.0,
            inv_width: 1.0 / 1280.0,
            inv_height: 1.0 / 720.0,
            ch1_input_width: 1280.0,
            ch1_input_height: 720.0,
            ch2_input_width: 1280.0,
            ch2_input_height: 720.0,
            ch1_aspect: 1.0,
            ch1_crib_x: 0.0,
            ch1_scale: 1.0,
            ch1_hd_zcrib: 0.0,
            ch1_xy_displace: [0.0, 0.0],
            ch1_z_displace: 1.0,
            ch1_rotate: 0.0,
            ch1_hsb_attenuate: Vec3::new(1.0, 1.0, 1.0),
            ch1_posterize: 16.0,
            ch1_posterize_inv: 1.0 / 16.0,
            ch1_kaleidoscope: 0.0,
            ch1_kaleidoscope_slice: 0.0,
            ch1_blur_amount: 0.0,
            ch1_blur_radius: 1.0,
            ch1_sharpen_amount: 0.0,
            ch1_sharpen_radius: 1.0,
            ch1_filters_boost: 0.0,
            ch1_switches: 0,
            ch1_geo_overflow: 0,
            ch1_hd_aspect_on: 0,
            _pad1: 0.0,
            
            ch2_mix_amount: 0.0,
            ch2_key_value: Vec3::new(0.0, 0.0, 0.0),
            ch2_key_threshold: 1.0,
            ch2_key_soft: 0.0,
            ch2_mix_type: 0,
            ch2_mix_overflow: 0,
            ch2_key_order: 0,
            ch2_key_mode: 0,
            
            ch2_aspect: 1.0,
            ch2_crib_x: 0.0,
            ch2_scale: 1.0,
            ch2_hd_zcrib: 0.0,
            ch2_xy_displace: [0.0, 0.0],
            ch2_z_displace: 1.0,
            ch2_rotate: 0.0,
            ch2_hsb_attenuate: Vec3::new(1.0, 1.0, 1.0),
            ch2_posterize: 16.0,
            ch2_posterize_inv: 1.0 / 16.0,
            ch2_kaleidoscope: 0.0,
            ch2_kaleidoscope_slice: 0.0,
            ch2_blur_amount: 0.0,
            ch2_blur_radius: 1.0,
            ch2_sharpen_amount: 0.0,
            ch2_sharpen_radius: 1.0,
            ch2_filters_boost: 0.0,
            ch2_switches: 0,
            ch2_geo_overflow: 0,
            ch2_hd_aspect_on: 0,
            _pad3: 0.0,
            
            fb1_mix_amount: 0.0,
            fb1_key_value: Vec3::new(0.0, 0.0, 0.0),
            fb1_key_threshold: 1.0,
            fb1_key_soft: 0.0,
            fb1_mix_type: 0,
            fb1_mix_overflow: 0,
            fb1_key_order: 0,
            _pad4: 0,
            
            fb1_xy_displace: [0.0, 0.0],
            fb1_z_displace: 1.0,
            fb1_rotate: 0.0,
            fb1_shear_matrix: Vec4::new(1.0, 0.0, 0.0, 1.0),
            fb1_kaleidoscope: 0.0,
            fb1_kaleidoscope_slice: 0.0,
            fb1_hsb_offset: Vec3::new(0.0, 0.0, 0.0),
            fb1_hue_shaper: 1.0,
            fb1_hsb_attenuate: Vec3::new(1.0, 1.0, 1.0),
            fb1_hsb_powmap: Vec3::new(1.0, 1.0, 1.0),
            fb1_posterize: 16.0,
            fb1_posterize_inv: 1.0 / 16.0,
            
            fb1_blur_amount: 0.0,
            fb1_blur_radius: 1.0,
            fb1_sharpen_amount: 0.0,
            fb1_sharpen_radius: 1.0,
            fb1_temporal1_amount: 0.0,
            fb1_temporal1_res: 0.0,
            fb1_temporal2_amount: 0.0,
            fb1_temporal2_res: 0.0,
            fb1_filters_boost: 0.0,
            fb1_switches: 0,
            fb1_rotate_mode: 0,
            fb1_geo_overflow: 0,
            _pad5: 0.0,
            
            ch1_input_select: 0,
            ch2_input_select: 1,
            _pad6: [0.0, 0.0],
        }
    }
}

/// Block 1 render pipeline
pub struct Block1Pipeline {
    pub pipeline: wgpu::RenderPipeline,
    pub bind_group: wgpu::BindGroup,
    pub uniform_buffer: wgpu::Buffer,
    pub bind_group_layout: wgpu::BindGroupLayout,
    width: u32,
    height: u32,
}

impl Block1Pipeline {
    /// Create new Block1 pipeline
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let shader_code = format!(r#"
// Block 1 Shader - ported from BLUEJAY_WAAAVES shader1.frag

struct VertexInput {{
    @location(0) position: vec2<f32>,
    @location(1) texcoord: vec2<f32>,
}}

struct VertexOutput {{
    @builtin(position) position: vec4<f32>,
    @location(0) texcoord: vec2<f32>,
}}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {{
    var output: VertexOutput;
    output.position = vec4<f32>(input.position, 0.0, 1.0);
    output.texcoord = input.texcoord;
    return output;
}}

@group(0) @binding(0)
var<uniform> uniforms: Block1Uniforms;

// Input textures
@group(0) @binding(1)
var ch1_tex: texture_2d<f32>;
@group(0) @binding(2)
var ch1_sampler: sampler;
@group(0) @binding(3)
var ch2_tex: texture_2d<f32>;
@group(0) @binding(4)
var ch2_sampler: sampler;

// Feedback textures
@group(0) @binding(5)
var fb1_tex: texture_2d<f32>;
@group(0) @binding(6)
var fb1_sampler: sampler;
@group(0) @binding(7)
var temporal_tex: texture_2d<f32>;
@group(0) @binding(8)
var temporal_sampler: sampler;

struct Block1Uniforms {{
    width: f32,
    height: f32,
    inv_width: f32,
    inv_height: f32,
    
    // Input texture dimensions for proper UV scaling
    ch1_input_width: f32,
    ch1_input_height: f32,
    ch2_input_width: f32,
    ch2_input_height: f32,
    
    ch1_aspect: f32,
    ch1_crib_x: f32,
    ch1_scale: f32,
    ch1_hd_zcrib: f32,
    ch1_xy_displace: vec2<f32>,
    ch1_z_displace: f32,
    ch1_rotate: f32,
    ch1_hsb_attenuate: vec3<f32>,
    ch1_posterize: f32,
    ch1_posterize_inv: f32,
    ch1_kaleidoscope: f32,
    ch1_kaleidoscope_slice: f32,
    ch1_blur_amount: f32,
    ch1_blur_radius: f32,
    ch1_sharpen_amount: f32,
    ch1_sharpen_radius: f32,
    ch1_filters_boost: f32,
    ch1_switches: u32,
    ch1_geo_overflow: i32,
    ch1_hd_aspect_on: i32,
    _pad1: f32,
    
    ch2_mix_amount: f32,
    ch2_key_value: vec3<f32>,
    ch2_key_threshold: f32,
    ch2_key_soft: f32,
    ch2_mix_type: i32,
    ch2_mix_overflow: i32,
    ch2_key_order: i32,
    ch2_key_mode: i32,
    
    ch2_aspect: f32,
    ch2_crib_x: f32,
    ch2_scale: f32,
    ch2_hd_zcrib: f32,
    ch2_xy_displace: vec2<f32>,
    ch2_z_displace: f32,
    ch2_rotate: f32,
    ch2_hsb_attenuate: vec3<f32>,
    ch2_posterize: f32,
    ch2_posterize_inv: f32,
    ch2_kaleidoscope: f32,
    ch2_kaleidoscope_slice: f32,
    ch2_blur_amount: f32,
    ch2_blur_radius: f32,
    ch2_sharpen_amount: f32,
    ch2_sharpen_radius: f32,
    ch2_filters_boost: f32,
    ch2_switches: u32,
    ch2_geo_overflow: i32,
    ch2_hd_aspect_on: i32,
    _pad3: f32,
    
    fb1_mix_amount: f32,
    fb1_key_value: vec3<f32>,
    fb1_key_threshold: f32,
    fb1_key_soft: f32,
    fb1_mix_type: i32,
    fb1_mix_overflow: i32,
    fb1_key_order: i32,
    _pad4: i32,
    
    fb1_xy_displace: vec2<f32>,
    fb1_z_displace: f32,
    fb1_rotate: f32,
    fb1_shear_matrix: vec4<f32>,
    fb1_kaleidoscope: f32,
    fb1_kaleidoscope_slice: f32,
    fb1_hsb_offset: vec3<f32>,
    fb1_hue_shaper: f32,
    fb1_hsb_attenuate: vec3<f32>,
    fb1_hsb_powmap: vec3<f32>,
    fb1_posterize: f32,
    fb1_posterize_inv: f32,
    fb1_blur_amount: f32,
    fb1_blur_radius: f32,
    fb1_sharpen_amount: f32,
    fb1_sharpen_radius: f32,
    fb1_temporal1_amount: f32,
    fb1_temporal1_res: f32,
    fb1_temporal2_amount: f32,
    fb1_temporal2_res: f32,
    fb1_filters_boost: f32,
    fb1_switches: u32,
    fb1_rotate_mode: i32,
    fb1_geo_overflow: i32,
    _pad5: f32,
    
    ch1_input_select: i32,
    ch2_input_select: i32,
    _pad6: vec2<f32>,
}}

const PI: f32 = 3.1415926535;
const TWO_PI: f32 = 6.2831855;

// Switch bit extraction
fn get_switch(switches: u32, bit: u32) -> bool {{
    return (switches & (1u << bit)) != 0u;
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

// Apply hue shaping
fn apply_hue_shaper(in_hue: f32, shaper: f32) -> f32 {{
    return fract(abs(in_hue + shaper * sin(in_hue * 0.3184713)));
}}

// Color quantization
fn color_quantize(in_color: vec3<f32>, amount: f32, amount_inv: f32) -> vec3<f32> {{
    var c = in_color * amount;
    c = floor(c);
    return c * amount_inv;
}}

// Solarize effect
fn apply_solarize(in_bright: f32) -> f32 {{
    if (in_bright > 0.5) {{
        return 1.0 - in_bright;
    }}
    return in_bright;
}}

// Rotate coordinates
fn do_rotate(coord: vec2<f32>, angle: f32) -> vec2<f32> {{
    let center = vec2<f32>(uniforms.width * 0.5, uniforms.height * 0.5);
    let centered = coord - center;
    let rotated_x = centered.x * cos(angle) - centered.y * sin(angle);
    let rotated_y = centered.x * sin(angle) + centered.y * cos(angle);
    return vec2<f32>(rotated_x + center.x, rotated_y + center.y);
}}

// Kaleidoscope effect
fn do_kaleidoscope(coord: vec2<f32>, segments: f32, slice: f32) -> vec2<f32> {{
    if (segments <= 0.0) {{
        return coord;
    }}
    
    var result = do_rotate(coord, slice);
    let norm = result / vec2<f32>(uniforms.width, uniforms.height);
    let centered = norm * 2.0 - 1.0;
    let radius = length(centered);
    var angle = atan2(centered.y, centered.x);
    let segment_angle = TWO_PI / segments;
    angle = angle - segment_angle * floor(angle / segment_angle);
    angle = min(angle, segment_angle - angle);
    result = radius * vec2<f32>(cos(angle), sin(angle));
    result = (result * 0.5 + 0.5) * vec2<f32>(uniforms.width, uniforms.height);
    return do_rotate(result, -slice);
}}

// Wrap coordinates
fn wrap_coord(coord: vec2<f32>) -> vec2<f32> {{
    return vec2<f32>(
        coord.x % uniforms.width,
        coord.y % uniforms.height
    );
}}

// Mirror function
fn mirror_val(a: f32) -> f32 {{
    if (a > 0.0) {{
        return a;
    }}
    return -(1.0 + a);
}}

// Mirror coordinates
fn mirror_coord(coord: vec2<f32>) -> vec2<f32> {{
    let w = uniforms.width - 1.0;
    let h = uniforms.height - 1.0;
    return vec2<f32>(
        w - mirror_val(coord.x % (2.0 * w) - w),
        h - mirror_val(coord.y % (2.0 * h) - h)
    );
}}

// Blur and sharpen function
fn blur_and_sharpen(tex: texture_2d<f32>, tex_sampler: sampler, coord: vec2<f32>, 
                    sharpen_amount: f32, sharpen_radius: f32, sharpen_boost: f32,
                    blur_radius: f32, blur_amount: f32) -> vec4<f32> {{
    let original_color = textureSample(tex, tex_sampler, coord);
    
    // Early exit if no filters
    if (blur_amount < 0.001 && sharpen_amount < 0.001) {{
        return original_color;
    }}
    
    let tex_size = vec2<f32>(uniforms.width, uniforms.height);
    let blur_size = vec2<f32>(blur_radius) / (tex_size - 1.0);
    let sharpen_size = vec2<f32>(sharpen_radius) / (tex_size - 1.0);
    
    // Box blur (8 samples)
    var color_blur = original_color;
    if (blur_amount >= 0.001) {{
        color_blur = textureSample(tex, tex_sampler, coord + blur_size * vec2<f32>( 1.0, 1.0))
                   + textureSample(tex, tex_sampler, coord + blur_size * vec2<f32>( 0.0, 1.0))
                   + textureSample(tex, tex_sampler, coord + blur_size * vec2<f32>(-1.0, 1.0))
                   + textureSample(tex, tex_sampler, coord + blur_size * vec2<f32>(-1.0, 0.0))
                   + textureSample(tex, tex_sampler, coord + blur_size * vec2<f32>(-1.0, -1.0))
                   + textureSample(tex, tex_sampler, coord + blur_size * vec2<f32>( 0.0, -1.0))
                   + textureSample(tex, tex_sampler, coord + blur_size * vec2<f32>( 1.0, -1.0))
                   + textureSample(tex, tex_sampler, coord + blur_size * vec2<f32>( 1.0, 0.0));
        color_blur *= 0.125;
        color_blur = mix(original_color, color_blur, blur_amount);
    }}
    
    // Sharpen
    var color_blur_hsb = rgb2hsb(color_blur.rgb);
    if (sharpen_amount >= 0.001) {{
        let lum_weights = vec3<f32>(0.299, 0.587, 0.114);
        var color_sharpen_bright = 
            dot(textureSample(tex, tex_sampler, coord + sharpen_size * vec2<f32>( 1.0, 0.0)).rgb, lum_weights) +
            dot(textureSample(tex, tex_sampler, coord + sharpen_size * vec2<f32>(-1.0, 0.0)).rgb, lum_weights) +
            dot(textureSample(tex, tex_sampler, coord + sharpen_size * vec2<f32>( 0.0, 1.0)).rgb, lum_weights) +
            dot(textureSample(tex, tex_sampler, coord + sharpen_size * vec2<f32>( 0.0, -1.0)).rgb, lum_weights) +
            dot(textureSample(tex, tex_sampler, coord + sharpen_size * vec2<f32>( 1.0, 1.0)).rgb, lum_weights) +
            dot(textureSample(tex, tex_sampler, coord + sharpen_size * vec2<f32>(-1.0, 1.0)).rgb, lum_weights) +
            dot(textureSample(tex, tex_sampler, coord + sharpen_size * vec2<f32>( 1.0, -1.0)).rgb, lum_weights) +
            dot(textureSample(tex, tex_sampler, coord + sharpen_size * vec2<f32>(-1.0, -1.0)).rgb, lum_weights);
        color_sharpen_bright *= 0.125;
        color_blur_hsb.z -= sharpen_amount * color_sharpen_bright;
    }}
    
    // Boost
    let boost_factor = mix(1.0, 1.0 + sharpen_amount + sharpen_boost, step(0.001, sharpen_amount));
    color_blur_hsb.z *= boost_factor;
    
    // Clamp HSB values before converting back to RGB
    color_blur_hsb.x = fract(color_blur_hsb.x);
    color_blur_hsb.y = clamp(color_blur_hsb.y, 0.0, 1.0);
    color_blur_hsb.z = clamp(color_blur_hsb.z, 0.0, 1.0);
    
    return vec4<f32>(hsb2rgb(color_blur_hsb), 1.0);
}}

// Shear transformation
fn shear_coord(coord: vec2<f32>, shear_matrix: vec4<f32>) -> vec2<f32> {{
    let center = vec2<f32>(uniforms.width * 0.5, uniforms.height * 0.5);
    var result = coord - center;
    result.x = shear_matrix.x * result.x + shear_matrix.y * result.y;
    result.y = shear_matrix.z * result.x + shear_matrix.w * result.y;
    return result + center;
}}

// Wrap value 0-1
fn wrap01(v: f32) -> f32 {{
    if (v < 0.0) {{
        return 1.0 - abs(v);
    }}
    if (v > 1.0) {{
        return fract(v);
    }}
    return v;
}}

// Foldover
fn fold01(v: f32) -> f32 {{
    var r = v;
    if (r < 0.0) {{
        r = abs(r);
    }}
    if (r > 1.0) {{
        r = 1.0 - fract(r);
    }}
    if (r < 0.0) {{
        r = abs(r);
    }}
    return r;
}}

// Calculate key mix amount based on chroma distance
fn calculate_key_mix(color: vec3<f32>, key_value: vec3<f32>, threshold: f32, softness: f32) -> f32 {{
    if (threshold < 0.001) {{
        return 0.0;
    }}
    let chroma_distance = distance(key_value, color);
    if (chroma_distance < threshold) {{
        // Key amount increases as we get closer to the key color
        return smoothstep(threshold, threshold * (1.0 - softness), chroma_distance);
    }}
    return 0.0;
}}

// Mix two colors with a specific blend mode and overflow handling
fn mix_with_mode(fg: vec4<f32>, bg: vec4<f32>, amount: f32, mix_type: i32, mix_overflow: i32) -> vec4<f32> {{
    var out_rgb: vec3<f32>;
    
    // Mix modes
    switch(mix_type) {{
        case 0: {{ // lerp
            out_rgb = mix(fg.rgb, bg.rgb, amount);
        }}
        case 1: {{ // add
            out_rgb = fg.rgb + amount * bg.rgb;
        }}
        case 2: {{ // diff
            out_rgb = abs(fg.rgb - amount * bg.rgb);
        }}
        case 3: {{ // mult
            out_rgb = mix(fg.rgb, fg.rgb * bg.rgb, amount);
        }}
        case 4: {{ // dodge
            out_rgb = mix(fg.rgb, fg.rgb / (1.00001 - bg.rgb), amount);
        }}
        default: {{
            out_rgb = mix(fg.rgb, bg.rgb, amount);
        }}
    }}
    
    // Overflow modes
    switch(mix_overflow) {{
        case 0: {{ // clamp
            out_rgb = clamp(out_rgb, vec3<f32>(0.0), vec3<f32>(1.0));
        }}
        case 1: {{ // wrap
            out_rgb = vec3<f32>(wrap01(out_rgb.x), wrap01(out_rgb.y), wrap01(out_rgb.z));
        }}
        case 2: {{ // fold
            out_rgb = vec3<f32>(fold01(out_rgb.x), fold01(out_rgb.y), fold01(out_rgb.z));
        }}
        default: {{
            out_rgb = clamp(out_rgb, vec3<f32>(0.0), vec3<f32>(1.0));
        }}
    }}
    
    return vec4<f32>(out_rgb, 1.0);
}}

// Mix and key function with OF-style integrated keying
// key_order: 0=Key First Then Mix, 1=Mix First Then Key
// mix_type: 0=lerp, 1=add, 2=diff, 3=mult, 4=dodge
fn mix_and_key(fg: vec4<f32>, bg: vec4<f32>, amount: f32, mix_type: i32, 
               key_threshold: f32, key_soft: f32, key_value: vec3<f32>,
               key_order: i32, mix_overflow: i32) -> vec4<f32> {{
    
    var out_color: vec4<f32>;
    
    if (key_order == 0) {{
        // Key First Then Mix: Key the foreground, then mix with background
        let key_amount = calculate_key_mix(fg.rgb, key_value, key_threshold, key_soft);
        let keyed_fg = mix(fg, bg, key_amount);
        out_color = mix_with_mode(keyed_fg, bg, amount, mix_type, mix_overflow);
    }} else {{
        // Mix First Then Key: Mix first, then key the result
        out_color = mix_with_mode(fg, bg, amount, mix_type, mix_overflow);
        let key_amount = calculate_key_mix(out_color.rgb, key_value, key_threshold, key_soft);
        out_color = mix(out_color, bg, key_amount);
    }}
    
    return out_color;
}}

// Process channel
fn process_channel(uv: vec2<f32>, coords: vec2<f32>, 
                   tex: texture_2d<f32>, tex_sampler: sampler,
                   input_width: f32, input_height: f32,
                   aspect: f32, crib_x: f32, scale: f32, hd_zcrib: f32,
                   xy_displace: vec2<f32>, z_displace: f32, rotate: f32,
                   hsb_attenuate: vec3<f32>, posterize: f32, posterize_inv: f32,
                   kaleidoscope: f32, kaleidoscope_slice: f32,
                   blur_amount: f32, blur_radius: f32, 
                   sharpen_amount: f32, sharpen_radius: f32, filters_boost: f32,
                   switches: u32, geo_overflow: i32, hd_aspect_on: i32) -> vec4<f32> {{
    
    var ch_coords = coords;
    
    // Apply aspect ratio
    ch_coords.x *= aspect;
    ch_coords.x -= crib_x;
    
    // Scale around center
    ch_coords -= vec2<f32>(uniforms.width * 0.5, uniforms.height * 0.5);
    ch_coords *= scale + hd_zcrib;
    ch_coords += vec2<f32>(uniforms.width * 0.5, uniforms.height * 0.5);
    
    // HD aspect fix
    if (hd_aspect_on == 1) {{
        ch_coords = uv * vec2<f32>(uniforms.width, uniforms.height);
    }}
    
    // H/V Flip
    if (get_switch(switches, 2u)) {{ // h_flip
        ch_coords.x = uniforms.width - ch_coords.x;
    }}
    if (get_switch(switches, 3u)) {{ // v_flip
        ch_coords.y = uniforms.height - ch_coords.y;
    }}
    
    // H/V Mirror
    if (get_switch(switches, 0u)) {{ // h_mirror
        if (ch_coords.x > uniforms.width * 0.5) {{
            ch_coords.x = abs(uniforms.width - ch_coords.x);
        }}
    }}
    if (get_switch(switches, 1u)) {{ // v_mirror
        if (ch_coords.y > uniforms.height * 0.5) {{
            ch_coords.y = abs(uniforms.height - ch_coords.y);
        }}
    }}
    
    // Kaleidoscope
    ch_coords = do_kaleidoscope(ch_coords, kaleidoscope, kaleidoscope_slice);
    
    // Displace
    ch_coords += xy_displace;
    
    // Z displace (zoom)
    ch_coords -= vec2<f32>(uniforms.width * 0.5, uniforms.height * 0.5);
    ch_coords *= z_displace;
    ch_coords += vec2<f32>(uniforms.width * 0.5, uniforms.height * 0.5);
    
    // Rotate
    ch_coords = do_rotate(ch_coords, rotate);
    
    // Geo overflow
    if (geo_overflow == 1) {{
        ch_coords = wrap_coord(ch_coords);
    }} else if (geo_overflow == 2) {{
        ch_coords = mirror_coord(ch_coords);
    }}
    
    // Sample with blur/sharpen
    // For video input, simply stretch to fit the output (no special scaling)
    let ch_uv = ch_coords / vec2<f32>(uniforms.width, uniforms.height);
    var ch_color = blur_and_sharpen(tex, tex_sampler, ch_uv,
                                     sharpen_amount, sharpen_radius, filters_boost,
                                     blur_radius, blur_amount);
    
    // Clamp if no overflow
    if (geo_overflow == 0) {{
        if (ch_coords.x > uniforms.width || ch_coords.y > uniforms.height || 
            ch_coords.x < 0.0 || ch_coords.y < 0.0) {{
            ch_color = vec4<f32>(0.0);
        }}
    }}
    
    // HSB processing with early exit optimization
    // Skip HSB conversion if no HSB operations needed
    let needs_hsb = hsb_attenuate.x != 1.0 || hsb_attenuate.y != 1.0 || hsb_attenuate.z != 1.0 ||
                    get_switch(switches, 4u) || get_switch(switches, 5u) || get_switch(switches, 6u) ||
                    get_switch(switches, 8u); // solarize
    
    var ch_rgb = ch_color.rgb;
    if (needs_hsb) {{
        var ch_hsb = rgb2hsb(ch_color.rgb);
        ch_hsb = pow(ch_hsb, hsb_attenuate);
        
        // Inverts
        if (get_switch(switches, 4u)) {{ // hue_invert
            ch_hsb.x = 1.0 - ch_hsb.x;
        }}
        if (get_switch(switches, 5u)) {{ // sat_invert
            ch_hsb.y = 1.0 - ch_hsb.y;
        }}
        if (get_switch(switches, 6u)) {{ // bright_invert
            ch_hsb.z = 1.0 - ch_hsb.z;
        }}
        
        ch_hsb.x = fract(ch_hsb.x);
        
        // Solarize
        if (get_switch(switches, 8u)) {{ // solarize
            ch_hsb.z = apply_solarize(ch_hsb.z);
        }}
        
        // Clamp saturation and brightness before converting back to RGB
        ch_hsb.y = clamp(ch_hsb.y, 0.0, 1.0);
        ch_hsb.z = clamp(ch_hsb.z, 0.0, 1.0);
        
        ch_rgb = hsb2rgb(ch_hsb);
    }}
    
    // RGB invert
    if (get_switch(switches, 7u)) {{ // rgb_invert
        ch_rgb = 1.0 - ch_rgb;
    }}
    
    // Posterize
    if (get_switch(switches, 9u)) {{ // posterize_switch
        ch_rgb = color_quantize(ch_rgb, posterize, posterize_inv);
    }}
    
    ch_color = vec4<f32>(ch_rgb, ch_color.a);
    
    return ch_color;
}}

@fragment
fn fs_main(@location(0) texcoord: vec2<f32>) -> @location(0) vec4<f32> {{
    // wgpu uses top-left as origin, but our vertex shader passes flipped V coordinates
    // Remap texcoord to account for this
    let uv = vec2<f32>(texcoord.x, 1.0 - texcoord.y);
    
    let coords = uv * vec2<f32>(uniforms.width, uniforms.height);
    
    // === CHANNEL 1 Processing ===
    let ch1_color = process_channel(
        uv, coords, ch1_tex, ch1_sampler,
        uniforms.ch1_input_width, uniforms.ch1_input_height,
        uniforms.ch1_aspect, uniforms.ch1_crib_x, uniforms.ch1_scale, uniforms.ch1_hd_zcrib,
        uniforms.ch1_xy_displace, uniforms.ch1_z_displace, uniforms.ch1_rotate,
        uniforms.ch1_hsb_attenuate, uniforms.ch1_posterize, uniforms.ch1_posterize_inv,
        uniforms.ch1_kaleidoscope, uniforms.ch1_kaleidoscope_slice,
        uniforms.ch1_blur_amount, uniforms.ch1_blur_radius,
        uniforms.ch1_sharpen_amount, uniforms.ch1_sharpen_radius, uniforms.ch1_filters_boost,
        uniforms.ch1_switches, uniforms.ch1_geo_overflow, uniforms.ch1_hd_aspect_on
    );
    
    var ch1_final_color = ch1_color;
    
    // === CHANNEL 2 Processing ===
    let ch2_color = process_channel(
        uv, coords, ch2_tex, ch2_sampler,
        uniforms.ch2_input_width, uniforms.ch2_input_height,
        uniforms.ch2_aspect, uniforms.ch2_crib_x, uniforms.ch2_scale, uniforms.ch2_hd_zcrib,
        uniforms.ch2_xy_displace, uniforms.ch2_z_displace, uniforms.ch2_rotate,
        uniforms.ch2_hsb_attenuate, uniforms.ch2_posterize, uniforms.ch2_posterize_inv,
        uniforms.ch2_kaleidoscope, uniforms.ch2_kaleidoscope_slice,
        uniforms.ch2_blur_amount, uniforms.ch2_blur_radius,
        uniforms.ch2_sharpen_amount, uniforms.ch2_sharpen_radius, uniforms.ch2_filters_boost,
        uniforms.ch2_switches, uniforms.ch2_geo_overflow, uniforms.ch2_hd_aspect_on
    );
    
    // === Mix CH1 and CH2 ===
    var mixed_color = mix_and_key(
        ch1_final_color, ch2_color, uniforms.ch2_mix_amount, uniforms.ch2_mix_type,
        uniforms.ch2_key_threshold, uniforms.ch2_key_soft, uniforms.ch2_key_value,
        uniforms.ch2_key_order, uniforms.ch2_mix_overflow
    );
    
    // === FB1 Processing ===
    var fb1_coords = coords;
    
    // FB1 H/V Flip
    if (get_switch(uniforms.fb1_switches, 2u)) {{ // fb1_h_flip
        fb1_coords.x = uniforms.width - fb1_coords.x;
    }}
    if (get_switch(uniforms.fb1_switches, 3u)) {{ // fb1_v_flip
        fb1_coords.y = uniforms.height - fb1_coords.y;
    }}
    
    // FB1 H/V Mirror
    if (get_switch(uniforms.fb1_switches, 0u)) {{ // fb1_h_mirror
        if (fb1_coords.x > uniforms.width * 0.5) {{
            fb1_coords.x = abs(uniforms.width - fb1_coords.x);
        }}
    }}
    if (get_switch(uniforms.fb1_switches, 1u)) {{ // fb1_v_mirror
        if (fb1_coords.y > uniforms.height * 0.5) {{
            fb1_coords.y = abs(uniforms.height - fb1_coords.y);
        }}
    }}
    
    // FB1 Kaleidoscope
    fb1_coords = do_kaleidoscope(fb1_coords, uniforms.fb1_kaleidoscope, uniforms.fb1_kaleidoscope_slice);
    
    // FB1 Displace
    fb1_coords += uniforms.fb1_xy_displace;
    
    // FB1 Z displace
    fb1_coords -= vec2<f32>(uniforms.width * 0.5, uniforms.height * 0.5);
    fb1_coords *= uniforms.fb1_z_displace;
    fb1_coords += vec2<f32>(uniforms.width * 0.5, uniforms.height * 0.5);
    
    // FB1 Rotate
    fb1_coords = do_rotate(fb1_coords, uniforms.fb1_rotate);
    
    // FB1 Shear
    fb1_coords = shear_coord(fb1_coords, uniforms.fb1_shear_matrix);
    
    // FB1 Geo overflow
    if (uniforms.fb1_geo_overflow == 1) {{
        fb1_coords = wrap_coord(fb1_coords);
    }} else if (uniforms.fb1_geo_overflow == 2) {{
        fb1_coords = mirror_coord(fb1_coords);
    }}
    
    // Sample FB1
    let fb1_uv = fb1_coords / vec2<f32>(uniforms.width, uniforms.height);
    var fb1_color = blur_and_sharpen(fb1_tex, fb1_sampler, fb1_uv,
                                      uniforms.fb1_sharpen_amount, uniforms.fb1_sharpen_radius, 
                                      uniforms.fb1_filters_boost,
                                      uniforms.fb1_blur_radius, uniforms.fb1_blur_amount);
    
    // Clamp FB1 if no overflow
    if (uniforms.fb1_geo_overflow == 0) {{
        if (fb1_coords.x > uniforms.width || fb1_coords.y > uniforms.height || 
            fb1_coords.x < 0.0 || fb1_coords.y < 0.0) {{
            fb1_color = vec4<f32>(0.0);
        }}
    }}
    
    // FB1 HSB processing
    var fb1_hsb = rgb2hsb(fb1_color.rgb);
    fb1_hsb += uniforms.fb1_hsb_offset;
    fb1_hsb = pow(fb1_hsb, uniforms.fb1_hsb_attenuate);
    fb1_hsb.x = apply_hue_shaper(fb1_hsb.x, uniforms.fb1_hue_shaper);
    
    // FB1 Inverts
    if (get_switch(uniforms.fb1_switches, 4u)) {{ // hue_invert
        fb1_hsb.x = 1.0 - fb1_hsb.x;
    }}
    if (get_switch(uniforms.fb1_switches, 5u)) {{ // sat_invert
        fb1_hsb.y = 1.0 - fb1_hsb.y;
    }}
    if (get_switch(uniforms.fb1_switches, 6u)) {{ // bright_invert
        fb1_hsb.z = 1.0 - fb1_hsb.z;
    }}
    
    fb1_hsb.x = fract(fb1_hsb.x);
    fb1_hsb.y = clamp(fb1_hsb.y, 0.0, 1.0);
    fb1_hsb.z = clamp(fb1_hsb.z, 0.0, 1.0);
    var fb1_rgb = hsb2rgb(fb1_hsb);
    
    // FB1 Posterize
    if (get_switch(uniforms.fb1_switches, 9u)) {{ // posterize_switch
        fb1_rgb = color_quantize(fb1_rgb, uniforms.fb1_posterize, uniforms.fb1_posterize_inv);
    }}
    
    fb1_color = vec4<f32>(fb1_rgb, fb1_color.a);
    
    // === Mix with FB1 ===
    var final_color = mix_and_key(
        mixed_color, fb1_color, uniforms.fb1_mix_amount, uniforms.fb1_mix_type,
        uniforms.fb1_key_threshold, uniforms.fb1_key_soft, uniforms.fb1_key_value,
        uniforms.fb1_key_order, uniforms.fb1_mix_overflow
    );
    
    return final_color;
}}
"#);
        
        // Create shader module
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Block1 Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_code.into()),
        });
        
        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Block1 Bind Group Layout"),
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
                // FB1 texture
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
                // FB1 sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // Temporal texture
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
                // Temporal sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        
        // Create pipeline layout
        let pipeline_layout = create_pipeline_layout(device, &bind_group_layout, "Block1 Pipeline Layout");
        
        // Create render pipeline
        let pipeline = create_render_pipeline(
            device,
            &pipeline_layout,
            &shader,
            &shader,
            wgpu::TextureFormat::Bgra8Unorm,
            "Block1 Pipeline",
        );
        
        // Create uniform buffer
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Block1 Uniform Buffer"),
            size: std::mem::size_of::<Block1Uniforms>() as u64,
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
            label: Some("Block1 Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&dummy_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&dummy_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&dummy_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&dummy_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&dummy_view),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::Sampler(&dummy_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::TextureView(&dummy_view),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::Sampler(&dummy_sampler),
                },
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
    pub fn update_params(&self, queue: &wgpu::Queue, params: &Block1Params, input1_size: (u32, u32), input2_size: (u32, u32)) {
        let uniforms = Block1Uniforms {
            width: self.width as f32,
            height: self.height as f32,
            inv_width: 1.0 / self.width as f32,
            inv_height: 1.0 / self.height as f32,
            
            // Input texture dimensions for proper UV scaling
            ch1_input_width: input1_size.0 as f32,
            ch1_input_height: input1_size.1 as f32,
            ch2_input_width: input2_size.0 as f32,
            ch2_input_height: input2_size.1 as f32,
            
            // Channel 1
            ch1_aspect: 1.0,
            ch1_crib_x: 0.0,
            ch1_scale: 1.0,
            ch1_hd_zcrib: 0.0,
            ch1_xy_displace: [params.ch1_x_displace, params.ch1_y_displace],
            ch1_z_displace: params.ch1_z_displace,
            ch1_rotate: params.ch1_rotate,
            ch1_hsb_attenuate: params.ch1_hsb_attenuate.into(),
            ch1_posterize: params.ch1_posterize,
            ch1_posterize_inv: 1.0 / params.ch1_posterize,
            ch1_kaleidoscope: params.ch1_kaleidoscope_amount,
            ch1_kaleidoscope_slice: params.ch1_kaleidoscope_slice,
            ch1_blur_amount: params.ch1_blur_amount,
            ch1_blur_radius: params.ch1_blur_radius,
            ch1_sharpen_amount: params.ch1_sharpen_amount,
            ch1_sharpen_radius: params.ch1_sharpen_radius,
            ch1_filters_boost: params.ch1_filters_boost,
            ch1_switches: pack_switches(
                params.ch1_h_mirror,
                params.ch1_v_mirror,
                params.ch1_h_flip,
                params.ch1_v_flip,
                params.ch1_hue_invert,
                params.ch1_saturation_invert,
                params.ch1_bright_invert,
                params.ch1_rgb_invert,
                params.ch1_solarize,
                params.ch1_posterize_switch,
            ),
            ch1_geo_overflow: params.ch1_geo_overflow,
            ch1_hd_aspect_on: if params.ch1_hd_aspect_on { 1 } else { 0 },
            _pad1: 0.0,
            
            // Channel 2 mix
            ch2_mix_amount: params.ch2_mix_amount,
            ch2_key_value: Vec3::new(params.ch2_key_value_red, params.ch2_key_value_green, params.ch2_key_value_blue),
            ch2_key_threshold: params.ch2_key_threshold,
            ch2_key_soft: params.ch2_key_soft,
            ch2_mix_type: params.ch2_mix_type,
            ch2_mix_overflow: params.ch2_mix_overflow,
            ch2_key_order: params.ch2_key_order,
            ch2_key_mode: params.ch2_key_mode,
            
            // Channel 2 adjust
            ch2_aspect: 1.0,
            ch2_crib_x: 0.0,
            ch2_scale: 1.0,
            ch2_hd_zcrib: 0.0,
            ch2_xy_displace: [params.ch2_x_displace, params.ch2_y_displace],
            ch2_z_displace: params.ch2_z_displace,
            ch2_rotate: params.ch2_rotate,
            ch2_hsb_attenuate: params.ch2_hsb_attenuate.into(),
            ch2_posterize: params.ch2_posterize,
            ch2_posterize_inv: 1.0 / params.ch2_posterize,
            ch2_kaleidoscope: params.ch2_kaleidoscope_amount,
            ch2_kaleidoscope_slice: params.ch2_kaleidoscope_slice,
            ch2_blur_amount: params.ch2_blur_amount,
            ch2_blur_radius: params.ch2_blur_radius,
            ch2_sharpen_amount: params.ch2_sharpen_amount,
            ch2_sharpen_radius: params.ch2_sharpen_radius,
            ch2_filters_boost: params.ch2_filters_boost,
            ch2_switches: pack_switches(
                params.ch2_h_mirror,
                params.ch2_v_mirror,
                params.ch2_h_flip,
                params.ch2_v_flip,
                params.ch2_hue_invert,
                params.ch2_saturation_invert,
                params.ch2_bright_invert,
                params.ch2_rgb_invert,
                params.ch2_solarize,
                params.ch2_posterize_switch,
            ),
            ch2_geo_overflow: params.ch2_geo_overflow,
            ch2_hd_aspect_on: if params.ch2_hd_aspect_on { 1 } else { 0 },
            _pad3: 0.0,
            
            // FB1 mix
            fb1_mix_amount: params.fb1_mix_amount,
            fb1_key_value: Vec3::new(params.fb1_key_value_red, params.fb1_key_value_green, params.fb1_key_value_blue),
            fb1_key_threshold: params.fb1_key_threshold,
            fb1_key_soft: params.fb1_key_soft,
            fb1_mix_type: params.fb1_mix_type,
            fb1_mix_overflow: params.fb1_mix_overflow,
            fb1_key_order: params.fb1_key_order,
            _pad4: 0,
            
            // FB1 geometry
            fb1_xy_displace: [params.fb1_x_displace, params.fb1_y_displace],
            fb1_z_displace: params.fb1_z_displace,
            fb1_rotate: params.fb1_rotate,
            fb1_shear_matrix: params.fb1_shear_matrix,
            fb1_kaleidoscope: params.fb1_kaleidoscope_amount,
            fb1_kaleidoscope_slice: params.fb1_kaleidoscope_slice,
            
            // FB1 color
            fb1_hsb_offset: params.fb1_hsb_offset.into(),
            fb1_hue_shaper: params.fb1_hue_shaper,
            fb1_hsb_attenuate: params.fb1_hsb_attenuate.into(),
            fb1_hsb_powmap: params.fb1_hsb_powmap.into(),
            fb1_posterize: params.fb1_posterize,
            fb1_posterize_inv: params.fb1_posterize_invert,
            
            // FB1 filters
            fb1_blur_amount: params.fb1_blur_amount,
            fb1_blur_radius: params.fb1_blur_radius,
            fb1_sharpen_amount: params.fb1_sharpen_amount,
            fb1_sharpen_radius: params.fb1_sharpen_radius,
            fb1_temporal1_amount: params.fb1_temporal_filter1_amount,
            fb1_temporal1_res: params.fb1_temporal_filter1_resonance,
            fb1_temporal2_amount: params.fb1_temporal_filter2_amount,
            fb1_temporal2_res: params.fb1_temporal_filter2_resonance,
            fb1_filters_boost: params.fb1_filters_boost,
            fb1_switches: pack_switches(
                params.fb1_h_mirror,
                params.fb1_v_mirror,
                params.fb1_h_flip,
                params.fb1_v_flip,
                params.fb1_hue_invert,
                params.fb1_saturation_invert,
                params.fb1_bright_invert,
                false, // placeholder
                false, // placeholder
                params.fb1_posterize_switch,
            ),
            fb1_rotate_mode: params.fb1_rotate_mode,
            fb1_geo_overflow: params.fb1_geo_overflow,
            _pad5: 0.0,
            
            // Input selection
            ch1_input_select: params.ch1_input_select,
            ch2_input_select: params.ch2_input_select,
            _pad6: [0.0, 0.0],
        };
        
        let bytes = unsafe {
            std::slice::from_raw_parts(
                &uniforms as *const _ as *const u8,
                std::mem::size_of::<Block1Uniforms>()
            )
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytes);
    }
    
    /// Update textures in bind group
    pub fn update_textures(
        &mut self, 
        device: &wgpu::Device, 
        input1_view: &wgpu::TextureView,
        input2_view: &wgpu::TextureView,
        fb1_view: &wgpu::TextureView, 
        temporal_view: &wgpu::TextureView
    ) {
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        
        self.bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Block1 Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                // Input 1
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(input1_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                // Input 2
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(input2_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                // FB1
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(fb1_view),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                // Temporal
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::TextureView(temporal_view),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });
    }
}

/// Pack boolean switches into a u32
fn pack_switches(
    h_mirror: bool,
    v_mirror: bool,
    h_flip: bool,
    v_flip: bool,
    hue_inv: bool,
    sat_inv: bool,
    bright_inv: bool,
    rgb_inv: bool,
    solarize: bool,
    posterize: bool,
) -> u32 {
    let mut result = 0u32;
    if h_mirror { result |= 1 << 0; }
    if v_mirror { result |= 1 << 1; }
    if h_flip { result |= 1 << 2; }
    if v_flip { result |= 1 << 3; }
    if hue_inv { result |= 1 << 4; }
    if sat_inv { result |= 1 << 5; }
    if bright_inv { result |= 1 << 6; }
    if rgb_inv { result |= 1 << 7; }
    if solarize { result |= 1 << 8; }
    if posterize { result |= 1 << 9; }
    result
}
