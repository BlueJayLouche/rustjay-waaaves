//! # Block 2 Pipeline
//!
//! Secondary input processing with WGSL shader.

use crate::engine::pipelines::{create_pipeline_layout, create_render_pipeline, COMMON_VERTEX_SHADER};
use crate::params::Block2Params;
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

/// Uniforms for Block2 shader
#[repr(C, align(16))]
#[derive(Debug, Copy, Clone, bytemuck::AnyBitPattern)]
pub struct Block2Uniforms {
    pub width: f32,
    pub height: f32,
    pub inv_width: f32,
    pub inv_height: f32,
    
    // Block 2 input
    pub input_aspect: f32,
    pub input_crib_x: f32,
    pub input_scale: f32,
    pub input_hd_zcrib: f32,
    pub input_xy_displace: [f32; 2],
    pub input_z_displace: f32,
    pub input_rotate: f32,
    pub input_hsb_attenuate: Vec3,
    pub input_posterize: f32,
    pub input_posterize_inv: f32,
    pub input_kaleidoscope: f32,
    pub input_kaleidoscope_slice: f32,
    pub input_blur_amount: f32,
    pub input_blur_radius: f32,
    pub input_sharpen_amount: f32,
    pub input_sharpen_radius: f32,
    pub input_filters_boost: f32,
    pub input_switches: u32,
    pub input_posterize_switch: i32,
    pub input_solarize: i32,
    pub input_geo_overflow: i32,
    pub input_hd_aspect_on: i32,
    pub _pad1: f32,
    
    // FB2
    pub fb2_mix_amount: f32,
    pub fb2_key_value: Vec3,
    pub fb2_key_threshold: f32,
    pub fb2_key_soft: f32,
    pub fb2_mix_type: i32,
    pub fb2_mix_overflow: i32,
    pub fb2_key_order: i32,
    pub _pad2: f32,
    
    pub fb2_xy_displace: [f32; 2],
    pub fb2_z_displace: f32,
    pub fb2_rotate: f32,
    pub fb2_shear_matrix: Vec4,
    pub fb2_kaleidoscope: f32,
    pub fb2_kaleidoscope_slice: f32,
    pub fb2_hsb_offset: Vec3,
    pub fb2_hsb_attenuate: Vec3,
    pub fb2_hsb_powmap: Vec3,
    pub fb2_hue_shaper: f32,
    pub fb2_posterize: f32,
    pub fb2_posterize_inv: f32,
    pub fb2_blur_amount: f32,
    pub fb2_blur_radius: f32,
    pub fb2_sharpen_amount: f32,
    pub fb2_sharpen_radius: f32,
    pub fb2_temporal1_amount: f32,
    pub fb2_temporal1_res: f32,
    pub fb2_temporal2_amount: f32,
    pub fb2_temporal2_res: f32,
    pub fb2_filters_boost: f32,
    pub fb2_switches: u32,
    pub fb2_posterize_switch: i32,
    pub fb2_rotate_mode: i32,
    pub fb2_geo_overflow: i32,
    
    // Input selection (0=block1, 1=input1, 2=input2)
    pub block2_input_select: i32,
    pub _pad4: [f32; 3],
}

impl Default for Block2Uniforms {
    fn default() -> Self {
        Self {
            width: 1280.0,
            height: 720.0,
            inv_width: 1.0 / 1280.0,
            inv_height: 1.0 / 720.0,
            input_aspect: 1.0,
            input_crib_x: 0.0,
            input_scale: 1.0,
            input_hd_zcrib: 0.0,
            input_xy_displace: [0.0, 0.0],
            input_z_displace: 1.0,
            input_rotate: 0.0,
            input_hsb_attenuate: Vec3::new(1.0, 1.0, 1.0),
            input_posterize: 16.0,
            input_posterize_inv: 1.0 / 16.0,
            input_kaleidoscope: 0.0,
            input_kaleidoscope_slice: 0.0,
            input_blur_amount: 0.0,
            input_blur_radius: 1.0,
            input_sharpen_amount: 0.0,
            input_sharpen_radius: 1.0,
            input_filters_boost: 0.0,
            input_switches: 0,
            input_posterize_switch: 0,
            input_solarize: 0,
            input_geo_overflow: 0,
            input_hd_aspect_on: 0,
            _pad1: 0.0,
            fb2_mix_amount: 0.0,
            fb2_key_value: Vec3::new(0.0, 0.0, 0.0),
            fb2_key_threshold: 1.0,
            fb2_key_soft: 0.0,
            fb2_mix_type: 0,
            fb2_mix_overflow: 0,
            fb2_key_order: 0,
            _pad2: 0.0,
            fb2_xy_displace: [0.0, 0.0],
            fb2_z_displace: 1.0,
            fb2_rotate: 0.0,
            fb2_shear_matrix: Vec4::new(1.0, 0.0, 0.0, 1.0),
            fb2_kaleidoscope: 0.0,
            fb2_kaleidoscope_slice: 0.0,
            fb2_hsb_offset: Vec3::new(0.0, 0.0, 0.0),
            fb2_hsb_attenuate: Vec3::new(1.0, 1.0, 1.0),
            fb2_hsb_powmap: Vec3::new(1.0, 1.0, 1.0),
            fb2_hue_shaper: 1.0,
            fb2_posterize: 16.0,
            fb2_posterize_inv: 1.0 / 16.0,
            fb2_blur_amount: 0.0,
            fb2_blur_radius: 1.0,
            fb2_sharpen_amount: 0.0,
            fb2_sharpen_radius: 1.0,
            fb2_temporal1_amount: 0.0,
            fb2_temporal1_res: 0.0,
            fb2_temporal2_amount: 0.0,
            fb2_temporal2_res: 0.0,
            fb2_filters_boost: 0.0,
            fb2_switches: 0,
            fb2_posterize_switch: 0,
            fb2_rotate_mode: 0,
            fb2_geo_overflow: 0,
            block2_input_select: 0,  // Default to block1
            _pad4: [0.0; 3],
        }
    }
}

/// Block 2 render pipeline
pub struct Block2Pipeline {
    pub pipeline: wgpu::RenderPipeline,
    pub bind_group: wgpu::BindGroup,
    pub uniform_buffer: wgpu::Buffer,
    pub bind_group_layout: wgpu::BindGroupLayout,
    width: u32,
    height: u32,
}

impl Block2Pipeline {
    /// Create new Block2 pipeline
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let shader_code = format!(r#"
{}

@group(0) @binding(0)
var<uniform> uniforms: Block2Uniforms;

@group(0) @binding(1)
var input_tex: texture_2d<f32>;
@group(0) @binding(2)
var input_sampler: sampler;
@group(0) @binding(3)
var fb2_tex: texture_2d<f32>;
@group(0) @binding(4)
var fb2_sampler: sampler;
@group(0) @binding(5)
var temporal_tex: texture_2d<f32>;
@group(0) @binding(6)
var temporal_sampler: sampler;

struct Block2Uniforms {{
    width: f32,
    height: f32,
    inv_width: f32,
    inv_height: f32,
    
    input_aspect: f32,
    input_crib_x: f32,
    input_scale: f32,
    input_hd_zcrib: f32,
    input_xy_displace: vec2<f32>,
    input_z_displace: f32,
    input_rotate: f32,
    input_hsb_attenuate: vec3<f32>,
    input_posterize: f32,
    input_posterize_inv: f32,
    input_kaleidoscope: f32,
    input_kaleidoscope_slice: f32,
    input_blur_amount: f32,
    input_blur_radius: f32,
    input_sharpen_amount: f32,
    input_sharpen_radius: f32,
    input_filters_boost: f32,
    input_switches: u32,
    input_posterize_switch: i32,
    input_solarize: i32,
    input_geo_overflow: i32,
    input_hd_aspect_on: i32,
    _pad1: f32,
    
    fb2_mix_amount: f32,
    fb2_key_value: vec3<f32>,
    fb2_key_threshold: f32,
    fb2_key_soft: f32,
    fb2_mix_type: i32,
    fb2_mix_overflow: i32,
    fb2_key_order: i32,
    _pad2: f32,
    
    fb2_xy_displace: vec2<f32>,
    fb2_z_displace: f32,
    fb2_rotate: f32,
    fb2_shear_matrix: vec4<f32>,
    fb2_kaleidoscope: f32,
    fb2_kaleidoscope_slice: f32,
    fb2_hsb_offset: vec3<f32>,
    fb2_hsb_attenuate: vec3<f32>,
    fb2_hsb_powmap: vec3<f32>,
    fb2_hue_shaper: f32,
    fb2_posterize: f32,
    fb2_posterize_inv: f32,
    fb2_blur_amount: f32,
    fb2_blur_radius: f32,
    fb2_sharpen_amount: f32,
    fb2_sharpen_radius: f32,
    fb2_temporal1_amount: f32,
    fb2_temporal1_res: f32,
    fb2_temporal2_amount: f32,
    fb2_temporal2_res: f32,
    fb2_filters_boost: f32,
    fb2_switches: u32,
    fb2_posterize_switch: i32,
    fb2_rotate_mode: i32,
    fb2_geo_overflow: i32,
    
    // Input selection (0=block1, 1=input1, 2=input2)
    block2_input_select: i32,
    _pad4: vec3<f32>,
}}

// Helper functions
const TWO_PI: f32 = 6.28318530718;

fn get_switch(switches: u32, bit: u32) -> bool {{
    return (switches & (1u << bit)) != 0u;
}}

fn rgb2hsb(c: vec3<f32>) -> vec3<f32> {{
    let K = vec4<f32>(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    let p = mix(vec4<f32>(c.bg, K.wz), vec4<f32>(c.gb, K.xy), step(c.b, c.g));
    let q = mix(vec4<f32>(p.xyw, c.r), vec4<f32>(c.r, p.yzx), step(p.x, c.r));
    let d = q.x - min(q.w, q.y);
    let e = 1.0e-10;
    return vec3<f32>(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}}

fn hsb2rgb(c: vec3<f32>) -> vec3<f32> {{
    let K = vec4<f32>(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    let p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, vec3<f32>(0.0), vec3<f32>(1.0)), c.y);
}}

fn do_rotate(coord: vec2<f32>, angle: f32, mode: i32) -> vec2<f32> {{
    let center = vec2<f32>(uniforms.width * 0.5, uniforms.height * 0.5);
    let rad = radians(angle);
    let c = cos(rad);
    let s = sin(rad);
    
    var rotate_coord = vec2<f32>(0.0, 0.0);
    
    // Mode 0: spiral effect (original)
    if (mode == 0) {{
        let delta = coord - center;
        rotate_coord.x = delta.x * c - delta.y * s + center.x;
        rotate_coord.y = delta.x * s + delta.y * c + center.y;
    }}
    // Mode 1: preserve aspect ratio
    else {{
        let center_coord = vec2<f32>(
            (coord.x - uniforms.width * 0.5) * uniforms.inv_width,
            (coord.y - uniforms.height * 0.5) * uniforms.inv_height
        );
        rotate_coord.x = uniforms.width * (center_coord.x * c - center_coord.y * s) + uniforms.width * 0.5;
        rotate_coord.y = uniforms.height * (center_coord.x * s + center_coord.y * c) + uniforms.height * 0.5;
    }}
    
    return rotate_coord;
}}

fn do_kaleidoscope(coord: vec2<f32>, segments: f32, slice: f32) -> vec2<f32> {{
    if (segments <= 0.0) {{
        return coord;
    }}
    
    var result = do_rotate(coord, slice, 1);  // Use mode 1 for kaleidoscope
    let norm = result / vec2<f32>(uniforms.width, uniforms.height);
    let centered = norm * 2.0 - 1.0;
    let radius = length(centered);
    var angle = atan2(centered.y, centered.x);
    let segment_angle = TWO_PI / segments;
    angle = angle - segment_angle * floor(angle / segment_angle);
    angle = min(angle, segment_angle - angle);
    result = radius * vec2<f32>(cos(angle), sin(angle));
    result = (result * 0.5 + 0.5) * vec2<f32>(uniforms.width, uniforms.height);
    return do_rotate(result, -slice, 1);
}}

fn color_quantize(in_color: vec3<f32>, amount: f32, amount_inv: f32) -> vec3<f32> {{
    var result = in_color * amount;
    result = floor(result);
    result = result * amount_inv;
    return result;
}}

fn solarize(in_bright: f32) -> f32 {{
    if (in_bright > 0.5) {{
        return 1.0 - in_bright;
    }}
    return in_bright;
}}

fn shear_coord(coord: vec2<f32>, shear_matrix: vec4<f32>) -> vec2<f32> {{
    let center = vec2<f32>(uniforms.width * 0.5, uniforms.height * 0.5);
    var result = coord - center;
    result.x = shear_matrix.x * result.x + shear_matrix.y * result.y;
    result.y = shear_matrix.z * result.x + shear_matrix.w * result.y;
    return result + center;
}}

fn wrap_coord(coord: vec2<f32>) -> vec2<f32> {{
    return vec2<f32>(coord.x % uniforms.width, coord.y % uniforms.height);
}}

fn mirror_val(a: f32) -> f32 {{
    if (a > 0.0) {{ return a; }}
    return -(1.0 + a);
}}

fn mirror_coord(coord: vec2<f32>) -> vec2<f32> {{
    let w = uniforms.width - 1.0;
    let h = uniforms.height - 1.0;
    return vec2<f32>(
        w - mirror_val(coord.x % (2.0 * w) - w),
        h - mirror_val(coord.y % (2.0 * h) - h)
    );
}}

fn blur_and_sharpen(tex: texture_2d<f32>, tex_sampler: sampler, coord: vec2<f32>, 
                    sharpen_amount: f32, sharpen_radius: f32, sharpen_boost: f32,
                    blur_radius: f32, blur_amount: f32) -> vec4<f32> {{
    let original_color = textureSample(tex, tex_sampler, coord);
    
    if (blur_amount < 0.001 && sharpen_amount < 0.001) {{
        return original_color;
    }}
    
    let tex_size = vec2<f32>(uniforms.width, uniforms.height);
    let blur_size = vec2<f32>(blur_radius) / (tex_size - 1.0);
    let sharpen_size = vec2<f32>(sharpen_radius) / (tex_size - 1.0);
    
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
    
    let boost_factor = mix(1.0, 1.0 + sharpen_amount + sharpen_boost, step(0.001, sharpen_amount));
    color_blur_hsb.z *= boost_factor;
    
    // Clamp HSB values before converting back to RGB
    color_blur_hsb.x = fract(color_blur_hsb.x);
    color_blur_hsb.y = clamp(color_blur_hsb.y, 0.0, 1.0);
    color_blur_hsb.z = clamp(color_blur_hsb.z, 0.0, 1.0);
    
    return vec4<f32>(hsb2rgb(color_blur_hsb), 1.0);
}}

fn wrap_color(in_color: f32) -> f32 {{
    var result = in_color;
    if (result < 0.0) {{
        result = 1.0 - abs(result);
    }}
    if (result > 1.0) {{
        result = fract(result);
    }}
    return result;
}}

fn foldover(in_color: f32) -> f32 {{
    var result = in_color;
    if (result < 0.0) {{
        result = abs(result);
    }}
    if (result > 1.0) {{
        result = 1.0 - fract(result);
    }}
    if (result < 0.0) {{
        result = abs(result);
    }}
    return result;
}}

fn mix_and_key(fg: vec4<f32>, bg: vec4<f32>, amount: f32, mix_type: i32, 
               key_threshold: f32, key_soft: f32, key_value: vec3<f32>,
               key_order: i32, mix_overflow: i32) -> vec4<f32> {{
    var foreground = fg;
    var background = bg;
    
    if (key_order == 1) {{
        let temp = foreground;
        foreground = background;
        background = temp;
    }}
    
    var out_color: vec3<f32>;
    
    // Mix modes: 0=lerp, 1=add/sub, 2=diff, 3=mult, 4=dodge, 5+=key
    if (mix_type == 0) {{ // Standard mix (lerp)
        out_color = mix(foreground.rgb, background.rgb, amount);
    }} else if (mix_type == 1) {{ // Add/Subtract
        out_color = foreground.rgb + amount * background.rgb;
    }} else if (mix_type == 2) {{ // Difference
        out_color = abs(foreground.rgb - amount * background.rgb);
    }} else if (mix_type == 3) {{ // Multiply
        out_color = mix(foreground.rgb, foreground.rgb * background.rgb, amount);
    }} else if (mix_type == 4) {{ // Dodge
        out_color = mix(foreground.rgb, foreground.rgb / (1.00001 - background.rgb), amount);
    }} else {{
        // Key mix (type 5+)
        let diff = length(foreground.rgb - key_value);
        let key = smoothstep(key_threshold - key_soft, key_threshold + key_soft, diff);
        out_color = mix(foreground.rgb, background.rgb, key * amount);
    }}
    
    // Overflow handling: 0=clamp, 1=wrap, 2=fold
    if (mix_overflow == 0) {{
        out_color = clamp(out_color, vec3<f32>(0.0), vec3<f32>(1.0));
    }} else if (mix_overflow == 1) {{
        out_color.r = wrap_color(out_color.r);
        out_color.g = wrap_color(out_color.g);
        out_color.b = wrap_color(out_color.b);
    }} else if (mix_overflow == 2) {{
        out_color.r = foldover(out_color.r);
        out_color.g = foldover(out_color.g);
        out_color.b = foldover(out_color.b);
    }}
    
    return vec4<f32>(out_color, 1.0);
}}

fn process_input(uv: vec2<f32>, coords: vec2<f32>) -> vec4<f32> {{
    var ch_coords = coords;
    
    // Aspect and scale
    ch_coords.x *= uniforms.input_aspect;
    ch_coords.x -= uniforms.input_crib_x;
    ch_coords -= vec2<f32>(uniforms.width * 0.5, uniforms.height * 0.5);
    ch_coords *= uniforms.input_scale + uniforms.input_hd_zcrib;
    ch_coords += vec2<f32>(uniforms.width * 0.5, uniforms.height * 0.5);
    
    if (uniforms.input_hd_aspect_on == 1) {{
        ch_coords = uv * vec2<f32>(uniforms.width, uniforms.height);
    }}
    
    // H/V Flip
    if (get_switch(uniforms.input_switches, 2u)) {{
        ch_coords.x = uniforms.width - ch_coords.x;
    }}
    if (get_switch(uniforms.input_switches, 3u)) {{
        ch_coords.y = uniforms.height - ch_coords.y;
    }}
    
    // H/V Mirror
    if (get_switch(uniforms.input_switches, 0u)) {{
        if (ch_coords.x > uniforms.width * 0.5) {{
            ch_coords.x = abs(uniforms.width - ch_coords.x);
        }}
    }}
    if (get_switch(uniforms.input_switches, 1u)) {{
        if (ch_coords.y > uniforms.height * 0.5) {{
            ch_coords.y = abs(uniforms.height - ch_coords.y);
        }}
    }}
    
    // Kaleidoscope
    ch_coords = do_kaleidoscope(ch_coords, uniforms.input_kaleidoscope, uniforms.input_kaleidoscope_slice);
    
    // Displace
    ch_coords += uniforms.input_xy_displace;
    
    // Z displace (zoom)
    ch_coords -= vec2<f32>(uniforms.width * 0.5, uniforms.height * 0.5);
    ch_coords *= uniforms.input_z_displace;
    ch_coords += vec2<f32>(uniforms.width * 0.5, uniforms.height * 0.5);
    
    // Rotate (mode 0 = spiral effect)
    ch_coords = do_rotate(ch_coords, uniforms.input_rotate, 0);
    
    // Geo overflow
    if (uniforms.input_geo_overflow == 1) {{
        ch_coords = wrap_coord(ch_coords);
    }} else if (uniforms.input_geo_overflow == 2) {{
        ch_coords = mirror_coord(ch_coords);
    }}
    
    // Sample with filters
    let ch_uv = ch_coords / vec2<f32>(uniforms.width, uniforms.height);
    var ch_color = blur_and_sharpen(input_tex, input_sampler, ch_uv,
                                     uniforms.input_sharpen_amount, uniforms.input_sharpen_radius, uniforms.input_filters_boost,
                                     uniforms.input_blur_radius, uniforms.input_blur_amount);
    
    // Clamp if no overflow
    if (uniforms.input_geo_overflow == 0) {{
        if (ch_coords.x > uniforms.width || ch_coords.y > uniforms.height || 
            ch_coords.x < 0.0 || ch_coords.y < 0.0) {{
            ch_color = vec4<f32>(0.0);
        }}
    }}
    
    // HSB processing with early exit optimization
    let needs_hsb = uniforms.input_hsb_attenuate.x != 1.0 || uniforms.input_hsb_attenuate.y != 1.0 || uniforms.input_hsb_attenuate.z != 1.0 ||
                    get_switch(uniforms.input_switches, 4u) || get_switch(uniforms.input_switches, 5u) || get_switch(uniforms.input_switches, 6u) ||
                    uniforms.input_solarize == 1;
    
    var ch_rgb = ch_color.rgb;
    if (needs_hsb) {{
        var ch_hsb = rgb2hsb(ch_color.rgb);
        ch_hsb = pow(ch_hsb, uniforms.input_hsb_attenuate);
        
        if (get_switch(uniforms.input_switches, 4u)) {{ ch_hsb.x = 1.0 - ch_hsb.x; }}
        if (get_switch(uniforms.input_switches, 5u)) {{ ch_hsb.y = 1.0 - ch_hsb.y; }}
        if (get_switch(uniforms.input_switches, 6u)) {{ ch_hsb.z = 1.0 - ch_hsb.z; }}
        
        ch_hsb.x = fract(ch_hsb.x);
        
        // Solarize
        if (uniforms.input_solarize == 1) {{
            ch_hsb.z = solarize(ch_hsb.z);
        }}
        
        // Clamp saturation and brightness before converting back to RGB
        ch_hsb.y = clamp(ch_hsb.y, 0.0, 1.0);
        ch_hsb.z = clamp(ch_hsb.z, 0.0, 1.0);
        
        ch_rgb = hsb2rgb(ch_hsb);
    }}
    
    if (get_switch(uniforms.input_switches, 7u)) {{ ch_rgb = 1.0 - ch_rgb; }}
    
    // Posterize
    if (uniforms.input_posterize_switch == 1) {{
        ch_rgb = color_quantize(ch_rgb, uniforms.input_posterize, uniforms.input_posterize_inv);
    }}
    
    return vec4<f32>(ch_rgb, ch_color.a);
}}

fn process_fb2(uv: vec2<f32>, coords: vec2<f32>) -> vec4<f32> {{
    var fb_coords = coords;
    
    // H/V Flip
    if (get_switch(uniforms.fb2_switches, 2u)) {{
        fb_coords.x = uniforms.width - fb_coords.x;
    }}
    if (get_switch(uniforms.fb2_switches, 3u)) {{
        fb_coords.y = uniforms.height - fb_coords.y;
    }}
    
    // H/V Mirror
    if (get_switch(uniforms.fb2_switches, 0u)) {{
        if (fb_coords.x > uniforms.width * 0.5) {{
            fb_coords.x = abs(uniforms.width - fb_coords.x);
        }}
    }}
    if (get_switch(uniforms.fb2_switches, 1u)) {{
        if (fb_coords.y > uniforms.height * 0.5) {{
            fb_coords.y = abs(uniforms.height - fb_coords.y);
        }}
    }}
    
    // Kaleidoscope
    fb_coords = do_kaleidoscope(fb_coords, uniforms.fb2_kaleidoscope, uniforms.fb2_kaleidoscope_slice);
    
    // Displace
    fb_coords += uniforms.fb2_xy_displace;
    
    // Z displace
    fb_coords -= vec2<f32>(uniforms.width * 0.5, uniforms.height * 0.5);
    fb_coords *= uniforms.fb2_z_displace;
    fb_coords += vec2<f32>(uniforms.width * 0.5, uniforms.height * 0.5);
    
    // Rotate (with mode selection)
    fb_coords = do_rotate(fb_coords, uniforms.fb2_rotate, uniforms.fb2_rotate_mode);
    
    // Shear
    fb_coords = shear_coord(fb_coords, uniforms.fb2_shear_matrix);
    
    // Geo overflow
    if (uniforms.fb2_geo_overflow == 1) {{
        fb_coords = wrap_coord(fb_coords);
    }} else if (uniforms.fb2_geo_overflow == 2) {{
        fb_coords = mirror_coord(fb_coords);
    }}
    
    // Sample with filters
    let fb_uv = fb_coords / vec2<f32>(uniforms.width, uniforms.height);
    var fb_color = blur_and_sharpen(fb2_tex, fb2_sampler, fb_uv,
                                     uniforms.fb2_sharpen_amount, uniforms.fb2_sharpen_radius, uniforms.fb2_filters_boost,
                                     uniforms.fb2_blur_radius, uniforms.fb2_blur_amount);
    
    // Clamp if no overflow
    if (uniforms.fb2_geo_overflow == 0) {{
        if (fb_coords.x > uniforms.width || fb_coords.y > uniforms.height ||
            fb_coords.x < 0.0 || fb_coords.y < 0.0) {{
            fb_color = vec4<f32>(0.0);
        }}
    }}
    
    // HSB processing
    var fb_hsb = rgb2hsb(fb_color.rgb);
    fb_hsb += uniforms.fb2_hsb_offset;
    fb_hsb = pow(fb_hsb, uniforms.fb2_hsb_attenuate);
    fb_hsb = pow(fb_hsb, uniforms.fb2_hsb_powmap);
    
    // Hue shaper
    fb_hsb.x = fb_hsb.x * uniforms.fb2_hue_shaper;
    
    if (get_switch(uniforms.fb2_switches, 4u)) {{ fb_hsb.x = 1.0 - fb_hsb.x; }}
    if (get_switch(uniforms.fb2_switches, 5u)) {{ fb_hsb.y = 1.0 - fb_hsb.y; }}
    if (get_switch(uniforms.fb2_switches, 6u)) {{ fb_hsb.z = 1.0 - fb_hsb.z; }}
    
    fb_hsb.x = fract(fb_hsb.x);
    fb_hsb.y = clamp(fb_hsb.y, 0.0, 1.0);
    fb_hsb.z = clamp(fb_hsb.z, 0.0, 1.0);
    var fb_rgb = hsb2rgb(fb_hsb);
    
    // Posterize
    if (uniforms.fb2_posterize_switch == 1) {{
        fb_rgb = color_quantize(fb_rgb, uniforms.fb2_posterize, uniforms.fb2_posterize_inv);
    }}
    
    return vec4<f32>(fb_rgb, fb_color.a);
}}

@fragment
fn fs_main(@location(0) texcoord: vec2<f32>) -> @location(0) vec4<f32> {{
    let uv = vec2<f32>(texcoord.x, 1.0 - texcoord.y);
    let coords = uv * vec2<f32>(uniforms.width, uniforms.height);
    
    // TEMP DEBUG: Force orange output to verify shader is running
    // Uncomment the next line to test
    // return vec4<f32>(1.0, 0.5, 0.0, 1.0);
    
    // Process input
    let input_color = process_input(uv, coords);
    
    // Process FB2
    let fb2_color = process_fb2(uv, coords);
    
    // Mix input and FB2
    var final_color = mix_and_key(
        input_color, fb2_color, uniforms.fb2_mix_amount, uniforms.fb2_mix_type,
        uniforms.fb2_key_threshold, uniforms.fb2_key_soft, uniforms.fb2_key_value,
        uniforms.fb2_key_order, uniforms.fb2_mix_overflow
    );
    
    // Temporal Filter 1
    let temporal1_color = textureSample(temporal_tex, temporal_sampler, uv);
    var temporal1_hsb = rgb2hsb(temporal1_color.rgb);
    // Apply resonance to saturation and brightness
    temporal1_hsb.y = clamp(temporal1_hsb.y * (1.0 + uniforms.fb2_temporal1_res * 0.25), 0.0, 1.0);
    temporal1_hsb.z = clamp(temporal1_hsb.z * (1.0 + uniforms.fb2_temporal1_res * 0.5), 0.0, 1.0);
    let temporal1_rgb = hsb2rgb(temporal1_hsb);
    final_color = mix(final_color, vec4<f32>(temporal1_rgb, 1.0), uniforms.fb2_temporal1_amount);
    final_color = clamp(final_color, vec4<f32>(0.0), vec4<f32>(1.0));
    
    // Temporal Filter 2
    var temporal2_hsb = temporal1_hsb;  // Reuse HSB from filter 1
    temporal2_hsb.y = clamp(temporal2_hsb.y * (1.0 + uniforms.fb2_temporal2_res * 0.25), 0.0, 1.0);
    temporal2_hsb.z = clamp(temporal2_hsb.z * (1.0 + uniforms.fb2_temporal2_res * 0.5), 0.0, 1.0);
    let temporal2_rgb = hsb2rgb(temporal2_hsb);
    final_color = mix(final_color, vec4<f32>(temporal2_rgb, 1.0), uniforms.fb2_temporal2_amount);
    final_color = clamp(final_color, vec4<f32>(0.0), vec4<f32>(1.0));
    
    final_color.a = 1.0;
    return final_color;
}}
"#, COMMON_VERTEX_SHADER);
        
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Block2 Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_code.into()),
        });
        
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Block2 Bind Group Layout"),
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
        
        let pipeline_layout = create_pipeline_layout(device, &bind_group_layout, "Block2 Pipeline Layout");
        
        let pipeline = create_render_pipeline(
            device,
            &pipeline_layout,
            &shader,
            &shader,
            wgpu::TextureFormat::Bgra8Unorm,
            "Block2 Pipeline",
        );
        
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Block2 Uniform Buffer"),
            size: std::mem::size_of::<Block2Uniforms>() as u64,
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
            label: Some("Block2 Bind Group"),
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
    pub fn update_params(&self, queue: &wgpu::Queue, params: &Block2Params) {
        // Build input switches bitmask
        let input_switches: u32 = 
            (if params.block2_input_h_mirror { 1u32 << 0 } else { 0 }) |
            (if params.block2_input_v_mirror { 1u32 << 1 } else { 0 }) |
            (if params.block2_input_h_flip { 1u32 << 2 } else { 0 }) |
            (if params.block2_input_v_flip { 1u32 << 3 } else { 0 }) |
            (if params.block2_input_hue_invert { 1u32 << 4 } else { 0 }) |
            (if params.block2_input_saturation_invert { 1u32 << 5 } else { 0 }) |
            (if params.block2_input_bright_invert { 1u32 << 6 } else { 0 }) |
            (if params.block2_input_rgb_invert { 1u32 << 7 } else { 0 });
        
        // Build FB2 switches bitmask
        let fb2_switches: u32 =
            (if params.fb2_h_mirror { 1u32 << 0 } else { 0 }) |
            (if params.fb2_v_mirror { 1u32 << 1 } else { 0 }) |
            (if params.fb2_h_flip { 1u32 << 2 } else { 0 }) |
            (if params.fb2_v_flip { 1u32 << 3 } else { 0 }) |
            (if params.fb2_hue_invert { 1u32 << 4 } else { 0 }) |
            (if params.fb2_saturation_invert { 1u32 << 5 } else { 0 }) |
            (if params.fb2_bright_invert { 1u32 << 6 } else { 0 });
        
        let uniforms = Block2Uniforms {
            width: self.width as f32,
            height: self.height as f32,
            inv_width: 1.0 / self.width as f32,
            inv_height: 1.0 / self.height as f32,
            
            input_aspect: 1.0,  // Default aspect
            input_crib_x: 0.0,
            input_scale: 1.0,
            input_hd_zcrib: 0.0,
            input_xy_displace: [params.block2_input_x_displace, params.block2_input_y_displace],
            input_z_displace: params.block2_input_z_displace,
            input_rotate: params.block2_input_rotate,
            input_hsb_attenuate: params.block2_input_hsb_attenuate.into(),
            input_posterize: params.block2_input_posterize,
            input_posterize_inv: 1.0 / params.block2_input_posterize.max(1.0),
            input_kaleidoscope: params.block2_input_kaleidoscope_amount,
            input_kaleidoscope_slice: params.block2_input_kaleidoscope_slice,
            input_blur_amount: params.block2_input_blur_amount,
            input_blur_radius: params.block2_input_blur_radius,
            input_sharpen_amount: params.block2_input_sharpen_amount,
            input_sharpen_radius: params.block2_input_sharpen_radius,
            input_filters_boost: params.block2_input_filters_boost,
            input_switches,
            input_posterize_switch: if params.block2_input_posterize_switch { 1 } else { 0 },
            input_solarize: if params.block2_input_solarize { 1 } else { 0 },
            input_geo_overflow: params.block2_input_geo_overflow,
            input_hd_aspect_on: if params.block2_input_hd_aspect_on { 1 } else { 0 },
            _pad1: 0.0,
            
            fb2_mix_amount: params.fb2_mix_amount,
            fb2_key_value: params.fb2_key_value.into(),
            fb2_key_threshold: params.fb2_key_threshold,
            fb2_key_soft: params.fb2_key_soft,
            fb2_mix_type: params.fb2_mix_type,
            fb2_mix_overflow: params.fb2_mix_overflow,
            fb2_key_order: params.fb2_key_order,
            _pad2: 0.0,
            
            fb2_xy_displace: [params.fb2_x_displace, params.fb2_y_displace],
            fb2_z_displace: params.fb2_z_displace,
            fb2_rotate: params.fb2_rotate,
            fb2_shear_matrix: params.fb2_shear_matrix,
            fb2_kaleidoscope: params.fb2_kaleidoscope_amount,
            fb2_kaleidoscope_slice: params.fb2_kaleidoscope_slice,
            fb2_hsb_offset: params.fb2_hsb_offset.into(),
            fb2_hsb_attenuate: params.fb2_hsb_attenuate.into(),
            fb2_hsb_powmap: params.fb2_hsb_powmap.into(),
            fb2_hue_shaper: params.fb2_hue_shaper,
            fb2_posterize: params.fb2_posterize,
            fb2_posterize_inv: params.fb2_posterize_invert,
            fb2_blur_amount: params.fb2_blur_amount,
            fb2_blur_radius: params.fb2_blur_radius,
            fb2_sharpen_amount: params.fb2_sharpen_amount,
            fb2_sharpen_radius: params.fb2_sharpen_radius,
            fb2_temporal1_amount: params.fb2_temporal_filter1_amount,
            fb2_temporal1_res: params.fb2_temporal_filter1_resonance,
            fb2_temporal2_amount: params.fb2_temporal_filter2_amount,
            fb2_temporal2_res: params.fb2_temporal_filter2_resonance,
            fb2_filters_boost: params.fb2_filters_boost,
            fb2_switches,
            fb2_posterize_switch: if params.fb2_posterize_switch { 1 } else { 0 },
            fb2_rotate_mode: params.fb2_rotate_mode,
            fb2_geo_overflow: params.fb2_geo_overflow,
            
            block2_input_select: params.block2_input_select,
            _pad4: [0.0; 3],
        };
        
        let bytes = unsafe {
            std::slice::from_raw_parts(
                &uniforms as *const _ as *const u8,
                std::mem::size_of::<Block2Uniforms>()
            )
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytes);
    }
    
    /// Update textures in bind group
    pub fn update_textures(
        &mut self,
        device: &wgpu::Device,
        input_view: &wgpu::TextureView,
        fb2_view: &wgpu::TextureView,
        temporal_view: &wgpu::TextureView,
    ) {
        self.bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Block2 Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.uniform_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(input_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&device.create_sampler(&wgpu::SamplerDescriptor::default())) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(fb2_view) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&device.create_sampler(&wgpu::SamplerDescriptor::default())) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(temporal_view) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::Sampler(&device.create_sampler(&wgpu::SamplerDescriptor::default())) },
            ],
        });
    }
}
