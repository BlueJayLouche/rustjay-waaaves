//! # Texture Module
//!
//! Texture management for wgpu.

#[derive(Clone, Copy, Debug)]
pub struct ReadbackLayout {
    pub row_bytes: u32,
    pub padded_bytes_per_row: u32,
    pub buffer_size: u64,
}

impl ReadbackLayout {
    pub fn new(width: u32, height: u32) -> Self {
        let row_bytes = width * 4;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bytes_per_row = row_bytes.div_ceil(align) * align;
        let buffer_size = (padded_bytes_per_row as u64) * (height as u64);

        Self {
            row_bytes,
            padded_bytes_per_row,
            buffer_size,
        }
    }
}

pub fn strip_readback_padding(data: &[u8], layout: ReadbackLayout, height: u32) -> Vec<u8> {
    if layout.padded_bytes_per_row == layout.row_bytes {
        return data.to_vec();
    }

    let mut out = Vec::with_capacity((layout.row_bytes * height) as usize);
    for row in 0..height as usize {
        let start = row * layout.padded_bytes_per_row as usize;
        let end = start + layout.row_bytes as usize;
        out.extend_from_slice(&data[start..end]);
    }
    out
}

/// Render target texture wrapper
pub struct Texture {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
    pub width: u32,
    pub height: u32,
}

impl Texture {
    /// Create a new render target texture (defaults to Bgra8Unorm, macOS native format)
    pub fn create_render_target(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        label: &str,
    ) -> Self {
        Self::create_render_target_with_format(device, width, height, label, wgpu::TextureFormat::Bgra8Unorm)
    }
    
    /// Create a new render target texture with specific format
    pub fn create_render_target_with_format(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        label: &str,
        format: wgpu::TextureFormat,
    ) -> Self {
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT 
                | wgpu::TextureUsages::TEXTURE_BINDING 
                | wgpu::TextureUsages::COPY_SRC 
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        
        Self {
            texture,
            view,
            sampler,
            width,
            height,
        }
    }
    
    /// Clear texture to black (for feedback texture initialization)
    pub fn clear_to_black(&self, queue: &wgpu::Queue) {
        let black_data = vec![0u8; (self.width * self.height * 4) as usize];
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &black_data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * self.width),
                rows_per_image: Some(self.height),
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
    }
    
    /// Create a texture from raw bytes (for input images)
    pub fn from_bytes(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bytes: &[u8],
        width: u32,
        height: u32,
        label: &str,
    ) -> Self {
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * width),
                rows_per_image: Some(height),
            },
            size,
        );
        
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        
        Self {
            texture,
            view,
            sampler,
            width,
            height,
        }
    }
}
