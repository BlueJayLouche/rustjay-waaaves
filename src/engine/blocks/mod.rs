//! # Modular Blocks
//!
//! Block implementations using the 3-stage modular architecture:
//! 1. Input Sampling
//! 2. Effects Processing (optional)
//! 3. Mixing & Feedback

pub mod block1;
pub mod block2;
pub mod block3;

pub use block1::ModularBlock1;
pub use block2::ModularBlock2;
pub use block3::ModularBlock3;


use crate::engine::texture::Texture;

/// Common resources for a modular block
pub struct BlockResources {
    /// Ping-pong buffer A for CH1
    pub buffer_a: Texture,
    /// Ping-pong buffer B for CH1
    pub buffer_b: Texture,
    /// Buffer for CH2 processing (output of CH2 Stage 2)
    pub ch2_buffer: Texture,
    /// Feedback texture (for sampling previous frame)
    pub feedback: Texture,
    /// Which buffer is currently the output (0=A, 1=B)
    pub current_output: usize,
    /// Delay buffer ring - stores multiple frames for delay effect
    pub delay_buffers: Vec<Texture>,
    /// Current write position in delay ring buffer
    pub delay_write_index: usize,
    /// Maximum delay frames supported
    pub max_delay_frames: usize,
}

impl BlockResources {
    /// Create resources for a block with given dimensions
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, width: u32, height: u32, label: &str) -> Self {
        let buffer_a = Texture::create_render_target_with_format(
            device, width, height, 
            &format!("{} Buffer A", label),
            wgpu::TextureFormat::Bgra8Unorm,
        );
        
        let buffer_b = Texture::create_render_target_with_format(
            device, width, height,
            &format!("{} Buffer B", label),
            wgpu::TextureFormat::Bgra8Unorm,
        );
        
        let feedback = Texture::create_render_target_with_format(
            device, width, height,
            &format!("{} Feedback", label),
            wgpu::TextureFormat::Bgra8Unorm,
        );
        
        // Clear to black
        buffer_a.clear_to_black(queue);
        buffer_b.clear_to_black(queue);
        feedback.clear_to_black(queue);
        
        let ch2_buffer = Texture::create_render_target_with_format(
            device, width, height,
            &format!("{} CH2 Buffer", label),
            wgpu::TextureFormat::Bgra8Unorm,
        );
        ch2_buffer.clear_to_black(queue);
        
        // Create delay buffer ring (default 120 frames = 2 seconds at 60fps)
        let max_delay_frames = 120;
        let mut delay_buffers = Vec::with_capacity(max_delay_frames);
        for i in 0..max_delay_frames {
            let delay_buf = Texture::create_render_target_with_format(
                device, width, height,
                &format!("{} Delay {}", label, i),
                wgpu::TextureFormat::Bgra8Unorm,
            );
            delay_buf.clear_to_black(queue);
            delay_buffers.push(delay_buf);
        }
        
        Self {
            buffer_a,
            buffer_b,
            ch2_buffer,
            feedback,
            current_output: 0,
            delay_buffers,
            delay_write_index: 0,
            max_delay_frames,
        }
    }
    
    /// Get the current output texture view
    pub fn get_output_view(&self) -> &wgpu::TextureView {
        if self.current_output == 0 {
            &self.buffer_a.view
        } else {
            &self.buffer_b.view
        }
    }
    
    /// Get the texture to use as input for next stage (opposite of output)
    pub fn get_input_view(&self) -> &wgpu::TextureView {
        if self.current_output == 0 {
            &self.buffer_b.view
        } else {
            &self.buffer_a.view
        }
    }
    
    /// Swap ping-pong buffers
    pub fn swap(&mut self) {
        self.current_output = 1 - self.current_output;
    }
    
    /// Get feedback texture view
    pub fn get_feedback_view(&self) -> &wgpu::TextureView {
        &self.feedback.view
    }

    /// Clear feedback and all internal ping-pong buffers to black.
    /// Call this when switching inputs so stale frames don't bleed through.
    pub fn clear_all(&self, queue: &wgpu::Queue) {
        self.feedback.clear_to_black(queue);
        self.buffer_a.clear_to_black(queue);
        self.buffer_b.clear_to_black(queue);
        self.ch2_buffer.clear_to_black(queue);
        for buf in &self.delay_buffers {
            buf.clear_to_black(queue);
        }
    }
    
    /// Get output texture (not view) for copy operations
    pub fn get_output_texture(&self) -> &wgpu::Texture {
        if self.current_output == 0 {
            &self.buffer_a.texture
        } else {
            &self.buffer_b.texture
        }
    }
    
    /// Get CH2 buffer view
    pub fn get_ch2_view(&self) -> &wgpu::TextureView {
        &self.ch2_buffer.view
    }
    
    /// Copy output to feedback for next frame
    pub fn update_feedback(&self, encoder: &mut wgpu::CommandEncoder) {
        let output_tex = if self.current_output == 0 {
            &self.buffer_a.texture
        } else {
            &self.buffer_b.texture
        };
        
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: output_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &self.feedback.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: self.buffer_a.width,
                height: self.buffer_a.height,
                depth_or_array_layers: 1,
            },
        );
    }
    
    /// Copy external texture to feedback buffer (for when output is rendered externally)
    pub fn update_feedback_from_external(&self, encoder: &mut wgpu::CommandEncoder, source_texture: &wgpu::Texture) {
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: source_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &self.feedback.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: self.buffer_a.width,
                height: self.buffer_a.height,
                depth_or_array_layers: 1,
            },
        );
    }
    
    /// Get delay buffer view for reading (based on delay_time frames ago)
    pub fn get_delay_view(&self, delay_time: usize) -> &wgpu::TextureView {
        let delay_frames = delay_time.min(self.max_delay_frames).max(1);
        // Read from (write_index - delay_frames) positions ago
        let read_index = if self.delay_write_index >= delay_frames {
            self.delay_write_index - delay_frames
        } else {
            self.max_delay_frames - (delay_frames - self.delay_write_index)
        };
        &self.delay_buffers[read_index].view
    }
    
    /// Update delay buffer ring with current output and advance write index
    pub fn update_delay_buffer(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let output_tex = if self.current_output == 0 {
            &self.buffer_a.texture
        } else {
            &self.buffer_b.texture
        };
        let current_delay_buf = &self.delay_buffers[self.delay_write_index];
        
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: output_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &current_delay_buf.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: self.buffer_a.width,
                height: self.buffer_a.height,
                depth_or_array_layers: 1,
            },
        );
        
        // Advance write index
        self.delay_write_index = (self.delay_write_index + 1) % self.max_delay_frames;
    }
    
    /// Update delay buffer from an external texture
    pub fn update_delay_buffer_from_external(&mut self, encoder: &mut wgpu::CommandEncoder, source_texture: &wgpu::Texture) {
        let current_delay_buf = &self.delay_buffers[self.delay_write_index];
        
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: source_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &current_delay_buf.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: self.buffer_a.width,
                height: self.buffer_a.height,
                depth_or_array_layers: 1,
            },
        );
        
        // Advance write index
        self.delay_write_index = (self.delay_write_index + 1) % self.max_delay_frames;
    }
}

/// Vertex buffer for full-screen quad (shared across all stages)
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct StageVertex {
    pub position: [f32; 2],
    pub texcoord: [f32; 2],
}

impl StageVertex {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        }
    }
    
    /// Create vertex buffer for full-screen quad
    pub fn create_quad_buffer(device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::Buffer {
        let vertices = [
            // Position (x, y), Texcoord (u, v)
            StageVertex { position: [-1.0, -1.0], texcoord: [0.0, 1.0] }, // Bottom-left
            StageVertex { position: [ 1.0, -1.0], texcoord: [1.0, 1.0] }, // Bottom-right
            StageVertex { position: [-1.0,  1.0], texcoord: [0.0, 0.0] }, // Top-left
            StageVertex { position: [ 1.0, -1.0], texcoord: [1.0, 1.0] }, // Bottom-right
            StageVertex { position: [ 1.0,  1.0], texcoord: [1.0, 0.0] }, // Top-right
            StageVertex { position: [-1.0,  1.0], texcoord: [0.0, 0.0] }, // Top-left
        ];
        
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Stage Quad Vertex Buffer"),
            size: std::mem::size_of_val(&vertices) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        queue.write_buffer(&buffer, 0, bytemuck::cast_slice(&vertices));
        buffer
    }
}
