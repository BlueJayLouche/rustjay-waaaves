//! # Preview Renderer
//!
//! Lightweight preview window for block output with color picker support.
//! Based on the oF implementation from BLUEJAY_WAAAVES.
//!
//! Design principles:
//! - Zero re-rendering: Copies from existing textures, never re-runs shaders
//! - Throttled updates: Limited to 30 FPS to save GPU
//! - Async readback: GPU->CPU pixel transfer without stalling
//! - Small fixed size: 320x180 (16:9) for performance

pub use crate::core::PreviewSource;
use crate::engine::texture::{ReadbackLayout, strip_readback_padding};
use std::sync::{mpsc::{channel, Receiver}, Arc};

/// Preview renderer that copies from pipeline textures
pub struct PreviewRenderer {
    /// Preview texture size
    width: u32,
    height: u32,
    
    /// The preview texture (GPU side) - Arc for sharing with ImGui
    texture: Arc<wgpu::Texture>,
    texture_view: Arc<wgpu::TextureView>,
    
    /// Buffer for async pixel readback
    readback_buffer: wgpu::Buffer,
    readback_layout: ReadbackLayout,
    
    /// CPU-side pixel buffer (RGBA8)
    pixels: Vec<u8>,
    
    /// Whether buffer is mapped and ready to read
    map_receiver: Option<Receiver<Result<(), wgpu::BufferAsyncError>>>,
    
    /// Whether the buffer is currently mapped (can't submit commands to mapped buffer)
    buffer_mapped: bool,
    
    /// Current source selection
    source: PreviewSource,
    
    /// Throttling
    last_update_time: instant::Instant,
    update_interval: std::time::Duration, // 30 FPS default
    
    /// Enabled flag
    enabled: bool,
    
    /// Bind group layout for copy shader
    bind_group_layout: wgpu::BindGroupLayout,
    
    /// Pipeline for copying textures
    pipeline: wgpu::RenderPipeline,
    
    /// Sampler for texture sampling
    sampler: wgpu::Sampler,
}

impl PreviewRenderer {
    /// Create a new preview renderer
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Preview Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT 
                 | wgpu::TextureUsages::COPY_SRC
                 | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        
        let texture = Arc::new(texture);
        let texture_view = Arc::new(texture.create_view(&wgpu::TextureViewDescriptor::default()));
        
        // Buffer for async readback (4 bytes per pixel RGBA)
        let readback_layout = ReadbackLayout::new(width, height);
        let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Preview Readback Buffer"),
            size: readback_layout.buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        
        // CPU pixel buffer
        let pixels = vec![0u8; (width * height * 4) as usize];
        
        // Create bind group layout for texture sampling
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Preview Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        
        // Create sampler
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Preview Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        
        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Preview Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        
        // Create render pipeline for copying
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Preview Copy Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("preview_copy.wgsl").into()),
        });
        
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Preview Copy Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        
        Self {
            width,
            height,
            texture,
            texture_view,
            readback_buffer,
            readback_layout,
            pixels,
            map_receiver: None,
            buffer_mapped: false,
            source: PreviewSource::Block3,
            last_update_time: instant::Instant::now(),
            update_interval: std::time::Duration::from_millis(33), // 30 FPS
            enabled: true,
            bind_group_layout,
            pipeline,
            sampler,
        }
    }
    
    /// Update preview from source texture
    /// Returns true if preview was updated
    pub fn update(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        source_texture: &wgpu::TextureView,
    ) -> bool {
        if !self.enabled {
            return false;
        }
        
        // Check throttle
        let now = instant::Instant::now();
        if now.duration_since(self.last_update_time) < self.update_interval {
            return false;
        }
        
        // Can't submit commands if buffer is still mapped from previous frame
        if self.buffer_mapped {
            // Try to process readback first to unmap the buffer
            if !self.process_readback_internal() {
                // Still mapped, skip this update
                return false;
            }
        }
        
        self.last_update_time = now;
        
        // Create bind group for this source
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Preview Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(source_texture),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        
        // Render source to preview texture (handles scaling)
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Preview Copy Encoder"),
        });
        
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Preview Copy Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.texture_view,
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
            
            render_pass.set_pipeline(&self.pipeline);
            render_pass.set_bind_group(0, &bind_group, &[]);
            render_pass.draw(0..3, 0..1); // Fullscreen triangle
        }
        
        // Copy texture to buffer for async readback
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.readback_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.readback_layout.padded_bytes_per_row),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
        
        queue.submit(std::iter::once(encoder.finish()));
        
        // Start async read of pixel data
        let (tx, rx) = channel();
        self.readback_buffer.slice(..).map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        self.map_receiver = Some(rx);
        self.buffer_mapped = true;
        
        true
    }
    
    /// Process pending pixel readback (public API)
    /// Call this every frame to get updated pixels
    /// Returns true if pixels were successfully read
    pub fn process_readback(&mut self) -> bool {
        self.process_readback_internal()
    }
    
    /// Internal readback processing
    fn process_readback_internal(&mut self) -> bool {
        if !self.buffer_mapped {
            return false;
        }
        
        let rx = match self.map_receiver.take() {
            Some(rx) => rx,
            None => return false,
        };
        
        // Try to receive the result without blocking
        match rx.try_recv() {
            Ok(Ok(())) => {
                // Buffer is mapped, safe to read
                {
                    let data = self.readback_buffer.slice(..).get_mapped_range();
                    let pixels = strip_readback_padding(&data, self.readback_layout, self.height);
                    self.pixels.copy_from_slice(&pixels);
                }
                self.readback_buffer.unmap();
                self.buffer_mapped = false;
                true
            }
            Ok(Err(e)) => {
                log::warn!("Buffer mapping failed: {:?}", e);
                self.buffer_mapped = false;
                false
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {
                // Not ready yet, put receiver back for next frame
                self.map_receiver = Some(rx);
                false
            }
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                self.buffer_mapped = false;
                false
            }
        }
    }
    
    /// Get color at pixel position
    pub fn pick_color(&self, x: u32, y: u32) -> Option<[u8; 3]> {
        if x >= self.width || y >= self.height {
            return None;
        }
        
        let idx = ((y * self.width + x) * 4) as usize;
        if idx + 3 < self.pixels.len() {
            Some([self.pixels[idx], self.pixels[idx + 1], self.pixels[idx + 2]])
        } else {
            None
        }
    }
    
    /// Get normalized color at UV coordinates (0-1)
    pub fn pick_color_uv(&self, u: f32, v: f32) -> Option<[u8; 3]> {
        let x = (u * self.width as f32).clamp(0.0, self.width as f32 - 1.0) as u32;
        let y = (v * self.height as f32).clamp(0.0, self.height as f32 - 1.0) as u32;
        self.pick_color(x, y)
    }
    
    /// Get the preview texture view for ImGui display
    pub fn get_texture_view(&self) -> &wgpu::TextureView {
        &self.texture_view
    }
    
    /// Get the texture Arc for registration with ImGui
    pub fn get_texture_arc(&self) -> Arc<wgpu::Texture> {
        Arc::clone(&self.texture)
    }
    
    /// Get the texture view Arc for registration with ImGui
    pub fn get_texture_view_arc(&self) -> Arc<wgpu::TextureView> {
        Arc::clone(&self.texture_view)
    }
    
    /// Get texture format
    pub fn format(&self) -> wgpu::TextureFormat {
        wgpu::TextureFormat::Rgba8Unorm
    }
    
    /// Get current source
    pub fn source(&self) -> PreviewSource {
        self.source
    }
    
    /// Set source
    pub fn set_source(&mut self, source: PreviewSource) {
        self.source = source;
    }
    
    /// Set update rate (FPS)
    pub fn set_fps(&mut self, fps: u32) {
        self.update_interval = std::time::Duration::from_millis(1000 / fps.max(1) as u64);
    }
    
    /// Enable/disable
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
    
    /// Get dimensions
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}
