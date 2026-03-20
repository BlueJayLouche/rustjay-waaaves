//! # Simplified Rendering Engine
//! 
//! Single-shader feedback loop with auto-webcam and tempo-synced LFOs

use std::sync::Arc;
use std::time::Instant;
use wgpu::util::DeviceExt;

use crate::engine::texture::Texture;
use crate::engine::simple_feedback::{SimpleFeedbackPipeline, SimpleFeedbackUniforms};
use crate::engine::lfo_tempo::TempoLfoBank;
use crate::input::InputManager;
use crate::core::SharedState;

// Input texture management for simple engine
pub struct SimpleInputTexture {
    pub texture: Texture,
    pub width: u32,
    pub height: u32,
}

/// Simplified engine configuration
pub struct SimpleEngineConfig {
    /// Output resolution
    pub output_width: u32,
    pub output_height: u32,
    /// Webcam capture resolution (will be scaled to output)
    pub webcam_width: u32,
    pub webcam_height: u32,
    /// Use ping-pong feedback (true) or single texture (false)
    pub use_ping_pong: bool,
    /// Target FPS
    pub target_fps: f32,
}

impl Default for SimpleEngineConfig {
    fn default() -> Self {
        Self {
            output_width: 1280,
            output_height: 720,
            webcam_width: 640,
            webcam_height: 480,
            use_ping_pong: true,
            target_fps: 60.0,
        }
    }
}

/// Simplified rendering engine
pub struct SimpleEngine {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    instance: wgpu::Instance,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    
    // Pipeline
    feedback_pipeline: SimpleFeedbackPipeline,
    
    // Render targets
    output_texture: Texture,
    feedback_texture_a: Texture,
    feedback_texture_b: Option<Texture>, // For ping-pong
    
    // Which feedback texture to read from
    current_feedback_idx: usize,
    
    // Input
    input_manager: InputManager,
    input1_texture: Option<SimpleInputTexture>,
    input2_texture: Option<SimpleInputTexture>,
    
    // LFOs
    lfo_bank: TempoLfoBank,
    
    // Parameter overrides from UI
    hue_amount_override: Option<f32>,
    
    // Shared state
    shared_state: Arc<std::sync::Mutex<SharedState>>,
    
    // Timing
    last_frame_time: Instant,
    frame_count: u64,
    
    // Config
    config_settings: SimpleEngineConfig,
    
    // Vertex buffer for full-screen quad
    vertex_buffer: wgpu::Buffer,
    
    // Dummy black texture for when there's no input
    dummy_black_texture: Texture,
}

impl SimpleEngine {
    pub async fn new(
        window: Arc<winit::window::Window>,
        shared_state: Arc<std::sync::Mutex<SharedState>>,
        settings: SimpleEngineConfig,
    ) -> anyhow::Result<Self> {
        log::info!("[SIMPLE_ENGINE] Creating simplified engine");
        log::info!("[SIMPLE_ENGINE] Output: {}x{}", settings.output_width, settings.output_height);
        log::info!("[SIMPLE_ENGINE] Webcam: {}x{}", settings.webcam_width, settings.webcam_height);
        log::info!("[SIMPLE_ENGINE] Ping-pong: {}", settings.use_ping_pong);
        
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        
        let surface = instance.create_surface(window)?;
        
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| anyhow::anyhow!("Failed to find suitable adapter: {:?}", e))?;
        
        log::info!("[SIMPLE_ENGINE] Using adapter: {:?}", adapter.get_info());
        
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    label: Some("Simple Engine Device"),
                    memory_hints: wgpu::MemoryHints::default(),
                    trace: wgpu::Trace::Off,
                },
            )
            .await?;
        
        let device = Arc::new(device);
        let queue = Arc::new(queue);
        
        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);
        
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: settings.output_width,
            height: settings.output_height,
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);
        
        // Use surface format for all textures to ensure copy compatibility
        let surface_format = config.format;
        log::info!("[SIMPLE_ENGINE] Using surface format: {:?}", surface_format);
        
        // Create render targets with surface format
        let output_texture = Texture::create_render_target_with_format(
            &device,
            settings.output_width,
            settings.output_height,
            "Output Texture",
            surface_format,
        );
        
        let feedback_texture_a = Texture::create_render_target_with_format(
            &device,
            settings.output_width,
            settings.output_height,
            "Feedback A",
            surface_format,
        );
        feedback_texture_a.clear_to_black(&queue);
        
        let feedback_texture_b = if settings.use_ping_pong {
            let tex = Texture::create_render_target_with_format(
                &device,
                settings.output_width,
                settings.output_height,
                "Feedback B",
                surface_format,
            );
            tex.clear_to_black(&queue);
            Some(tex)
        } else {
            None
        };
        
        // Create dummy black texture for when there's no input (use surface format)
        let dummy_black_texture = Texture::create_render_target_with_format(
            &device,
            64,  // Small texture
            64,
            "Dummy Black",
            surface_format,
        );
        dummy_black_texture.clear_to_black(&queue);
        
        // Create pipeline with surface format
        let feedback_pipeline = SimpleFeedbackPipeline::new_with_format(
            &device,
            settings.output_width,
            settings.output_height,
            surface_format,
        );
        
        // Create input manager
        let mut input_manager = InputManager::new();
        
        // Initialize input manager with device and queue
        if let Err(e) = input_manager.initialize(device.clone(), queue.clone()) {
            log::warn!("[SIMPLE_ENGINE] Failed to initialize input manager: {}", e);
        } else {
            log::info!("[SIMPLE_ENGINE] Input manager initialized");
        }
        
        // Create LFO bank at 120 BPM
        let lfo_bank = TempoLfoBank::with_bpm(120.0);
        
        // Create vertex buffer for full-screen quad
        // Position (x, y) + Texcoord (u, v)
        let vertices: &[f32] = &[
            // Tri 1
            -1.0, -1.0, 0.0, 1.0,  // bottom-left
             1.0, -1.0, 1.0, 1.0,  // bottom-right
            -1.0,  1.0, 0.0, 0.0,  // top-left
            // Tri 2
             1.0, -1.0, 1.0, 1.0,  // bottom-right
             1.0,  1.0, 1.0, 0.0,  // top-right
            -1.0,  1.0, 0.0, 0.0,  // top-left
        ];
        
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Simple Engine Vertex Buffer"),
            contents: bytemuck::cast_slice(vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        
        log::info!("[SIMPLE_ENGINE] Engine created successfully");
        
        Ok(Self {
            device,
            queue,
            instance,
            surface,
            config,
            feedback_pipeline,
            output_texture,
            feedback_texture_a,
            feedback_texture_b,
            current_feedback_idx: 0,
            input_manager,
            input1_texture: None,
            input2_texture: None,
            lfo_bank,
            hue_amount_override: None,
            shared_state,
            last_frame_time: Instant::now(),
            frame_count: 0,
            config_settings: settings,
            vertex_buffer,
            dummy_black_texture,
        })
    }
    
    /// Auto-start webcam on input 1
    pub fn auto_start_webcam(&mut self) {
        log::info!("[SIMPLE_ENGINE] Auto-starting webcam...");
        
        // Start webcam directly on input 1 at VGA resolution
        let result = self.input_manager.start_input1_webcam(
            0, // First camera
            self.config_settings.webcam_width,
            self.config_settings.webcam_height,
            30 // 30 fps
        );
        
        match result {
            Ok(_) => log::info!("[SIMPLE_ENGINE] Webcam 1 started successfully"),
            Err(e) => log::error!("[SIMPLE_ENGINE] Failed to start webcam 1: {}", e),
        }
    }
    
    /// Start Syphon input on input 1 (macOS only)
    #[cfg(target_os = "macos")]
    pub fn start_input1_syphon(&mut self, server_name: &str) {
        log::info!("[SIMPLE_ENGINE] Starting Syphon input from '{}'...", server_name);
        
        // Initialize input manager with device/queue if not already done
        if let Err(e) = self.input_manager.initialize(self.device.clone(), self.queue.clone()) {
            log::error!("[SIMPLE_ENGINE] Failed to initialize input manager: {}", e);
            return;
        }
        
        let result = self.input_manager.start_input1_syphon(server_name);
        
        match result {
            Ok(_) => log::info!("[SIMPLE_ENGINE] Syphon input 1 started successfully from '{}'", server_name),
            Err(e) => log::error!("[SIMPLE_ENGINE] Failed to start Syphon input 1: {}", e),
        }
    }
    
    /// Start webcam on input 2
    pub fn start_input2_webcam(&mut self, device_index: usize) {
        log::info!("[SIMPLE_ENGINE] Starting webcam on input 2 (device {})...", device_index);
        
        let result = self.input_manager.start_input2_webcam(
            device_index,
            self.config_settings.webcam_width,
            self.config_settings.webcam_height,
            30 // 30 fps
        );
        
        match result {
            Ok(_) => log::info!("[SIMPLE_ENGINE] Webcam 2 started successfully"),
            Err(e) => log::error!("[SIMPLE_ENGINE] Failed to start webcam 2: {}", e),
        }
    }
    
    /// Resize output
    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            log::info!("[SIMPLE_ENGINE] Resizing to {}x{}", width, height);
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
            
            // Use surface format for all textures
            let surface_format = self.config.format;
            
            // Recreate output texture
            self.output_texture = Texture::create_render_target_with_format(
                &self.device,
                width,
                height,
                "Output Texture",
                surface_format,
            );
            
            // Recreate feedback textures
            self.feedback_texture_a = Texture::create_render_target_with_format(
                &self.device,
                width,
                height,
                "Feedback A",
                surface_format,
            );
            self.feedback_texture_a.clear_to_black(&self.queue);
            
            if let Some(ref mut tex_b) = self.feedback_texture_b {
                *tex_b = Texture::create_render_target_with_format(
                    &self.device,
                    width,
                    height,
                    "Feedback B",
                    surface_format,
                );
                tex_b.clear_to_black(&self.queue);
            }
        }
    }
    
    /// Main render loop
    pub fn render(&mut self) {
        let frame_start = Instant::now();
        
        // Calculate delta time
        let delta_time = self.last_frame_time.elapsed().as_secs_f32();
        self.last_frame_time = Instant::now();
        
        // Update LFOs
        self.lfo_bank.update(delta_time);
        let (hue_lfo, rotate_lfo, zoom_lfo) = self.lfo_bank.values();
        
        // Update inputs (poll for new frames)
        self.update_inputs();
        
        // Upload webcam frames to GPU
        self.upload_input_frames();
        
        // Get shared state params
        let params = self.get_params_from_shared_state(hue_lfo, rotate_lfo, zoom_lfo);
        
        // Update shader params first (before any view borrows)
        self.feedback_pipeline.update_params(&self.queue, &params);
        
        // Get texture views — prefer zero-copy GPU Syphon texture when available.
        // On macOS the SyphonWgpuInput holds the texture; we create a temporary view
        // without any GPU copy.  For all other inputs we use the CPU-uploaded texture.
        #[cfg(target_os = "macos")]
        let syphon_input1_view = self.input_manager
            .get_input1_syphon_texture()
            .map(|t| t.create_view(&wgpu::TextureViewDescriptor::default()));

        let input1_view_ref: &wgpu::TextureView = {
            #[cfg(target_os = "macos")]
            if let Some(ref v) = syphon_input1_view { v } else {
                match &self.input1_texture {
                    Some(tex) => &tex.texture.view,
                    None => &self.dummy_black_texture.view,
                }
            }
            #[cfg(not(target_os = "macos"))]
            match &self.input1_texture {
                Some(tex) => &tex.texture.view,
                None => &self.dummy_black_texture.view,
            }
        };

        // Input 2 uses the same texture as Input 1 (for self-mixing / fallback)
        let input2_view_ref = input1_view_ref;
        
        let feedback_read_view_ref: &wgpu::TextureView = if self.config_settings.use_ping_pong {
            if self.current_feedback_idx == 0 {
                &self.feedback_texture_a.view
            } else {
                &self.feedback_texture_b.as_ref().unwrap().view
            }
        } else {
            &self.feedback_texture_a.view
        };
        
        // Update shader textures
        self.feedback_pipeline.update_textures(
            &self.device,
            input1_view_ref,
            input2_view_ref,
            feedback_read_view_ref,
        );
        
        // Now get the write view (after texture update is done)
        let feedback_write_view = self.get_feedback_write_view();
        
        // Render
        self.do_render_pass(feedback_write_view);
        
        // Swap feedback textures for ping-pong
        if self.config_settings.use_ping_pong {
            self.current_feedback_idx = 1 - self.current_feedback_idx;
        }
        
        // Copy to surface
        self.present_output();
        
        self.frame_count += 1;
    }
    
    fn update_inputs(&mut self) {
        // Update input manager to poll for new frames
        self.input_manager.update();
    }
    
    fn upload_input_frames(&mut self) {
        // Upload input 1 frame if available
        if self.input_manager.input1_has_new_frame() {
            if let Some(frame_data) = self.input_manager.take_input1_frame() {
                let (width, height) = self.input_manager.get_input1_resolution();
                

                
                // Create or resize texture if needed
                let current_size = self.input1_texture.as_ref()
                    .map(|t| (t.width, t.height));
                let needs_create = current_size.map(|(w, h)| w != width || h != height).unwrap_or(true);
                
                if needs_create {
                    log::info!("[SIMPLE_ENGINE] Creating input 1 texture: {}x{} (current: {:?})", 
                        width, height, current_size);
                    // Use surface format for input texture to match pipeline
                    let surface_format = self.config.format;
                    self.input1_texture = Some(SimpleInputTexture {
                        texture: Texture::create_render_target_with_format(
                            &self.device, width, height, "Input 1 Texture", surface_format,
                        ),
                        width,
                        height,
                    });
                }
                
                // Verify texture size matches data size
                let expected_bytes = (width * height * 4) as usize;
                if frame_data.len() != expected_bytes {
                    log::error!("[SIMPLE_ENGINE] SIZE MISMATCH! Texture: {}x{} ({} bytes expected), Data: {} bytes",
                        width, height, expected_bytes, frame_data.len());
                }
                
                // Upload data
                if let Some(ref input_tex) = self.input1_texture {
                    log::debug!("[SIMPLE_ENGINE] Writing {} bytes to texture", frame_data.len());
                    self.queue.write_texture(
                        wgpu::TexelCopyTextureInfo {
                            texture: &input_tex.texture.texture,
                            mip_level: 0,
                            origin: wgpu::Origin3d::ZERO,
                            aspect: wgpu::TextureAspect::All,
                        },
                        &frame_data,
                        wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(4 * width),
                            rows_per_image: Some(height),
                        },
                        wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                    );
                    log::debug!("[SIMPLE_ENGINE] Texture upload complete");
                }
            } else {
                log::warn!("[SIMPLE_ENGINE] Has new frame but no frame data");
            }
        }
        
        // Upload input 2 frame if available
        if self.input_manager.input2_has_new_frame() {
            if let Some(frame_data) = self.input_manager.take_input2_frame() {
                let (width, height) = self.input_manager.get_input2_resolution();
                
                // Create or resize texture if needed
                let current_size = self.input2_texture.as_ref()
                    .map(|t| (t.width, t.height));
                let needs_create = current_size.map(|(w, h)| w != width || h != height).unwrap_or(true);
                
                if needs_create {
                    log::info!("[SIMPLE_ENGINE] Creating input 2 texture: {}x{} (current: {:?})", 
                        width, height, current_size);
                    let surface_format = self.config.format;
                    self.input2_texture = Some(SimpleInputTexture {
                        texture: Texture::create_render_target_with_format(
                            &self.device, width, height, "Input 2 Texture", surface_format,
                        ),
                        width,
                        height,
                    });
                }
                
                // Upload data
                if let Some(ref input_tex) = self.input2_texture {
                    self.queue.write_texture(
                        wgpu::TexelCopyTextureInfo {
                            texture: &input_tex.texture.texture,
                            mip_level: 0,
                            origin: wgpu::Origin3d::ZERO,
                            aspect: wgpu::TextureAspect::All,
                        },
                        &frame_data,
                        wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(4 * width),
                            rows_per_image: Some(height),
                        },
                        wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                    );
                }
            }
        }
        
        // GPU Syphon path (macOS only): the texture is already updated in-place
        // by SyphonWgpuInput.receive_texture() during InputSource::update().
        // No copy needed — the view is used directly in render().
    }
    
    fn get_input_view(&self) -> &wgpu::TextureView {
        self.input1_texture
            .as_ref()
            .map(|t| &t.texture.view)
            .unwrap_or(&self.dummy_black_texture.view) // Fallback to black if no input
    }
    
    fn get_feedback_read_view(&self) -> &wgpu::TextureView {
        if self.config_settings.use_ping_pong {
            if self.current_feedback_idx == 0 {
                &self.feedback_texture_a.view
            } else {
                &self.feedback_texture_b.as_ref().unwrap().view
            }
        } else {
            &self.feedback_texture_a.view
        }
    }
    
    fn get_feedback_write_view(&self) -> &wgpu::TextureView {
        if self.config_settings.use_ping_pong {
            if self.current_feedback_idx == 0 {
                &self.feedback_texture_b.as_ref().unwrap().view
            } else {
                &self.feedback_texture_a.view
            }
        } else {
            &self.output_texture.view
        }
    }
    
    fn get_params_from_shared_state(&self, hue_lfo: f32, rotate_lfo: f32, zoom_lfo: f32) -> SimpleFeedbackUniforms {
        // Default params
        let mut params = SimpleFeedbackUniforms::default();
        
        // Set resolution
        params.width = self.config_settings.output_width as f32;
        params.height = self.config_settings.output_height as f32;
        params.inv_width = 1.0 / params.width;
        params.inv_height = 1.0 / params.height;
        
        // Apply hue amount override from UI if set
        if let Some(hue_amount) = self.hue_amount_override {
            params.hue_amount = hue_amount;
        }
        
        // Set LFO values
        params.hue_lfo = hue_lfo;
        params.rotate_lfo = rotate_lfo;
        params.zoom_lfo = zoom_lfo;
        
        // Get mixing and keying parameters from shared state (Block 1 CH2 mix settings)
        if let Ok(state) = self.shared_state.lock() {
            params.mix_amount = state.block1.ch2_mix_amount;
            params.mix_type = state.block1.ch2_mix_type;
            params.key_threshold = state.block1.ch2_key_threshold;
            params.key_softness = state.block1.ch2_key_soft;
            // Key type: 0=Lumakey, 1=Chromakey
            params.key_type = state.block1.ch2_key_mode;
            params.key_order = state.block1.ch2_key_order;
        }
        
        params
    }
    
    fn do_render_pass(&self, output_view: &wgpu::TextureView) {
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Simple Feedback Render"),
        });
        
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Feedback Pass"),
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
            
            render_pass.set_pipeline(&self.feedback_pipeline.pipeline);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_bind_group(0, &self.feedback_pipeline.bind_group, &[]);
            render_pass.draw(0..6, 0..1);
        }
        
        self.queue.submit(std::iter::once(encoder.finish()));
    }
    
    fn present_output(&self) {
        // Get the texture to present (the one we just rendered to)
        let source_view = if self.config_settings.use_ping_pong {
            // In ping-pong mode, we need to read from the texture we just wrote to
            // current_feedback_idx was already swapped, so we need the opposite logic
            if self.current_feedback_idx == 0 {
                // After swap, idx is 0, meaning we wrote to A (and will read from B next frame)
                // But we want to display what we JUST wrote, which is A
                &self.feedback_texture_a.view
            } else {
                // After swap, idx is 1, meaning we wrote to B
                &self.feedback_texture_b.as_ref().unwrap().view
            }
        } else {
            // In single texture mode, we rendered to output_texture
            &self.output_texture.view
        };
        
        let output = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(_) => {
                self.surface.configure(&self.device, &self.config);
                return;
            }
        };
        
        let surface_view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        // Create blit bind group for presenting
        let blit_bind_group = self.feedback_pipeline.create_blit_bind_group(&self.device, source_view);
        
        // Blit the source texture to the surface using a render pass
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Present Output"),
        });
        
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Blit to Surface"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &surface_view,
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
            
            render_pass.set_pipeline(&self.feedback_pipeline.blit_pipeline);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_bind_group(0, &blit_bind_group, &[]);
            render_pass.draw(0..6, 0..1);
        }
        
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
    }
    
    /// Set the hue amount override from UI
    pub fn set_hue_amount(&mut self, amount: f32) {
        self.hue_amount_override = Some(amount);
    }
    
    /// Set the BPM for tempo-synced LFOs
    pub fn set_bpm(&mut self, bpm: f32) {
        self.lfo_bank.set_bpm(bpm);
        log::info!("[SIMPLE_ENGINE] BPM set to {:.1}", bpm);
    }
    
    /// Reset LFO phases to 0
    pub fn reset_lfo_phase(&mut self) {
        self.lfo_bank.reset_all();
        log::debug!("[SIMPLE_ENGINE] LFO phases reset");
    }
    
    /// Get reference to the device (for sharing with control window)
    pub fn device(&self) -> Arc<wgpu::Device> {
        Arc::clone(&self.device)
    }
    
    /// Get reference to the queue (for sharing with control window)
    pub fn queue(&self) -> Arc<wgpu::Queue> {
        Arc::clone(&self.queue)
    }
    
    /// Get reference to the instance (for creating control window surface)
    pub fn instance(&self) -> &wgpu::Instance {
        &self.instance
    }
}
