//! # ImGui Renderer Module
//!
//! Handles rendering the ImGui interface to the control window using wgpu.
//! Uses a shared device/queue from the output window to avoid creating multiple devices.

use imgui::{Context, TextureId};
use imgui_wgpu::{Renderer, RendererConfig, Texture};
use std::sync::Arc;
use winit::window::Window;

/// ImGui renderer for the control window
pub struct ImGuiRenderer {
    context: Context,
    renderer: Renderer,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
}

impl ImGuiRenderer {
    /// Create a new ImGui renderer using a shared device/queue
    /// 
    /// # Arguments
    /// * `instance` - wgpu instance
    /// * `device` - Shared wgpu device
    /// * `queue` - Shared wgpu queue
    /// * `window` - Window to render to
    /// * `ui_scale` - UI scaling factor (1.0 = default, 0.5 = half size, 2.0 = double size)
    pub async fn new(
        instance: &wgpu::Instance,
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        window: Arc<Window>,
        ui_scale: f32,
    ) -> anyhow::Result<Self> {
        let surface = instance.create_surface(Arc::clone(&window))?;
        
        // Get surface capabilities - we need an adapter for this
        // Use the first available adapter
        let adapters = instance.enumerate_adapters(wgpu::Backends::all());
        let surface_caps = if let Some(adapter) = adapters.first() {
            surface.get_capabilities(adapter)
        } else {
            anyhow::bail!("No adapters available for ImGui renderer");
        };
        let surface_format = surface_caps.formats[0];
        
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: window.inner_size().width.max(1),
            height: window.inner_size().height.max(1),
            // The control window shares the same device/queue as the output window.
            // Running both surfaces with VSync can throttle the whole app to the
            // slowest presentation cadence, so keep the UI surface non-vsynced.
            present_mode: wgpu::PresentMode::AutoNoVsync,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 1,
        };
        surface.configure(&device, &surface_config);
        
        // Create imgui context
        let mut context = Context::create();
        context.set_ini_filename(None); // Don't save window positions
        
        // Apply UI scaling
        let ui_scale = ui_scale.clamp(0.5, 2.0); // Clamp to reasonable range
        {
            let io = context.io_mut();
            io.font_global_scale = ui_scale;
        }
        
        // Configure ImGui style to match OF version
        let style = context.style_mut();
        style.window_rounding = 4.0;
        style.frame_rounding = 2.0;
        style.scrollbar_rounding = 2.0;
        style.grab_rounding = 2.0;
        style.window_border_size = 1.0;
        style.frame_border_size = 0.0;
        
        // Dark theme with colored accents
        use imgui::StyleColor;
        style.colors[StyleColor::WindowBg as usize] = [0.10, 0.10, 0.10, 1.0];
        style.colors[StyleColor::TitleBg as usize] = [0.15, 0.15, 0.15, 1.0];
        style.colors[StyleColor::TitleBgActive as usize] = [0.20, 0.20, 0.25, 1.0];
        style.colors[StyleColor::FrameBg as usize] = [0.20, 0.20, 0.20, 1.0];
        style.colors[StyleColor::FrameBgHovered as usize] = [0.25, 0.25, 0.25, 1.0];
        style.colors[StyleColor::FrameBgActive as usize] = [0.30, 0.30, 0.35, 1.0];
        style.colors[StyleColor::Button as usize] = [0.25, 0.25, 0.30, 1.0];
        style.colors[StyleColor::ButtonHovered as usize] = [0.35, 0.35, 0.40, 1.0];
        style.colors[StyleColor::ButtonActive as usize] = [0.40, 0.40, 0.50, 1.0];
        style.colors[StyleColor::Header as usize] = [0.30, 0.30, 0.35, 1.0];
        style.colors[StyleColor::HeaderHovered as usize] = [0.40, 0.40, 0.50, 1.0];
        style.colors[StyleColor::HeaderActive as usize] = [0.45, 0.45, 0.55, 1.0];
        style.colors[StyleColor::SliderGrab as usize] = [0.50, 0.50, 0.60, 1.0];
        style.colors[StyleColor::SliderGrabActive as usize] = [0.60, 0.60, 0.70, 1.0];
        
        // Create renderer
        let renderer_config = RendererConfig {
            texture_format: surface_format,
            ..Default::default()
        };
        let renderer = Renderer::new(&mut context, &device, &queue, renderer_config);
        
        Ok(Self {
            context,
            renderer,
            device,
            queue,
            surface,
            surface_config,
        })
    }
    
    /// Resize the surface
    pub fn resize(&mut self, width: u32, height: u32) {
        self.surface_config.width = width.max(1);
        self.surface_config.height = height.max(1);
        self.surface.configure(&self.device, &self.surface_config);
    }
    
    /// Update UI scale at runtime
    pub fn set_ui_scale(&mut self, scale: f32) {
        let scale = scale.clamp(0.5, 2.0);
        let io = self.context.io_mut();
        io.font_global_scale = scale;
    }
    
    /// Get current UI scale
    pub fn ui_scale(&self) -> f32 {
        self.context.io().font_global_scale
    }
    
    /// Register an external wgpu texture with ImGui
    /// Returns a TextureId that can be used with imgui::Image
    pub fn register_external_texture(
        &mut self,
        texture: Arc<wgpu::Texture>,
        view: Arc<wgpu::TextureView>,
        width: u32,
        height: u32,
    ) -> TextureId {
        use imgui_wgpu::RawTextureConfig;
        
        let size = wgpu::Extent3d { width, height, depth_or_array_layers: 1 };
        
        // Create sampler config for bind group creation
        let sampler_desc = wgpu::SamplerDescriptor {
            label: Some("imgui preview sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            lod_min_clamp: 0.0,
            lod_max_clamp: 100.0,
            compare: None,
            anisotropy_clamp: 1,
            border_color: None,
        };
        
        let raw_config = RawTextureConfig {
            label: Some("preview texture bind group"),
            sampler_desc,
        };
        
        let imgui_texture = Texture::from_raw_parts(
            &self.device,
            &self.renderer,
            texture,
            view,
            None, // Create bind group automatically
            Some(&raw_config),
            size,
        );
        
        // Insert into renderer's texture map
        self.renderer.textures.insert(imgui_texture)
    }
    
    /// Remove a previously registered texture
    pub fn unregister_texture(&mut self, texture_id: TextureId) {
        self.renderer.textures.remove(texture_id);
    }
    
    /// Render a frame with the given UI builder function
    pub fn render_frame<F>(&mut self, build_ui: F) -> anyhow::Result<()>
    where
        F: FnOnce(&mut imgui::Ui),
    {
        // Update display size
        let surface_texture = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(_) => {
                self.surface.configure(&self.device, &self.surface_config);
                return Ok(());
            }
        };
        
        // Build UI frame
        let ui = self.context.new_frame();
        build_ui(ui);
        let draw_data = self.context.render();
        
        let view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ImGui Encoder"),
        });
        
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ImGui Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.1,
                            b: 0.1,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            
            let _ = self.renderer
                .render(draw_data, &self.queue, &self.device, &mut render_pass);
        }
        
        self.queue.submit(std::iter::once(encoder.finish()));
        surface_texture.present();
        
        Ok(())
    }
    
    /// Update display size in IO
    pub fn set_display_size(&mut self, width: f32, height: f32) {
        self.context.io_mut().display_size = [width, height];
    }
    
    /// Handle a winit window event
    pub fn handle_event(&mut self, event: &winit::event::WindowEvent, _window: &Window) {
        use winit::event::*;
        
        let io = self.context.io_mut();
        
        match event {
            WindowEvent::CursorMoved { position, .. } => {
                io.mouse_pos = [position.x as f32, position.y as f32];
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let pressed = *state == ElementState::Pressed;
                match button {
                    MouseButton::Left => io.mouse_down[0] = pressed,
                    MouseButton::Right => io.mouse_down[1] = pressed,
                    MouseButton::Middle => io.mouse_down[2] = pressed,
                    _ => {}
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                match delta {
                    MouseScrollDelta::LineDelta(x, y) => {
                        io.mouse_wheel_h = *x;
                        io.mouse_wheel = *y;
                    }
                    MouseScrollDelta::PixelDelta(pos) => {
                        io.mouse_wheel_h = pos.x as f32 / 100.0;
                        io.mouse_wheel = pos.y as f32 / 100.0;
                    }
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                let pressed = event.state == ElementState::Pressed;
                
                // Handle modifiers
                match &event.logical_key {
                    winit::keyboard::Key::Named(winit::keyboard::NamedKey::Shift) => io.key_shift = pressed,
                    winit::keyboard::Key::Named(winit::keyboard::NamedKey::Control) => io.key_ctrl = pressed,
                    winit::keyboard::Key::Named(winit::keyboard::NamedKey::Alt) => io.key_alt = pressed,
                    winit::keyboard::Key::Named(winit::keyboard::NamedKey::Super) => io.key_super = pressed,
                    _ => {}
                }
                
                // Handle text input
                if pressed {
                    if let winit::keyboard::Key::Character(c) = &event.logical_key {
                        for ch in c.chars() {
                            io.add_input_character(ch);
                        }
                    }
                }
            }
            _ => {}
        }
    }
}
