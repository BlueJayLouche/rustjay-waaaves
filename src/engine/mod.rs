//! # Engine Module
//!
//! The core rendering engine using wgpu for cross-platform GPU acceleration.
//! Supports dual-window architecture: output window + control window with imgui.

use crate::audio::AudioInput;
use crate::config::AppConfig;
use crate::core::lfo_engine::{update_lfo_phases, apply_lfos_to_block1, apply_lfos_to_block2, apply_lfos_to_block3};
use crate::midi::MidiInputHandler;
use crate::params::preset::{apply_audio_modulations, ParamModulationData};
use std::collections::HashMap;
use crate::core::{OutputMode, SharedState, Vertex};
use crate::engine::imgui_renderer::ImGuiRenderer;

use crate::engine::blocks::{ModularBlock1, ModularBlock2, ModularBlock3};

use crate::engine::texture::{ReadbackLayout, Texture, strip_readback_padding};
use crate::gui::ControlGui;
use crate::input::{InputManager, InputTextureManager};
// NDI output is managed through AsyncNdiOutput
use anyhow::Result;
use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

pub mod imgui_renderer;
pub mod pipelines;
pub mod preview;
pub mod texture;
pub mod simple_feedback;
pub mod lfo_tempo;
pub mod simple_engine;
pub mod stages;
pub mod blocks;

/// Main application runner
pub fn run_app(
    config: AppConfig,
    shared_state: Arc<std::sync::Mutex<SharedState>>,
) -> Result<()> {
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    
    let mut app = App::new(config, shared_state);
    event_loop.run_app(&mut app)?;
    
    Ok(())
}

/// Application state managing all windows
struct App {
    config: AppConfig,
    shared_state: Arc<std::sync::Mutex<SharedState>>,
    
    // Shared wgpu resources
    wgpu_instance: Option<wgpu::Instance>,
    wgpu_adapter: Option<wgpu::Adapter>,
    wgpu_device: Option<Arc<wgpu::Device>>,
    wgpu_queue: Option<Arc<wgpu::Queue>>,
    
    // Output window (main display)
    output_window: Option<Arc<Window>>,
    output_engine: Option<WgpuEngine>,
    output_fullscreen: bool,  // Track fullscreen state
    output_occluded: bool,
    
    // Frame rate limiting for output window
    output_last_frame_time: Option<std::time::Instant>,
    output_target_frame_duration: std::time::Duration,
    output_next_frame_time: std::time::Instant,
    output_vsync_enabled: bool,
    
    // Control window (imgui)
    control_window: Option<Arc<Window>>,
    control_gui: Option<ControlGui>,
    imgui_renderer: Option<ImGuiRenderer>,
    control_needs_redraw: bool,
    control_active_until: Option<std::time::Instant>,
    control_last_frame_time: Option<std::time::Instant>,
    control_target_frame_duration: std::time::Duration,
    
    // Audio input
    audio_input: Option<AudioInput>,
    
    // Video input
    video_input: Option<InputManager>,
    
    // MIDI input handler
    midi_input: Option<MidiInputHandler>,
    
    // Keyboard modifier state tracking
    shift_pressed: bool,
}

impl App {
    fn new(config: AppConfig, shared_state: Arc<std::sync::Mutex<SharedState>>) -> Self {
        // Initialize audio input (wrapped in catch_unwind to prevent panic)
        let audio_input = if config.audio.enable_input {
            match std::panic::catch_unwind(|| {
                let mut audio = AudioInput::new(config.audio.fft_size);
                if let Err(e) = audio.initialize() {
                    log::error!("Failed to initialize audio input: {:?}", e);
                    None
                } else {
                    log::info!("Audio input initialized successfully");
                    Some(audio)
                }
            }) {
                Ok(result) => result,
                Err(_) => {
                    log::error!("Audio initialization panicked");
                    None
                }
            }
        } else {
            log::info!("Audio input disabled in config");
            None
        };
        
        // Video input will be initialized after wgpu device is created
        let video_input: Option<InputManager> = None;
        
        let target_fps = config.output_window.fps.max(1);
        let control_fps = config.control_window.fps.max(1);
        let now = std::time::Instant::now();
        let vsync = config.output_window.vsync;
        
        Self {
            config,
            shared_state,
            wgpu_instance: None,
            wgpu_adapter: None,
            wgpu_device: None,
            wgpu_queue: None,
            output_window: None,
            output_engine: None,
            output_fullscreen: false,
            output_occluded: false,
            output_last_frame_time: None,
            output_target_frame_duration: std::time::Duration::from_secs_f64(1.0 / target_fps as f64),
            output_next_frame_time: now,
            output_vsync_enabled: vsync,
            control_window: None,
            control_gui: None,
            imgui_renderer: None,
            control_needs_redraw: true,
            control_active_until: None,
            control_last_frame_time: None,
            control_target_frame_duration: std::time::Duration::from_secs_f64(1.0 / control_fps.max(1) as f64),
            audio_input,
            video_input,
            shift_pressed: false,
            midi_input: None,
        }
    }
    
    /// Render the preview window with modular Block 1 debug output
    /// Toggle fullscreen mode for the output window
    fn toggle_fullscreen(&mut self) {
        if let Some(ref output_window) = self.output_window {
            self.output_fullscreen = !self.output_fullscreen;
            
            let fullscreen_mode = if self.output_fullscreen {
                // Use borderless fullscreen on the current monitor
                Some(winit::window::Fullscreen::Borderless(None))
            } else {
                None
            };
            
            output_window.set_fullscreen(fullscreen_mode);
            log::info!("Output window fullscreen: {}", self.output_fullscreen);
            
            // Show a brief notification (could add on-screen display here)
            if let Some(ref mut gui) = self.control_gui {
                let msg = if self.output_fullscreen { "Fullscreen ON" } else { "Fullscreen OFF" };
                gui.show_status(msg);
            }
        }
    }
    
    /// Trigger tap tempo from keyboard shortcut
    fn trigger_tap_tempo(&mut self) {
        if let Some(ref mut gui) = self.control_gui {
            // Call the existing tap tempo handler
            gui.trigger_tap_tempo();
            log::debug!("Tap tempo triggered from keyboard");
        }
    }
    
    /// Toggle recording from keyboard shortcut (Shift+R)
    fn toggle_recording(&mut self) {
        // Send toggle command to engine through shared state
        if let Ok(mut state) = self.shared_state.lock() {
            state.recording_command = crate::core::RecordingCommand::Toggle;
            log::info!("Recording toggle requested (Shift+R)");
        }
    }
    
    /// Apply a MIDI value to a parameter based on its ID
    fn apply_midi_value_to_param(state: &mut SharedState, param_id: &str, value: f32) {
        // Helper macro to set parameter values
        macro_rules! set_param {
            ($params:expr, $field:ident) => {
                $params.$field = value
            };
        }
        
        // Helper to set Vec3 components
        macro_rules! set_vec3 {
            ($params:expr, $base:ident, $comp:ident, $x:ident, $y:ident, $z:ident) => {
                $params.$base = glam::Vec3::new($params.$x, $params.$y, $params.$z)
            };
        }
        
        // Parse the parameter ID and set the value
        match param_id {
            // ===========================================
            // BLOCK 1 - CHANNEL 1
            // ===========================================
            // Geometry
            "block1.ch1_x_displace" => set_param!(state.block1, ch1_x_displace),
            "block1.ch1_y_displace" => set_param!(state.block1, ch1_y_displace),
            "block1.ch1_z_displace" => set_param!(state.block1, ch1_z_displace),
            "block1.ch1_rotate" => set_param!(state.block1, ch1_rotate),
            "block1.ch1_kaleidoscope_amount" => set_param!(state.block1, ch1_kaleidoscope_amount),
            "block1.ch1_kaleidoscope_slice" => set_param!(state.block1, ch1_kaleidoscope_slice),
            
            // Color - HSB Attenuate (individual components)
            "block1.ch1_hsb_attenuate_x" => { state.block1.ch1_hsb_attenuate_x = value; state.block1.ch1_hsb_attenuate.x = value; }
            "block1.ch1_hsb_attenuate_y" => { state.block1.ch1_hsb_attenuate_y = value; state.block1.ch1_hsb_attenuate.y = value; }
            "block1.ch1_hsb_attenuate_z" => { state.block1.ch1_hsb_attenuate_z = value; state.block1.ch1_hsb_attenuate.z = value; }
            
            // Posterize
            "block1.ch1_posterize" => set_param!(state.block1, ch1_posterize),
            
            // Filters
            "block1.ch1_blur_amount" => set_param!(state.block1, ch1_blur_amount),
            "block1.ch1_blur_radius" => set_param!(state.block1, ch1_blur_radius),
            "block1.ch1_sharpen_amount" => set_param!(state.block1, ch1_sharpen_amount),
            "block1.ch1_sharpen_radius" => set_param!(state.block1, ch1_sharpen_radius),
            "block1.ch1_filters_boost" => set_param!(state.block1, ch1_filters_boost),
            
            // ===========================================
            // BLOCK 1 - CHANNEL 2
            // ===========================================
            // Mix
            "block1.ch2_mix_amount" => set_param!(state.block1, ch2_mix_amount),
            
            // Key
            "block1.ch2_key_value_red" => set_param!(state.block1, ch2_key_value_red),
            "block1.ch2_key_value_green" => set_param!(state.block1, ch2_key_value_green),
            "block1.ch2_key_value_blue" => set_param!(state.block1, ch2_key_value_blue),
            "block1.ch2_key_threshold" => set_param!(state.block1, ch2_key_threshold),
            "block1.ch2_key_soft" => set_param!(state.block1, ch2_key_soft),
            
            // Geometry
            "block1.ch2_x_displace" => set_param!(state.block1, ch2_x_displace),
            "block1.ch2_y_displace" => set_param!(state.block1, ch2_y_displace),
            "block1.ch2_z_displace" => set_param!(state.block1, ch2_z_displace),
            "block1.ch2_rotate" => set_param!(state.block1, ch2_rotate),
            "block1.ch2_kaleidoscope_amount" => set_param!(state.block1, ch2_kaleidoscope_amount),
            "block1.ch2_kaleidoscope_slice" => set_param!(state.block1, ch2_kaleidoscope_slice),
            
            // Color - HSB Attenuate
            "block1.ch2_hsb_attenuate_x" => { state.block1.ch2_hsb_attenuate_x = value; state.block1.ch2_hsb_attenuate.x = value; }
            "block1.ch2_hsb_attenuate_y" => { state.block1.ch2_hsb_attenuate_y = value; state.block1.ch2_hsb_attenuate.y = value; }
            "block1.ch2_hsb_attenuate_z" => { state.block1.ch2_hsb_attenuate_z = value; state.block1.ch2_hsb_attenuate.z = value; }
            
            // Posterize
            "block1.ch2_posterize" => set_param!(state.block1, ch2_posterize),
            
            // Filters
            "block1.ch2_blur_amount" => set_param!(state.block1, ch2_blur_amount),
            "block1.ch2_blur_radius" => set_param!(state.block1, ch2_blur_radius),
            "block1.ch2_sharpen_amount" => set_param!(state.block1, ch2_sharpen_amount),
            "block1.ch2_sharpen_radius" => set_param!(state.block1, ch2_sharpen_radius),
            "block1.ch2_filters_boost" => set_param!(state.block1, ch2_filters_boost),
            
            // ===========================================
            // BLOCK 1 - FB1
            // ===========================================
            // Mix
            "block1.fb1_mix_amount" => set_param!(state.block1, fb1_mix_amount),
            
            // Key
            "block1.fb1_key_value_red" => set_param!(state.block1, fb1_key_value_red),
            "block1.fb1_key_value_green" => set_param!(state.block1, fb1_key_value_green),
            "block1.fb1_key_value_blue" => set_param!(state.block1, fb1_key_value_blue),
            "block1.fb1_key_threshold" => set_param!(state.block1, fb1_key_threshold),
            "block1.fb1_key_soft" => set_param!(state.block1, fb1_key_soft),
            
            // Geometry
            "block1.fb1_x_displace" => set_param!(state.block1, fb1_x_displace),
            "block1.fb1_y_displace" => set_param!(state.block1, fb1_y_displace),
            "block1.fb1_z_displace" => set_param!(state.block1, fb1_z_displace),
            "block1.fb1_rotate" => set_param!(state.block1, fb1_rotate),
            "block1.fb1_shear_matrix_x" => { state.block1.fb1_shear_matrix_x = value; state.block1.fb1_shear_matrix.x = value; }
            "block1.fb1_shear_matrix_y" => { state.block1.fb1_shear_matrix_y = value; state.block1.fb1_shear_matrix.y = value; }
            "block1.fb1_shear_matrix_z" => { state.block1.fb1_shear_matrix_z = value; state.block1.fb1_shear_matrix.z = value; }
            "block1.fb1_shear_matrix_w" => { state.block1.fb1_shear_matrix_w = value; state.block1.fb1_shear_matrix.w = value; }
            "block1.fb1_kaleidoscope_amount" => set_param!(state.block1, fb1_kaleidoscope_amount),
            "block1.fb1_kaleidoscope_slice" => set_param!(state.block1, fb1_kaleidoscope_slice),
            
            // Color - HSB Offset
            "block1.fb1_hsb_offset_x" => { state.block1.fb1_hsb_offset_x = value; state.block1.fb1_hsb_offset.x = value; }
            "block1.fb1_hsb_offset_y" => { state.block1.fb1_hsb_offset_y = value; state.block1.fb1_hsb_offset.y = value; }
            "block1.fb1_hsb_offset_z" => { state.block1.fb1_hsb_offset_z = value; state.block1.fb1_hsb_offset.z = value; }
            
            // Color - HSB Attenuate
            "block1.fb1_hsb_attenuate_x" => { state.block1.fb1_hsb_attenuate_x = value; state.block1.fb1_hsb_attenuate.x = value; }
            "block1.fb1_hsb_attenuate_y" => { state.block1.fb1_hsb_attenuate_y = value; state.block1.fb1_hsb_attenuate.y = value; }
            "block1.fb1_hsb_attenuate_z" => { state.block1.fb1_hsb_attenuate_z = value; state.block1.fb1_hsb_attenuate.z = value; }
            
            // Color - HSB PowMap
            "block1.fb1_hsb_powmap_x" => { state.block1.fb1_hsb_powmap_x = value; state.block1.fb1_hsb_powmap.x = value; }
            "block1.fb1_hsb_powmap_y" => { state.block1.fb1_hsb_powmap_y = value; state.block1.fb1_hsb_powmap.y = value; }
            "block1.fb1_hsb_powmap_z" => { state.block1.fb1_hsb_powmap_z = value; state.block1.fb1_hsb_powmap.z = value; }
            
            // Color - Other
            "block1.fb1_hue_shaper" => set_param!(state.block1, fb1_hue_shaper),
            
            // Filters
            "block1.fb1_blur_amount" => set_param!(state.block1, fb1_blur_amount),
            "block1.fb1_blur_radius" => set_param!(state.block1, fb1_blur_radius),
            "block1.fb1_sharpen_amount" => set_param!(state.block1, fb1_sharpen_amount),
            "block1.fb1_sharpen_radius" => set_param!(state.block1, fb1_sharpen_radius),
            "block1.fb1_temporal_filter1_amount" => set_param!(state.block1, fb1_temporal_filter1_amount),
            "block1.fb1_temporal_filter1_resonance" => set_param!(state.block1, fb1_temporal_filter1_resonance),
            "block1.fb1_temporal_filter2_amount" => set_param!(state.block1, fb1_temporal_filter2_amount),
            "block1.fb1_temporal_filter2_resonance" => set_param!(state.block1, fb1_temporal_filter2_resonance),
            "block1.fb1_filters_boost" => set_param!(state.block1, fb1_filters_boost),
            
            // Delay
            "block1.fb1_delay_time" => state.block1.fb1_delay_time = value as i32,
            
            // ===========================================
            // BLOCK 2 - INPUT
            // ===========================================
            "block2.input_x_displace" => set_param!(state.block2, block2_input_x_displace),
            "block2.input_y_displace" => set_param!(state.block2, block2_input_y_displace),
            "block2.input_z_displace" => set_param!(state.block2, block2_input_z_displace),
            "block2.input_rotate" => set_param!(state.block2, block2_input_rotate),
            "block2.input_kaleidoscope_amount" => set_param!(state.block2, block2_input_kaleidoscope_amount),
            "block2.input_kaleidoscope_slice" => set_param!(state.block2, block2_input_kaleidoscope_slice),
            
            // HSB
            "block2.input_hsb_attenuate_x" => { state.block2.block2_input_hsb_attenuate_x = value; state.block2.block2_input_hsb_attenuate.x = value; }
            "block2.input_hsb_attenuate_y" => { state.block2.block2_input_hsb_attenuate_y = value; state.block2.block2_input_hsb_attenuate.y = value; }
            "block2.input_hsb_attenuate_z" => { state.block2.block2_input_hsb_attenuate_z = value; state.block2.block2_input_hsb_attenuate.z = value; }
            
            // Filters
            "block2.input_blur_amount" => set_param!(state.block2, block2_input_blur_amount),
            "block2.input_blur_radius" => set_param!(state.block2, block2_input_blur_radius),
            "block2.input_sharpen_amount" => set_param!(state.block2, block2_input_sharpen_amount),
            "block2.input_sharpen_radius" => set_param!(state.block2, block2_input_sharpen_radius),
            "block2.input_filters_boost" => set_param!(state.block2, block2_input_filters_boost),
            
            // ===========================================
            // BLOCK 2 - FB2
            // ===========================================
            "block2.fb2_mix_amount" => set_param!(state.block2, fb2_mix_amount),
            "block2.fb2_key_threshold" => set_param!(state.block2, fb2_key_threshold),
            "block2.fb2_key_soft" => set_param!(state.block2, fb2_key_soft),
            "block2.fb2_key_value_red" => set_param!(state.block2, fb2_key_value_red),
            "block2.fb2_key_value_green" => set_param!(state.block2, fb2_key_value_green),
            "block2.fb2_key_value_blue" => set_param!(state.block2, fb2_key_value_blue),
            
            "block2.fb2_x_displace" => set_param!(state.block2, fb2_x_displace),
            "block2.fb2_y_displace" => set_param!(state.block2, fb2_y_displace),
            "block2.fb2_z_displace" => set_param!(state.block2, fb2_z_displace),
            "block2.fb2_rotate" => set_param!(state.block2, fb2_rotate),
            "block2.fb2_shear_matrix_x" => { state.block2.fb2_shear_matrix_x = value; state.block2.fb2_shear_matrix.x = value; }
            "block2.fb2_shear_matrix_y" => { state.block2.fb2_shear_matrix_y = value; state.block2.fb2_shear_matrix.y = value; }
            "block2.fb2_shear_matrix_z" => { state.block2.fb2_shear_matrix_z = value; state.block2.fb2_shear_matrix.z = value; }
            "block2.fb2_shear_matrix_w" => { state.block2.fb2_shear_matrix_w = value; state.block2.fb2_shear_matrix.w = value; }
            "block2.fb2_kaleidoscope_amount" => set_param!(state.block2, fb2_kaleidoscope_amount),
            "block2.fb2_kaleidoscope_slice" => set_param!(state.block2, fb2_kaleidoscope_slice),
            
            // HSB
            "block2.fb2_hsb_offset_x" => { state.block2.fb2_hsb_offset_x = value; state.block2.fb2_hsb_offset.x = value; }
            "block2.fb2_hsb_offset_y" => { state.block2.fb2_hsb_offset_y = value; state.block2.fb2_hsb_offset.y = value; }
            "block2.fb2_hsb_offset_z" => { state.block2.fb2_hsb_offset_z = value; state.block2.fb2_hsb_offset.z = value; }
            "block2.fb2_hsb_attenuate_x" => { state.block2.fb2_hsb_attenuate_x = value; state.block2.fb2_hsb_attenuate.x = value; }
            "block2.fb2_hsb_attenuate_y" => { state.block2.fb2_hsb_attenuate_y = value; state.block2.fb2_hsb_attenuate.y = value; }
            "block2.fb2_hsb_attenuate_z" => { state.block2.fb2_hsb_attenuate_z = value; state.block2.fb2_hsb_attenuate.z = value; }
            "block2.fb2_hsb_powmap_x" => { state.block2.fb2_hsb_powmap_x = value; state.block2.fb2_hsb_powmap.x = value; }
            "block2.fb2_hsb_powmap_y" => { state.block2.fb2_hsb_powmap_y = value; state.block2.fb2_hsb_powmap.y = value; }
            "block2.fb2_hsb_powmap_z" => { state.block2.fb2_hsb_powmap_z = value; state.block2.fb2_hsb_powmap.z = value; }
            
            // Filters
            "block2.fb2_blur_amount" => set_param!(state.block2, fb2_blur_amount),
            "block2.fb2_blur_radius" => set_param!(state.block2, fb2_blur_radius),
            "block2.fb2_sharpen_amount" => set_param!(state.block2, fb2_sharpen_amount),
            "block2.fb2_sharpen_radius" => set_param!(state.block2, fb2_sharpen_radius),
            "block2.fb2_temporal_filter1_amount" => set_param!(state.block2, fb2_temporal_filter1_amount),
            "block2.fb2_temporal_filter1_resonance" => set_param!(state.block2, fb2_temporal_filter1_resonance),
            "block2.fb2_temporal_filter2_amount" => set_param!(state.block2, fb2_temporal_filter2_amount),
            "block2.fb2_temporal_filter2_resonance" => set_param!(state.block2, fb2_temporal_filter2_resonance),
            "block2.fb2_filters_boost" => set_param!(state.block2, fb2_filters_boost),
            "block2.fb2_delay_time" => state.block2.fb2_delay_time = value as i32,
            
            // ===========================================
            // BLOCK 3 - BLOCK 1 RE-PROCESS
            // ===========================================
            "block3.block1_x_displace" => set_param!(state.block3, block1_x_displace),
            "block3.block1_y_displace" => set_param!(state.block3, block1_y_displace),
            "block3.block1_z_displace" => set_param!(state.block3, block1_z_displace),
            "block3.block1_rotate" => set_param!(state.block3, block1_rotate),
            "block3.block1_shear_matrix_x" => { state.block3.block1_shear_matrix_x = value; state.block3.block1_shear_matrix.x = value; }
            "block3.block1_shear_matrix_y" => { state.block3.block1_shear_matrix_y = value; state.block3.block1_shear_matrix.y = value; }
            "block3.block1_shear_matrix_z" => { state.block3.block1_shear_matrix_z = value; state.block3.block1_shear_matrix.z = value; }
            "block3.block1_shear_matrix_w" => { state.block3.block1_shear_matrix_w = value; state.block3.block1_shear_matrix.w = value; }
            "block3.block1_kaleidoscope_amount" => set_param!(state.block3, block1_kaleidoscope_amount),
            "block3.block1_kaleidoscope_slice" => set_param!(state.block3, block1_kaleidoscope_slice),
            "block3.block1_blur_amount" => set_param!(state.block3, block1_blur_amount),
            "block3.block1_blur_radius" => set_param!(state.block3, block1_blur_radius),
            "block3.block1_sharpen_amount" => set_param!(state.block3, block1_sharpen_amount),
            "block3.block1_sharpen_radius" => set_param!(state.block3, block1_sharpen_radius),
            
            // ===========================================
            // BLOCK 3 - BLOCK 2 RE-PROCESS
            // ===========================================
            "block3.block2_x_displace" => set_param!(state.block3, block2_x_displace),
            "block3.block2_y_displace" => set_param!(state.block3, block2_y_displace),
            "block3.block2_z_displace" => set_param!(state.block3, block2_z_displace),
            "block3.block2_rotate" => set_param!(state.block3, block2_rotate),
            "block3.block2_shear_matrix_x" => { state.block3.block2_shear_matrix_x = value; state.block3.block2_shear_matrix.x = value; }
            "block3.block2_shear_matrix_y" => { state.block3.block2_shear_matrix_y = value; state.block3.block2_shear_matrix.y = value; }
            "block3.block2_shear_matrix_z" => { state.block3.block2_shear_matrix_z = value; state.block3.block2_shear_matrix.z = value; }
            "block3.block2_shear_matrix_w" => { state.block3.block2_shear_matrix_w = value; state.block3.block2_shear_matrix.w = value; }
            "block3.block2_kaleidoscope_amount" => set_param!(state.block3, block2_kaleidoscope_amount),
            "block3.block2_kaleidoscope_slice" => set_param!(state.block3, block2_kaleidoscope_slice),
            "block3.block2_blur_amount" => set_param!(state.block3, block2_blur_amount),
            "block3.block2_blur_radius" => set_param!(state.block3, block2_blur_radius),
            "block3.block2_sharpen_amount" => set_param!(state.block3, block2_sharpen_amount),
            "block3.block2_sharpen_radius" => set_param!(state.block3, block2_sharpen_radius),
            
            // ===========================================
            // BLOCK 3 - MATRIX MIXER
            // ===========================================
            "block3.matrix_mix_r_to_r" => set_param!(state.block3, matrix_mix_r_to_r),
            "block3.matrix_mix_r_to_g" => set_param!(state.block3, matrix_mix_r_to_g),
            "block3.matrix_mix_r_to_b" => set_param!(state.block3, matrix_mix_r_to_b),
            "block3.matrix_mix_g_to_r" => set_param!(state.block3, matrix_mix_g_to_r),
            "block3.matrix_mix_g_to_g" => set_param!(state.block3, matrix_mix_g_to_g),
            "block3.matrix_mix_g_to_b" => set_param!(state.block3, matrix_mix_g_to_b),
            "block3.matrix_mix_b_to_r" => set_param!(state.block3, matrix_mix_b_to_r),
            "block3.matrix_mix_b_to_g" => set_param!(state.block3, matrix_mix_b_to_g),
            "block3.matrix_mix_b_to_b" => set_param!(state.block3, matrix_mix_b_to_b),
            
            // ===========================================
            // BLOCK 3 - BLOCK 1 RE-PROCESS
            // ===========================================
            "block3.block1_dither" => set_param!(state.block3, block1_dither),
            
            // ===========================================
            // BLOCK 3 - BLOCK 2 RE-PROCESS
            // ===========================================
            "block3.block2_dither" => set_param!(state.block3, block2_dither),
            
            // ===========================================
            // BLOCK 3 - FINAL MIX
            // ===========================================
            "block3.final_mix_amount" => set_param!(state.block3, final_mix_amount),
            "block3.final_key_order" => state.block3.final_key_order = value as i32,
            "block3.final_key_threshold" => set_param!(state.block3, final_key_threshold),
            "block3.final_key_soft" => set_param!(state.block3, final_key_soft),
            "block3.final_dither" => set_param!(state.block3, final_dither),
            
            // ===========================================
            // GLOBAL
            // ===========================================
            "global.bpm" => state.bpm = value.clamp(20.0, 300.0),
            
            // Unknown parameter
            _ => {
                log::warn!("MIDI: Unknown parameter ID: {}", param_id);
            }
        }
    }
    
    /// Update output window target FPS
    fn set_output_fps(&mut self, fps: u32) {
        let fps = fps.max(1).min(240);
        self.output_target_frame_duration = std::time::Duration::from_secs_f64(1.0 / fps as f64);
        self.config.output_window.fps = fps;
        
        // Update shared state so GUI can see the change
        if let Ok(mut state) = self.shared_state.lock() {
            state.output_fps = fps;
        }
        
        // Also update the engine if it exists
        if let Some(ref mut engine) = self.output_engine {
            engine.set_target_fps(fps);
        }
        
        // Recalculate next frame time based on current time
        self.output_next_frame_time = std::time::Instant::now() + self.output_target_frame_duration;
        
        log::info!("Output FPS set to {} (target frame duration: {:?})", fps, self.output_target_frame_duration);
    }
    
    /// Update output window VSync
    fn set_output_vsync(&mut self, enabled: bool) {
        self.output_vsync_enabled = enabled;
        self.config.output_window.vsync = enabled;
        
        // Update shared state so GUI can see the change
        if let Ok(mut state) = self.shared_state.lock() {
            state.output_vsync = enabled;
        }
        
        // Update the engine if it exists
        if let Some(ref mut engine) = self.output_engine {
            engine.set_vsync(enabled);
        }
        
        log::info!("Output VSync {} (FPS limit: {})", 
            if enabled { "enabled" } else { "disabled" },
            if enabled { "display refresh rate".to_string() } else { format!("{} (software limited)", self.config.output_window.fps) }
        );
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // Create shared wgpu instance first
        if self.wgpu_instance.is_none() {
            self.wgpu_instance = Some(wgpu::Instance::new(&wgpu::InstanceDescriptor {
                backends: wgpu::Backends::all(),
                ..Default::default()
            }));
        }
        let Some(instance) = self.wgpu_instance.as_ref() else {
            log::error!("Failed to initialize wgpu instance");
            event_loop.exit();
            return;
        };
        
        // Create output window first
        if self.output_window.is_none() {
            let window_attrs = winit::window::WindowAttributes::default()
                .with_title(&self.config.output_window.title)
                .with_inner_size(winit::dpi::LogicalSize::new(
                    self.config.output_window.width,
                    self.config.output_window.height,
                ))
                .with_resizable(self.config.output_window.resizable)
                .with_decorations(self.config.output_window.decorated);
            
            let window = match event_loop.create_window(window_attrs) {
                Ok(window) => Arc::new(window),
                Err(err) => {
                    log::error!("Failed to create output window: {}", err);
                    event_loop.exit();
                    return;
                }
            };
            
            // Hide cursor by default for output window
            window.set_cursor_visible(false);
            
            self.output_window = Some(Arc::clone(&window));
            
            // Initialize output engine using shared instance
            let shared_state = Arc::clone(&self.shared_state);
            let config = self.config.clone();
            match pollster::block_on(WgpuEngine::new(
                instance, 
                window, 
                &config, 
                shared_state,
                &mut self.wgpu_adapter,
                &mut self.wgpu_device,
                &mut self.wgpu_queue,
            )) {
                Ok(engine) => {
                    // Initialize video input manager now that we have device/queue
                    if self.video_input.is_none() {
                        let mut input_manager = InputManager::new();
                        if let (Some(device), Some(queue)) = (&self.wgpu_device, &self.wgpu_queue) {
                            if let Err(e) = input_manager.initialize(Arc::clone(device), Arc::clone(queue)) {
                                log::error!("Failed to initialize video input: {:?}", e);
                            } else {
                                log::info!("Video input manager initialized with GPU device");
                            }
                        }
                        self.video_input = Some(input_manager);
                    }
                    self.output_engine = Some(engine);
                    
                    // Initialize MIDI input (only if enabled in config)
                    if self.config.control.midi_enabled {
                        // Wrap in catch_unwind to prevent panic from crashing the app
                        match std::panic::catch_unwind(|| {
                            MidiInputHandler::new()
                        }) {
                            Ok(Ok(mut midi)) => {
                                // Also wrap connect_all as it can panic in CoreMIDI callbacks
                                match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                    midi.connect_all()
                                })) {
                                    Ok(connected) => {
                                        if connected > 0 {
                                            log::info!("MIDI: Connected to {} device(s)", connected);
                                        } else {
                                            log::info!("MIDI: No devices found, will scan for hot-plugged devices");
                                        }
                                    }
                                    Err(_) => {
                                        log::warn!("MIDI: Panic during device connection - MIDI may be in use by another application (DAW)");
                                    }
                                }
                                self.midi_input = Some(midi);
                            }
                            Ok(Err(e)) => {
                                log::warn!("MIDI: Failed to initialize: {}", e);
                            }
                            Err(_) => {
                                log::warn!("MIDI: Panic during initialization - MIDI may be in use by another application (DAW). Disable MIDI in config or close other MIDI applications.");
                            }
                        }
                    } else {
                        log::info!("MIDI: Disabled in config");
                    }
                }
                Err(err) => {
                    eprintln!("Failed to create output engine: {}", err);
                    event_loop.exit();
                    return;
                }
            }
        }
        
        // Create control window after output window is ready (shares device/queue)
        if self.control_window.is_none() {
            if let (Some(device), Some(queue)) = (&self.wgpu_device, &self.wgpu_queue) {
                let window_attrs = winit::window::WindowAttributes::default()
                    .with_title("RUSTJAY WAAAVES - Control")
                    .with_inner_size(winit::dpi::LogicalSize::new(
                        self.config.control_window.width,
                        self.config.control_window.height,
                    ))
                    .with_resizable(true)
                    .with_decorations(true);
                
                let window = match event_loop.create_window(window_attrs) {
                    Ok(window) => Arc::new(window),
                    Err(err) => {
                        log::error!("Failed to create control window: {}", err);
                        event_loop.exit();
                        return;
                    }
                };
                self.control_window = Some(Arc::clone(&window));
                
                // Initialize ImGui renderer using shared device/queue
                let ui_scale = self.config.ui_scale;
                match pollster::block_on(ImGuiRenderer::new(
                    instance, 
                    Arc::clone(device), 
                    Arc::clone(queue), 
                    Arc::clone(&window),
                    ui_scale
                )) {
                    Ok(mut renderer) => {
                        // Initialize ControlGui with the imgui context
                        match ControlGui::new(&self.config, Arc::clone(&self.shared_state)) {
                            Ok(mut gui) => {
                                // Auto-start webcams based on saved config
                                gui.auto_start_webcams();
                                
                                // Register preview texture with imgui renderer if preview is available
                                if let Some(ref output_engine) = self.output_engine {
                                    if let Some(ref preview) = output_engine.preview_renderer {
                                        let texture_id = renderer.register_external_texture(
                                            preview.get_texture_arc(),
                                            preview.get_texture_view_arc(),
                                            320, 180
                                        );
                                        gui.set_preview_texture_id(texture_id);
                                        log::info!("Registered preview texture with ImGui: {:?}", texture_id);
                                    }
                                }
                                
                                self.control_gui = Some(gui);
                                self.imgui_renderer = Some(renderer);
                            }
                            Err(err) => {
                                eprintln!("Failed to create control GUI: {}", err);
                            }
                        }
                    }
                    Err(err) => {
                        eprintln!("Failed to create ImGui renderer: {}", err);
                    }
                }
            }
        }
    }
    
    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        // Handle output window events
        if let Some(ref output_window) = self.output_window {
            if window_id == output_window.id() {
                match event {
                    WindowEvent::CloseRequested => {
                        event_loop.exit();
                    }
                    WindowEvent::Occluded(occluded) => {
                        self.output_occluded = occluded;
                        if !occluded {
                            self.output_last_frame_time = None;
                            if let Some(ref window) = self.output_window {
                                window.request_redraw();
                            }
                        }
                    }
                    WindowEvent::CursorEntered { .. } => {
                        // Hide cursor when entering output window
                        output_window.set_cursor_visible(false);
                    }
                    WindowEvent::CursorLeft { .. } => {
                        // Show cursor when leaving output window
                        output_window.set_cursor_visible(true);
                    }
                    WindowEvent::KeyboardInput { event, .. } => {
                        // Track modifier state
                        match &event.logical_key {
                            winit::keyboard::Key::Named(winit::keyboard::NamedKey::Shift) => {
                                self.shift_pressed = event.state == winit::event::ElementState::Pressed;
                            }
                            _ => {}
                        }
                        
                        if event.state == winit::event::ElementState::Pressed {
                            match &event.logical_key {
                                winit::keyboard::Key::Named(winit::keyboard::NamedKey::Escape) => {
                                    event_loop.exit();
                                }
                                winit::keyboard::Key::Character(ch) => {
                                    let key = ch.to_lowercase();
                                    
                                    if self.shift_pressed && key == "f" {
                                        // Shift+F: Toggle fullscreen
                                        self.toggle_fullscreen();
                                    } else if self.shift_pressed && key == "t" {
                                        // Shift+T: Tap tempo
                                        self.trigger_tap_tempo();
                                    } else if self.shift_pressed && key == "r" {
                                        // Shift+R: Toggle recording
                                        self.toggle_recording();
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                    WindowEvent::Resized(size) => {
                        if let Some(ref mut engine) = self.output_engine {
                            engine.resize(size.width, size.height);
                        }
                    }
                    WindowEvent::RedrawRequested => {
                        if let Some(ref mut engine) = self.output_engine {
                            engine.render(self.output_occluded);
                        }
                    }
                    _ => {}
                }
                return;
            }
        }
        
        // Handle control window events
        if let Some(ref control_window) = self.control_window {
            if window_id == control_window.id() {
                if !matches!(event, WindowEvent::RedrawRequested) {
                    self.control_needs_redraw = true;
                    self.control_active_until = Some(
                        std::time::Instant::now() + std::time::Duration::from_millis(500)
                    );
                }

                // Pass events to ImGui
                if let Some(ref mut renderer) = self.imgui_renderer {
                    renderer.handle_event(&event, control_window);
                }
                
                match event {
                    WindowEvent::CloseRequested => {
                        // Just close the control window, keep output running
                        self.control_window = None;
                        self.control_gui = None;
                        self.imgui_renderer = None;
                    }
                    WindowEvent::KeyboardInput { event, .. } => {
                        // Track modifier state
                        match &event.logical_key {
                            winit::keyboard::Key::Named(winit::keyboard::NamedKey::Shift) => {
                                self.shift_pressed = event.state == winit::event::ElementState::Pressed;
                            }
                            _ => {}
                        }
                        
                        if event.state == winit::event::ElementState::Pressed {
                            if let winit::keyboard::Key::Character(ch) = &event.logical_key {
                                let key = ch.to_lowercase();
                                
                                if self.shift_pressed && key == "t" {
                                    // Shift+T: Tap tempo (also works in control window)
                                    self.trigger_tap_tempo();
                                }
                            }
                        }
                    }
                    WindowEvent::Resized(size) => {
                        if let Some(ref mut renderer) = self.imgui_renderer {
                            renderer.resize(size.width, size.height);
                        }
                        self.control_needs_redraw = true;
                    }
                    WindowEvent::RedrawRequested => {
                        // Render imgui
                        if let (Some(ref mut renderer), Some(ref mut gui)) = 
                            (self.imgui_renderer.as_mut(), self.control_gui.as_mut()) 
                        {
                            // Check for UI scale changes from shared state
                            if let Ok(state) = self.shared_state.lock() {
                                let desired_scale = state.ui_scale;
                                if (renderer.ui_scale() - desired_scale).abs() > 0.01 {
                                    renderer.set_ui_scale(desired_scale);
                                }
                            }
                            
                            // Update display size and render
                            let window_size = control_window.inner_size();
                            renderer.set_display_size(window_size.width as f32, window_size.height as f32);
                            
                            if let Err(err) = renderer.render_frame(|ui| gui.build_ui(ui)) {
                                eprintln!("ImGui render error: {}", err);
                            }
                        }
                        self.control_needs_redraw = false;
                        self.control_last_frame_time = Some(std::time::Instant::now());
                    }
                    _ => {}
                }
                return;
            }
        }
    }
    
    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        // Update audio input
        if let Some(ref mut audio) = self.audio_input {
            // Get processed 8-band FFT (with amplitude/smoothing/normalization applied)
            let fft_8band = audio.get_8band_fft();
            let beat_state = audio.get_beat_state();
            
            // Update shared state with audio data
            if let Ok(mut state) = self.shared_state.lock() {
                // Copy 8-band FFT to first 8 slots of shared state
                for i in 0..8.min(state.audio.fft.len()) {
                    state.audio.fft[i] = fft_8band[i];
                }
                // Fill remaining slots with zeros if needed
                for i in 8..state.audio.fft.len() {
                    state.audio.fft[i] = 0.0;
                }
                
                state.audio.volume = beat_state.energy;
                state.audio.beat = beat_state.beat;
                state.audio.bpm = beat_state.bpm;
                state.audio.beat_phase = beat_state.phase;
                
                // Sync audio processing settings from SharedState to AudioInput
                let amplitude = state.audio.amplitude;
                let smoothing = state.audio.smoothing;
                let normalization = state.audio.normalization;
                let pink_compensation = state.audio.pink_compensation;
                audio.set_amplitude(amplitude);
                audio.set_smoothing(smoothing);
                audio.set_normalization(normalization);
                audio.set_pink_compensation(pink_compensation);
            }
        }
        
        // Handle input change requests from GUI
        {
            let mut state = self.shared_state.lock().unwrap();
            
            // Handle input 1 change request
            let input1_request = std::mem::replace(&mut state.input1_change_request, crate::core::InputChangeRequest::None);
            drop(state); // Drop lock before calling into video input
            
            match input1_request {
                crate::core::InputChangeRequest::StartWebcam { device_index, width, height, fps, .. } => {
                    log::info!("[INPUT] Received StartWebcam request for input 1, device {}", device_index);
                    if let Some(ref mut engine) = self.output_engine {
                        engine.input_texture_manager.clear_input1();
                        let queue = std::sync::Arc::clone(&engine.queue);
                        engine.modular_block1.clear_all(&queue);
                        engine.modular_block1.invalidate_bind_group_caches();
                        engine.modular_block2.clear_all(&queue);
                    }
                    if let Some(ref mut video) = self.video_input {
                        log::info!("[INPUT] Processing webcam start request for input 1, device {}", device_index);
                        match video.start_input1_webcam(device_index, width, height, fps) {
                            Ok(_) => log::info!("[INPUT] Successfully started webcam input 1"),
                            Err(e) => log::error!("[INPUT] Failed to start webcam input 1: {:?}", e),
                        }
                    } else {
                        log::error!("[INPUT] Video input manager not initialized! Cannot start webcam.");
                    }
                }
                crate::core::InputChangeRequest::StartNdi { source_name, .. } => {
                    if let Some(ref mut engine) = self.output_engine {
                        engine.input_texture_manager.clear_input1();
                        let queue = std::sync::Arc::clone(&engine.queue);
                        engine.modular_block1.clear_all(&queue);
                        engine.modular_block1.invalidate_bind_group_caches();
                        engine.modular_block2.clear_all(&queue);
                    }
                    if let Some(ref mut video) = self.video_input {
                        match video.start_input1_ndi(&source_name) {
                            Ok(_) => log::info!("[INPUT] Started NDI input 1: {}", source_name),
                            Err(e) => log::error!("[INPUT] Failed to start NDI input 1: {:?}", e),
                        }
                    } else {
                        log::error!("[INPUT] Video input manager not initialized");
                    }
                }
                crate::core::InputChangeRequest::StartSyphon { server_name, .. } => {
                    if let Some(ref mut engine) = self.output_engine {
                        engine.input_texture_manager.clear_input1();
                        let queue = std::sync::Arc::clone(&engine.queue);
                        engine.modular_block1.clear_all(&queue);
                        engine.modular_block1.invalidate_bind_group_caches();
                        engine.modular_block2.clear_all(&queue);
                    }
                    if let Some(ref mut video) = self.video_input {
                        match video.start_input1_syphon(&server_name) {
                            Ok(_) => log::info!("[INPUT] Started Syphon input 1: {}", server_name),
                            Err(e) => log::error!("[INPUT] Failed to start Syphon input 1: {:?}", e),
                        }
                    } else {
                        log::error!("[INPUT] Video input manager not initialized");
                    }
                }
                crate::core::InputChangeRequest::StopInput { .. } => {
                    if let Some(ref mut engine) = self.output_engine {
                        engine.input_texture_manager.clear_input1();
                        let queue = std::sync::Arc::clone(&engine.queue);
                        engine.modular_block1.clear_all(&queue);
                        engine.modular_block1.invalidate_bind_group_caches();
                        engine.modular_block2.clear_all(&queue);
                    }
                    if let Some(ref mut video) = self.video_input {
                        video.stop_input1();
                    }
                }
                crate::core::InputChangeRequest::SetVsync(enabled) => {
                    self.set_output_vsync(enabled);
                }
                crate::core::InputChangeRequest::SetOutputFps(fps) => {
                    self.set_output_fps(fps);
                }
                _ => {}
            }
            
            // Handle input 2 change request
            let mut state = self.shared_state.lock().unwrap();
            let input2_request = std::mem::replace(&mut state.input2_change_request, crate::core::InputChangeRequest::None);
            drop(state);
            
            match input2_request {
                crate::core::InputChangeRequest::StartWebcam { device_index, width, height, fps, .. } => {
                    if let Some(ref mut engine) = self.output_engine {
                        engine.input_texture_manager.clear_input2();
                        let queue = std::sync::Arc::clone(&engine.queue);
                        engine.modular_block1.clear_all(&queue);
                        engine.modular_block1.invalidate_bind_group_caches();
                        engine.modular_block2.clear_all(&queue);
                    }
                    if let Some(ref mut video) = self.video_input {
                        log::info!("[INPUT] Processing webcam start request for input 2, device {}", device_index);
                        match video.start_input2_webcam(device_index, width, height, fps) {
                            Ok(_) => log::info!("[INPUT] Successfully started webcam input 2"),
                            Err(e) => log::error!("[INPUT] Failed to start webcam input 2: {:?}", e),
                        }
                    } else {
                        log::error!("[INPUT] Video input manager not initialized!");
                    }
                }
                crate::core::InputChangeRequest::StartNdi { source_name, .. } => {
                    if let Some(ref mut engine) = self.output_engine {
                        engine.input_texture_manager.clear_input2();
                        let queue = std::sync::Arc::clone(&engine.queue);
                        engine.modular_block1.clear_all(&queue);
                        engine.modular_block1.invalidate_bind_group_caches();
                        engine.modular_block2.clear_all(&queue);
                    }
                    if let Some(ref mut video) = self.video_input {
                        match video.start_input2_ndi(&source_name) {
                            Ok(_) => log::info!("[INPUT] Started NDI input 2: {}", source_name),
                            Err(e) => log::error!("[INPUT] Failed to start NDI input 2: {:?}", e),
                        }
                    } else {
                        log::error!("[INPUT] Video input manager not initialized");
                    }
                }
                crate::core::InputChangeRequest::StartSyphon { server_name, .. } => {
                    if let Some(ref mut engine) = self.output_engine {
                        engine.input_texture_manager.clear_input2();
                        let queue = std::sync::Arc::clone(&engine.queue);
                        engine.modular_block1.clear_all(&queue);
                        engine.modular_block1.invalidate_bind_group_caches();
                        engine.modular_block2.clear_all(&queue);
                    }
                    if let Some(ref mut video) = self.video_input {
                        match video.start_input2_syphon(&server_name) {
                            Ok(_) => log::info!("[INPUT] Started Syphon input 2: {}", server_name),
                            Err(e) => log::error!("[INPUT] Failed to start Syphon input 2: {:?}", e),
                        }
                    } else {
                        log::error!("[INPUT] Video input manager not initialized");
                    }
                }
                crate::core::InputChangeRequest::StopInput { .. } => {
                    if let Some(ref mut engine) = self.output_engine {
                        engine.input_texture_manager.clear_input2();
                        let queue = std::sync::Arc::clone(&engine.queue);
                        engine.modular_block1.clear_all(&queue);
                        engine.modular_block1.invalidate_bind_group_caches();
                        engine.modular_block2.clear_all(&queue);
                    }
                    if let Some(ref mut video) = self.video_input {
                        video.stop_input2();
                    }
                }
                _ => {}
            }
            
            // Handle audio change request
            let mut state = self.shared_state.lock().unwrap();
            let audio_request = std::mem::replace(&mut state.audio_change_request, crate::core::AudioChangeRequest::None);
            drop(state);
            
            match audio_request {
                crate::core::AudioChangeRequest::ChangeDevice { device_index } => {
                    log::info!("Processing audio device change request to index {}", device_index);
                    // Stop current audio
                    if let Some(ref mut audio) = self.audio_input {
                        let _ = audio.stop();
                    }
                    // Create new audio input with specific device
                    let mut new_audio = AudioInput::new(self.config.audio.fft_size);
                    match new_audio.initialize_with_device(device_index) {
                        Ok(_) => {
                            log::info!("Audio reinitialized successfully with device {}", device_index);
                            self.audio_input = Some(new_audio);
                        }
                        Err(e) => {
                            log::error!("Failed to reinitialize audio: {:?}", e);
                        }
                    }
                }
                crate::core::AudioChangeRequest::ChangeFftSize { fft_size } => {
                    log::info!("Processing FFT size change request to {}", fft_size);
                    self.config.audio.fft_size = fft_size;
                    // Stop current audio
                    if let Some(ref mut audio) = self.audio_input {
                        let _ = audio.stop();
                    }
                    // Rebuild with new FFT size
                    let mut new_audio = AudioInput::new(fft_size);
                    match new_audio.initialize() {
                        Ok(_) => {
                            log::info!("Audio reinitialized with FFT size {}", fft_size);
                            self.audio_input = Some(new_audio);
                        }
                        Err(e) => {
                            log::error!("Failed to reinitialize audio with FFT size {}: {:?}", fft_size, e);
                        }
                    }
                }
                _ => {}
            }
            
            // Handle output commands (NDI + Syphon)
            {
                let mut state = self.shared_state.lock().unwrap();
                let output_cmd = std::mem::replace(&mut state.output_command, crate::core::OutputCommand::None);
                drop(state);

                if let Some(ref mut engine) = self.output_engine {
                    match output_cmd {
                        crate::core::OutputCommand::StartNdi { name, include_alpha, frame_skip } => {
                            #[cfg(feature = "ndi")]
                            match engine.start_ndi_output(&name, include_alpha, frame_skip) {
                                Ok(_) => log::info!("[ENGINE] NDI output started: '{}'", name),
                                Err(e) => log::error!("[ENGINE] Failed to start NDI output: {:?}", e),
                            }
                        }
                        crate::core::OutputCommand::StopNdi => {
                            #[cfg(feature = "ndi")]
                            engine.stop_ndi_output();
                        }
                        crate::core::OutputCommand::StartSyphon { name } => {
                            match engine.start_syphon_output(&name) {
                                Ok(_) => log::info!("[ENGINE] Syphon output started: '{}'", name),
                                Err(e) => log::error!("[ENGINE] Failed to start Syphon output: {:?}", e),
                            }
                        }
                        crate::core::OutputCommand::StopSyphon => {
                            engine.stop_syphon_output();
                        }
                        crate::core::OutputCommand::None => {}
                    }
                }
            }
        }
        
        // Update video input and upload frames to GPU
        if let (Some(ref mut video), Some(ref mut engine)) = (self.video_input.as_mut(), self.output_engine.as_mut()) {
            video.update();
            
            // Check for GPU-accelerated Syphon input first (macOS + syphon feature only)
            #[cfg(all(target_os = "macos", feature = "syphon"))]
            {
                // Input 1: GPU Syphon texture
                if video.input1_is_gpu_syphon() {
                    if let Some(texture) = video.get_input1_syphon_texture() {
                        engine.input_texture_manager.update_input1_from_texture(texture);
                    }
                }
                // Input 1: CPU frame data (webcam, NDI, or fallback Syphon)
                else if video.input1_has_new_frame() {
                    if let Some(frame_data) = video.take_input1_frame() {
                        let (width, height) = video.get_input1_resolution();
                        engine.input_texture_manager.update_input1(&frame_data, width, height);
                    }
                }
                
                // Input 2: GPU Syphon texture
                if video.input2_is_gpu_syphon() {
                    if let Some(texture) = video.get_input2_syphon_texture() {
                        engine.input_texture_manager.update_input2_from_texture(texture);
                    }
                }
                // Input 2: CPU frame data
                else if video.input2_has_new_frame() {
                    if let Some(frame_data) = video.take_input2_frame() {
                        let (width, height) = video.get_input2_resolution();
                        engine.input_texture_manager.update_input2(&frame_data, width, height);
                    }
                }
            }
            
            // Non-macOS or syphon disabled: CPU frame data only
            #[cfg(not(all(target_os = "macos", feature = "syphon")))]
            {
                // Upload input 1 frame if available
                if video.input1_has_new_frame() {
                    if let Some(frame_data) = video.take_input1_frame() {
                        let (width, height) = video.get_input1_resolution();
                        engine.input_texture_manager.update_input1(&frame_data, width, height);
                    }
                }
                
                // Upload input 2 frame if available
                if video.input2_has_new_frame() {
                    if let Some(frame_data) = video.take_input2_frame() {
                        let (width, height) = video.get_input2_resolution();
                        engine.input_texture_manager.update_input2(&frame_data, width, height);
                    }
                }
            }
        }
        
        // Poll for MIDI events and update parameters
        if let Some(ref mut midi) = self.midi_input {
            let events = midi.poll_events();
            if !events.is_empty() {
                log::trace!("MIDI: Polled {} events", events.len());
                if let Ok(mut state) = self.shared_state.lock() {
                    // Update connected device list
                    state.midi.connected_devices = midi.connected_devices();
                    
                    // Process MIDI events
                    if state.midi.enabled {
                        for event in events {
                            log::trace!("MIDI: Processing event from {}: {:?}", event.device_id, event.message);
                            if let Some((param_id, value)) = state.midi.process_midi_event(event.clone()) {
                                // Apply the MIDI value to the appropriate parameter
                                log::debug!("MIDI: Mapping found - param_id={}, value={:.3}", param_id, value);
                                Self::apply_midi_value_to_param(&mut state, &param_id, value);
                                log::debug!("MIDI: Applied {} = {:.3} to parameter", param_id, value);
                            } else {
                                log::trace!("MIDI: No mapping found for event from {}", event.device_id);
                            }
                        }
                    } else {
                        log::trace!("MIDI: Events received but MIDI is disabled");
                    }
                }
            }
        }
        
        // Request redraw for control window only on activity, plus a slow idle refresh
        if let Some(ref window) = self.control_window {
            let now = std::time::Instant::now();
            let idle_due = match self.control_last_frame_time {
                None => true,
                Some(last_time) => {
                    let elapsed = now.duration_since(last_time);
                    elapsed >= self.control_target_frame_duration
                }
            };
            let active = self.control_active_until.is_some_and(|until| now < until);
            
            if self.control_needs_redraw || active || idle_due {
                window.request_redraw();
            }
        }
        
        // Request redraw for output window.
        // Always request redraws even when occluded — the render function
        // skips only the surface blit, keeping outputs streaming.
        if let Some(ref window) = self.output_window {
            if self.output_vsync_enabled {
                // VSync mode: let the display compositor be the throttle.
                // Always request redraw; present() will block to refresh rate.
                window.request_redraw();
            } else {
                // Software frame limiter: only render when target duration has elapsed.
                let now = std::time::Instant::now();
                let should_render = match self.output_last_frame_time {
                    None => true,
                    Some(last_time) => now.duration_since(last_time) >= self.output_target_frame_duration,
                };

                if should_render {
                    self.output_last_frame_time = Some(now);
                    self.output_next_frame_time = now + self.output_target_frame_duration;
                    window.request_redraw();
                    use winit::event_loop::ControlFlow;
                    event_loop.set_control_flow(ControlFlow::Poll);
                } else {
                    use winit::event_loop::ControlFlow;
                    event_loop.set_control_flow(ControlFlow::WaitUntil(self.output_next_frame_time));
                }
            }
        }
    }
    
}

/// wgpu-based rendering engine for output
pub struct WgpuEngine {
    #[allow(dead_code)]
    instance: wgpu::Instance,
    #[allow(dead_code)]
    adapter: wgpu::Adapter,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    
    // VSync and frame rate settings
    vsync: bool,
    target_fps: u32,
    
    shared_state: Arc<std::sync::Mutex<SharedState>>,
    
    // Modular blocks (new 3-stage implementation)
    modular_block1: ModularBlock1,
    modular_block2: ModularBlock2,
    modular_block3: ModularBlock3,
    

    
    // Simple blit pipeline for output
    blit_pipeline: wgpu::RenderPipeline,
    blit_bind_group_layout: wgpu::BindGroupLayout,
    
    // Render targets
    block1_texture: Texture,
    block2_texture: Texture,
    block3_texture: Texture,
    
    // Feedback texture (legacy - modular blocks manage their own feedback)
    fb1_texture: Texture,
    
    /// Input texture manager for video sources
    pub input_texture_manager: InputTextureManager,
    
    vertex_buffer: wgpu::Buffer,
    
    frame_count: u64,

    // Output FPS measurement
    fps_ring: [f32; 60],
    fps_ring_index: usize,
    fps_ring_count: usize,
    last_render_time: Option<std::time::Instant>,
    
    /// Preview renderer for color picker
    preview_renderer: Option<crate::engine::preview::PreviewRenderer>,
    
    /// Video recorder
    recorder: Option<crate::recorder::Recorder>,
    
    /// Unified output manager (NDI + Syphon dispatch, async readback pool)
    output_manager: crate::output::OutputManager,
}

impl WgpuEngine {
    pub async fn new(
        instance: &wgpu::Instance,
        window: Arc<Window>,
        app_config: &AppConfig,
        shared_state: Arc<std::sync::Mutex<SharedState>>,
        adapter_out: &mut Option<wgpu::Adapter>,
        device_out: &mut Option<Arc<wgpu::Device>>,
        queue_out: &mut Option<Arc<wgpu::Queue>>,
    ) -> Result<Self> {
        let size = window.inner_size();
        
        let surface = instance.create_surface(window)?;
        
        // Get or create adapter
        let adapter = if let Some(adapter) = adapter_out.take() {
            adapter
        } else {
            instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: Some(&surface),
                    force_fallback_adapter: false,
                })
                .await?
        };
        
        // Get or create device/queue
        let (device, queue) = if let (Some(device), Some(queue)) = (device_out.clone(), queue_out.clone()) {
            (device, queue)
        } else {
            let (device, queue) = adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        required_features: wgpu::Features::empty(),
                        required_limits: wgpu::Limits::default(),
                        label: Some("Device"),
                        memory_hints: wgpu::MemoryHints::default(),
                        trace: wgpu::Trace::Off,
                    },
                )
                .await?;
            let device = Arc::new(device);
            let queue = Arc::new(queue);
            *device_out = Some(Arc::clone(&device));
            *queue_out = Some(Arc::clone(&queue));
            (device, queue)
        };
        
        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);
        
        // Determine present mode based on vsync setting
        let vsync = app_config.output_window.vsync;
        let target_fps = app_config.output_window.fps;
        let present_mode = if vsync {
            wgpu::PresentMode::AutoVsync
        } else {
            wgpu::PresentMode::AutoNoVsync
        };
        
        log::info!("Output surface configured: vsync={}, target_fps={}, present_mode={:?}", 
            vsync, target_fps, present_mode);
        
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);
        
        // Get dimensions from the new resolution config
        let (internal_width, internal_height) = app_config.resolution.internal.dimensions();
        let (output_width, output_height) = app_config.resolution.output.dimensions();
        
        log::info!("WgpuEngine::new() - Creating textures with resolution:");
        log::info!("  Internal: {}x{} (from config.resolution.internal)", internal_width, internal_height);
        log::info!("  Output/Surface: {}x{} (from config.resolution.output)", output_width, output_height);
        log::info!("  Legacy pipeline config was: {}x{}", app_config.pipeline.internal_width, app_config.pipeline.internal_height);
        
        // Render targets
        let block1_texture = Texture::create_render_target(
            &device,
            internal_width,
            internal_height,
            "Block1 Texture",
        );
        let block2_texture = Texture::create_render_target(
            &device,
            internal_width,
            internal_height,
            "Block2 Texture",
        );
        let block3_texture = Texture::create_render_target(
            &device,
            output_width,
            output_height,
            "Block3 Texture",
        );
        
        // Feedback texture (legacy - modular blocks manage their own feedback)
        let fb1_texture = Texture::create_render_target(
            &device,
            internal_width,
            internal_height,
            "FB1 Texture",
        );
        fb1_texture.clear_to_black(&queue);
        
        // Create modular blocks (new 3-stage implementation)
        let modular_block1 = ModularBlock1::new(&device, &queue, internal_width, internal_height);
        let modular_block2 = ModularBlock2::new(&device, &queue, internal_width, internal_height);
        // Block 3 renders to configured output resolution (will be upscaled to window size in blit)
        let modular_block3 = ModularBlock3::new(&device, &queue, output_width, output_height);
        
        // Create simple blit pipeline for final output
        let blit_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Blit Bind Group Layout"),
            entries: &[
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
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        
        let blit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Blit Shader"),
            source: wgpu::ShaderSource::Wgsl(r#"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) texcoord: vec2<f32>,
};

@vertex
fn vs_main(@location(0) position: vec2<f32>, @location(1) texcoord: vec2<f32>) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4<f32>(position, 0.0, 1.0);
    out.texcoord = texcoord;
    return out;
}

@group(0) @binding(0)
var source_tex: texture_2d<f32>;
@group(0) @binding(1)
var source_sampler: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(source_tex, source_sampler, in.texcoord);
}
"#.into()),
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
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
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
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &blit_shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
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
        
        let vertices = Vertex::quad_vertices();
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Quad Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        
        // Create input texture manager
        let input_texture_manager = InputTextureManager::new(
            Arc::clone(&device),
            Arc::clone(&queue),
        );
        
        // Store adapter for potential reuse
        *adapter_out = Some(adapter.clone());
        
        Ok(Self {
            instance: instance.clone(),
            adapter,
            device: Arc::clone(&device),
            queue: Arc::clone(&queue),
            surface,
            config,
            vsync,
            target_fps,

            shared_state,
            modular_block1,
            modular_block2,
            modular_block3,
            blit_pipeline,
            blit_bind_group_layout,
            block1_texture,
            block2_texture,
            block3_texture,
            fb1_texture,
            input_texture_manager,
            vertex_buffer,
            frame_count: 0,
            fps_ring: [0.0; 60],
            fps_ring_index: 0,
            fps_ring_count: 0,
            last_render_time: None,
            
            // Initialize preview renderer (320x180 = 16:9 aspect)
            preview_renderer: Some(crate::engine::preview::PreviewRenderer::new(
                &device, 320, 180
            )),
            
            // Initialize recorder (will be created when recording starts)
            recorder: None,
            
            output_manager: crate::output::OutputManager::new(),
        })
    }
    
    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            // Check if surface capabilities have changed (e.g., moved to different monitor)
            let surface_caps = self.surface.get_capabilities(&self.adapter);
            let preferred_format = surface_caps
                .formats
                .iter()
                .copied()
                .find(|f| f.is_srgb())
                .unwrap_or(surface_caps.formats[0]);
            
            // Update format if it changed (e.g., moving between displays)
            if preferred_format != self.config.format {
                println!("Surface format changed from {:?} to {:?}", self.config.format, preferred_format);
                self.config.format = preferred_format;
            }
            
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
            
            // Note: Block3 texture is NOT resized here - it stays at configured output resolution
            // The blit pass handles upscaling/downscaling to the window surface size
            // This allows users to set a fixed render resolution independent of window size
            log::debug!("Window resized to {}x{}, but Block3 remains at configured output resolution (blit will scale)", width, height);
        }
    }
    
    /// Resize internal textures (called when pipeline config changes)
    pub fn resize_internal(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            // Recreate render targets
            self.block1_texture = Texture::create_render_target(
                &self.device,
                width,
                height,
                "Block1 Texture",
            );
            self.block2_texture = Texture::create_render_target(
                &self.device,
                width,
                height,
                "Block2 Texture",
            );
            
            // Recreate feedback texture
            self.fb1_texture = Texture::create_render_target(
                &self.device,
                width,
                height,
                "FB1 Texture",
            );
            
            // Update shared state
            if let Ok(mut state) = self.shared_state.lock() {
                state.internal_size = (width, height);
            }
        }
    }
    
    /// Get current VSync setting
    pub fn vsync(&self) -> bool {
        self.vsync
    }
    
    /// Set VSync on/off (updates present mode)
    pub fn set_vsync(&mut self, enabled: bool) {
        if self.vsync != enabled {
            self.vsync = enabled;
            self.config.present_mode = if enabled {
                wgpu::PresentMode::AutoVsync
            } else {
                wgpu::PresentMode::AutoNoVsync
            };
            self.surface.configure(&self.device, &self.config);
            log::info!("VSync {}", if enabled { "enabled" } else { "disabled" });
        }
    }
    
    /// Get target FPS
    pub fn target_fps(&self) -> u32 {
        self.target_fps
    }
    
    /// Set target FPS
    pub fn set_target_fps(&mut self, fps: u32) {
        self.target_fps = fps.max(1).min(240);
        log::info!("Target FPS set to {}", self.target_fps);
    }
    
    pub fn render(&mut self, occluded: bool) {
        // Measure actual output FPS
        let now = std::time::Instant::now();
        if let Some(last) = self.last_render_time {
            let delta = now.duration_since(last).as_secs_f32();
            if delta > 0.0 {
                self.fps_ring[self.fps_ring_index] = 1.0 / delta;
                self.fps_ring_index = (self.fps_ring_index + 1) % 60;
                if self.fps_ring_count < 60 {
                    self.fps_ring_count += 1;
                }
            }
        }
        self.last_render_time = Some(now);

        let mut state = self.shared_state.lock().unwrap();

        // Publish measured FPS to shared state for GUI display
        if self.fps_ring_count > 0 {
            let sum: f32 = self.fps_ring[..self.fps_ring_count].iter().sum();
            state.output_actual_fps = sum / self.fps_ring_count as f32;
        }

        let output_mode = state.output_mode;
        let block2_input_select = state.block2.block2_input_select;
        
        // Update LFO phases using BPM from GUI
        let bpm = state.bpm;
        update_lfo_phases(&mut state.lfo_banks, bpm, 1.0 / 60.0); // Assuming 60fps
        
        // Clone base parameters and apply LFO modulation
        let mut modulated_block1 = state.block1.clone();
        let mut modulated_block2 = state.block2.clone();
        let mut modulated_block3 = state.block3.clone();
        
        apply_lfos_to_block1(&mut modulated_block1, &state.lfo_banks, &state.block1_lfo_map);
        apply_lfos_to_block2(&mut modulated_block2, &state.lfo_banks, &state.block2_lfo_map);
        apply_lfos_to_block3(&mut modulated_block3, &state.lfo_banks, &state.block3_lfo_map);
        
        // Apply audio modulations - get FFT values from shared state (already processed)
        let fft_values: [f32; 8] = std::array::from_fn(|i| state.audio.fft.get(i).copied().unwrap_or(0.0));
        
        // Apply audio modulations directly without cloning HashMaps
        // (functions only read from the maps, so we can pass references)
        apply_audio_modulations_to_block1(&mut modulated_block1, &state.block1_modulations, &fft_values);
        apply_audio_modulations_to_block2(&mut modulated_block2, &state.block2_modulations, &fft_values);
        apply_audio_modulations_to_block3(&mut modulated_block3, &state.block3_modulations, &fft_values);
        
        // Apply delay time tempo sync
        // If sync is enabled, calculate delay frames from BPM
        if modulated_block1.fb1_delay_time_sync {
            modulated_block1.fb1_delay_time = crate::core::lfo_engine::calculate_delay_frames_from_tempo(
                bpm, modulated_block1.fb1_delay_time_division, 60.0
            );
        }
        if modulated_block2.fb2_delay_time_sync {
            modulated_block2.fb2_delay_time = crate::core::lfo_engine::calculate_delay_frames_from_tempo(
                bpm, modulated_block2.fb2_delay_time_division, 60.0
            );
        }
        

        
        // Get input texture sizes for proper UV scaling
        let input1_size = self.input_texture_manager.get_input1_resolution();
        let input2_size = self.input_texture_manager.get_input2_resolution();
        
        // Use modulated parameters for rendering
        self.modular_block2.update_params(&self.queue, &modulated_block2);
        // Block 3 params are passed to render method
        
        // Get preview source for modular Block 1 before dropping state
        let preview_source = state.preview_source;
        
        drop(state);
        
        // Update input textures in the pipeline
        let input1_view = self.input_texture_manager.get_input1_view();
        let input2_view = self.input_texture_manager.get_input2_view();
        let _input1_has_data = self.input_texture_manager.input1_has_data();
        let _input2_has_data = self.input_texture_manager.input2_has_data();
        
        if self.frame_count % 60 == 0 {
            log::info!("[RENDER] Frame {}: input1_has_data={}, input2_has_data={}",
                self.frame_count, _input1_has_data, _input2_has_data);
        }
        
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Pipeline Encoder"),
        });
        
        // Render modular Block 1
        // This renders to modular_block1's internal buffers
        {
            // Render modular Block 1
            self.modular_block1.render(
                &mut encoder,
                &self.device,
                &self.queue,
                input1_view,
                input2_view,
                &modulated_block1,
            );
            
            // Update feedback for next frame
            self.modular_block1.update_feedback(&mut encoder);
        }
        
        // Copy modular Block 1 output to block1_texture for downstream use
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.modular_block1.resources.get_output_texture(),
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &self.block1_texture.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: self.block1_texture.width,
                height: self.block1_texture.height,
                depth_or_array_layers: 1,
            },
        );
        
        // Determine Block2 input based on block2_input_select
        // 0 = block1, 1 = input1, 2 = input2
        let _input1_has_data = self.input_texture_manager.input1_has_data();
        let _input2_has_data = self.input_texture_manager.input2_has_data();
        
        let block2_input_view: &wgpu::TextureView = match block2_input_select {
            1 => input1_view,
            2 => input2_view,
            _ => &self.block1_texture.view,
        };
        
        // Render Block 2 using modular implementation
        self.modular_block2.render(
            &mut encoder,
            &self.device,
            &self.queue,
            &self.block1_texture.view,
            input1_view,
            input2_view,
            &self.block2_texture.view,
            &modulated_block2,
        );
        
        // Update FB2 feedback for next frame
        // Block 2 renders to block2_texture (not ping-pong buffers), so copy from there
        self.modular_block2.resources.update_feedback_from_external(&mut encoder, &self.block2_texture.texture);
        self.modular_block2.resources.update_delay_buffer_from_external(&mut encoder, &self.block2_texture.texture);
        
        // Determine which block to display based on output_mode
        let (output_texture, _output_view): (&wgpu::Texture, &wgpu::TextureView) = match output_mode {
            OutputMode::Block1 => {
                log::debug!("Output mode: Block1 (texture: {}x{})", self.block1_texture.width, self.block1_texture.height);
                (&self.block1_texture.texture, &self.block1_texture.view)
            }
            OutputMode::Block2 => {
                log::debug!("Output mode: Block2 (texture: {}x{})", self.block2_texture.width, self.block2_texture.height);
                (&self.block2_texture.texture, &self.block2_texture.view)
            }
            OutputMode::Block3 => {
                log::debug!("Output mode: Block3 (texture: {}x{})", self.block3_texture.width, self.block3_texture.height);
                (&self.block3_texture.texture, &self.block3_texture.view)
            }
            OutputMode::PreviewInput1 => {
                if let Some(ref input) = self.input_texture_manager.input1 {
                    (&input.texture.texture, &input.texture.view)
                } else {
                    (&self.block1_texture.texture, &self.block1_texture.view)
                }
            },
            OutputMode::PreviewInput2 => {
                if let Some(ref input) = self.input_texture_manager.input2 {
                    (&input.texture.texture, &input.texture.view)
                } else {
                    (&self.block1_texture.texture, &self.block1_texture.view)
                }
            },
        };
        
        // Render Block 3 using modular implementation
        log::debug!("Block3 input: block1_texture={}x{}, block2_texture={}x{}",
            self.block1_texture.width, self.block1_texture.height,
            self.block2_texture.width, self.block2_texture.height);
        let block3_output_view = self.modular_block3.render(
            &mut encoder,
            &self.device,
            &self.queue,
            &self.block1_texture.view,
            &self.block2_texture.view,
            &modulated_block3,
        );
        
        // Copy modular Block 3 output to block3_texture for display
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: self.modular_block3.get_output_texture(),
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &self.block3_texture.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: self.block3_texture.width,
                height: self.block3_texture.height,
                depth_or_array_layers: 1,
            },
        );
        
        // NOTE: Surface blit is deferred until after pipeline submit (see below).
        // This allows the shader pipeline + outputs to always run even when occluded.

        // Copy render targets to feedback textures for next frame
        // This must happen after all render passes are done
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.block1_texture.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &self.fb1_texture.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: self.block1_texture.width,
                height: self.block1_texture.height,
                depth_or_array_layers: 1,
            },
        );
        
        // Note: Block 2 feedback is managed by modular_block2.resources
        // The update_feedback() call above handles copying output to feedback
        
        // Check if preview is enabled before computing
        let preview_enabled = if let Ok(state) = self.shared_state.lock() {
            state.preview_enabled
        } else {
            true // Default to enabled if can't read state
        };
        
        // Update preview renderer only if enabled (saves GPU when window closed)
        if preview_enabled {
            if let Some(ref mut preview) = self.preview_renderer {
                // Sync preview source (captured before dropping state)
                preview.set_source(preview_source);
                
                // Get source texture based on preview source
                let source_view: &wgpu::TextureView = match preview_source {
                    crate::engine::preview::PreviewSource::Block1 => &self.block1_texture.view,
                    crate::engine::preview::PreviewSource::Block2 => &self.block2_texture.view,
                    crate::engine::preview::PreviewSource::Block3 => &self.block3_texture.view,
                    crate::engine::preview::PreviewSource::Input1 => input1_view,
                    crate::engine::preview::PreviewSource::Input2 => input2_view,
                };
                
                preview.update(&self.device, &self.queue, source_view);
                preview.process_readback();
                
                // Handle color picking - only sample when mouse click is requested
                let pick_request = if let Ok(state) = self.shared_state.lock() {
                    if state.preview_pick_requested {
                        Some(state.preview_pick_uv)
                    } else {
                        None
                    }
                } else {
                    None
                };
                
                // Only sample color when a pick request is active
                if let Some(pick_uv) = pick_request {
                    if let Some(color) = preview.pick_color_uv(pick_uv[0], pick_uv[1]) {
                        // Convert u8 RGB to f32 0-1 range
                        let color_f32 = [
                            color[0] as f32 / 255.0,
                            color[1] as f32 / 255.0,
                            color[2] as f32 / 255.0,
                        ];
                        // Write back to shared state and clear the request
                        if let Ok(mut state) = self.shared_state.lock() {
                            state.preview_sampled_color = color_f32;
                            state.preview_pick_requested = false;
                            log::debug!("Color picked at UV [{:.3}, {:.3}]: [{:.3}, {:.3}, {:.3}]", 
                                pick_uv[0], pick_uv[1], color_f32[0], color_f32[1], color_f32[2]);
                        }
                    }
                }
            }
        }
        
        // NOTE: Output submission (NDI readback + Syphon publish) happens after
        // the pipeline encoder is submitted — see output_manager.submit_frame() below.
        
        // Handle recording commands
        let recording_command = {
            if let Ok(mut state) = self.shared_state.lock() {
                let cmd = state.recording_command;
                state.recording_command = crate::core::RecordingCommand::None;
                cmd
            } else {
                crate::core::RecordingCommand::None
            }
        };
        
        match recording_command {
            crate::core::RecordingCommand::Start | crate::core::RecordingCommand::Toggle => {
                if self.recorder.is_none() {
                    // Determine resolution based on current output mode
                    // Different blocks may have different texture sizes
                    let resolution = match output_mode {
                        OutputMode::Block1 => (self.block1_texture.width, self.block1_texture.height),
                        OutputMode::Block2 => (self.block2_texture.width, self.block2_texture.height),
                        OutputMode::Block3 => (self.block3_texture.width, self.block3_texture.height),
                        OutputMode::PreviewInput1 => {
                            if let Some(ref input) = self.input_texture_manager.input1 {
                                (input.texture.width, input.texture.height)
                            } else {
                                (self.block1_texture.width, self.block1_texture.height)
                            }
                        }
                        OutputMode::PreviewInput2 => {
                            if let Some(ref input) = self.input_texture_manager.input2 {
                                (input.texture.width, input.texture.height)
                            } else {
                                (self.block1_texture.width, self.block1_texture.height)
                            }
                        }
                    };
                    
                    log::info!("Starting recording at resolution: {}x{} (output mode: {:?})", 
                        resolution.0, resolution.1, output_mode);
                    
                    if let Ok(state) = self.shared_state.lock() {
                        let settings = state.recording_settings.clone();
                        drop(state);
                        
                        let mut recorder = crate::recorder::Recorder::new(settings, resolution);
                        match recorder.start() {
                            Ok(_) => {
                                log::info!("Recording started");
                                if let Ok(mut state) = self.shared_state.lock() {
                                    state.is_recording = true;
                                }
                                self.recorder = Some(recorder);
                            }
                            Err(e) => {
                                log::error!("Failed to start recording: {}", e);
                                if let Ok(mut state) = self.shared_state.lock() {
                                    state.is_recording = false;
                                }
                            }
                        }
                    }
                } else {
                    // Already recording, toggle means stop
                    if let Some(recorder) = self.recorder.take() {
                        drop(recorder); // This will stop recording
                        log::info!("Recording stopped");
                        if let Ok(mut state) = self.shared_state.lock() {
                            state.is_recording = false;
                        }
                    }
                }
            }
            crate::core::RecordingCommand::Stop => {
                if let Some(recorder) = self.recorder.take() {
                    drop(recorder); // This will stop recording
                    log::info!("Recording stopped");
                    if let Ok(mut state) = self.shared_state.lock() {
                        state.is_recording = false;
                    }
                }
            }
            crate::core::RecordingCommand::None => {}
        }
        
        // Capture frame if recording is active
        if self.recorder.is_some() {
            // Get actual texture dimensions from the output texture
            let texture_size = output_texture.size();
            let width = texture_size.width;
            let height = texture_size.height;
            
            // Validate dimensions
            if width == 0 || height == 0 {
                log::error!("Invalid output texture dimensions: {}x{}", width, height);
                if let Some(recorder) = self.recorder.take() {
                    drop(recorder);
                }
                if let Ok(mut state) = self.shared_state.lock() {
                    state.is_recording = false;
                }
            } else {
                let layout = ReadbackLayout::new(width, height);
                let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Recording Staging Buffer"),
                    size: layout.buffer_size,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                });

                encoder.copy_texture_to_buffer(
                    wgpu::TexelCopyTextureInfo {
                        texture: output_texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::TexelCopyBufferInfo {
                        buffer: &staging_buffer,
                        layout: wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(layout.padded_bytes_per_row),
                            rows_per_image: Some(height),
                        },
                    },
                    wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                );

                // Submit pipeline encoder (includes recording copy + NDI copy)
                self.queue.submit(std::iter::once(encoder.finish()));

                // Sync readback for recorder (must be synchronous for frame ordering)
                let buffer_slice = staging_buffer.slice(..);
                let (tx, rx) = std::sync::mpsc::channel();
                buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                    let _ = tx.send(result);
                });

                if self.device.poll(wgpu::PollType::Wait).is_ok() {
                    if let Ok(Ok(())) = rx.recv() {
                        let data = buffer_slice.get_mapped_range();
                        let rgba_data = strip_readback_padding(&data, layout, height);
                        if let Some(ref mut recorder) = self.recorder {
                            if let Err(e) = recorder.write_frame(&rgba_data) {
                                log::error!("Failed to write frame to recorder: {}", e);
                                drop(self.recorder.take());
                                if let Ok(mut state) = self.shared_state.lock() {
                                    state.is_recording = false;
                                }
                            }
                        }
                    }
                }
                let _ = buffer_slice;
                staging_buffer.unmap();

                // Submit to output sinks (NDI readback + Syphon zero-copy)
                self.output_manager.submit_frame(
                    &self.block3_texture.texture, &self.device, &self.queue,
                );

                // ── Blit to screen (skip when occluded) ──────────────────
                if !occluded {
                    match self.surface.get_current_texture() {
                        Ok(surface_texture) => {
                            let surface_view = surface_texture
                                .texture
                                .create_view(&wgpu::TextureViewDescriptor::default());

                            let blit_sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
                                mag_filter: wgpu::FilterMode::Linear,
                                min_filter: wgpu::FilterMode::Linear,
                                ..Default::default()
                            });
                            let blit_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                                label: Some("Blit Bind Group"),
                                layout: &self.blit_bind_group_layout,
                                entries: &[
                                    wgpu::BindGroupEntry {
                                        binding: 0,
                                        resource: wgpu::BindingResource::TextureView(_output_view),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 1,
                                        resource: wgpu::BindingResource::Sampler(&blit_sampler),
                                    },
                                ],
                            });

                            let mut blit_encoder = self.device.create_command_encoder(
                                &wgpu::CommandEncoderDescriptor { label: Some("Blit Encoder") },
                            );
                            {
                                let mut render_pass = blit_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                    label: Some("Blit Pass"),
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
                                render_pass.set_pipeline(&self.blit_pipeline);
                                render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
                                render_pass.set_bind_group(0, &blit_bind_group, &[]);
                                render_pass.draw(0..6, 0..1);
                            }
                            self.queue.submit(std::iter::once(blit_encoder.finish()));
                            surface_texture.present();
                        }
                        Err(e) => {
                            log::debug!("Surface unavailable, reconfiguring: {}", e);
                            self.surface.configure(&self.device, &self.config);
                        }
                    }
                }

                self.frame_count += 1;
                return;
            }
        }

        // Non-recording path: submit pipeline encoder, then outputs, then blit
        self.queue.submit(std::iter::once(encoder.finish()));

        // Submit to output sinks (NDI readback + Syphon zero-copy)
        self.output_manager.submit_frame(
            &self.block3_texture.texture, &self.device, &self.queue,
        );

        // ── Blit to screen (skip when occluded) ──────────────────────
        if !occluded {
            match self.surface.get_current_texture() {
                Ok(surface_texture) => {
                    let surface_view = surface_texture
                        .texture
                        .create_view(&wgpu::TextureViewDescriptor::default());

                    let blit_sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
                        mag_filter: wgpu::FilterMode::Linear,
                        min_filter: wgpu::FilterMode::Linear,
                        ..Default::default()
                    });
                    let blit_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("Blit Bind Group"),
                        layout: &self.blit_bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(_output_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Sampler(&blit_sampler),
                            },
                        ],
                    });

                    let mut blit_encoder = self.device.create_command_encoder(
                        &wgpu::CommandEncoderDescriptor { label: Some("Blit Encoder") },
                    );
                    {
                        let mut render_pass = blit_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("Blit Pass"),
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
                        render_pass.set_pipeline(&self.blit_pipeline);
                        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
                        render_pass.set_bind_group(0, &blit_bind_group, &[]);
                        render_pass.draw(0..6, 0..1);
                    }
                    self.queue.submit(std::iter::once(blit_encoder.finish()));
                    surface_texture.present();
                }
                Err(e) => {
                    log::debug!("Surface unavailable, reconfiguring: {}", e);
                    self.surface.configure(&self.device, &self.config);
                }
            }
        }

        self.frame_count += 1;
    }
    
    /// Get the modular Block 1 output view for preview window
    pub fn get_modular_block1_output_view(&self) -> &wgpu::TextureView {
        self.modular_block1.get_output_view()
    }
    
    /// Get the device for creating preview resources
    pub fn get_device(&self) -> &wgpu::Device {
        &self.device
    }
    
    /// Get the queue for preview rendering
    pub fn get_queue(&self) -> &wgpu::Queue {
        &self.queue
    }
    
    /// Get the preview renderer
    pub fn get_preview_renderer(&self) -> Option<&crate::engine::preview::PreviewRenderer> {
        self.preview_renderer.as_ref()
    }
    
    /// Get the preview renderer (mutable)
    pub fn get_preview_renderer_mut(&mut self) -> Option<&mut crate::engine::preview::PreviewRenderer> {
        self.preview_renderer.as_mut()
    }
    
    /// Start NDI output
    #[cfg(feature = "ndi")]
    pub fn start_ndi_output(&mut self, name: &str, include_alpha: bool, frame_skip: u8) -> anyhow::Result<()> {
        let width = self.block3_texture.width;
        let height = self.block3_texture.height;
        self.output_manager.start_ndi(name, width, height, include_alpha, frame_skip)?;
        if let Ok(mut state) = self.shared_state.lock() {
            state.ndi_output_active = true;
        }
        Ok(())
    }

    /// Stop NDI output
    #[cfg(feature = "ndi")]
    pub fn stop_ndi_output(&mut self) {
        self.output_manager.stop_ndi();
        if let Ok(mut state) = self.shared_state.lock() {
            state.ndi_output_active = false;
        }
    }

    /// Start Syphon output (macOS only, requires syphon feature)
    #[cfg(all(target_os = "macos", feature = "syphon"))]
    pub fn start_syphon_output(&mut self, name: &str) -> anyhow::Result<()> {
        let width = self.block3_texture.texture.width();
        let height = self.block3_texture.texture.height();
        self.output_manager.start_syphon(name, &self.device, &self.queue, width, height)?;
        if let Ok(mut state) = self.shared_state.lock() {
            state.syphon_output_active = true;
        }
        log::info!("[Engine] Syphon output started (zero-copy: {})",
            self.output_manager.syphon_is_zero_copy());
        Ok(())
    }

    /// Stop Syphon output (macOS only, requires syphon feature)
    #[cfg(all(target_os = "macos", feature = "syphon"))]
    pub fn stop_syphon_output(&mut self) {
        self.output_manager.stop_syphon();
        if let Ok(mut state) = self.shared_state.lock() {
            state.syphon_output_active = false;
        }
    }

    /// Stub for non-macOS platforms or when syphon feature is disabled
    #[cfg(not(all(target_os = "macos", feature = "syphon")))]
    pub fn start_syphon_output(&mut self, _name: &str) -> anyhow::Result<()> {
        Err(anyhow::anyhow!("Syphon is only available on macOS with the 'syphon' feature enabled"))
    }

    /// Stub for non-macOS platforms or when syphon feature is disabled
    #[cfg(not(all(target_os = "macos", feature = "syphon")))]
    pub fn stop_syphon_output(&mut self) {}

    /// Drain readback pool while the GPU device is still alive.
    pub fn drain_readback(&mut self) {
        self.output_manager.drain_readback(&self.device);
    }
    
    /// Stub for when NDI feature is disabled
    #[cfg(not(feature = "ndi"))]
    pub fn start_ndi_output(&mut self, _name: &str, _include_alpha: bool, _frame_skip: u8) -> anyhow::Result<()> {
        Err(anyhow::anyhow!("NDI support not compiled. Enable the 'ndi' feature."))
    }
}

/// Apply audio modulations to Block 1 parameters
fn apply_audio_modulations_to_block1(
    params: &mut crate::params::Block1Params,
    modulations: &HashMap<String, ParamModulationData>,
    fft_values: &[f32; 8],
) {
    for (param_name, modulation) in modulations {
        if !modulation.audio_enabled {
            continue;
        }
        
        let fft_band = modulation.audio_fft_band.clamp(0, 7) as usize;
        let fft_value = fft_values[fft_band];
        let modulated = apply_audio_modulations(0.0, modulation, fft_value, 0.0);
        
        // Apply modulation to the parameter
        match param_name.as_str() {
            "ch1_x_displace" => params.ch1_x_displace += modulated,
            "ch1_y_displace" => params.ch1_y_displace += modulated,
            "ch1_z_displace" => params.ch1_z_displace += modulated,
            "ch1_rotate" => params.ch1_rotate += modulated,
            "ch1_hsb_attenuate.x" => params.ch1_hsb_attenuate.x += modulated,
            "ch1_hsb_attenuate.y" => params.ch1_hsb_attenuate.y += modulated,
            "ch1_hsb_attenuate.z" => params.ch1_hsb_attenuate.z += modulated,
            "ch1_kaleidoscope_amount" => params.ch1_kaleidoscope_amount += modulated,
            "ch1_blur_amount" => params.ch1_blur_amount += modulated,
            "ch2_mix_amount" => params.ch2_mix_amount += modulated,
            "ch2_x_displace" => params.ch2_x_displace += modulated,
            "ch2_y_displace" => params.ch2_y_displace += modulated,
            "ch2_rotate" => params.ch2_rotate += modulated,
            "fb1_mix_amount" => params.fb1_mix_amount += modulated,
            "fb1_x_displace" => params.fb1_x_displace += modulated,
            "fb1_y_displace" => params.fb1_y_displace += modulated,
            "fb1_rotate" => params.fb1_rotate += modulated,
            _ => {} // Unknown parameter
        }
    }
}

/// Apply audio modulations to Block 2 parameters
fn apply_audio_modulations_to_block2(
    params: &mut crate::params::Block2Params,
    modulations: &HashMap<String, ParamModulationData>,
    fft_values: &[f32; 8],
) {
    for (param_name, modulation) in modulations {
        if !modulation.audio_enabled {
            continue;
        }
        
        let fft_band = modulation.audio_fft_band.clamp(0, 7) as usize;
        let fft_value = fft_values[fft_band];
        let modulated = apply_audio_modulations(0.0, modulation, fft_value, 0.0);
        
        match param_name.as_str() {
            "block2_input_x_displace" => params.block2_input_x_displace += modulated,
            "block2_input_y_displace" => params.block2_input_y_displace += modulated,
            "block2_input_rotate" => params.block2_input_rotate += modulated,
            "block2_input_blur_amount" => params.block2_input_blur_amount += modulated,
            "fb2_mix_amount" => params.fb2_mix_amount += modulated,
            "fb2_x_displace" => params.fb2_x_displace += modulated,
            "fb2_y_displace" => params.fb2_y_displace += modulated,
            "fb2_rotate" => params.fb2_rotate += modulated,
            _ => {}
        }
    }
}

/// Apply audio modulations to Block 3 parameters
fn apply_audio_modulations_to_block3(
    params: &mut crate::params::Block3Params,
    modulations: &HashMap<String, ParamModulationData>,
    fft_values: &[f32; 8],
) {
    for (param_name, modulation) in modulations {
        if !modulation.audio_enabled {
            continue;
        }
        
        let fft_band = modulation.audio_fft_band.clamp(0, 7) as usize;
        let fft_value = fft_values[fft_band];
        let modulated = apply_audio_modulations(0.0, modulation, fft_value, 0.0);
        
        match param_name.as_str() {
            "block1_reprocess_x_displace" => params.block1_x_displace += modulated,
            "block1_reprocess_y_displace" => params.block1_y_displace += modulated,
            "block1_reprocess_rotate" => params.block1_rotate += modulated,
            "block2_reprocess_x_displace" => params.block2_x_displace += modulated,
            "block2_reprocess_y_displace" => params.block2_y_displace += modulated,
            "block2_reprocess_rotate" => params.block2_rotate += modulated,
            "final_mix_amount" => params.final_mix_amount += modulated,
            _ => {}
        }
    }
}
