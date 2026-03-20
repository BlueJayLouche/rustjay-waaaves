//! # Simple Main Entry Point
//! 
//! Minimal application using the simplified feedback engine.
//! Auto-starts webcam and runs a single feedback shader with LFOs.
//! Includes a second control window with ImGui for hue slider and tap tempo.

// Allow deprecated ComboBox API - imgui 0.12 uses the older API
#![allow(deprecated)]

use std::sync::Arc;
use std::sync::Mutex;
use std::time::Instant;
use std::collections::VecDeque;

use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

use crate::core::SharedState;
use crate::engine::simple_engine::{SimpleEngine, SimpleEngineConfig};
use crate::engine::imgui_renderer::ImGuiRenderer;
use crate::config::AppConfig;
/// Syphon types (real when syphon feature enabled, stubs otherwise)
use crate::input::{SyphonDiscovery, SyphonServerInfo};
use imgui::ComboBox;

/// Simple app that runs the feedback engine with a control window
pub struct SimpleApp {
    shared_state: Arc<Mutex<SharedState>>,
    engine: Option<SimpleEngine>,
    output_window: Option<Arc<Window>>,
    
    // Control window
    control_window: Option<Arc<Window>>,
    imgui_renderer: Option<ImGuiRenderer>,
    
    // UI State
    hue_amount: f32,
    mix_amount: f32,
    mix_type: i32,
    key_threshold: f32,
    key_softness: f32,
    tap_times: VecDeque<Instant>,
    current_bpm: f32,
    last_tap_info: String,
    
    // Deferred input auto-start (to avoid event loop conflicts)
    pending_input_autostart: Option<String>, // None = webcam, Some(server) = Syphon
    
    // Input selection UI state
    syphon_discovery: SyphonDiscovery,
    available_syphon_servers: Vec<SyphonServerInfo>,
    selected_syphon_server: i32,
    current_input_source: String, // "webcam" or syphon server name
    last_server_refresh: Instant,
    input_switch_requested: Option<String>, // None = no switch, Some("webcam") or Some(server_name)
}

impl SimpleApp {
    pub fn new(shared_state: Arc<Mutex<SharedState>>) -> Self {
        // Check for Syphon input env var at startup
        // Empty string means use webcam, any other value is the Syphon server name
        let pending_input_autostart = match std::env::var("SYPHON_INPUT") {
            Ok(val) if !val.is_empty() => {
                log::info!("[SIMPLE_APP] Will auto-start Syphon input from '{}' (deferred)", val);
                Some(val)
            }
            Ok(_) => {
                log::info!("[SIMPLE_APP] SYPHON_INPUT set but empty - will use webcam");
                Some(String::new()) // Empty string = webcam
            }
            Err(_) => {
                log::info!("[SIMPLE_APP] No SYPHON_INPUT set - will use webcam");
                Some(String::new()) // No env var = webcam
            }
        };
        
        Self {
            shared_state,
            engine: None,
            output_window: None,
            control_window: None,
            imgui_renderer: None,
            hue_amount: 0.5,
            mix_amount: 0.0,
            mix_type: 0,
            key_threshold: 0.0,
            key_softness: 0.0,
            tap_times: VecDeque::with_capacity(8),
            current_bpm: 120.0,
            last_tap_info: String::from("Tap to set tempo"),
            pending_input_autostart,
            syphon_discovery: SyphonDiscovery::new(),
            available_syphon_servers: Vec::new(),
            selected_syphon_server: -1, // -1 means webcam
            current_input_source: String::from("webcam"),
            last_server_refresh: Instant::now(),
            input_switch_requested: None,
        }
    }
    
    /// Handle a tap tempo button press
    fn handle_tap_tempo(&mut self) {
        let now = Instant::now();
        
        // Clear taps if it's been too long since last tap (2 seconds)
        if let Some(&last_tap) = self.tap_times.back() {
            if now.duration_since(last_tap).as_secs_f32() > 2.0 {
                self.tap_times.clear();
                self.last_tap_info = "Reset: new tempo sequence".to_string();
            }
        }
        
        self.tap_times.push_back(now);
        
        // Keep only last 8 taps
        if self.tap_times.len() > 8 {
            self.tap_times.pop_front();
        }
        
        // Reset LFO phase on every tap
        if let Some(ref mut engine) = self.engine {
            engine.reset_lfo_phase();
        }
        
        // Calculate BPM if we have at least 4 taps
        if self.tap_times.len() >= 4 {
            let mut intervals = Vec::new();
            let taps: Vec<_> = self.tap_times.iter().collect();
            
            for i in 1..taps.len() {
                let interval = taps[i].duration_since(*taps[i-1]).as_secs_f32();
                intervals.push(interval);
            }
            
            // Calculate average interval
            let avg_interval: f32 = intervals.iter().sum::<f32>() / intervals.len() as f32;
            
            if avg_interval > 0.0 {
                let bpm = 60.0 / avg_interval;
                // Clamp to reasonable range
                self.current_bpm = bpm.clamp(40.0, 200.0);
                
                // Update engine BPM
                if let Some(ref mut engine) = self.engine {
                    engine.set_bpm(self.current_bpm);
                }
                
                self.last_tap_info = format!("BPM: {:.1} ({} taps)", self.current_bpm, self.tap_times.len());
            }
        } else {
            self.last_tap_info = format!("Tap {} more...", 4 - self.tap_times.len());
        }
    }
}

impl ApplicationHandler for SimpleApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        log::info!("[SIMPLE_APP] Resumed - creating windows");
        
        // Create output window
        let output_attrs = Window::default_attributes()
            .with_title("RustJay Simple - Output")
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 720));
        
        let output_window = match event_loop.create_window(output_attrs) {
            Ok(window) => Arc::new(window),
            Err(err) => {
                log::error!("[SIMPLE_APP] Failed to create output window: {}", err);
                event_loop.exit();
                return;
            }
        };
        self.output_window = Some(output_window.clone());
        
        // Create control window
        let control_attrs = Window::default_attributes()
            .with_title("RustJay Simple - Controls")
            .with_inner_size(winit::dpi::LogicalSize::new(400, 300));
        
        let control_window = match event_loop.create_window(control_attrs) {
            Ok(window) => Arc::new(window),
            Err(err) => {
                log::error!("[SIMPLE_APP] Failed to create control window: {}", err);
                event_loop.exit();
                return;
            }
        };
        self.control_window = Some(control_window.clone());
        
        // Create config
        let config = SimpleEngineConfig {
            output_width: 1280,
            output_height: 720,
            webcam_width: 640,
            webcam_height: 480,
            use_ping_pong: true,
            target_fps: 60.0,
        };
        
        // Create engine
        let shared_state = self.shared_state.clone();
        
        let engine = match pollster::block_on(async {
            SimpleEngine::new(output_window, shared_state, config).await
        }) {
            Ok(engine) => engine,
            Err(err) => {
                log::error!("[SIMPLE_APP] Failed to create simple engine: {}", err);
                event_loop.exit();
                return;
            }
        };
        
        // Create ImGui renderer for control window using shared device/queue
        let control_size = control_window.inner_size();
        let mut imgui_renderer = match pollster::block_on(async {
            let device = engine.device();
            let queue = engine.queue();
            let instance = engine.instance();
            
            ImGuiRenderer::new(
                instance,
                device,
                queue,
                control_window,
                1.0, // UI scale (default)
            ).await
        }) {
            Ok(renderer) => renderer,
            Err(err) => {
                log::error!("[SIMPLE_APP] Failed to create ImGui renderer: {}", err);
                event_loop.exit();
                return;
            }
        };
        
        // Set initial display size
        imgui_renderer.set_display_size(control_size.width as f32, control_size.height as f32);
        
        self.engine = Some(engine);
        self.imgui_renderer = Some(imgui_renderer);
        
        // Note: Input auto-start is deferred to first frame to avoid event loop conflicts
        // See about_to_wait() for the actual connection logic
        
        log::info!("[SIMPLE_APP] Windows created, engine ready");
    }
    
    fn window_event(&mut self, event_loop: &ActiveEventLoop, window_id: WindowId, event: WindowEvent) {
        // Handle control window events
        if let Some(ref control_window) = self.control_window {
            if window_id == control_window.id() {
                // Pass events to ImGui
                if let Some(ref mut renderer) = self.imgui_renderer {
                    renderer.handle_event(&event, control_window);
                }
                
                match event {
                    WindowEvent::CloseRequested => {
                        log::info!("[SIMPLE_APP] Control window close requested");
                        event_loop.exit();
                    }
                    WindowEvent::Resized(size) => {
                        if let Some(ref mut renderer) = self.imgui_renderer {
                            renderer.resize(size.width, size.height);
                            renderer.set_display_size(size.width as f32, size.height as f32);
                        }
                    }
                    WindowEvent::RedrawRequested => {
                        // Render ImGui
                        if let Some(ref mut renderer) = self.imgui_renderer {
                            let bpm = self.current_bpm;
                            let tap_info = self.last_tap_info.clone();
                            let mut hue_amount = self.hue_amount;
                            let mut mix_amount = self.mix_amount;
                            let mut mix_type = self.mix_type as usize;
                            let mut key_threshold = self.key_threshold;
                            let mut key_softness = self.key_softness;
                            let mut tap_pressed = false;
                            
                            // Refresh Syphon servers periodically (every 2 seconds)
                            if self.last_server_refresh.elapsed().as_secs() >= 2 {
                                self.available_syphon_servers = self.syphon_discovery.discover_servers();
                                self.last_server_refresh = Instant::now();
                            }
                            
                            // Prepare input source options
                            let input_sources: Vec<String> = std::iter::once("Webcam".to_string())
                                .chain(self.available_syphon_servers.iter().map(|s| {
                                    if s.name.is_empty() {
                                        format!("{} (app: {})", s.name, s.app_name)
                                    } else {
                                        s.name.clone()
                                    }
                                }))
                                .collect();
                            
                            let mut selected_input = self.selected_syphon_server + 1; // +1 because -1 (webcam) -> 0
                            let mut input_switch_requested: Option<String> = None;
                            
                            let _ = renderer.render_frame(|ui| {
                                ui.window("Controls")
                                    .size([380.0, 500.0], imgui::Condition::FirstUseEver)
                                    .build(|| {
                                        ui.text("RustJay Simple Feedback");
                                        ui.separator();
                                        
                                        // Input source selection
                                        ui.text("Input Source");
                                        let input_preview = if selected_input == 0 {
                                            "Webcam"
                                        } else if let Some(server) = self.available_syphon_servers.get((selected_input - 1) as usize) {
                                            if server.name.is_empty() {
                                                &server.app_name
                                            } else {
                                                &server.name
                                            }
                                        } else {
                                            "Webcam"
                                        };
                                        
                                        ComboBox::new(ui, "##input_source")
                                            .preview_value(input_preview)
                                            .build(|| {
                                                // Webcam option
                                                if ui.selectable_config("Webcam")
                                                    .selected(selected_input == 0)
                                                    .build() {
                                                    selected_input = 0;
                                                }
                                                
                                                // Syphon servers
                                                for (idx, server) in self.available_syphon_servers.iter().enumerate() {
                                                    let label = if server.name.is_empty() {
                                                        format!("{} (app: {})", server.name, server.app_name)
                                                    } else {
                                                        server.name.clone()
                                                    };
                                                    if ui.selectable_config(&label)
                                                        .selected(selected_input == (idx + 1) as i32)
                                                        .build() {
                                                        selected_input = (idx + 1) as i32;
                                                    }
                                                }
                                            });
                                        
                                        // Show current input status
                                        ui.text(format!("Current: {}", self.current_input_source));
                                        
                                        // Apply button for input switch
                                        if ui.button_with_size("Switch Input", [120.0, 25.0]) {
                                            if selected_input == 0 {
                                                input_switch_requested = Some("webcam".to_string());
                                            } else if let Some(server) = self.available_syphon_servers.get((selected_input - 1) as usize) {
                                                let server_name = if server.name.is_empty() {
                                                    server.app_name.clone()
                                                } else {
                                                    server.name.clone()
                                                };
                                                input_switch_requested = Some(server_name);
                                            }
                                        }
                                        
                                        ui.separator();
                                        
                                        // Mix amount slider
                                        ui.text("Mix Amount (Input 1 → Input 2)");
                                        ui.slider("##mix_amount", 0.0, 1.0, &mut mix_amount);
                                        ui.same_line();
                                        ui.text(format!("{:.2}", mix_amount));
                                        
                                        // Mix type dropdown
                                        ui.text("Mix Type");
                                        let mix_types = ["Lerp", "Add", "Diff", "Mult", "Dodge"];
                                        let preview = mix_types[mix_type];
                                        ComboBox::new(ui, "##mix_type")
                                            .preview_value(&preview)
                                            .build(|| {
                                                for (idx, opt) in mix_types.iter().enumerate() {
                                                    if ui.selectable_config(opt)
                                                        .selected(idx == mix_type)
                                                        .build() {
                                                        mix_type = idx;
                                                    }
                                                }
                                            });
                                        
                                        ui.separator();
                                        
                                        // Keying controls
                                        ui.text("Keying");
                                        ui.text("Threshold");
                                        ui.slider("##key_threshold", -1.0, 1.0, &mut key_threshold);
                                        ui.same_line();
                                        ui.text(format!("{:.2}", key_threshold));
                                        
                                        ui.text("Softness");
                                        ui.slider("##key_softness", 0.0, 1.0, &mut key_softness);
                                        ui.same_line();
                                        ui.text(format!("{:.2}", key_softness));
                                        
                                        ui.separator();
                                        
                                        // Hue amount slider
                                        ui.text("Hue Shift Amount");
                                        ui.slider("##hue_amount", 0.0, 1.0, &mut hue_amount);
                                        ui.same_line();
                                        ui.text(format!("{:.2}", hue_amount));
                                        
                                        ui.separator();
                                        
                                        // Tap tempo
                                        ui.text("Tap Tempo");
                                        if ui.button_with_size("TAP", [100.0, 40.0]) {
                                            tap_pressed = true;
                                        }
                                        ui.same_line();
                                        ui.text(&tap_info);
                                        
                                        ui.text(format!("Current BPM: {:.1}", bpm));
                                        
                                        ui.separator();
                                        
                                        // Instructions
                                        ui.text_wrapped("Tap the button 4+ times in rhythm to set the tempo. Each tap resets the LFO phase.");
                                    });
                            });
                            
                            // Apply changes after render frame
                            self.hue_amount = hue_amount;
                            self.mix_amount = mix_amount;
                            self.mix_type = mix_type as i32;
                            self.key_threshold = key_threshold;
                            self.key_softness = key_softness;
                            self.selected_syphon_server = selected_input - 1; // Convert back to -1-based
                            
                            // Handle input switch request
                            if let Some(source) = input_switch_requested {
                                self.input_switch_requested = Some(source);
                            }
                            
                            // Update shared state for mixing/keying
                            if let Ok(mut state) = self.shared_state.lock() {
                                state.block1.ch2_mix_amount = mix_amount;
                                state.block1.ch2_mix_type = mix_type as i32;
                                state.block1.ch2_key_threshold = key_threshold;
                                state.block1.ch2_key_soft = key_softness;
                            }
                            
                            if let Some(ref mut engine) = self.engine {
                                engine.set_hue_amount(hue_amount);
                            }
                            
                            if tap_pressed {
                                self.handle_tap_tempo();
                            }
                        }
                    }
                    _ => {}
                }
                return;
            }
        }
        
        // Handle output window events
        match event {
            WindowEvent::CloseRequested => {
                log::info!("[SIMPLE_APP] Output window close requested");
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                if let Some(ref mut engine) = self.engine {
                    engine.resize(size.width, size.height);
                }
            }
            WindowEvent::RedrawRequested => {
                if let Some(ref mut engine) = self.engine {
                    engine.render();
                }
            }
            _ => {}
        }
    }
    
    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        // Handle deferred input auto-start (must be done outside of event handlers)
        if let Some(server_name) = self.pending_input_autostart.take() {
            if let Some(ref mut engine) = self.engine {
                if server_name.is_empty() {
                    log::info!("[SIMPLE_APP] Auto-starting webcam");
                    engine.auto_start_webcam();
                    self.current_input_source = String::from("webcam");
                } else {
                    log::info!("[SIMPLE_APP] Auto-starting Syphon input from: {}", server_name);
                    #[cfg(target_os = "macos")]
                    {
                        engine.start_input1_syphon(&server_name);
                        self.current_input_source = format!("Syphon: {}", server_name);
                    }
                    #[cfg(not(target_os = "macos"))]
                    {
                        log::warn!("[SIMPLE_APP] Syphon input only available on macOS, falling back to webcam");
                        engine.auto_start_webcam();
                        self.current_input_source = String::from("webcam");
                    }
                }
            }
        }
        
        // Handle input switch requests from UI
        if let Some(source) = self.input_switch_requested.take() {
            if let Some(ref mut engine) = self.engine {
                if source == "webcam" {
                    log::info!("[SIMPLE_APP] Switching to webcam");
                    engine.auto_start_webcam();
                    self.current_input_source = String::from("webcam");
                } else {
                    log::info!("[SIMPLE_APP] Switching to Syphon: {}", source);
                    #[cfg(target_os = "macos")]
                    {
                        engine.start_input1_syphon(&source);
                        self.current_input_source = format!("Syphon: {}", source);
                    }
                    #[cfg(not(target_os = "macos"))]
                    {
                        log::warn!("[SIMPLE_APP] Syphon not available on this platform");
                    }
                }
            }
        }
        
        // Request continuous redraw for both windows
        if let Some(ref window) = self.output_window {
            window.request_redraw();
        }
        if let Some(ref window) = self.control_window {
            window.request_redraw();
        }
    }
}

/// Run the simple app
pub fn run_simple_app(
    _config: AppConfig,
    shared_state: Arc<Mutex<SharedState>>,
) -> anyhow::Result<()> {
    log::info!("[SIMPLE_MAIN] Starting simple feedback app with control window");
    
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    
    let mut app = SimpleApp::new(shared_state);
    event_loop.run_app(&mut app)?;
    
    Ok(())
}
