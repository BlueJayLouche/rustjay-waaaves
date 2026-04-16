//! # GUI Module
//!
//! ImGui-based control interface for the VJ application.
//! Provides real-time parameter control, preset management, and input configuration.

// Allow deprecated ComboBox API - imgui 0.12 uses the older API
#![allow(deprecated)]

use crate::config::{LayoutManager, ResolutionPreset, TabId};
use crate::core::{InputChangeRequest, OutputMode, PreviewSource, SharedState};
use crate::input::InputType;
use crate::midi::{cc_name, MidiEvent, MidiMapping, MidiMessageType};
use crate::midi::learn::{LearnableParam, LearnableParams};
use crate::midi::mapping::ParamRange;
use crate::params::preset::{PresetData, PresetManager};
use crate::params::{Block1Params, Block2Params, Block3Params};
use glam::{Vec3, Vec4};
use imgui::{CollapsingHeader, ComboBox, Condition, Drag, Ui};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

// =============================================================================
// CONSTANTS
// =============================================================================

/// Waveform names for LFO selection
pub const WAVEFORM_NAMES: &[&str] = &["Sine", "Triangle", "Ramp", "Saw", "Square"];

/// Beat divisions for tempo-synced LFOs
pub const BEAT_DIVISIONS: &[&str] = &["1/16", "1/8", "1/4", "1/2", "1", "2", "4", "8"];

/// FFT band names for audio modulation
pub const FFT_BAND_NAMES: &[&str] = &[
    "Sub Bass (20-60Hz)",
    "Bass (60-120Hz)",
    "Low Mid (120-250Hz)",
    "Mid (250-500Hz)",
    "High Mid (500-2kHz)",
    "High (2k-4kHz)",
    "Very High (4k-8kHz)",
    "Presence (8k-16kHz)",
];

/// Geometric overflow modes
pub const GEO_OVERFLOW_MODES: &[&str] = &["Clamp", "Toroid", "Mirror"];

/// Mix/blend types
pub const MIX_TYPES: &[&str] = &["Linear", "Additive", "Difference", "Multiplicative", "Dodge"];

/// Keying modes (OF-style: 0=Lumakey, 1=Chromakey)
pub const KEY_MODES: &[&str] = &["Lumakey", "Chromakey"];

// =============================================================================
// TYPE DEFINITIONS
// =============================================================================

/// Audio modulation state for a parameter
#[derive(Debug, Clone)]
pub struct ParamAudioModulation {
    pub enabled: bool,
    pub fft_band: i32,
    pub amount: f32,
    pub attack: f32,
    pub release: f32,
}

impl Default for ParamAudioModulation {
    fn default() -> Self {
        Self {
            enabled: false,
            fft_band: 0,
            amount: 0.0,
            attack: 0.01,
            release: 0.15,
        }
    }
}

/// LFO state for a parameter
#[derive(Debug, Clone)]
pub struct LfoState {
    pub enabled: bool,
    pub amplitude: f32,
    pub rate: f32,
    pub waveform: i32,       // 0=Sine, 1=Triangle, 2=Ramp, 3=Saw, 4=Square
    pub tempo_sync: bool,
    pub division: i32,       // 0=1/16, 1=1/8, 2=1/4, 3=1/2, 4=1, 5=2, 6=4, 7=8
    pub bank_index: i32,     // Which LFO bank (0-15) to use
}

impl Default for LfoState {
    fn default() -> Self {
        Self {
            enabled: false,
            amplitude: 0.0,
            rate: 0.5,
            waveform: 0,
            tempo_sync: false,
            division: 2, // 1/4 note default
            bank_index: 0, // Default to bank 0
        }
    }
}

/// Block 1 sub-tabs
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Block1Tab {
    Ch1Adjust,
    Ch2MixAndKey,
    Ch2Adjust,
    Fb1Parameters,
    Lfo,
}

/// Block 2 sub-tabs
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Block2Tab {
    InputAdjust,
    Fb2Parameters,
    Lfo,
}

/// Block 3 sub-tabs
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Block3Tab {
    Block1Reprocess,
    Block2Reprocess,
    MatrixMixer,
    FinalMix,
    Lfo,
}



/// Main tab selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MainTab {
    Block1,
    Block2,
    Block3,
    Macros,
    Inputs,
    Presets,
    Settings,
    Midi,
}

// =============================================================================
// CONTROL GUI STRUCT
// =============================================================================

/// Main GUI controller for the application
pub struct ControlGui {
    /// Shared state with the engine
    pub shared_state: Arc<Mutex<SharedState>>,
    /// Application configuration (for saving input settings)
    config: crate::config::AppConfig,
    /// Layout manager for saving/recalling window layouts
    layout_manager: LayoutManager,
    /// New layout name input
    layout_name_input: String,
    
    /// Show ImGui demo window
    pub show_demo: bool,
    
    /// Currently selected main tab
    pub selected_tab: MainTab,
    
    /// Block 1 sub-tab
    pub block1_tab: Block1Tab,
    /// Block 2 sub-tab
    pub block2_tab: Block2Tab,
    /// Block 3 sub-tab
    pub block3_tab: Block3Tab,
    
    // Preset management
    preset_manager: PresetManager,
    preset_name_input: String,
    preset_description_input: String,
    selected_bank: String,
    selected_preset_index: i32,
    preset_status_message: String,
    preset_status_timer: f32,
    new_bank_name_input: String,
    preset_filter_text: String,
    favorite_presets: Vec<String>, // Format: "bank_name/preset_name"
    
    // Parameter copies for editing (to reduce lock contention)
    pub block1_edit: Block1Params,
    pub block2_edit: Block2Params,
    pub block3_edit: Block3Params,
    
    // Input selection
    pub input1_type: InputType,
    pub input2_type: InputType,
    pub selected_webcam1: i32,
    pub selected_webcam2: i32,
    pub webcam_devices: Vec<String>,
    webcam_discovery: Option<JoinHandle<Vec<String>>>,
    
    // NDI source selection
    pub ndi_sources: Vec<String>,
    pub selected_ndi_source1: i32,
    pub selected_ndi_source2: i32,
    pub ndi_sources_dirty: bool,
    ndi_discovery: Option<JoinHandle<Vec<String>>>,
    // Saved NDI source names from config (for matching after discovery)
    pub saved_ndi_source1: String,
    pub saved_ndi_source2: String,
    
    // Syphon source selection (macOS only)
    pub syphon_sources: Vec<String>,
    pub selected_syphon_source1: i32,
    pub selected_syphon_source2: i32,
    pub syphon_sources_dirty: bool,
    syphon_discovery: Option<JoinHandle<Vec<String>>>,
    // Saved Syphon source names from config (for matching after discovery)
    pub saved_syphon_source1: String,
    pub saved_syphon_source2: String,

    // Spout source selection (Windows only)
    pub spout_sources: Vec<String>,
    pub selected_spout_source1: i32,
    pub selected_spout_source2: i32,
    /// Spout output sender name
    pub spout_output_name: String,

    // Audio device selection
    pub audio_devices: Vec<String>,
    pub selected_audio_device: i32,
    pub audio_device_dirty: bool,
    audio_discovery: Option<JoinHandle<Vec<String>>>,
    
    // Audio modulation UI state
    show_audio_panel: bool,
    selected_block1_param: i32,
    selected_block2_param: i32,
    selected_block3_param: i32,
    // Per-block audio mod settings (keyed by parameter name)
    block1_audio_mods: HashMap<String, ParamAudioModulation>,
    block2_audio_mods: HashMap<String, ParamAudioModulation>,
    block3_audio_mods: HashMap<String, ParamAudioModulation>,
    
    // LFO editor state
    selected_lfo_bank: i32,
    
    // Tempo / Tap Tempo state
    bpm: f32,
    bpm_enabled: bool,
    bpm_playing: bool,
    tap_times: Vec<f64>,
    last_tap_time: f64,
    beat_flash: f32,
    
    // LFO parameter states (amplitude, rate, waveform, sync)
    // Block 1 LFOs
    ch1_lfo_params: LfoParamGroup,      // CH1 Adjust LFOs
    ch2_mix_lfo_params: LfoParamGroup,  // CH2 Mix LFOs
    ch2_adj_lfo_params: LfoParamGroup,  // CH2 Adjust LFOs
    fb1_lfo_params: LfoParamGroup,      // FB1 LFOs
    
    // Block 2 LFOs
    b2_input_lfo_params: LfoParamGroup,
    fb2_lfo_params: LfoParamGroup,
    
    // Block 3 LFOs
    b3_b1_lfo_params: LfoParamGroup,
    b3_b2_lfo_params: LfoParamGroup,
    
    // Per-parameter LFO states for detailed LFO control
    block1_lfos: HashMap<String, LfoState>,
    block2_lfos: HashMap<String, LfoState>,
    block3_lfos: HashMap<String, LfoState>,
    
    // Status message
    status_message: String,
    status_timer: f32,
    
    // Main FPS counter (displayed in Settings tab)
    main_fps: f32,
    frame_times: [f32; 60], // Ring buffer for last 60 frames
    frame_time_index: usize, // Current index in ring buffer
    frame_time_count: usize, // Number of valid entries
    last_frame_time: std::time::Instant,
    
    // Preview window state
    show_preview_window: bool,
    preview_source: PreviewSource,
    preview_texture_id: Option<imgui::TextureId>,
    preview_sampled_color: [f32; 3],
    preview_crosshair_uv: [f32; 2], // Crosshair position in UV coordinates (0-1)
    preview_image_pos: [f32; 2],   // Position of preview image in window (for mouse picking)
    preview_image_size: [f32; 2],  // Size of preview image
    selected_key_target: i32,       // Selected key target for "Apply to Key"
    preview_fps: f32,              // FPS counter for preview
    preview_last_frame_time: std::time::Instant, // For FPS calculation
    
    // MIDI learn mode
    midi_learn_mode: bool,         // Is MIDI learn mode active?
    midi_learn_target: Option<String>, // Current parameter being learned (if any)
    
    // Deferred input startup (to avoid reentrant event handling during app init)
    startup_frame_count: u32,      // Frames since GUI started
    syphon_start_pending: [bool; 2], // [Input1, Input2] - true if Syphon should start after delay
}

/// LFO parameters for a group of controls
#[derive(Debug, Clone)]
pub struct LfoParamGroup {
    pub amplitude: f32,
    pub rate: f32,
    pub waveform: i32,
    pub tempo_sync: bool,
    pub division: i32,
}

impl Default for LfoParamGroup {
    fn default() -> Self {
        Self {
            amplitude: 0.0,
            rate: 0.15,
            waveform: 0,
            tempo_sync: false,
            division: 2, // 1/4 note default
        }
    }
}

// =============================================================================
// IMPLEMENTATION
// =============================================================================

impl ControlGui {
    /// Create a new ControlGui instance
    pub fn new(config: &crate::config::AppConfig, shared_state: Arc<Mutex<SharedState>>) -> anyhow::Result<Self> {
        let preset_manager = PresetManager::new();
        let selected_bank = preset_manager.get_current_bank().to_string();
        
        // Get initial state from shared state
        let (block1_edit, block2_edit, block3_edit) = {
            let state = shared_state.lock().unwrap();
            (state.block1, state.block2, state.block3)
        };
        
        let webcam_devices: Vec<String> = Vec::new();
        let audio_devices: Vec<String> = Vec::new();
        
        // Load input settings from config
        // Note: Input type values: 0=None, 1=Webcam, 2=NDI, 3=Syphon, 4=Spout, 5=VideoFile
        let input1_type = match config.inputs.input1_type {
            0 => InputType::None,
            1 => InputType::Webcam,
            2 => InputType::Ndi,
            3 => InputType::Syphon,
            4 => InputType::Spout,
            5 => InputType::VideoFile,
            _ => InputType::None,
        };
        let input2_type = match config.inputs.input2_type {
            0 => InputType::None,
            1 => InputType::Webcam,
            2 => InputType::Ndi,
            3 => InputType::Syphon,
            4 => InputType::Spout,
            5 => InputType::VideoFile,
            _ => InputType::None,
        };
        
        // Validate device indices
        let selected_webcam1 = if config.inputs.input1_device >= 0 {
            config.inputs.input1_device
        } else {
            -1
        };
        let selected_webcam2 = if config.inputs.input2_device >= 0 {
            config.inputs.input2_device
        } else {
            -1
        };
        
        // Store saved NDI/Syphon source names for later matching after discovery
        let saved_ndi_source1 = config.inputs.input1_ndi_source.clone();
        let saved_ndi_source2 = config.inputs.input2_ndi_source.clone();
        let saved_syphon_source1 = config.inputs.input1_syphon_source.clone();
        let saved_syphon_source2 = config.inputs.input2_syphon_source.clone();
        
        log::info!("Config: Input1={:?} (device {}), Input2={:?} (device {}), AutoStart={}",
            input1_type, selected_webcam1, input2_type, selected_webcam2, config.inputs.auto_start_webcams);
        log::info!("Saved sources: NDI1='{}', NDI2='{}', Syphon1='{}', Syphon2='{}'",
            saved_ndi_source1, saved_ndi_source2, saved_syphon_source1, saved_syphon_source2);
        
        let mut gui = Self {
            shared_state,
            config: config.clone(),
            show_demo: false,
            selected_tab: MainTab::Block1,
            block1_tab: Block1Tab::Ch1Adjust,
            block2_tab: Block2Tab::InputAdjust,
            block3_tab: Block3Tab::FinalMix,
            preset_manager,
            preset_name_input: String::new(),
            preset_description_input: String::new(),
            selected_bank,
            selected_preset_index: -1,
            preset_status_message: String::new(),
            preset_status_timer: 0.0,
            new_bank_name_input: String::new(),
            preset_filter_text: String::new(),
            favorite_presets: Vec::new(),
            block1_edit,
            block2_edit,
            block3_edit,
            input1_type,
            input2_type,
            selected_webcam1,
            selected_webcam2,
            webcam_devices,
            webcam_discovery: None,
            ndi_sources: Vec::new(),
            selected_ndi_source1: -1,
            selected_ndi_source2: -1,
            ndi_sources_dirty: true, // Mark as dirty to trigger initial scan
            ndi_discovery: None,
            saved_ndi_source1,
            saved_ndi_source2,
            syphon_sources: Vec::new(),
            selected_syphon_source1: -1,
            selected_syphon_source2: -1,
            syphon_sources_dirty: true,
            syphon_discovery: None,
            saved_syphon_source1,
            saved_syphon_source2,
            spout_sources: Vec::new(),
            selected_spout_source1: -1,
            selected_spout_source2: -1,
            spout_output_name: String::from("RustJay Waaaves"),
            audio_devices,
            selected_audio_device: -1,
            audio_device_dirty: true,
            audio_discovery: None,
            show_audio_panel: false,
            selected_block1_param: 0,
            selected_block2_param: 0,
            selected_block3_param: 0,
            block1_audio_mods: HashMap::new(),
            block2_audio_mods: HashMap::new(),
            block3_audio_mods: HashMap::new(),
            selected_lfo_bank: 0,
            
            // Tempo state
            bpm: 120.0,
            bpm_enabled: true,
            bpm_playing: true,
            tap_times: Vec::new(),
            last_tap_time: 0.0,
            beat_flash: 0.0,
            
            // Block 1 LFOs
            ch1_lfo_params: LfoParamGroup::default(),
            ch2_mix_lfo_params: LfoParamGroup::default(),
            ch2_adj_lfo_params: LfoParamGroup::default(),
            fb1_lfo_params: LfoParamGroup::default(),
            
            // Block 2 LFOs
            b2_input_lfo_params: LfoParamGroup::default(),
            fb2_lfo_params: LfoParamGroup::default(),
            
            // Block 3 LFOs
            b3_b1_lfo_params: LfoParamGroup::default(),
            b3_b2_lfo_params: LfoParamGroup::default(),
            
            // Per-parameter LFO states
            block1_lfos: HashMap::new(),
            block2_lfos: HashMap::new(),
            block3_lfos: HashMap::new(),
            
            status_message: String::new(),
            status_timer: 0.0,
            
            // Main FPS counter
            main_fps: 0.0,
            frame_times: [0.0; 60],
            frame_time_index: 0,
            frame_time_count: 0,
            last_frame_time: std::time::Instant::now(),
            
            // Layout manager for popped-out tabs
            layout_manager: LayoutManager::new(),
            layout_name_input: String::new(),
            
            // Preview window state
            show_preview_window: false,
            preview_source: PreviewSource::Block3,
            preview_texture_id: None,
            preview_sampled_color: [1.0, 1.0, 1.0],
            preview_crosshair_uv: [0.5, 0.5], // Start at center
            preview_image_pos: [0.0, 0.0],
            preview_image_size: [320.0, 180.0],
            selected_key_target: 0, // Default to first key target
            preview_fps: 0.0,
            preview_last_frame_time: std::time::Instant::now(),
            
            // MIDI learn mode
            midi_learn_mode: false,
            midi_learn_target: None,
            
            // Deferred input startup
            startup_frame_count: 0,
            syphon_start_pending: [false, false],
        };
        
        gui.refresh_devices();
        gui.refresh_syphon_sources();
        
        Ok(gui)
    }
    
    /// Set the preview texture ID (registered with imgui-wgpu)
    pub fn set_preview_texture_id(&mut self, texture_id: imgui::TextureId) {
        self.preview_texture_id = Some(texture_id);
        log::info!("Preview texture ID set: {:?}", texture_id);
    }
    
    /// Sync parameters from shared state to local copies
    pub fn sync_from_shared_state(&mut self) {
        if let Ok(state) = self.shared_state.lock() {
            self.block1_edit = state.block1;
            self.block2_edit = state.block2;
            self.block3_edit = state.block3;
        }
    }
    
    /// Sync parameters from local copies to shared state
    pub fn sync_to_shared_state(&mut self) {
        if let Ok(mut state) = self.shared_state.lock() {
            state.block1 = self.block1_edit;
            state.block2 = self.block2_edit;
            state.block3 = self.block3_edit;
            
            // Sync LFO assignments
            self.sync_lfo_map(&self.block1_lfos, &mut state.block1_lfo_map);
            self.sync_lfo_map(&self.block2_lfos, &mut state.block2_lfo_map);
            self.sync_lfo_map(&self.block3_lfos, &mut state.block3_lfo_map);
        }
    }
    
    /// Sync GUI LFO states to shared LFO parameter map
    fn sync_lfo_map(
        &self,
        gui_lfos: &HashMap<String, LfoState>,
        shared_map: &mut crate::core::LfoParameterMap,
    ) {
        shared_map.clear();
        for (param_id, lfo_state) in gui_lfos.iter() {
            if lfo_state.enabled {
                shared_map.insert(
                    param_id.clone(),
                    crate::core::LfoAssignment {
                        bank_index: lfo_state.bank_index,
                        amplitude: lfo_state.amplitude,
                        enabled: true,
                    },
                );
            }
        }
    }
    
    /// Show a status message
    pub fn show_status(&mut self, message: &str) {
        self.status_message = message.to_string();
        self.status_timer = 3.0; // Show for 3 seconds
    }
    
    /// Format Unix timestamp to readable string (YYYY-MM-DD HH:MM)
    fn format_timestamp(timestamp: u64) -> String {
        // Simple timestamp formatting without chrono
        let days_since_epoch = timestamp / 86400;
        let seconds_in_day = timestamp % 86400;
        let hours = seconds_in_day / 3600;
        let minutes = (seconds_in_day % 3600) / 60;
        
        // Approximate date calculation (good enough for display)
        let days_since_1970 = days_since_epoch as i64;
        let year = 1970 + (days_since_1970 / 365) as i64;
        let day_of_year = days_since_1970 % 365;
        
        // Simple month approximation
        let month = (day_of_year / 30 + 1).min(12).max(1);
        let day = (day_of_year % 30 + 1).min(28).max(1);
        
        format!("{:04}-{:02}-{:02} {:02}:{:02}", year, month, day, hours, minutes)
    }
    
    /// Refresh the list of available devices
    fn refresh_devices(&mut self) {
        self.begin_webcam_refresh();
        self.begin_audio_refresh();
        self.refresh_ndi_sources();
    }
    
    /// Refresh the list of available NDI sources
    fn refresh_ndi_sources(&mut self) {
        if self.ndi_discovery.is_some() {
            return;
        }
        self.ndi_sources_dirty = true;
        self.ndi_discovery = Some(std::thread::spawn(|| crate::input::list_ndi_sources(1000)));
    }
    
    /// Refresh the list of available Syphon servers (macOS only, requires syphon feature)
    #[cfg(all(target_os = "macos", feature = "syphon"))]
    fn refresh_syphon_sources(&mut self) {
        if self.syphon_discovery.is_some() {
            return;
        }
        self.syphon_sources_dirty = true;
        self.syphon_discovery = Some(std::thread::spawn(|| {
            use crate::input::SyphonServerInfo;
            let discovery = crate::input::SyphonDiscovery::new();
            let servers: Vec<SyphonServerInfo> = discovery.discover_servers();
            servers
                .into_iter()
                .map(|s| s.display_name().to_string())
                .collect()
        }));
    }
    
    /// Stub for non-macOS platforms or when syphon feature is disabled
    #[cfg(not(all(target_os = "macos", feature = "syphon")))]
    fn refresh_syphon_sources(&mut self) {
        self.syphon_sources.clear();
        self.syphon_sources_dirty = false;
    }

    /// Refresh Spout sender list (Windows only)
    fn refresh_spout_sources(&mut self) {
        self.spout_sources.clear();
        #[cfg(all(target_os = "windows", feature = "ipc-spout"))]
        {
            use crate::ipc::IpcDiscovery;
            let mut discovery = crate::ipc::spout::SpoutDiscovery::new();
            let sources = discovery.discover_sources(100);
            self.spout_sources = sources.into_iter().map(|s| s.name).collect();
            log::debug!("[GUI] Spout sources: {:?}", self.spout_sources);
        }
    }

    fn begin_webcam_refresh(&mut self) {
        if self.webcam_discovery.is_some() {
            return;
        }
        #[cfg(feature = "webcam")]
        {
            self.webcam_discovery = Some(std::thread::spawn(|| crate::input::webcam::list_cameras()));
        }
        #[cfg(not(feature = "webcam"))]
        {
            self.webcam_devices.clear();
        }
    }

    fn begin_audio_refresh(&mut self) {
        if self.audio_discovery.is_some() {
            return;
        }
        self.audio_device_dirty = true;
        self.audio_discovery = Some(std::thread::spawn(crate::audio::AudioInput::list_devices));
    }

    fn poll_async_discovery(&mut self) {
        if let Some(handle) = self.webcam_discovery.as_ref() {
            if handle.is_finished() {
                if let Some(handle) = self.webcam_discovery.take() {
                    match handle.join() {
                        Ok(devices) => {
                            log::info!("Refreshed device list: {} webcam(s) found", devices.len());
                            self.webcam_devices = devices;
                            self.clamp_webcam_selection();
                        }
                        Err(_) => {
                            log::error!("[GUI] Webcam discovery thread panicked");
                            self.webcam_devices.clear();
                            self.clamp_webcam_selection();
                        }
                    }
                }
            }
        }

        if let Some(handle) = self.audio_discovery.as_ref() {
            if handle.is_finished() {
                if let Some(handle) = self.audio_discovery.take() {
                    match handle.join() {
                        Ok(devices) => {
                            log::info!("Refreshed device list: {} audio device(s) found", devices.len());
                            self.audio_devices = devices;
                        }
                        Err(_) => {
                            log::error!("[GUI] Audio discovery thread panicked");
                            self.audio_devices.clear();
                        }
                    }
                    self.audio_device_dirty = false;
                    if self.selected_audio_device >= 0 && (self.selected_audio_device as usize) >= self.audio_devices.len() {
                        self.selected_audio_device = -1;
                    }
                }
            }
        }

        if let Some(handle) = self.ndi_discovery.as_ref() {
            if handle.is_finished() {
                if let Some(handle) = self.ndi_discovery.take() {
                    match handle.join() {
                        Ok(sources) => {
                            self.ndi_sources = sources;
                            self.ndi_sources_dirty = false;
                            self.match_saved_ndi_sources();
                        }
                        Err(_) => {
                            log::error!("[GUI] NDI discovery thread panicked");
                            self.ndi_sources.clear();
                            self.ndi_sources_dirty = false;
                        }
                    }
                    if self.selected_ndi_source1 >= 0 && (self.selected_ndi_source1 as usize) >= self.ndi_sources.len() {
                        self.selected_ndi_source1 = -1;
                    }
                    if self.selected_ndi_source2 >= 0 && (self.selected_ndi_source2 as usize) >= self.ndi_sources.len() {
                        self.selected_ndi_source2 = -1;
                    }
                }
            }
        }

        #[cfg(all(target_os = "macos", feature = "syphon"))]
        if let Some(handle) = self.syphon_discovery.as_ref() {
            if handle.is_finished() {
                if let Some(handle) = self.syphon_discovery.take() {
                    match handle.join() {
                        Ok(sources) => {
                            self.syphon_sources = sources;
                            self.syphon_sources_dirty = false;
                            self.match_saved_syphon_sources();
                        }
                        Err(_) => {
                            log::error!("[GUI] Syphon discovery thread panicked");
                            self.syphon_sources.clear();
                            self.syphon_sources_dirty = false;
                        }
                    }
                    if self.selected_syphon_source1 >= 0 && (self.selected_syphon_source1 as usize) >= self.syphon_sources.len() {
                        self.selected_syphon_source1 = -1;
                    }
                    if self.selected_syphon_source2 >= 0 && (self.selected_syphon_source2 as usize) >= self.syphon_sources.len() {
                        self.selected_syphon_source2 = -1;
                    }
                }
            }
        }
    }

    fn clamp_webcam_selection(&mut self) {
        if self.selected_webcam1 >= 0 && (self.selected_webcam1 as usize) >= self.webcam_devices.len() {
            self.selected_webcam1 = -1;
        }
        if self.selected_webcam2 >= 0 && (self.selected_webcam2 as usize) >= self.webcam_devices.len() {
            self.selected_webcam2 = -1;
        }
    }

    fn match_saved_ndi_sources(&mut self) {
        if self.selected_ndi_source1 < 0 && !self.saved_ndi_source1.is_empty() {
            if let Some(idx) = self.ndi_sources.iter().position(|s| s == &self.saved_ndi_source1) {
                self.selected_ndi_source1 = idx as i32;
                log::info!("[GUI] Matched saved NDI source 1: {} at index {}", self.saved_ndi_source1, idx);
            }
        }
        if self.selected_ndi_source2 < 0 && !self.saved_ndi_source2.is_empty() {
            if let Some(idx) = self.ndi_sources.iter().position(|s| s == &self.saved_ndi_source2) {
                self.selected_ndi_source2 = idx as i32;
                log::info!("[GUI] Matched saved NDI source 2: {} at index {}", self.saved_ndi_source2, idx);
            }
        }
    }

    #[cfg(all(target_os = "macos", feature = "syphon"))]
    fn match_saved_syphon_sources(&mut self) {
        if self.selected_syphon_source1 < 0 && !self.saved_syphon_source1.is_empty() {
            if let Some(idx) = self.syphon_sources.iter().position(|s| s == &self.saved_syphon_source1) {
                self.selected_syphon_source1 = idx as i32;
                log::info!("[GUI] Matched saved Syphon source 1: {} at index {}", self.saved_syphon_source1, idx);
            }
        }
        if self.selected_syphon_source2 < 0 && !self.saved_syphon_source2.is_empty() {
            if let Some(idx) = self.syphon_sources.iter().position(|s| s == &self.saved_syphon_source2) {
                self.selected_syphon_source2 = idx as i32;
                log::info!("[GUI] Matched saved Syphon source 2: {} at index {}", self.saved_syphon_source2, idx);
            }
        }
    }
    
    /// Save current input settings to config file
    fn save_input_config(&self) {
        // Update config with current values
        // Input type values: 0=None, 1=Webcam, 2=NDI, 3=Syphon, 4=Spout, 5=VideoFile
        let input1_type_int = match self.input1_type {
            InputType::None => 0,
            InputType::Webcam => 1,
            InputType::Ndi => 2,
            InputType::Syphon => 3,
            InputType::Spout => 4,
            InputType::VideoFile => 5,
        };
        let input2_type_int = match self.input2_type {
            InputType::None => 0,
            InputType::Webcam => 1,
            InputType::Ndi => 2,
            InputType::Syphon => 3,
            InputType::Spout => 4,
            InputType::VideoFile => 5,
        };
        
        // Get current NDI source names if selected
        let ndi_source1 = if self.selected_ndi_source1 >= 0 && (self.selected_ndi_source1 as usize) < self.ndi_sources.len() {
            self.ndi_sources[self.selected_ndi_source1 as usize].clone()
        } else {
            self.saved_ndi_source1.clone()
        };
        let ndi_source2 = if self.selected_ndi_source2 >= 0 && (self.selected_ndi_source2 as usize) < self.ndi_sources.len() {
            self.ndi_sources[self.selected_ndi_source2 as usize].clone()
        } else {
            self.saved_ndi_source2.clone()
        };
        
        // Get current Syphon source names if selected
        let syphon_source1 = if self.selected_syphon_source1 >= 0 && (self.selected_syphon_source1 as usize) < self.syphon_sources.len() {
            self.syphon_sources[self.selected_syphon_source1 as usize].clone()
        } else {
            self.saved_syphon_source1.clone()
        };
        let syphon_source2 = if self.selected_syphon_source2 >= 0 && (self.selected_syphon_source2 as usize) < self.syphon_sources.len() {
            self.syphon_sources[self.selected_syphon_source2 as usize].clone()
        } else {
            self.saved_syphon_source2.clone()
        };
        
        // We can't modify self.config directly since it's used immutably,
        // so we save directly using the current values
        let mut config = crate::config::AppConfig::load_or_default();
        config.inputs.input1_type = input1_type_int;
        config.inputs.input2_type = input2_type_int;
        config.inputs.input1_device = self.selected_webcam1;
        config.inputs.input2_device = self.selected_webcam2;
        config.inputs.input1_ndi_source = ndi_source1;
        config.inputs.input2_ndi_source = ndi_source2;
        config.inputs.input1_syphon_source = syphon_source1;
        config.inputs.input2_syphon_source = syphon_source2;
        config.inputs.auto_start_webcams = true; // Once user sets up, auto-start is enabled
        
        if let Err(e) = config.save() {
            log::warn!("Failed to save input config: {}", e);
        } else {
            log::debug!("Input config saved: Input1={} (dev {}), Input2={} (dev {})",
                input1_type_int, self.selected_webcam1, input2_type_int, self.selected_webcam2);
        }
    }
    
    /// Auto-start inputs based on saved config (called once at startup)
    /// Only auto-starts webcams immediately. NDI and Syphon are deferred
    /// to avoid reentrant event handling issues during app initialization.
    pub fn auto_start_webcams(&mut self) {
        if !self.config.inputs.auto_start_webcams {
            log::info!("Auto-start inputs disabled in config");
            return;
        }
        
        log::info!("Auto-starting webcams...");
        
        // Only auto-start webcams during initialization
        // NDI and Syphon require manual start to avoid event loop issues
        if self.input1_type == InputType::Webcam && self.selected_webcam1 >= 0 {
            let device_index = self.selected_webcam1 as usize;
            if device_index < self.webcam_devices.len() {
                log::info!("Auto-starting Webcam 1 (device {}: {})", device_index, self.webcam_devices[device_index]);
            } else {
                log::info!("Auto-starting Webcam 1 using saved device index {}", device_index);
            }
            if let Ok(mut state) = self.shared_state.lock() {
                state.input1_change_request = crate::core::InputChangeRequest::StartWebcam { 
                    input_id: 1,
                    device_index,
                    width: 1280,
                    height: 720,
                    fps: 30,
                };
            }
        }
        
        if self.input2_type == InputType::Webcam && self.selected_webcam2 >= 0 {
            let device_index = self.selected_webcam2 as usize;
            if device_index < self.webcam_devices.len() {
                log::info!("Auto-starting Webcam 2 (device {}: {})", device_index, self.webcam_devices[device_index]);
            } else {
                log::info!("Auto-starting Webcam 2 using saved device index {}", device_index);
            }
            if let Ok(mut state) = self.shared_state.lock() {
                state.input2_change_request = crate::core::InputChangeRequest::StartWebcam { 
                    input_id: 2,
                    device_index,
                    width: 1280,
                    height: 720,
                    fps: 30,
                };
            }
        }
        
        // Mark Syphon inputs for deferred startup (to avoid event loop issues)
        // We wait ~2 seconds (120 frames at 60fps) before starting Syphon
        if self.input1_type == InputType::Syphon && !self.saved_syphon_source1.is_empty() {
            self.syphon_start_pending[0] = true;
            log::info!("Syphon Input 1 marked for deferred startup (source: {})", self.saved_syphon_source1);
        }
        if self.input2_type == InputType::Syphon && !self.saved_syphon_source2.is_empty() {
            self.syphon_start_pending[1] = true;
            log::info!("Syphon Input 2 marked for deferred startup (source: {})", self.saved_syphon_source2);
        }
        
        // Note: NDI is also deferred to avoid potential issues
        if self.input1_type == InputType::Ndi && !self.saved_ndi_source1.is_empty() {
            log::info!("NDI Input 1 configured but requires manual start (source: {})", self.saved_ndi_source1);
        }
        if self.input2_type == InputType::Ndi && !self.saved_ndi_source2.is_empty() {
            log::info!("NDI Input 2 configured but requires manual start (source: {})", self.saved_ndi_source2);
        }
    }
    
    /// Process deferred input startups (called every frame)
    /// This avoids reentrant event handling during app initialization
    fn process_deferred_startups(&mut self) {
        self.startup_frame_count += 1;
        
        // Wait ~120 frames (~2 seconds at 60fps) before starting deferred inputs
        // This gives the event loop time to stabilize
        if self.startup_frame_count == 120 {
            // Start Syphon Input 1 if pending
            if self.syphon_start_pending[0] {
                self.syphon_start_pending[0] = false;
                self.start_deferred_syphon(1);
            }
        }
        
        if self.startup_frame_count == 150 {
            // Start Syphon Input 2 if pending (slight delay after input 1)
            if self.syphon_start_pending[1] {
                self.syphon_start_pending[1] = false;
                self.start_deferred_syphon(2);
            }
        }
    }
    
    /// Start a deferred Syphon input (called after app is fully initialized)
    /// This avoids the reentrant event handling issue during startup
    pub fn start_deferred_syphon(&mut self, input_id: u8) {
        match input_id {
            1 if self.input1_type == InputType::Syphon && !self.saved_syphon_source1.is_empty() => {
                log::info!("Starting deferred Syphon Input 1: {}", self.saved_syphon_source1);
                if let Ok(mut state) = self.shared_state.lock() {
                    state.input1_change_request = crate::core::InputChangeRequest::StartSyphon {
                        input_id: 1,
                        server_name: self.saved_syphon_source1.clone(),
                    };
                }
            }
            2 if self.input2_type == InputType::Syphon && !self.saved_syphon_source2.is_empty() => {
                log::info!("Starting deferred Syphon Input 2: {}", self.saved_syphon_source2);
                if let Ok(mut state) = self.shared_state.lock() {
                    state.input2_change_request = crate::core::InputChangeRequest::StartSyphon {
                        input_id: 2,
                        server_name: self.saved_syphon_source2.clone(),
                    };
                }
            }
            _ => {
                log::warn!("Invalid deferred Syphon start request for input {}", input_id);
            }
        }
    }
    
    /// Send an input change request to the engine
    fn send_input_request(&mut self, request: crate::core::InputChangeRequest) {
        if let Ok(mut state) = self.shared_state.lock() {
            // Use input1_change_request as a general channel for output settings
            state.input1_change_request = request.clone();
            
            // Also update our local config copy so it gets saved
            match &request {
                crate::core::InputChangeRequest::SetVsync(enabled) => {
                    self.config.output_window.vsync = *enabled;
                    state.output_vsync = *enabled;
                }
                crate::core::InputChangeRequest::SetOutputFps(fps) => {
                    self.config.output_window.fps = *fps;
                    state.output_fps = *fps;
                }
                _ => {}
            }
        }
    }
    
    /// Build the complete UI
    pub fn build_ui(&mut self, ui: &mut Ui) {
        // Process deferred startups (e.g., Syphon inputs that need to wait for event loop stabilization)
        self.process_deferred_startups();
        self.poll_async_discovery();
        
        // Update FPS counter (average over last 60 frames for smooth display)
        let now = std::time::Instant::now();
        let delta = now.duration_since(self.last_frame_time).as_secs_f32();
        self.last_frame_time = now;
        if delta > 0.0 {
            let fps = 1.0 / delta;
            // Ring buffer: store at current index and advance
            self.frame_times[self.frame_time_index] = fps;
            self.frame_time_index = (self.frame_time_index + 1) % 60;
            if self.frame_time_count < 60 {
                self.frame_time_count += 1;
            }
            // Calculate average over valid entries
            if self.frame_time_count > 0 {
                let sum: f32 = self.frame_times[..self.frame_time_count].iter().sum();
                self.main_fps = sum / self.frame_time_count as f32;
            }
        }
        
        // Update status timer
        if self.status_timer > 0.0 {
            self.status_timer -= ui.io().delta_time;
            if self.status_timer < 0.0 {
                self.status_timer = 0.0;
                self.status_message.clear();
            }
        }
        
        // Update beat flash
        if self.beat_flash > 0.0 {
            self.beat_flash -= ui.io().delta_time;
            if self.beat_flash < 0.0 {
                self.beat_flash = 0.0;
            }
        }
        
        // Sync from shared state at start of frame
        self.sync_from_shared_state();
        
        // Build UI components
        self.build_menu_bar(ui);
        self.build_top_bar(ui);
        self.build_main_tabs(ui);
        
        // Check if MIDI learning completed (do this every frame, not just in MIDI tab)
        self.check_midi_learn_completion();
        
        // Note: Demo window removed for cleaner UI
        
        // Sync back to shared state at end of frame
        self.sync_to_shared_state();
    }
    
    /// Build the menu bar
    fn build_menu_bar(&mut self, ui: &Ui) {
        ui.menu_bar(|| {
            ui.menu("File", || {
                if ui.menu_item("Exit") {
                    // Exit handled by main loop
                }
            });
            
            // View menu - currently empty, demo window removed for cleaner UI
            ui.menu("View", || {
                // Placeholder for view options
            });
        });
    }
    
    /// Build the top bar with preset controls and status
    fn build_top_bar(&mut self, ui: &Ui) {
        ui.group(|| {
            // Preset section
            self.build_preset_section(ui);
            
            // Status message
            if !self.status_message.is_empty() {
                ui.same_line_with_pos(600.0);
                ui.text_colored([0.0, 1.0, 0.0, 1.0], &self.status_message);
            }
        });
        
        ui.separator();
    }
    
    /// Build preset management section - compact quick access bar
    fn build_preset_section(&mut self, ui: &Ui) {
        // Show current bank and quick preset access
        let current_bank = self.preset_manager.get_current_bank();
        ui.text_colored([0.5, 0.8, 1.0, 1.0], &format!("Bank: {}", current_bank));
        
        ui.same_line();
        ui.separator();
        ui.same_line();
        
        // Quick load dropdown
        let preset_names = self.preset_manager.get_preset_names();
        let load_preview = if self.selected_preset_index >= 0 && 
                              (self.selected_preset_index as usize) < preset_names.len() {
            preset_names[self.selected_preset_index as usize].clone()
        } else {
            "Quick Load...".to_string()
        };
        
        let mut selected_idx = self.selected_preset_index;
        ui.set_next_item_width(180.0);
        ComboBox::new(ui, "##quick_load_preset")
            .preview_value(&load_preview)
            .build(|| {
                for (idx, name) in preset_names.iter().enumerate() {
                    if ui.selectable_config(name)
                        .selected(idx == selected_idx as usize)
                        .build() {
                        selected_idx = idx as i32;
                    }
                }
            });
        
        if selected_idx != self.selected_preset_index {
            self.selected_preset_index = selected_idx;
            // Auto-load on selection
            if self.selected_preset_index >= 0 {
                let idx = self.selected_preset_index as usize;
                if idx < preset_names.len() {
                    let name = &preset_names[idx];
                    match self.preset_manager.load_preset(name) {
                        Ok(data) => {
                            self.block1_edit = data.block1;
                            self.block2_edit = data.block2;
                            self.block3_edit = data.block3;
                            
                            if let Ok(mut state) = self.shared_state.lock() {
                                // Restore LFO banks
                                for (idx, lfo_bank) in data.lfo_banks.iter().enumerate() {
                                    if idx < state.lfo_banks.len() {
                                        state.lfo_banks[idx] = *lfo_bank;
                                    }
                                }
                                
                                state.block1_modulations = data.block1_modulations;
                                state.block2_modulations = data.block2_modulations;
                                state.block3_modulations = data.block3_modulations;
                                state.audio.amplitude = data.audio.amplitude;
                                state.audio.smoothing = data.audio.smoothing;
                                state.audio.normalization = data.audio.normalization;
                                state.audio.pink_compensation = data.audio.pink_compensation;
                                state.bpm = data.tempo.bpm;
                                
                                // Restore MIDI mappings
                                state.midi.mappings.clear();
                                for mapping in &data.midi_mappings {
                                    state.midi.mappings.insert(mapping.param_id.clone(), mapping.clone());
                                }
                            }
                            
                            self.sync_to_shared_state();
                            self.show_status(&format!("Loaded: {}", name));
                        }
                        Err(e) => self.show_status(&format!("Load failed: {}", e)),
                    }
                }
            }
        }
        
        ui.same_line();
        ui.separator();
        ui.same_line();
        
        // Quick save with timestamp
        if ui.button("💾 Quick Save") {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            let name = format!("Preset_{}", now);
            self.preset_name_input = name.clone();
            // Trigger save immediately
            let _ = self.save_current_preset(&name);
        }
        if ui.is_item_hovered() {
            ui.tooltip_text("Save current state with auto-generated name");
        }
        
        ui.same_line();
        ui.text("|");
        ui.same_line();
        
        // Link to full presets tab
        ui.text_colored([0.9, 0.5, 0.7, 1.0], "→ Full presets in Presets tab");
    }
    
    /// Build main tab bar
    fn build_main_tabs(&mut self, ui: &Ui) {
        let tab_labels = ["Block 1", "Block 2", "Block 3", "Macros", "Inputs", "Presets", "Settings", "MIDI"];
        let tab_ids = [
            TabId::Block1,
            TabId::Block2,
            TabId::Block3,
            TabId::Macros,
            TabId::Inputs,
            TabId::Presets,
            TabId::Settings,
            TabId::Midi,
        ];
        let mut selected_tab_idx = self.selected_tab as usize;
        let mut context_menu_tab: Option<TabId> = None;
        
        if let Some(_tab_bar) = ui.tab_bar("##main_tabs") {
            for (idx, (label, tab_id)) in tab_labels.iter().zip(tab_ids.iter()).enumerate() {
                let is_selected = idx == selected_tab_idx;
                let is_popped = self.layout_manager.current().is_popped(tab_id);
                
                // Build tab item label with optional pop-out indicator
                let tab_label = if is_popped {
                    format!("{} ⧉", label)  // Show indicator when popped
                } else {
                    label.to_string()
                };
                
                // Build tab item and check if clicked
                if let Some(_tab) = ui.tab_item(&tab_label) {
                    if !is_selected {
                        selected_tab_idx = idx;
                        self.selected_tab = match idx {
                            0 => MainTab::Block1,
                            1 => MainTab::Block2,
                            2 => MainTab::Block3,
                            3 => MainTab::Macros,
                            4 => MainTab::Inputs,
                            5 => MainTab::Presets,
                            6 => MainTab::Settings,
                            7 => MainTab::Midi,
                            _ => MainTab::Block1,
                        };
                    }
                }
                
                // Check for right-click on tab item
                if ui.is_item_hovered() && ui.is_mouse_released(imgui::MouseButton::Right) {
                    ui.open_popup(&format!("##context_{:?}", tab_id));
                }
                
                // Context menu for this tab
                ui.popup(&format!("##context_{:?}", tab_id), || {
                    if ui.menu_item("Pop Out") {
                        if !is_popped {
                            self.layout_manager.current_mut().pop_tab(*tab_id);
                            if let Err(e) = self.layout_manager.auto_save() {
                                log::warn!("Failed to save layout: {}", e);
                            }
                        }
                    }
                    if ui.menu_item_config("Dock").enabled(is_popped).build() {
                        if is_popped {
                            self.layout_manager.current_mut().dock_tab(tab_id);
                            if let Err(e) = self.layout_manager.auto_save() {
                                log::warn!("Failed to save layout: {}", e);
                            }
                        }
                    }
                });
            }
        }
        
        // Build content based on selected tab
        match self.selected_tab {
            MainTab::Block1 => self.build_block1_panel(ui),
            MainTab::Block2 => self.build_block2_panel(ui),
            MainTab::Block3 => self.build_block3_panel(ui),
            MainTab::Macros => self.build_macros_panel(ui),
            MainTab::Inputs => self.build_inputs_panel(ui),
            MainTab::Presets => self.build_presets_panel(ui),
            MainTab::Settings => self.build_settings_panel(ui),
            MainTab::Midi => self.build_midi_panel(ui),
        }
        
        // Render popped-out tab windows
        self.render_popped_tabs(ui);
        
        // Audio panel button at bottom
        ui.separator();
        
        if ui.button("Audio Reactivity") {
            self.show_audio_panel = !self.show_audio_panel;
        }
        
        ui.same_line();
        
        if ui.button("Preview & Color Picker") {
            self.show_preview_window = !self.show_preview_window;
        }
        
        if self.show_audio_panel {
            match self.selected_tab {
                MainTab::Block1 => self.draw_block1_audio_panel(ui),
                MainTab::Block2 => self.draw_block2_audio_panel(ui),
                MainTab::Block3 => self.draw_block3_audio_panel(ui),
                _ => {}
            }
        }
        
        // Draw preview window if enabled
        if self.show_preview_window {
            // Get frame count from shared state for throttling
            let frame_count = if let Ok(state) = self.shared_state.lock() {
                state.frame_count
            } else {
                0
            };
            self.draw_preview_window(ui, frame_count);
        }
    }
    
    /// Render popped-out tab windows
    fn render_popped_tabs(&mut self, ui: &Ui) {
        // Collect tabs to render (to avoid borrow issues)
        let popped_tabs: Vec<TabId> = self.layout_manager.current().popped_tabs.keys().cloned().collect();
        let mut layout_changed = false;
        
        // Check if we should force positions (e.g., after loading a layout)
        let force_positions = self.layout_manager.should_force_positions();
        if force_positions {
            log::debug!("Forcing window positions for layout load");
        }
        
        for tab_id in popped_tabs {
            let window_state = self.layout_manager.current().get_window_state(&tab_id);
            let title = tab_id.window_title();
            let bg_color = tab_id.bg_color();
            let border_color = tab_id.border_color();
            
            let mut opened = true;
            let mut new_pos: Option<[f32; 2]> = None;
            let mut new_size: Option<[f32; 2]> = None;
            
            // Apply color styling by pushing style vars
            let _style_bg = ui.push_style_color(imgui::StyleColor::WindowBg, bg_color);
            let _style_border = ui.push_style_color(imgui::StyleColor::Border, border_color);
            let _style_title_bg = ui.push_style_color(imgui::StyleColor::TitleBg, border_color);
            let _style_title_bg_active = ui.push_style_color(imgui::StyleColor::TitleBgActive, border_color);
            
            // Choose condition: Always if forcing positions, otherwise FirstUseEver
            let condition = if force_positions {
                log::debug!("Forcing position for {:?}: pos=[{:.1}, {:.1}] size=[{:.1}, {:.1}]", 
                    tab_id, window_state.pos_x, window_state.pos_y, window_state.width, window_state.height);
                Condition::Always
            } else {
                Condition::FirstUseEver
            };
            
            // Build window with saved position/size
            ui.window(&title)
                .size([window_state.width, window_state.height], condition)
                .position([window_state.pos_x, window_state.pos_y], condition)
                .opened(&mut opened)
                .build(|| {
                    // Render tab content based on ID
                    match tab_id {
                        TabId::Block1 => self.build_block1_panel(ui),
                        TabId::Block2 => self.build_block2_panel(ui),
                        TabId::Block3 => self.build_block3_panel(ui),
                        TabId::Macros => self.build_macros_panel(ui),
                        TabId::Inputs => self.build_inputs_panel(ui),
                        TabId::Presets => self.build_presets_panel(ui),
                        TabId::Settings => self.build_settings_panel(ui),
                        TabId::Midi => self.build_midi_panel(ui),
                        // Sub-tabs not supported for pop-out yet
                        _ => {
                            ui.text("This tab cannot be popped out.");
                            ui.text("Pop out the parent tab instead.");
                        }
                    }
                    
                    // Capture window state inside the closure
                    new_pos = Some(ui.window_pos());
                    new_size = Some(ui.window_size());
                });
            
            // Style colors are automatically popped when _style vars go out of scope
            
            // Update window state if we got valid position/size
            if let (Some(pos), Some(size)) = (new_pos, new_size) {
                // Check if position or size changed significantly
                let pos_changed = (pos[0] - window_state.pos_x).abs() > 1.0
                    || (pos[1] - window_state.pos_y).abs() > 1.0;
                let size_changed = (size[0] - window_state.width).abs() > 1.0
                    || (size[1] - window_state.height).abs() > 1.0;
                
                if pos_changed || size_changed {
                    log::debug!("Window {:?} changed: pos={:?}, size={:?}", tab_id, pos, size);
                    self.layout_manager.current_mut().update_from_imgui(
                        &tab_id,
                        pos,
                        size,
                        false
                    );
                    layout_changed = true;
                }
            }
            
            // If window was closed, dock the tab
            if !opened {
                self.layout_manager.current_mut().dock_tab(&tab_id);
                layout_changed = true;
            }
        }
        
        // Save layout if anything changed
        if layout_changed {
            if let Err(e) = self.layout_manager.auto_save() {
                log::warn!("Failed to save layout: {}", e);
            } else {
                log::debug!("Layout saved successfully");
            }
        }
    }
    
    /// Build Block 1 panel with sub-tabs
    fn build_block1_panel(&mut self, ui: &Ui) {
        // Sub-tab bar
        let subtab_labels = ["Ch 1 Adjust", "Ch 2 Mix & Key", "Ch 2 Adjust", "FB1", "LFO"];
        let mut subtab_idx = self.block1_tab as usize;
        
        if let Some(_tab_bar) = ui.tab_bar("##block1_tabs") {
            for (idx, label) in subtab_labels.iter().enumerate() {
                let is_selected = idx == subtab_idx;
                
                if let Some(_tab) = ui.tab_item(label) {
                    if !is_selected {
                        subtab_idx = idx;
                        self.block1_tab = match idx {
                            0 => Block1Tab::Ch1Adjust,
                            1 => Block1Tab::Ch2MixAndKey,
                            2 => Block1Tab::Ch2Adjust,
                            3 => Block1Tab::Fb1Parameters,
                            4 => Block1Tab::Lfo,
                            _ => Block1Tab::Ch1Adjust,
                        };
                    }
                }
            }
        }
        
        // Build content based on sub-tab
        match self.block1_tab {
            Block1Tab::Ch1Adjust => self.build_block1_ch1_adjust(ui),
            Block1Tab::Ch2MixAndKey => self.build_block1_ch2_mix_key(ui),
            Block1Tab::Ch2Adjust => self.build_block1_ch2_adjust(ui),
            Block1Tab::Fb1Parameters => self.build_block1_fb1_params(ui),
            Block1Tab::Lfo => self.build_block1_lfo(ui),
        }
    }
    
    /// Build Block 1 Channel 1 Adjust panel
    fn build_block1_ch1_adjust(&mut self, ui: &Ui) {
        // Extract config values and MIDI learn state before borrowing self mutably
        let show_osc = self.config.show_osc_addresses;
        let learn_mode = self.midi_learn_mode;
        let learn_target = self.midi_learn_target.clone();
        
        // Collect clicks for deferred processing
        let mut midi_learn_clicks: Vec<(String, f32, f32)> = Vec::new();
        
        let p = &mut self.block1_edit;
        
        // Input selection
        ui.text("Input Source:");
        ui.same_line();
        let input_options = ["Input 1", "Input 2"];
        let mut input_idx = p.ch1_input_select as usize;
        let preview = input_options[input_idx].to_string();
        ComboBox::new(ui, "##ch1_input_select")
            .preview_value(&preview)
            .build(|| {
                for (idx, opt) in input_options.iter().enumerate() {
                    if ui.selectable_config(opt).selected(idx == input_idx).build() {
                        input_idx = idx;
                    }
                }
            });
        p.ch1_input_select = input_idx.clamp(0, 1) as i32;
        
        ui.separator();
        
        // Helper closure for OSC tooltips
        let osc_tooltip = |ui: &Ui, address: &str, value: Option<f32>| {
            if show_osc && ui.is_item_hovered() {
                let mut tooltip = format!("OSC: {}", address);
                if let Some(val) = value {
                    tooltip.push_str(&format!("\nValue: {:.3}", val));
                }
                ui.tooltip_text(tooltip);
            }
        };
        
        // Helper to check if a parameter is being learned
        let is_learning = |param_id: &str| -> bool {
            learn_mode && learn_target.as_deref() == Some(param_id)
        };
        
        // Geometry section
        if CollapsingHeader::new("Geometry").default_open(true).build(ui) {
            // X Displace with MIDI learn
            let _text_color = if is_learning("block1.ch1_x_displace") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("X Displace##ch1").speed(0.01).range(-2.0, 2.0).build(ui, &mut p.ch1_x_displace);
            drop(_text_color);
            osc_tooltip(ui, "/block1/ch1/x_displace", Some(p.ch1_x_displace));
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block1.ch1_x_displace".to_string(), -2.0, 2.0));
            }
            
            // Y Displace with MIDI learn
            let _text_color = if is_learning("block1.ch1_y_displace") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Y Displace##ch1").speed(0.01).range(-2.0, 2.0).build(ui, &mut p.ch1_y_displace);
            drop(_text_color);
            osc_tooltip(ui, "/block1/ch1/y_displace", Some(p.ch1_y_displace));
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block1.ch1_y_displace".to_string(), -2.0, 2.0));
            }
            
            // Z Displace with MIDI learn
            let _text_color = if is_learning("block1.ch1_z_displace") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Z Displace##ch1").speed(0.01).range(0.0, 10.0).build(ui, &mut p.ch1_z_displace);
            drop(_text_color);
            osc_tooltip(ui, "/block1/ch1/z_displace", Some(p.ch1_z_displace));
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block1.ch1_z_displace".to_string(), 0.0, 10.0));
            }
            
            // Rotate with MIDI learn
            let _text_color = if is_learning("block1.ch1_rotate") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Rotate##ch1").speed(0.1).range(-360.0, 360.0).build(ui, &mut p.ch1_rotate);
            drop(_text_color);
            osc_tooltip(ui, "/block1/ch1/rotate", Some(p.ch1_rotate));
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block1.ch1_rotate".to_string(), -360.0, 360.0));
            }
            
            // Kaleidoscope Amount
            let _text_color = if is_learning("block1.ch1_kaleidoscope_amount") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Kaleidoscope Amount##ch1").speed(0.01).range(0.0, 1.0).build(ui, &mut p.ch1_kaleidoscope_amount);
            drop(_text_color);
            osc_tooltip(ui, "/block1/ch1/kaleidoscope_amount", Some(p.ch1_kaleidoscope_amount));
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block1.ch1_kaleidoscope_amount".to_string(), 0.0, 1.0));
            }
            
            // Kaleidoscope Slice
            let _text_color = if is_learning("block1.ch1_kaleidoscope_slice") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Kaleidoscope Slice##ch1").speed(0.01).range(0.0, 1.0).build(ui, &mut p.ch1_kaleidoscope_slice);
            drop(_text_color);
            osc_tooltip(ui, "/block1/ch1/kaleidoscope_slice", Some(p.ch1_kaleidoscope_slice));
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block1.ch1_kaleidoscope_slice".to_string(), 0.0, 1.0));
            }
            
            // Geo overflow
            let mut overflow_idx = p.ch1_geo_overflow as usize;
            let preview = GEO_OVERFLOW_MODES[overflow_idx].to_string();
            ComboBox::new(ui, "Overflow Mode##ch1")
                .preview_value(&preview)
                .build(|| {
                    for (idx, opt) in GEO_OVERFLOW_MODES.iter().enumerate() {
                        if ui.selectable_config(opt).selected(idx == overflow_idx).build() {
                            overflow_idx = idx;
                        }
                    }
                });
            p.ch1_geo_overflow = overflow_idx.clamp(0, GEO_OVERFLOW_MODES.len() - 1) as i32;
        }
        
        // Color section
        if CollapsingHeader::new("Color").default_open(true).build(ui) {
            // HSB Attenuate X
            let _text_color = if is_learning("block1.ch1_hsb_attenuate_x") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            let mut hsb_x = p.ch1_hsb_attenuate_x;
            Drag::new("HSB Attenuate H##ch1").speed(0.01).range(0.0, 2.0).build(ui, &mut hsb_x);
            drop(_text_color);
            p.ch1_hsb_attenuate_x = hsb_x;
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block1.ch1_hsb_attenuate_x".to_string(), 0.0, 2.0));
            }
            
            // HSB Attenuate Y
            let _text_color = if is_learning("block1.ch1_hsb_attenuate_y") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            let mut hsb_y = p.ch1_hsb_attenuate_y;
            Drag::new("HSB Attenuate S##ch1").speed(0.01).range(0.0, 2.0).build(ui, &mut hsb_y);
            drop(_text_color);
            p.ch1_hsb_attenuate_y = hsb_y;
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block1.ch1_hsb_attenuate_y".to_string(), 0.0, 2.0));
            }
            
            // HSB Attenuate Z
            let _text_color = if is_learning("block1.ch1_hsb_attenuate_z") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            let mut hsb_z = p.ch1_hsb_attenuate_z;
            Drag::new("HSB Attenuate B##ch1").speed(0.01).range(0.0, 2.0).build(ui, &mut hsb_z);
            drop(_text_color);
            p.ch1_hsb_attenuate_z = hsb_z;
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block1.ch1_hsb_attenuate_z".to_string(), 0.0, 2.0));
            }
            
            // Update the Vec3 from components
            p.ch1_hsb_attenuate = Vec3::new(p.ch1_hsb_attenuate_x, p.ch1_hsb_attenuate_y, p.ch1_hsb_attenuate_z);
            
            ui.checkbox("Hue Invert##ch1", &mut p.ch1_hue_invert);
            ui.checkbox("Saturation Invert##ch1", &mut p.ch1_saturation_invert);
            ui.checkbox("Brightness Invert##ch1", &mut p.ch1_bright_invert);
            ui.checkbox("RGB Invert##ch1", &mut p.ch1_rgb_invert);
            
            ui.checkbox("Solarize##ch1", &mut p.ch1_solarize);
            ui.checkbox("Posterize##ch1", &mut p.ch1_posterize_switch);
            if p.ch1_posterize_switch {
                let _text_color = if is_learning("block1.ch1_posterize") {
                    Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
                } else { None };
                Drag::new("Posterize Levels##ch1").speed(0.1).range(2.0, 32.0).build(ui, &mut p.ch1_posterize);
                drop(_text_color);
                if learn_mode && ui.is_item_clicked() { 
                    midi_learn_clicks.push(("block1.ch1_posterize".to_string(), 2.0, 32.0));
                }
            }
        }
        
        // Filters section
        if CollapsingHeader::new("Filters").default_open(true).build(ui) {
            // Blur Amount
            let _text_color = if is_learning("block1.ch1_blur_amount") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Blur Amount##ch1").speed(0.01).range(0.0, 1.0).build(ui, &mut p.ch1_blur_amount);
            drop(_text_color);
            osc_tooltip(ui, "/block1/ch1/blur_amount", Some(p.ch1_blur_amount));
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block1.ch1_blur_amount".to_string(), 0.0, 1.0));
            }
            
            // Blur Radius
            let _text_color = if is_learning("block1.ch1_blur_radius") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Blur Radius##ch1").speed(0.01).range(0.0, 5.0).build(ui, &mut p.ch1_blur_radius);
            drop(_text_color);
            osc_tooltip(ui, "/block1/ch1/blur_radius", Some(p.ch1_blur_radius));
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block1.ch1_blur_radius".to_string(), 0.0, 5.0));
            }
            
            // Sharpen Amount
            let _text_color = if is_learning("block1.ch1_sharpen_amount") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Sharpen Amount##ch1").speed(0.01).range(0.0, 1.0).build(ui, &mut p.ch1_sharpen_amount);
            drop(_text_color);
            osc_tooltip(ui, "/block1/ch1/sharpen_amount", Some(p.ch1_sharpen_amount));
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block1.ch1_sharpen_amount".to_string(), 0.0, 1.0));
            }
            
            // Sharpen Radius
            let _text_color = if is_learning("block1.ch1_sharpen_radius") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Sharpen Radius##ch1").speed(0.01).range(0.0, 5.0).build(ui, &mut p.ch1_sharpen_radius);
            drop(_text_color);
            osc_tooltip(ui, "/block1/ch1/sharpen_radius", Some(p.ch1_sharpen_radius));
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block1.ch1_sharpen_radius".to_string(), 0.0, 5.0));
            }
            
            // Filters Boost
            let _text_color = if is_learning("block1.ch1_filters_boost") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Filters Boost##ch1").speed(0.01).range(0.0, 2.0).build(ui, &mut p.ch1_filters_boost);
            drop(_text_color);
            osc_tooltip(ui, "/block1/ch1/filters_boost", Some(p.ch1_filters_boost));
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block1.ch1_filters_boost".to_string(), 0.0, 2.0));
            }
        }
        
        // Switches section
        if CollapsingHeader::new("Switches").default_open(false).build(ui) {
            ui.checkbox("H Mirror##ch1", &mut p.ch1_h_mirror);
            ui.checkbox("V Mirror##ch1", &mut p.ch1_v_mirror);
            ui.checkbox("H Flip##ch1", &mut p.ch1_h_flip);
            ui.checkbox("V Flip##ch1", &mut p.ch1_v_flip);
            ui.checkbox("HD Aspect##ch1", &mut p.ch1_hd_aspect_on);
        }
        
        // Process MIDI learn clicks (deferred to avoid borrow issues)
        for (param_id, min, max) in midi_learn_clicks {
            self.handle_midi_learn_click(&param_id, min, max);
        }
    }
    
    /// Build Block 1 Channel 2 Mix & Key panel
    fn build_block1_ch2_mix_key(&mut self, ui: &Ui) {
        // Extract config values and MIDI learn state before borrowing self mutably
        let show_osc = self.config.show_osc_addresses;
        let learn_mode = self.midi_learn_mode;
        let learn_target = self.midi_learn_target.clone();
        
        // Collect clicks for deferred processing
        let mut midi_learn_clicks: Vec<(String, f32, f32)> = Vec::new();
        
        let p = &mut self.block1_edit;
        
        // Helper closure for OSC tooltips
        let osc_tooltip = |ui: &Ui, address: &str, value: Option<f32>| {
            if show_osc && ui.is_item_hovered() {
                let mut tooltip = format!("OSC: {}", address);
                if let Some(val) = value {
                    tooltip.push_str(&format!("\nValue: {:.3}", val));
                }
                ui.tooltip_text(tooltip);
            }
        };
        
        // Helper to check if a parameter is being learned
        let is_learning = |param_id: &str| -> bool {
            learn_mode && learn_target.as_deref() == Some(param_id)
        };
        
        // Mix section
        if CollapsingHeader::new("Mix").default_open(true).build(ui) {
            // Mix amount with fine control (slower speed for precision near 0 and 1)
            let _text_color = if is_learning("block1.ch2_mix_amount") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Mix Amount##ch2mix").speed(0.002).range(0.0, 1.0).build(ui, &mut p.ch2_mix_amount);
            drop(_text_color);
            osc_tooltip(ui, "/block1/ch2/mix_amount", Some(p.ch2_mix_amount));
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block1.ch2_mix_amount".to_string(), 0.0, 1.0));
            }
            
            let mut mix_type = p.ch2_mix_type as usize;
            let preview = MIX_TYPES[mix_type].to_string();
            ComboBox::new(ui, "Mix Type##ch2")
                .preview_value(&preview)
                .build(|| {
                    for (idx, opt) in MIX_TYPES.iter().enumerate() {
                        if ui.selectable_config(opt).selected(idx == mix_type).build() {
                            mix_type = idx;
                        }
                    }
                });
            p.ch2_mix_type = mix_type.clamp(0, MIX_TYPES.len() - 1) as i32;
            
            let mut overflow = p.ch2_mix_overflow as usize;
            let preview = GEO_OVERFLOW_MODES[overflow].to_string();
            ComboBox::new(ui, "Mix Overflow##ch2")
                .preview_value(&preview)
                .build(|| {
                    for (idx, opt) in GEO_OVERFLOW_MODES.iter().enumerate() {
                        if ui.selectable_config(opt).selected(idx == overflow).build() {
                            overflow = idx;
                        }
                    }
                });
            p.ch2_mix_overflow = overflow.clamp(0, GEO_OVERFLOW_MODES.len() - 1) as i32;
        }
        
        // Key section - OF-style (always active)
        if CollapsingHeader::new("Key").default_open(true).build(ui) {
            // Key mode selector (OF: 0=lumakey, 1=chromakey)
            let mut key_mode = (p.ch2_key_mode.clamp(0, 1)) as usize;
            let key_modes = ["Lumakey", "Chromakey"];
            let preview = key_modes[key_mode].to_string();
            ComboBox::new(ui, "Key Mode##ch2")
                .preview_value(&preview)
                .build(|| {
                    for (idx, opt) in key_modes.iter().enumerate() {
                        if ui.selectable_config(*opt).selected(idx == key_mode).build() {
                            key_mode = idx;
                        }
                    }
                });
            p.ch2_key_mode = key_mode as i32;
            
            // Key value (OF uses -1.0 to 1.0 range)
            let mut key_color = [p.ch2_key_value_red, p.ch2_key_value_green, p.ch2_key_value_blue];
            
            if p.ch2_key_mode == 0 {
                // Lumakey mode - single slider controls all channels
                ui.text("Key Value:");
                let _text_color = if is_learning("block1.ch2_key_value_red") {
                    Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
                } else { None };
                Drag::new("Key Value##ch2").speed(0.01).range(-1.0, 1.0).build(ui, &mut key_color[0]);
                drop(_text_color);
                if learn_mode && ui.is_item_clicked() { 
                    midi_learn_clicks.push(("block1.ch2_key_value_red".to_string(), -1.0, 1.0));
                    midi_learn_clicks.push(("block1.ch2_key_value_green".to_string(), -1.0, 1.0));
                    midi_learn_clicks.push(("block1.ch2_key_value_blue".to_string(), -1.0, 1.0));
                }
                key_color[1] = key_color[0];
                key_color[2] = key_color[0];
            } else {
                // Chromakey mode - RGB sliders
                ui.text("Key Color (RGB -1 to 1):");
                let _text_color = if is_learning("block1.ch2_key_value_red") {
                    Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
                } else { None };
                Drag::new("Red##ch2key").speed(0.01).range(-1.0, 1.0).build(ui, &mut key_color[0]);
                drop(_text_color);
                if learn_mode && ui.is_item_clicked() { 
                    midi_learn_clicks.push(("block1.ch2_key_value_red".to_string(), -1.0, 1.0));
                }
                
                let _text_color = if is_learning("block1.ch2_key_value_green") {
                    Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
                } else { None };
                Drag::new("Green##ch2key").speed(0.01).range(-1.0, 1.0).build(ui, &mut key_color[1]);
                drop(_text_color);
                if learn_mode && ui.is_item_clicked() { 
                    midi_learn_clicks.push(("block1.ch2_key_value_green".to_string(), -1.0, 1.0));
                }
                
                let _text_color = if is_learning("block1.ch2_key_value_blue") {
                    Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
                } else { None };
                Drag::new("Blue##ch2key").speed(0.01).range(-1.0, 1.0).build(ui, &mut key_color[2]);
                drop(_text_color);
                if learn_mode && ui.is_item_clicked() { 
                    midi_learn_clicks.push(("block1.ch2_key_value_blue".to_string(), -1.0, 1.0));
                }
            }
            
            // Color preview (mapped to 0-1 for display)
            let preview_color = [
                (key_color[0] + 1.0) * 0.5,
                (key_color[1] + 1.0) * 0.5,
                (key_color[2] + 1.0) * 0.5,
                1.0, // Alpha
            ];
            ui.color_button("Key Color##ch2preview", preview_color);
            ui.same_line();
            ui.text("preview");
            
            p.ch2_key_value_red = key_color[0];
            p.ch2_key_value_green = key_color[1];
            p.ch2_key_value_blue = key_color[2];
            
            ui.separator();
            
            // Key Order dropdown
            let key_orders = ["Key First, Then Mix", "Mix First, Then Key"];
            let mut order_idx = p.ch2_key_order as usize;
            let preview = key_orders[order_idx.clamp(0, key_orders.len() - 1)].to_string();
            ComboBox::new(ui, "Key Order##ch2")
                .preview_value(&preview)
                .build(|| {
                    for (idx, opt) in key_orders.iter().enumerate() {
                        if ui.selectable_config(*opt).selected(idx == order_idx).build() {
                            order_idx = idx;
                        }
                    }
                });
            p.ch2_key_order = order_idx.clamp(0, key_orders.len() - 1) as i32;
            
            // Key threshold and soft (OF uses -1.0 to 1.0)
            ui.text("Key Parameters:");
            let _text_color = if is_learning("block1.ch2_key_threshold") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Threshold##ch2").speed(0.01).range(-1.0, 1.0).build(ui, &mut p.ch2_key_threshold);
            drop(_text_color);
            osc_tooltip(ui, "/block1/ch2/key_threshold", Some(p.ch2_key_threshold));
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block1.ch2_key_threshold".to_string(), -1.0, 1.0));
            }
            
            let _text_color = if is_learning("block1.ch2_key_soft") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Soft##ch2").speed(0.01).range(-1.0, 1.0).build(ui, &mut p.ch2_key_soft);
            drop(_text_color);
            osc_tooltip(ui, "/block1/ch2/key_soft", Some(p.ch2_key_soft));
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block1.ch2_key_soft".to_string(), -1.0, 1.0));
            }
        }
        
        // Process MIDI learn clicks (deferred to avoid borrow issues)
        for (param_id, min, max) in midi_learn_clicks {
            self.handle_midi_learn_click(&param_id, min, max);
        }
    }
    
    /// Build Block 1 Channel 2 Adjust panel
    fn build_block1_ch2_adjust(&mut self, ui: &Ui) {
        // Extract config values and MIDI learn state before borrowing self mutably
        let show_osc = self.config.show_osc_addresses;
        let learn_mode = self.midi_learn_mode;
        let learn_target = self.midi_learn_target.clone();
        
        // Collect clicks for deferred processing
        let mut midi_learn_clicks: Vec<(String, f32, f32)> = Vec::new();
        
        let p = &mut self.block1_edit;
        
        // Helper closure for OSC tooltips
        let osc_tooltip = |ui: &Ui, address: &str, value: Option<f32>| {
            if show_osc && ui.is_item_hovered() {
                let mut tooltip = format!("OSC: {}", address);
                if let Some(val) = value {
                    tooltip.push_str(&format!("\nValue: {:.3}", val));
                }
                ui.tooltip_text(tooltip);
            }
        };
        
        // Helper to check if a parameter is being learned
        let is_learning = |param_id: &str| -> bool {
            learn_mode && learn_target.as_deref() == Some(param_id)
        };
        
        // Input selection (moved here to match CH1 layout)
        ui.text("Input Source:");
        ui.same_line();
        let input_options = ["Input 1", "Input 2"];
        let mut input_idx = p.ch2_input_select as usize;
        let preview = input_options[input_idx].to_string();
        ComboBox::new(ui, "##ch2_input_select")
            .preview_value(&preview)
            .build(|| {
                for (idx, opt) in input_options.iter().enumerate() {
                    if ui.selectable_config(opt).selected(idx == input_idx).build() {
                        input_idx = idx;
                    }
                }
            });
        p.ch2_input_select = input_idx.clamp(0, 1) as i32;
        
        ui.separator();
        
        // Geometry section
        if CollapsingHeader::new("Geometry").default_open(true).build(ui) {
            let _text_color = if is_learning("block1.ch2_x_displace") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("X Displace##ch2adj").speed(0.01).range(-2.0, 2.0).build(ui, &mut p.ch2_x_displace);
            drop(_text_color);
            osc_tooltip(ui, "/block1/ch2/x_displace", Some(p.ch2_x_displace));
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block1.ch2_x_displace".to_string(), -2.0, 2.0));
            }
            
            let _text_color = if is_learning("block1.ch2_y_displace") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Y Displace##ch2adj").speed(0.01).range(-2.0, 2.0).build(ui, &mut p.ch2_y_displace);
            drop(_text_color);
            osc_tooltip(ui, "/block1/ch2/y_displace", Some(p.ch2_y_displace));
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block1.ch2_y_displace".to_string(), -2.0, 2.0));
            }
            
            let _text_color = if is_learning("block1.ch2_z_displace") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Z Displace##ch2adj").speed(0.01).range(0.0, 10.0).build(ui, &mut p.ch2_z_displace);
            drop(_text_color);
            osc_tooltip(ui, "/block1/ch2/z_displace", Some(p.ch2_z_displace));
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block1.ch2_z_displace".to_string(), 0.0, 10.0));
            }
            
            let _text_color = if is_learning("block1.ch2_rotate") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Rotate##ch2adj").speed(0.1).range(-360.0, 360.0).build(ui, &mut p.ch2_rotate);
            drop(_text_color);
            osc_tooltip(ui, "/block1/ch2/rotate", Some(p.ch2_rotate));
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block1.ch2_rotate".to_string(), -360.0, 360.0));
            }
            
            let _text_color = if is_learning("block1.ch2_kaleidoscope_amount") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Kaleidoscope Amount##ch2adj").speed(0.01).range(0.0, 1.0).build(ui, &mut p.ch2_kaleidoscope_amount);
            drop(_text_color);
            osc_tooltip(ui, "/block1/ch2/kaleidoscope_amount", Some(p.ch2_kaleidoscope_amount));
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block1.ch2_kaleidoscope_amount".to_string(), 0.0, 1.0));
            }
            
            let _text_color = if is_learning("block1.ch2_kaleidoscope_slice") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Kaleidoscope Slice##ch2adj").speed(0.01).range(0.0, 1.0).build(ui, &mut p.ch2_kaleidoscope_slice);
            drop(_text_color);
            osc_tooltip(ui, "/block1/ch2/kaleidoscope_slice", Some(p.ch2_kaleidoscope_slice));
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block1.ch2_kaleidoscope_slice".to_string(), 0.0, 1.0));
            }
            
            let mut overflow_idx = p.ch2_geo_overflow as usize;
            let preview = GEO_OVERFLOW_MODES[overflow_idx].to_string();
            ComboBox::new(ui, "Overflow Mode##ch2adj")
                .preview_value(&preview)
                .build(|| {
                    for (idx, opt) in GEO_OVERFLOW_MODES.iter().enumerate() {
                        if ui.selectable_config(opt).selected(idx == overflow_idx).build() {
                            overflow_idx = idx;
                        }
                    }
                });
            p.ch2_geo_overflow = overflow_idx.clamp(0, GEO_OVERFLOW_MODES.len() - 1) as i32;
        }
        
        // Color section
        if CollapsingHeader::new("Color").default_open(true).build(ui) {
            let mut hsb = [p.ch2_hsb_attenuate.x, p.ch2_hsb_attenuate.y, p.ch2_hsb_attenuate.z];
            let _text_color = if is_learning("block1.ch2_hsb_attenuate_x") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("HSB Attenuate##ch2adj").speed(0.01).range(0.0, 2.0).build_array(ui, &mut hsb);
            drop(_text_color);
            p.ch2_hsb_attenuate = Vec3::new(hsb[0], hsb[1], hsb[2]);
            // Sync individual components for LFO modulation
            p.ch2_hsb_attenuate_x = hsb[0];
            p.ch2_hsb_attenuate_y = hsb[1];
            p.ch2_hsb_attenuate_z = hsb[2];
            osc_tooltip(ui, "/block1/ch2/hsb_attenuate", Some(hsb[0]));
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block1.ch2_hsb_attenuate_x".to_string(), 0.0, 2.0));
                midi_learn_clicks.push(("block1.ch2_hsb_attenuate_y".to_string(), 0.0, 2.0));
                midi_learn_clicks.push(("block1.ch2_hsb_attenuate_z".to_string(), 0.0, 2.0));
            }
            
            ui.checkbox("Hue Invert##ch2adj", &mut p.ch2_hue_invert);
            ui.checkbox("Saturation Invert##ch2adj", &mut p.ch2_saturation_invert);
            ui.checkbox("Brightness Invert##ch2adj", &mut p.ch2_bright_invert);
            ui.checkbox("RGB Invert##ch2adj", &mut p.ch2_rgb_invert);
            
            ui.checkbox("Solarize##ch2adj", &mut p.ch2_solarize);
            ui.checkbox("Posterize##ch2adj", &mut p.ch2_posterize_switch);
            if p.ch2_posterize_switch {
                let _text_color = if is_learning("block1.ch2_posterize") {
                    Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
                } else { None };
                Drag::new("Posterize Levels##ch2adj").speed(0.1).range(2.0, 32.0).build(ui, &mut p.ch2_posterize);
                drop(_text_color);
                osc_tooltip(ui, "/block1/ch2/posterize", Some(p.ch2_posterize));
                if learn_mode && ui.is_item_clicked() { 
                    midi_learn_clicks.push(("block1.ch2_posterize".to_string(), 2.0, 32.0));
                }
            }
        }
        
        // Filters section
        if CollapsingHeader::new("Filters").default_open(true).build(ui) {
            let _text_color = if is_learning("block1.ch2_blur_amount") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Blur Amount##ch2adj").speed(0.01).range(0.0, 1.0).build(ui, &mut p.ch2_blur_amount);
            drop(_text_color);
            osc_tooltip(ui, "/block1/ch2/blur_amount", Some(p.ch2_blur_amount));
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block1.ch2_blur_amount".to_string(), 0.0, 1.0));
            }
            
            let _text_color = if is_learning("block1.ch2_blur_radius") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Blur Radius##ch2adj").speed(0.01).range(0.0, 5.0).build(ui, &mut p.ch2_blur_radius);
            drop(_text_color);
            osc_tooltip(ui, "/block1/ch2/blur_radius", Some(p.ch2_blur_radius));
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block1.ch2_blur_radius".to_string(), 0.0, 5.0));
            }
            
            let _text_color = if is_learning("block1.ch2_sharpen_amount") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Sharpen Amount##ch2adj").speed(0.01).range(0.0, 1.0).build(ui, &mut p.ch2_sharpen_amount);
            drop(_text_color);
            osc_tooltip(ui, "/block1/ch2/sharpen_amount", Some(p.ch2_sharpen_amount));
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block1.ch2_sharpen_amount".to_string(), 0.0, 1.0));
            }
            
            let _text_color = if is_learning("block1.ch2_sharpen_radius") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Sharpen Radius##ch2adj").speed(0.01).range(0.0, 5.0).build(ui, &mut p.ch2_sharpen_radius);
            drop(_text_color);
            osc_tooltip(ui, "/block1/ch2/sharpen_radius", Some(p.ch2_sharpen_radius));
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block1.ch2_sharpen_radius".to_string(), 0.0, 5.0));
            }
            
            let _text_color = if is_learning("block1.ch2_filters_boost") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Filters Boost##ch2adj").speed(0.01).range(0.0, 2.0).build(ui, &mut p.ch2_filters_boost);
            drop(_text_color);
            osc_tooltip(ui, "/block1/ch2/filters_boost", Some(p.ch2_filters_boost));
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block1.ch2_filters_boost".to_string(), 0.0, 2.0));
            }
        }
        
        // Process MIDI learn clicks (deferred to avoid borrow issues)
        for (param_id, min, max) in midi_learn_clicks {
            self.handle_midi_learn_click(&param_id, min, max);
        }
    }
    
    /// Build Block 1 FB1 Parameters panel
    fn build_block1_fb1_params(&mut self, ui: &Ui) {
        // Extract config values and MIDI learn state before borrowing self mutably
        let show_osc = self.config.show_osc_addresses;
        let learn_mode = self.midi_learn_mode;
        let learn_target = self.midi_learn_target.clone();
        
        // Collect clicks for deferred processing
        let mut midi_learn_clicks: Vec<(String, f32, f32)> = Vec::new();
        
        let p = &mut self.block1_edit;
        
        // Helper closure for OSC tooltips
        let osc_tooltip = |ui: &Ui, address: &str, value: Option<f32>| {
            if show_osc && ui.is_item_hovered() {
                let mut tooltip = format!("OSC: {}", address);
                if let Some(val) = value {
                    tooltip.push_str(&format!("\nValue: {:.3}", val));
                }
                ui.tooltip_text(tooltip);
            }
        };
        
        // Helper to check if a parameter is being learned
        let is_learning = |param_id: &str| -> bool {
            learn_mode && learn_target.as_deref() == Some(param_id)
        };
        
        // Mix section
        if CollapsingHeader::new("Feedback Mix").default_open(true).build(ui) {
            // Mix amount with fine control
            let _text_color = if is_learning("block1.fb1_mix_amount") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Mix Amount##fb1").speed(0.002).range(0.0, 1.0).build(ui, &mut p.fb1_mix_amount);
            drop(_text_color);
            osc_tooltip(ui, "/block1/fb1/mix_amount", Some(p.fb1_mix_amount));
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block1.fb1_mix_amount".to_string(), 0.0, 1.0));
            }
            
            let mut mix_type = p.fb1_mix_type as usize;
            let preview = MIX_TYPES[mix_type].to_string();
            ComboBox::new(ui, "Mix Type##fb1")
                .preview_value(&preview)
                .build(|| {
                    for (idx, opt) in MIX_TYPES.iter().enumerate() {
                        if ui.selectable_config(opt).selected(idx == mix_type).build() {
                            mix_type = idx;
                        }
                    }
                });
            p.fb1_mix_type = mix_type.clamp(0, MIX_TYPES.len() - 1) as i32;
            
            let mut overflow = p.fb1_mix_overflow as usize;
            let preview = GEO_OVERFLOW_MODES[overflow].to_string();
            ComboBox::new(ui, "Mix Overflow##fb1")
                .preview_value(&preview)
                .build(|| {
                    for (idx, opt) in GEO_OVERFLOW_MODES.iter().enumerate() {
                        if ui.selectable_config(opt).selected(idx == overflow).build() {
                            overflow = idx;
                        }
                    }
                });
            p.fb1_mix_overflow = overflow.clamp(0, GEO_OVERFLOW_MODES.len() - 1) as i32;
        }
        
        // FB1 Key section - OF-style (always active)
        if CollapsingHeader::new("Feedback Key").default_open(true).build(ui) {
            // Key mode selector (OF: 0=lumakey, 1=chromakey)
            let mut key_mode = (p.fb1_key_mode.clamp(0, 1)) as usize;
            let key_modes = ["Lumakey", "Chromakey"];
            let preview = key_modes[key_mode].to_string();
            ComboBox::new(ui, "Key Mode##fb1")
                .preview_value(&preview)
                .build(|| {
                    for (idx, opt) in key_modes.iter().enumerate() {
                        if ui.selectable_config(*opt).selected(idx == key_mode).build() {
                            key_mode = idx;
                        }
                    }
                });
            p.fb1_key_mode = key_mode as i32;
            
            // Key value (OF uses -1.0 to 1.0 range)
            let mut key_color = [p.fb1_key_value_red, p.fb1_key_value_green, p.fb1_key_value_blue];
            
            if p.fb1_key_mode == 0 {
                // Lumakey mode - single slider controls all channels
                ui.text("Key Value:");
                let _text_color = if is_learning("block1.fb1_key_value") {
                    Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
                } else { None };
                Drag::new("Key Value##fb1").speed(0.01).range(-1.0, 1.0).build(ui, &mut key_color[0]);
                drop(_text_color);
                if learn_mode && ui.is_item_clicked() { 
                    midi_learn_clicks.push(("block1.fb1_key_value".to_string(), -1.0, 1.0));
                }
                key_color[1] = key_color[0];
                key_color[2] = key_color[0];
            } else {
                // Chromakey mode - RGB sliders
                ui.text("Key Color (RGB -1 to 1):");
                let _text_color = if is_learning("block1.fb1_key_value_red") {
                    Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
                } else { None };
                Drag::new("Red##fb1key").speed(0.01).range(-1.0, 1.0).build(ui, &mut key_color[0]);
                drop(_text_color);
                if learn_mode && ui.is_item_clicked() { 
                    midi_learn_clicks.push(("block1.fb1_key_value_red".to_string(), -1.0, 1.0));
                }
                
                let _text_color = if is_learning("block1.fb1_key_value_green") {
                    Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
                } else { None };
                Drag::new("Green##fb1key").speed(0.01).range(-1.0, 1.0).build(ui, &mut key_color[1]);
                drop(_text_color);
                if learn_mode && ui.is_item_clicked() { 
                    midi_learn_clicks.push(("block1.fb1_key_value_green".to_string(), -1.0, 1.0));
                }
                
                let _text_color = if is_learning("block1.fb1_key_value_blue") {
                    Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
                } else { None };
                Drag::new("Blue##fb1key").speed(0.01).range(-1.0, 1.0).build(ui, &mut key_color[2]);
                drop(_text_color);
                if learn_mode && ui.is_item_clicked() { 
                    midi_learn_clicks.push(("block1.fb1_key_value_blue".to_string(), -1.0, 1.0));
                }
            }
            
            // Color preview (mapped to 0-1 for display)
            let preview_color = [
                (key_color[0] + 1.0) * 0.5,
                (key_color[1] + 1.0) * 0.5,
                (key_color[2] + 1.0) * 0.5,
                1.0, // Alpha
            ];
            ui.color_button("Key Color##fb1preview", preview_color);
            ui.same_line();
            ui.text("preview");
            
            p.fb1_key_value_red = key_color[0];
            p.fb1_key_value_green = key_color[1];
            p.fb1_key_value_blue = key_color[2];
            
            ui.separator();
            
            // Key Order dropdown
            let key_orders = ["Key First, Then Mix", "Mix First, Then Key"];
            let mut order_idx = p.fb1_key_order as usize;
            let preview = key_orders[order_idx.clamp(0, key_orders.len() - 1)].to_string();
            ComboBox::new(ui, "Key Order##fb1")
                .preview_value(&preview)
                .build(|| {
                    for (idx, opt) in key_orders.iter().enumerate() {
                        if ui.selectable_config(*opt).selected(idx == order_idx).build() {
                            order_idx = idx;
                        }
                    }
                });
            p.fb1_key_order = order_idx.clamp(0, key_orders.len() - 1) as i32;
            
            // Key threshold and soft (OF uses -1.0 to 1.0)
            ui.text("Key Parameters:");
            let _text_color = if is_learning("block1.fb1_key_threshold") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Threshold##fb1").speed(0.01).range(-1.0, 1.0).build(ui, &mut p.fb1_key_threshold);
            drop(_text_color);
            osc_tooltip(ui, "/block1/fb1/key_threshold", Some(p.fb1_key_threshold));
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block1.fb1_key_threshold".to_string(), -1.0, 1.0));
            }
            
            let _text_color = if is_learning("block1.fb1_key_soft") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Soft##fb1").speed(0.01).range(-1.0, 1.0).build(ui, &mut p.fb1_key_soft);
            drop(_text_color);
            osc_tooltip(ui, "/block1/fb1/key_soft", Some(p.fb1_key_soft));
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block1.fb1_key_soft".to_string(), -1.0, 1.0));
            }
        }
        
        // FB1 Geometry section
        if CollapsingHeader::new("Feedback Geometry").default_open(true).build(ui) {
            let _text_color = if is_learning("block1.fb1_x_displace") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("X Displace##fb1").speed(0.01).range(-2.0, 2.0).build(ui, &mut p.fb1_x_displace);
            drop(_text_color);
            osc_tooltip(ui, "/block1/fb1/x_displace", Some(p.fb1_x_displace));
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block1.fb1_x_displace".to_string(), -2.0, 2.0));
            }
            
            let _text_color = if is_learning("block1.fb1_y_displace") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Y Displace##fb1").speed(0.01).range(-2.0, 2.0).build(ui, &mut p.fb1_y_displace);
            drop(_text_color);
            osc_tooltip(ui, "/block1/fb1/y_displace", Some(p.fb1_y_displace));
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block1.fb1_y_displace".to_string(), -2.0, 2.0));
            }
            
            let _text_color = if is_learning("block1.fb1_z_displace") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Z Displace##fb1").speed(0.01).range(0.0, 10.0).build(ui, &mut p.fb1_z_displace);
            drop(_text_color);
            osc_tooltip(ui, "/block1/fb1/z_displace", Some(p.fb1_z_displace));
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block1.fb1_z_displace".to_string(), 0.0, 10.0));
            }
            
            let _text_color = if is_learning("block1.fb1_rotate") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Rotate##fb1").speed(0.1).range(-360.0, 360.0).build(ui, &mut p.fb1_rotate);
            drop(_text_color);
            osc_tooltip(ui, "/block1/fb1/rotate", Some(p.fb1_rotate));
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block1.fb1_rotate".to_string(), -360.0, 360.0));
            }
            
            // Shear matrix - individual sliders
            ui.text("Shear Matrix:");
            let mut shear_x = p.fb1_shear_matrix_x;
            let _text_color = if is_learning("block1.fb1_shear_matrix_x") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("X##fb1shear").speed(0.01).range(-2.0, 2.0).build(ui, &mut shear_x);
            drop(_text_color);
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block1.fb1_shear_matrix_x".to_string(), -2.0, 2.0));
            }
            p.fb1_shear_matrix_x = shear_x;
            
            let mut shear_y = p.fb1_shear_matrix_y;
            let _text_color = if is_learning("block1.fb1_shear_matrix_y") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Y##fb1shear").speed(0.01).range(-2.0, 2.0).build(ui, &mut shear_y);
            drop(_text_color);
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block1.fb1_shear_matrix_y".to_string(), -2.0, 2.0));
            }
            p.fb1_shear_matrix_y = shear_y;
            
            let mut shear_z = p.fb1_shear_matrix_z;
            let _text_color = if is_learning("block1.fb1_shear_matrix_z") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Z##fb1shear").speed(0.01).range(-2.0, 2.0).build(ui, &mut shear_z);
            drop(_text_color);
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block1.fb1_shear_matrix_z".to_string(), -2.0, 2.0));
            }
            p.fb1_shear_matrix_z = shear_z;
            
            let mut shear_w = p.fb1_shear_matrix_w;
            let _text_color = if is_learning("block1.fb1_shear_matrix_w") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("W##fb1shear").speed(0.01).range(-2.0, 2.0).build(ui, &mut shear_w);
            drop(_text_color);
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block1.fb1_shear_matrix_w".to_string(), -2.0, 2.0));
            }
            p.fb1_shear_matrix_w = shear_w;
            
            p.fb1_shear_matrix = Vec4::new(shear_x, shear_y, shear_z, shear_w);
            
            let _text_color = if is_learning("block1.fb1_kaleidoscope_amount") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Kaleidoscope Amount##fb1").speed(0.01).range(0.0, 1.0).build(ui, &mut p.fb1_kaleidoscope_amount);
            drop(_text_color);
            osc_tooltip(ui, "/block1/fb1/kaleidoscope_amount", Some(p.fb1_kaleidoscope_amount));
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block1.fb1_kaleidoscope_amount".to_string(), 0.0, 1.0));
            }
            
            let _text_color = if is_learning("block1.fb1_kaleidoscope_slice") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Kaleidoscope Slice##fb1").speed(0.01).range(0.0, 1.0).build(ui, &mut p.fb1_kaleidoscope_slice);
            drop(_text_color);
            osc_tooltip(ui, "/block1/fb1/kaleidoscope_slice", Some(p.fb1_kaleidoscope_slice));
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block1.fb1_kaleidoscope_slice".to_string(), 0.0, 1.0));
            }
        }
        
        // FB1 Color section
        if CollapsingHeader::new("Feedback Color").default_open(true).build(ui) {
            // HSB Offset - individual sliders
            ui.text("HSB Offset:");
            let mut hsb_offset_x = p.fb1_hsb_offset_x;
            let _text_color = if is_learning("block1.fb1_hsb_offset_x") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("H##fb1hsboff").speed(0.01).range(-1.0, 1.0).build(ui, &mut hsb_offset_x);
            drop(_text_color);
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block1.fb1_hsb_offset_x".to_string(), -1.0, 1.0));
            }
            p.fb1_hsb_offset_x = hsb_offset_x;
            
            let mut hsb_offset_y = p.fb1_hsb_offset_y;
            let _text_color = if is_learning("block1.fb1_hsb_offset_y") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("S##fb1hsboff").speed(0.01).range(-1.0, 1.0).build(ui, &mut hsb_offset_y);
            drop(_text_color);
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block1.fb1_hsb_offset_y".to_string(), -1.0, 1.0));
            }
            p.fb1_hsb_offset_y = hsb_offset_y;
            
            let mut hsb_offset_z = p.fb1_hsb_offset_z;
            let _text_color = if is_learning("block1.fb1_hsb_offset_z") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("B##fb1hsboff").speed(0.01).range(-1.0, 1.0).build(ui, &mut hsb_offset_z);
            drop(_text_color);
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block1.fb1_hsb_offset_z".to_string(), -1.0, 1.0));
            }
            p.fb1_hsb_offset_z = hsb_offset_z;
            
            p.fb1_hsb_offset = Vec3::new(hsb_offset_x, hsb_offset_y, hsb_offset_z);
            
            // HSB Attenuate - individual sliders
            ui.text("HSB Attenuate:");
            let mut hsb_att_x = p.fb1_hsb_attenuate_x;
            let _text_color = if is_learning("block1.fb1_hsb_attenuate_x") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("H##fb1hsbatt").speed(0.01).range(0.0, 2.0).build(ui, &mut hsb_att_x);
            drop(_text_color);
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block1.fb1_hsb_attenuate_x".to_string(), 0.0, 2.0));
            }
            p.fb1_hsb_attenuate_x = hsb_att_x;
            
            let mut hsb_att_y = p.fb1_hsb_attenuate_y;
            let _text_color = if is_learning("block1.fb1_hsb_attenuate_y") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("S##fb1hsbatt").speed(0.01).range(0.0, 2.0).build(ui, &mut hsb_att_y);
            drop(_text_color);
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block1.fb1_hsb_attenuate_y".to_string(), 0.0, 2.0));
            }
            p.fb1_hsb_attenuate_y = hsb_att_y;
            
            let mut hsb_att_z = p.fb1_hsb_attenuate_z;
            let _text_color = if is_learning("block1.fb1_hsb_attenuate_z") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("B##fb1hsbatt").speed(0.01).range(0.0, 2.0).build(ui, &mut hsb_att_z);
            drop(_text_color);
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block1.fb1_hsb_attenuate_z".to_string(), 0.0, 2.0));
            }
            p.fb1_hsb_attenuate_z = hsb_att_z;
            
            p.fb1_hsb_attenuate = Vec3::new(hsb_att_x, hsb_att_y, hsb_att_z);
            
            // HSB PowMap - individual sliders
            ui.text("HSB PowMap:");
            let mut hsb_pow_x = p.fb1_hsb_powmap.x;
            let _text_color = if is_learning("block1.fb1_hsb_powmap_x") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("H##fb1hsbpow").speed(0.01).range(0.0, 5.0).build(ui, &mut hsb_pow_x);
            drop(_text_color);
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block1.fb1_hsb_powmap_x".to_string(), 0.0, 5.0));
            }
            
            let mut hsb_pow_y = p.fb1_hsb_powmap.y;
            let _text_color = if is_learning("block1.fb1_hsb_powmap_y") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("S##fb1hsbpow").speed(0.01).range(0.0, 5.0).build(ui, &mut hsb_pow_y);
            drop(_text_color);
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block1.fb1_hsb_powmap_y".to_string(), 0.0, 5.0));
            }
            
            let mut hsb_pow_z = p.fb1_hsb_powmap.z;
            let _text_color = if is_learning("block1.fb1_hsb_powmap_z") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("B##fb1hsbpow").speed(0.01).range(0.0, 5.0).build(ui, &mut hsb_pow_z);
            drop(_text_color);
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block1.fb1_hsb_powmap_z".to_string(), 0.0, 5.0));
            }
            
            p.fb1_hsb_powmap = Vec3::new(hsb_pow_x, hsb_pow_y, hsb_pow_z);
            
            let _text_color = if is_learning("block1.fb1_hue_shaper") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Hue Shaper##fb1").speed(0.01).range(0.0, 2.0).build(ui, &mut p.fb1_hue_shaper);
            drop(_text_color);
            osc_tooltip(ui, "/block1/fb1/hue_shaper", Some(p.fb1_hue_shaper));
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block1.fb1_hue_shaper".to_string(), 0.0, 2.0));
            }
            
            ui.checkbox("Hue Invert##fb1", &mut p.fb1_hue_invert);
            ui.checkbox("Saturation Invert##fb1", &mut p.fb1_saturation_invert);
            ui.checkbox("Brightness Invert##fb1", &mut p.fb1_bright_invert);
        }
        
        // FB1 Filters section
        if CollapsingHeader::new("Feedback Filters").default_open(true).build(ui) {
            let _text_color = if is_learning("block1.fb1_blur_amount") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Blur Amount##fb1").speed(0.01).range(0.0, 1.0).build(ui, &mut p.fb1_blur_amount);
            drop(_text_color);
            osc_tooltip(ui, "/block1/fb1/blur_amount", Some(p.fb1_blur_amount));
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block1.fb1_blur_amount".to_string(), 0.0, 1.0));
            }
            
            let _text_color = if is_learning("block1.fb1_blur_radius") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Blur Radius##fb1").speed(0.01).range(0.0, 5.0).build(ui, &mut p.fb1_blur_radius);
            drop(_text_color);
            osc_tooltip(ui, "/block1/fb1/blur_radius", Some(p.fb1_blur_radius));
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block1.fb1_blur_radius".to_string(), 0.0, 5.0));
            }
            
            let _text_color = if is_learning("block1.fb1_sharpen_amount") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Sharpen Amount##fb1").speed(0.01).range(0.0, 1.0).build(ui, &mut p.fb1_sharpen_amount);
            drop(_text_color);
            osc_tooltip(ui, "/block1/fb1/sharpen_amount", Some(p.fb1_sharpen_amount));
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block1.fb1_sharpen_amount".to_string(), 0.0, 1.0));
            }
            
            let _text_color = if is_learning("block1.fb1_sharpen_radius") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Sharpen Radius##fb1").speed(0.01).range(0.0, 5.0).build(ui, &mut p.fb1_sharpen_radius);
            drop(_text_color);
            osc_tooltip(ui, "/block1/fb1/sharpen_radius", Some(p.fb1_sharpen_radius));
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block1.fb1_sharpen_radius".to_string(), 0.0, 5.0));
            }
            
            let _text_color = if is_learning("block1.fb1_temporal_filter1_amount") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Temp Filter 1 Amount##fb1").speed(0.01).range(0.0, 1.0).build(ui, &mut p.fb1_temporal_filter1_amount);
            drop(_text_color);
            osc_tooltip(ui, "/block1/fb1/temp_filter1_amount", Some(p.fb1_temporal_filter1_amount));
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block1.fb1_temporal_filter1_amount".to_string(), 0.0, 1.0));
            }
            
            let _text_color = if is_learning("block1.fb1_temporal_filter1_resonance") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Temp Filter 1 Res##fb1").speed(0.01).range(0.0, 1.0).build(ui, &mut p.fb1_temporal_filter1_resonance);
            drop(_text_color);
            osc_tooltip(ui, "/block1/fb1/temp_filter1_res", Some(p.fb1_temporal_filter1_resonance));
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block1.fb1_temporal_filter1_resonance".to_string(), 0.0, 1.0));
            }
            
            let _text_color = if is_learning("block1.fb1_temporal_filter2_amount") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Temp Filter 2 Amount##fb1").speed(0.01).range(0.0, 1.0).build(ui, &mut p.fb1_temporal_filter2_amount);
            drop(_text_color);
            osc_tooltip(ui, "/block1/fb1/temp_filter2_amount", Some(p.fb1_temporal_filter2_amount));
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block1.fb1_temporal_filter2_amount".to_string(), 0.0, 1.0));
            }
            
            let _text_color = if is_learning("block1.fb1_temporal_filter2_resonance") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Temp Filter 2 Res##fb1").speed(0.01).range(0.0, 1.0).build(ui, &mut p.fb1_temporal_filter2_resonance);
            drop(_text_color);
            osc_tooltip(ui, "/block1/fb1/temp_filter2_res", Some(p.fb1_temporal_filter2_resonance));
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block1.fb1_temporal_filter2_resonance".to_string(), 0.0, 1.0));
            }
            
            let _text_color = if is_learning("block1.fb1_filters_boost") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Filters Boost##fb1").speed(0.01).range(0.0, 2.0).build(ui, &mut p.fb1_filters_boost);
            drop(_text_color);
            osc_tooltip(ui, "/block1/fb1/filters_boost", Some(p.fb1_filters_boost));
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block1.fb1_filters_boost".to_string(), 0.0, 2.0));
            }
            
            ui.separator();
            // Delay section with tempo sync
            ui.text("Feedback Delay");
            
            // Tempo sync toggle
            let mut sync_enabled = p.fb1_delay_time_sync;
            if ui.checkbox("Sync to BPM##fb1delay", &mut sync_enabled) {
                p.fb1_delay_time_sync = sync_enabled;
            }
            if ui.is_item_hovered() {
                ui.tooltip_text("Enable to sync delay time to BPM");
            }
            
            if p.fb1_delay_time_sync {
                // Show beat division dropdown when sync is enabled
                let beat_divisions = ["1/16", "1/8", "1/4", "1/2", "1", "2", "4", "8"];
                let mut div_idx = p.fb1_delay_time_division as usize;
                let preview = beat_divisions[div_idx.min(beat_divisions.len() - 1)].to_string();
                ComboBox::new(ui, "##fb1_delay_division")
                    .preview_value(&preview)
                    .build(|| {
                        for (idx, opt) in beat_divisions.iter().enumerate() {
                            if ui.selectable_config(opt).selected(idx == div_idx).build() {
                                div_idx = idx;
                            }
                        }
                    });
                p.fb1_delay_time_division = div_idx.clamp(0, beat_divisions.len() - 1) as i32;
                
                // Show calculated delay time
                let calculated_frames = crate::core::lfo_engine::calculate_delay_frames_from_tempo(
                    self.bpm, p.fb1_delay_time_division, 60.0
                );
                ui.text_disabled(format!("Calculated: {} frames (≈ {:.2}s at {} BPM)", 
                    calculated_frames, calculated_frames as f32 / 60.0, self.bpm as i32));
            } else {
                // Show frame slider when sync is disabled
                let _text_color = if is_learning("block1.fb1_delay_time") {
                    Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
                } else { None };
                Drag::new("Delay Time (frames)##fb1").speed(1.0).range(0, 120).build(ui, &mut p.fb1_delay_time);
                drop(_text_color);
                osc_tooltip(ui, "/block1/fb1/delay_time", Some(p.fb1_delay_time as f32));
                if learn_mode && ui.is_item_clicked() { 
                    midi_learn_clicks.push(("block1.fb1_delay_time".to_string(), 0.0, 120.0));
                }
                if p.fb1_delay_time > 0 {
                    ui.text_disabled(format!("≈ {:.2} seconds at 60fps", p.fb1_delay_time as f32 / 60.0));
                } else {
                    ui.text_disabled("No delay (use immediate feedback)");
                }
            }
        }
        
        // Process MIDI learn clicks (deferred to avoid borrow issues)
        for (param_id, min, max) in midi_learn_clicks {
            self.handle_midi_learn_click(&param_id, min, max);
        }
    }

    /// Build Block 2 panel with sub-tabs
    fn build_block2_panel(&mut self, ui: &Ui) {
        let subtab_labels = ["Input Adjust", "FB2", "LFO"];
        let mut subtab_idx = self.block2_tab as usize;
        
        if let Some(_tab_bar) = ui.tab_bar("##block2_tabs") {
            for (idx, label) in subtab_labels.iter().enumerate() {
                let is_selected = idx == subtab_idx;
                
                if let Some(_tab) = ui.tab_item(label) {
                    if !is_selected {
                        subtab_idx = idx;
                        self.block2_tab = match idx {
                            0 => Block2Tab::InputAdjust,
                            1 => Block2Tab::Fb2Parameters,
                            2 => Block2Tab::Lfo,
                            _ => Block2Tab::InputAdjust,
                        };
                    }
                }
            }
        }
        
        match self.block2_tab {
            Block2Tab::InputAdjust => self.build_block2_input_adjust(ui),
            Block2Tab::Fb2Parameters => self.build_block2_fb2_params(ui),
            Block2Tab::Lfo => self.build_block2_lfo(ui),
        }
    }
    
    /// Build Block 2 Input Adjust panel
    fn build_block2_input_adjust(&mut self, ui: &Ui) {
        // Extract config values and MIDI learn state before borrowing self mutably
        let show_osc = self.config.show_osc_addresses;
        let learn_mode = self.midi_learn_mode;
        let learn_target = self.midi_learn_target.clone();
        
        // Collect clicks for deferred processing
        let mut midi_learn_clicks: Vec<(String, f32, f32)> = Vec::new();
        
        let p = &mut self.block2_edit;
        
        // Helper closure for OSC tooltips
        let osc_tooltip = |ui: &Ui, address: &str, value: Option<f32>| {
            if show_osc && ui.is_item_hovered() {
                let mut tooltip = format!("OSC: {}", address);
                if let Some(val) = value {
                    tooltip.push_str(&format!("\nValue: {:.3}", val));
                }
                ui.tooltip_text(tooltip);
            }
        };
        
        // Helper to check if a parameter is being learned
        let is_learning = |param_id: &str| -> bool {
            learn_mode && learn_target.as_deref() == Some(param_id)
        };
        
        // Input selection
        ui.text("Input Source:");
        ui.same_line();
        let input_options = ["Block 1", "Input 1", "Input 2"];
        let mut input_idx = p.block2_input_select as usize;
        let preview = input_options[input_idx].to_string();
        ComboBox::new(ui, "##b2_input_select")
            .preview_value(&preview)
            .build(|| {
                for (idx, opt) in input_options.iter().enumerate() {
                    if ui.selectable_config(opt).selected(idx == input_idx).build() {
                        input_idx = idx;
                    }
                }
            });
        p.block2_input_select = input_idx.clamp(0, 2) as i32;
        
        ui.separator();
        
        // Geometry section
        if CollapsingHeader::new("Geometry").default_open(true).build(ui) {
            let _text_color = if is_learning("block2.block2_input_x_displace") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("X Displace##b2in").speed(0.01).range(-2.0, 2.0).build(ui, &mut p.block2_input_x_displace);
            drop(_text_color);
            osc_tooltip(ui, "/block2/input/x_displace", Some(p.block2_input_x_displace));
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block2.block2_input_x_displace".to_string(), -2.0, 2.0));
            }
            
            let _text_color = if is_learning("block2.block2_input_y_displace") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Y Displace##b2in").speed(0.01).range(-2.0, 2.0).build(ui, &mut p.block2_input_y_displace);
            drop(_text_color);
            osc_tooltip(ui, "/block2/input/y_displace", Some(p.block2_input_y_displace));
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block2.block2_input_y_displace".to_string(), -2.0, 2.0));
            }
            
            let _text_color = if is_learning("block2.block2_input_z_displace") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Z Displace##b2in").speed(0.01).range(0.0, 10.0).build(ui, &mut p.block2_input_z_displace);
            drop(_text_color);
            osc_tooltip(ui, "/block2/input/z_displace", Some(p.block2_input_z_displace));
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block2.block2_input_z_displace".to_string(), 0.0, 10.0));
            }
            
            let _text_color = if is_learning("block2.block2_input_rotate") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Rotate##b2in").speed(0.1).range(-360.0, 360.0).build(ui, &mut p.block2_input_rotate);
            drop(_text_color);
            osc_tooltip(ui, "/block2/input/rotate", Some(p.block2_input_rotate));
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block2.block2_input_rotate".to_string(), -360.0, 360.0));
            }
            
            let _text_color = if is_learning("block2.block2_input_kaleidoscope_amount") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Kaleidoscope Amount##b2in").speed(0.01).range(0.0, 1.0).build(ui, &mut p.block2_input_kaleidoscope_amount);
            drop(_text_color);
            osc_tooltip(ui, "/block2/input/kaleidoscope_amount", Some(p.block2_input_kaleidoscope_amount));
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block2.block2_input_kaleidoscope_amount".to_string(), 0.0, 1.0));
            }
            
            let _text_color = if is_learning("block2.block2_input_kaleidoscope_slice") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Kaleidoscope Slice##b2in").speed(0.01).range(0.0, 1.0).build(ui, &mut p.block2_input_kaleidoscope_slice);
            drop(_text_color);
            osc_tooltip(ui, "/block2/input/kaleidoscope_slice", Some(p.block2_input_kaleidoscope_slice));
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block2.block2_input_kaleidoscope_slice".to_string(), 0.0, 1.0));
            }
            
            let mut overflow_idx = p.block2_input_geo_overflow as usize;
            let preview = GEO_OVERFLOW_MODES[overflow_idx].to_string();
            ComboBox::new(ui, "Overflow Mode##b2in")
                .preview_value(&preview)
                .build(|| {
                    for (idx, opt) in GEO_OVERFLOW_MODES.iter().enumerate() {
                        if ui.selectable_config(opt).selected(idx == overflow_idx).build() {
                            overflow_idx = idx;
                        }
                    }
                });
            p.block2_input_geo_overflow = overflow_idx.clamp(0, GEO_OVERFLOW_MODES.len() - 1) as i32;
        }
        
        // Color section
        if CollapsingHeader::new("Color").default_open(true).build(ui) {
            // HSB Attenuate - individual sliders
            ui.text("HSB Attenuate:");
            let mut hsb_x = p.block2_input_hsb_attenuate_x;
            let _text_color = if is_learning("block2.block2_input_hsb_attenuate_x") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("H##b2inhsb").speed(0.01).range(0.0, 2.0).build(ui, &mut hsb_x);
            drop(_text_color);
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block2.block2_input_hsb_attenuate_x".to_string(), 0.0, 2.0));
            }
            p.block2_input_hsb_attenuate_x = hsb_x;
            
            let mut hsb_y = p.block2_input_hsb_attenuate_y;
            let _text_color = if is_learning("block2.block2_input_hsb_attenuate_y") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("S##b2inhsb").speed(0.01).range(0.0, 2.0).build(ui, &mut hsb_y);
            drop(_text_color);
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block2.block2_input_hsb_attenuate_y".to_string(), 0.0, 2.0));
            }
            p.block2_input_hsb_attenuate_y = hsb_y;
            
            let mut hsb_z = p.block2_input_hsb_attenuate_z;
            let _text_color = if is_learning("block2.block2_input_hsb_attenuate_z") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("B##b2inhsb").speed(0.01).range(0.0, 2.0).build(ui, &mut hsb_z);
            drop(_text_color);
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block2.block2_input_hsb_attenuate_z".to_string(), 0.0, 2.0));
            }
            p.block2_input_hsb_attenuate_z = hsb_z;
            
            p.block2_input_hsb_attenuate = Vec3::new(hsb_x, hsb_y, hsb_z);
            
            ui.checkbox("Hue Invert##b2in", &mut p.block2_input_hue_invert);
            ui.checkbox("Saturation Invert##b2in", &mut p.block2_input_saturation_invert);
            ui.checkbox("Brightness Invert##b2in", &mut p.block2_input_bright_invert);
            ui.checkbox("RGB Invert##b2in", &mut p.block2_input_rgb_invert);
            
            ui.checkbox("Solarize##b2in", &mut p.block2_input_solarize);
            ui.checkbox("Posterize##b2in", &mut p.block2_input_posterize_switch);
            if p.block2_input_posterize_switch {
                let _text_color = if is_learning("block2.block2_input_posterize") {
                    Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
                } else { None };
                Drag::new("Posterize Levels##b2in").speed(0.1).range(2.0, 32.0).build(ui, &mut p.block2_input_posterize);
                drop(_text_color);
                osc_tooltip(ui, "/block2/input/posterize", Some(p.block2_input_posterize));
                if learn_mode && ui.is_item_clicked() { 
                    midi_learn_clicks.push(("block2.block2_input_posterize".to_string(), 2.0, 32.0));
                }
            }
        }
        
        // Filters section
        if CollapsingHeader::new("Filters").default_open(true).build(ui) {
            let _text_color = if is_learning("block2.block2_input_blur_amount") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Blur Amount##b2in").speed(0.01).range(0.0, 1.0).build(ui, &mut p.block2_input_blur_amount);
            drop(_text_color);
            osc_tooltip(ui, "/block2/input/blur_amount", Some(p.block2_input_blur_amount));
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block2.block2_input_blur_amount".to_string(), 0.0, 1.0));
            }
            
            let _text_color = if is_learning("block2.block2_input_blur_radius") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Blur Radius##b2in").speed(0.01).range(0.0, 5.0).build(ui, &mut p.block2_input_blur_radius);
            drop(_text_color);
            osc_tooltip(ui, "/block2/input/blur_radius", Some(p.block2_input_blur_radius));
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block2.block2_input_blur_radius".to_string(), 0.0, 5.0));
            }
            
            let _text_color = if is_learning("block2.block2_input_sharpen_amount") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Sharpen Amount##b2in").speed(0.01).range(0.0, 1.0).build(ui, &mut p.block2_input_sharpen_amount);
            drop(_text_color);
            osc_tooltip(ui, "/block2/input/sharpen_amount", Some(p.block2_input_sharpen_amount));
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block2.block2_input_sharpen_amount".to_string(), 0.0, 1.0));
            }
            
            let _text_color = if is_learning("block2.block2_input_sharpen_radius") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Sharpen Radius##b2in").speed(0.01).range(0.0, 5.0).build(ui, &mut p.block2_input_sharpen_radius);
            drop(_text_color);
            osc_tooltip(ui, "/block2/input/sharpen_radius", Some(p.block2_input_sharpen_radius));
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block2.block2_input_sharpen_radius".to_string(), 0.0, 5.0));
            }
            
            let _text_color = if is_learning("block2.block2_input_filters_boost") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Filters Boost##b2in").speed(0.01).range(0.0, 2.0).build(ui, &mut p.block2_input_filters_boost);
            drop(_text_color);
            osc_tooltip(ui, "/block2/input/filters_boost", Some(p.block2_input_filters_boost));
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block2.block2_input_filters_boost".to_string(), 0.0, 2.0));
            }
        }
        
        // Process MIDI learn clicks (deferred to avoid borrow issues)
        for (param_id, min, max) in midi_learn_clicks {
            self.handle_midi_learn_click(&param_id, min, max);
        }
    }

    fn build_block2_fb2_params(&mut self, ui: &Ui) {
        // Extract config values and MIDI learn state before borrowing self mutably
        let show_osc = self.config.show_osc_addresses;
        let learn_mode = self.midi_learn_mode;
        let learn_target = self.midi_learn_target.clone();
        
        // Collect clicks for deferred processing
        let mut midi_learn_clicks: Vec<(String, f32, f32)> = Vec::new();
        
        let p = &mut self.block2_edit;
        
        // Helper closure for OSC tooltips
        let osc_tooltip = |ui: &Ui, address: &str, value: Option<f32>| {
            if show_osc && ui.is_item_hovered() {
                let mut tooltip = format!("OSC: {}", address);
                if let Some(val) = value {
                    tooltip.push_str(&format!("\nValue: {:.3}", val));
                }
                ui.tooltip_text(tooltip);
            }
        };
        
        // Helper to check if a parameter is being learned
        let is_learning = |param_id: &str| -> bool {
            learn_mode && learn_target.as_deref() == Some(param_id)
        };
        
        // Mix section
        if CollapsingHeader::new("Feedback Mix").default_open(true).build(ui) {
            // Mix amount with fine control
            let _text_color = if is_learning("block2.fb2_mix_amount") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Mix Amount##fb2").speed(0.002).range(0.0, 1.0).build(ui, &mut p.fb2_mix_amount);
            drop(_text_color);
            osc_tooltip(ui, "/block2/fb2/mix_amount", Some(p.fb2_mix_amount));
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block2.fb2_mix_amount".to_string(), 0.0, 1.0));
            }
            
            let mut mix_type = p.fb2_mix_type as usize;
            let preview = MIX_TYPES[mix_type].to_string();
            ComboBox::new(ui, "Mix Type##fb2")
                .preview_value(&preview)
                .build(|| {
                    for (idx, opt) in MIX_TYPES.iter().enumerate() {
                        if ui.selectable_config(opt).selected(idx == mix_type).build() {
                            mix_type = idx;
                        }
                    }
                });
            p.fb2_mix_type = mix_type.clamp(0, MIX_TYPES.len() - 1) as i32;
            
            let mut overflow = p.fb2_mix_overflow as usize;
            let preview = GEO_OVERFLOW_MODES[overflow].to_string();
            ComboBox::new(ui, "Mix Overflow##fb2")
                .preview_value(&preview)
                .build(|| {
                    for (idx, opt) in GEO_OVERFLOW_MODES.iter().enumerate() {
                        if ui.selectable_config(opt).selected(idx == overflow).build() {
                            overflow = idx;
                        }
                    }
                });
            p.fb2_mix_overflow = overflow.clamp(0, GEO_OVERFLOW_MODES.len() - 1) as i32;
        }
        
        // Key section
        if CollapsingHeader::new("Feedback Key").default_open(true).build(ui) {
            // Key mode selector
            let mut key_mode = (p.fb2_key_mode.clamp(0, 1)) as usize;
            let key_modes = ["Lumakey", "Chromakey"];
            let preview = key_modes[key_mode].to_string();
            ComboBox::new(ui, "Key Mode##fb2")
                .preview_value(&preview)
                .build(|| {
                    for (idx, opt) in key_modes.iter().enumerate() {
                        if ui.selectable_config(*opt).selected(idx == key_mode).build() {
                            key_mode = idx;
                        }
                    }
                });
            p.fb2_key_mode = key_mode as i32;
            
            let mut key_color = [p.fb2_key_value.x, p.fb2_key_value.y, p.fb2_key_value.z];
            
            if p.fb2_key_mode == 0 {
                // Lumakey mode - single slider controls all channels
                ui.text("Key Value:");
                let _text_color = if is_learning("block2.fb2_key_value") {
                    Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
                } else { None };
                Drag::new("Key Value##fb2").speed(0.01).range(-1.0, 1.0).build(ui, &mut key_color[0]);
                drop(_text_color);
                if learn_mode && ui.is_item_clicked() { 
                    midi_learn_clicks.push(("block2.fb2_key_value".to_string(), -1.0, 1.0));
                }
                key_color[1] = key_color[0];
                key_color[2] = key_color[0];
            } else {
                // Chromakey mode - RGB sliders
                ui.color_edit3("Key Color##fb2", &mut key_color);
            }
            
            // Color preview (mapped to 0-1 for display)
            let preview_color = [
                (key_color[0] + 1.0) * 0.5,
                (key_color[1] + 1.0) * 0.5,
                (key_color[2] + 1.0) * 0.5,
                1.0, // Alpha
            ];
            ui.color_button("Key Color##fb2preview", preview_color);
            ui.same_line();
            ui.text("preview");
            
            p.fb2_key_value = Vec3::new(key_color[0], key_color[1], key_color[2]);
            
            ui.separator();
            
            // Key Order dropdown
            let key_orders = ["Key First, Then Mix", "Mix First, Then Key"];
            let mut order_idx = p.fb2_key_order as usize;
            let preview = key_orders[order_idx.clamp(0, key_orders.len() - 1)].to_string();
            ComboBox::new(ui, "Key Order##fb2")
                .preview_value(&preview)
                .build(|| {
                    for (idx, opt) in key_orders.iter().enumerate() {
                        if ui.selectable_config(*opt).selected(idx == order_idx).build() {
                            order_idx = idx;
                        }
                    }
                });
            p.fb2_key_order = order_idx.clamp(0, key_orders.len() - 1) as i32;
            
            // Key threshold with fine control
            let _text_color = if is_learning("block2.fb2_key_threshold") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Key Threshold##fb2").speed(0.001).range(0.0, 1.0).build(ui, &mut p.fb2_key_threshold);
            drop(_text_color);
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block2.fb2_key_threshold".to_string(), 0.0, 1.0));
            }
            
            let _text_color = if is_learning("block2.fb2_key_soft") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Key Soft##fb2").speed(0.002).range(0.0, 1.0).build(ui, &mut p.fb2_key_soft);
            drop(_text_color);
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block2.fb2_key_soft".to_string(), 0.0, 1.0));
            }
        }
        
        // Geometry section
        if CollapsingHeader::new("Feedback Geometry").default_open(true).build(ui) {
            let _text_color = if is_learning("block2.fb2_x_displace") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("X Displace##fb2").speed(0.01).range(-2.0, 2.0).build(ui, &mut p.fb2_x_displace);
            drop(_text_color);
            osc_tooltip(ui, "/block2/fb2/x_displace", Some(p.fb2_x_displace));
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block2.fb2_x_displace".to_string(), -2.0, 2.0));
            }
            
            let _text_color = if is_learning("block2.fb2_y_displace") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Y Displace##fb2").speed(0.01).range(-2.0, 2.0).build(ui, &mut p.fb2_y_displace);
            drop(_text_color);
            osc_tooltip(ui, "/block2/fb2/y_displace", Some(p.fb2_y_displace));
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block2.fb2_y_displace".to_string(), -2.0, 2.0));
            }
            
            let _text_color = if is_learning("block2.fb2_z_displace") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Z Displace##fb2").speed(0.01).range(0.0, 10.0).build(ui, &mut p.fb2_z_displace);
            drop(_text_color);
            osc_tooltip(ui, "/block2/fb2/z_displace", Some(p.fb2_z_displace));
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block2.fb2_z_displace".to_string(), 0.0, 10.0));
            }
            
            let _text_color = if is_learning("block2.fb2_rotate") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Rotate##fb2").speed(0.1).range(-360.0, 360.0).build(ui, &mut p.fb2_rotate);
            drop(_text_color);
            osc_tooltip(ui, "/block2/fb2/rotate", Some(p.fb2_rotate));
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block2.fb2_rotate".to_string(), -360.0, 360.0));
            }
            
            // Shear matrix - individual sliders
            ui.text("Shear Matrix:");
            let mut shear_x = p.fb2_shear_matrix_x;
            let _text_color = if is_learning("block2.fb2_shear_matrix_x") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("X##fb2shear").speed(0.01).range(-2.0, 2.0).build(ui, &mut shear_x);
            drop(_text_color);
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block2.fb2_shear_matrix_x".to_string(), -2.0, 2.0));
            }
            p.fb2_shear_matrix_x = shear_x;
            
            let mut shear_y = p.fb2_shear_matrix_y;
            let _text_color = if is_learning("block2.fb2_shear_matrix_y") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Y##fb2shear").speed(0.01).range(-2.0, 2.0).build(ui, &mut shear_y);
            drop(_text_color);
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block2.fb2_shear_matrix_y".to_string(), -2.0, 2.0));
            }
            p.fb2_shear_matrix_y = shear_y;
            
            let mut shear_z = p.fb2_shear_matrix_z;
            let _text_color = if is_learning("block2.fb2_shear_matrix_z") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Z##fb2shear").speed(0.01).range(-2.0, 2.0).build(ui, &mut shear_z);
            drop(_text_color);
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block2.fb2_shear_matrix_z".to_string(), -2.0, 2.0));
            }
            p.fb2_shear_matrix_z = shear_z;
            
            let mut shear_w = p.fb2_shear_matrix_w;
            let _text_color = if is_learning("block2.fb2_shear_matrix_w") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("W##fb2shear").speed(0.01).range(-2.0, 2.0).build(ui, &mut shear_w);
            drop(_text_color);
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block2.fb2_shear_matrix_w".to_string(), -2.0, 2.0));
            }
            p.fb2_shear_matrix_w = shear_w;
            
            p.fb2_shear_matrix = Vec4::new(shear_x, shear_y, shear_z, shear_w);
            
            let _text_color = if is_learning("block2.fb2_kaleidoscope_amount") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Kaleidoscope Amount##fb2").speed(0.01).range(0.0, 1.0).build(ui, &mut p.fb2_kaleidoscope_amount);
            drop(_text_color);
            osc_tooltip(ui, "/block2/fb2/kaleidoscope_amount", Some(p.fb2_kaleidoscope_amount));
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block2.fb2_kaleidoscope_amount".to_string(), 0.0, 1.0));
            }
            
            let _text_color = if is_learning("block2.fb2_kaleidoscope_slice") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Kaleidoscope Slice##fb2").speed(0.01).range(0.0, 1.0).build(ui, &mut p.fb2_kaleidoscope_slice);
            drop(_text_color);
            osc_tooltip(ui, "/block2/fb2/kaleidoscope_slice", Some(p.fb2_kaleidoscope_slice));
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block2.fb2_kaleidoscope_slice".to_string(), 0.0, 1.0));
            }
            
            ui.checkbox("H Mirror##fb2", &mut p.fb2_h_mirror);
            ui.checkbox("V Mirror##fb2", &mut p.fb2_v_mirror);
            ui.checkbox("H Flip##fb2", &mut p.fb2_h_flip);
            ui.checkbox("V Flip##fb2", &mut p.fb2_v_flip);
        }
        
        // Color section
        if CollapsingHeader::new("Feedback Color").default_open(true).build(ui) {
            // HSB Offset - individual sliders
            ui.text("HSB Offset:");
            let mut hsb_offset_x = p.fb2_hsb_offset_x;
            let _text_color = if is_learning("block2.fb2_hsb_offset_x") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("H##fb2hsboff").speed(0.01).range(-1.0, 1.0).build(ui, &mut hsb_offset_x);
            drop(_text_color);
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block2.fb2_hsb_offset_x".to_string(), -1.0, 1.0));
            }
            p.fb2_hsb_offset_x = hsb_offset_x;
            
            let mut hsb_offset_y = p.fb2_hsb_offset_y;
            let _text_color = if is_learning("block2.fb2_hsb_offset_y") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("S##fb2hsboff").speed(0.01).range(-1.0, 1.0).build(ui, &mut hsb_offset_y);
            drop(_text_color);
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block2.fb2_hsb_offset_y".to_string(), -1.0, 1.0));
            }
            p.fb2_hsb_offset_y = hsb_offset_y;
            
            let mut hsb_offset_z = p.fb2_hsb_offset_z;
            let _text_color = if is_learning("block2.fb2_hsb_offset_z") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("B##fb2hsboff").speed(0.01).range(-1.0, 1.0).build(ui, &mut hsb_offset_z);
            drop(_text_color);
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block2.fb2_hsb_offset_z".to_string(), -1.0, 1.0));
            }
            p.fb2_hsb_offset_z = hsb_offset_z;
            
            p.fb2_hsb_offset = Vec3::new(hsb_offset_x, hsb_offset_y, hsb_offset_z);
            
            // HSB Attenuate - individual sliders
            ui.text("HSB Attenuate:");
            let mut hsb_att_x = p.fb2_hsb_attenuate_x;
            let _text_color = if is_learning("block2.fb2_hsb_attenuate_x") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("H##fb2hsbatt").speed(0.01).range(0.0, 2.0).build(ui, &mut hsb_att_x);
            drop(_text_color);
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block2.fb2_hsb_attenuate_x".to_string(), 0.0, 2.0));
            }
            p.fb2_hsb_attenuate_x = hsb_att_x;
            
            let mut hsb_att_y = p.fb2_hsb_attenuate_y;
            let _text_color = if is_learning("block2.fb2_hsb_attenuate_y") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("S##fb2hsbatt").speed(0.01).range(0.0, 2.0).build(ui, &mut hsb_att_y);
            drop(_text_color);
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block2.fb2_hsb_attenuate_y".to_string(), 0.0, 2.0));
            }
            p.fb2_hsb_attenuate_y = hsb_att_y;
            
            let mut hsb_att_z = p.fb2_hsb_attenuate_z;
            let _text_color = if is_learning("block2.fb2_hsb_attenuate_z") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("B##fb2hsbatt").speed(0.01).range(0.0, 2.0).build(ui, &mut hsb_att_z);
            drop(_text_color);
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block2.fb2_hsb_attenuate_z".to_string(), 0.0, 2.0));
            }
            p.fb2_hsb_attenuate_z = hsb_att_z;
            
            p.fb2_hsb_attenuate = Vec3::new(hsb_att_x, hsb_att_y, hsb_att_z);
            
            // HSB PowMap - individual sliders
            ui.text("HSB PowMap:");
            let mut hsb_pow_x = p.fb2_hsb_powmap.x;
            let _text_color = if is_learning("block2.fb2_hsb_powmap_x") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("H##fb2hsbpow").speed(0.01).range(0.0, 5.0).build(ui, &mut hsb_pow_x);
            drop(_text_color);
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block2.fb2_hsb_powmap_x".to_string(), 0.0, 5.0));
            }
            
            let mut hsb_pow_y = p.fb2_hsb_powmap.y;
            let _text_color = if is_learning("block2.fb2_hsb_powmap_y") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("S##fb2hsbpow").speed(0.01).range(0.0, 5.0).build(ui, &mut hsb_pow_y);
            drop(_text_color);
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block2.fb2_hsb_powmap_y".to_string(), 0.0, 5.0));
            }
            
            let mut hsb_pow_z = p.fb2_hsb_powmap.z;
            let _text_color = if is_learning("block2.fb2_hsb_powmap_z") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("B##fb2hsbpow").speed(0.01).range(0.0, 5.0).build(ui, &mut hsb_pow_z);
            drop(_text_color);
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block2.fb2_hsb_powmap_z".to_string(), 0.0, 5.0));
            }
            
            p.fb2_hsb_powmap = Vec3::new(hsb_pow_x, hsb_pow_y, hsb_pow_z);
            
            let _text_color = if is_learning("block2.fb2_hue_shaper") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Hue Shaper##fb2").speed(0.01).range(0.0, 2.0).build(ui, &mut p.fb2_hue_shaper);
            drop(_text_color);
            osc_tooltip(ui, "/block2/fb2/hue_shaper", Some(p.fb2_hue_shaper));
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block2.fb2_hue_shaper".to_string(), 0.0, 2.0));
            }
            
            ui.checkbox("Hue Invert##fb2", &mut p.fb2_hue_invert);
            ui.checkbox("Saturation Invert##fb2", &mut p.fb2_saturation_invert);
            ui.checkbox("Brightness Invert##fb2", &mut p.fb2_bright_invert);
            ui.checkbox("RGB Invert##fb2", &mut p.fb2_rgb_invert);
        }
        
        // Filters section
        if CollapsingHeader::new("Feedback Filters").default_open(true).build(ui) {
            let _text_color = if is_learning("block2.fb2_blur_amount") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Blur Amount##fb2").speed(0.01).range(0.0, 1.0).build(ui, &mut p.fb2_blur_amount);
            drop(_text_color);
            osc_tooltip(ui, "/block2/fb2/blur_amount", Some(p.fb2_blur_amount));
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block2.fb2_blur_amount".to_string(), 0.0, 1.0));
            }
            
            let _text_color = if is_learning("block2.fb2_blur_radius") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Blur Radius##fb2").speed(0.01).range(0.0, 5.0).build(ui, &mut p.fb2_blur_radius);
            drop(_text_color);
            osc_tooltip(ui, "/block2/fb2/blur_radius", Some(p.fb2_blur_radius));
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block2.fb2_blur_radius".to_string(), 0.0, 5.0));
            }
            
            let _text_color = if is_learning("block2.fb2_sharpen_amount") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Sharpen Amount##fb2").speed(0.01).range(0.0, 1.0).build(ui, &mut p.fb2_sharpen_amount);
            drop(_text_color);
            osc_tooltip(ui, "/block2/fb2/sharpen_amount", Some(p.fb2_sharpen_amount));
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block2.fb2_sharpen_amount".to_string(), 0.0, 1.0));
            }
            
            let _text_color = if is_learning("block2.fb2_sharpen_radius") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Sharpen Radius##fb2").speed(0.01).range(0.0, 5.0).build(ui, &mut p.fb2_sharpen_radius);
            drop(_text_color);
            osc_tooltip(ui, "/block2/fb2/sharpen_radius", Some(p.fb2_sharpen_radius));
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block2.fb2_sharpen_radius".to_string(), 0.0, 5.0));
            }
            
            let _text_color = if is_learning("block2.fb2_filters_boost") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Filters Boost##fb2").speed(0.01).range(0.0, 2.0).build(ui, &mut p.fb2_filters_boost);
            drop(_text_color);
            osc_tooltip(ui, "/block2/fb2/filters_boost", Some(p.fb2_filters_boost));
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block2.fb2_filters_boost".to_string(), 0.0, 2.0));
            }
            
            ui.separator();
            
            // Temporal Filters
            let _text_color = if is_learning("block2.fb2_temporal_filter1_amount") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Temp Filter 1 Amount##fb2").speed(0.01).range(0.0, 1.0).build(ui, &mut p.fb2_temporal_filter1_amount);
            drop(_text_color);
            osc_tooltip(ui, "/block2/fb2/temp_filter1_amount", Some(p.fb2_temporal_filter1_amount));
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block2.fb2_temporal_filter1_amount".to_string(), 0.0, 1.0));
            }
            
            let _text_color = if is_learning("block2.fb2_temporal_filter1_resonance") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Temp Filter 1 Res##fb2").speed(0.01).range(0.0, 1.0).build(ui, &mut p.fb2_temporal_filter1_resonance);
            drop(_text_color);
            osc_tooltip(ui, "/block2/fb2/temp_filter1_res", Some(p.fb2_temporal_filter1_resonance));
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block2.fb2_temporal_filter1_resonance".to_string(), 0.0, 1.0));
            }
            
            let _text_color = if is_learning("block2.fb2_temporal_filter2_amount") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Temp Filter 2 Amount##fb2").speed(0.01).range(0.0, 1.0).build(ui, &mut p.fb2_temporal_filter2_amount);
            drop(_text_color);
            osc_tooltip(ui, "/block2/fb2/temp_filter2_amount", Some(p.fb2_temporal_filter2_amount));
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block2.fb2_temporal_filter2_amount".to_string(), 0.0, 1.0));
            }
            
            let _text_color = if is_learning("block2.fb2_temporal_filter2_resonance") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Temp Filter 2 Res##fb2").speed(0.01).range(0.0, 1.0).build(ui, &mut p.fb2_temporal_filter2_resonance);
            drop(_text_color);
            osc_tooltip(ui, "/block2/fb2/temp_filter2_res", Some(p.fb2_temporal_filter2_resonance));
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block2.fb2_temporal_filter2_resonance".to_string(), 0.0, 1.0));
            }
            
            ui.separator();
            // Delay section with tempo sync
            ui.text("Feedback Delay");
            
            // Tempo sync toggle
            let mut sync_enabled = p.fb2_delay_time_sync;
            if ui.checkbox("Sync to BPM##fb2delay", &mut sync_enabled) {
                p.fb2_delay_time_sync = sync_enabled;
            }
            if ui.is_item_hovered() {
                ui.tooltip_text("Enable to sync delay time to BPM");
            }
            
            if p.fb2_delay_time_sync {
                // Show beat division dropdown when sync is enabled
                let beat_divisions = ["1/16", "1/8", "1/4", "1/2", "1", "2", "4", "8"];
                let mut div_idx = p.fb2_delay_time_division as usize;
                let preview = beat_divisions[div_idx.min(beat_divisions.len() - 1)].to_string();
                ComboBox::new(ui, "##fb2_delay_division")
                    .preview_value(&preview)
                    .build(|| {
                        for (idx, opt) in beat_divisions.iter().enumerate() {
                            if ui.selectable_config(opt).selected(idx == div_idx).build() {
                                div_idx = idx;
                            }
                        }
                    });
                p.fb2_delay_time_division = div_idx.clamp(0, beat_divisions.len() - 1) as i32;
                
                // Show calculated delay time
                let calculated_frames = crate::core::lfo_engine::calculate_delay_frames_from_tempo(
                    self.bpm, p.fb2_delay_time_division, 60.0
                );
                ui.text_disabled(format!("Calculated: {} frames (≈ {:.2}s at {} BPM)", 
                    calculated_frames, calculated_frames as f32 / 60.0, self.bpm as i32));
            } else {
                // Show frame slider when sync is disabled
                let _text_color = if is_learning("block2.fb2_delay_time") {
                    Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
                } else { None };
                Drag::new("Delay Time (frames)##fb2").speed(1.0).range(0, 120).build(ui, &mut p.fb2_delay_time);
                drop(_text_color);
                if learn_mode && ui.is_item_clicked() { 
                    midi_learn_clicks.push(("block2.fb2_delay_time".to_string(), 0.0, 120.0));
                }
                if p.fb2_delay_time > 0 {
                    ui.text_disabled(format!("≈ {:.2} seconds at 60fps", p.fb2_delay_time as f32 / 60.0));
                } else {
                    ui.text_disabled("No delay (use immediate feedback)");
                }
            }
        }
        
        // Process MIDI learn clicks (deferred to avoid borrow issues)
        for (param_id, min, max) in midi_learn_clicks {
            self.handle_midi_learn_click(&param_id, min, max);
        }
    }

    /// Build Block 3 panel with sub-tabs
    fn build_block3_panel(&mut self, ui: &Ui) {
        let subtab_labels = ["Block 1 Re-process", "Block 2 Re-process", "Matrix Mixer", "Final Mix", "LFO"];
        let mut subtab_idx = self.block3_tab as usize;
        
        if let Some(_tab_bar) = ui.tab_bar("##block3_tabs") {
            for (idx, label) in subtab_labels.iter().enumerate() {
                let is_selected = idx == subtab_idx;
                
                if let Some(_tab) = ui.tab_item(label) {
                    if !is_selected {
                        subtab_idx = idx;
                        self.block3_tab = match idx {
                            0 => Block3Tab::Block1Reprocess,
                            1 => Block3Tab::Block2Reprocess,
                            2 => Block3Tab::MatrixMixer,
                            3 => Block3Tab::FinalMix,
                            4 => Block3Tab::Lfo,
                            _ => Block3Tab::FinalMix,
                        };
                    }
                }
            }
        }
        
        match self.block3_tab {
            Block3Tab::Block1Reprocess => self.build_block3_b1_reprocess(ui),
            Block3Tab::Block2Reprocess => self.build_block3_b2_reprocess(ui),
            Block3Tab::MatrixMixer => self.build_block3_matrix_mixer(ui),
            Block3Tab::FinalMix => self.build_block3_final_mix(ui),
            Block3Tab::Lfo => self.build_block3_lfo(ui),
        }
    }
    
    /// Build Block 3 Block 1 Re-process panel
    fn build_block3_b1_reprocess(&mut self, ui: &Ui) {
        // Extract MIDI learn state before borrowing self mutably
        let learn_mode = self.midi_learn_mode;
        let learn_target = self.midi_learn_target.clone();
        
        // Collect clicks for deferred processing
        let mut midi_learn_clicks: Vec<(String, f32, f32)> = Vec::new();
        
        let p = &mut self.block3_edit;
        
        // Helper to check if a parameter is being learned
        let is_learning = |param_id: &str| -> bool {
            learn_mode && learn_target.as_deref() == Some(param_id)
        };
        
        // Geometry section
        if CollapsingHeader::new("Geometry").default_open(true).build(ui) {
            let _text_color = if is_learning("block3.block1_x_displace") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("X Displace##b3b1").speed(0.01).range(-2.0, 2.0).build(ui, &mut p.block1_x_displace);
            drop(_text_color);
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block3.block1_x_displace".to_string(), -2.0, 2.0));
            }
            
            let _text_color = if is_learning("block3.block1_y_displace") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Y Displace##b3b1").speed(0.01).range(-2.0, 2.0).build(ui, &mut p.block1_y_displace);
            drop(_text_color);
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block3.block1_y_displace".to_string(), -2.0, 2.0));
            }
            
            let _text_color = if is_learning("block3.block1_z_displace") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Z Displace##b3b1").speed(0.01).range(0.0, 10.0).build(ui, &mut p.block1_z_displace);
            drop(_text_color);
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block3.block1_z_displace".to_string(), 0.0, 10.0));
            }
            
            let _text_color = if is_learning("block3.block1_rotate") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Rotate##b3b1").speed(0.1).range(-360.0, 360.0).build(ui, &mut p.block1_rotate);
            drop(_text_color);
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block3.block1_rotate".to_string(), -360.0, 360.0));
            }
            
            // Shear matrix - individual sliders
            ui.text("Shear Matrix:");
            let mut shear_x = p.block1_shear_matrix_x;
            let _text_color = if is_learning("block3.block1_shear_matrix_x") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("X##b3b1shear").speed(0.01).range(-2.0, 2.0).build(ui, &mut shear_x);
            drop(_text_color);
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block3.block1_shear_matrix_x".to_string(), -2.0, 2.0));
            }
            p.block1_shear_matrix_x = shear_x;
            
            let mut shear_y = p.block1_shear_matrix_y;
            let _text_color = if is_learning("block3.block1_shear_matrix_y") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Y##b3b1shear").speed(0.01).range(-2.0, 2.0).build(ui, &mut shear_y);
            drop(_text_color);
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block3.block1_shear_matrix_y".to_string(), -2.0, 2.0));
            }
            p.block1_shear_matrix_y = shear_y;
            
            let mut shear_z = p.block1_shear_matrix_z;
            let _text_color = if is_learning("block3.block1_shear_matrix_z") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Z##b3b1shear").speed(0.01).range(-2.0, 2.0).build(ui, &mut shear_z);
            drop(_text_color);
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block3.block1_shear_matrix_z".to_string(), -2.0, 2.0));
            }
            p.block1_shear_matrix_z = shear_z;
            
            let mut shear_w = p.block1_shear_matrix_w;
            let _text_color = if is_learning("block3.block1_shear_matrix_w") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("W##b3b1shear").speed(0.01).range(-2.0, 2.0).build(ui, &mut shear_w);
            drop(_text_color);
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block3.block1_shear_matrix_w".to_string(), -2.0, 2.0));
            }
            p.block1_shear_matrix_w = shear_w;
            
            p.block1_shear_matrix = Vec4::new(shear_x, shear_y, shear_z, shear_w);
            
            let _text_color = if is_learning("block3.block1_kaleidoscope_amount") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Kaleidoscope Amount##b3b1").speed(0.01).range(0.0, 1.0).build(ui, &mut p.block1_kaleidoscope_amount);
            drop(_text_color);
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block3.block1_kaleidoscope_amount".to_string(), 0.0, 1.0));
            }
            
            let _text_color = if is_learning("block3.block1_kaleidoscope_slice") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Kaleidoscope Slice##b3b1").speed(0.01).range(0.0, 1.0).build(ui, &mut p.block1_kaleidoscope_slice);
            drop(_text_color);
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block3.block1_kaleidoscope_slice".to_string(), 0.0, 1.0));
            }
            
            let mut overflow_idx = p.block1_geo_overflow as usize;
            let preview = GEO_OVERFLOW_MODES[overflow_idx].to_string();
            ComboBox::new(ui, "Overflow Mode##b3b1")
                .preview_value(&preview)
                .build(|| {
                    for (idx, opt) in GEO_OVERFLOW_MODES.iter().enumerate() {
                        if ui.selectable_config(opt).selected(idx == overflow_idx).build() {
                            overflow_idx = idx;
                        }
                    }
                });
            p.block1_geo_overflow = overflow_idx.clamp(0, GEO_OVERFLOW_MODES.len() - 1) as i32;
            
            ui.checkbox("H Mirror##b3b1", &mut p.block1_h_mirror);
            ui.checkbox("V Mirror##b3b1", &mut p.block1_v_mirror);
            ui.checkbox("H Flip##b3b1", &mut p.block1_h_flip);
            ui.checkbox("V Flip##b3b1", &mut p.block1_v_flip);
        }
        
        // Colorize section
        if CollapsingHeader::new("Colorize").default_open(true).build(ui) {
            ui.checkbox("Enable Colorize##b3b1", &mut p.block1_colorize_switch);
            
            if p.block1_colorize_switch {
                let colorize_modes = ["HSB", "RGB"];
                let mut mode_idx = p.block1_colorize_hsb_rgb as usize;
                let preview = colorize_modes[mode_idx].to_string();
                ComboBox::new(ui, "Colorize Mode##b3b1")
                    .preview_value(&preview)
                    .build(|| {
                        for (idx, opt) in colorize_modes.iter().enumerate() {
                            if ui.selectable_config(opt).selected(idx == mode_idx).build() {
                                mode_idx = idx;
                            }
                        }
                    });
                p.block1_colorize_hsb_rgb = mode_idx.clamp(0, 1) as i32;
                
                // Colorize bands are stored as HSB values (matching the shader)
                // Convert RGB color picker output to HSB
                let mut band1_rgb = [0.0f32; 3];
                // Convert current HSB to RGB for display
                let (r, g, b) = crate::utils::color::hsb_to_rgb(p.block1_colorize_band1.x, p.block1_colorize_band1.y, p.block1_colorize_band1.z);
                band1_rgb = [r, g, b];
                ui.color_edit3("Band 1##b3b1", &mut band1_rgb);
                // Convert back to HSB for storage
                let (h, s, v) = crate::utils::color::rgb_to_hsb(band1_rgb[0], band1_rgb[1], band1_rgb[2]);
                p.block1_colorize_band1 = Vec3::new(h, s, v);
                p.block1_colorize_band1_x = h;
                p.block1_colorize_band1_y = s;
                p.block1_colorize_band1_z = v;
                
                let mut band2_rgb = [0.0f32; 3];
                let (r, g, b) = crate::utils::color::hsb_to_rgb(p.block1_colorize_band2.x, p.block1_colorize_band2.y, p.block1_colorize_band2.z);
                band2_rgb = [r, g, b];
                ui.color_edit3("Band 2##b3b1", &mut band2_rgb);
                let (h, s, v) = crate::utils::color::rgb_to_hsb(band2_rgb[0], band2_rgb[1], band2_rgb[2]);
                p.block1_colorize_band2 = Vec3::new(h, s, v);
                p.block1_colorize_band2_x = h;
                p.block1_colorize_band2_y = s;
                p.block1_colorize_band2_z = v;
                
                let mut band3_rgb = [0.0f32; 3];
                let (r, g, b) = crate::utils::color::hsb_to_rgb(p.block1_colorize_band3.x, p.block1_colorize_band3.y, p.block1_colorize_band3.z);
                band3_rgb = [r, g, b];
                ui.color_edit3("Band 3##b3b1", &mut band3_rgb);
                let (h, s, v) = crate::utils::color::rgb_to_hsb(band3_rgb[0], band3_rgb[1], band3_rgb[2]);
                p.block1_colorize_band3 = Vec3::new(h, s, v);
                p.block1_colorize_band3_x = h;
                p.block1_colorize_band3_y = s;
                p.block1_colorize_band3_z = v;
                
                let mut band4_rgb = [0.0f32; 3];
                let (r, g, b) = crate::utils::color::hsb_to_rgb(p.block1_colorize_band4.x, p.block1_colorize_band4.y, p.block1_colorize_band4.z);
                band4_rgb = [r, g, b];
                ui.color_edit3("Band 4##b3b1", &mut band4_rgb);
                let (h, s, v) = crate::utils::color::rgb_to_hsb(band4_rgb[0], band4_rgb[1], band4_rgb[2]);
                p.block1_colorize_band4 = Vec3::new(h, s, v);
                p.block1_colorize_band4_x = h;
                p.block1_colorize_band4_y = s;
                p.block1_colorize_band4_z = v;
                
                let mut band5_rgb = [0.0f32; 3];
                let (r, g, b) = crate::utils::color::hsb_to_rgb(p.block1_colorize_band5.x, p.block1_colorize_band5.y, p.block1_colorize_band5.z);
                band5_rgb = [r, g, b];
                ui.color_edit3("Band 5##b3b1", &mut band5_rgb);
                let (h, s, v) = crate::utils::color::rgb_to_hsb(band5_rgb[0], band5_rgb[1], band5_rgb[2]);
                p.block1_colorize_band5 = Vec3::new(h, s, v);
                p.block1_colorize_band5_x = h;
                p.block1_colorize_band5_y = s;
                p.block1_colorize_band5_z = v;
            }
        }
        
        // Filters section
        if CollapsingHeader::new("Filters").default_open(true).build(ui) {
            let _text_color = if is_learning("block3.block1_blur_amount") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Blur Amount##b3b1").speed(0.01).range(0.0, 1.0).build(ui, &mut p.block1_blur_amount);
            drop(_text_color);
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block3.block1_blur_amount".to_string(), 0.0, 1.0));
            }
            
            let _text_color = if is_learning("block3.block1_blur_radius") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Blur Radius##b3b1").speed(0.01).range(0.0, 5.0).build(ui, &mut p.block1_blur_radius);
            drop(_text_color);
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block3.block1_blur_radius".to_string(), 0.0, 5.0));
            }
            
            let _text_color = if is_learning("block3.block1_sharpen_amount") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Sharpen Amount##b3b1").speed(0.01).range(0.0, 1.0).build(ui, &mut p.block1_sharpen_amount);
            drop(_text_color);
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block3.block1_sharpen_amount".to_string(), 0.0, 1.0));
            }
            
            let _text_color = if is_learning("block3.block1_sharpen_radius") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Sharpen Radius##b3b1").speed(0.01).range(0.0, 5.0).build(ui, &mut p.block1_sharpen_radius);
            drop(_text_color);
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block3.block1_sharpen_radius".to_string(), 0.0, 5.0));
            }
            
            let _text_color = if is_learning("block3.block1_filters_boost") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Filters Boost##b3b1").speed(0.01).range(0.0, 2.0).build(ui, &mut p.block1_filters_boost);
            drop(_text_color);
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block3.block1_filters_boost".to_string(), 0.0, 2.0));
            }
            
            ui.checkbox("Dither##b3b1", &mut p.block1_dither_switch);
            if p.block1_dither_switch {
                let _text_color = if is_learning("block3.block1_dither") {
                    Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
                } else { None };
                Drag::new("Dither Amount##b3b1").speed(0.1).range(1.0, 64.0).build(ui, &mut p.block1_dither);
                drop(_text_color);
                if learn_mode && ui.is_item_clicked() { 
                    midi_learn_clicks.push(("block3.block1_dither".to_string(), 1.0, 64.0));
                }
                let dither_types = [
                    "Bayer 4x4",
                    "Bayer 8x8", 
                    "Blue Noise",
                    "White Noise",
                    "IGN",
                    "Scanlines",
                    "Checkerboard",
                    "Stripes",
                    "Bit Crush",
                    "1-Bit Threshold",
                    "Pixel Sort",
                    "Atkinson",
                    "RGB Split"
                ];
                let mut dither_idx = p.block1_dither_type as usize;
                dither_idx = dither_idx.min(dither_types.len() - 1);
                let preview = dither_types[dither_idx].to_string();
                ComboBox::new(ui, "Dither Type##b3b1")
                    .preview_value(&preview)
                    .build(|| {
                        for (idx, opt) in dither_types.iter().enumerate() {
                            if ui.selectable_config(opt).selected(idx == dither_idx).build() {
                                dither_idx = idx;
                            }
                        }
                    });
                p.block1_dither_type = dither_idx.clamp(0, dither_types.len() - 1) as i32;
            }
        }
        
        // Process MIDI learn clicks (deferred to avoid borrow issues)
        for (param_id, min, max) in midi_learn_clicks {
            self.handle_midi_learn_click(&param_id, min, max);
        }
    }

    /// Build Block 3 Block 2 Re-process panel
    fn build_block3_b2_reprocess(&mut self, ui: &Ui) {
        // Extract MIDI learn state before borrowing self mutably
        let learn_mode = self.midi_learn_mode;
        let learn_target = self.midi_learn_target.clone();
        
        // Collect clicks for deferred processing
        let mut midi_learn_clicks: Vec<(String, f32, f32)> = Vec::new();
        
        let p = &mut self.block3_edit;
        
        // Helper to check if a parameter is being learned
        let is_learning = |param_id: &str| -> bool {
            learn_mode && learn_target.as_deref() == Some(param_id)
        };
        
        // Geometry section
        if CollapsingHeader::new("Geometry").default_open(true).build(ui) {
            let _text_color = if is_learning("block3.block2_x_displace") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("X Displace##b3b2").speed(0.01).range(-2.0, 2.0).build(ui, &mut p.block2_x_displace);
            drop(_text_color);
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block3.block2_x_displace".to_string(), -2.0, 2.0));
            }
            
            let _text_color = if is_learning("block3.block2_y_displace") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Y Displace##b3b2").speed(0.01).range(-2.0, 2.0).build(ui, &mut p.block2_y_displace);
            drop(_text_color);
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block3.block2_y_displace".to_string(), -2.0, 2.0));
            }
            
            let _text_color = if is_learning("block3.block2_z_displace") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Z Displace##b3b2").speed(0.01).range(0.0, 10.0).build(ui, &mut p.block2_z_displace);
            drop(_text_color);
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block3.block2_z_displace".to_string(), 0.0, 10.0));
            }
            
            let _text_color = if is_learning("block3.block2_rotate") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Rotate##b3b2").speed(0.1).range(-360.0, 360.0).build(ui, &mut p.block2_rotate);
            drop(_text_color);
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block3.block2_rotate".to_string(), -360.0, 360.0));
            }
            
            // Shear matrix - individual sliders
            ui.text("Shear Matrix:");
            let mut shear_x = p.block2_shear_matrix_x;
            let _text_color = if is_learning("block3.block2_shear_matrix_x") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("X##b3b2shear").speed(0.01).range(-2.0, 2.0).build(ui, &mut shear_x);
            drop(_text_color);
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block3.block2_shear_matrix_x".to_string(), -2.0, 2.0));
            }
            p.block2_shear_matrix_x = shear_x;
            
            let mut shear_y = p.block2_shear_matrix_y;
            let _text_color = if is_learning("block3.block2_shear_matrix_y") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Y##b3b2shear").speed(0.01).range(-2.0, 2.0).build(ui, &mut shear_y);
            drop(_text_color);
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block3.block2_shear_matrix_y".to_string(), -2.0, 2.0));
            }
            p.block2_shear_matrix_y = shear_y;
            
            let mut shear_z = p.block2_shear_matrix_z;
            let _text_color = if is_learning("block3.block2_shear_matrix_z") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Z##b3b2shear").speed(0.01).range(-2.0, 2.0).build(ui, &mut shear_z);
            drop(_text_color);
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block3.block2_shear_matrix_z".to_string(), -2.0, 2.0));
            }
            p.block2_shear_matrix_z = shear_z;
            
            let mut shear_w = p.block2_shear_matrix_w;
            let _text_color = if is_learning("block3.block2_shear_matrix_w") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("W##b3b2shear").speed(0.01).range(-2.0, 2.0).build(ui, &mut shear_w);
            drop(_text_color);
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block3.block2_shear_matrix_w".to_string(), -2.0, 2.0));
            }
            p.block2_shear_matrix_w = shear_w;
            
            p.block2_shear_matrix = Vec4::new(shear_x, shear_y, shear_z, shear_w);
            
            let _text_color = if is_learning("block3.block2_kaleidoscope_amount") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Kaleidoscope Amount##b3b2").speed(0.01).range(0.0, 1.0).build(ui, &mut p.block2_kaleidoscope_amount);
            drop(_text_color);
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block3.block2_kaleidoscope_amount".to_string(), 0.0, 1.0));
            }
            
            let _text_color = if is_learning("block3.block2_kaleidoscope_slice") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Kaleidoscope Slice##b3b2").speed(0.01).range(0.0, 1.0).build(ui, &mut p.block2_kaleidoscope_slice);
            drop(_text_color);
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block3.block2_kaleidoscope_slice".to_string(), 0.0, 1.0));
            }
            
            let mut overflow_idx = p.block2_geo_overflow as usize;
            let preview = GEO_OVERFLOW_MODES[overflow_idx].to_string();
            ComboBox::new(ui, "Overflow Mode##b3b2")
                .preview_value(&preview)
                .build(|| {
                    for (idx, opt) in GEO_OVERFLOW_MODES.iter().enumerate() {
                        if ui.selectable_config(opt).selected(idx == overflow_idx).build() {
                            overflow_idx = idx;
                        }
                    }
                });
            p.block2_geo_overflow = overflow_idx.clamp(0, GEO_OVERFLOW_MODES.len() - 1) as i32;
            
            ui.checkbox("H Mirror##b3b2", &mut p.block2_h_mirror);
            ui.checkbox("V Mirror##b3b2", &mut p.block2_v_mirror);
            ui.checkbox("H Flip##b3b2", &mut p.block2_h_flip);
            ui.checkbox("V Flip##b3b2", &mut p.block2_v_flip);
        }
        
        // Colorize section
        if CollapsingHeader::new("Colorize").default_open(true).build(ui) {
            ui.checkbox("Enable Colorize##b3b2", &mut p.block2_colorize_switch);
            
            if p.block2_colorize_switch {
                let colorize_modes = ["HSB", "RGB"];
                let mut mode_idx = p.block2_colorize_hsb_rgb as usize;
                let preview = colorize_modes[mode_idx].to_string();
                ComboBox::new(ui, "Colorize Mode##b3b2")
                    .preview_value(&preview)
                    .build(|| {
                        for (idx, opt) in colorize_modes.iter().enumerate() {
                            if ui.selectable_config(opt).selected(idx == mode_idx).build() {
                                mode_idx = idx;
                            }
                        }
                    });
                p.block2_colorize_hsb_rgb = mode_idx.clamp(0, 1) as i32;
                
                // Colorize bands are stored as HSB values (matching the shader)
                // Convert RGB color picker output to HSB
                let mut band1_rgb = [0.0f32; 3];
                let (r, g, b) = crate::utils::color::hsb_to_rgb(p.block2_colorize_band1.x, p.block2_colorize_band1.y, p.block2_colorize_band1.z);
                band1_rgb = [r, g, b];
                ui.color_edit3("Band 1##b3b2", &mut band1_rgb);
                let (h, s, v) = crate::utils::color::rgb_to_hsb(band1_rgb[0], band1_rgb[1], band1_rgb[2]);
                p.block2_colorize_band1 = Vec3::new(h, s, v);
                p.block2_colorize_band1_x = h;
                p.block2_colorize_band1_y = s;
                p.block2_colorize_band1_z = v;
                
                let mut band2_rgb = [0.0f32; 3];
                let (r, g, b) = crate::utils::color::hsb_to_rgb(p.block2_colorize_band2.x, p.block2_colorize_band2.y, p.block2_colorize_band2.z);
                band2_rgb = [r, g, b];
                ui.color_edit3("Band 2##b3b2", &mut band2_rgb);
                let (h, s, v) = crate::utils::color::rgb_to_hsb(band2_rgb[0], band2_rgb[1], band2_rgb[2]);
                p.block2_colorize_band2 = Vec3::new(h, s, v);
                p.block2_colorize_band2_x = h;
                p.block2_colorize_band2_y = s;
                p.block2_colorize_band2_z = v;
                
                let mut band3_rgb = [0.0f32; 3];
                let (r, g, b) = crate::utils::color::hsb_to_rgb(p.block2_colorize_band3.x, p.block2_colorize_band3.y, p.block2_colorize_band3.z);
                band3_rgb = [r, g, b];
                ui.color_edit3("Band 3##b3b2", &mut band3_rgb);
                let (h, s, v) = crate::utils::color::rgb_to_hsb(band3_rgb[0], band3_rgb[1], band3_rgb[2]);
                p.block2_colorize_band3 = Vec3::new(h, s, v);
                p.block2_colorize_band3_x = h;
                p.block2_colorize_band3_y = s;
                p.block2_colorize_band3_z = v;
                
                let mut band4_rgb = [0.0f32; 3];
                let (r, g, b) = crate::utils::color::hsb_to_rgb(p.block2_colorize_band4.x, p.block2_colorize_band4.y, p.block2_colorize_band4.z);
                band4_rgb = [r, g, b];
                ui.color_edit3("Band 4##b3b2", &mut band4_rgb);
                let (h, s, v) = crate::utils::color::rgb_to_hsb(band4_rgb[0], band4_rgb[1], band4_rgb[2]);
                p.block2_colorize_band4 = Vec3::new(h, s, v);
                p.block2_colorize_band4_x = h;
                p.block2_colorize_band4_y = s;
                p.block2_colorize_band4_z = v;
                
                let mut band5_rgb = [0.0f32; 3];
                let (r, g, b) = crate::utils::color::hsb_to_rgb(p.block2_colorize_band5.x, p.block2_colorize_band5.y, p.block2_colorize_band5.z);
                band5_rgb = [r, g, b];
                ui.color_edit3("Band 5##b3b2", &mut band5_rgb);
                let (h, s, v) = crate::utils::color::rgb_to_hsb(band5_rgb[0], band5_rgb[1], band5_rgb[2]);
                p.block2_colorize_band5 = Vec3::new(h, s, v);
                p.block2_colorize_band5_x = h;
                p.block2_colorize_band5_y = s;
                p.block2_colorize_band5_z = v;
            }
        }
        
        // Filters section
        if CollapsingHeader::new("Filters").default_open(true).build(ui) {
            let _text_color = if is_learning("block3.block2_blur_amount") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Blur Amount##b3b2").speed(0.01).range(0.0, 1.0).build(ui, &mut p.block2_blur_amount);
            drop(_text_color);
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block3.block2_blur_amount".to_string(), 0.0, 1.0));
            }
            
            let _text_color = if is_learning("block3.block2_blur_radius") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Blur Radius##b3b2").speed(0.01).range(0.0, 5.0).build(ui, &mut p.block2_blur_radius);
            drop(_text_color);
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block3.block2_blur_radius".to_string(), 0.0, 5.0));
            }
            
            let _text_color = if is_learning("block3.block2_sharpen_amount") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Sharpen Amount##b3b2").speed(0.01).range(0.0, 1.0).build(ui, &mut p.block2_sharpen_amount);
            drop(_text_color);
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block3.block2_sharpen_amount".to_string(), 0.0, 1.0));
            }
            
            let _text_color = if is_learning("block3.block2_sharpen_radius") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Sharpen Radius##b3b2").speed(0.01).range(0.0, 5.0).build(ui, &mut p.block2_sharpen_radius);
            drop(_text_color);
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block3.block2_sharpen_radius".to_string(), 0.0, 5.0));
            }
            
            let _text_color = if is_learning("block3.block2_filters_boost") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Filters Boost##b3b2").speed(0.01).range(0.0, 2.0).build(ui, &mut p.block2_filters_boost);
            drop(_text_color);
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block3.block2_filters_boost".to_string(), 0.0, 2.0));
            }
            
            ui.checkbox("Dither##b3b2", &mut p.block2_dither_switch);
            if p.block2_dither_switch {
                let _text_color = if is_learning("block3.block2_dither") {
                    Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
                } else { None };
                Drag::new("Dither Amount##b3b2").speed(0.1).range(1.0, 64.0).build(ui, &mut p.block2_dither);
                drop(_text_color);
                if learn_mode && ui.is_item_clicked() { 
                    midi_learn_clicks.push(("block3.block2_dither".to_string(), 1.0, 64.0));
                }
                let dither_types = [
                    "Bayer 4x4",
                    "Bayer 8x8", 
                    "Blue Noise",
                    "White Noise",
                    "IGN",
                    "Scanlines",
                    "Checkerboard",
                    "Stripes",
                    "Bit Crush",
                    "1-Bit Threshold",
                    "Pixel Sort",
                    "Atkinson",
                    "RGB Split"
                ];
                let mut dither_idx = p.block2_dither_type as usize;
                dither_idx = dither_idx.min(dither_types.len() - 1);
                let preview = dither_types[dither_idx].to_string();
                ComboBox::new(ui, "Dither Type##b3b2")
                    .preview_value(&preview)
                    .build(|| {
                        for (idx, opt) in dither_types.iter().enumerate() {
                            if ui.selectable_config(opt).selected(idx == dither_idx).build() {
                                dither_idx = idx;
                            }
                        }
                    });
                p.block2_dither_type = dither_idx.clamp(0, dither_types.len() - 1) as i32;
            }
        }
        
        // Process MIDI learn clicks (deferred to avoid borrow issues)
        for (param_id, min, max) in midi_learn_clicks {
            self.handle_midi_learn_click(&param_id, min, max);
        }
    }

    /// Build Block 3 Matrix Mixer panel
    fn build_block3_matrix_mixer(&mut self, ui: &Ui) {
        // Extract MIDI learn state before borrowing self mutably
        let learn_mode = self.midi_learn_mode;
        let learn_target = self.midi_learn_target.clone();
        
        // Collect clicks for deferred processing
        let mut midi_learn_clicks: Vec<(String, f32, f32)> = Vec::new();
        
        let p = &mut self.block3_edit;
        
        // Helper to check if a parameter is being learned
        let is_learning = |param_id: &str| -> bool {
            learn_mode && learn_target.as_deref() == Some(param_id)
        };
        
        // Mix Order (which block is foreground/background)
        if CollapsingHeader::new("Mix Order").default_open(true).build(ui) {
            let order_labels = ["Block 1 → Block 2 (B1 FG, B2 BG)", "Block 2 → Block 1 (B2 FG, B1 BG)"];
            let mut order_idx = p.final_key_order as usize;
            
            ui.radio_button(&order_labels[0], &mut order_idx, 0);
            ui.radio_button(&order_labels[1], &mut order_idx, 1);
            
            p.final_key_order = order_idx.clamp(0, 1) as i32;
            
            ui.text_disabled("Changes which block is foreground/background in matrix mix");
        }
        
        ui.separator();
        ui.text("Matrix Mix Type:");
        let mut mix_type = p.matrix_mix_type as usize;
        let preview = MIX_TYPES[mix_type].to_string();
        ComboBox::new(ui, "##matrix_mix_type")
            .preview_value(&preview)
            .build(|| {
                for (idx, opt) in MIX_TYPES.iter().enumerate() {
                    if ui.selectable_config(opt).selected(idx == mix_type).build() {
                        mix_type = idx;
                    }
                }
            });
        p.matrix_mix_type = mix_type.clamp(0, MIX_TYPES.len() - 1) as i32;
        
        let mut overflow = p.matrix_mix_overflow as usize;
        let preview2 = GEO_OVERFLOW_MODES[overflow].to_string();
        ComboBox::new(ui, "##matrix_overflow")
            .preview_value(&preview2)
            .build(|| {
                for (idx, opt) in GEO_OVERFLOW_MODES.iter().enumerate() {
                    if ui.selectable_config(opt).selected(idx == overflow).build() {
                        overflow = idx;
                    }
                }
            });
        p.matrix_mix_overflow = overflow.clamp(0, GEO_OVERFLOW_MODES.len() - 1) as i32;
        
        ui.separator();
        ui.text("RGB Channel Mixing:");
        
        if CollapsingHeader::new("Background RGB into Foreground Red").default_open(true).build(ui) {
            // Individual sliders for matrix mix
            let mut red_r = p.matrix_mix_r_to_r;
            let _text_color = if is_learning("block3.matrix_mix_r_to_r") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("R→R##matrix").speed(0.002).range(-2.0, 2.0).build(ui, &mut red_r);
            drop(_text_color);
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block3.matrix_mix_r_to_r".to_string(), -2.0, 2.0));
            }
            p.matrix_mix_r_to_r = red_r;
            
            let mut green_r = p.matrix_mix_g_to_r;
            let _text_color = if is_learning("block3.matrix_mix_g_to_r") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("G→R##matrix").speed(0.002).range(-2.0, 2.0).build(ui, &mut green_r);
            drop(_text_color);
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block3.matrix_mix_g_to_r".to_string(), -2.0, 2.0));
            }
            p.matrix_mix_g_to_r = green_r;
            
            let mut blue_r = p.matrix_mix_b_to_r;
            let _text_color = if is_learning("block3.matrix_mix_b_to_r") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("B→R##matrix").speed(0.002).range(-2.0, 2.0).build(ui, &mut blue_r);
            drop(_text_color);
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block3.matrix_mix_b_to_r".to_string(), -2.0, 2.0));
            }
            p.matrix_mix_b_to_r = blue_r;
            
            p.bg_rgb_into_fg_red = Vec3::new(red_r, green_r, blue_r);
        }
        
        if CollapsingHeader::new("Background RGB into Foreground Green").default_open(true).build(ui) {
            let mut red_g = p.matrix_mix_r_to_g;
            let _text_color = if is_learning("block3.matrix_mix_r_to_g") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("R→G##matrix").speed(0.002).range(-2.0, 2.0).build(ui, &mut red_g);
            drop(_text_color);
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block3.matrix_mix_r_to_g".to_string(), -2.0, 2.0));
            }
            p.matrix_mix_r_to_g = red_g;
            
            let mut green_g = p.matrix_mix_g_to_g;
            let _text_color = if is_learning("block3.matrix_mix_g_to_g") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("G→G##matrix").speed(0.002).range(-2.0, 2.0).build(ui, &mut green_g);
            drop(_text_color);
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block3.matrix_mix_g_to_g".to_string(), -2.0, 2.0));
            }
            p.matrix_mix_g_to_g = green_g;
            
            let mut blue_g = p.matrix_mix_b_to_g;
            let _text_color = if is_learning("block3.matrix_mix_b_to_g") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("B→G##matrix").speed(0.002).range(-2.0, 2.0).build(ui, &mut blue_g);
            drop(_text_color);
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block3.matrix_mix_b_to_g".to_string(), -2.0, 2.0));
            }
            p.matrix_mix_b_to_g = blue_g;
            
            p.bg_rgb_into_fg_green = Vec3::new(red_g, green_g, blue_g);
        }
        
        if CollapsingHeader::new("Background RGB into Foreground Blue").default_open(true).build(ui) {
            let mut red_b = p.matrix_mix_r_to_b;
            let _text_color = if is_learning("block3.matrix_mix_r_to_b") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("R→B##matrix").speed(0.002).range(-2.0, 2.0).build(ui, &mut red_b);
            drop(_text_color);
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block3.matrix_mix_r_to_b".to_string(), -2.0, 2.0));
            }
            p.matrix_mix_r_to_b = red_b;
            
            let mut green_b = p.matrix_mix_g_to_b;
            let _text_color = if is_learning("block3.matrix_mix_g_to_b") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("G→B##matrix").speed(0.002).range(-2.0, 2.0).build(ui, &mut green_b);
            drop(_text_color);
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block3.matrix_mix_g_to_b".to_string(), -2.0, 2.0));
            }
            p.matrix_mix_g_to_b = green_b;
            
            let mut blue_b = p.matrix_mix_b_to_b;
            let _text_color = if is_learning("block3.matrix_mix_b_to_b") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("B→B##matrix").speed(0.002).range(-2.0, 2.0).build(ui, &mut blue_b);
            drop(_text_color);
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block3.matrix_mix_b_to_b".to_string(), -2.0, 2.0));
            }
            p.matrix_mix_b_to_b = blue_b;
            
            p.bg_rgb_into_fg_blue = Vec3::new(red_b, green_b, blue_b);
        }
        
        // Process MIDI learn clicks (deferred to avoid borrow issues)
        for (param_id, min, max) in midi_learn_clicks {
            self.handle_midi_learn_click(&param_id, min, max);
        }
    }

    /// Build Block 3 Final Mix panel
    fn build_block3_final_mix(&mut self, ui: &Ui) {
        // Extract MIDI learn state before borrowing self mutably
        let learn_mode = self.midi_learn_mode;
        let learn_target = self.midi_learn_target.clone();
        
        // Collect clicks for deferred processing
        let mut midi_learn_clicks: Vec<(String, f32, f32)> = Vec::new();
        
        let p = &mut self.block3_edit;
        
        // Helper to check if a parameter is being learned
        let is_learning = |param_id: &str| -> bool {
            learn_mode && learn_target.as_deref() == Some(param_id)
        };
        
        // Final mix section
        if CollapsingHeader::new("Final Mix").default_open(true).build(ui) {
            // Mix amount with fine control
            let _text_color = if is_learning("block3.final_mix_amount") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Mix Amount##final").speed(0.002).range(0.0, 1.0).build(ui, &mut p.final_mix_amount);
            drop(_text_color);
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block3.final_mix_amount".to_string(), 0.0, 1.0));
            }
            
            let mut mix_type = p.final_mix_type as usize;
            let preview = MIX_TYPES[mix_type].to_string();
            ComboBox::new(ui, "Mix Type##final")
                .preview_value(&preview)
                .build(|| {
                    for (idx, opt) in MIX_TYPES.iter().enumerate() {
                        if ui.selectable_config(opt).selected(idx == mix_type).build() {
                            mix_type = idx;
                        }
                    }
                });
            p.final_mix_type = mix_type.clamp(0, MIX_TYPES.len() - 1) as i32;
            
            let mut overflow = p.final_mix_overflow as usize;
            let preview2 = GEO_OVERFLOW_MODES[overflow].to_string();
            ComboBox::new(ui, "Mix Overflow##final")
                .preview_value(&preview2)
                .build(|| {
                    for (idx, opt) in GEO_OVERFLOW_MODES.iter().enumerate() {
                        if ui.selectable_config(opt).selected(idx == overflow).build() {
                            overflow = idx;
                        }
                    }
                });
            p.final_mix_overflow = overflow.clamp(0, GEO_OVERFLOW_MODES.len() - 1) as i32;
            
            let _text_color = if is_learning("block3.final_key_order") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Key Order##final").speed(1.0).range(0, 10).build(ui, &mut p.final_key_order);
            drop(_text_color);
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block3.final_key_order".to_string(), 0.0, 10.0));
            }
        }
        
        // Final key section
        if CollapsingHeader::new("Final Key").default_open(true).build(ui) {
            let mut key_color = [p.final_key_value.x, p.final_key_value.y, p.final_key_value.z];
            ui.color_edit3("Key Color##final", &mut key_color);
            p.final_key_value = Vec3::new(key_color[0], key_color[1], key_color[2]);
            
            // Key threshold with fine control
            let _text_color = if is_learning("block3.final_key_threshold") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Key Threshold##final").speed(0.001).range(0.0, 1.0).build(ui, &mut p.final_key_threshold);
            drop(_text_color);
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block3.final_key_threshold".to_string(), 0.0, 1.0));
            }
            
            let _text_color = if is_learning("block3.final_key_soft") {
                Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
            } else { None };
            Drag::new("Key Soft##final").speed(0.002).range(0.0, 1.0).build(ui, &mut p.final_key_soft);
            drop(_text_color);
            if learn_mode && ui.is_item_clicked() { 
                midi_learn_clicks.push(("block3.final_key_soft".to_string(), 0.0, 1.0));
            }
        }
        
        // Output Dither section
        if CollapsingHeader::new("Output Dither").default_open(true).build(ui) {
            ui.checkbox("Dither##final", &mut p.final_dither_switch);
            if p.final_dither_switch {
                let _text_color = if is_learning("block3.final_dither") {
                    Some(ui.push_style_color(imgui::StyleColor::Text, [0.0, 1.0, 0.0, 1.0]))
                } else { None };
                Drag::new("Dither Amount##final").speed(0.1).range(1.0, 64.0).build(ui, &mut p.final_dither);
                drop(_text_color);
                if learn_mode && ui.is_item_clicked() { 
                    midi_learn_clicks.push(("block3.final_dither".to_string(), 1.0, 64.0));
                }
                let dither_types = [
                    "Bayer 4x4",
                    "Bayer 8x8", 
                    "Blue Noise",
                    "White Noise",
                    "IGN",
                    "Scanlines",
                    "Checkerboard",
                    "Stripes",
                    "Bit Crush",
                    "1-Bit Threshold",
                    "Pixel Sort",
                    "Atkinson",
                    "RGB Split"
                ];
                let mut dither_idx = p.final_dither_type as usize;
                dither_idx = dither_idx.min(dither_types.len() - 1);
                let preview = dither_types[dither_idx].to_string();
                ComboBox::new(ui, "Dither Type##final")
                    .preview_value(&preview)
                    .build(|| {
                        for (idx, opt) in dither_types.iter().enumerate() {
                            if ui.selectable_config(opt).selected(idx == dither_idx).build() {
                                dither_idx = idx;
                            }
                        }
                    });
                p.final_dither_type = dither_idx.clamp(0, dither_types.len() - 1) as i32;
            }
        }
        
        // Process MIDI learn clicks (deferred to avoid borrow issues)
        for (param_id, min, max) in midi_learn_clicks {
            self.handle_midi_learn_click(&param_id, min, max);
        }
    }

    /// Build Macros panel (LFO controls)
    fn build_macros_panel(&mut self, ui: &Ui) {
        ui.text("LFO Banks (0-15)");
        ui.separator();
        
        // LFO bank selector
        ui.text("Select LFO Bank:");
        for i in 0..16 {
            if i > 0 && i % 8 != 0 {
                ui.same_line();
            }
            let label = format!("{}", i);
            if ui.radio_button_bool(&label, self.selected_lfo_bank == i) {
                self.selected_lfo_bank = i;
            }
        }
        
        ui.separator();
        
        // Edit selected LFO bank
        let bank_idx = self.selected_lfo_bank as usize;
        if let Ok(mut state) = self.shared_state.lock() {
            if bank_idx < state.lfo_banks.len() {
                let lfo = &mut state.lfo_banks[bank_idx];
                
                Drag::new("Rate").speed(0.01).range(-1.0, 1.0).build(ui, &mut lfo.rate);
                Drag::new("Amplitude").speed(0.01).range(0.0, 2.0).build(ui, &mut lfo.amplitude);
                Drag::new("Phase").speed(0.01).range(0.0, 1.0).build(ui, &mut lfo.phase);
                
                let mut waveform = lfo.waveform as usize;
                let preview = WAVEFORM_NAMES[waveform].to_string();
                ComboBox::new(ui, "Waveform")
                    .preview_value(&preview)
                    .build(|| {
                        for (idx, opt) in WAVEFORM_NAMES.iter().enumerate() {
                            if ui.selectable_config(opt).selected(idx == waveform).build() {
                                waveform = idx;
                            }
                        }
                    });
                lfo.waveform = waveform.clamp(0, WAVEFORM_NAMES.len() - 1) as i32;
                
                ui.checkbox("Tempo Sync", &mut lfo.tempo_sync);
                
                if lfo.tempo_sync {
                    let mut division = lfo.division as usize;
                    let preview = BEAT_DIVISIONS[division].to_string();
                    ComboBox::new(ui, "Beat Division")
                        .preview_value(&preview)
                        .build(|| {
                            for (idx, opt) in BEAT_DIVISIONS.iter().enumerate() {
                                if ui.selectable_config(opt).selected(idx == division).build() {
                                    division = idx;
                                }
                            }
                        });
                    lfo.division = division.clamp(0, BEAT_DIVISIONS.len() - 1) as i32;
                }
            }
        }
    }
    
    /// Build Inputs panel
    fn build_inputs_panel(&mut self, ui: &Ui) {
        // Input 1 section
        if CollapsingHeader::new("Input 1").default_open(true).build(ui) {
            let input_types = ["None", "Webcam", "NDI", "Syphon", "Spout", "Video File"];
            let mut type_idx = self.input1_type as usize;
            let old_type = type_idx;
            let preview = input_types[type_idx].to_string();
            
            ComboBox::new(ui, "##input1_type")
                .preview_value(&preview)
                .build(|| {
                    for (idx, opt) in input_types.iter().enumerate() {
                        if ui.selectable_config(opt).selected(idx == type_idx).build() {
                            type_idx = idx;
                        }
                    }
                });
            
            let type_changed = type_idx != old_type;
            self.input1_type = match type_idx {
                0 => InputType::None,
                1 => InputType::Webcam,
                2 => InputType::Ndi,
                3 => InputType::Syphon,
                4 => InputType::Spout,
                5 => InputType::VideoFile,
                _ => InputType::None,
            };
            
            // Auto-select first webcam if Webcam is chosen but no device selected
            if type_changed && self.input1_type == InputType::Webcam 
                && self.selected_webcam1 < 0 && !self.webcam_devices.is_empty() {
                self.selected_webcam1 = 0;
            }
            
            // Auto-select first NDI source if NDI is chosen but no source selected
            if type_changed && self.input1_type == InputType::Ndi 
                && self.selected_ndi_source1 < 0 && !self.ndi_sources.is_empty() {
                self.selected_ndi_source1 = 0;
            }
            
            // Auto-select first Syphon source if Syphon is chosen but no source selected
            if type_changed && self.input1_type == InputType::Syphon 
                && self.selected_syphon_source1 < 0 && !self.syphon_sources.is_empty() {
                self.selected_syphon_source1 = 0;
            }
            
            // Refresh Syphon sources when switching to Syphon
            if type_changed && self.input1_type == InputType::Syphon {
                self.refresh_syphon_sources();
            }

            // Auto-select first Spout source if Spout is chosen but no source selected
            if type_changed && self.input1_type == InputType::Spout
                && self.selected_spout_source1 < 0 && !self.spout_sources.is_empty() {
                self.selected_spout_source1 = 0;
            }

            // Refresh Spout sources when switching to Spout
            if type_changed && self.input1_type == InputType::Spout {
                self.refresh_spout_sources();
            }

            // Refresh NDI sources when switching to NDI
            if type_changed && self.input1_type == InputType::Ndi {
                self.refresh_ndi_sources();
            }
            
            // Stop current input and save config when input type changes
            if type_changed {
                log::info!("[GUI] Input 1 type changed from {} to {}, stopping current input", 
                    old_type, type_idx);
                // Stop the current input before switching
                if let Ok(mut state) = self.shared_state.lock() {
                    state.input1_change_request = InputChangeRequest::StopInput { input_id: 1 };
                }
                self.save_input_config();
            }
            
            // Webcam device selection
            if self.input1_type == InputType::Webcam {
                let devices: Vec<&str> = self.webcam_devices.iter().map(|s| s.as_str()).collect();
                if !devices.is_empty() {
                    let preview = if self.selected_webcam1 >= 0 { 
                        self.webcam_devices[self.selected_webcam1 as usize].clone()
                    } else { "Select device...".to_string() };
                    
                    // Check if selected device is a virtual camera
                    let is_virtual = self.selected_webcam1 >= 0 && 
                        self.webcam_devices[self.selected_webcam1 as usize].to_lowercase().contains("virtual");
                    
                    if is_virtual {
                        ui.text_colored([1.0, 0.5, 0.0, 1.0], 
                            "⚠️ Virtual cameras may not work on macOS.\nUse NDI from OBS instead.");
                    }
                    
                    let mut selected = self.selected_webcam1;
                    ComboBox::new(ui, "##webcam1_select")
                        .preview_value(&preview)
                        .build(|| {
                            for (idx, opt) in devices.iter().enumerate() {
                                if ui.selectable_config(opt).selected(idx == selected as usize).build() {
                                    selected = idx as i32;
                                }
                            }
                        });
                    let device_changed = self.selected_webcam1 != selected;
                    self.selected_webcam1 = selected;
                    
                    // Save config when device selection changes
                    if device_changed {
                        self.save_input_config();
                    }
                    
                    if ui.button("Start Webcam 1") && self.selected_webcam1 >= 0 {
                        if let Ok(mut state) = self.shared_state.lock() {
                            state.input1_change_request = InputChangeRequest::StartWebcam {
                                input_id: 1,
                                device_index: self.selected_webcam1 as usize,
                                width: 1280,
                                height: 720,
                                fps: 30,
                            };
                        }
                    }
                } else {
                    ui.text_disabled("No webcam devices found");
                }
            }
            
            // NDI source selection
            if self.input1_type == InputType::Ndi {
                let sources: Vec<&str> = self.ndi_sources.iter().map(|s| s.as_str()).collect();
                if !sources.is_empty() {
                    let preview = if self.selected_ndi_source1 >= 0 { 
                        self.ndi_sources[self.selected_ndi_source1 as usize].clone()
                    } else { "Select NDI source...".to_string() };
                    
                    let mut selected = self.selected_ndi_source1;
                    ComboBox::new(ui, "##ndi1_select")
                        .preview_value(&preview)
                        .build(|| {
                            for (idx, opt) in sources.iter().enumerate() {
                                if ui.selectable_config(opt).selected(idx == selected as usize).build() {
                                    selected = idx as i32;
                                }
                            }
                        });
                    let source_changed = self.selected_ndi_source1 != selected;
                    self.selected_ndi_source1 = selected;
                    
                    // Save config when source selection changes
                    if source_changed {
                        self.save_input_config();
                    }
                    
                    if ui.button("Start NDI Input 1") && self.selected_ndi_source1 >= 0 {
                        let source_name = self.ndi_sources[self.selected_ndi_source1 as usize].clone();
                        if let Ok(mut state) = self.shared_state.lock() {
                            state.input1_change_request = InputChangeRequest::StartNdi {
                                input_id: 1,
                                source_name,
                            };
                        }
                    }
                } else {
                    ui.text_disabled("No NDI sources found");
                    if ui.button("Refresh NDI Sources") {
                        self.refresh_ndi_sources();
                    }
                }
            }
            
            // Syphon source selection (macOS only)
            #[cfg(target_os = "macos")]
            if self.input1_type == InputType::Syphon {
                // Build safe sources list - never pass empty strings to ImGui
                let sources: Vec<(usize, &str)> = self.syphon_sources.iter()
                    .enumerate()
                    .map(|(i, s)| (i, if s.is_empty() { "(unnamed)" } else { s.as_str() }))
                    .collect();
                if !sources.is_empty() {
                    let preview = if self.selected_syphon_source1 >= 0 && 
                                     (self.selected_syphon_source1 as usize) < self.syphon_sources.len() {
                        let name = &self.syphon_sources[self.selected_syphon_source1 as usize];
                        if name.is_empty() { "(unnamed server)".to_string() } else { name.clone() }
                    } else { "Select Syphon source...".to_string() };
                    
                    let mut selected = self.selected_syphon_source1;
                    ComboBox::new(ui, "##syphon1_select")
                        .preview_value(&preview)
                        .build(|| {
                            for (idx, label) in sources.iter() {
                                if ui.selectable_config(label).selected(*idx == selected as usize).build() {
                                    selected = *idx as i32;
                                }
                            }
                        });
                    let source_changed = self.selected_syphon_source1 != selected;
                    self.selected_syphon_source1 = selected;
                    
                    // Save config when source selection changes
                    if source_changed {
                        self.save_input_config();
                    }
                    
                    // Refresh button next to dropdown
                    ui.same_line();
                    if ui.button("🔄") {
                        log::info!("[GUI] Refreshing Syphon sources for Input 1");
                        self.refresh_syphon_sources();
                    }
                    
                    if ui.button("Start Syphon Input 1") && self.selected_syphon_source1 >= 0 {
                        let source_name = self.syphon_sources[self.selected_syphon_source1 as usize].clone();
                        log::info!("[GUI] Requesting Syphon Input 1: {}", source_name);
                        if let Ok(mut state) = self.shared_state.lock() {
                            state.input1_change_request = InputChangeRequest::StartSyphon {
                                input_id: 1,
                                server_name: source_name,
                            };
                        }
                    }
                } else {
                    ui.text_disabled("No Syphon sources found");
                    if ui.button("Refresh Syphon Sources") {
                        self.refresh_syphon_sources();
                    }
                }
            }

            // Spout input source selection (Windows only)
            if self.input1_type == InputType::Spout {
                if !self.spout_sources.is_empty() {
                    let preview = if self.selected_spout_source1 >= 0 &&
                                     (self.selected_spout_source1 as usize) < self.spout_sources.len() {
                        self.spout_sources[self.selected_spout_source1 as usize].clone()
                    } else { "Select Spout sender...".to_string() };

                    let mut selected = self.selected_spout_source1;
                    ComboBox::new(ui, "##spout1_select")
                        .preview_value(&preview)
                        .build(|| {
                            for (idx, name) in self.spout_sources.iter().enumerate() {
                                if ui.selectable_config(name).selected(idx == selected as usize).build() {
                                    selected = idx as i32;
                                }
                            }
                        });
                    self.selected_spout_source1 = selected;

                    ui.same_line();
                    if ui.button("Refresh##spout1") {
                        self.refresh_spout_sources();
                    }

                    if ui.button("Start Spout Input 1") && self.selected_spout_source1 >= 0 {
                        let sender_name = self.spout_sources[self.selected_spout_source1 as usize].clone();
                        log::info!("[GUI] Requesting Spout Input 1: {}", sender_name);
                        if let Ok(mut state) = self.shared_state.lock() {
                            state.input1_change_request = InputChangeRequest::StartSpout {
                                input_id: 1,
                                sender_name,
                            };
                        }
                    }
                } else {
                    ui.text_disabled("No Spout senders found");
                    if ui.button("Refresh Spout Senders##1") {
                        self.refresh_spout_sources();
                    }
                }
            }

            // Stop button
            if ui.button("Stop Input 1") {
                if let Ok(mut state) = self.shared_state.lock() {
                    state.input1_change_request = InputChangeRequest::StopInput { input_id: 1 };
                }
            }
        }

        // Input 2 section
        if CollapsingHeader::new("Input 2").default_open(true).build(ui) {
            let input_types = ["None", "Webcam", "NDI", "Syphon", "Spout", "Video File"];
            let mut type_idx = self.input2_type as usize;
            let old_type = type_idx;
            let preview = input_types[type_idx].to_string();
            
            ComboBox::new(ui, "##input2_type")
                .preview_value(&preview)
                .build(|| {
                    for (idx, opt) in input_types.iter().enumerate() {
                        if ui.selectable_config(opt).selected(idx == type_idx).build() {
                            type_idx = idx;
                        }
                    }
                });
            
            let type_changed = type_idx != old_type;
            self.input2_type = match type_idx {
                0 => InputType::None,
                1 => InputType::Webcam,
                2 => InputType::Ndi,
                3 => InputType::Syphon,
                4 => InputType::Spout,
                5 => InputType::VideoFile,
                _ => InputType::None,
            };
            
            // Auto-select first webcam if Webcam is chosen but no device selected
            if type_changed && self.input2_type == InputType::Webcam 
                && self.selected_webcam2 < 0 && !self.webcam_devices.is_empty() {
                self.selected_webcam2 = 0;
            }
            
            // Auto-select first NDI source if NDI is chosen but no source selected
            if type_changed && self.input2_type == InputType::Ndi 
                && self.selected_ndi_source2 < 0 && !self.ndi_sources.is_empty() {
                self.selected_ndi_source2 = 0;
            }
            
            // Auto-select first Syphon source if Syphon is chosen but no source selected
            if type_changed && self.input2_type == InputType::Syphon 
                && self.selected_syphon_source2 < 0 && !self.syphon_sources.is_empty() {
                self.selected_syphon_source2 = 0;
            }
            
            // Refresh NDI sources when switching to NDI
            if type_changed && self.input2_type == InputType::Ndi {
                self.refresh_ndi_sources();
            }
            
            // Refresh Syphon sources when switching to Syphon
            if type_changed && self.input2_type == InputType::Syphon {
                self.refresh_syphon_sources();
            }

            // Auto-select first Spout source if Spout is chosen but no source selected
            if type_changed && self.input2_type == InputType::Spout
                && self.selected_spout_source2 < 0 && !self.spout_sources.is_empty() {
                self.selected_spout_source2 = 0;
            }

            // Refresh Spout sources when switching to Spout
            if type_changed && self.input2_type == InputType::Spout {
                self.refresh_spout_sources();
            }

            // Stop current input and save config when input type changes
            if type_changed {
                log::info!("[GUI] Input 2 type changed from {} to {}, stopping current input", 
                    old_type, type_idx);
                // Stop the current input before switching
                if let Ok(mut state) = self.shared_state.lock() {
                    state.input2_change_request = InputChangeRequest::StopInput { input_id: 2 };
                }
                self.save_input_config();
            }
            
            // Webcam device selection
            if self.input2_type == InputType::Webcam {
                let devices: Vec<&str> = self.webcam_devices.iter().map(|s| s.as_str()).collect();
                if !devices.is_empty() {
                    let preview = if self.selected_webcam2 >= 0 { 
                        self.webcam_devices[self.selected_webcam2 as usize].clone()
                    } else { "Select device...".to_string() };
                    
                    // Check if selected device is a virtual camera
                    let is_virtual = self.selected_webcam2 >= 0 && 
                        self.webcam_devices[self.selected_webcam2 as usize].to_lowercase().contains("virtual");
                    
                    if is_virtual {
                        ui.text_colored([1.0, 0.5, 0.0, 1.0], 
                            "⚠️ Virtual cameras may not work on macOS.\nUse NDI from OBS instead.");
                    }
                    
                    let mut selected = self.selected_webcam2;
                    ComboBox::new(ui, "##webcam2_select")
                        .preview_value(&preview)
                        .build(|| {
                            for (idx, opt) in devices.iter().enumerate() {
                                if ui.selectable_config(opt).selected(idx == selected as usize).build() {
                                    selected = idx as i32;
                                }
                            }
                        });
                    let device_changed = self.selected_webcam2 != selected;
                    self.selected_webcam2 = selected;
                    
                    // Save config when device selection changes
                    if device_changed {
                        self.save_input_config();
                    }
                    
                    if ui.button("Start Webcam 2") && self.selected_webcam2 >= 0 {
                        if let Ok(mut state) = self.shared_state.lock() {
                            state.input2_change_request = InputChangeRequest::StartWebcam {
                                input_id: 2,
                                device_index: self.selected_webcam2 as usize,
                                width: 1280,
                                height: 720,
                                fps: 30,
                            };
                        }
                    }
                } else {
                    ui.text_disabled("No webcam devices found");
                }
            }
            
            // NDI source selection
            if self.input2_type == InputType::Ndi {
                let sources: Vec<&str> = self.ndi_sources.iter().map(|s| s.as_str()).collect();
                if !sources.is_empty() {
                    let preview = if self.selected_ndi_source2 >= 0 { 
                        self.ndi_sources[self.selected_ndi_source2 as usize].clone()
                    } else { "Select NDI source...".to_string() };
                    
                    let mut selected = self.selected_ndi_source2;
                    ComboBox::new(ui, "##ndi2_select")
                        .preview_value(&preview)
                        .build(|| {
                            for (idx, opt) in sources.iter().enumerate() {
                                if ui.selectable_config(opt).selected(idx == selected as usize).build() {
                                    selected = idx as i32;
                                }
                            }
                        });
                    let source_changed = self.selected_ndi_source2 != selected;
                    self.selected_ndi_source2 = selected;
                    
                    // Save config when source selection changes
                    if source_changed {
                        self.save_input_config();
                    }
                    
                    if ui.button("Start NDI Input 2") && self.selected_ndi_source2 >= 0 {
                        let source_name = self.ndi_sources[self.selected_ndi_source2 as usize].clone();
                        if let Ok(mut state) = self.shared_state.lock() {
                            state.input2_change_request = InputChangeRequest::StartNdi {
                                input_id: 2,
                                source_name,
                            };
                        }
                    }
                } else {
                    ui.text_disabled("No NDI sources found");
                    if ui.button("Refresh NDI Sources") {
                        self.refresh_ndi_sources();
                    }
                }
            }
            
            // Syphon source selection (macOS only)
            #[cfg(target_os = "macos")]
            if self.input2_type == InputType::Syphon {
                // Build safe sources list - never pass empty strings to ImGui
                let sources: Vec<(usize, &str)> = self.syphon_sources.iter()
                    .enumerate()
                    .map(|(i, s)| (i, if s.is_empty() { "(unnamed)" } else { s.as_str() }))
                    .collect();
                if !sources.is_empty() {
                    let preview = if self.selected_syphon_source2 >= 0 &&
                                     (self.selected_syphon_source2 as usize) < self.syphon_sources.len() {
                        let name = &self.syphon_sources[self.selected_syphon_source2 as usize];
                        if name.is_empty() { "(unnamed server)".to_string() } else { name.clone() }
                    } else { "Select Syphon source...".to_string() };
                    
                    let mut selected = self.selected_syphon_source2;
                    ComboBox::new(ui, "##syphon2_select")
                        .preview_value(&preview)
                        .build(|| {
                            for (idx, label) in sources.iter() {
                                if ui.selectable_config(label).selected(*idx == selected as usize).build() {
                                    selected = *idx as i32;
                                }
                            }
                        });
                    let source_changed = self.selected_syphon_source2 != selected;
                    self.selected_syphon_source2 = selected;
                    
                    // Save config when source selection changes
                    if source_changed {
                        self.save_input_config();
                    }
                    
                    // Refresh button next to dropdown
                    ui.same_line();
                    if ui.button("🔄") {
                        log::info!("[GUI] Refreshing Syphon sources for Input 2");
                        self.refresh_syphon_sources();
                    }
                    
                    if ui.button("Start Syphon Input 2") && self.selected_syphon_source2 >= 0 {
                        let source_name = self.syphon_sources[self.selected_syphon_source2 as usize].clone();
                        log::info!("[GUI] Requesting Syphon Input 2: {}", source_name);
                        if let Ok(mut state) = self.shared_state.lock() {
                            state.input2_change_request = InputChangeRequest::StartSyphon {
                                input_id: 2,
                                server_name: source_name,
                            };
                        }
                    }
                } else {
                    ui.text_disabled("No Syphon sources found");
                    if ui.button("Refresh Syphon Sources") {
                        self.refresh_syphon_sources();
                    }
                }
            }

            // Spout input source selection (Windows only)
            if self.input2_type == InputType::Spout {
                if !self.spout_sources.is_empty() {
                    let preview = if self.selected_spout_source2 >= 0 &&
                                     (self.selected_spout_source2 as usize) < self.spout_sources.len() {
                        self.spout_sources[self.selected_spout_source2 as usize].clone()
                    } else { "Select Spout sender...".to_string() };

                    let mut selected = self.selected_spout_source2;
                    ComboBox::new(ui, "##spout2_select")
                        .preview_value(&preview)
                        .build(|| {
                            for (idx, name) in self.spout_sources.iter().enumerate() {
                                if ui.selectable_config(name).selected(idx == selected as usize).build() {
                                    selected = idx as i32;
                                }
                            }
                        });
                    self.selected_spout_source2 = selected;

                    ui.same_line();
                    if ui.button("Refresh##spout2") {
                        self.refresh_spout_sources();
                    }

                    if ui.button("Start Spout Input 2") && self.selected_spout_source2 >= 0 {
                        let sender_name = self.spout_sources[self.selected_spout_source2 as usize].clone();
                        log::info!("[GUI] Requesting Spout Input 2: {}", sender_name);
                        if let Ok(mut state) = self.shared_state.lock() {
                            state.input2_change_request = InputChangeRequest::StartSpout {
                                input_id: 2,
                                sender_name,
                            };
                        }
                    }
                } else {
                    ui.text_disabled("No Spout senders found");
                    if ui.button("Refresh Spout Senders##2") {
                        self.refresh_spout_sources();
                    }
                }
            }

            // Stop button
            if ui.button("Stop Input 2") {
                if let Ok(mut state) = self.shared_state.lock() {
                    state.input2_change_request = InputChangeRequest::StopInput { input_id: 2 };
                }
            }
        }
        
        // Device refresh button
        ui.separator();
        if ui.button("Refresh Device List") {
            self.refresh_devices();
        }
        ui.same_line();
        ui.text(format!("Found {} webcam(s), {} NDI source(s)", 
            self.webcam_devices.len(), 
            self.ndi_sources.len()));
        
        // Audio input section
        if CollapsingHeader::new("Audio Input").default_open(true).build(ui) {
            // Audio device selection
            let audio_devices: Vec<&str> = self.audio_devices.iter().map(|s| s.as_str()).collect();
            if !audio_devices.is_empty() {
                let preview = if self.selected_audio_device >= 0 { 
                    self.audio_devices[self.selected_audio_device as usize].clone()
                } else { "Select audio device...".to_string() };
                
                let mut selected = self.selected_audio_device;
                ComboBox::new(ui, "##audio_device_select")
                    .preview_value(&preview)
                    .build(|| {
                        for (idx, opt) in audio_devices.iter().enumerate() {
                            if ui.selectable_config(opt).selected(idx == selected as usize).build() {
                                selected = idx as i32;
                            }
                        }
                    });
                let device_changed = self.selected_audio_device != selected;
                self.selected_audio_device = selected;
                
                // Send change request to engine
                if device_changed && self.selected_audio_device >= 0 {
                    if let Ok(mut state) = self.shared_state.lock() {
                        state.audio_change_request = crate::core::AudioChangeRequest::ChangeDevice {
                            device_index: self.selected_audio_device,
                        };
                    }
                }
            } else {
                ui.text_disabled("No audio devices found");
            }
            
            ui.separator();
            
            // Audio processing controls
            if let Ok(mut state) = self.shared_state.lock() {
                // Amplitude slider (0-1x, unity gain at 1.0)
                let mut amplitude = state.audio.amplitude;
                if imgui::Drag::new("Amplitude##audio")
                    .range(0.0, 1.0)
                    .speed(0.01)
                    .build(ui, &mut amplitude) 
                {
                    state.audio.amplitude = amplitude.clamp(0.0, 1.0);
                }
                
                // Smoothing slider (0-0.99)
                let mut smoothing = state.audio.smoothing;
                if imgui::Drag::new("Smoothing##audio")
                    .range(0.0, 0.99)
                    .speed(0.01)
                    .build(ui, &mut smoothing)
                {
                    state.audio.smoothing = smoothing.clamp(0.0, 0.99);
                }
                
                // Normalization toggle
                let mut normalization = state.audio.normalization;
                if ui.checkbox("Normalization##audio", &mut normalization) {
                    state.audio.normalization = normalization;
                }
                ui.same_line();
                ui.text_disabled("(scales to min/max range)");
                
                // Pink noise compensation toggle
                let mut pink_comp = state.audio.pink_compensation;
                if ui.checkbox("Pink Noise Comp##audio", &mut pink_comp) {
                    state.audio.pink_compensation = pink_comp;
                }
                ui.same_line();
                ui.text_disabled("(flat response for pink noise)");
            }
            
            ui.separator();
            
            // Audio status display
            if let Ok(state) = self.shared_state.lock() {
                ui.text(format!("Volume: {:.3}", state.audio.volume));
                ui.text(format!("BPM: {:.1}", state.audio.bpm));
                
                // FFT visualization
                ui.text("FFT Bands:");
                for (i, val) in state.audio.fft.iter().enumerate().take(8) {
                    let bar_width = 200.0 * val.min(1.0);
                    ui.text(format!("Band {}: ", i));
                    ui.same_line();
                    let draw_list = ui.get_window_draw_list();
                    let pos = ui.cursor_screen_pos();
                    draw_list.add_rect(
                        [pos[0], pos[1]],
                        [pos[0] + bar_width, pos[1] + 10.0],
                        [0.0, 1.0, 0.0, 1.0],
                    ).filled(true).build();
                    ui.new_line();
                }
            }
        }
    }
    
    /// Build Settings panel
    fn build_settings_panel(&mut self, ui: &Ui) {
        // Performance stats at the top — show actual output window FPS from engine
        let output_fps = self.shared_state.lock().map(|s| s.output_actual_fps).unwrap_or(0.0);
        ui.text(format!("Output FPS: {:.1}", output_fps));
        ui.text(format!("GUI FPS: {:.1}", self.main_fps));
        ui.separator();
        
        // Output mode selection
        if CollapsingHeader::new("Output Mode").default_open(true).build(ui) {
            if let Ok(mut state) = self.shared_state.lock() {
                let mut selected_mode = match state.output_mode {
                    OutputMode::Block1 => 0,
                    OutputMode::Block2 => 1,
                    OutputMode::Block3 => 2,
                    OutputMode::PreviewInput1 => 3,
                    OutputMode::PreviewInput2 => 4,
                };
                let old_mode = selected_mode;
                
                ui.radio_button("Block 1##out", &mut selected_mode, 0);
                ui.radio_button("Block 2##out", &mut selected_mode, 1);
                ui.radio_button("Block 3##out", &mut selected_mode, 2);
                ui.radio_button("Preview Input 1##out", &mut selected_mode, 3);
                ui.radio_button("Preview Input 2##out", &mut selected_mode, 4);
                
                if selected_mode != old_mode {
                    state.output_mode = match selected_mode {
                        0 => OutputMode::Block1,
                        1 => OutputMode::Block2,
                        2 => OutputMode::Block3,
                        3 => OutputMode::PreviewInput1,
                        4 => OutputMode::PreviewInput2,
                        _ => OutputMode::Block3,
                    };
                }
            }
        }
        
        // Display info
        if CollapsingHeader::new("Display Info").default_open(true).build(ui) {
            if let Ok(state) = self.shared_state.lock() {
                ui.text(format!("Output Size: {}x{}", state.output_size.0, state.output_size.1));
                ui.text(format!("Internal Size: {}x{}", state.internal_size.0, state.internal_size.1));
                ui.text(format!("Frame Count: {}", state.frame_count));
            }
        }
        
        // Output Frame Rate Settings
        if CollapsingHeader::new("Output Frame Rate").default_open(true).build(ui) {
            // Read current values from shared state
            let (current_vsync, current_fps) = if let Ok(state) = self.shared_state.lock() {
                (state.output_vsync, state.output_fps)
            } else {
                (self.config.output_window.vsync, self.config.output_window.fps)
            };
            
            // VSync toggle
            let mut vsync = current_vsync;
            if ui.checkbox("VSync", &mut vsync) && vsync != current_vsync {
                self.send_input_request(crate::core::InputChangeRequest::SetVsync(vsync));
            }
            if ui.is_item_hovered() {
                ui.tooltip_text("Enable VSync to match display refresh rate (reduces tearing)");
            }
            
            ui.separator();
            
            // Common framerate presets
            let fps_presets = [24u32, 30, 60, 120, 144];
            let fps_labels = ["24 FPS (Cinematic)", "30 FPS", "60 FPS", "120 FPS", "144 FPS"];
            let custom_label = "Custom...";
            
            // Find if current FPS matches a preset
            let preset_idx = fps_presets.iter().position(|&fps| fps == current_fps);
            
            let preview = match preset_idx {
                Some(idx) => fps_labels[idx].to_string(),
                None => format!("{} FPS (Custom)", current_fps),
            };
            
            ui.text("Target Frame Rate:");
            ComboBox::new(ui, "##fps_combo")
                .preview_value(&preview)
                .build(|| {
                    // Preset options
                    for (idx, label) in fps_labels.iter().enumerate() {
                        let is_selected = preset_idx == Some(idx);
                        if ui.selectable_config(label).selected(is_selected).build() {
                            self.send_input_request(crate::core::InputChangeRequest::SetOutputFps(fps_presets[idx]));
                        }
                    }
                    // Custom option
                    ui.separator();
                    let is_custom = preset_idx.is_none();
                    if ui.selectable_config(custom_label).selected(is_custom).build() {
                        // Keep current custom value selected
                    }
                });
            
            // Custom FPS input (shown when custom or always available)
            let mut custom_fps = current_fps as i32;
            ui.text("Custom FPS:");
            if Drag::new("##custom_fps").speed(1.0).range(1, 240).build(ui, &mut custom_fps) {
                if custom_fps != current_fps as i32 {
                    self.send_input_request(crate::core::InputChangeRequest::SetOutputFps(custom_fps as u32));
                }
            }
            
            ui.text_disabled(format!("Current: {} FPS", current_fps));
            if !current_vsync {
                ui.text_disabled("Frame rate limiting is active (VSync off)");
            } else {
                ui.text_disabled("Frame rate controlled by VSync");
            }
        }
        
        // Resolution Configuration
        if CollapsingHeader::new("Resolution Settings").default_open(true).build(ui) {
            self.build_resolution_panel(ui);
        }
        
        // UI Scale - Match oF version with discrete presets
        if CollapsingHeader::new("UI Scale").default_open(true).build(ui) {
            ui.text("Adjust UI scale for better visibility on high-DPI displays:");
            
            // oF-style scale presets: 100%, 150%, 200%, 250%, 300%
            let scale_presets = [1.0f32, 1.5, 2.0, 2.5, 3.0];
            let scale_labels = ["100%", "150%", "200%", "250%", "300%"];
            
            // Find current preset index (closest match)
            let current_scale = if self.config.ui_scale.is_finite() {
                self.config.ui_scale
            } else {
                2.0
            };
            let mut selected_idx = scale_presets.iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    let diff_a = (**a - current_scale).abs();
                    let diff_b = (**b - current_scale).abs();
                    diff_a.total_cmp(&diff_b)
                })
                .map(|(i, _)| i)
                .unwrap_or(2); // Default to 200%
            
            let preview = scale_labels[selected_idx];
            ComboBox::new(ui, "##ui_scale_combo")
                .preview_value(preview)
                .build(|| {
                    for (idx, label) in scale_labels.iter().enumerate() {
                        if ui.selectable_config(label).selected(idx == selected_idx).build() {
                            selected_idx = idx;
                        }
                    }
                });
            
            // Apply new scale if changed
            let new_scale = scale_presets[selected_idx];
            if new_scale != current_scale {
                self.config.ui_scale = new_scale;
                // Save config immediately
                if let Err(e) = self.config.save() {
                    log::warn!("Failed to save config: {}", e);
                }
                // Update shared state so engine can apply it at runtime
                if let Ok(mut state) = self.shared_state.lock() {
                    state.ui_scale = new_scale;
                }
            }
            
            ui.text_disabled("Scale changes apply immediately");
        }
        
        // OSC Address Display Toggle
        if CollapsingHeader::new("OSC Control").default_open(true).build(ui) {
            let mut show_osc = self.config.show_osc_addresses;
            if ui.checkbox("Show OSC addresses on hover", &mut show_osc) {
                self.config.show_osc_addresses = show_osc;
                if let Err(e) = self.config.save() {
                    log::warn!("Failed to save config: {}", e);
                }
            }
            if ui.is_item_hovered() {
                ui.tooltip_text("When enabled, hover over any parameter to see its OSC address for remote control");
            }

            ui.separator();

            // Receive port
            ui.text("Receive Port:");
            ui.same_line();
            let mut recv_port = self.config.control.osc_receive_port as i32;
            ui.set_next_item_width(100.0);
            if ui.input_int("##osc_recv_port", &mut recv_port).build() {
                let new_port = recv_port.clamp(1024, 65535) as u16;
                if new_port != self.config.control.osc_receive_port {
                    self.config.control.osc_receive_port = new_port;
                    if let Err(e) = self.config.save() {
                        log::warn!("Failed to save config: {}", e);
                    }
                }
            }

            // Send port
            ui.text("Send Port:");
            ui.same_line();
            let mut send_port = self.config.control.osc_send_port as i32;
            ui.set_next_item_width(100.0);
            if ui.input_int("##osc_send_port", &mut send_port).build() {
                let new_port = send_port.clamp(1024, 65535) as u16;
                if new_port != self.config.control.osc_send_port {
                    self.config.control.osc_send_port = new_port;
                    if let Err(e) = self.config.save() {
                        log::warn!("Failed to save config: {}", e);
                    }
                }
            }

            // Send IP
            ui.text("Send IP:");
            ui.same_line();
            let mut send_ip = self.config.control.osc_send_ip.clone();
            ui.set_next_item_width(150.0);
            if ui.input_text("##osc_send_ip", &mut send_ip).build() {
                if send_ip != self.config.control.osc_send_ip {
                    self.config.control.osc_send_ip = send_ip;
                    if let Err(e) = self.config.save() {
                        log::warn!("Failed to save config: {}", e);
                    }
                }
            }

            ui.text_disabled("Changes take effect on next launch");
        }
        
        // Preview Window Settings
        if CollapsingHeader::new("Preview & Color Picker").default_open(true).build(ui) {
            ui.text("The Preview window allows you to:");
            ui.bullet_text("View output from any Block or Input");
            ui.bullet_text("Sample colors for keying");
            ui.bullet_text("Copy color values to clipboard");
            
            ui.separator();
            
            if ui.button("Open Preview Window") {
                self.show_preview_window = true;
                // Re-enable preview computation
                if let Ok(mut state) = self.shared_state.lock() {
                    state.preview_enabled = true;
                }
            }
            
            ui.text_disabled("Note: Color sampling requires GPU readback (not yet implemented)");
        }
        
        // NDI Output Settings
        if CollapsingHeader::new("NDI Output").default_open(true).build(ui) {
            // Read NDI output status
            let is_active = self.shared_state.lock()
                .map(|s| s.ndi_output_active)
                .unwrap_or(false);
            
            // Status indicator
            if is_active {
                ui.text_colored([0.0, 1.0, 0.0, 1.0], "● Streaming");
            } else {
                ui.text("○ Not streaming");
            }
            
            ui.separator();
            
            // NDI output name
            ui.text("Output Name:");
            let mut ndi_name = self.config.ndi.output_name.clone();
            ui.input_text("##ndi_name", &mut ndi_name)
                .build();
            if ndi_name != self.config.ndi.output_name {
                self.config.ndi.output_name = ndi_name;
            }
            
            // Alpha channel option
            let mut include_alpha = self.config.ndi.output_alpha;
            ui.checkbox("Include Alpha Channel", &mut include_alpha);
            if include_alpha != self.config.ndi.output_alpha {
                self.config.ndi.output_alpha = include_alpha;
            }
            
            // Frame skip option (for performance)
            ui.text("Frame Skip (performance):");
            let frame_skip_options = ["1 (60fps)", "2 (30fps)", "3 (20fps)", "4 (15fps)", "6 (10fps)"];
            let current_skip = self.config.ndi.frame_skip.clamp(1, 6) as usize;
            let skip_idx = match current_skip {
                1 => 0,
                2 => 1,
                3 => 2,
                4 => 3,
                5 | 6 => 4,
                _ => 1, // Default to 2 (30fps)
            };
            let preview = frame_skip_options[skip_idx];
            
            ComboBox::new(ui, "##ndi_frame_skip")
                .preview_value(preview)
                .build(|| {
                    for (idx, label) in frame_skip_options.iter().enumerate() {
                        if ui.selectable_config(label).selected(idx == skip_idx).build() {
                            let new_skip = match idx {
                                0 => 1,
                                1 => 2,
                                2 => 3,
                                3 => 4,
                                4 => 6,
                                _ => 2,
                            };
                            self.config.ndi.frame_skip = new_skip;
                        }
                    }
                });
            
            if ui.is_item_hovered() {
                ui.tooltip_text("Higher frame skip = better performance but lower NDI output framerate");
            }
            
            ui.separator();
            
            // Start/Stop button
            if is_active {
                if ui.button("Stop NDI Output") {
                    if let Ok(mut state) = self.shared_state.lock() {
                        state.output_command = crate::core::OutputCommand::StopNdi;
                    }
                }
            } else {
                if ui.button("Start NDI Output") {
                    let name = self.config.ndi.output_name.clone();
                    let alpha = self.config.ndi.output_alpha;
                    let skip = self.config.ndi.frame_skip.max(1);
                    if let Ok(mut state) = self.shared_state.lock() {
                        state.output_command = crate::core::OutputCommand::StartNdi {
                            name,
                            include_alpha: alpha,
                            frame_skip: skip,
                        };
                    }
                }
            }
            
            ui.text_disabled("NDI output streams the final output at display resolution");
        }
        
        // Syphon Output Settings (macOS only)
        #[cfg(target_os = "macos")]
        if CollapsingHeader::new("Syphon Output (macOS)").default_open(true).build(ui) {
            // Read Syphon output status
            let is_active = self.shared_state.lock()
                .map(|s| s.syphon_output_active)
                .unwrap_or(false);
            
            // Check if Syphon is available (requires syphon feature)
            #[cfg(all(target_os = "macos", feature = "syphon"))]
            let available = crate::output::SyphonSender::is_syphon_available();
            #[cfg(not(all(target_os = "macos", feature = "syphon")))]
            let available = false;
            
            if !available {
                ui.text_colored([1.0, 0.5, 0.0, 1.0], "⚠ Syphon.framework not available");
                ui.text_disabled("Install to: /Library/Frameworks/Syphon.framework");
                ui.text_disabled("Download from: github.com/Syphon/Syphon-Framework");
                if ui.button("Check Again") {
                    // Force recheck - reload the library
                    log::info!("[GUI] Rechecking Syphon availability...");
                }
            } else {
                // Status indicator
                if is_active {
                    ui.text_colored([0.0, 1.0, 0.0, 1.0], "● Streaming");
                } else {
                    ui.text("○ Not streaming");
                }
                
                ui.separator();
                
                // Syphon server name
                ui.text("Server Name:");
                let mut server_name = self.config.syphon.server_name.clone();
                ui.input_text("##syphon_name", &mut server_name)
                    .build();
                if server_name != self.config.syphon.server_name {
                    self.config.syphon.server_name = server_name;
                }
                
                ui.separator();
                
                // Start/Stop button
                if is_active {
                    if ui.button("Stop Syphon Output") {
                        if let Ok(mut state) = self.shared_state.lock() {
                            state.output_command = crate::core::OutputCommand::StopSyphon;
                        }
                    }
                } else {
                    if ui.button("Start Syphon Output") {
                        let name = self.config.syphon.server_name.clone();
                        if let Ok(mut state) = self.shared_state.lock() {
                            state.output_command = crate::core::OutputCommand::StartSyphon { name };
                        }
                    }
                }
                
                ui.text_disabled("Syphon output streams to other macOS apps (Resolume, MadMapper, etc.)");
            }
        }
        
        // Spout Output Settings (Windows only)
        #[cfg(target_os = "windows")]
        if CollapsingHeader::new("Spout Output (Windows)").default_open(true).build(ui) {
            let is_active = self.shared_state.lock()
                .map(|s| s.spout_output_active)
                .unwrap_or(false);

            // Status indicator
            if is_active {
                ui.text_colored([0.0, 1.0, 0.0, 1.0], "● Streaming");
            } else {
                ui.text("○ Not streaming");
            }

            ui.separator();

            // Spout sender name
            ui.text("Sender Name:");
            let mut sender_name = self.spout_output_name.clone();
            ui.input_text("##spout_output_name", &mut sender_name)
                .build();
            if sender_name != self.spout_output_name {
                self.spout_output_name = sender_name;
            }

            ui.separator();

            // Start/Stop button
            if is_active {
                if ui.button("Stop Spout Output") {
                    if let Ok(mut state) = self.shared_state.lock() {
                        state.output_command = crate::core::OutputCommand::StopSpout;
                    }
                }
            } else {
                if ui.button("Start Spout Output") {
                    let name = self.spout_output_name.clone();
                    if let Ok(mut state) = self.shared_state.lock() {
                        state.output_command = crate::core::OutputCommand::StartSpout { name };
                    }
                }
            }

            ui.text_disabled("Spout output streams to other Windows apps (Resolume, OBS, etc.)");
        }

        // Clear feedback button
        if ui.button("Clear Feedback") {
            if let Ok(mut state) = self.shared_state.lock() {
                state.clear_feedback = true;
            }
        }
        
        // Recording Section
        if CollapsingHeader::new("Recording (Shift+R)").default_open(true).build(ui) {
            // Read recording state
            let is_recording = self.shared_state.lock()
                .map(|s| s.is_recording)
                .unwrap_or(false);
            
            // Recording status with red indicator
            if is_recording {
                ui.text_colored([1.0, 0.0, 0.0, 1.0], "● REC");
            } else {
                ui.text("○ Ready");
            }
            
            // Start/Stop button
            let button_label = if is_recording { "Stop Recording" } else { "Start Recording" };
            if ui.button(button_label) {
                if let Ok(mut state) = self.shared_state.lock() {
                    state.recording_command = crate::core::RecordingCommand::Toggle;
                }
            }
            ui.same_line();
            ui.text_disabled("or press Shift+R");
            
            ui.separator();
            
            // Read current settings
            let (codec, quality, include_audio, filename) = self.shared_state.lock()
                .map(|s| (
                    s.recording_settings.codec,
                    s.recording_settings.quality,
                    s.recording_settings.include_audio,
                    s.recording_settings.filename.clone(),
                ))
                .unwrap_or_default();
            
            // Codec selection
            let codec_names = ["H.264 (AVC)", "H.265 (HEVC)", "ProRes", "VP9", "AV1"];
            let mut codec_idx = codec as usize;
            let codec_preview = codec_names[codec_idx.min(4)];
            
            ComboBox::new(ui, "Codec##rec")
                .preview_value(codec_preview)
                .build(|| {
                    for (idx, name) in codec_names.iter().enumerate() {
                        if ui.selectable_config(name).selected(idx == codec_idx).build() {
                            codec_idx = idx;
                        }
                    }
                });
            
            if codec_idx != codec as usize {
                if let Ok(mut state) = self.shared_state.lock() {
                    state.recording_settings.codec = match codec_idx {
                        0 => crate::core::VideoCodec::H264,
                        1 => crate::core::VideoCodec::H265,
                        2 => crate::core::VideoCodec::ProRes,
                        3 => crate::core::VideoCodec::VP9,
                        4 => crate::core::VideoCodec::AV1,
                        _ => crate::core::VideoCodec::H264,
                    };
                }
            }
            
            // Quality selection
            let quality_names = ["Lossless", "High", "Medium", "Low"];
            let mut quality_idx = quality as usize;
            let quality_preview = quality_names[quality_idx.min(3)];
            
            ComboBox::new(ui, "Quality##rec")
                .preview_value(quality_preview)
                .build(|| {
                    for (idx, name) in quality_names.iter().enumerate() {
                        if ui.selectable_config(name).selected(idx == quality_idx).build() {
                            quality_idx = idx;
                        }
                    }
                });
            
            if quality_idx != quality as usize {
                if let Ok(mut state) = self.shared_state.lock() {
                    state.recording_settings.quality = match quality_idx {
                        0 => crate::core::RecordingQuality::Lossless,
                        1 => crate::core::RecordingQuality::High,
                        2 => crate::core::RecordingQuality::Medium,
                        3 => crate::core::RecordingQuality::Low,
                        _ => crate::core::RecordingQuality::High,
                    };
                }
            }
            
            // Include audio toggle (disabled - not yet implemented)
            let mut include_audio_mut = include_audio;
            ui.checkbox("Include Audio (coming soon)", &mut include_audio_mut);
            ui.text_disabled("Audio recording will be added in a future update");
            
            // Filename input
            ui.text("Filename:");
            let mut filename_mut = filename.clone();
            imgui::InputText::new(ui, "##rec_filename", &mut filename_mut)
                .hint("output")
                .build();
            if filename_mut != filename {
                if let Ok(mut state) = self.shared_state.lock() {
                    state.recording_settings.filename = filename_mut;
                }
            }
            ui.same_line();
            ui.text(".mp4");
        }
        
        // Window Layout Management
        if CollapsingHeader::new("Window Layout").default_open(true).build(ui) {
            ui.text("Popped-out tabs: Click 'Pop Out' from tab context menu");
            ui.text("Indicator ⧉ shows which tabs are floating");
            
            if ui.button("Save Layout") {
                if let Err(e) = self.layout_manager.auto_save() {
                    self.show_status(&format!("Failed to save layout: {}", e));
                } else {
                    self.show_status("Layout saved!");
                }
            }
            ui.same_line();
            if ui.button("Reset Layout") {
                *self.layout_manager.current_mut() = crate::config::LayoutConfig::default();
                if let Err(e) = self.layout_manager.auto_save() {
                    self.show_status(&format!("Failed to reset layout: {}", e));
                } else {
                    self.show_status("Layout reset!");
                }
            }
            
            // Show currently popped tabs
            if !self.layout_manager.current().popped_tabs.is_empty() {
                ui.text("Currently floating:");
                for tab_id in self.layout_manager.current().popped_tabs.keys() {
                    ui.bullet_text(tab_id.display_name());
                }
            }
        }
    }
    
    /// Build comprehensive presets management panel
    fn build_presets_panel(&mut self, ui: &Ui) {
        use imgui::CollapsingHeader;
        
        // === SECTION 1: Quick Actions ===
        if CollapsingHeader::new("Quick Actions").default_open(true).build(ui) {
            ui.spacing();
            
            // Quick Save button
            if ui.button("💾 Quick Save") {
                // Use simple timestamp without chrono
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                let name = format!("Preset_{}", now);
                self.preset_name_input = name.clone();
                // Trigger save immediately
                let _ = self.save_current_preset(&name);
            }
            if ui.is_item_hovered() {
                ui.tooltip_text("Save current state to current bank with timestamp");
            }
            
            ui.same_line();
            
            // Refresh button
            if ui.button("🔄 Refresh") {
                self.preset_manager.scan_banks();
                self.show_status("Banks refreshed");
            }
            
            // Recently used presets (would need tracking - simplified for now)
            ui.spacing();
            ui.text_disabled("Recently saved presets appear in the bank below");
        }
        
        // === SECTION 2: Bank/Folder Management ===
        if CollapsingHeader::new("Banks (Folders)").default_open(true).build(ui) {
            ui.spacing();
            
            // Current bank selector
            let banks = self.preset_manager.get_bank_names();
            let current_bank = self.preset_manager.get_current_bank().to_string();
            
            ui.text("Current Bank:");
            let mut bank_idx = banks.iter().position(|b| b == &current_bank).unwrap_or(0);
            let bank_preview = &banks[bank_idx];
            
            ComboBox::new(ui, "##bank_selector")
                .preview_value(bank_preview)
                .build(|| {
                    for (idx, bank) in banks.iter().enumerate() {
                        let is_selected = idx == bank_idx;
                        if ui.selectable_config(bank).selected(is_selected).build() {
                            bank_idx = idx;
                        }
                    }
                });
            
            if banks.get(bank_idx) != Some(&current_bank) {
                let new_bank = banks[bank_idx].clone();
                self.preset_manager.switch_bank(&new_bank);
                self.selected_bank = new_bank;
                self.selected_preset_index = -1;
                self.show_status(&format!("Switched to bank: {}", &banks[bank_idx]));
            }
            
            ui.spacing();
            
            // Create new bank
            ui.text("Create New Bank:");
            imgui::InputText::new(ui, "##new_bank_name", &mut self.new_bank_name_input)
                .hint("Enter folder name...")
                .build();
            ui.same_line();
            if ui.button("Create Folder") {
                if !self.new_bank_name_input.is_empty() {
                    if self.preset_manager.create_bank(&self.new_bank_name_input) {
                        self.preset_manager.switch_bank(&self.new_bank_name_input);
                        self.selected_bank = self.new_bank_name_input.clone();
                        self.show_status(&format!("Created and switched to bank: {}", self.new_bank_name_input));
                        self.new_bank_name_input.clear();
                    } else {
                        self.show_status(&format!("Failed to create bank '{}' (may already exist)", self.new_bank_name_input));
                    }
                }
            }
            
            ui.spacing();
            ui.text_disabled(format!("Available banks: {}", banks.len()));
            for bank in &banks {
                ui.bullet_text(bank);
            }
        }
        
        // === SECTION 3: Preset Browser ===
        if CollapsingHeader::new("Preset Browser").default_open(true).build(ui) {
            ui.spacing();
            
            // Filter/search
            ui.text("Filter:");
            imgui::InputText::new(ui, "##preset_filter", &mut self.preset_filter_text)
                .hint("Search presets...")
                .build();
            
            ui.spacing();
            
            // Get presets with optional filtering - preserve original indices
            let preset_names = self.preset_manager.get_preset_names();
            let filtered_presets: Vec<(usize, &String)> = preset_names.iter()
                .enumerate()
                .filter(|(_, name)| {
                    if self.preset_filter_text.is_empty() {
                        true
                    } else {
                        name.to_lowercase().contains(&self.preset_filter_text.to_lowercase())
                    }
                })
                .collect();
            
            if filtered_presets.is_empty() {
                ui.text_disabled("No presets found");
            } else {
                ui.text(&format!("Presets in {} ({}):", self.preset_manager.get_current_bank(), filtered_presets.len()));
                
                // List presets with separators
                for (display_idx, (original_idx, preset_name)) in filtered_presets.iter().enumerate() {
                    ui.separator();
                    
                    // Load button
                    let load_label = format!("Load##load_{}", display_idx);
                    if ui.button(&load_label) {
                        match self.preset_manager.load_preset(preset_name) {
                            Ok(data) => {
                                // Apply loaded preset
                                if let Ok(mut state) = self.shared_state.lock() {
                                    state.block1 = data.block1;
                                    state.block2 = data.block2;
                                    state.block3 = data.block3;
                                    
                                    // Restore LFO banks
                                    for (idx, lfo_bank) in data.lfo_banks.iter().enumerate() {
                                        if idx < state.lfo_banks.len() {
                                            state.lfo_banks[idx] = *lfo_bank;
                                        }
                                    }
                                    
                                    // Restore MIDI mappings
                                    state.midi.mappings.clear();
                                    for mapping in &data.midi_mappings {
                                        state.midi.mappings.insert(mapping.param_id.clone(), mapping.clone());
                                    }
                                }
                                self.block1_edit = data.block1;
                                self.block2_edit = data.block2;
                                self.block3_edit = data.block3;
                                
                                self.show_status(&format!("Loaded preset: {}", preset_name));
                            }
                            Err(e) => {
                                self.show_status(&format!("Failed to load: {}", e));
                            }
                        }
                    }
                    
                    ui.same_line();
                    
                    // Delete button
                    let delete_label = format!("Del##del_{}", display_idx);
                    if ui.button(&delete_label) {
                        if let Err(e) = self.preset_manager.delete_preset(*original_idx) {
                            self.show_status(&format!("Failed to delete: {}", e));
                        } else {
                            self.show_status(&format!("Deleted preset: {}", preset_name));
                        }
                    }
                    
                    ui.same_line();
                    
                    // Preset name
                    ui.text(preset_name);
                }
            }
        }
        
        // === SECTION 4: Save New Preset ===
        if CollapsingHeader::new("Save New Preset").default_open(true).build(ui) {
            ui.spacing();
            
            ui.text("Preset Name:");
            imgui::InputText::new(ui, "##preset_name_input", &mut self.preset_name_input)
                .hint("Enter preset name...")
                .build();
            
            ui.text("Description (optional):");
            imgui::InputText::new(ui, "##preset_desc", &mut self.preset_description_input)
                .hint("Enter description...")
                .build();
            
            ui.spacing();
            
            if ui.button("💾 Save Preset") {
                if !self.preset_name_input.is_empty() {
                    match self.save_current_preset(&self.preset_name_input.clone()) {
                        Ok(_) => {
                            self.show_status(&format!("Saved preset: {}", self.preset_name_input));
                            self.preset_name_input.clear();
                            self.preset_description_input.clear();
                        }
                        Err(e) => {
                            self.show_status(&format!("Failed to save: {}", e));
                        }
                    }
                }
            }
        }
        
        // === SECTION 5: Layout Management ===
        if CollapsingHeader::new("Layout Management").default_open(true).build(ui) {
            ui.spacing();
            
            // Save new layout
            ui.text("Save Current Layout:");
            imgui::InputText::new(ui, "##layout_name", &mut self.layout_name_input)
                .hint("Enter layout name...")
                .build();
            ui.same_line();
            if ui.button("💾 Save") && !self.layout_name_input.is_empty() {
                match self.layout_manager.save_named(&self.layout_name_input) {
                    Ok(_) => {
                        self.show_status(&format!("Saved layout: {}", self.layout_name_input));
                        self.layout_manager.set_selected(self.layout_name_input.clone());
                    }
                    Err(e) => self.show_status(&format!("Failed to save layout: {}", e)),
                }
            }
            
            ui.spacing();
            ui.separator();
            ui.spacing();
            
            // Load/Delete existing layouts
            ui.text("Saved Layouts:");
            let layouts = self.layout_manager.list_layouts();
            
            if layouts.is_empty() {
                ui.text_disabled("No saved layouts yet");
            } else {
                for (name, created_at) in &layouts {
                    ui.separator();
                    
                    // Load button
                    let load_label = format!("Load##layout_{}", name);
                    if ui.button(&load_label) {
                        match self.layout_manager.load_named(name) {
                            Ok(_) => {
                                self.show_status(&format!("Loaded layout: {}", name));
                            }
                            Err(e) => self.show_status(&format!("Failed to load: {}", e)),
                        }
                    }
                    
                    ui.same_line();
                    
                    // Delete button
                    let delete_label = format!("Del##layout_{}", name);
                    if ui.button(&delete_label) {
                        match self.layout_manager.delete_named(name) {
                            Ok(_) => self.show_status(&format!("Deleted layout: {}", name)),
                            Err(e) => self.show_status(&format!("Failed to delete: {}", e)),
                        }
                    }
                    
                    ui.same_line();
                    
                    // Layout name with indicator if currently selected
                    let selected_marker = if self.layout_manager.selected() == name {
                        "● "
                    } else {
                        "  "
                    };
                    ui.text(format!("{}{}", selected_marker, name));
                    
                    // Show creation date
                    ui.same_line_with_pos(300.0);
                    let datetime = Self::format_timestamp(*created_at);
                    ui.text_disabled(format!("({})", datetime));
                }
            }
            
            ui.spacing();
            ui.separator();
            ui.spacing();
            
            // Quick actions
            if ui.button("🔄 Reset to Default") {
                *self.layout_manager.current_mut() = crate::config::LayoutConfig::default();
                if let Err(e) = self.layout_manager.auto_save() {
                    self.show_status(&format!("Failed to reset: {}", e));
                } else {
                    self.show_status("Layout reset to default!");
                }
            }
            
            ui.spacing();
            
            // Show currently popped tabs
            if !self.layout_manager.current().popped_tabs.is_empty() {
                ui.text("Currently floating tabs:");
                for tab_id in self.layout_manager.current().popped_tabs.keys() {
                    ui.bullet_text(tab_id.display_name());
                }
            } else {
                ui.text_disabled("No floating tabs - right-click any tab to pop out");
            }
        }
        
        // === SECTION 6: Import/Export ===
        if CollapsingHeader::new("Import / Export").default_open(false).build(ui) {
            ui.spacing();
            
            if ui.button("📥 Import OF Presets") {
                // Trigger OF preset import
                if let Ok(mut state) = self.shared_state.lock() {
                    // Use input2_change_request as a signal for OF import
                    // This is a hack - in a real implementation we'd add a dedicated request type
                    log::info!("OF preset import requested from Presets tab");
                }
                self.show_status("Use Inputs tab to import OF presets");
            }
            if ui.is_item_hovered() {
                ui.tooltip_text("Go to Inputs tab → Import OF Dir button");
            }
            
            ui.spacing();
            
            ui.text_disabled("Backup/Restore: Copy the 'presets' folder manually");
            ui.text_disabled("Location: ./presets/");
        }
    }
    
    /// Helper to save current preset
    fn save_current_preset(&mut self, name: &str) -> anyhow::Result<()> {
        use crate::params::preset::{PresetAudioSettings, PresetData, PresetTempoData};
        
        let (block1, block2, block3, midi_mappings, lfo_banks) = {
            let state = self.shared_state.lock().unwrap();
            let mappings: Vec<crate::midi::MidiMapping> = state.midi.mappings.values().cloned().collect();
            let lfos = state.lfo_banks.clone();
            (state.block1, state.block2, state.block3, mappings, lfos)
        };
        
        let audio_settings = PresetAudioSettings {
            amplitude: 1.0,
            smoothing: 0.7,
            normalization: false,
            pink_compensation: false,
        };
        
        let tempo = PresetTempoData {
            bpm: self.bpm,
            enabled: self.bpm_enabled,
        };
        
        let preset_data = PresetData {
            block1,
            block2,
            block3,
            block1_modulations: HashMap::new(),
            block2_modulations: HashMap::new(),
            block3_modulations: HashMap::new(),
            audio: audio_settings,
            tempo,
            lfo_banks,
            midi_mappings,
            version: env!("CARGO_PKG_VERSION").to_string(),
            name: name.to_string(),
        };
        
        self.preset_manager.save_preset(name, &preset_data)?;
        Ok(())
    }
    
    /// Build resolution configuration panel
    fn build_resolution_panel(&mut self, ui: &Ui) {
        use crate::config::ResolutionPreset;
        
        let presets = ResolutionPreset::all();
        
        // Helper to handle resolution dropdown
        let mut handle_res_dropdown = |ui: &Ui, current_preset: ResolutionPreset, id: &str| -> ResolutionPreset {
            let mut selected_idx = presets.iter().position(|&p| p == current_preset).unwrap_or(0);
            let preview = presets[selected_idx].name();
            
            ComboBox::new(ui, &format!("##res_{}", id))
                .preview_value(preview)
                .build(|| {
                    for (idx, preset) in presets.iter().enumerate() {
                        if ui.selectable_config(preset.name()).selected(idx == selected_idx).build() {
                            selected_idx = idx;
                        }
                    }
                });
            
            presets[selected_idx]
        };
        
        // Input Resolution
        ui.text("Input Resolution:");
        let input_preset = self.config.resolution.input.preset;
        let new_input_preset = handle_res_dropdown(ui, input_preset, "input");
        
        if !new_input_preset.is_custom() {
            if let Some((w, h)) = new_input_preset.dimensions() {
                ui.same_line();
                ui.text_disabled(&format!("({}x{})", w, h));
            }
        }
        
        if new_input_preset != input_preset {
            self.config.resolution.input.set_preset(new_input_preset);
            let _ = self.config.save();
        }
        
        if self.config.resolution.input.preset.is_custom() {
            ui.indent();
            let mut w = self.config.resolution.input.custom_width;
            let mut h = self.config.resolution.input.custom_height;
            ui.text("Custom:");
            ui.same_line();
            ui.set_next_item_width(80.0);
            if Drag::new("Width##input").range(1, 7680).speed(1.0).build(ui, &mut w) {
                self.config.resolution.input.custom_width = w;
                let _ = self.config.save();
            }
            ui.same_line();
            ui.text("x");
            ui.same_line();
            ui.set_next_item_width(80.0);
            if Drag::new("Height##input").range(1, 4320).speed(1.0).build(ui, &mut h) {
                self.config.resolution.input.custom_height = h;
                let _ = self.config.save();
            }
            ui.unindent();
        }
        ui.text_disabled("Resolution for camera/video inputs");
        
        ui.separator();
        
        // Internal Resolution
        ui.text("Internal Resolution:");
        let internal_preset = self.config.resolution.internal.preset;
        let new_internal_preset = handle_res_dropdown(ui, internal_preset, "internal");
        
        if !new_internal_preset.is_custom() {
            if let Some((w, h)) = new_internal_preset.dimensions() {
                ui.same_line();
                ui.text_disabled(&format!("({}x{})", w, h));
            }
        }
        
        if new_internal_preset != internal_preset {
            self.config.resolution.internal.set_preset(new_internal_preset);
            let _ = self.config.save();
        }
        
        if self.config.resolution.internal.preset.is_custom() {
            ui.indent();
            let mut w = self.config.resolution.internal.custom_width;
            let mut h = self.config.resolution.internal.custom_height;
            ui.text("Custom:");
            ui.same_line();
            ui.set_next_item_width(80.0);
            if Drag::new("Width##internal").range(1, 7680).speed(1.0).build(ui, &mut w) {
                self.config.resolution.internal.custom_width = w;
                let _ = self.config.save();
            }
            ui.same_line();
            ui.text("x");
            ui.same_line();
            ui.set_next_item_width(80.0);
            if Drag::new("Height##internal").range(1, 4320).speed(1.0).build(ui, &mut h) {
                self.config.resolution.internal.custom_height = h;
                let _ = self.config.save();
            }
            ui.unindent();
        }
        ui.text_disabled("Resolution for texture processing (affects performance)");
        
        ui.separator();
        
        // Output Resolution
        ui.text("Output Resolution:");
        let output_preset = self.config.resolution.output.preset;
        let new_output_preset = handle_res_dropdown(ui, output_preset, "output");
        
        if !new_output_preset.is_custom() {
            if let Some((w, h)) = new_output_preset.dimensions() {
                ui.same_line();
                ui.text_disabled(&format!("({}x{})", w, h));
            }
        }
        
        if new_output_preset != output_preset {
            self.config.resolution.output.set_preset(new_output_preset);
            let _ = self.config.save();
        }
        
        if self.config.resolution.output.preset.is_custom() {
            ui.indent();
            let mut w = self.config.resolution.output.custom_width;
            let mut h = self.config.resolution.output.custom_height;
            ui.text("Custom:");
            ui.same_line();
            ui.set_next_item_width(80.0);
            if Drag::new("Width##output").range(1, 7680).speed(1.0).build(ui, &mut w) {
                self.config.resolution.output.custom_width = w;
                let _ = self.config.save();
            }
            ui.same_line();
            ui.text("x");
            ui.same_line();
            ui.set_next_item_width(80.0);
            if Drag::new("Height##output").range(1, 4320).speed(1.0).build(ui, &mut h) {
                self.config.resolution.output.custom_height = h;
                let _ = self.config.save();
            }
            ui.unindent();
        }
        ui.text_disabled("Resolution for display and recording output");
        
        ui.separator();
        
        // Apply button
        if ui.button("Apply Resolution Changes") {
            self.show_status("Resolution changes will take effect on next restart");
        }
        ui.same_line();
        ui.text_disabled("(Restart required)");
    }
    
    /// Draw Block 1 audio modulation panel
    fn draw_block1_audio_panel(&mut self, ui: &Ui) {
        ui.window("Audio Reactivity - Block 1")
            .size([400.0, 500.0], Condition::FirstUseEver)
            .build(|| {
                let param_names = get_block1_param_names();
                
                // Parameter selection
                ui.text("Select Parameter:");
                let preview = param_names[self.selected_block1_param as usize].clone();
                let mut selected = self.selected_block1_param;
                ComboBox::new(ui, "##b1_param_select")
                    .preview_value(&preview)
                    .build(|| {
                        for (idx, name) in param_names.iter().enumerate() {
                            if ui.selectable_config(name).selected(idx == selected as usize).build() {
                                selected = idx as i32;
                            }
                        }
                    });
                self.selected_block1_param = selected;
                
                let param_name = param_names[self.selected_block1_param as usize].clone();
                
                // Get or create modulation settings for this parameter
                let mod_settings = self.block1_audio_mods.entry(param_name.clone()).or_default();
                
                ui.separator();
                
                // Enable checkbox
                ui.checkbox("Enable Audio Mod", &mut mod_settings.enabled);
                
                // FFT Band dropdown
                let mut band_idx = mod_settings.fft_band as usize;
                let band_preview = FFT_BAND_NAMES[band_idx.min(7)].to_string();
                ComboBox::new(ui, "FFT Band")
                    .preview_value(&band_preview)
                    .build(|| {
                        for (idx, name) in FFT_BAND_NAMES.iter().enumerate() {
                            if ui.selectable_config(name).selected(idx == band_idx).build() {
                                band_idx = idx;
                            }
                        }
                    });
                mod_settings.fft_band = band_idx.clamp(0, 7) as i32;
                
                // Modulation Amount (bipolar: negative = inverse, positive = direct)
                Drag::new("Modulation Amount")
                    .speed(0.01)
                    .range(-2.0, 2.0)
                    .build(ui, &mut mod_settings.amount);

                // Attack/Release envelope controls
                Drag::new("Attack (s)")
                    .speed(0.001)
                    .range(0.001, 1.0)
                    .build(ui, &mut mod_settings.attack);
                Drag::new("Release (s)")
                    .speed(0.001)
                    .range(0.001, 1.0)
                    .build(ui, &mut mod_settings.release);

                ui.separator();

                // Apply button - adds to shared state
                let mut applied = false;
                if ui.button("Apply Modulation") {
                    if let Ok(mut state) = self.shared_state.lock() {
                        let mod_data = crate::params::preset::ParamModulationData {
                            audio_enabled: mod_settings.enabled,
                            audio_fft_band: mod_settings.fft_band,
                            audio_amount: mod_settings.amount,
                            audio_use_normalization: false,
                            audio_attack: mod_settings.attack,
                            audio_release: mod_settings.release,
                            audio_range_scale: 1.0,
                            audio_smoothed_value: 0.0,
                            bpm_enabled: false,
                            bpm_division_index: 2,
                            bpm_phase: 0.0,
                            bpm_waveform: 0,
                            bpm_min_value: 0.0,
                            bpm_max_value: 1.0,
                            bpm_bipolar: false,
                        };
                        state.block1_modulations.insert(param_name.clone(), mod_data);
                        applied = true;
                    }
                }
                if applied {
                    self.show_status(&format!("Applied audio mod: {}", param_name));
                }
                
                ui.separator();
                ui.text("Active Modulations:");
                
                // Show active modulations
                let active_mods: Vec<(String, i32, f32, f32, f32, bool)> = if let Ok(state) = self.shared_state.lock() {
                    state.block1_modulations
                        .iter()
                        .map(|(k, v)| (k.clone(), v.audio_fft_band, v.audio_amount, v.audio_attack, v.audio_release, v.audio_enabled))
                        .collect()
                } else {
                    Vec::new()
                };

                if active_mods.is_empty() {
                    ui.text_disabled("No active modulations");
                } else {
                    for (key, band, amount, attack, release, enabled) in active_mods {
                        let enabled_str = if enabled { "" } else { " [OFF]" };
                        ui.text(&format!("{}{}: Band {} @ {:.2}x  A:{:.3} R:{:.3}",
                            key, enabled_str, band, amount, attack, release));
                        ui.same_line();
                        if ui.small_button(&format!("Remove##b1_{}", key)) {
                            if let Ok(mut state) = self.shared_state.lock() {
                                state.block1_modulations.remove(&key);
                            }
                            self.block1_audio_mods.remove(&key);
                            self.show_status(&format!("Removed modulation: {}", key));
                        }
                    }
                }
            });
    }
    
    /// Draw Block 2 audio modulation panel
    fn draw_block2_audio_panel(&mut self, ui: &Ui) {
        ui.window("Audio Reactivity - Block 2")
            .size([400.0, 500.0], Condition::FirstUseEver)
            .build(|| {
                let param_names = get_block2_param_names();
                
                // Parameter selection
                ui.text("Select Parameter:");
                let preview = param_names[self.selected_block2_param as usize].clone();
                let mut selected = self.selected_block2_param;
                ComboBox::new(ui, "##b2_param_select")
                    .preview_value(&preview)
                    .build(|| {
                        for (idx, name) in param_names.iter().enumerate() {
                            if ui.selectable_config(name).selected(idx == selected as usize).build() {
                                selected = idx as i32;
                            }
                        }
                    });
                self.selected_block2_param = selected;
                
                let param_name = param_names[self.selected_block2_param as usize].clone();
                
                // Get or create modulation settings
                let mod_settings = self.block2_audio_mods.entry(param_name.clone()).or_default();
                
                ui.separator();
                
                ui.checkbox("Enable Audio Mod##b2", &mut mod_settings.enabled);
                let mut band_idx_b2 = mod_settings.fft_band as usize;
                let band_preview_b2 = FFT_BAND_NAMES[band_idx_b2.min(7)].to_string();
                ComboBox::new(ui, "FFT Band##b2")
                    .preview_value(&band_preview_b2)
                    .build(|| {
                        for (idx, name) in FFT_BAND_NAMES.iter().enumerate() {
                            if ui.selectable_config(name).selected(idx == band_idx_b2).build() {
                                band_idx_b2 = idx;
                            }
                        }
                    });
                mod_settings.fft_band = band_idx_b2.clamp(0, 7) as i32;
                Drag::new("Modulation Amount##b2").speed(0.01).range(-2.0, 2.0).build(ui, &mut mod_settings.amount);
                Drag::new("Attack (s)##b2").speed(0.001).range(0.001, 1.0).build(ui, &mut mod_settings.attack);
                Drag::new("Release (s)##b2").speed(0.001).range(0.001, 1.0).build(ui, &mut mod_settings.release);

                ui.separator();

                let mut applied_b2 = false;
                if ui.button("Apply Modulation##b2") {
                    if let Ok(mut state) = self.shared_state.lock() {
                        let mod_data = crate::params::preset::ParamModulationData {
                            audio_enabled: mod_settings.enabled,
                            audio_fft_band: mod_settings.fft_band,
                            audio_amount: mod_settings.amount,
                            audio_use_normalization: false,
                            audio_attack: mod_settings.attack,
                            audio_release: mod_settings.release,
                            audio_range_scale: 1.0,
                            audio_smoothed_value: 0.0,
                            bpm_enabled: false,
                            bpm_division_index: 2,
                            bpm_phase: 0.0,
                            bpm_waveform: 0,
                            bpm_min_value: 0.0,
                            bpm_max_value: 1.0,
                            bpm_bipolar: false,
                        };
                        state.block2_modulations.insert(param_name.clone(), mod_data);
                        applied_b2 = true;
                    }
                }
                if applied_b2 {
                    self.show_status(&format!("Applied audio mod: {}", param_name));
                }
                
                ui.separator();
                ui.text("Active Modulations:");
                
                let active_mods: Vec<(String, i32, f32, f32, f32, bool)> = if let Ok(state) = self.shared_state.lock() {
                    state.block2_modulations
                        .iter()
                        .map(|(k, v)| (k.clone(), v.audio_fft_band, v.audio_amount, v.audio_attack, v.audio_release, v.audio_enabled))
                        .collect()
                } else {
                    Vec::new()
                };

                if active_mods.is_empty() {
                    ui.text_disabled("No active modulations");
                } else {
                    for (key, band, amount, attack, release, enabled) in active_mods {
                        let enabled_str = if enabled { "" } else { " [OFF]" };
                        ui.text(&format!("{}{}: Band {} @ {:.2}x  A:{:.3} R:{:.3}",
                            key, enabled_str, band, amount, attack, release));
                        ui.same_line();
                        if ui.small_button(&format!("Remove##b2_{}", key)) {
                            if let Ok(mut state) = self.shared_state.lock() {
                                state.block2_modulations.remove(&key);
                            }
                            self.block2_audio_mods.remove(&key);
                            self.show_status(&format!("Removed modulation: {}", key));
                        }
                    }
                }
            });
    }
    
    /// Draw Block 3 audio modulation panel
    fn draw_block3_audio_panel(&mut self, ui: &Ui) {
        ui.window("Audio Reactivity - Block 3")
            .size([400.0, 500.0], Condition::FirstUseEver)
            .build(|| {
                let param_names = get_block3_param_names();
                
                // Parameter selection
                ui.text("Select Parameter:");
                let preview = param_names[self.selected_block3_param as usize].clone();
                let mut selected = self.selected_block3_param;
                ComboBox::new(ui, "##b3_param_select")
                    .preview_value(&preview)
                    .build(|| {
                        for (idx, name) in param_names.iter().enumerate() {
                            if ui.selectable_config(name).selected(idx == selected as usize).build() {
                                selected = idx as i32;
                            }
                        }
                    });
                self.selected_block3_param = selected;
                
                let param_name = param_names[self.selected_block3_param as usize].clone();
                
                // Get or create modulation settings
                let mod_settings = self.block3_audio_mods.entry(param_name.clone()).or_default();
                
                ui.separator();
                
                ui.checkbox("Enable Audio Mod##b3", &mut mod_settings.enabled);
                let mut band_idx_b3 = mod_settings.fft_band as usize;
                let band_preview_b3 = FFT_BAND_NAMES[band_idx_b3.min(7)].to_string();
                ComboBox::new(ui, "FFT Band##b3")
                    .preview_value(&band_preview_b3)
                    .build(|| {
                        for (idx, name) in FFT_BAND_NAMES.iter().enumerate() {
                            if ui.selectable_config(name).selected(idx == band_idx_b3).build() {
                                band_idx_b3 = idx;
                            }
                        }
                    });
                mod_settings.fft_band = band_idx_b3.clamp(0, 7) as i32;
                Drag::new("Modulation Amount##b3").speed(0.01).range(-2.0, 2.0).build(ui, &mut mod_settings.amount);
                Drag::new("Attack (s)##b3").speed(0.001).range(0.001, 1.0).build(ui, &mut mod_settings.attack);
                Drag::new("Release (s)##b3").speed(0.001).range(0.001, 1.0).build(ui, &mut mod_settings.release);

                ui.separator();

                let mut applied_b3 = false;
                if ui.button("Apply Modulation##b3") {
                    if let Ok(mut state) = self.shared_state.lock() {
                        let mod_data = crate::params::preset::ParamModulationData {
                            audio_enabled: mod_settings.enabled,
                            audio_fft_band: mod_settings.fft_band,
                            audio_amount: mod_settings.amount,
                            audio_use_normalization: false,
                            audio_attack: mod_settings.attack,
                            audio_release: mod_settings.release,
                            audio_range_scale: 1.0,
                            audio_smoothed_value: 0.0,
                            bpm_enabled: false,
                            bpm_division_index: 2,
                            bpm_phase: 0.0,
                            bpm_waveform: 0,
                            bpm_min_value: 0.0,
                            bpm_max_value: 1.0,
                            bpm_bipolar: false,
                        };
                        state.block3_modulations.insert(param_name.clone(), mod_data);
                        applied_b3 = true;
                    }
                }
                if applied_b3 {
                    self.show_status(&format!("Applied audio mod: {}", param_name));
                }
                
                ui.separator();
                ui.text("Active Modulations:");
                
                let active_mods: Vec<(String, i32, f32, f32, f32, bool)> = if let Ok(state) = self.shared_state.lock() {
                    state.block3_modulations
                        .iter()
                        .map(|(k, v)| (k.clone(), v.audio_fft_band, v.audio_amount, v.audio_attack, v.audio_release, v.audio_enabled))
                        .collect()
                } else {
                    Vec::new()
                };

                if active_mods.is_empty() {
                    ui.text_disabled("No active modulations");
                } else {
                    for (key, band, amount, attack, release, enabled) in active_mods {
                        let enabled_str = if enabled { "" } else { " [OFF]" };
                        ui.text(&format!("{}{}: Band {} @ {:.2}x  A:{:.3} R:{:.3}",
                            key, enabled_str, band, amount, attack, release));
                        ui.same_line();
                        if ui.small_button(&format!("Remove##b3_{}", key)) {
                            if let Ok(mut state) = self.shared_state.lock() {
                                state.block3_modulations.remove(&key);
                            }
                            self.block3_audio_mods.remove(&key);
                            self.show_status(&format!("Removed modulation: {}", key));
                        }
                    }
                }
            });
    }
    
    /// Draw Preview Window with color picker
    fn draw_preview_window(&mut self, ui: &Ui, frame_count: u64) {
        let sources = [
            PreviewSource::Block1,
            PreviewSource::Block2,
            PreviewSource::Block3,
            PreviewSource::Input1,
            PreviewSource::Input2,
        ];
        
        // Track window open state
        let mut is_open = self.show_preview_window;
        
        ui.window("Preview & Color Picker")
            .size([400.0, 350.0], Condition::FirstUseEver)
            .position([100.0, 100.0], Condition::FirstUseEver)
            .opened(&mut is_open)
            .build(|| {
                // Close button at top right
                if ui.button("X##close_preview") {
                    self.show_preview_window = false;
                    // Sync to shared state so engine stops computing
                    if let Ok(mut state) = self.shared_state.lock() {
                        state.preview_enabled = false;
                    }
                }
                
                ui.separator();
                // Get current preview source from shared state
                let mut preview_source = if let Ok(state) = self.shared_state.lock() {
                    state.preview_source
                } else {
                    crate::core::PreviewSource::Block3
                };
                
                // Source selection dropdown
                let preview_names: Vec<&str> = sources.iter().map(|s| s.display_name()).collect();
                let mut selected_idx = sources.iter().position(|&s| s == preview_source).unwrap_or(2);
                
                ui.text("Preview Source:");
                let preview = preview_names[selected_idx];
                ComboBox::new(ui, "##preview_source")
                    .preview_value(preview)
                    .build(|| {
                        for (idx, name) in preview_names.iter().enumerate() {
                            if ui.selectable_config(*name).selected(idx == selected_idx).build() {
                                selected_idx = idx;
                            }
                        }
                    });
                
                // Update shared state if changed
                let new_source = sources[selected_idx];
                if new_source != preview_source {
                    if let Ok(mut state) = self.shared_state.lock() {
                        state.preview_source = new_source;
                    }
                }
                
                ui.separator();
                
                // Preview image display
                ui.text("Preview Area:");
                
                // Calculate preview size maintaining 16:9 aspect ratio
                let available_width = ui.content_region_avail()[0];
                let preview_width = available_width.min(320.0);
                let preview_height = preview_width * (9.0 / 16.0);
                
                // Store image position and size for mouse picking
                let cursor_pos = ui.cursor_screen_pos();
                self.preview_image_pos = [cursor_pos[0], cursor_pos[1]];
                self.preview_image_size = [preview_width, preview_height];
                
                // Display the preview texture if available
                if let Some(texture_id) = self.preview_texture_id {
                    // Store values locally to avoid borrow issues
                    let image_pos = self.preview_image_pos;
                    let crosshair_uv = self.preview_crosshair_uv;
                    
                    // Draw the preview image
                    imgui::Image::new(texture_id, [preview_width, preview_height])
                        .uv0([0.0, 0.0])  // Top-left
                        .uv1([1.0, 1.0])  // Bottom-right
                        .build(ui);
                    
                    // Handle mouse click on preview for color picking
                    let mouse_pos = ui.io().mouse_pos;
                    let is_hovering = mouse_pos[0] >= image_pos[0] 
                        && mouse_pos[0] < image_pos[0] + preview_width
                        && mouse_pos[1] >= image_pos[1]
                        && mouse_pos[1] < image_pos[1] + preview_height;
                    
                    if is_hovering && ui.is_mouse_clicked(imgui::MouseButton::Left) {
                        // Calculate UV coordinates from mouse position
                        let u = (mouse_pos[0] - image_pos[0]) / preview_width;
                        let v = (mouse_pos[1] - image_pos[1]) / preview_height;
                        
                        // Store the crosshair position
                        self.preview_crosshair_uv = [u, v];
                        
                        // Store the pick position in shared state for the engine to process
                        if let Ok(mut state) = self.shared_state.lock() {
                            state.preview_pick_uv = [u, v];
                            state.preview_pick_requested = true;
                        }
                        
                        log::debug!("Color pick at UV: [{:.3}, {:.3}]", u, v);
                    }
                    
                    // Draw crosshair overlay at stored position
                    let draw_list = ui.get_window_draw_list();
                    let crosshair_x = image_pos[0] + preview_width * crosshair_uv[0];
                    let crosshair_y = image_pos[1] + preview_height * crosshair_uv[1];
                    let crosshair_color = 0xFF00FF00; // Green, ABGR format
                    let crosshair_size = 10.0;
                    
                    // Horizontal line
                    draw_list.add_line(
                        [crosshair_x - crosshair_size, crosshair_y],
                        [crosshair_x + crosshair_size, crosshair_y],
                        crosshair_color,
                    ).build();
                    
                    // Vertical line
                    draw_list.add_line(
                        [crosshair_x, crosshair_y - crosshair_size],
                        [crosshair_x, crosshair_y + crosshair_size],
                        crosshair_color,
                    ).build();
                    
                    // Small circle at center
                    draw_list.add_circle([crosshair_x, crosshair_y], 3.0, crosshair_color)
                        .filled(true)
                        .build();
                } else {
                    // Placeholder when texture not available
                    ui.child_window("PreviewImagePlaceholder")
                        .size([preview_width, preview_height])
                        .border(true)
                        .build(|| {
                            ui.text("Preview texture not available");
                            ui.text("(Restart may be required)");
                        });
                }
                
                ui.separator();
                
                // Color picker section
                ui.text("Color Picker:");
                ui.text_disabled("Click on preview to sample color");
                
                // Get sampled color from shared state
                let sampled_color = if let Ok(state) = self.shared_state.lock() {
                    state.preview_sampled_color
                } else {
                    [1.0, 1.0, 1.0]
                };
                
                // Display sampled color
                ui.color_button("Sampled Color", [sampled_color[0], 
                    sampled_color[1], sampled_color[2], 1.0]);
                ui.same_line();
                ui.text(format!("R: {:.2} G: {:.2} B: {:.2}", 
                    sampled_color[0],
                    sampled_color[1],
                    sampled_color[2]));
                
                // Color edit for manual adjustment
                let mut editable_color = sampled_color;
                if ui.color_edit3("Edit Color", &mut editable_color) {
                    // Update shared state with edited color
                    if let Ok(mut state) = self.shared_state.lock() {
                        state.preview_sampled_color = editable_color;
                    }
                }
                
                // Key target selection and apply
                let key_targets = [
                    "Block 1 - CH2 Key",
                    "Block 1 - FB1 Key",
                    "Block 2 - FB2 Key",
                    "Block 3 - Final Key",
                ];
                
                ui.text("Apply to Key:");
                
                // Target dropdown
                ComboBox::new(ui, "##key_target")
                    .preview_value(key_targets[self.selected_key_target as usize])
                    .build(|| {
                        for (idx, name) in key_targets.iter().enumerate() {
                            if ui.selectable_config(*name).selected(idx == self.selected_key_target as usize).build() {
                                self.selected_key_target = idx as i32;
                            }
                        }
                    });
                
                ui.same_line();
                
                // Apply button
                if ui.button("Apply") {
                    let color = if let Ok(state) = self.shared_state.lock() {
                        state.preview_sampled_color
                    } else {
                        [1.0, 1.0, 1.0]
                    };
                    
                    // Apply color to selected key target (both shared state AND local edit copy)
                    let status_msg = match self.selected_key_target {
                        0 => { // Block 1 - CH2 Key
                            // Update shared state
                            if let Ok(mut state) = self.shared_state.lock() {
                                state.block1.ch2_key_value_red = color[0];
                                state.block1.ch2_key_value_green = color[1];
                                state.block1.ch2_key_value_blue = color[2];
                            }
                            // Also update local edit copy so GUI shows the new value immediately
                            self.block1_edit.ch2_key_value_red = color[0];
                            self.block1_edit.ch2_key_value_green = color[1];
                            self.block1_edit.ch2_key_value_blue = color[2];
                            "Color applied to Block 1 CH2 Key"
                        }
                        1 => { // Block 1 - FB1 Key
                            if let Ok(mut state) = self.shared_state.lock() {
                                state.block1.fb1_key_value_red = color[0];
                                state.block1.fb1_key_value_green = color[1];
                                state.block1.fb1_key_value_blue = color[2];
                            }
                            self.block1_edit.fb1_key_value_red = color[0];
                            self.block1_edit.fb1_key_value_green = color[1];
                            self.block1_edit.fb1_key_value_blue = color[2];
                            "Color applied to Block 1 FB1 Key"
                        }
                        2 => { // Block 2 - FB2 Key
                            use glam::Vec3;
                            if let Ok(mut state) = self.shared_state.lock() {
                                state.block2.fb2_key_value = Vec3::new(color[0], color[1], color[2]);
                            }
                            self.block2_edit.fb2_key_value = Vec3::new(color[0], color[1], color[2]);
                            "Color applied to Block 2 FB2 Key"
                        }
                        3 => { // Block 3 - Final Key
                            use glam::Vec3;
                            if let Ok(mut state) = self.shared_state.lock() {
                                state.block3.final_key_value = Vec3::new(color[0], color[1], color[2]);
                            }
                            self.block3_edit.final_key_value = Vec3::new(color[0], color[1], color[2]);
                            "Color applied to Block 3 Final Key"
                        }
                        _ => ""
                    };
                    
                    if !status_msg.is_empty() {
                        self.show_status(status_msg);
                    }
                }
            });
        
        // Handle window close (user clicked X button or pressed Esc)
        if !is_open && self.show_preview_window {
            self.show_preview_window = false;
            // Disable preview computation to save GPU
            if let Ok(mut state) = self.shared_state.lock() {
                state.preview_enabled = false;
            }
        }
    }
    
    /// Build Block 1 LFO panel - Organized by OF LFO groups
    fn build_block1_lfo(&mut self, ui: &Ui) {
        // Global tempo control
        self.draw_tempo_control(ui);
        ui.separator();
        
        // === CH1 Adjust LFOs (16 params) ===
        if CollapsingHeader::new("CH1 Adjust LFOs").default_open(true).build(ui) {
            let ch1_params = [
                // X/Y/Z Displace <->
                ("ch1_x_displace", "X Displace"),
                ("ch1_y_displace", "Y Displace"),
                ("ch1_z_displace", "Z Displace"),
                // Rotate
                ("ch1_rotate", "Rotate"),
                // HSB Hue ^^
                ("ch1_hsb_attenuate_x", "HSB Hue"),
                // HSB Sat ^^
                ("ch1_hsb_attenuate_y", "HSB Sat"),
                // HSB Bri ^^
                ("ch1_hsb_attenuate_z", "HSB Bri"),
                // Kaleidoscope Slice
                ("ch1_kaleidoscope_slice", "Kaleidoscope Slice"),
                // Blur
                ("ch1_blur_amount", "Blur Amount"),
                ("ch1_blur_radius", "Blur Radius"),
                // Sharpen
                ("ch1_sharpen_amount", "Sharpen Amount"),
                ("ch1_sharpen_radius", "Sharpen Radius"),
            ];
            
            for (param_id, label) in &ch1_params {
                self.draw_lfo_control(ui, param_id, label, 1);
            }
        }
        
        // === CH2 Mix & Key LFOs (6 params) ===
        if CollapsingHeader::new("CH2 Mix & Key LFOs").default_open(true).build(ui) {
            let ch2_mix_params = [
                ("ch2_mix_amount", "Mix Amount"),
                ("ch2_key_threshold", "Key Threshold"),
                ("ch2_key_soft", "Key Soft"),
            ];
            
            for (param_id, label) in &ch2_mix_params {
                self.draw_lfo_control(ui, param_id, label, 1);
            }
        }
        
        // === CH2 Adjust LFOs (16 params) ===
        if CollapsingHeader::new("CH2 Adjust LFOs").default_open(true).build(ui) {
            let ch2_params = [
                // X/Y/Z Displace <->
                ("ch2_x_displace", "X Displace"),
                ("ch2_y_displace", "Y Displace"),
                ("ch2_z_displace", "Z Displace"),
                // Rotate
                ("ch2_rotate", "Rotate"),
                // HSB Hue ^^
                ("ch2_hsb_attenuate_x", "HSB Hue"),
                // HSB Sat ^^
                ("ch2_hsb_attenuate_y", "HSB Sat"),
                // HSB Bri ^^
                ("ch2_hsb_attenuate_z", "HSB Bri"),
                // Kaleidoscope Slice
                ("ch2_kaleidoscope_slice", "Kaleidoscope Slice"),
                // Blur
                ("ch2_blur_amount", "Blur Amount"),
                ("ch2_blur_radius", "Blur Radius"),
                // Sharpen
                ("ch2_sharpen_amount", "Sharpen Amount"),
                ("ch2_sharpen_radius", "Sharpen Radius"),
            ];
            
            for (param_id, label) in &ch2_params {
                self.draw_lfo_control(ui, param_id, label, 1);
            }
        }
        
        // === FB1 Mix & Key LFOs ===
        if CollapsingHeader::new("FB1 Mix & Key LFOs").default_open(true).build(ui) {
            let fb1_mix_params = [
                ("fb1_mix_amount", "Mix Amount"),
                ("fb1_key_threshold", "Key Threshold"),
                ("fb1_key_soft", "Key Soft"),
            ];
            
            for (param_id, label) in &fb1_mix_params {
                self.draw_lfo_control(ui, param_id, label, 1);
            }
        }
        
        // === FB1 Geo1 LFOs (8 params) ===
        if CollapsingHeader::new("FB1 Geo1 LFOs").default_open(true).build(ui) {
            let fb1_geo1_params = [
                ("fb1_x_displace", "X Displace"),
                ("fb1_y_displace", "Y Displace"),
                ("fb1_z_displace", "Z Displace"),
                ("fb1_rotate", "Rotate"),
            ];
            
            for (param_id, label) in &fb1_geo1_params {
                self.draw_lfo_control(ui, param_id, label, 1);
            }
        }
        
        // === FB1 Geo2 LFOs (10 params) ===
        if CollapsingHeader::new("FB1 Geo2 LFOs").default_open(true).build(ui) {
            let fb1_geo2_params = [
                // Stretch (X/Y) - from shear matrix x, w
                ("fb1_shear_matrix_x", "X Stretch"),
                ("fb1_shear_matrix_w", "Y Stretch"),
                // Shear (X/Y) - from shear matrix y, z
                ("fb1_shear_matrix_y", "X Shear"),
                ("fb1_shear_matrix_z", "Y Shear"),
                // Kaleidoscope Slice
                ("fb1_kaleidoscope_slice", "Kaleidoscope Slice"),
            ];
            
            for (param_id, label) in &fb1_geo2_params {
                self.draw_lfo_control(ui, param_id, label, 1);
            }
        }
        
        // === FB1 Color1 LFOs (6 params) ===
        if CollapsingHeader::new("FB1 Color LFOs").default_open(true).build(ui) {
            let fb1_color_params = [
                ("fb1_hsb_offset_x", "HSB Offset Hue"),
                ("fb1_hsb_offset_y", "HSB Offset Sat"),
                ("fb1_hsb_offset_z", "HSB Offset Bri"),
                ("fb1_hsb_attenuate_x", "HSB Attenuate Hue"),
                ("fb1_hsb_attenuate_y", "HSB Attenuate Sat"),
                ("fb1_hsb_attenuate_z", "HSB Attenuate Bri"),
            ];
            
            for (param_id, label) in &fb1_color_params {
                self.draw_lfo_control(ui, param_id, label, 1);
            }
        }
        
        // === FB1 Delay Time LFO ===
        if CollapsingHeader::new("FB1 Delay LFO").default_open(true).build(ui) {
            self.draw_lfo_control(ui, "fb1_delay_time", "Delay Time", 1);
        }
    }
    
    /// Build Block 2 LFO panel - Organized by OF LFO groups
    fn build_block2_lfo(&mut self, ui: &Ui) {
        // Global tempo control
        self.draw_tempo_control(ui);
        ui.separator();
        
        // === Block2 Input Adjust LFOs (16 params) ===
        if CollapsingHeader::new("Input Adjust LFOs").default_open(true).build(ui) {
            let input_params = [
                // X/Y/Z Displace <->
                ("block2_input_x_displace", "X Displace"),
                ("block2_input_y_displace", "Y Displace"),
                ("block2_input_z_displace", "Z Displace"),
                // Rotate
                ("block2_input_rotate", "Rotate"),
                // HSB Hue ^^
                ("block2_input_hsb_attenuate_x", "HSB Hue"),
                // HSB Sat ^^
                ("block2_input_hsb_attenuate_y", "HSB Sat"),
                // HSB Bri ^^
                ("block2_input_hsb_attenuate_z", "HSB Bri"),
                // Kaleidoscope Slice
                ("block2_input_kaleidoscope_slice", "Kaleidoscope Slice"),
                // Blur
                ("block2_input_blur_amount", "Blur Amount"),
                ("block2_input_blur_radius", "Blur Radius"),
                // Sharpen
                ("block2_input_sharpen_amount", "Sharpen Amount"),
                ("block2_input_sharpen_radius", "Sharpen Radius"),
            ];
            
            for (param_id, label) in &input_params {
                self.draw_lfo_control(ui, param_id, label, 2);
            }
        }
        
        // === FB2 Mix & Key LFOs ===
        if CollapsingHeader::new("FB2 Mix & Key LFOs").default_open(true).build(ui) {
            let fb2_mix_params = [
                ("fb2_mix_amount", "Mix Amount"),
                ("fb2_key_threshold", "Key Threshold"),
                ("fb2_key_soft", "Key Soft"),
            ];
            
            for (param_id, label) in &fb2_mix_params {
                self.draw_lfo_control(ui, param_id, label, 2);
            }
        }
        
        // === FB2 Geo1 LFOs (8 params) ===
        if CollapsingHeader::new("FB2 Geo1 LFOs").default_open(true).build(ui) {
            let fb2_geo1_params = [
                ("fb2_x_displace", "X Displace"),
                ("fb2_y_displace", "Y Displace"),
                ("fb2_z_displace", "Z Displace"),
                ("fb2_rotate", "Rotate"),
            ];
            
            for (param_id, label) in &fb2_geo1_params {
                self.draw_lfo_control(ui, param_id, label, 2);
            }
        }
        
        // === FB2 Geo2 LFOs (10 params) ===
        if CollapsingHeader::new("FB2 Geo2 LFOs").default_open(true).build(ui) {
            let fb2_geo2_params = [
                // Stretch (X/Y) - from shear matrix x, w
                ("fb2_shear_matrix_x", "X Stretch"),
                ("fb2_shear_matrix_w", "Y Stretch"),
                // Shear (X/Y) - from shear matrix y, z
                ("fb2_shear_matrix_y", "X Shear"),
                ("fb2_shear_matrix_z", "Y Shear"),
                // Kaleidoscope Slice
                ("fb2_kaleidoscope_slice", "Kaleidoscope Slice"),
            ];
            
            for (param_id, label) in &fb2_geo2_params {
                self.draw_lfo_control(ui, param_id, label, 2);
            }
        }
        
        // === FB2 Color LFOs (6 params) ===
        if CollapsingHeader::new("FB2 Color LFOs").default_open(true).build(ui) {
            let fb2_color_params = [
                ("fb2_hsb_offset_x", "HSB Offset Hue"),
                ("fb2_hsb_offset_y", "HSB Offset Sat"),
                ("fb2_hsb_offset_z", "HSB Offset Bri"),
                ("fb2_hsb_attenuate_x", "HSB Attenuate Hue"),
                ("fb2_hsb_attenuate_y", "HSB Attenuate Sat"),
                ("fb2_hsb_attenuate_z", "HSB Attenuate Bri"),
            ];
            
            for (param_id, label) in &fb2_color_params {
                self.draw_lfo_control(ui, param_id, label, 2);
            }
        }
        
        // === FB2 Delay Time LFO ===
        if CollapsingHeader::new("FB2 Delay LFO").default_open(true).build(ui) {
            self.draw_lfo_control(ui, "fb2_delay_time", "Delay Time", 2);
        }
    }
    
    /// Build Block 3 LFO panel - Organized by OF LFO groups
    fn build_block3_lfo(&mut self, ui: &Ui) {
        // Global tempo control
        self.draw_tempo_control(ui);
        ui.separator();
        
        // === Block 1 Re-process Geo1 LFOs (8 params) ===
        if CollapsingHeader::new("Block1 Geo1 LFOs").default_open(true).build(ui) {
            let b1_geo1_params = [
                ("block1_x_displace", "X Displace"),
                ("block1_y_displace", "Y Displace"),
                ("block1_z_displace", "Z Displace"),
                ("block1_rotate", "Rotate"),
            ];
            
            for (param_id, label) in &b1_geo1_params {
                self.draw_lfo_control(ui, param_id, label, 3);
            }
        }
        
        // === Block 1 Re-process Geo2 LFOs (10 params) ===
        if CollapsingHeader::new("Block1 Geo2 LFOs").default_open(true).build(ui) {
            let b1_geo2_params = [
                // Stretch (X/Y) - from shear matrix x, w
                ("block1_shear_matrix_x", "X Stretch"),
                ("block1_shear_matrix_w", "Y Stretch"),
                // Shear (X/Y) - from shear matrix y, z
                ("block1_shear_matrix_y", "X Shear"),
                ("block1_shear_matrix_z", "Y Shear"),
                // Kaleidoscope Slice
                ("block1_kaleidoscope_slice", "Kaleidoscope Slice"),
            ];
            
            for (param_id, label) in &b1_geo2_params {
                self.draw_lfo_control(ui, param_id, label, 3);
            }
        }
        
        // === Block 1 Colorize LFOs (30 params total) ===
        if CollapsingHeader::new("Block1 Colorize LFOs").default_open(true).build(ui) {
            let b1_color_params = [
                // Band 1
                ("block1_colorize_band1_x", "Band1 Hue/Red"),
                ("block1_colorize_band1_y", "Band1 Sat/Green"),
                ("block1_colorize_band1_z", "Band1 Bri/Blue"),
                // Band 2
                ("block1_colorize_band2_x", "Band2 Hue/Red"),
                ("block1_colorize_band2_y", "Band2 Sat/Green"),
                ("block1_colorize_band2_z", "Band2 Bri/Blue"),
                // Band 3
                ("block1_colorize_band3_x", "Band3 Hue/Red"),
                ("block1_colorize_band3_y", "Band3 Sat/Green"),
                ("block1_colorize_band3_z", "Band3 Bri/Blue"),
                // Band 4
                ("block1_colorize_band4_x", "Band4 Hue/Red"),
                ("block1_colorize_band4_y", "Band4 Sat/Green"),
                ("block1_colorize_band4_z", "Band4 Bri/Blue"),
                // Band 5
                ("block1_colorize_band5_x", "Band5 Hue/Red"),
                ("block1_colorize_band5_y", "Band5 Sat/Green"),
                ("block1_colorize_band5_z", "Band5 Bri/Blue"),
            ];
            
            for (param_id, label) in &b1_color_params {
                self.draw_lfo_control(ui, param_id, label, 3);
            }
        }
        
        // === Block 2 Re-process Geo1 LFOs (8 params) ===
        if CollapsingHeader::new("Block2 Geo1 LFOs").default_open(true).build(ui) {
            let b2_geo1_params = [
                ("block2_x_displace", "X Displace"),
                ("block2_y_displace", "Y Displace"),
                ("block2_z_displace", "Z Displace"),
                ("block2_rotate", "Rotate"),
            ];
            
            for (param_id, label) in &b2_geo1_params {
                self.draw_lfo_control(ui, param_id, label, 3);
            }
        }
        
        // === Block 2 Re-process Geo2 LFOs (10 params) ===
        if CollapsingHeader::new("Block2 Geo2 LFOs").default_open(true).build(ui) {
            let b2_geo2_params = [
                // Stretch (X/Y) - from shear matrix x, w
                ("block2_shear_matrix_x", "X Stretch"),
                ("block2_shear_matrix_w", "Y Stretch"),
                // Shear (X/Y) - from shear matrix y, z
                ("block2_shear_matrix_y", "X Shear"),
                ("block2_shear_matrix_z", "Y Shear"),
                // Kaleidoscope Slice
                ("block2_kaleidoscope_slice", "Kaleidoscope Slice"),
            ];
            
            for (param_id, label) in &b2_geo2_params {
                self.draw_lfo_control(ui, param_id, label, 3);
            }
        }
        
        // === Block 2 Colorize LFOs (30 params total) ===
        if CollapsingHeader::new("Block2 Colorize LFOs").default_open(true).build(ui) {
            let b2_color_params = [
                // Band 1
                ("block2_colorize_band1_x", "Band1 Hue/Red"),
                ("block2_colorize_band1_y", "Band1 Sat/Green"),
                ("block2_colorize_band1_z", "Band1 Bri/Blue"),
                // Band 2
                ("block2_colorize_band2_x", "Band2 Hue/Red"),
                ("block2_colorize_band2_y", "Band2 Sat/Green"),
                ("block2_colorize_band2_z", "Band2 Bri/Blue"),
                // Band 3
                ("block2_colorize_band3_x", "Band3 Hue/Red"),
                ("block2_colorize_band3_y", "Band3 Sat/Green"),
                ("block2_colorize_band3_z", "Band3 Bri/Blue"),
                // Band 4
                ("block2_colorize_band4_x", "Band4 Hue/Red"),
                ("block2_colorize_band4_y", "Band4 Sat/Green"),
                ("block2_colorize_band4_z", "Band4 Bri/Blue"),
                // Band 5
                ("block2_colorize_band5_x", "Band5 Hue/Red"),
                ("block2_colorize_band5_y", "Band5 Sat/Green"),
                ("block2_colorize_band5_z", "Band5 Bri/Blue"),
            ];
            
            for (param_id, label) in &b2_color_params {
                self.draw_lfo_control(ui, param_id, label, 3);
            }
        }
        
        // === Matrix Mix LFOs (18 params) ===
        if CollapsingHeader::new("Matrix Mix LFOs").default_open(true).build(ui) {
            let matrix_params = [
                // B1 R -> B2 R/G/B
                ("matrix_mix_r_to_r", "B1 Red -> B2 Red"),
                ("matrix_mix_r_to_g", "B1 Red -> B2 Green"),
                ("matrix_mix_r_to_b", "B1 Red -> B2 Blue"),
                // B1 G -> B2 R/G/B
                ("matrix_mix_g_to_r", "B1 Green -> B2 Red"),
                ("matrix_mix_g_to_g", "B1 Green -> B2 Green"),
                ("matrix_mix_g_to_b", "B1 Green -> B2 Blue"),
                // B1 B -> B2 R/G/B
                ("matrix_mix_b_to_r", "B1 Blue -> B2 Red"),
                ("matrix_mix_b_to_g", "B1 Blue -> B2 Green"),
                ("matrix_mix_b_to_b", "B1 Blue -> B2 Blue"),
            ];
            
            for (param_id, label) in &matrix_params {
                self.draw_lfo_control(ui, param_id, label, 3);
            }
        }
        
        // === Final Mix & Key LFOs (6 params) ===
        if CollapsingHeader::new("Final Mix & Key LFOs").default_open(true).build(ui) {
            let final_params = [
                ("final_mix_amount", "Mix Amount"),
                ("final_key_threshold", "Key Threshold"),
                ("final_key_soft", "Key Soft"),
            ];
            
            for (param_id, label) in &final_params {
                self.draw_lfo_control(ui, param_id, label, 3);
            }
        }
        
        // === Dither LFOs ===
        if CollapsingHeader::new("Dither LFOs").default_open(true).build(ui) {
            let dither_params = [
                ("block1_dither", "Block1 Dither"),
                ("block2_dither", "Block2 Dither"),
                ("final_dither", "Final Dither"),
            ];
            
            for (param_id, label) in &dither_params {
                self.draw_lfo_control(ui, param_id, label, 3);
            }
        }
    }
    
    /// Draw LFO control for a single parameter
    fn draw_lfo_control(&mut self, ui: &Ui, param_id: &str, label: &str, block: i32) {
        let lfo_map = match block {
            1 => &mut self.block1_lfos,
            2 => &mut self.block2_lfos,
            3 => &mut self.block3_lfos,
            _ => &mut self.block1_lfos,
        };
        
        // Get or create LFO state
        let lfo = lfo_map.entry(param_id.to_string()).or_insert_with(|| LfoState {
            enabled: false,
            amplitude: 0.0,
            rate: 0.5,
            waveform: 0,
            tempo_sync: false,
            division: 2, // 1/4 note default
            bank_index: 0,
        });
        
        ui.text(label);
        
        ui.checkbox(&format!("Enable##{}_{}", param_id, block), &mut lfo.enabled);
        
        if lfo.enabled {
            ui.indent();
            
            // LFO Bank selector (0-15)
            let mut bank_idx = lfo.bank_index as usize;
            ui.text("LFO Bank:");
            ui.same_line();
            for i in 0..8 {
                if i > 0 { ui.same_line(); }
                let label = format!("{}##{}_{}_{}", i, param_id, block, i);
                if ui.radio_button_bool(&label, bank_idx == i) {
                    bank_idx = i;
                }
            }
            lfo.bank_index = bank_idx.clamp(0, 15) as i32;
            
            Drag::new(&format!("Amplitude##{}_{}", param_id, block))
                .speed(0.01).range(0.0, 1.0).build(ui, &mut lfo.amplitude);
            
            ui.checkbox(&format!("Tempo Sync##{}_{}", param_id, block), &mut lfo.tempo_sync);
            
            if lfo.tempo_sync {
                // Beat division selector
                let mut div_idx = lfo.division as usize;
                let preview = BEAT_DIVISIONS[div_idx.min(BEAT_DIVISIONS.len() - 1)].to_string();
                ComboBox::new(ui, &format!("Division##{}_{}", param_id, block))
                    .preview_value(&preview)
                    .build(|| {
                        for (idx, opt) in BEAT_DIVISIONS.iter().enumerate() {
                            if ui.selectable_config(opt).selected(idx == div_idx).build() {
                                div_idx = idx;
                            }
                        }
                    });
                lfo.division = div_idx.clamp(0, BEAT_DIVISIONS.len() - 1) as i32;
            } else {
                // Free rate control
                Drag::new(&format!("Rate##{}_{}", param_id, block))
                    .speed(0.01).range(0.0, 10.0).build(ui, &mut lfo.rate);
            }
            
            // Waveform selector
            let mut wave_idx = lfo.waveform as usize;
            let preview = WAVEFORM_NAMES[wave_idx.min(WAVEFORM_NAMES.len() - 1)].to_string();
            ComboBox::new(ui, &format!("Waveform##{}_{}", param_id, block))
                .preview_value(&preview)
                .build(|| {
                    for (idx, opt) in WAVEFORM_NAMES.iter().enumerate() {
                        if ui.selectable_config(opt).selected(idx == wave_idx).build() {
                            wave_idx = idx;
                        }
                    }
                });
            let new_waveform = wave_idx.clamp(0, WAVEFORM_NAMES.len() - 1) as i32;
            if new_waveform != lfo.waveform {
                lfo.waveform = new_waveform;
                // Sync waveform to the LFO bank in shared state
                if lfo.bank_index >= 0 {
                    if let Ok(mut state) = self.shared_state.lock() {
                        if (lfo.bank_index as usize) < state.lfo_banks.len() {
                            state.lfo_banks[lfo.bank_index as usize].waveform = new_waveform;
                        }
                    }
                }
            }
            
            // Sync all LFO parameters to the bank in shared state
            if lfo.bank_index >= 0 {
                if let Ok(mut state) = self.shared_state.lock() {
                    if (lfo.bank_index as usize) < state.lfo_banks.len() {
                        let bank = &mut state.lfo_banks[lfo.bank_index as usize];
                        bank.rate = lfo.rate;
                        bank.tempo_sync = lfo.tempo_sync;
                        bank.division = lfo.division;
                        // waveform is already synced above
                    }
                }
            }
            
            // Show live LFO output visualization
            if lfo.bank_index >= 0 {
                let bank_index = lfo.bank_index;
                let waveform = lfo.waveform;
                let amplitude = lfo.amplitude;
                if let Ok(state) = self.shared_state.lock() {
                    if (bank_index as usize) < state.lfo_banks.len() {
                        let bank = &state.lfo_banks[bank_index as usize];
                        let lfo_value = Self::calculate_lfo_value(bank.phase, waveform);
                        let display_value = lfo_value * amplitude;
                        
                        ui.text("LFO Output:");
                        ui.same_line();
                        
                        // Draw a simple bar showing current LFO value (-1 to 1)
                        let bar_width = 100.0;
                        let center_x = ui.cursor_screen_pos()[0] + bar_width / 2.0;
                        let y = ui.cursor_screen_pos()[1];
                        
                        // Background bar
                        let draw_list = ui.get_window_draw_list();
                        draw_list.add_rect(
                            [center_x - bar_width/2.0, y],
                            [center_x + bar_width/2.0, y + 10.0],
                            [0.3, 0.3, 0.3, 1.0]
                        ).filled(true).build();
                        
                        // Value indicator
                        let value_x = center_x + (display_value * bar_width / 2.0);
                        draw_list.add_rect(
                            [center_x.min(value_x), y],
                            [center_x.max(value_x), y + 10.0],
                            if display_value > 0.0 { [0.0, 1.0, 0.0, 1.0] } else { [1.0, 0.0, 0.0, 1.0] }
                        ).filled(true).build();
                        
                        ui.dummy([bar_width + 10.0, 15.0]);
                        
                        // Show numeric value
                        ui.same_line();
                        ui.text(format!("{:.2}", display_value));
                    }
                }
            }
            
            ui.unindent();
        }
    }
    
    /// Calculate LFO value from phase and waveform
    fn calculate_lfo_value(phase: f32, waveform: i32) -> f32 {
        let phase = phase.fract();
        match waveform {
            0 => (phase * 2.0 * std::f32::consts::PI).sin(), // Sine
            1 => { // Triangle
                if phase < 0.5 {
                    (phase * 4.0) - 1.0
                } else {
                    3.0 - (phase * 4.0)
                }
            }
            2 => phase * 2.0 - 1.0, // Ramp
            3 => 1.0 - phase * 2.0, // Saw
            4 => { // Square
                if phase < 0.5 { 1.0 } else { -1.0 }
            }
            _ => (phase * 2.0 * std::f32::consts::PI).sin(),
        }
    }
    
    /// Draw global tempo control with tap tempo
    fn draw_tempo_control(&mut self, ui: &Ui) {
        // Extract config value first
        let show_osc = self.config.show_osc_addresses;
        
        ui.text("Global Tempo");
        
        // Helper closure for OSC tooltips
        let osc_tooltip = |ui: &Ui, address: &str, value: Option<f32>, show: bool| {
            if show && ui.is_item_hovered() {
                let mut tooltip = format!("OSC: {}", address);
                if let Some(val) = value {
                    tooltip.push_str(&format!("\nValue: {:.3}", val));
                }
                ui.tooltip_text(tooltip);
            }
        };
        
        // BPM display and edit
        ui.same_line_with_pos(120.0);
        let mut bpm = self.bpm;
        if Drag::new("BPM").speed(1.0).range(20.0, 300.0).build(ui, &mut bpm) {
            self.bpm = bpm;
            // Update shared state so engine can use it
            if let Ok(mut state) = self.shared_state.lock() {
                state.bpm = bpm;
            }
        }
        osc_tooltip(ui, "/global/bpm", Some(self.bpm), show_osc);
        
        // Tap tempo button
        ui.same_line();
        let button_label = if self.beat_flash > 0.0 {
            "TAP FLASH!"
        } else {
            "TAP TEMPO"
        };
        
        if ui.button(button_label) {
            self.handle_tap_tempo();
        }
        osc_tooltip(ui, "/global/tap_tempo", None, show_osc);
        
        // Play/Pause button
        ui.same_line();
        let play_label = if self.bpm_playing { "PAUSE" } else { "PLAY" };
        if ui.button(play_label) {
            self.bpm_playing = !self.bpm_playing;
        }
        
        // Sync enable
        ui.same_line();
        ui.checkbox("Sync", &mut self.bpm_enabled);
    }
    
    /// Handle tap tempo button press
    fn handle_tap_tempo(&mut self) {
        use std::time::{SystemTime, UNIX_EPOCH};
        
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();
        
        // Clear taps if it's been too long since last tap (2 seconds)
        if now - self.last_tap_time > 2.0 {
            self.tap_times.clear();
        }
        
        // Add tap time
        self.tap_times.push(now);
        
        // Keep only last 8 taps (more taps = more accurate average)
        if self.tap_times.len() > 8 {
            self.tap_times.remove(0);
        }
        
        // Update last tap time
        self.last_tap_time = now;
        
        // Reset all LFO phases on every tap (global sync)
        if let Ok(mut state) = self.shared_state.lock() {
            for lfo in &mut state.lfo_banks {
                lfo.phase = 0.0;
            }
        }
        
        // Calculate BPM from tap intervals (need at least 4 taps for accuracy)
        if self.tap_times.len() >= 4 {
            let mut intervals = Vec::new();
            for i in 1..self.tap_times.len() {
                intervals.push(self.tap_times[i] - self.tap_times[i-1]);
            }
            
            // Average interval
            let avg_interval: f64 = intervals.iter().sum::<f64>() / intervals.len() as f64;
            
            if avg_interval > 0.1 && avg_interval < 3.0 { // Reasonable range
                let new_bpm = (60.0 / avg_interval) as f32;
                self.bpm = new_bpm.clamp(40.0, 200.0);
                // Update shared state so engine can use it
                if let Ok(mut state) = self.shared_state.lock() {
                    state.bpm = self.bpm;
                }
            }
        }
        
        // Flash the button
        self.beat_flash = 0.2;
    }
    
    /// Public method to trigger tap tempo from external sources (e.g., keyboard shortcut)
    pub fn trigger_tap_tempo(&mut self) {
        self.handle_tap_tempo();
    }
    
    /// Show OSC address tooltip if the feature is enabled
    /// 
    /// # Arguments
    /// * `ui` - The ImGui UI context
    /// * `address` - The OSC address path (e.g., "/block1/ch1/x_displace")
    /// * `current_value` - Optional current parameter value to display
    pub fn show_osc_tooltip(&self, ui: &Ui, address: &str, current_value: Option<f32>) {
        if !self.config.show_osc_addresses {
            return;
        }
        
        if ui.is_item_hovered() {
            let mut tooltip = format!("OSC: {}", address);
            if let Some(val) = current_value {
                tooltip.push_str(&format!("\nValue: {:.3}", val));
            }
            ui.tooltip_text(tooltip);
        }
    }
    
    /// Generate OSC address for a Block 1 parameter
    pub fn get_osc_address_block1(param_id: &str) -> String {
        format!("/block1/{}", param_id.replace('_', "/").replace(".", "/"))
    }
    
    /// Generate OSC address for a Block 2 parameter
    pub fn get_osc_address_block2(param_id: &str) -> String {
        format!("/block2/{}", param_id.replace('_', "/").replace(".", "/"))
    }
    
    /// Generate OSC address for a Block 3 parameter
    pub fn get_osc_address_block3(param_id: &str) -> String {
        format!("/block3/{}", param_id.replace('_', "/").replace(".", "/"))
    }
    
    /// Check if MIDI learn mode is active and handle parameter click
    /// Returns true if the click was consumed (learn mode was active)
    pub fn handle_midi_learn_click(&mut self, param_id: &str, param_min: f32, param_max: f32) -> bool {
        if self.midi_learn_mode {
            // Start learning this parameter
            self.midi_learn_target = Some(param_id.to_string());
            if let Ok(mut state) = self.shared_state.lock() {
                state.midi.start_learning(param_id.to_string(), param_min, param_max);
            }
            self.show_status(&format!("Learning MIDI for: {}", 
                crate::midi::mapping::param_display_name(param_id)));
            log::info!("MIDI Learn: Started learning for parameter '{}'", param_id);
            true
        } else {
            false
        }
    }
    
    /// Check if a parameter is currently being learned (for visual feedback)
    pub fn is_learning_param(&self, param_id: &str) -> bool {
        self.midi_learn_mode && self.midi_learn_target.as_deref() == Some(param_id)
    }
    
    /// Check if MIDI learning has completed and show success message
    fn check_midi_learn_completion(&mut self) {
        if !self.midi_learn_mode {
            return;
        }
        
        // Collect information while holding the lock
        let learn_completed: bool;
        let target_name: Option<String>;
        let has_mapping: bool;
        
        {
            if let Ok(state) = self.shared_state.lock() {
                learn_completed = !state.midi.learn.is_active() && self.midi_learn_target.is_some();
                target_name = self.midi_learn_target.clone();
                has_mapping = target_name.as_ref()
                    .map(|t| state.midi.mappings.contains_key(t))
                    .unwrap_or(false);
            } else {
                return;
            }
        } // Lock dropped here
        
        // Now process the results without holding the lock
        if learn_completed {
            if let Some(ref target) = target_name {
                if has_mapping {
                    self.show_status(&format!("✅ MIDI mapped to: {}", 
                        crate::midi::mapping::param_display_name(target)));
                    log::info!("MIDI Learn: Successfully mapped parameter '{}'", target);
                }
            }
            // Reset learn mode
            self.midi_learn_mode = false;
            self.midi_learn_target = None;
        }
    }
    
    /// Build MIDI panel with learn mode and mapping management
    fn build_midi_panel(&mut self, ui: &Ui) {
        // Check if MIDI is enabled in config
        if !self.config.control.midi_enabled {
            ui.text_colored([1.0, 0.5, 0.0, 1.0], "⚠️ MIDI DISABLED IN CONFIG");
            ui.text_disabled("MIDI is disabled in config.toml to prevent conflicts with DAWs");
            ui.text_disabled("To enable MIDI, set midi_enabled = true in [control] section");
            ui.separator();
        }
        
        // MIDI Learn Mode Toggle
        let learn_active = self.midi_learn_mode;
        
        if learn_active {
            ui.text_colored([0.0, 1.0, 0.0, 1.0], "🎹 MIDI LEARN MODE ACTIVE");
            ui.same_line();
            if ui.button("Cancel Learn##midi") {
                self.midi_learn_mode = false;
                self.midi_learn_target = None;
                if let Ok(mut state) = self.shared_state.lock() {
                    state.midi.cancel_learning();
                }
                self.show_status("MIDI Learn cancelled");
            }
            
            // Show current target if any
            if let Some(ref target) = self.midi_learn_target {
                ui.text_colored([0.0, 1.0, 0.5, 1.0], &format!("Waiting for MIDI input for: {}", 
                    crate::midi::mapping::param_display_name(target)));
            } else {
                ui.text_disabled("Click any parameter (slider, checkbox, etc.) to select it for mapping");
            }
        } else {
            let button_color = if self.config.control.midi_enabled {
                [0.2, 0.8, 0.2, 1.0] // Green when MIDI enabled
            } else {
                [0.5, 0.5, 0.5, 1.0] // Gray when MIDI disabled
            };
            let _button_style = ui.push_style_color(imgui::StyleColor::Button, button_color);
            if ui.button("🎹 Enable MIDI Learn Mode") {
                if self.config.control.midi_enabled {
                    self.midi_learn_mode = true;
                    self.show_status("MIDI Learn enabled - click a parameter to map");
                } else {
                    self.show_status("MIDI is disabled in config.toml");
                }
            }
            drop(_button_style);
        }
        
        ui.separator();
        
        // MIDI Settings
        if CollapsingHeader::new("MIDI Settings").default_open(true).build(ui) {
            // Enable/disable MIDI
            let mut midi_enabled = if let Ok(state) = self.shared_state.lock() {
                state.midi.enabled
            } else {
                true
            };
            
            if ui.checkbox("Enable MIDI Input", &mut midi_enabled) {
                if let Ok(mut state) = self.shared_state.lock() {
                    state.midi.enabled = midi_enabled;
                }
            }
            
            // Channel filter
            let mut channel_filter = if let Ok(state) = self.shared_state.lock() {
                state.midi.channel_filter
            } else {
                0
            };
            
            let channel_preview = if channel_filter == 0 {
                "Omni (all channels)".to_string()
            } else {
                format!("Channel {}", channel_filter)
            };
            
            let mut channel_idx = channel_filter as usize;
            ComboBox::new(ui, "Channel Filter")
                .preview_value(&channel_preview)
                .build(|| {
                    if ui.selectable_config("Omni (all channels)").selected(channel_filter == 0).build() {
                        channel_idx = 0;
                    }
                    for ch in 1..=16 {
                        let label = format!("Channel {}", ch);
                        if ui.selectable_config(&label).selected(channel_filter == ch).build() {
                            channel_idx = ch as usize;
                        }
                    }
                });
            
            if channel_idx as u8 != channel_filter {
                if let Ok(mut state) = self.shared_state.lock() {
                    state.midi.channel_filter = channel_idx as u8;
                }
            }
            
            // High-resolution CC
            let mut high_res = if let Ok(state) = self.shared_state.lock() {
                state.midi.high_resolution_cc
            } else {
                true
            };
            
            if ui.checkbox("Enable 14-bit CC (High Resolution)", &mut high_res) {
                if let Ok(mut state) = self.shared_state.lock() {
                    state.midi.high_resolution_cc = high_res;
                }
            }
            if ui.is_item_hovered() {
                ui.tooltip_text("Enables 14-bit resolution for CC messages 0-31 (MSB) and 32-63 (LSB)");
            }
            
            // Last Note Priority for Channel Aftertouch
            let mut last_note_priority = if let Ok(state) = self.shared_state.lock() {
                state.midi.last_note_priority
            } else {
                true
            };
            
            if ui.checkbox("Last Note Priority (Aftertouch)", &mut last_note_priority) {
                if let Ok(mut state) = self.shared_state.lock() {
                    state.midi.last_note_priority = last_note_priority;
                }
            }
            if ui.is_item_hovered() {
                ui.tooltip_text("Routes Channel Aftertouch to the most recently played note's mapping. Allows per-pad aftertouch control on controllers that only send Channel Aftertouch.");
            }
        }
        
        ui.separator();
        
        // Connected Devices
        if CollapsingHeader::new("Connected Devices").default_open(true).build(ui) {
            let devices = if let Ok(state) = self.shared_state.lock() {
                state.midi.connected_devices.clone()
            } else {
                Vec::new()
            };
            
            if devices.is_empty() {
                ui.text_disabled("No MIDI devices connected");
            } else {
                ui.text(format!("Connected devices ({}):", devices.len()));
                for device in &devices {
                    ui.bullet_text(device);
                }
            }
            
            if ui.button("Refresh Devices") {
                // Trigger device scan
                self.show_status("Scanning for MIDI devices...");
            }
        }
        
        ui.separator();
        
        // Active Mappings
        if CollapsingHeader::new("Active Mappings").default_open(true).build(ui) {
            let mappings: Vec<(String, crate::midi::MidiMapping)> = if let Ok(state) = self.shared_state.lock() {
                state.midi.mappings.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
            } else {
                Vec::new()
            };
            
            if mappings.is_empty() {
                ui.text_disabled("No MIDI mappings configured");
                ui.text_disabled("Enable MIDI Learn and click a parameter to create mappings");
            } else {
                ui.text(format!("Active mappings ({}):", mappings.len()));
                
                // Create a scrollable area for mappings
                ui.child_window("mappings_list")
                    .size([0.0, 200.0])
                    .build(|| {
                        for (param_id, mapping) in &mappings {
                            ui.separator();
                            
                            // Delete button
                            if ui.small_button(&format!("X##del_{}", param_id)) {
                                if let Ok(mut state) = self.shared_state.lock() {
                                    state.midi.remove_mapping(param_id);
                                }
                            }
                            
                            ui.same_line();
                            
                            // Parameter name
                            let display_name = crate::midi::mapping::param_display_name(param_id);
                            ui.text_colored([0.0, 1.0, 0.5, 1.0], &display_name);
                            
                            ui.same_line_with_pos(250.0);
                            
                            // Mapping details
                            let note_details = match mapping.message_type {
                                MidiMessageType::ControlChange => format!("CC{} ({})", mapping.controller, cc_name(mapping.controller)),
                                MidiMessageType::NoteOn | MidiMessageType::NoteOff => format!("Note {}", mapping.controller),
                                MidiMessageType::PolyAftertouch => format!("PolyAT Note{}", mapping.controller),
                                MidiMessageType::ChannelAftertouch => "ChanAT".to_string(),
                                MidiMessageType::PitchBend => "PitchBend".to_string(),
                                _ => String::new(),
                            };
                            
                            ui.text(format!(
                                "{} Ch{} {}",
                                mapping.message_type.name(),
                                if mapping.channel == 0 { "Omni".to_string() } else { mapping.channel.to_string() },
                                note_details
                            ));
                            
                            // Range
                            ui.text_disabled(format!(
                                "  Range: {:.2} - {:.2}",
                                mapping.min_value, mapping.max_value
                            ));
                            
                            // Use Aftertouch toggle
                            let mut use_at = mapping.use_aftertouch;
                            if ui.checkbox(&format!("Use AT##useat_{}", param_id), &mut use_at) {
                                if let Ok(mut state) = self.shared_state.lock() {
                                    if let Some(m) = state.midi.mappings.get_mut(param_id) {
                                        m.use_aftertouch = use_at;
                                    }
                                }
                            }
                            if ui.is_item_hovered() {
                                ui.tooltip_text("Only respond to aftertouch for this mapping");
                            }
                        }
                    });
                
                ui.separator();
                
                if ui.button("Clear All Mappings") {
                    if let Ok(mut state) = self.shared_state.lock() {
                        state.midi.clear_mappings();
                    }
                    self.show_status("All MIDI mappings cleared");
                }
                
                ui.same_line();
                
                if ui.button("Save Mappings") {
                    let save_result = if let Ok(state) = self.shared_state.lock() {
                        state.midi.save_mappings("midi_mappings.toml")
                    } else {
                        Err(anyhow::anyhow!("Failed to lock state"))
                    };
                    match save_result {
                        Ok(_) => self.show_status("MIDI mappings saved"),
                        Err(e) => self.show_status(&format!("Failed to save: {}", e)),
                    }
                }
            }
        }
        
        ui.separator();
        
        // Last MIDI Message
        if CollapsingHeader::new("Last MIDI Message").default_open(false).build(ui) {
            let last_msg = if let Ok(state) = self.shared_state.lock() {
                state.midi.last_message.clone()
            } else {
                None
            };
            
            if let Some(event) = last_msg {
                ui.text(format!("Device: {}", event.device_id));
                ui.text(format!("Message: {:?}", event.message));
                ui.text(format!("Channel: {}", event.message.channel()));
                ui.text(format!("Value: {}", event.message.value()));
            } else {
                ui.text_disabled("No MIDI message received yet");
            }
        }
        
        ui.separator();
        
        // Quick Map Section - Direct mapping without learn mode
        if CollapsingHeader::new("Quick Map").default_open(false).build(ui) {
            ui.text("Quickly map a parameter:");
            ui.text_disabled("Select a parameter and MIDI control manually");
            
            // Parameter selector
            let learnable = LearnableParams::all();
            let param_names: Vec<String> = learnable.params().iter().map(|p| p.display_name.clone()).collect();
            
            static mut SELECTED_PARAM: usize = 0;
            let selected = unsafe { SELECTED_PARAM };
            
            let preview = &param_names[selected.min(param_names.len() - 1)];
            let mut new_selected = selected;
            
            ComboBox::new(ui, "Parameter")
                .preview_value(preview)
                .build(|| {
                    for (idx, name) in param_names.iter().enumerate() {
                        if ui.selectable_config(name).selected(idx == selected).build() {
                            new_selected = idx;
                        }
                    }
                });
            
            unsafe { SELECTED_PARAM = new_selected; }
            
            if let Some(param) = learnable.params().get(new_selected) {
                ui.text_disabled(&param.param_id);
                
                // MIDI Control selector
                let cc_numbers: Vec<u8> = (0..128).collect();
                let mut selected_cc: usize = 16; // Default to CC16 (General Purpose 1)
                
                let cc_preview = format!("CC{} - {}", selected_cc, cc_name(selected_cc as u8));
                ComboBox::new(ui, "MIDI CC")
                    .preview_value(&cc_preview)
                    .build(|| {
                        for cc in 0..128u8 {
                            let name = cc_name(cc);
                            let label = format!("CC{:3} - {}", cc, name);
                            if ui.selectable_config(&label).selected(cc as usize == selected_cc).build() {
                                selected_cc = cc as usize;
                            }
                        }
                    });
                
                if ui.button("Create Mapping") {
                    let mapping = MidiMapping::new(
                        param.param_id.clone(),
                        MidiMessageType::ControlChange,
                        0, // Omni channel
                        selected_cc as u8,
                    );
                    
                    if let Ok(mut state) = self.shared_state.lock() {
                        state.midi.add_mapping(param.param_id.clone(), mapping);
                    }
                    self.show_status(&format!("Mapped {} to CC{}", param.display_name, selected_cc));
                }
            }
        }
    }

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

}

/// Get list of Block 1 parameter names for audio modulation
pub fn get_block1_param_names() -> Vec<String> {
    vec![
        "ch1_x_displace".to_string(),
        "ch1_y_displace".to_string(),
        "ch1_z_displace".to_string(),
        "ch1_rotate".to_string(),
        "ch1_hsb_attenuate.x".to_string(),
        "ch1_hsb_attenuate.y".to_string(),
        "ch1_hsb_attenuate.z".to_string(),
        "ch1_kaleidoscope_amount".to_string(),
        "ch1_blur_amount".to_string(),
        "ch2_mix_amount".to_string(),
        "ch2_x_displace".to_string(),
        "ch2_y_displace".to_string(),
        "ch2_rotate".to_string(),
        "fb1_mix_amount".to_string(),
        "fb1_x_displace".to_string(),
        "fb1_y_displace".to_string(),
        "fb1_rotate".to_string(),
    ]
}

/// Get list of Block 2 parameter names for audio modulation
pub fn get_block2_param_names() -> Vec<String> {
    vec![
        "block2_input_x_displace".to_string(),
        "block2_input_y_displace".to_string(),
        "block2_input_rotate".to_string(),
        "block2_input_blur_amount".to_string(),
        "fb2_mix_amount".to_string(),
        "fb2_x_displace".to_string(),
        "fb2_y_displace".to_string(),
        "fb2_rotate".to_string(),
    ]
}

/// Get list of Block 3 parameter names for audio modulation
pub fn get_block3_param_names() -> Vec<String> {
    vec![
        "block1_x_displace".to_string(),
        "block1_y_displace".to_string(),
        "block1_rotate".to_string(),
        "block2_x_displace".to_string(),
        "block2_y_displace".to_string(),
        "block2_rotate".to_string(),
        "final_mix_amount".to_string(),
    ]
}
