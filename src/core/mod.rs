//! # Core Module
//!
//! Contains shared state and core types used throughout the application.

use crate::config::AppConfig;
use crate::midi::MidiState;
use crate::params::{Block1Params, Block2Params, Block3Params, LfoBank, ParamModulationData};
use std::collections::HashMap;

pub mod lfo_engine;

/// Preview source selection (defined here to avoid circular deps)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PreviewSource {
    #[default]
    Block1,
    Block2,
    Block3, // Final output
    Input1,
    Input2,
}

impl PreviewSource {
    pub fn display_name(&self) -> &'static str {
        match self {
            PreviewSource::Block1 => "Block 1",
            PreviewSource::Block2 => "Block 2",
            PreviewSource::Block3 => "Block 3 (Final)",
            PreviewSource::Input1 => "Input 1",
            PreviewSource::Input2 => "Input 2",
        }
    }
}

/// Input change request
#[derive(Debug, Clone, PartialEq)]
pub enum InputChangeRequest {
    None,
    StartWebcam { input_id: u8, device_index: usize, width: u32, height: u32, fps: u32 },
    StartNdi { input_id: u8, source_name: String },
    StartSyphon { input_id: u8, server_name: String },
    StopInput { input_id: u8 },
    /// Set output window VSync
    SetVsync(bool),
    /// Set output window target FPS
    SetOutputFps(u32),
}

/// Unified output command
#[derive(Debug, Clone, PartialEq)]
pub enum OutputCommand {
    None,
    StartNdi { name: String, include_alpha: bool, frame_skip: u8 },
    StopNdi,
    StartSyphon { name: String },
    StopSyphon,
}

/// Audio change request
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AudioChangeRequest {
    None,
    ChangeDevice { device_index: i32 },
}

/// Output display mode - which block to show
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum OutputMode {
    Block1,
    Block2,
    #[default]
    Block3,
    PreviewInput1,
    PreviewInput2,
}

/// LFO assignment for a single parameter
#[derive(Debug, Clone, Copy, Default)]
pub struct LfoAssignment {
    /// Which LFO bank (0-15) to use, or -1 for none
    pub bank_index: i32,
    /// Amplitude/scaling of the modulation
    pub amplitude: f32,
    /// Whether this assignment is active
    pub enabled: bool,
}

/// LFO parameter mappings for a block
pub type LfoParameterMap = HashMap<String, LfoAssignment>;

/// Recording command from GUI to engine
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RecordingCommand {
    #[default]
    None,
    Start,
    Stop,
    Toggle,
}

/// Video codec options
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VideoCodec {
    #[default]
    H264,       // H.264 (AVC) - good compatibility
    H265,       // H.265 (HEVC) - better compression
    ProRes,     // Apple ProRes - professional editing
    VP9,        // VP9 - web optimized
    AV1,        // AV1 - next gen compression
}

/// Recording quality preset
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RecordingQuality {
    #[default]
    High,       // High quality, larger file
    Medium,     // Balanced
    Low,        // Smaller file, lower quality
    Lossless,   // Uncompressed (very large)
}

/// Recording settings
#[derive(Debug, Clone)]
pub struct RecordingSettings {
    /// Video codec to use
    pub codec: VideoCodec,
    /// Quality preset
    pub quality: RecordingQuality,
    /// Output filename (without extension)
    pub filename: String,
    /// Include audio in recording
    pub include_audio: bool,
    /// Frame rate (usually matches display refresh)
    pub fps: u32,
}

impl Default for RecordingSettings {
    fn default() -> Self {
        Self {
            codec: VideoCodec::H264,
            quality: RecordingQuality::High,
            filename: String::from("output"),
            include_audio: true,
            fps: 60,
        }
    }
}

/// Shared application state between windows
/// 
/// This struct is wrapped in an Arc<Mutex<>> and shared between
/// the output window (engine) and control window (GUI).
#[derive(Debug)]
pub struct SharedState {
    /// Block 1 parameters (channel mixing)
    pub block1: Block1Params,
    /// Block 2 parameters (secondary processing)
    pub block2: Block2Params,
    /// Block 3 parameters (final mixing)
    pub block3: Block3Params,
    /// LFO banks for modulation
    pub lfo_banks: Vec<LfoBank>,
    /// LFO to parameter mappings for Block 1
    pub block1_lfo_map: LfoParameterMap,
    /// LFO to parameter mappings for Block 2
    pub block2_lfo_map: LfoParameterMap,
    /// LFO to parameter mappings for Block 3
    pub block3_lfo_map: LfoParameterMap,
    /// Audio analysis data
    pub audio: AudioState,
    /// Current frame number
    pub frame_count: u64,
    /// Whether to clear feedback buffers
    pub clear_feedback: bool,
    /// Output resolution
    pub output_size: (u32, u32),
    /// Internal processing resolution
    pub internal_size: (u32, u32),
    /// Recording state
    pub is_recording: bool,
    /// Recording command from GUI (Start/Stop/Toggle)
    pub recording_command: RecordingCommand,
    /// Recording settings (codec, quality, etc.)
    pub recording_settings: RecordingSettings,
    /// Input 1 change request (GUI -> Engine)
    pub input1_change_request: InputChangeRequest,
    /// Input 2 change request (GUI -> Engine)
    pub input2_change_request: InputChangeRequest,
    /// Audio change request (GUI -> Engine)
    pub audio_change_request: AudioChangeRequest,
    /// Output command (GUI -> Engine)
    pub output_command: OutputCommand,
    /// NDI output status (Engine -> GUI)
    pub ndi_output_active: bool,
    /// Syphon output status (Engine -> GUI)
    pub syphon_output_active: bool,
    /// Output display mode (which block to show)
    pub output_mode: OutputMode,
    /// Global BPM for tempo-synced LFOs
    pub bpm: f32,
    /// Active audio/BPM modulations for Block 1
    pub block1_modulations: HashMap<String, ParamModulationData>,
    /// Active audio/BPM modulations for Block 2
    pub block2_modulations: HashMap<String, ParamModulationData>,
    /// Active audio/BPM modulations for Block 3
    pub block3_modulations: HashMap<String, ParamModulationData>,
    /// UI scale for ImGui (1.0 = 100%, 2.0 = 200%, etc.)
    /// This allows GUI to request scale changes that the engine applies
    pub ui_scale: f32,
    /// Preview window state - sampled color from preview (RGB 0-1)
    pub preview_sampled_color: [f32; 3],
    /// Preview source selection (which block/input to preview)
    pub preview_source: PreviewSource,
    /// Mouse pick UV coordinates [u, v] where user clicked on preview (0-1 range)
    pub preview_pick_uv: [f32; 2],
    /// Flag set by GUI when a color pick is requested
    pub preview_pick_requested: bool,
    /// Whether preview window is open (if false, engine skips preview computation)
    pub preview_enabled: bool,
    /// Output window target FPS (for GUI display)
    pub output_fps: u32,
    /// Output window actual measured FPS (engine → GUI)
    pub output_actual_fps: f32,
    /// Output window VSync enabled (for GUI display)
    pub output_vsync: bool,
    /// MIDI state for parameter mapping and learn
    pub midi: MidiState,
}

/// Audio analysis state
#[derive(Debug, Clone)]
pub struct AudioState {
    /// FFT data (frequency bands)
    pub fft: Vec<f32>,
    /// Overall volume
    pub volume: f32,
    /// Beat detection
    pub beat: bool,
    /// BPM estimate
    pub bpm: f32,
    /// Beat phase (0-1)
    pub beat_phase: f32,
    /// Amplitude multiplier (0-10x)
    pub amplitude: f32,
    /// Smoothing factor (0-1)
    pub smoothing: f32,
    /// Normalization enabled
    pub normalization: bool,
    /// Pink noise compensation (makes pink noise appear flat)
    pub pink_compensation: bool,
}

impl SharedState {
    /// Create new shared state from configuration
    pub fn new(config: &AppConfig) -> Self {
        // Get dimensions from the new resolution config (with fallback to legacy config)
        let (internal_width, internal_height) = config.resolution.internal.dimensions();
        let (output_width, output_height) = config.resolution.output.dimensions();
        
        log::info!("SharedState::new() - Resolution config:");
        log::info!("  Internal: {}x{} (preset: {:?})", internal_width, internal_height, config.resolution.internal.preset);
        log::info!("  Output: {}x{} (preset: {:?})", output_width, output_height, config.resolution.output.preset);
        log::info!("  Input: {:?} (preset: {:?})", config.resolution.input.dimensions(), config.resolution.input.preset);
        
        Self {
            block1: Block1Params::default(),
            block2: Block2Params::default(),
            block3: Block3Params::default(),
            lfo_banks: vec![LfoBank::default(); 16], // 16 macro banks
            block1_lfo_map: HashMap::new(),
            block2_lfo_map: HashMap::new(),
            block3_lfo_map: HashMap::new(),
            audio: AudioState::default(),
            frame_count: 0,
            clear_feedback: false,
            output_size: (output_width, output_height),
            internal_size: (internal_width, internal_height),
            is_recording: false,
            recording_command: RecordingCommand::None,
            recording_settings: RecordingSettings::default(),
            input1_change_request: InputChangeRequest::None,
            input2_change_request: InputChangeRequest::None,
            audio_change_request: AudioChangeRequest::None,
            output_command: OutputCommand::None,
            ndi_output_active: false,
            syphon_output_active: false,
            output_mode: OutputMode::default(),
            bpm: 120.0,
            block1_modulations: HashMap::new(),
            block2_modulations: HashMap::new(),
            block3_modulations: HashMap::new(),
            ui_scale: config.ui_scale,
            preview_sampled_color: [1.0, 1.0, 1.0], // Default white
            preview_source: PreviewSource::Block3,   // Default to final output
            preview_pick_uv: [0.5, 0.5],             // Default to center
            preview_pick_requested: false,
            preview_enabled: true,                   // Preview starts enabled
            output_fps: config.output_window.fps,    // Initial FPS from config
            output_actual_fps: 0.0,
            output_vsync: config.output_window.vsync, // Initial VSync from config
            midi: {
                let mut midi = MidiState::new();
                // Try to load existing MIDI mappings
                if let Err(e) = midi.load_mappings("midi_mappings.toml") {
                    log::warn!("Failed to load MIDI mappings: {}", e);
                }
                midi
            },
        }
    }
}

impl Default for AudioState {
    fn default() -> Self {
        Self {
            fft: vec![0.0; 16], // 16 frequency bands
            volume: 0.0,
            beat: false,
            bpm: 120.0,
            beat_phase: 0.0,
            amplitude: 1.0,      // Default: 1x amplitude
            smoothing: 0.7,      // Default: 70% smoothing
            normalization: false, // Default: off
            pink_compensation: false, // Default: off
        }
    }
}

/// Vertex data for full-screen quad
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 2],
    pub texcoord: [f32; 2],
}

impl Vertex {
    /// Create a vertex buffer for a full-screen quad
    pub fn quad_vertices() -> Vec<Vertex> {
        vec![
            // Triangle 1
            Vertex { position: [-1.0, -1.0], texcoord: [0.0, 1.0] },  // Bottom-left (V=1)
            Vertex { position: [ 1.0, -1.0], texcoord: [1.0, 1.0] },  // Bottom-right (V=1)
            Vertex { position: [-1.0,  1.0], texcoord: [0.0, 0.0] },  // Top-left (V=0)
            // Triangle 2
            Vertex { position: [-1.0,  1.0], texcoord: [0.0, 0.0] },  // Top-left (V=0)
            Vertex { position: [ 1.0, -1.0], texcoord: [1.0, 1.0] },  // Bottom-right (V=1)
            Vertex { position: [ 1.0,  1.0], texcoord: [1.0, 0.0] },  // Top-right (V=0)
        ]
    }
    
    /// Vertex buffer layout for wgpu
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                // Position at location 0
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                },
                // Texcoord at location 1
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        }
    }
}
