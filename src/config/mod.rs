//! # Configuration Module
//!
//! Handles application configuration including:
//! - Window settings (resolution, position)
//! - Shader pipeline configuration
//! - Input/output settings
//! - Default parameter values
//! - Floating window layout for popped-out tabs

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

pub mod layout;
pub mod resolution;

// Re-export layout types
pub use layout::{LayoutConfig, LayoutManager, TabId};
// Re-export resolution types
pub use resolution::{ResolutionConfig, ResolutionPreset, ResolutionSettings};

/// Main application configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    /// Output window configuration
    pub output_window: WindowConfig,
    /// Control window configuration
    pub control_window: WindowConfig,
    /// Shader pipeline settings
    pub pipeline: PipelineConfig,
    /// Input source settings
    pub inputs: InputConfig,
    /// Audio input settings
    pub audio: AudioConfig,
    /// NDI output settings
    pub ndi: NdiConfig,
    /// Syphon output settings (macOS only)
    pub syphon: SyphonConfig,
    /// OSC/MIDI settings
    pub control: ControlConfig,
    /// UI scaling factor (0.5 - 2.0)
    pub ui_scale: f32,
    /// Show OSC addresses on parameter hover
    pub show_osc_addresses: bool,
    /// Resolution settings (input, internal, output)
    pub resolution: ResolutionSettings,
}

/// Window configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowConfig {
    /// Window width in pixels
    pub width: u32,
    /// Window height in pixels
    pub height: u32,
    /// X position (None for default)
    pub x: Option<i32>,
    /// Y position (None for default)
    pub y: Option<i32>,
    /// Window title
    pub title: String,
    /// Whether window is resizable
    pub resizable: bool,
    /// Whether window has decorations
    pub decorated: bool,
    /// Target frame rate
    pub fps: u32,
    /// Enable VSync
    pub vsync: bool,
}

/// Pipeline configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Internal processing resolution width
    pub internal_width: u32,
    /// Internal processing resolution height
    pub internal_height: u32,
    /// Maximum feedback delay frames
    pub max_delay_frames: usize,
    /// Enable temporal filtering
    pub temporal_filter: bool,
    /// Default mix mode (0=lerp, 1=add, 2=diff, 3=mult, 4=dodge)
    pub default_mix_mode: i32,
}

/// Input source configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputConfig {
    /// Primary input device ID (webcam index)
    pub input1_device: i32,
    /// Secondary input device ID (webcam index)
    pub input2_device: i32,
    /// Input source type (0=None, 1=Webcam, 2=NDI, 3=Syphon, 4=Spout, 5=Video File)
    pub input1_type: i32,
    pub input2_type: i32,
    /// NDI source names (selected source for each input)
    pub input1_ndi_source: String,
    pub input2_ndi_source: String,
    /// Syphon source names (selected source for each input)
    pub input1_syphon_source: String,
    pub input2_syphon_source: String,
    /// Legacy field - kept for backward compatibility (deprecated)
    pub ndi_sources: Vec<String>,
    /// Input resolution
    pub input_width: u32,
    pub input_height: u32,
    /// Whether to auto-start inputs on startup (webcam, ndi, syphon)
    pub auto_start_webcams: bool,
}

/// NDI output configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NdiConfig {
    /// Enable NDI output
    pub enabled: bool,
    /// NDI output width
    pub width: u32,
    /// NDI output height
    pub height: u32,
    /// Frame rate
    pub fps: u32,
    /// NDI output source name
    pub output_name: String,
    /// Include alpha channel in NDI output
    pub output_alpha: bool,
    /// Frame skip factor (send every Nth frame, 1=no skip, 2=30fps, 3=20fps, etc)
    pub frame_skip: u8,
}

/// Syphon output configuration (macOS only)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyphonConfig {
    /// Enable Syphon output
    pub enabled: bool,
    /// Syphon server name
    pub server_name: String,
}

impl Default for SyphonConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            server_name: "RustJay Waaaves".to_string(),
        }
    }
}

/// Audio input configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioConfig {
    /// Enable audio input
    pub enable_input: bool,
    /// Audio device index (None for default)
    pub device_index: Option<usize>,
    /// FFT size (power of 2: 256, 512, 1024, 2048)
    pub fft_size: usize,
    /// Number of FFT bands for visualization
    pub fft_bands: usize,
    /// Beat detection sensitivity (0.0 - 2.0)
    pub beat_sensitivity: f32,
}

/// Control interface configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlConfig {
    /// OSC receive port
    pub osc_receive_port: u16,
    /// OSC send port
    pub osc_send_port: u16,
    /// OSC send IP
    pub osc_send_ip: String,
    /// Enable OSC
    pub osc_enabled: bool,
    /// MIDI input device
    pub midi_device: Option<String>,
    /// Enable MIDI input
    pub midi_enabled: bool,
    /// Presets directory
    pub presets_dir: PathBuf,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            output_window: WindowConfig {
                width: 1280,
                height: 720,
                x: Some(100),
                y: Some(100),
                title: "RustJay Waaaves - Output".to_string(),
                resizable: false,
                decorated: true,
                fps: 60,
                vsync: true,
            },
            control_window: WindowConfig {
                width: 1920,
                height: 1080,
                x: None,
                y: None,
                title: "RustJay Waaaves - Control".to_string(),
                resizable: true,
                decorated: true,
                fps: 30,
                vsync: true,
            },
            pipeline: PipelineConfig {
                internal_width: 1280,
                internal_height: 720,
                max_delay_frames: 120,
                temporal_filter: true,
                default_mix_mode: 0,
            },
            inputs: InputConfig {
                input1_device: 0,      // Default to first webcam
                input2_device: 1,      // Default to second webcam
                input1_type: 1,        // Default to Webcam (1)
                input2_type: 0,        // Default to None (0)
                input1_ndi_source: String::new(),  // No default NDI source
                input2_ndi_source: String::new(),
                input1_syphon_source: String::new(), // No default Syphon source
                input2_syphon_source: String::new(),
                ndi_sources: vec![],   // Legacy field
                input_width: 640,
                input_height: 480,
                auto_start_webcams: true, // Auto-start on launch
            },
            audio: AudioConfig {
                enable_input: true,
                device_index: None,
                fft_size: 1024,
                fft_bands: 16,
                beat_sensitivity: 1.0,
            },
            ndi: NdiConfig {
                enabled: false,
                width: 1280,
                height: 720,
                fps: 30,
                output_name: "RustJay Waaaves".to_string(),
                output_alpha: false,
                frame_skip: 2, // Default to 30fps output (skip every other frame)
            },
            syphon: SyphonConfig::default(),
            control: ControlConfig {
                osc_receive_port: 7000,
                osc_send_port: 7001,
                osc_send_ip: "127.0.0.1".to_string(),
                osc_enabled: false,
                midi_device: None,
                midi_enabled: true, // MIDI enabled by default, but can be disabled if DAW is running
                presets_dir: PathBuf::from("presets"),
            },
            ui_scale: 1.0,
            show_osc_addresses: false,
            resolution: ResolutionSettings::default(),
        }
    }
}

impl AppConfig {
    /// Resolve the config file path:
    /// 1. `~/Library/Application Support/rustjay/waaaves.toml` (macOS) via `dirs::config_dir()`
    /// 2. Falls back to `./config.toml` in the working directory
    fn config_path() -> PathBuf {
        if let Some(config_dir) = dirs::config_dir() {
            let app_dir = config_dir.join("rustjay");
            let _ = std::fs::create_dir_all(&app_dir);
            // Migrate legacy CWD config on first run
            let legacy = PathBuf::from("config.toml");
            let target = app_dir.join("waaaves.toml");
            if legacy.exists() && !target.exists() {
                log::info!("[Config] Migrating config.toml → {}", target.display());
                let _ = std::fs::copy(&legacy, &target);
            }
            target
        } else {
            PathBuf::from("config.toml")
        }
    }

    /// Load configuration from the OS config directory, or return defaults.
    pub fn load_or_default() -> Self {
        let config_path = Self::config_path();

        if config_path.exists() {
            match std::fs::read_to_string(&config_path) {
                Ok(contents) => {
                    match toml::from_str(&contents) {
                        Ok(config) => {
                            log::info!("[Config] Loaded from {}", config_path.display());
                            return config;
                        }
                        Err(e) => {
                            log::warn!(
                                "[Config] Failed to parse config at {}: {}. Preserving existing file and using defaults in-memory",
                                config_path.display(),
                                e
                            );
                            return Self::default();
                        }
                    }
                }
                Err(e) => {
                    log::warn!(
                        "[Config] Failed to read config at {}: {}. Preserving existing file and using defaults in-memory",
                        config_path.display(),
                        e
                    );
                    return Self::default();
                }
            }
        }

        let config = Self::default();
        // Save defaults so the file exists for next launch
        if let Ok(toml) = toml::to_string_pretty(&config) {
            if let Err(e) = std::fs::write(&config_path, toml) {
                log::warn!("[Config] Failed to write default config to {}: {}", config_path.display(), e);
            }
        }
        config
    }

    /// Atomically save configuration to the OS config directory.
    pub fn save(&self) -> anyhow::Result<()> {
        let config_path = Self::config_path();
        let toml = toml::to_string_pretty(self)?;
        // Write to a temp file then rename for atomicity
        let tmp_path = config_path.with_extension("toml.tmp");
        std::fs::write(&tmp_path, &toml)?;
        std::fs::rename(&tmp_path, &config_path)?;
        Ok(())
    }
}
