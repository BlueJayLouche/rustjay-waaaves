//! # Preset Management
//!
//! Handles saving and loading of preset files, including all parameter values
//! and audio/BPM modulation assignments. Ported from the original OF app's
//! PresetManager.

use crate::midi::MidiMapping;
use crate::params::{Block1Params, Block2Params, Block3Params, LfoBank};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Audio/BPM modulation data for a single parameter
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub struct ParamModulationData {
    // Audio modulation
    pub audio_enabled: bool,
    pub audio_fft_band: i32,
    pub audio_amount: f32,
    pub audio_use_normalization: bool,
    pub audio_attack: f32,
    pub audio_release: f32,
    pub audio_range_scale: f32,

    // Runtime envelope state (not serialized)
    #[serde(skip)]
    pub audio_smoothed_value: f32,

    // BPM modulation
    pub bpm_enabled: bool,
    pub bpm_division_index: i32,
    pub bpm_phase: f32,
    pub bpm_waveform: i32,
    pub bpm_min_value: f32,
    pub bpm_max_value: f32,
    pub bpm_bipolar: bool,
}

/// Audio settings for presets
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PresetAudioSettings {
    pub amplitude: f32,
    pub smoothing: f32,
    pub normalization: bool,
    pub pink_compensation: bool,
}

impl Default for PresetAudioSettings {
    fn default() -> Self {
        Self {
            amplitude: 1.0,
            smoothing: 0.7,
            normalization: false,
            pink_compensation: false,
        }
    }
}

/// Tempo settings for presets
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PresetTempoData {
    pub bpm: f32,
    pub enabled: bool,
}

impl Default for PresetTempoData {
    fn default() -> Self {
        Self {
            bpm: 120.0,
            enabled: true,
        }
    }
}

/// Complete preset data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PresetData {
    // Block parameters
    pub block1: Block1Params,
    pub block2: Block2Params,
    pub block3: Block3Params,
    
    // Audio/BPM modulations for all blocks (key = parameter name)
    // Backward compatible - uses empty HashMap if missing
    #[serde(default)]
    pub block1_modulations: HashMap<String, ParamModulationData>,
    #[serde(default)]
    pub block2_modulations: HashMap<String, ParamModulationData>,
    #[serde(default)]
    pub block3_modulations: HashMap<String, ParamModulationData>,
    
    // Audio processing settings (backward compatible - uses default if missing)
    #[serde(default)]
    pub audio: PresetAudioSettings,
    
    // Tempo settings (backward compatible - uses default if missing)
    #[serde(default)]
    pub tempo: PresetTempoData,
    
    // LFO banks (16 banks, backward compatible - uses default if missing)
    #[serde(default = "default_lfo_banks")]
    pub lfo_banks: Vec<LfoBank>,
    
    // MIDI mappings (backward compatible - uses empty Vec if missing)
    #[serde(default)]
    pub midi_mappings: Vec<MidiMapping>,
    
    // Metadata
    pub version: String,
    pub name: String,
}

/// Default LFO banks (16 banks with default values)
pub fn default_lfo_banks() -> Vec<LfoBank> {
    vec![
        LfoBank { rate: 0.15, amplitude: 1.0, waveform: 0, tempo_sync: false, division: 2, phase: 0.0 },
        LfoBank { rate: 0.15, amplitude: 1.0, waveform: 0, tempo_sync: false, division: 2, phase: 0.0 },
        LfoBank { rate: 0.15, amplitude: 1.0, waveform: 0, tempo_sync: false, division: 2, phase: 0.0 },
        LfoBank { rate: 0.15, amplitude: 1.0, waveform: 0, tempo_sync: false, division: 2, phase: 0.0 },
        LfoBank { rate: 0.15, amplitude: 1.0, waveform: 0, tempo_sync: false, division: 2, phase: 0.0 },
        LfoBank { rate: 0.15, amplitude: 1.0, waveform: 0, tempo_sync: false, division: 2, phase: 0.0 },
        LfoBank { rate: 0.15, amplitude: 1.0, waveform: 0, tempo_sync: false, division: 2, phase: 0.0 },
        LfoBank { rate: 0.15, amplitude: 1.0, waveform: 0, tempo_sync: false, division: 2, phase: 0.0 },
        LfoBank { rate: 0.15, amplitude: 1.0, waveform: 0, tempo_sync: false, division: 2, phase: 0.0 },
        LfoBank { rate: 0.15, amplitude: 1.0, waveform: 0, tempo_sync: false, division: 2, phase: 0.0 },
        LfoBank { rate: 0.15, amplitude: 1.0, waveform: 0, tempo_sync: false, division: 2, phase: 0.0 },
        LfoBank { rate: 0.15, amplitude: 1.0, waveform: 0, tempo_sync: false, division: 2, phase: 0.0 },
        LfoBank { rate: 0.15, amplitude: 1.0, waveform: 0, tempo_sync: false, division: 2, phase: 0.0 },
        LfoBank { rate: 0.15, amplitude: 1.0, waveform: 0, tempo_sync: false, division: 2, phase: 0.0 },
        LfoBank { rate: 0.15, amplitude: 1.0, waveform: 0, tempo_sync: false, division: 2, phase: 0.0 },
        LfoBank { rate: 0.15, amplitude: 1.0, waveform: 0, tempo_sync: false, division: 2, phase: 0.0 },
    ]
}

impl Default for PresetData {
    fn default() -> Self {
        Self {
            block1: Block1Params::default(),
            block2: Block2Params::default(),
            block3: Block3Params::default(),
            block1_modulations: HashMap::new(),
            block2_modulations: HashMap::new(),
            block3_modulations: HashMap::new(),
            audio: PresetAudioSettings::default(),
            tempo: PresetTempoData::default(),
            lfo_banks: default_lfo_banks(),
            midi_mappings: Vec::new(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            name: "Untitled".to_string(),
        }
    }
}

/// Preset bank information
#[derive(Debug, Clone)]
pub struct PresetBank {
    pub name: String,
    pub path: PathBuf,
    pub preset_files: Vec<String>,
    pub preset_display_names: Vec<String>,
}

/// Centralized preset management
pub struct PresetManager {
    banks: HashMap<String, PresetBank>,
    current_bank: String,
    base_path: PathBuf,
}

impl PresetManager {
    /// Create a new preset manager
    pub fn new() -> Self {
        let base_path = PathBuf::from("presets");
        
        // Ensure presets directory exists
        if !base_path.exists() {
            let _ = std::fs::create_dir_all(&base_path);
        }
        
        let mut manager = Self {
            banks: HashMap::new(),
            current_bank: "Default".to_string(),
            base_path,
        };
        
        // Scan for banks and ensure Default exists
        manager.scan_banks();
        
        manager
    }
    
    /// Scan for available preset banks
    pub fn scan_banks(&mut self) {
        self.banks.clear();
        
        if let Ok(entries) = std::fs::read_dir(&self.base_path) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    let name = path.file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("Unknown")
                        .to_string();
                    
                    let mut bank = PresetBank {
                        name: name.clone(),
                        path: path.clone(),
                        preset_files: Vec::new(),
                        preset_display_names: Vec::new(),
                    };
                    
                    // Index presets in this bank
                    Self::index_presets_in_bank(&mut bank);
                    
                    self.banks.insert(name, bank);
                }
            }
        }
        
        // Ensure Default bank exists
        if !self.banks.contains_key("Default") {
            let default_path = self.base_path.join("Default");
            let _ = std::fs::create_dir_all(&default_path);
            
            let bank = PresetBank {
                name: "Default".to_string(),
                path: default_path,
                preset_files: Vec::new(),
                preset_display_names: Vec::new(),
            };
            
            self.banks.insert("Default".to_string(), bank);
        }
        
        log::info!("Scanned {} preset banks", self.banks.len());
    }
    
    /// Get list of bank names
    pub fn get_bank_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.banks.keys().cloned().collect();
        names.sort();
        names
    }
    
    /// Switch to a different bank
    pub fn switch_bank(&mut self, bank_name: &str) -> bool {
        if self.banks.contains_key(bank_name) {
            self.current_bank = bank_name.to_string();
            
            // Re-index presets
            if let Some(bank) = self.banks.get_mut(bank_name) {
                Self::index_presets_in_bank(bank);
            }
            
            log::info!("Switched to preset bank: {}", bank_name);
            true
        } else {
            log::warn!("Preset bank not found: {}", bank_name);
            false
        }
    }
    
    /// Create a new bank
    pub fn create_bank(&mut self, name: &str) -> bool {
        let path = self.base_path.join(name);
        
        if path.exists() {
            log::warn!("Bank '{}' already exists", name);
            return false;
        }
        
        match std::fs::create_dir_all(&path) {
            Ok(_) => {
                let bank = PresetBank {
                    name: name.to_string(),
                    path,
                    preset_files: Vec::new(),
                    preset_display_names: Vec::new(),
                };
                
                self.banks.insert(name.to_string(), bank);
                log::info!("Created preset bank: {}", name);
                true
            }
            Err(e) => {
                log::error!("Failed to create bank '{}': {:?}", name, e);
                false
            }
        }
    }
    
    /// Get current bank name
    pub fn get_current_bank(&self) -> &str {
        &self.current_bank
    }
    
    /// Get preset names in current bank
    pub fn get_preset_names(&self) -> Vec<String> {
        if let Some(bank) = self.banks.get(&self.current_bank) {
            bank.preset_display_names.clone()
        } else {
            Vec::new()
        }
    }
    
    /// Save a preset
    pub fn save_preset(
        &mut self,
        name: &str,
        data: &PresetData,
    ) -> anyhow::Result<()> {
        let bank = self.banks.get(&self.current_bank)
            .ok_or_else(|| anyhow::anyhow!("Current bank not found"))?;
        
        let filename = Self::generate_preset_filename(name);
        let full_path = bank.path.join(&filename);
        
        // Serialize to JSON
        let json = serde_json::to_string_pretty(data)?;
        std::fs::write(&full_path, json)?;
        
        // Re-index
        if let Some(bank) = self.banks.get_mut(&self.current_bank) {
            Self::index_presets_in_bank(bank);
        }
        
        log::info!("Saved preset '{}' to {:?}", name, full_path);
        Ok(())
    }
    
    /// Load a preset
    pub fn load_preset(&self, name: &str) -> anyhow::Result<PresetData> {
        let bank = self.banks.get(&self.current_bank)
            .ok_or_else(|| anyhow::anyhow!("Current bank not found"))?;
        
        // Find preset file by display name
        let index = bank.preset_display_names.iter()
            .position(|n| n == name)
            .or_else(|| {
                bank.preset_display_names.iter()
                    .position(|n| n == &Self::clean_display_name(name))
            })
            .ok_or_else(|| anyhow::anyhow!("Preset '{}' not found", name))?;
        
        let filename = &bank.preset_files[index];
        let full_path = bank.path.join(filename);
        
        let json = std::fs::read_to_string(&full_path)?;
        let data: PresetData = serde_json::from_str(&json)?;
        
        log::info!("Loaded preset '{}' from {:?}", name, full_path);
        Ok(data)
    }
    
    /// Delete a preset
    pub fn delete_preset(&mut self, index: usize) -> anyhow::Result<()> {
        let bank = self.banks.get(&self.current_bank)
            .ok_or_else(|| anyhow::anyhow!("Current bank not found"))?;
        
        if index >= bank.preset_files.len() {
            return Err(anyhow::anyhow!("Invalid preset index"));
        }
        
        let filename = &bank.preset_files[index];
        let full_path = bank.path.join(filename);
        
        std::fs::remove_file(&full_path)?;
        
        // Re-index
        if let Some(bank) = self.banks.get_mut(&self.current_bank) {
            Self::index_presets_in_bank(bank);
        }
        
        log::info!("Deleted preset at index {}", index);
        Ok(())
    }
    
    /// Index presets in a bank
    fn index_presets_in_bank(bank: &mut PresetBank) {
        bank.preset_files.clear();
        bank.preset_display_names.clear();
        
        if let Ok(entries) = std::fs::read_dir(&bank.path) {
            let mut files: Vec<_> = entries
                .flatten()
                .filter(|e| {
                    e.path().extension()
                        .and_then(|e| e.to_str())
                        .map(|e| e == "json")
                        .unwrap_or(false)
                })
                .map(|e| e.file_name().to_string_lossy().to_string())
                .collect();
            
            files.sort();
            
            for filename in files {
                bank.preset_display_names.push(Self::clean_display_name(&filename));
                bank.preset_files.push(filename);
            }
        }
    }
    
    /// Generate a filename from a display name
    fn generate_preset_filename(display_name: &str) -> String {
        // Sanitize name
        let sanitized: String = display_name
            .chars()
            .map(|c| match c {
                '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' => '_',
                _ => c,
            })
            .collect();
        
        format!("{}.json", sanitized)
    }
    
    /// Clean a filename to get display name
    fn clean_display_name(filename: &str) -> String {
        // Remove .json extension
        let name = filename.strip_suffix(".json").unwrap_or(filename);
        
        // Remove numeric prefix (###_)
        let name = if let Some(pos) = name.find('_') {
            if name[..pos].chars().all(|c| c.is_ascii_digit()) {
                &name[pos + 1..]
            } else {
                name
            }
        } else {
            name
        };
        
        // Remove legacy gwSaveState prefix
        let name = if name.starts_with("gwSaveState") {
            let rest = &name[11..];
            rest.trim_start_matches(|c: char| c.is_ascii_digit())
        } else {
            name
        };
        
        if name.is_empty() {
            "Preset".to_string()
        } else {
            name.to_string()
        }
    }
    
    /// Import an OpenFrameworks preset file
    pub fn import_of_preset(
        &mut self,
        of_path: &std::path::Path,
        name: Option<&str>,
    ) -> anyhow::Result<()> {
        use crate::params::of_preset_compat::OfPreset;
        
        let of_preset = OfPreset::from_file(of_path)?;
        
        // Generate name from filename if not provided
        let preset_name = name.map(|s| s.to_string()).unwrap_or_else(|| {
            of_path.file_stem()
                .and_then(|s| s.to_str())
                .map(|s| {
                    // Clean up OF preset names like "gwSaveState041"
                    let s = s.trim_start_matches("gwSaveState");
                    let s = s.trim_start_matches(|c: char| c.is_ascii_digit());
                    if s.is_empty() { "Imported".to_string() } else { s.to_string() }
                })
                .unwrap_or_else(|| "Imported".to_string())
        });
        
        // Convert to Rust preset format
        let preset_data = PresetData {
            block1: of_preset.to_block1_params(),
            block2: of_preset.to_block2_params(),
            block3: of_preset.to_block3_params(),
            block1_modulations: HashMap::new(), // TODO: Extract LFO modulation data
            block2_modulations: HashMap::new(),
            block3_modulations: HashMap::new(),
            audio: PresetAudioSettings::default(),
            tempo: PresetTempoData::default(),
            lfo_banks: default_lfo_banks(),
            midi_mappings: Vec::new(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            name: preset_name.clone(),
        };
        
        // Save to current bank
        self.save_preset(&preset_name, &preset_data)?;
        
        log::info!("Imported OF preset '{}' to bank '{}'", preset_name, self.current_bank);
        Ok(())
    }
    
    /// Batch import all OF presets from a directory
    pub fn batch_import_of_presets(
        &mut self,
        of_dir: &std::path::Path,
    ) -> anyhow::Result<(usize, usize)> {
        let mut success = 0;
        let mut failed = 0;
        
        if !of_dir.exists() {
            anyhow::bail!("Source directory does not exist: {:?}", of_dir);
        }
        
        for entry in std::fs::read_dir(of_dir)? {
            let entry = entry?;
            let path = entry.path();
            
            if path.extension().and_then(|e| e.to_str()) == Some("json") {
                match self.import_of_preset(&path, None) {
                    Ok(_) => success += 1,
                    Err(e) => {
                        log::warn!("Failed to import {:?}: {}", path, e);
                        failed += 1;
                    }
                }
            }
        }
        
        log::info!("Batch import complete: {} succeeded, {} failed", success, failed);
        Ok((success, failed))
    }
}

impl Default for PresetManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Apply audio modulations to parameters with attack/release envelope following.
///
/// Uses asymmetric exponential smoothing: `attack` controls how fast the envelope
/// rises to follow transients, `release` controls how slowly it decays. This makes
/// audio-reactive parameters feel more musical and less jittery.
pub fn apply_audio_modulations(
    base_value: f32,
    modulation: &mut ParamModulationData,
    fft_value: f32,
    delta_time: f32,
) -> f32 {
    if !modulation.audio_enabled {
        modulation.audio_smoothed_value = 0.0;
        return base_value;
    }

    let target = fft_value * modulation.audio_amount * modulation.audio_range_scale;

    // Asymmetric envelope: use attack when rising, release when falling
    let smoothing = if target > modulation.audio_smoothed_value {
        modulation.audio_attack
    } else {
        modulation.audio_release
    };

    // Exponential smoothing (frame-rate independent)
    let smoothing_factor = (-delta_time / smoothing.max(0.001)).exp();
    modulation.audio_smoothed_value = modulation.audio_smoothed_value * smoothing_factor
        + target * (1.0 - smoothing_factor);

    base_value + modulation.audio_smoothed_value
}

/// Get list of modulated parameters for display
pub fn get_modulated_params(
    modulations: &HashMap<String, ParamModulationData>,
) -> Vec<(String, ParamModulationData)> {
    let mut result: Vec<_> = modulations
        .iter()
        .filter(|(_, m)| m.audio_enabled || m.bpm_enabled)
        .map(|(name, data)| (name.clone(), *data))
        .collect();
    
    result.sort_by(|a, b| a.0.cmp(&b.0));
    result
}
