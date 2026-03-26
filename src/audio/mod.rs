//! # Audio Module
//!
//! Handles audio input capture and FFT analysis for audio-reactive visuals.
//! Uses cpal for audio capture and rustfft for spectrum analysis.
//! 
//! Features 8-band FFT matching BLUEJAY_WAAAVES:
//! - Sub Bass: 20-60 Hz
//! - Bass: 60-120 Hz  
//! - Low Mid: 120-250 Hz
//! - Mid: 250-500 Hz
//! - High Mid: 500-2000 Hz
//! - High: 2000-4000 Hz
//! - Very High: 4000-8000 Hz
//! - Presence: 8000-16000 Hz

use anyhow::{anyhow, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use realfft::{num_complex::Complex32, RealFftPlanner};
use std::sync::{Arc, Mutex};

/// Available FFT sizes (must be powers of 2)
pub const FFT_SIZES: &[usize] = &[1024, 2048, 4096, 8192];

/// Human-readable labels for the FFT size dropdown
pub const FFT_SIZE_LABELS: &[&str] = &[
    "1024  (43 Hz, 23ms)",
    "2048  (21 Hz, 46ms)",
    "4096  (11 Hz, 93ms)",
    "8192  (5 Hz, 186ms)",
];

/// 8 FFT Bands matching the OF app
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FftBand {
    SubBass = 0,    // 20-60 Hz
    Bass = 1,       // 60-120 Hz
    LowMid = 2,     // 120-250 Hz
    Mid = 3,        // 250-500 Hz
    HighMid = 4,    // 500-2000 Hz
    High = 5,       // 2000-4000 Hz
    VeryHigh = 6,   // 4000-8000 Hz
    Presence = 7,   // 8000-16000 Hz
}

impl FftBand {
    /// Get frequency range for this band (min, max) in Hz
    pub fn freq_range(&self) -> (f32, f32) {
        match self {
            FftBand::SubBass => (20.0, 60.0),
            FftBand::Bass => (60.0, 120.0),
            FftBand::LowMid => (120.0, 250.0),
            FftBand::Mid => (250.0, 500.0),
            FftBand::HighMid => (500.0, 2000.0),
            FftBand::High => (2000.0, 4000.0),
            FftBand::VeryHigh => (4000.0, 8000.0),
            FftBand::Presence => (8000.0, 16000.0),
        }
    }
    
    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            FftBand::SubBass => "Sub Bass (20-60Hz)",
            FftBand::Bass => "Bass (60-120Hz)",
            FftBand::LowMid => "Low Mid (120-250Hz)",
            FftBand::Mid => "Mid (250-500Hz)",
            FftBand::HighMid => "High Mid (500-2kHz)",
            FftBand::High => "High (2k-4kHz)",
            FftBand::VeryHigh => "Very High (4k-8kHz)",
            FftBand::Presence => "Presence (8k-16kHz)",
        }
    }
    
    /// Get short name for display
    pub fn short_name(&self) -> &'static str {
        match self {
            FftBand::SubBass => "Sub",
            FftBand::Bass => "Bass",
            FftBand::LowMid => "LoMid",
            FftBand::Mid => "Mid",
            FftBand::HighMid => "HiMid",
            FftBand::High => "High",
            FftBand::VeryHigh => "VHigh",
            FftBand::Presence => "Presence",
        }
    }
    
    /// Get all bands as array
    pub fn all() -> [FftBand; 8] {
        [
            FftBand::SubBass,
            FftBand::Bass,
            FftBand::LowMid,
            FftBand::Mid,
            FftBand::HighMid,
            FftBand::High,
            FftBand::VeryHigh,
            FftBand::Presence,
        ]
    }
}

/// Audio modulation for a single parameter
#[derive(Debug, Clone, Copy)]
pub struct AudioModulation {
    /// Whether modulation is enabled
    pub enabled: bool,
    /// Which FFT band to use (0-7)
    pub fft_band: usize,
    /// Modulation depth (-1 to 1, bipolar)
    pub amount: f32,
    /// Attack smoothing (0-1)
    pub attack: f32,
    /// Release smoothing (0-1)
    pub release: f32,
    /// Scale factor for parameter range
    pub range_scale: f32,
    /// Current modulated value (runtime)
    pub current_value: f32,
}

impl Default for AudioModulation {
    fn default() -> Self {
        Self {
            enabled: false,
            fft_band: 0,
            amount: 0.0,
            attack: 0.1,
            release: 0.1,
            range_scale: 1.0,
            current_value: 0.0,
        }
    }
}

impl AudioModulation {
    /// Create new modulation with defaults
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Process modulation with FFT band value
    pub fn process(&mut self, fft_value: f32, delta_time: f32) -> f32 {
        if !self.enabled {
            self.current_value = 0.0;
            return 0.0;
        }
        
        // Calculate target based on FFT value and amount
        let target = fft_value * self.amount * self.range_scale;
        
        // Apply attack/release smoothing
        let smoothing = if target > self.current_value {
            self.attack
        } else {
            self.release
        };
        
        let smoothing_factor = (-delta_time / smoothing.max(0.001)).exp();
        self.current_value = self.current_value * smoothing_factor + target * (1.0 - smoothing_factor);
        
        self.current_value
    }
    
    /// Reset current value
    pub fn reset(&mut self) {
        self.current_value = 0.0;
    }
}

/// 8-band FFT analysis results
#[derive(Debug, Clone)]
pub struct FftBands {
    /// 8 frequency bands (0-1 normalized)
    pub bands: [f32; 8],
    /// Smoothed values
    pub smoothed: [f32; 8],
    /// Peak values (for normalization)
    pub peaks: [f32; 8],
    /// Min values (for normalization)
    pub mins: [f32; 8],
    /// Normalized values (when normalization is enabled)
    pub normalized: [f32; 8],
}

impl Default for FftBands {
    fn default() -> Self {
        Self {
            bands: [0.0; 8],
            smoothed: [0.0; 8],
            peaks: [0.01; 8], // Start with small value to avoid division by zero
            mins: [1.0; 8],
            normalized: [0.0; 8],
        }
    }
}

/// Audio processing settings (shared between threads)
#[derive(Debug, Clone)]
pub struct AudioSettings {
    /// Smoothing factor (0-1)
    pub smoothing: f32,
    /// Amplitude multiplier
    pub amplitude: f32,
    /// Normalization enabled
    pub normalization: bool,
    /// Pink noise compensation (makes pink noise appear flat)
    pub pink_compensation: bool,
}

impl Default for AudioSettings {
    fn default() -> Self {
        Self {
            smoothing: 0.7,
            amplitude: 1.0,
            normalization: false,
            pink_compensation: false,
        }
    }
}

/// Audio input manager
pub struct AudioInput {
    /// Whether audio capture is active
    active: bool,
    /// Sample rate
    sample_rate: u32,
    /// FFT size (number of samples per FFT)
    fft_size: usize,
    /// Audio stream
    stream: Option<cpal::Stream>,
    /// Shared audio buffer for FFT
    audio_buffer: Arc<Mutex<Vec<f32>>>,
    /// FFT output (frequency bins)
    fft_output: Arc<Mutex<Vec<f32>>>,
    /// 8-band FFT analysis
    fft_bands: Arc<Mutex<FftBands>>,
    /// Beat detection state
    beat_state: Arc<Mutex<BeatState>>,
    /// Audio processing settings (shared with audio thread)
    settings: Arc<Mutex<AudioSettings>>,
}

/// Beat detection state
#[derive(Debug, Clone)]
pub struct BeatState {
    /// Current energy level
    pub energy: f32,
    /// Beat detected this frame
    pub beat: bool,
    /// Beat phase (0-1)
    pub phase: f32,
    /// Beat tempo estimate (BPM)
    pub bpm: f32,
    /// History for beat detection
    energy_history: Vec<f32>,
    /// Last beat time
    last_beat_time: std::time::Instant,
}

/// FFT analysis results
#[derive(Debug, Clone)]
pub struct FftAnalysis {
    /// Frequency bins (linear scale)
    pub bins: Vec<f32>,
    /// Number of bins
    pub num_bins: usize,
    /// Sample rate
    pub sample_rate: u32,
}

impl AudioInput {
    /// Create a new audio input manager
    pub fn new(fft_size: usize) -> Self {
        Self {
            active: false,
            sample_rate: 44100,
            fft_size,
            stream: None,
            audio_buffer: Arc::new(Mutex::new(Vec::with_capacity(fft_size))),
            fft_output: Arc::new(Mutex::new(vec![0.0; fft_size / 2])),
            fft_bands: Arc::new(Mutex::new(FftBands::default())),
            beat_state: Arc::new(Mutex::new(BeatState::new())),
            settings: Arc::new(Mutex::new(AudioSettings::default())),
        }
    }
    
    /// Get 8-band FFT values (0-1 range, smoothed)
    /// Returns normalized values if normalization is enabled
    pub fn get_8band_fft(&self) -> [f32; 8] {
        let bands = self.fft_bands.lock().unwrap();
        let use_normalization = self.settings.lock().map(|s| s.normalization).unwrap_or(false);
        if use_normalization {
            bands.normalized
        } else {
            bands.smoothed
        }
    }
    
    /// Get specific FFT band value
    pub fn get_band(&self, band: FftBand) -> f32 {
        self.fft_bands.lock().unwrap().smoothed[band as usize]
    }
    
    /// Get specific FFT band by index (0-7)
    pub fn get_band_by_index(&self, index: usize) -> f32 {
        if index < 8 {
            self.fft_bands.lock().unwrap().smoothed[index]
        } else {
            0.0
        }
    }
    
    /// Set smoothing factor (0-1)
    pub fn set_smoothing(&mut self, smoothing: f32) {
        if let Ok(mut settings) = self.settings.lock() {
            settings.smoothing = smoothing.clamp(0.0, 0.99);
        }
    }
    
    /// Get smoothing factor
    pub fn get_smoothing(&self) -> f32 {
        self.settings.lock().map(|s| s.smoothing).unwrap_or(0.7)
    }
    
    /// Set amplitude multiplier (0-1, unity gain at 1.0)
    pub fn set_amplitude(&mut self, amplitude: f32) {
        if let Ok(mut settings) = self.settings.lock() {
            settings.amplitude = amplitude.clamp(0.0, 1.0);
        }
    }
    
    /// Get amplitude multiplier
    pub fn get_amplitude(&self) -> f32 {
        self.settings.lock().map(|s| s.amplitude).unwrap_or(1.0)
    }
    
    /// Set normalization enabled
    pub fn set_normalization(&mut self, enabled: bool) {
        if let Ok(mut settings) = self.settings.lock() {
            settings.normalization = enabled;
        }
    }
    
    /// Get normalization enabled
    pub fn get_normalization(&self) -> bool {
        self.settings.lock().map(|s| s.normalization).unwrap_or(false)
    }
    
    /// Set pink noise compensation
    pub fn set_pink_compensation(&mut self, enabled: bool) {
        if let Ok(mut settings) = self.settings.lock() {
            settings.pink_compensation = enabled;
        }
    }
    
    /// Get pink noise compensation
    pub fn get_pink_compensation(&self) -> bool {
        self.settings.lock().map(|s| s.pink_compensation).unwrap_or(false)
    }
    
    /// Initialize audio input with default device
    pub fn initialize(&mut self) -> Result<()> {
        self.initialize_with_device(-1)
    }
    
    /// Initialize audio input with specific device index (-1 for default)
    pub fn initialize_with_device(&mut self, device_index: i32) -> Result<()> {
        let host = cpal::default_host();
        
        // Get input device based on index
        let device = if device_index < 0 {
            // Use default device
            match host.default_input_device() {
                Some(d) => d,
                None => {
                    log::warn!("No audio input device available - audio features disabled");
                    return Err(anyhow!("No audio input device"));
                }
            }
        } else {
            // Use specific device by index
            let devices = match host.input_devices() {
                Ok(d) => d.collect::<Vec<_>>(),
                Err(e) => {
                    log::warn!("Failed to enumerate audio devices: {:?}", e);
                    return Err(anyhow!("Failed to enumerate audio devices: {:?}", e));
                }
            };
            
            let idx = device_index as usize;
            if idx >= devices.len() {
                log::warn!("Audio device index {} out of range (max {}), using default", idx, devices.len());
                match host.default_input_device() {
                    Some(d) => d,
                    None => {
                        log::warn!("No audio input device available");
                        return Err(anyhow!("No audio input device"));
                    }
                }
            } else {
                devices[idx].clone()
            }
        };
        
        let device_name = device.name().unwrap_or_else(|_| "Unknown".to_string());
        log::info!("Using audio input device: {} (index: {})", device_name, device_index);
        
        // Get default input config
        let config = match device.default_input_config() {
            Ok(c) => c,
            Err(e) => {
                log::warn!("Failed to get audio config: {:?} - audio features disabled", e);
                return Err(anyhow!("Failed to get audio config: {:?}", e));
            }
        };
        log::info!("Audio config: sample_rate={}, channels={}, format={:?}", 
            config.sample_rate().0, config.channels(), config.sample_format());
        
        self.sample_rate = config.sample_rate().0;
        
        // Build stream - using f32 format
        if config.sample_format() != cpal::SampleFormat::F32 {
            log::warn!("Audio device doesn't support f32 format, trying anyway");
        }
        let stream = match self.build_stream(&device, &config.into()) {
            Ok(s) => s,
            Err(e) => {
                log::warn!("Failed to build audio stream: {:?} - audio features disabled", e);
                return Err(e);
            }
        };
        
        self.stream = Some(stream);
        self.active = true;
        
        Ok(())
    }
    
    /// Build audio stream for f32 samples
    fn build_stream(&self, device: &cpal::Device, config: &cpal::StreamConfig) -> Result<cpal::Stream> {
        let fft_size = self.fft_size;
        let sample_rate = self.sample_rate;
        let settings = Arc::clone(&self.settings);
        let audio_buffer = Arc::clone(&self.audio_buffer);
        let fft_output = Arc::clone(&self.fft_output);
        let fft_bands = Arc::clone(&self.fft_bands);
        let beat_state = Arc::clone(&self.beat_state);
        
        // Create FFT planner
        let mut fft_planner = RealFftPlanner::<f32>::new();
        let fft = fft_planner.plan_fft_forward(fft_size);
        
        let err_fn = |err| log::error!("Audio stream error: {}", err);
        
        let stream = device.build_input_stream(
            config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                // Read current settings (amplitude, smoothing, normalization, pink_compensation)
                let (amplitude, smoothing, normalization, pink_compensation) = settings.lock()
                    .map(|s| (s.amplitude, s.smoothing, s.normalization, s.pink_compensation))
                    .unwrap_or((1.0, 0.7, false, false));
                
                // Samples are already f32
                let samples: Vec<f32> = data.to_vec();
                
                // Add to audio buffer
                {
                    let mut buffer = audio_buffer.lock().unwrap();
                    buffer.extend_from_slice(&samples);
                    
                    // Process when we have enough samples
                    while buffer.len() >= fft_size {
                        let frame: Vec<f32> = buffer.drain(0..fft_size).collect();
                        
                        // Perform FFT
                        let mut input = frame.clone();
                        let mut output = vec![Complex32::new(0.0, 0.0); fft_size / 2 + 1];
                        
                        if fft.process(&mut input, &mut output).is_ok() {
                            // Calculate magnitude spectrum
                            // Normalize by fft_size (realfft does NOT divide by N)
                            let fft_norm = 1.0 / fft_size as f32;
                            let gain = amplitude.max(0.0001); // Prevent zero gain issues
                            let mut bins = Vec::with_capacity(fft_size / 2);
                            let mut total_energy = 0.0f32;

                            for i in 0..(fft_size / 2) {
                                let magnitude = output[i].norm() * fft_norm * gain;
                                let db = 20.0 * (magnitude + 1e-10).log10();
                                let normalized = ((db + 60.0) / 60.0).clamp(0.0, 1.0);
                                bins.push(normalized);
                                total_energy += normalized;
                            }
                            
                            // Update FFT output
                            if let Ok(mut fft_out) = fft_output.lock() {
                                *fft_out = bins.clone();
                            }
                            
                            // Compute 8-band FFT
                            if let Ok(mut bands) = fft_bands.lock() {
                                compute_8band_fft(&bins, sample_rate, fft_size, &mut bands, smoothing, normalization, pink_compensation);
                            }
                            
                            // Update beat detection
                            if let Ok(mut beat) = beat_state.lock() {
                                beat.update(total_energy / (fft_size as f32 / 2.0));
                            }
                        }
                    }
                }
            },
            err_fn,
            None,
        )?;
        
        stream.play()?;
        
        Ok(stream)
    }
    
    /// Start audio capture
    pub fn start(&mut self) -> Result<()> {
        if let Some(stream) = &self.stream {
            stream.play()?;
            self.active = true;
        }
        Ok(())
    }
    
    /// Stop audio capture
    pub fn stop(&mut self) -> Result<()> {
        if let Some(stream) = &self.stream {
            stream.pause()?;
            self.active = false;
        }
        Ok(())
    }
    
    /// Get current FFT bins
    pub fn get_fft(&self) -> Vec<f32> {
        self.fft_output.lock().unwrap().clone()
    }
    
    /// Get beat detection state
    pub fn get_beat_state(&self) -> BeatState {
        self.beat_state.lock().unwrap().clone()
    }
    
    /// Check if beat was detected
    pub fn is_beat(&self) -> bool {
        self.beat_state.lock().unwrap().beat
    }
    
    /// Get current energy level (0-1)
    pub fn get_energy(&self) -> f32 {
        self.beat_state.lock().unwrap().energy
    }
    
    /// Get estimated BPM
    pub fn get_bpm(&self) -> f32 {
        self.beat_state.lock().unwrap().bpm
    }
    
    /// List available audio input devices (safe wrapper)
    pub fn list_devices() -> Vec<String> {
        match std::panic::catch_unwind(audio_list_devices_internal) {
            Ok(result) => result,
            Err(_) => {
                log::error!("Audio device enumeration panicked");
                Vec::new()
            }
        }
    }
    
    /// Check if audio is active
    pub fn is_active(&self) -> bool {
        self.active
    }
    
    /// Get sample rate
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
    
    /// Get FFT size
    pub fn fft_size(&self) -> usize {
        self.fft_size
    }
}

impl BeatState {
    /// Create new beat state
    pub fn new() -> Self {
        Self {
            energy: 0.0,
            beat: false,
            phase: 0.0,
            bpm: 120.0,
            energy_history: Vec::with_capacity(43), // ~1 second at 44.1kHz with 1024 FFT
            last_beat_time: std::time::Instant::now(),
        }
    }
    
    /// Update beat detection with new energy level
    fn update(&mut self, energy: f32) {
        self.beat = false;
        self.energy = energy;
        
        // Add to history
        self.energy_history.push(energy);
        if self.energy_history.len() > 43 {
            self.energy_history.remove(0);
        }
        
        // Calculate average energy
        if self.energy_history.len() >= 10 {
            let avg_energy: f32 = self.energy_history.iter().sum::<f32>() / self.energy_history.len() as f32;
            
            // Simple beat detection: energy spike above 1.3x average
            if energy > avg_energy * 1.3 && energy > 0.1 {
                let now = std::time::Instant::now();
                let time_since_last = now.duration_since(self.last_beat_time).as_secs_f32();
                
                // Debounce: minimum 200ms between beats
                if time_since_last > 0.2 {
                    self.beat = true;
                    
                    // Estimate BPM
                    if time_since_last > 0.25 && time_since_last < 2.0 {
                        self.bpm = 60.0 / time_since_last;
                    }
                    
                    self.last_beat_time = now;
                    self.phase = 0.0;
                }
            }
        }
        
        // Update phase (0-1 based on current BPM)
        let time_since_beat = self.last_beat_time.elapsed().as_secs_f32();
        let beat_duration = 60.0 / self.bpm.max(60.0);
        self.phase = (time_since_beat / beat_duration).fract();
    }
}

impl Default for BeatState {
    fn default() -> Self {
        Self::new()
    }
}

/// Get frequency for a given FFT bin
pub fn bin_to_frequency(bin_index: usize, fft_size: usize, sample_rate: u32) -> f32 {
    (bin_index as f32 * sample_rate as f32) / fft_size as f32
}

/// Get FFT bin index for a given frequency
pub fn frequency_to_bin(frequency: f32, fft_size: usize, sample_rate: u32) -> usize {
    ((frequency * fft_size as f32) / sample_rate as f32) as usize
}

/// Smooth FFT data with simple moving average
pub fn smooth_fft(bins: &[f32], smoothing: f32) -> Vec<f32> {
    let mut smoothed = Vec::with_capacity(bins.len());
    let mut prev = 0.0f32;
    
    for &bin in bins {
        prev = prev * smoothing + bin * (1.0 - smoothing);
        smoothed.push(prev);
    }
    
    smoothed
}

/// Get bass energy (20-250 Hz)
pub fn get_bass_energy(bins: &[f32], sample_rate: u32, fft_size: usize) -> f32 {
    let start_bin = frequency_to_bin(20.0, fft_size, sample_rate);
    let end_bin = frequency_to_bin(250.0, fft_size, sample_rate).min(bins.len());
    
    if end_bin > start_bin {
        bins[start_bin..end_bin].iter().sum::<f32>() / (end_bin - start_bin) as f32
    } else {
        0.0
    }
}

/// Get mid energy (250 Hz - 4 kHz)
pub fn get_mid_energy(bins: &[f32], sample_rate: u32, fft_size: usize) -> f32 {
    let start_bin = frequency_to_bin(250.0, fft_size, sample_rate);
    let end_bin = frequency_to_bin(4000.0, fft_size, sample_rate).min(bins.len());
    
    if end_bin > start_bin {
        bins[start_bin..end_bin].iter().sum::<f32>() / (end_bin - start_bin) as f32
    } else {
        0.0
    }
}

/// Get treble energy (4-20 kHz)
pub fn get_treble_energy(bins: &[f32], sample_rate: u32, fft_size: usize) -> f32 {
    let start_bin = frequency_to_bin(4000.0, fft_size, sample_rate);
    let end_bin = frequency_to_bin(20000.0, fft_size, sample_rate).min(bins.len());
    
    if end_bin > start_bin {
        bins[start_bin..end_bin].iter().sum::<f32>() / (end_bin - start_bin) as f32
    } else {
        0.0
    }
}

/// Pink noise compensation gains for each band.
/// Moderate multiplicative gains that work with dB-normalized 0-1 values.
/// Old values (up to 11.3) were designed for linear space and clamped everything;
/// these are gentler and let the .min(1.0) clamp catch only actual peaks.
const PINK_NOISE_GAINS: [f32; 8] = [
    1.0,       // Sub Bass  ~40 Hz (reference)
    1.15,      // Bass      ~90 Hz
    1.30,      // Low Mid   ~185 Hz
    1.50,      // Mid       ~375 Hz
    1.80,      // High Mid  ~1000 Hz
    2.20,      // High      ~3000 Hz
    2.60,      // Very High ~6000 Hz
    3.00,      // Presence  ~12000 Hz
];

/// Compute 8-band FFT from raw FFT bins
fn compute_8band_fft(bins: &[f32], sample_rate: u32, fft_size: usize, bands: &mut FftBands, smoothing: f32, normalization: bool, pink_compensation: bool) {
    // Define frequency ranges for each band (min, max) in Hz
    let band_ranges = [
        (20.0_f32, 60.0_f32),      // Sub Bass
        (60.0_f32, 120.0_f32),     // Bass
        (120.0_f32, 250.0_f32),    // Low Mid
        (250.0_f32, 500.0_f32),    // Mid
        (500.0_f32, 2000.0_f32),   // High Mid
        (2000.0_f32, 4000.0_f32),  // High
        (4000.0_f32, 8000.0_f32),  // Very High
        (8000.0_f32, 16000.0_f32), // Presence
    ];
    
    // Calculate bin indices for each frequency
    let nyquist = sample_rate as f32 / 2.0;
    let bins_per_hz = bins.len() as f32 / nyquist;
    
    for (i, (min_freq, max_freq)) in band_ranges.iter().enumerate() {
        let start_bin = ((min_freq * bins_per_hz) as usize).min(bins.len() - 1);
        let end_bin = ((max_freq * bins_per_hz) as usize).min(bins.len());

        if end_bin > start_bin {
            // Use peak (max) per band instead of average.
            // This prevents dilution in high-frequency bands that span many bins.
            let peak = bins[start_bin..end_bin]
                .iter()
                .cloned()
                .fold(0.0f32, f32::max);

            // Apply pink noise compensation if enabled (multiplicative, clamped to 1.0)
            let compensated = if pink_compensation {
                (peak * PINK_NOISE_GAINS[i]).min(1.0)
            } else {
                peak
            };

            // Update raw band value
            bands.bands[i] = compensated;
            
            // Update smoothed value
            bands.smoothed[i] = bands.smoothed[i] * smoothing + compensated * (1.0 - smoothing);
            
            // Update peak (slow decay)
            if bands.smoothed[i] > bands.peaks[i] {
                bands.peaks[i] = bands.smoothed[i];
            } else {
                bands.peaks[i] *= 0.995; // Slow decay
            }
            
            // Update min (for normalization)
            if bands.smoothed[i] < bands.mins[i] && bands.smoothed[i] > 0.001 {
                bands.mins[i] = bands.smoothed[i];
            }
            
            // Calculate normalized value (0-1 based on min/max range)
            // Ensure peaks >= mins to avoid negative ranges
            let peaks = bands.peaks[i].max(bands.mins[i]);
            let range = peaks - bands.mins[i];
            if range > 0.001 {
                bands.normalized[i] = ((bands.smoothed[i] - bands.mins[i]) / range).clamp(0.0, 1.0);
            } else {
                bands.normalized[i] = bands.smoothed[i].clamp(0.0, 1.0);
            }
        }
    }
}

/// Internal function to list audio devices
fn audio_list_devices_internal() -> Vec<String> {
    use cpal::traits::{DeviceTrait, HostTrait};
    
    let host = cpal::default_host();
    let mut devices = Vec::new();
    
    match host.input_devices() {
        Ok(input_devices) => {
            for (idx, device) in input_devices.enumerate() {
                if let Ok(name) = device.name() {
                    devices.push(format!("{}: {}", idx, name));
                }
            }
        }
        Err(e) => {
            log::error!("Failed to list audio devices: {:?}", e);
        }
    }
    
    devices
}
