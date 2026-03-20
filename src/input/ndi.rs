//! # NDI Input Implementation
//!
//! Provides NDI (Network Device Interface) video input support using grafton-ndi 0.11.
//! This module is only compiled when the `ndi` feature is enabled.

#![cfg(feature = "ndi")]

//!
//! 
//! Architecture:
//! - NdiSourceFinder: Discovers available NDI sources on the network
//! - NdiReceiver: Receives video frames from an NDI source in a background thread
//! - Frame queue with latest-frame-only semantics to prevent buildup

use grafton_ndi::{NDI, Finder, FinderOptions, Receiver, ReceiverOptions, ReceiverColorFormat, ReceiverBandwidth};
use crossbeam::channel::{self, Sender, Receiver as CrossbeamReceiver};
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// Information about an available NDI source
#[derive(Debug, Clone)]
pub struct NdiSourceInfo {
    pub name: String,
    pub url: String,
}

/// A received NDI video frame
pub struct NdiFrame {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>, // BGRA format (native macOS IOSurface format, matches Bgra8Unorm texture)
    pub timestamp: Instant,
}

/// NDI source finder for discovering sources on the network
pub struct NdiSourceFinder;

impl NdiSourceFinder {
    /// Find available NDI sources on the network
    /// 
    /// # Arguments
    /// * `timeout_ms` - How long to wait for sources to appear
    /// 
    /// # Returns
    /// List of available NDI source names
    pub fn find_sources(timeout_ms: u32) -> Vec<NdiSourceInfo> {
        let ndi = match NDI::new() {
            Ok(ndi) => ndi,
            Err(e) => {
                log::error!("Failed to initialize NDI: {:?}", e);
                return Vec::new();
            }
        };

        let options = FinderOptions::builder()
            .show_local_sources(true)
            .build();

        let finder = match Finder::new(&ndi, &options) {
            Ok(f) => f,
            Err(e) => {
                log::error!("Failed to create NDI finder: {:?}", e);
                return Vec::new();
            }
        };

        // Get sources with timeout
        match finder.find_sources(Duration::from_millis(timeout_ms as u64)) {
            Ok(sources) => {
                sources.into_iter()
                    .map(|source| {
                        let name = source.name.clone();
                        let url = format!("{:?}", source.address);
                        NdiSourceInfo { name, url }
                    })
                    .collect()
            }
            Err(e) => {
                log::error!("Failed to find NDI sources: {:?}", e);
                Vec::new()
            }
        }
    }
}

/// NDI receiver that captures video frames from a source
pub struct NdiReceiver {
    source_name: String,
    receiver_thread: Option<JoinHandle<()>>,
    frame_tx: Sender<NdiFrame>,
    frame_rx: CrossbeamReceiver<NdiFrame>,
    running: Arc<AtomicBool>,
    resolution: (u32, u32),
}

impl NdiReceiver {
    /// Create a new NDI receiver (does not start receiving yet)
    pub fn new(source_name: impl Into<String>) -> Self {
        let (frame_tx, frame_rx) = channel::bounded(5); // Bounded channel with 5 frame capacity
        
        Self {
            source_name: source_name.into(),
            receiver_thread: None,
            frame_tx,
            frame_rx,
            running: Arc::new(AtomicBool::new(false)),
            resolution: (1920, 1080), // Default, updated on first frame
        }
    }

    /// Start receiving from the NDI source
    /// 
    /// This spawns a background thread that continuously receives frames
    /// from the NDI source and sends them through the channel.
    pub fn start(&mut self) -> anyhow::Result<()> {
        if self.receiver_thread.is_some() {
            return Err(anyhow::anyhow!("NDI receiver already started"));
        }

        let ndi = NDI::new().map_err(|e| {
            anyhow::anyhow!("Failed to initialize NDI: {:?}", e)
        })?;

        let source_name = self.source_name.clone();
        let frame_tx = self.frame_tx.clone();
        let running = Arc::clone(&self.running);
        running.store(true, Ordering::SeqCst);

        let thread_handle = thread::spawn(move || {

            // Find the source
            let options = FinderOptions::builder()
                .show_local_sources(true)
                .build();

            let finder = match Finder::new(&ndi, &options) {
                Ok(f) => f,
                Err(e) => {
                    log::error!("[NDI] Failed to create finder: {:?}", e);
                    return;
                }
            };

            // Wait for the specific source
            let mut found_source = None;
            let search_start = Instant::now();
            
            while running.load(Ordering::SeqCst) && search_start.elapsed().as_secs() < 10 {
                match finder.find_sources(Duration::from_millis(100)) {
                    Ok(sources) => {
                        for source in sources {
                            if source.name.contains(&source_name) || source_name.contains(&source.name) {
                                found_source = Some(source);
                                break;
                            }
                        }
                    }
                    Err(e) => {
                        log::debug!("[NDI] Error finding sources: {:?}", e);
                    }
                }
                
                if found_source.is_some() {
                    break;
                }
                
                thread::sleep(Duration::from_millis(50));
            }

            let source = match found_source {
                Some(s) => s,
                None => {
                    log::error!("[NDI] Could not find source: {}", source_name);
                    return;
                }
            };

            // Create receiver with BGRA format (native macOS IOSurface format)
            let options = ReceiverOptions::builder(source)
                .color(ReceiverColorFormat::BGRX_BGRA)
                .bandwidth(ReceiverBandwidth::Highest)
                .build();

            let receiver = match Receiver::new(&ndi, &options) {
                Ok(r) => r,
                Err(e) => {
                    log::error!("[NDI] Failed to create receiver: {:?}", e);
                    return;
                }
            };

            log::info!("[NDI] Connected to: {}", source_name);

            // Receive loop
            while running.load(Ordering::SeqCst) {
                match receiver.capture_video_ref(Duration::from_millis(100)) {
                    Ok(Some(video_frame)) => {
                        // Get frame dimensions
                        let width = video_frame.width() as u32;
                        let height = video_frame.height() as u32;
                        
                        // Get frame data — already BGRA (ReceiverColorFormat::BGRX_BGRA)
                        // Strip any row stride/padding to produce tightly-packed BGRA bytes
                        // matching the Bgra8Unorm texture format. No channel swap needed.
                        let frame_data = video_frame.data();
                        let data = strip_stride_bgra(frame_data, width, height);

                        let frame = NdiFrame {
                            width,
                            height,
                            data,
                            timestamp: Instant::now(),
                        };

                        // Send frame (non-blocking, drop if channel full)
                        let _ = frame_tx.try_send(frame);
                    }
                    Ok(None) => {
                        // No frame available (timeout)
                    }
                    Err(e) => {
                        log::error!("[NDI] Frame capture error: {:?}", e);
                        thread::sleep(Duration::from_millis(10));
                    }
                }
            }

        });

        self.receiver_thread = Some(thread_handle);
        Ok(())
    }

    /// Stop receiving frames
    pub fn stop(&mut self) {
        self.running.store(false, Ordering::SeqCst);
        
        if let Some(handle) = self.receiver_thread.take() {
            // Wait for thread to finish (with timeout)
            let _ = handle.join();
        }

        log::info!("[NDI] Receiver stopped for source: {}", self.source_name);
    }

    /// Get the latest frame (non-blocking, consumes the frame)
    pub fn get_latest_frame(&mut self) -> Option<NdiFrame> {
        // Drain all available frames and return only the most recent
        let mut latest: Option<NdiFrame> = None;
        while let Ok(frame) = self.frame_rx.try_recv() {
            self.resolution = (frame.width, frame.height);
            latest = Some(frame);
        }
        latest
    }

    /// Check if a new frame is available
    pub fn has_frame(&self) -> bool {
        !self.frame_rx.is_empty()
    }

    /// Get current resolution
    pub fn resolution(&self) -> (u32, u32) {
        self.resolution
    }

    /// Check if receiver is running
    pub fn is_running(&self) -> bool {
        self.receiver_thread.is_some() && self.running.load(Ordering::SeqCst)
    }
}

impl Drop for NdiReceiver {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Strip NDI row stride/padding from BGRA data.
///
/// NDI frames may have row-aligned padding. This produces tightly-packed BGRA
/// bytes ready to upload to a `Bgra8Unorm` wgpu texture. No channel swap needed
/// since BGRX_BGRA receiver color format already matches `Bgra8Unorm`.
fn strip_stride_bgra(bgra_data: &[u8], width: u32, height: u32) -> Vec<u8> {
    let row_bytes = width as usize * 4;
    let pixel_count = (width * height) as usize;
    let mut out = vec![0u8; pixel_count * 4];

    // Derive actual stride from total data length
    let actual_stride = if height > 0 {
        bgra_data.len() / height as usize
    } else {
        row_bytes
    };

    log::debug!("[NDI] Frame: {}x{}, data_len={}, stride={}, row_bytes={}",
        width, height, bgra_data.len(), actual_stride, row_bytes);

    for y in 0..height as usize {
        let src_start = y * actual_stride;
        let dst_start = y * row_bytes;
        let src_end = src_start + row_bytes;
        if src_end <= bgra_data.len() {
            out[dst_start..dst_start + row_bytes]
                .copy_from_slice(&bgra_data[src_start..src_end]);
        }
    }

    out
}

/// Global NDI availability check
pub fn is_ndi_available() -> bool {
    NDI::new().is_ok()
}

/// Quick function to list available NDI sources
pub fn list_ndi_sources(timeout_ms: u32) -> Vec<String> {
    NdiSourceFinder::find_sources(timeout_ms)
        .into_iter()
        .map(|info| info.name)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_stride_bgra_no_padding() {
        // Tightly-packed BGRA — should pass through unchanged
        let bgra = vec![
            255, 0, 0, 255,    // pixel 0: B=255, G=0, R=0, A=255
            0, 255, 0, 255,    // pixel 1: B=0,   G=255, R=0, A=255
        ];
        let out = strip_stride_bgra(&bgra, 2, 1);
        assert_eq!(&out, &bgra);
    }

    #[test]
    fn test_strip_stride_bgra_with_padding() {
        // 1-pixel-wide row with 4 bytes of padding per row
        let bgra = vec![
            10, 20, 30, 255,   // pixel 0
            0,  0,  0,  0,     // padding
        ];
        let out = strip_stride_bgra(&bgra, 1, 1);
        assert_eq!(out, vec![10, 20, 30, 255]);
    }
}
