//! # Async Syphon Output (macOS)
//!
//! High-performance Syphon output with zero-copy GPU texture publishing.
//! This module is only compiled when the `syphon` feature is enabled on macOS.

#![cfg(all(target_os = "macos", feature = "syphon"))]
//!
//! ## Architecture
//!
//! This module provides two modes:
//! 1. **Zero-copy mode** (recommended): Uses syphon-wgpu's `SyphonWgpuOutput` for direct
//!    GPU-to-GPU texture sharing via IOSurface.
//! 2. **Async readback mode**: Triple-buffered GPU readback with background Syphon publishing
//!    (fallback for compatibility).
//! 
//! The zero-copy mode is automatically used when a wgpu device is available.

use crate::engine::texture::{ReadbackLayout, strip_readback_padding};
use std::sync::{Arc, Mutex, atomic::{AtomicBool, Ordering}};
use std::thread::{self, JoinHandle};

// Re-export high-performance sender
pub use crate::output::syphon_sender::{SyphonWgpuSender, SyphonSender};

/// Buffer state for async readback mode
#[derive(Clone, Copy, Debug, PartialEq)]
enum BufferState {
    /// Buffer is free and ready for copy
    Free,
    /// Copy submitted to GPU, map_async started
    InFlight,
}

/// Syphon buffer with state tracking
struct TrackedBuffer {
    buffer: Arc<wgpu::Buffer>,
    state: Arc<Mutex<BufferState>>,
}

/// Async Syphon output processor (readback mode)
///
/// Triple-buffered GPU readback with concurrent Syphon publishing.
/// This is the fallback mode - prefer `SyphonWgpuSender` for zero-copy.
pub struct AsyncSyphonOutput {
    /// Triple buffered wgpu buffers
    buffers: Vec<TrackedBuffer>,
    /// Syphon sender (CPU-based)
    syphon_sender: SyphonSender,
    /// Device for polling
    _device: Arc<wgpu::Device>,
    /// Frame dimensions
    width: u32,
    height: u32,
    layout: ReadbackLayout,
    /// Background poll thread
    _poll_thread: Option<JoinHandle<()>>,
    /// Shutdown signal
    running: Arc<AtomicBool>,
}

impl AsyncSyphonOutput {
    /// Create new async Syphon output (readback mode)
    ///
    /// # Arguments
    /// * `device` - wgpu device
    /// * `syphon_sender` - Syphon sender instance
    /// * `width` - Frame width
    /// * `height` - Frame height
    ///
    /// # Deprecated
    /// Use `SyphonWgpuSender` for zero-copy GPU publishing instead.
    pub fn new(
        device: &wgpu::Device,
        syphon_sender: SyphonSender,
        width: u32,
        height: u32,
    ) -> Self {
        // Create triple-buffered readback buffers
        let layout = ReadbackLayout::new(width, height);
        let mut buffers = Vec::with_capacity(3);
        
        for i in 0..3 {
                let buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("Syphon Readback Buffer {}", i)),
                    size: layout.buffer_size,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                }));
            buffers.push(TrackedBuffer {
                buffer,
                state: Arc::new(Mutex::new(BufferState::Free)),
            });
        }
        
        let running = Arc::new(AtomicBool::new(true));
        
        // Start a poll thread
        let device_arc = Arc::new(device.clone());
        let running_poll = Arc::clone(&running);
        let poll_thread = thread::spawn(move || {
            while running_poll.load(Ordering::Relaxed) {
                let _ = device_arc.poll(wgpu::PollType::Poll);
                thread::sleep(std::time::Duration::from_micros(100));
            }
        });
        
        log::info!("[Syphon] Async CPU output created: {}x{} (consider zero-copy mode)", width, height);
        
        Self {
            buffers,
            syphon_sender,
            _device: Arc::new(device.clone()),
            width,
            height,
            layout,
            _poll_thread: Some(poll_thread),
            running,
        }
    }
    
    /// Find a free buffer for copying
    ///
    /// Returns the buffer index and reference if available.
    pub fn acquire_buffer(&self) -> Option<(usize, Arc<wgpu::Buffer>)> {
        for (idx, tracked) in self.buffers.iter().enumerate() {
            let mut state = tracked.state.lock().unwrap();
            if *state == BufferState::Free {
                *state = BufferState::InFlight;
                return Some((idx, Arc::clone(&tracked.buffer)));
            }
        }
        
        // All buffers in flight - log occasionally
        static LAST_WARN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let last = LAST_WARN.load(std::sync::atomic::Ordering::Relaxed);
        if now > last + 5 {
            LAST_WARN.store(now, std::sync::atomic::Ordering::Relaxed);
            log::warn!("[Syphon] All buffers in flight - frame dropped");
        }
        
        None
    }

    pub fn readback_layout(&self) -> ReadbackLayout {
        self.layout
    }
    
    /// Start async processing of a buffer (call AFTER queue.submit)
    ///
    /// This spawns a thread that handles map completion and Syphon publishing.
    pub fn process_buffer_async(&self, buffer_idx: usize) {
        if buffer_idx >= self.buffers.len() {
            return;
        }
        
        let buffer = Arc::clone(&self.buffers[buffer_idx].buffer);
        let state = Arc::clone(&self.buffers[buffer_idx].state);
        let syphon_sender = self.syphon_sender.clone();
        let width = self.width;
        let height = self.height;
        
        // Spawn thread for this buffer's processing
        thread::spawn(move || {
            let slice = buffer.slice(..);
            
            // Channel for map completion
            let (tx, rx) = std::sync::mpsc::channel::<bool>();
            
            // Start async map
            slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result.is_ok());
            });
            
            // Wait for map with polling
            let mut mapped = false;
            for _ in 0..10000 {
                match rx.try_recv() {
                    Ok(true) => {
                        mapped = true;
                        break;
                    }
                    Ok(false) => {
                        log::warn!("[Syphon] Buffer {} map failed", buffer_idx);
                        break;
                    }
                    Err(std::sync::mpsc::TryRecvError::Empty) => {
                        thread::sleep(std::time::Duration::from_micros(100));
                    }
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        break;
                    }
                }
            }
            
            if mapped {
                // Read data
                let data = slice.get_mapped_range();
                let frame_data = strip_readback_padding(&data, ReadbackLayout::new(width, height), height);
                drop(data);
                buffer.unmap();
                
                // Send to Syphon (BGRA data)
                syphon_sender.submit_frame(&frame_data, width, height);
            } else {
                buffer.unmap();
                log::warn!("[Syphon] Buffer {} map timeout", buffer_idx);
            }
            
            // Mark buffer as free
            let mut state = state.lock().unwrap();
            *state = BufferState::Free;
        });
    }
    
    /// Check if all buffers are busy (for monitoring)
    pub fn is_overloaded(&self) -> bool {
        self.buffers.iter().all(|b| {
            *b.state.lock().unwrap() == BufferState::InFlight
        })
    }
    
    /// Get number of free buffers
    pub fn free_buffer_count(&self) -> usize {
        self.buffers.iter().filter(|b| {
            *b.state.lock().unwrap() == BufferState::Free
        }).count()
    }
}

impl Drop for AsyncSyphonOutput {
    fn drop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
        if let Some(handle) = self._poll_thread.take() {
            let _ = handle.join();
        }
    }
}

/// Helper to integrate with the render loop
///
/// This integration uses zero-copy when possible, falling back to
/// async readback mode if needed.
pub struct SyphonOutputIntegration {
    /// Zero-copy wgpu output (preferred)
    zero_copy_output: Option<SyphonWgpuSender>,
    /// Async readback output (fallback)
    async_output: Option<AsyncSyphonOutput>,
    enabled: bool,
    /// Output dimensions
    width: u32,
    height: u32,
}

impl SyphonOutputIntegration {
    /// Create new integration (disabled by default)
    pub fn new() -> Self {
        Self {
            zero_copy_output: None,
            async_output: None,
            enabled: false,
            width: 1920,
            height: 1080,
        }
    }
    
    /// Enable Syphon output with zero-copy
    ///
    /// This is the recommended method - it uses syphon-wgpu's zero-copy
    /// GPU texture sharing for maximum performance.
    pub fn enable(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        name: &str,
        width: u32,
        height: u32,
    ) -> anyhow::Result<()> {
        // Try zero-copy mode first
        match SyphonWgpuSender::new(name, device, queue, width, height) {
            Ok(sender) => {
                self.zero_copy_output = Some(sender);
                self.async_output = None;
                self.enabled = true;
                self.width = width;
                self.height = height;
                log::info!("[Syphon] Output enabled: {} at {}x{} (zero-copy)", name, width, height);
                Ok(())
            }
            Err(e) => {
                log::warn!("[Syphon] Zero-copy failed ({}), falling back to CPU mode", e);
                self.enable_fallback(device, name, width, height)
            }
        }
    }
    
    /// Enable with CPU fallback (async readback)
    fn enable_fallback(
        &mut self,
        device: &wgpu::Device,
        name: &str,
        width: u32,
        height: u32,
    ) -> anyhow::Result<()> {
        let sender = SyphonSender::new(name, width, height)?;
        self.async_output = Some(AsyncSyphonOutput::new(device, sender, width, height));
        self.zero_copy_output = None;
        self.enabled = true;
        self.width = width;
        self.height = height;
        log::info!("[Syphon] Output enabled: {} at {}x{} (CPU fallback)", name, width, height);
        Ok(())
    }
    
    /// Disable output
    pub fn disable(&mut self) {
        self.zero_copy_output = None;
        self.async_output = None;
        self.enabled = false;
        log::info!("[Syphon] Output disabled");
    }
    
    /// Check if enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled && (self.zero_copy_output.is_some() || self.async_output.is_some())
    }
    
    /// Check if using zero-copy mode
    pub fn is_zero_copy(&self) -> bool {
        self.zero_copy_output.as_ref().map_or(false, |o| o.is_zero_copy())
    }
    
    /// Submit a frame (auto-detects zero-copy vs readback)
    ///
    /// For zero-copy mode, this just publishes the texture.
    /// For readback mode, you need to manually manage the buffer lifecycle.
    pub fn submit_frame(&mut self, texture: &wgpu::Texture, device: &wgpu::Device, queue: &wgpu::Queue) {
        // Use zero-copy path if available
        if let Some(ref mut output) = self.zero_copy_output {
            output.publish(texture, device, queue);
            return;
        }
        
        // Fall back to async readback
        let Some(ref output) = self.async_output else {
            return;
        };
        
        let Some((idx, buffer)) = output.acquire_buffer() else {
            return;
        };
        let layout = output.readback_layout();
        
        // Create encoder for copy
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Syphon Copy"),
        });
        
        // Copy texture to buffer
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: buffer.as_ref(),
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(layout.padded_bytes_per_row),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
        
        // Submit and start async processing
        queue.submit(std::iter::once(encoder.finish()));
        output.process_buffer_async(idx);
    }
    
    /// Get client count (for monitoring)
    pub fn client_count(&self) -> usize {
        if let Some(ref output) = self.zero_copy_output {
            output.client_count()
        } else {
            0
        }
    }
    
    /// Check if any clients are connected
    pub fn has_clients(&self) -> bool {
        self.client_count() > 0
    }
}

impl Default for SyphonOutputIntegration {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_state() {
        assert_ne!(BufferState::Free, BufferState::InFlight);
    }

    #[test]
    fn test_integration_creation() {
        let integration = SyphonOutputIntegration::new();
        assert!(!integration.is_enabled());
        assert!(!integration.is_zero_copy());
    }
}
