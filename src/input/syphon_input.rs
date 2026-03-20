//! # Syphon Input Receiver (macOS)
//!
//! Receives video frames from Syphon servers on the local machine.
//! This module is only compiled when the `syphon` feature is enabled on macOS.

#![cfg(all(target_os = "macos", feature = "syphon"))]
//!
//! This module wraps the syphon-core crate's SyphonClient for integration
//! with the input system. It provides:
//! - Background frame polling
//! - Frame queuing for main thread consumption
//! - Server discovery and caching
//!
//! ## Architecture
//!
//! The implementation uses syphon_core::SyphonClient which handles:
//! - Objective-C runtime interop
//! - IOSurface-based frame delivery
//! - Server directory queries
//!
//! ## Format Note
//!
//! Syphon delivers frames in BGRA format (native macOS IOSurface format).
//! Frames are passed directly to the GPU as BGRA data — no CPU conversion needed.
//! Upload to `Bgra8Unorm` textures for zero-copy path.

use crossbeam::channel::{self, Sender, Receiver};
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// Re-export ServerInfo from external crate
pub use syphon_core::ServerInfo as SyphonServerInfo;

/// A received Syphon frame
pub struct SyphonFrame {
    pub width: u32,
    pub height: u32,
    /// BGRA pixel data (native macOS IOSurface format)
    pub data: Vec<u8>,
    pub timestamp: Instant,
}

/// Syphon input receiver
///
/// Connects to a Syphon server and receives frames in a background thread.
/// Uses syphon_core::SyphonClient for the actual communication.
pub struct SyphonInputReceiver {
    server_name: Option<String>,
    receiver_thread: Option<JoinHandle<()>>,
    frame_tx: Sender<SyphonFrame>,
    frame_rx: Receiver<SyphonFrame>,
    running: Arc<AtomicBool>,
    resolution: (u32, u32),
}

impl SyphonInputReceiver {
    /// Create a new Syphon input receiver
    pub fn new() -> Self {
        // Use bounded channel to prevent memory growth
        let (frame_tx, frame_rx) = channel::bounded(5);
        
        Self {
            server_name: None,
            receiver_thread: None,
            frame_tx,
            frame_rx,
            running: Arc::new(AtomicBool::new(false)),
            resolution: (1920, 1080),
        }
    }
    
    /// Check if Syphon is available
    pub fn is_available() -> bool {
        syphon_core::is_available()
    }
    
    /// Connect to a Syphon server by name
    pub fn connect(&mut self, server_name: impl Into<String>) -> anyhow::Result<()> {
        let server_name = server_name.into();
        
        if self.is_connected() {
            self.disconnect();
        }
        
        // Check if Syphon is available first
        if !Self::is_available() {
            return Err(anyhow::anyhow!("Syphon framework not available"));
        }
        
        log::info!("[Syphon Input] Connecting to server: {}", server_name);
        
        let running = Arc::clone(&self.running);
        running.store(true, Ordering::SeqCst);
        
        let frame_tx = self.frame_tx.clone();
        let name_clone = server_name.clone();
        
        let thread_handle = thread::spawn(move || {
            Self::receive_thread(name_clone, frame_tx, running);
        });
        
        self.server_name = Some(server_name);
        self.receiver_thread = Some(thread_handle);
        
        Ok(())
    }
    
    /// Receive thread that polls SyphonClient
    #[cfg(target_os = "macos")]
    fn receive_thread(
        server_name: String,
        frame_tx: Sender<SyphonFrame>,
        running: Arc<AtomicBool>,
    ) {
        use objc::rc::autoreleasepool;
        
        log::info!("[Syphon Input] Receive thread started for '{}'", server_name);
        
        // Wrap the entire thread in an autoreleasepool
        autoreleasepool(|| {
            // Create Syphon client using external crate
            let client = match SyphonClient::connect(&server_name) {
                Ok(c) => c,
                Err(e) => {
                    log::error!("[Syphon Input] Failed to create client for '{}': {}", server_name, e);
                    return;
                }
            };
            
            log::info!("[Syphon Input] Client created for '{}'", server_name);
            let mut frame_count = 0u64;
            
            while running.load(Ordering::SeqCst) {
                // Try to receive a frame (non-blocking)
                match client.try_receive() {
                    Ok(Some(mut frame)) => {
                        // Convert IOSurface to CPU buffer
                        match frame.to_vec() {
                            Ok(bgra_data) => {
                                // Pass BGRA data directly — native macOS IOSurface format,
                                // matches Bgra8Unorm texture format. No CPU conversion needed.
                                let syphon_frame = SyphonFrame {
                                    width: frame.width,
                                    height: frame.height,
                                    data: bgra_data,
                                    timestamp: Instant::now(),
                                };
                                
                                // Send to main thread (non-blocking, drop if queue full)
                                frame_count += 1;
                                if frame_tx.try_send(syphon_frame).is_ok() {
                                    if frame_count <= 5 || frame_count % 60 == 0 {
                                        log::info!("[Syphon Input] Frame {} sent: {}x{}", frame_count, frame.width, frame.height);
                                    }
                                } else {
                                    log::debug!("[Syphon Input] Frame dropped - queue full");
                                }
                            }
                            Err(e) => {
                                log::warn!("[Syphon Input] Failed to read frame data: {}", e);
                            }
                        }
                    }
                    Ok(None) => {
                        // No new frame available
                    }
                    Err(e) => {
                        log::warn!("[Syphon Input] Error receiving frame: {}", e);
                    }
                }
                
                // Small sleep to prevent busy-waiting
                // 100μs gives us up to ~10kHz polling which should catch 60fps easily
                thread::sleep(Duration::from_micros(100));
            }
            
            log::info!("[Syphon Input] Receive thread stopped for '{}'", server_name);
        });
    }
    
    /// Receive thread stub for non-macOS
    #[cfg(not(target_os = "macos"))]
    fn receive_thread(
        server_name: String,
        _frame_tx: Sender<SyphonFrame>,
        running: Arc<AtomicBool>,
    ) {
        log::warn!("[Syphon Input] Not available on this platform for '{}'", server_name);
        
        while running.load(Ordering::SeqCst) {
            thread::sleep(Duration::from_millis(100));
        }
    }
    
    /// Disconnect from current server
    pub fn disconnect(&mut self) {
        if !self.is_connected() {
            return;
        }
        
        log::info!("[Syphon Input] Disconnecting from: {:?}", self.server_name);
        
        self.running.store(false, Ordering::SeqCst);
        
        if let Some(handle) = self.receiver_thread.take() {
            let _ = handle.join();
        }
        
        // Clear any pending frames
        while self.frame_rx.try_recv().is_ok() {}
        
        self.server_name = None;
    }
    
    /// Check if connected
    pub fn is_connected(&self) -> bool {
        self.server_name.is_some()
    }
    
    /// Get the latest frame (non-blocking, consumes the frame)
    pub fn get_latest_frame(&mut self) -> Option<SyphonFrame> {
        // Drain all frames and return only the most recent
        let mut latest: Option<SyphonFrame> = None;
        let mut count = 0;
        
        while let Ok(frame) = self.frame_rx.try_recv() {
            self.resolution = (frame.width, frame.height);
            latest = Some(frame);
            count += 1;
        }
        
        if count > 0 {
            log::debug!("[Syphon Input] Retrieved {} frame(s) from queue", count);
        }
        
        latest
    }
    
    /// Check if a new frame is available (approximate)
    pub fn has_frame(&self) -> bool {
        !self.frame_rx.is_empty()
    }
    
    /// Get current resolution
    pub fn resolution(&self) -> (u32, u32) {
        self.resolution
    }
    
    /// Get connected server name
    pub fn server_name(&self) -> Option<&str> {
        self.server_name.as_deref()
    }
}

impl Default for SyphonInputReceiver {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for SyphonInputReceiver {
    fn drop(&mut self) {
        self.disconnect();
    }
}

/// Syphon server discovery
///
/// Scans for available Syphon servers on the local machine.
/// Wraps syphon_core::SyphonServerDirectory.
pub struct SyphonDiscovery;

impl SyphonDiscovery {
    /// Create new discovery
    pub fn new() -> Self {
        Self
    }
    
    /// Discover available Syphon servers
    pub fn discover_servers(&self) -> Vec<SyphonServerInfo> {
        log::debug!("[Syphon] Discovering servers...");
        
        // Check if Syphon is available before trying to discover
        if !SyphonInputReceiver::is_available() {
            log::warn!("[Syphon] Framework not available, skipping discovery");
            return Vec::new();
        }
        
        let servers = SyphonServerDirectory::servers();
        
        log::info!("[Syphon] Discovered {} servers", servers.len());
        for server in &servers {
            log::debug!("  - {} ({})", server.name, server.app_name);
        }
        
        servers
    }
    
    /// Check if a specific server is still available
    pub fn is_server_available(&self, name: &str) -> bool {
        // Safety check
        if !syphon_core::is_available() {
            return false;
        }
        SyphonServerDirectory::server_exists(name)
    }
}

impl Default for SyphonDiscovery {
    fn default() -> Self {
        Self::new()
    }
}

/// Integration with the input system
///
/// Wraps SyphonInputReceiver with convenience methods for the engine.
pub struct SyphonInputIntegration {
    receiver: Option<SyphonInputReceiver>,
    discovery: SyphonDiscovery,
    cached_servers: Vec<SyphonServerInfo>,
    last_discovery: Option<Instant>,
}

impl SyphonInputIntegration {
    /// Create new integration
    pub fn new() -> Self {
        Self {
            receiver: None,
            discovery: SyphonDiscovery::new(),
            cached_servers: Vec::new(),
            last_discovery: None,
        }
    }
    
    /// Check if Syphon is available
    pub fn is_available() -> bool {
        SyphonInputReceiver::is_available()
    }
    
    /// Refresh the list of available servers
    pub fn refresh_servers(&mut self) {
        self.cached_servers = self.discovery.discover_servers();
        self.last_discovery = Some(Instant::now());
    }
    
    /// Get cached server list
    pub fn servers(&self) -> &[SyphonServerInfo] {
        &self.cached_servers
    }
    
    /// Connect to a server by name
    pub fn connect(&mut self, server_name: &str) -> anyhow::Result<()> {
        if self.receiver.is_some() {
            self.disconnect();
        }
        
        let mut receiver = SyphonInputReceiver::new();
        receiver.connect(server_name)?;
        self.receiver = Some(receiver);
        
        Ok(())
    }
    
    /// Disconnect from current server
    pub fn disconnect(&mut self) {
        self.receiver = None;
    }
    
    /// Check if connected
    pub fn is_connected(&self) -> bool {
        self.receiver.as_ref().map_or(false, |r| r.is_connected())
    }
    
    /// Get latest frame data for GPU upload
    pub fn get_frame_data(&mut self) -> Option<(u32, u32, Vec<u8>)> {
        let receiver = self.receiver.as_mut()?;
        let frame = receiver.get_latest_frame()?;
        Some((frame.width, frame.height, frame.data))
    }
    
    /// Update (called each frame by engine)
    pub fn update(&mut self) {
        // Auto-refresh discovery every 5 seconds
        if self.last_discovery.map_or(true, |t| t.elapsed().as_secs() > 5) {
            self.refresh_servers();
        }
    }
    
    /// Get connected server name
    pub fn connected_server(&self) -> Option<&str> {
        self.receiver.as_ref()?.server_name()
    }
}

impl Default for SyphonInputIntegration {
    fn default() -> Self {
        Self::new()
    }
}

// Re-export syphon_core types that input users might need
pub use syphon_core::{SyphonClient, SyphonServerDirectory};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_receiver_creation() {
        let receiver = SyphonInputReceiver::new();
        assert!(!receiver.is_connected());
    }

    #[test]
    fn test_discovery_creation() {
        let discovery = SyphonDiscovery::new();
        let servers = discovery.discover_servers();
        // Should return empty list on non-macOS or when no servers available
        println!("Found {} servers", servers.len());
    }

    #[test]
    fn test_integration_creation() {
        let integration = SyphonInputIntegration::new();
        assert!(!integration.is_connected());
        assert!(integration.servers().is_empty());
    }

}
