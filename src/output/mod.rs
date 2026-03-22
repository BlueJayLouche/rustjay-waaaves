//! # Output Module
//!
//! Handles video output to external systems:
//! - NDI output (cross-platform, CPU readback path)
//! - Syphon output (macOS, zero-copy GPU path)
//!
//! GPU readback uses a double-buffered staging pool so the render thread
//! never blocks waiting for a GPU→CPU copy to complete.

pub mod readback;

// NDI output (requires ndi feature)
#[cfg(feature = "ndi")]
pub mod ndi_sender;
#[cfg(feature = "ndi")]
pub use ndi_sender::{NdiOutputSender, is_ndi_output_available};

// Syphon output (macOS only, requires syphon feature)
#[cfg(all(target_os = "macos", feature = "syphon"))]
pub mod syphon_sender;
#[cfg(all(target_os = "macos", feature = "syphon"))]
pub use syphon_sender::{SyphonSender, SyphonWgpuSender};

// Legacy modules kept for compatibility but no longer used by OutputManager
#[cfg(all(target_os = "macos", feature = "syphon"))]
pub mod syphon_async;
#[cfg(all(target_os = "macos", feature = "syphon"))]
pub use syphon_async::{AsyncSyphonOutput, SyphonOutputIntegration};

use readback::ReadbackPool;

// ---------------------------------------------------------------------------
// OutputManager
// ---------------------------------------------------------------------------

/// Manages all video outputs with async readback for CPU-path sinks.
pub struct OutputManager {
    /// NDI network output
    #[cfg(feature = "ndi")]
    ndi: Option<NdiOutputSender>,

    /// Syphon zero-copy GPU output (macOS)
    #[cfg(all(target_os = "macos", feature = "syphon"))]
    syphon: Option<SyphonWgpuSender>,

    /// Async readback pool for CPU-path outputs (NDI).
    readback_pool: ReadbackPool,

    /// Frame skip factor for NDI (0 = every frame, 1 = every 2nd, etc.)
    #[cfg(feature = "ndi")]
    frame_skip: u8,
    #[cfg(feature = "ndi")]
    skip_counter: u8,
}

impl OutputManager {
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "ndi")]
            ndi: None,
            #[cfg(all(target_os = "macos", feature = "syphon"))]
            syphon: None,
            readback_pool: ReadbackPool::new(),
            #[cfg(feature = "ndi")]
            frame_skip: 1,
            #[cfg(feature = "ndi")]
            skip_counter: 0,
        }
    }

    // ── NDI ───────────────────────────────────────────────────────────

    /// Start (or restart) NDI output.
    #[cfg(feature = "ndi")]
    pub fn start_ndi(
        &mut self,
        name: &str,
        width: u32,
        height: u32,
        include_alpha: bool,
        frame_skip: u8,
    ) -> anyhow::Result<()> {
        self.stop_ndi();
        let sender = NdiOutputSender::new(name, width, height, include_alpha)?;
        self.ndi = Some(sender);
        self.frame_skip = frame_skip.max(1);
        self.skip_counter = 0;
        log::info!("NDI output started: {} ({}x{}, alpha={}, skip={})",
            name, width, height, include_alpha, self.frame_skip);
        Ok(())
    }

    /// Stop NDI output.
    #[cfg(feature = "ndi")]
    pub fn stop_ndi(&mut self) {
        if self.ndi.take().is_some() {
            self.frame_skip = 1;
            self.skip_counter = 0;
            log::info!("NDI output stopped");
        }
    }

    /// Whether NDI output is currently active.
    #[cfg(feature = "ndi")]
    pub fn ndi_active(&self) -> bool {
        self.ndi.is_some()
    }

    // ── Syphon ────────────────────────────────────────────────────────

    /// Start (or restart) Syphon output.
    #[cfg(all(target_os = "macos", feature = "syphon"))]
    pub fn start_syphon(
        &mut self,
        name: &str,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
    ) -> anyhow::Result<()> {
        self.stop_syphon();
        let sender = SyphonWgpuSender::new(name, device, queue, width, height)?;
        self.syphon = Some(sender);
        log::info!("Syphon output started: {}", name);
        Ok(())
    }

    /// Stop Syphon output.
    #[cfg(all(target_os = "macos", feature = "syphon"))]
    pub fn stop_syphon(&mut self) {
        if self.syphon.take().is_some() {
            log::info!("Syphon output stopped");
        }
    }

    /// Whether Syphon output is currently active.
    #[cfg(all(target_os = "macos", feature = "syphon"))]
    pub fn syphon_active(&self) -> bool {
        self.syphon.is_some()
    }

    #[cfg(all(target_os = "macos", feature = "syphon"))]
    pub fn syphon_is_zero_copy(&self) -> bool {
        self.syphon.as_ref().map_or(false, |s| s.is_zero_copy())
    }

    // ── Frame submission ──────────────────────────────────────────────

    /// Returns true if any CPU-path output needs readback.
    fn needs_readback(&self) -> bool {
        #[cfg(feature = "ndi")]
        if self.ndi.is_some() {
            return true;
        }
        false
    }

    /// Submit frame to all active outputs.
    ///
    /// GPU-path outputs (Syphon) receive the texture directly.
    /// CPU-path outputs (NDI) use the async readback pool — the
    /// render thread never blocks waiting for a GPU→CPU copy.
    pub fn submit_frame(
        &mut self,
        texture: &wgpu::Texture,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        // CPU-path outputs: harvest previous frame's readback, then
        // submit a new copy for this frame.
        if self.needs_readback() {
            // Non-blocking poll to nudge the GPU.
            device.poll(wgpu::PollType::Poll).ok();

            // Harvest the previous frame (never blocks).
            if let Some((data, width, height)) = self.readback_pool.harvest_previous() {
                #[cfg(feature = "ndi")]
                if let Some(ref ndi) = self.ndi {
                    ndi.submit_frame(&data, width, height);
                }
            }

            // Apply frame skip then submit a new copy.
            #[cfg(feature = "ndi")]
            {
                self.skip_counter = self.skip_counter.wrapping_add(1);
                if self.skip_counter % (self.frame_skip + 1) == 0 {
                    self.readback_pool.submit_copy(texture, device, queue);
                    self.skip_counter = 0;
                }
            }
            #[cfg(not(feature = "ndi"))]
            {
                self.readback_pool.submit_copy(texture, device, queue);
            }
        }

        // Syphon: zero-copy GPU path
        #[cfg(all(target_os = "macos", feature = "syphon"))]
        if let Some(ref mut syphon) = self.syphon {
            syphon.publish(texture, device, queue);
        }
    }

    /// Shutdown all outputs.
    pub fn shutdown(&mut self) {
        #[cfg(feature = "ndi")]
        self.stop_ndi();
        #[cfg(all(target_os = "macos", feature = "syphon"))]
        self.stop_syphon();
    }

    /// Drain readback pool (call when GPU device is still alive).
    pub fn drain_readback(&mut self, device: &wgpu::Device) {
        self.readback_pool.drain(device);
    }
}

impl Default for OutputManager {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for OutputManager {
    fn drop(&mut self) {
        self.shutdown();
    }
}
