//! # Output Module
//!
//! Handles video output to external systems including:
//! - NDI output
//! - Syphon output (macOS)
//! - Spout output (Windows)
//! - Video recording (via FFmpeg)

use crate::engine::texture::{ReadbackLayout, strip_readback_padding};

// NDI output (requires ndi feature)
#[cfg(feature = "ndi")]
pub mod ndi_sender;
#[cfg(feature = "ndi")]
pub use ndi_sender::{NdiOutputSender, is_ndi_output_available};

#[cfg(feature = "ndi")]
pub mod ndi_async;
#[cfg(feature = "ndi")]
pub use ndi_async::AsyncNdiOutput;

// Platform-specific IPC outputs (macOS only, requires syphon feature)
#[cfg(all(target_os = "macos", feature = "syphon"))]
pub mod syphon_sender;
#[cfg(all(target_os = "macos", feature = "syphon"))]
pub use syphon_sender::{SyphonSender, SyphonWgpuSender};

#[cfg(all(target_os = "macos", feature = "syphon"))]
pub mod syphon_async;
#[cfg(all(target_os = "macos", feature = "syphon"))]
pub use syphon_async::{AsyncSyphonOutput, SyphonOutputIntegration};

/// Consolidated output manager for live outputs.
///
/// Holds optional NDI and Syphon outputs and dispatches `submit_frame()`
/// to whichever are currently active.
pub struct OutputManager {
    #[cfg(all(target_os = "macos", feature = "syphon"))]
    syphon: SyphonOutputIntegration,
    #[cfg(feature = "ndi")]
    ndi: Option<NdiOutputSender>,
}

impl OutputManager {
    pub fn new() -> Self {
        Self {
            #[cfg(all(target_os = "macos", feature = "syphon"))]
            syphon: SyphonOutputIntegration::new(),
            #[cfg(feature = "ndi")]
            ndi: None,
        }
    }

    /// Start (or restart) Syphon output.
    #[cfg(all(target_os = "macos", feature = "syphon"))]
    pub fn start_syphon(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        name: &str,
        width: u32,
        height: u32,
    ) -> anyhow::Result<()> {
        self.syphon.enable(device, queue, name, width, height)
    }

    /// Stop Syphon output.
    #[cfg(all(target_os = "macos", feature = "syphon"))]
    pub fn stop_syphon(&mut self) {
        self.syphon.disable();
    }

    /// Whether Syphon output is currently active.
    #[cfg(all(target_os = "macos", feature = "syphon"))]
    pub fn syphon_active(&self) -> bool {
        self.syphon.is_enabled()
    }

    /// Start (or restart) NDI output.
    #[cfg(feature = "ndi")]
    pub fn start_ndi(
        &mut self,
        name: &str,
        width: u32,
        height: u32,
        include_alpha: bool,
    ) -> anyhow::Result<()> {
        let sender = NdiOutputSender::new(name, width, height, include_alpha)?;
        self.ndi = Some(sender);
        Ok(())
    }

    /// Stop NDI output.
    #[cfg(feature = "ndi")]
    pub fn stop_ndi(&mut self) {
        self.ndi = None;
    }

    /// Whether NDI output is currently active.
    #[cfg(feature = "ndi")]
    pub fn ndi_active(&self) -> bool {
        self.ndi.is_some()
    }

    /// Submit a frame to all active outputs.
    ///
    /// `texture` must be in `Bgra8Unorm` format (the pipeline native format).
    pub fn submit_frame(
        &mut self,
        texture: &wgpu::Texture,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        #[cfg(all(target_os = "macos", feature = "syphon"))]
        self.syphon.submit_frame(texture, device, queue);

        // NDI currently requires CPU-side pixel data, so this path performs
        // a synchronous readback and channel swap before enqueueing the frame.
        #[cfg(feature = "ndi")]
        if let Some(ref ndi) = self.ndi {
            let size = texture.size();
            let bgra = Self::read_texture_bgra(device, queue, texture, size.width, size.height);
            let rgba = Self::bgra_to_rgba(&bgra);
            ndi.submit_frame(&rgba, size.width, size.height);
        }
    }

    /// Read a texture's pixel data as BGRA bytes (for CPU-side outputs like NDI).
    ///
    /// This is a synchronous readback — use sparingly (not every frame at 60fps).
    pub fn read_texture_bgra(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
        width: u32,
        height: u32,
    ) -> Vec<u8> {
        let layout = ReadbackLayout::new(width, height);

        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("OutputManager Readback"),
            size: layout.buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("OutputManager Readback Encoder"),
        });
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(layout.padded_bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        );
        queue.submit(std::iter::once(encoder.finish()));

        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        device.poll(wgpu::PollType::Wait).expect("device poll failed");
        let _ = rx.recv();

        let data = slice.get_mapped_range();
        strip_readback_padding(&data, layout, height)
    }

    #[cfg(feature = "ndi")]
    fn bgra_to_rgba(bgra: &[u8]) -> Vec<u8> {
        let mut rgba = Vec::with_capacity(bgra.len());
        for pixel in bgra.chunks_exact(4) {
            rgba.extend_from_slice(&[pixel[2], pixel[1], pixel[0], pixel[3]]);
        }
        rgba
    }
}

impl Default for OutputManager {
    fn default() -> Self {
        Self::new()
    }
}
