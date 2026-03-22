//! # Async GPU Readback Pool
//!
//! Double-buffered staging pool for non-blocking GPU→CPU readback.
//! The render thread never blocks waiting for a GPU→CPU copy to complete.
//! Each frame we submit a copy into the *current* slot and harvest the
//! *previous* slot's data (which has had a full frame to finish mapping).

use crate::engine::texture::{ReadbackLayout, strip_readback_padding};

/// Number of staging buffer slots. Two is enough: one being filled by the
/// GPU while the CPU reads the other.
const READBACK_SLOTS: usize = 2;

/// State of a single staging buffer slot.
enum SlotState {
    /// Buffer is idle and available for a new copy.
    Available,
    /// A copy has been submitted and `map_async` requested; waiting for GPU.
    Pending {
        buffer: wgpu::Buffer,
        width: u32,
        height: u32,
        ready: std::sync::mpsc::Receiver<bool>,
    },
}

/// Double-buffered staging pool for non-blocking GPU→CPU readback.
pub struct ReadbackPool {
    slots: Vec<SlotState>,
    /// Index of the slot to write into this frame.
    current: usize,
}

impl ReadbackPool {
    pub fn new() -> Self {
        let mut slots = Vec::with_capacity(READBACK_SLOTS);
        for _ in 0..READBACK_SLOTS {
            slots.push(SlotState::Available);
        }
        Self { slots, current: 0 }
    }

    /// Harvest the *previous* slot if its map has completed, returning the
    /// BGRA pixel data. This never blocks — if the GPU hasn't finished yet
    /// we simply skip this frame's readback.
    pub fn harvest_previous(&mut self) -> Option<(Vec<u8>, u32, u32)> {
        let prev = (self.current + READBACK_SLOTS - 1) % READBACK_SLOTS;
        let slot = &mut self.slots[prev];

        match slot {
            SlotState::Pending { buffer, width, height, ready } => {
                match ready.try_recv() {
                    Ok(true) => {
                        let w = *width;
                        let h = *height;
                        let layout = ReadbackLayout::new(w, h);
                        let raw = buffer.slice(..).get_mapped_range().to_vec();
                        buffer.unmap();
                        // Strip wgpu row-alignment padding if present.
                        let data = strip_readback_padding(&raw, layout, h);
                        *slot = SlotState::Available;
                        Some((data, w, h))
                    }
                    _ => None,
                }
            }
            SlotState::Available => None,
        }
    }

    /// Submit a non-blocking copy from `texture` into the current staging
    /// slot and request an async map.
    pub fn submit_copy(
        &mut self,
        texture: &wgpu::Texture,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        let width = texture.width();
        let height = texture.height();
        let layout = ReadbackLayout::new(width, height);

        // If the current slot is still pending (GPU too slow), drop it.
        if matches!(self.slots[self.current], SlotState::Pending { .. }) {
            self.slots[self.current] = SlotState::Available;
            log::debug!("Readback slot {} overwritten (GPU too slow)", self.current);
        }

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Readback Staging"),
            size: layout.buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Readback Copy"),
        });

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(layout.padded_bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        queue.submit(std::iter::once(encoder.finish()));

        // Request async map — the callback signals via channel.
        let (tx, rx) = std::sync::mpsc::channel::<bool>();
        staging_buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result.is_ok());
            });

        self.slots[self.current] = SlotState::Pending {
            buffer: staging_buffer,
            width,
            height,
            ready: rx,
        };

        self.current = (self.current + 1) % READBACK_SLOTS;
    }

    /// Drain any pending slots (used during shutdown / output stop).
    pub fn drain(&mut self, device: &wgpu::Device) {
        for slot in &mut self.slots {
            if matches!(slot, SlotState::Pending { .. }) {
                device.poll(wgpu::PollType::Wait).ok();
                if let SlotState::Pending { buffer, .. } = std::mem::replace(slot, SlotState::Available) {
                    drop(buffer);
                }
            }
        }
    }
}
