//! # V4L2 Loopback Implementation (Linux)
//!
//! Provides video sharing on Linux via v4l2loopback kernel module.
//!
//! ## Overview
//!
//! v4l2loopback creates virtual video devices that appear as regular cameras
//! to V4L2-compatible applications. This allows video to be shared between
//! applications using standard Linux video APIs.
//!
//! ## Prerequisites
//!
//! ```bash
//! # Install v4l2loopback kernel module
//! sudo apt-get install v4l2loopback-dkms  # Debian/Ubuntu
//! sudo modprobe v4l2loopback devices=2     # Create 2 virtual devices
//! ```
//!
//! ## Architecture
//!
//! Unlike Syphon/Spout, v4l2loopback uses CPU buffers and requires format
//! conversion. The typical path is:
//!
//! ```
//! RustJay (wgpu/Vulkan) ────► GPU Readback ────► RGB Buffer ────► /dev/videoX
//!                                                    │
//!                                              (YUYV conversion)
//! ```

use super::{IpcDiscovery, IpcError, IpcFrame, IpcInput, IpcOutput, IpcResult, IpcSourceInfo, PixelFormat};
use crate::engine::texture::{ReadbackLayout, strip_readback_padding};
use std::fmt::Debug;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::PathBuf;

/// Default v4l2 devices to scan
const V4L2_DEVICE_PATHS: &[&str] = &[
    "/dev/video0",
    "/dev/video1",
    "/dev/video2",
    "/dev/video3",
    "/dev/video4",
    "/dev/video5",
    "/dev/video6",
    "/dev/video7",
];

/// V4L2 input (capture from virtual device)
///
/// Reads video frames from a v4l2loopback device or physical camera.
pub struct V4L2Input {
    /// Device path (e.g., "/dev/video0")
    device_path: Option<String>,
    /// File handle
    #[allow(dead_code)]
    file: Option<File>,
    /// Current dimensions
    dimensions: Option<(u32, u32)>,
    /// Pixel format
    format: PixelFormat,
    /// Buffer for reading
    buffer: Vec<u8>,
}

impl V4L2Input {
    /// Create a new V4L2 input
    pub fn new() -> Self {
        Self {
            device_path: None,
            file: None,
            dimensions: None,
            format: PixelFormat::RGB24,
            buffer: Vec::new(),
        }
    }

    /// Check if v4l2 is available on this system
    pub fn is_available() -> bool {
        // Check if any video devices exist
        V4L2_DEVICE_PATHS.iter().any(|path| std::path::Path::new(path).exists())
    }

    /// Get list of available V4L2 devices
    pub fn list_devices() -> Vec<(String, String)> {
        let mut devices = Vec::new();

        for path in V4L2_DEVICE_PATHS {
            if std::path::Path::new(path).exists() {
                // Try to get device name from /sys or ioctl
                let name = format!("V4L2 Device {}", path);
                devices.push((path.to_string(), name));
            }
        }

        devices
    }

    /// Set desired capture format
    pub fn set_format(&mut self, format: PixelFormat) {
        self.format = format;
    }
}

impl Debug for V4L2Input {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("V4L2Input")
            .field("device_path", &self.device_path)
            .field("dimensions", &self.dimensions)
            .field("format", &self.format)
            .field("connected", &self.is_connected())
            .finish()
    }
}

impl IpcInput for V4L2Input {
    fn connect(&mut self, source: &str) -> IpcResult<()> {
        if self.is_connected() {
            return Err(IpcError::AlreadyConnected);
        }

        log::info!("[V4L2] Opening device: {}", source);

        // Open the V4L2 device
        let file = OpenOptions::new()
            .read(true)
            .write(false)
            .open(source)
            .map_err(|e| IpcError::ConnectionFailed(e.to_string()))?;

        // TODO: Configure V4L2 format using ioctl:
        // 1. VIDIOC_G_FMT to get current format
        // 2. VIDIOC_S_FMT to set desired format
        // 3. VIDIOC_REQBUFS to allocate buffers
        // 4. VIDIOC_QBUF/STREAMON to start capture

        self.device_path = Some(source.to_string());
        self.file = Some(file);

        Ok(())
    }

    fn disconnect(&mut self) {
        if !self.is_connected() {
            return;
        }

        log::info!("[V4L2] Closing device: {:?}", self.device_path);

        // TODO: Stop streaming with VIDIOC_STREAMOFF

        self.device_path = None;
        self.file = None;
        self.dimensions = None;
    }

    fn is_connected(&self) -> bool {
        self.device_path.is_some()
    }

    fn receive_frame(&mut self) -> Option<IpcFrame> {
        if !self.is_connected() {
            return None;
        }

        let (width, height) = self.dimensions?;
        let expected_size = self.format.buffer_size(width, height);

        // Resize buffer if needed
        if self.buffer.len() < expected_size {
            self.buffer.resize(expected_size, 0);
        }

        // TODO: Read frame using V4L2:
        // 1. VIDIOC_DQBUF to dequeue a buffer
        // 2. Read data from mmap'd buffer
        // 3. VIDIOC_QBUF to requeue the buffer

        // For now, return None (placeholder)
        // In real implementation, read into self.buffer

        // Some(IpcFrame::CpuBuffer {
        //     data: self.buffer.clone(),
        //     format: self.format,
        //     width,
        //     height,
        // })

        None
    }

    fn resolution(&self) -> Option<(u32, u32)> {
        self.dimensions
    }

    fn source_name(&self) -> Option<&str> {
        self.device_path.as_deref()
    }
}

impl Default for V4L2Input {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for V4L2Input {
    fn drop(&mut self) {
        self.disconnect();
    }
}

/// V4L2 output (write to virtual device)
///
/// Writes video frames to a v4l2loopback virtual device.
pub struct V4L2Output {
    /// Device path (e.g., "/dev/video0")
    device_path: Option<String>,
    /// File handle
    #[allow(dead_code)]
    file: Option<File>,
    /// Current dimensions
    dimensions: Option<(u32, u32)>,
    /// Pixel format
    format: PixelFormat,
    /// Whether device is a loopback device
    is_loopback: bool,
}

impl V4L2Output {
    /// Create a new V4L2 output
    pub fn new() -> Self {
        Self {
            device_path: None,
            file: None,
            dimensions: None,
            format: PixelFormat::RGB24,
            is_loopback: false,
        }
    }

    /// Check if v4l2loopback module is loaded
    pub fn is_loopback_available() -> bool {
        // Check if loopback module is loaded
        std::path::Path::new("/sys/module/v4l2loopback").exists()
    }

    /// Check if v4l2 is available
    pub fn is_available() -> bool {
        V4L2Input::is_available()
    }

    /// Get list of available output devices (loopback devices only)
    pub fn list_output_devices() -> Vec<(String, String)> {
        V4L2Input::list_devices()
            .into_iter()
            .filter(|(path, _)| {
                // Check if it's a loopback device by looking at driver
                // In /sys/class/video4linux/videoX/device/driver
                Self::is_loopback_device(path)
            })
            .collect()
    }

    fn is_loopback_device(_path: &str) -> bool {
        // TODO: Check if device uses v4l2loopback driver
        // For now, assume all video devices are valid outputs
        true
    }

    /// Set output format
    pub fn set_format(&mut self, format: PixelFormat) {
        self.format = format;
    }
}

impl Debug for V4L2Output {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("V4L2Output")
            .field("device_path", &self.device_path)
            .field("dimensions", &self.dimensions)
            .field("format", &self.format)
            .field("active", &self.is_active())
            .field("is_loopback", &self.is_loopback)
            .finish()
    }
}

impl IpcOutput for V4L2Output {
    fn create_server(&mut self, name: &str, width: u32, height: u32) -> IpcResult<()> {
        if self.is_active() {
            self.destroy_server();
        }

        if width == 0 || height == 0 {
            return Err(IpcError::InvalidDimensions { width, height });
        }

        // For V4L2, the "name" is actually the device path
        let device_path = if name.starts_with("/dev/video") {
            name.to_string()
        } else {
            // Find first available loopback device
            let devices = Self::list_output_devices();
            devices
                .first()
                .map(|(path, _)| path.clone())
                .ok_or_else(|| IpcError::ServerNotFound(
                    "No v4l2loopback devices available. Load module with: sudo modprobe v4l2loopback".to_string()
                ))?
        };

        log::info!("[V4L2] Opening output device '{}' at {}x{}", device_path, width, height);

        // Open device for writing
        let file = OpenOptions::new()
            .read(false)
            .write(true)
            .open(&device_path)
            .map_err(|e| IpcError::ConnectionFailed(e.to_string()))?;

        // TODO: Configure output format with VIDIOC_S_FMT

        self.device_path = Some(device_path);
        self.file = Some(file);
        self.dimensions = Some((width, height));
        self.is_loopback = true;

        Ok(())
    }

    fn destroy_server(&mut self) {
        if !self.is_active() {
            return;
        }

        log::info!("[V4L2] Closing output device: {:?}", self.device_path);

        // TODO: VIDIOC_STREAMOFF if streaming

        self.device_path = None;
        self.file = None;
        self.dimensions = None;
        self.is_loopback = false;
    }

    fn is_active(&self) -> bool {
        self.device_path.is_some()
    }

    fn send_texture(
        &mut self,
        texture: &wgpu::Texture,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> IpcResult<()> {
        if !self.is_active() {
            return Err(IpcError::NotInitialized);
        }

        // V4L2 doesn't support direct GPU sharing
        // We need to readback to CPU first

        let (width, height) = self.dimensions.unwrap();
        let layout = ReadbackLayout::new(width, height);

        // Create readback buffer
        let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("V4L2 Readback"),
            size: layout.buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Copy texture to buffer
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("V4L2 Copy"),
        });

        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &readback_buffer,
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

        // Map and read (blocking for simplicity, could be async)
        let buffer_slice = readback_buffer.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::Wait);

        let data = buffer_slice.get_mapped_range();
        let rgba = strip_readback_padding(&data, layout, height);

        // Convert format and send
        let result = self.send_buffer(&rgba, PixelFormat::RGBA, width, height);

        drop(data);
        readback_buffer.unmap();

        result
    }

    fn send_buffer(
        &mut self,
        data: &[u8],
        format: PixelFormat,
        width: u32,
        height: u32,
    ) -> IpcResult<()> {
        if !self.is_active() {
            return Err(IpcError::NotInitialized);
        }

        // Validate dimensions match
        if let Some((w, h)) = self.dimensions {
            if w != width || h != height {
                return Err(IpcError::InvalidDimensions { width, height });
            }
        }

        let file = self.file.as_mut().ok_or(IpcError::NotInitialized)?;

        // Convert to output format if needed
        let output_data = if format != self.format {
            convert_pixel_format(data, format, self.format, width, height)
        } else {
            data.to_vec()
        };

        // Write to V4L2 device
        // For v4l2loopback, we simply write the frame data
        file.write_all(&output_data)
            .map_err(|e| IpcError::NativeError(e.to_string()))?;

        log::debug!("[V4L2] Wrote frame: {}x{} {:?}", width, height, self.format);

        Ok(())
    }

    fn server_name(&self) -> Option<&str> {
        self.device_path.as_deref()
    }

    fn dimensions(&self) -> Option<(u32, u32)> {
        self.dimensions
    }
}

impl Default for V4L2Output {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for V4L2Output {
    fn drop(&mut self) {
        self.destroy_server();
    }
}

/// V4L2 device discovery
pub struct V4L2Discovery {
    /// Cache of discovered devices
    devices: Vec<IpcSourceInfo>,
}

impl V4L2Discovery {
    /// Create a new V4L2 discovery
    pub fn new() -> Self {
        Self {
            devices: Vec::new(),
        }
    }
}

impl Default for V4L2Discovery {
    fn default() -> Self {
        Self::new()
    }
}

impl IpcDiscovery for V4L2Discovery {
    fn discover_sources(&mut self, _timeout_ms: u32) -> Vec<IpcSourceInfo> {
        log::debug!("[V4L2] Discovering devices...");

        let devices = V4L2Input::list_devices();
        self.devices = devices
            .into_iter()
            .map(|(path, name)| IpcSourceInfo {
                name: name.clone(),
                app_name: "V4L2 Device".to_string(),
                dimensions: None, // Would need VIDIOC_G_FMT to get
                handle: path,
            })
            .collect();

        self.devices.clone()
    }

    fn is_source_available(&self, name: &str) -> bool {
        self.devices.iter().any(|s| s.name == name || s.handle == name)
    }
}

/// Convert between pixel formats
fn convert_pixel_format(
    input: &[u8],
    from: PixelFormat,
    to: PixelFormat,
    width: u32,
    height: u32,
) -> Vec<u8> {
    if from == to {
        return input.to_vec();
    }

    // Simple RGBA->RGB24 conversion
    match (from, to) {
        (PixelFormat::RGBA, PixelFormat::RGB24) => {
            let mut output = Vec::with_capacity((width * height * 3) as usize);
            for chunk in input.chunks_exact(4) {
                output.push(chunk[0]); // R
                output.push(chunk[1]); // G
                output.push(chunk[2]); // B
            }
            output
        }
        (PixelFormat::RGBA, PixelFormat::YUYV) => {
            // RGBA to YUYV conversion
            // This is a simplified version
            let pixel_count = (width * height) as usize;
            let mut output = Vec::with_capacity(pixel_count * 2);
            // TODO: Proper YUV conversion
            output
        }
        _ => {
            log::warn!("[V4L2] Unsupported format conversion: {:?} -> {:?}", from, to);
            input.to_vec()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_v4l2_input_creation() {
        let input = V4L2Input::new();
        assert!(!input.is_connected());
        assert_eq!(input.source_name(), None);
    }

    #[test]
    fn test_v4l2_output_creation() {
        let output = V4L2Output::new();
        assert!(!output.is_active());
        assert_eq!(output.server_name(), None);
    }

    #[test]
    fn test_device_listing() {
        let devices = V4L2Input::list_devices();
        // Should at least not panic
        println!("Found {} V4L2 devices", devices.len());
    }

    #[test]
    fn test_format_conversion() {
        // RGBA -> RGB24
        let rgba = vec![255, 0, 0, 255, 0, 255, 0, 255]; // 2 red and green pixels
        let rgb24 = convert_pixel_format(&rgba, PixelFormat::RGBA, PixelFormat::RGB24, 2, 1);
        assert_eq!(rgb24, vec![255, 0, 0, 0, 255, 0]);
    }
}
