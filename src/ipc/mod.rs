//! # Inter-Process Communication Video Module
//!
//! Provides cross-platform video sharing between applications using:
//! - **macOS**: Syphon (OpenGL-based, IOSurface sharing)
//! - **Windows**: Spout (DirectX/OpenGL texture sharing)
//! - **Linux**: v4l2loopback (V4L2 virtual video devices)
//!
//! ## Architecture
//!
//! The module uses a trait-based abstraction with platform-specific implementations:
//!
//! ```
//! IpcInput trait  ───────┐
//!                        ├──► Platform implementations (Syphon/Spout/V4L2)
//! IpcOutput trait ───────┘
//! ```
//!
//! ## Feature Flags
//!
//! - `ipc-syphon` - Enable Syphon support (macOS only)
//! - `ipc-spout` - Enable Spout support (Windows only)
//! - `ipc-v4l2` - Enable v4l2loopback support (Linux only)

use std::fmt::Debug;

/// Result type for IPC operations
pub type IpcResult<T> = Result<T, IpcError>;

/// Error types for IPC operations
#[derive(Debug, Clone)]
pub enum IpcError {
    /// Platform not supported
    PlatformNotSupported,
    /// Server/sender not found
    ServerNotFound(String),
    /// Connection failed
    ConnectionFailed(String),
    /// Texture sharing not available
    SharingNotAvailable,
    /// Invalid dimensions
    InvalidDimensions { width: u32, height: u32 },
    /// Format not supported
    UnsupportedFormat(String),
    /// Native API error
    NativeError(String),
    /// Not initialized
    NotInitialized,
    /// Already connected
    AlreadyConnected,
}

impl std::fmt::Display for IpcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IpcError::PlatformNotSupported => write!(f, "Platform not supported for this IPC method"),
            IpcError::ServerNotFound(name) => write!(f, "Server/sender not found: {}", name),
            IpcError::ConnectionFailed(msg) => write!(f, "Connection failed: {}", msg),
            IpcError::SharingNotAvailable => write!(f, "GPU texture sharing not available"),
            IpcError::InvalidDimensions { width, height } => {
                write!(f, "Invalid dimensions: {}x{}", width, height)
            }
            IpcError::UnsupportedFormat(fmt) => write!(f, "Unsupported format: {}", fmt),
            IpcError::NativeError(msg) => write!(f, "Native API error: {}", msg),
            IpcError::NotInitialized => write!(f, "IPC not initialized"),
            IpcError::AlreadyConnected => write!(f, "Already connected to a source"),
        }
    }
}

impl std::error::Error for IpcError {}

/// Pixel formats supported by IPC
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    /// 8-bit RGB (24 bpp)
    RGB24,
    /// 8-bit RGBA (32 bpp)
    RGBA,
    /// 8-bit BGRA (32 bpp) - Windows native
    BGRA,
    /// YUYV 4:2:2 (16 bpp) - V4L2 common
    YUYV,
    /// NV12 (12 bpp) - DirectX common
    NV12,
}

impl PixelFormat {
    /// Get bytes per pixel (approximate for planar formats)
    pub fn bytes_per_pixel(&self) -> usize {
        match self {
            PixelFormat::RGB24 => 3,
            PixelFormat::RGBA | PixelFormat::BGRA => 4,
            PixelFormat::YUYV => 2,
            PixelFormat::NV12 => 1, // Actually 1.5, use 1 for simplicity
        }
    }

    /// Get total buffer size for dimensions
    pub fn buffer_size(&self, width: u32, height: u32) -> usize {
        match self {
            PixelFormat::RGB24 => (width * height) as usize * 3,
            PixelFormat::RGBA | PixelFormat::BGRA => (width * height) as usize * 4,
            PixelFormat::YUYV => (width * height) as usize * 2,
            PixelFormat::NV12 => (width * height) as usize * 3 / 2, // 12 bits per pixel
        }
    }
}

/// Information about an available IPC source
#[derive(Debug, Clone)]
pub struct IpcSourceInfo {
    /// Human-readable name
    pub name: String,
    /// Application that created it
    pub app_name: String,
    /// Current dimensions (if known)
    pub dimensions: Option<(u32, u32)>,
    /// Native handle for connection
    pub handle: String,
}

/// Frame data received from IPC
pub enum IpcFrame {
    /// CPU-accessible buffer (always available)
    CpuBuffer {
        /// Raw pixel data
        data: Vec<u8>,
        /// Pixel format
        format: PixelFormat,
        /// Width in pixels
        width: u32,
        /// Height in pixels
        height: u32,
    },
    // GPU texture handle (platform-specific, zero-copy)
    // TODO: Implement zero-copy GPU texture handles using syphon-metal types
    // #[cfg(target_os = "macos")]
    // GpuTexture(...),
}

/// Core trait for IPC video input
///
/// Implementations provide platform-specific ways to receive video from other applications.
pub trait IpcInput: Send + Debug {
    /// Connect to a named source/server
    ///
    /// # Arguments
    /// * `source` - Source name/identifier from discovery
    fn connect(&mut self, source: &str) -> IpcResult<()>;

    /// Disconnect from current source
    fn disconnect(&mut self);

    /// Check if connected to a source
    fn is_connected(&self) -> bool;

    /// Receive a frame if available
    ///
    /// Returns `None` if no new frame is available (non-blocking).
    fn receive_frame(&mut self) -> Option<IpcFrame>;

    /// Get current resolution if known
    fn resolution(&self) -> Option<(u32, u32)>;

    /// Get the name of the connected source
    fn source_name(&self) -> Option<&str>;
}

/// Core trait for IPC video output
///
/// Implementations provide platform-specific ways to send video to other applications.
pub trait IpcOutput: Send + Debug {
    /// Create a server/sender with the given name
    ///
    /// # Arguments
    /// * `name` - Unique name for this server/sender
    /// * `width` - Output width in pixels
    /// * `height` - Output height in pixels
    fn create_server(&mut self, name: &str, width: u32, height: u32) -> IpcResult<()>;

    /// Destroy the server/sender
    fn destroy_server(&mut self);

    /// Check if server is active
    fn is_active(&self) -> bool;

    /// Send a frame from a wgpu texture
    ///
    /// This is the preferred method when GPU sharing is available.
    ///
    /// # Arguments
    /// * `texture` - The wgpu texture to share
    /// * `device` - wgpu device
    /// * `queue` - wgpu queue
    fn send_texture(
        &mut self,
        texture: &wgpu::Texture,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> IpcResult<()>;

    /// Send frame data from CPU buffer
    ///
    /// This is the fallback method when GPU sharing is not available.
    ///
    /// # Arguments
    /// * `data` - Raw pixel data
    /// * `format` - Pixel format
    /// * `width` - Width in pixels
    /// * `height` - Height in pixels
    fn send_buffer(
        &mut self,
        data: &[u8],
        format: PixelFormat,
        width: u32,
        height: u32,
    ) -> IpcResult<()>;

    /// Get server name
    fn server_name(&self) -> Option<&str>;

    /// Get current dimensions
    fn dimensions(&self) -> Option<(u32, u32)>;
}

/// Trait for discovering available IPC sources
pub trait IpcDiscovery: Send {
    /// Scan for available sources/servers
    ///
    /// # Arguments
    /// * `timeout_ms` - Maximum time to wait for discovery
    fn discover_sources(&mut self, timeout_ms: u32) -> Vec<IpcSourceInfo>;

    /// Check if a specific source is still available
    fn is_source_available(&self, name: &str) -> bool;
}

/// Platform detection
pub mod platform {
    /// Check if this platform supports Syphon
    pub const HAS_SYPHON: bool = cfg!(target_os = "macos");

    /// Check if this platform supports Spout
    pub const HAS_SPOUT: bool = cfg!(target_os = "windows");

    /// Check if this platform supports v4l2loopback
    pub const HAS_V4L2: bool = cfg!(target_os = "linux");

    /// Get the primary IPC method for this platform
    pub fn primary_ipc_method() -> &'static str {
        if HAS_SYPHON {
            "Syphon (macOS)"
        } else if HAS_SPOUT {
            "Spout (Windows)"
        } else if HAS_V4L2 {
            "v4l2loopback (Linux)"
        } else {
            "None"
        }
    }
}

// Platform-specific implementations
#[cfg(all(target_os = "macos", feature = "ipc-syphon"))]
pub mod syphon;
#[cfg(all(target_os = "windows", feature = "ipc-spout"))]
pub mod spout;
#[cfg(all(target_os = "linux", feature = "ipc-v4l2"))]
pub mod v4l2;

/// Factory functions for creating platform-specific implementations
pub mod factory {
    use super::*;

    /// Create the best available IPC input for this platform
    pub fn create_input() -> Option<Box<dyn IpcInput>> {
        #[cfg(all(target_os = "macos", feature = "ipc-syphon"))]
        {
            return Some(Box::new(syphon::SyphonInput::new()));
        }
        #[cfg(all(target_os = "windows", feature = "ipc-spout"))]
        {
            return Some(Box::new(spout::SpoutInput::new()));
        }
        #[cfg(all(target_os = "linux", feature = "ipc-v4l2"))]
        {
            return Some(Box::new(v4l2::V4L2Input::new()));
        }
        None
    }

    /// Create the best available IPC output for this platform
    pub fn create_output() -> Option<Box<dyn IpcOutput>> {
        #[cfg(all(target_os = "macos", feature = "ipc-syphon"))]
        {
            return Some(Box::new(syphon::SyphonOutput::new()));
        }
        #[cfg(all(target_os = "windows", feature = "ipc-spout"))]
        {
            return Some(Box::new(spout::SpoutOutput::new()));
        }
        #[cfg(all(target_os = "linux", feature = "ipc-v4l2"))]
        {
            return Some(Box::new(v4l2::V4L2Output::new()));
        }
        None
    }

    /// Create the best available discovery for this platform
    pub fn create_discovery() -> Option<Box<dyn IpcDiscovery>> {
        #[cfg(all(target_os = "macos", feature = "ipc-syphon"))]
        {
            return Some(Box::new(syphon::SyphonDiscovery::new()));
        }
        #[cfg(all(target_os = "windows", feature = "ipc-spout"))]
        {
            return Some(Box::new(spout::SpoutDiscovery::new()));
        }
        #[cfg(all(target_os = "linux", feature = "ipc-v4l2"))]
        {
            return Some(Box::new(v4l2::V4L2Discovery::new()));
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pixel_format_sizes() {
        assert_eq!(PixelFormat::RGB24.buffer_size(1920, 1080), 1920 * 1080 * 3);
        assert_eq!(PixelFormat::RGBA.buffer_size(1920, 1080), 1920 * 1080 * 4);
        assert_eq!(PixelFormat::YUYV.buffer_size(1920, 1080), 1920 * 1080 * 2);
    }

    #[test]
    fn test_platform_detection() {
        #[cfg(target_os = "macos")]
        {
            assert!(platform::HAS_SYPHON);
            assert!(!platform::HAS_SPOUT);
            assert!(!platform::HAS_V4L2);
        }
        #[cfg(target_os = "windows")]
        {
            assert!(!platform::HAS_SYPHON);
            assert!(platform::HAS_SPOUT);
            assert!(!platform::HAS_V4L2);
        }
        #[cfg(target_os = "linux")]
        {
            assert!(!platform::HAS_SYPHON);
            assert!(!platform::HAS_SPOUT);
            assert!(platform::HAS_V4L2);
        }
    }
}
