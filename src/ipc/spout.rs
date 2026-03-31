//! # Spout Implementation (Windows)
//!
//! Provides Spout2 video sharing on Windows by directly implementing the
//! Spout shared-memory protocol — no Spout SDK or DLL required.
//!
//! ## How Spout Works
//!
//! Spout senders register in two Windows named shared-memory mappings:
//!   - `"SpoutSenderNames"` — flat array of `char[256]` name slots (no header)
//!   - `"<sender_name>"`    — per-sender `SharedTextureInfo` (280 bytes)
//!
//! The per-sender info contains the DXGI shared handle. We create a standalone
//! D3D11 device (wgpu uses D3D12) for texture sharing with keyed mutex
//! synchronization.

use super::{IpcDiscovery, IpcError, IpcFrame, IpcInput, IpcOutput, IpcResult, IpcSourceInfo, PixelFormat};
use std::fmt::Debug;

use windows::core::Interface;
use windows::Win32::Foundation::{CloseHandle, HANDLE, HMODULE, INVALID_HANDLE_VALUE};
use windows::Win32::Graphics::Direct3D::D3D_DRIVER_TYPE_HARDWARE;
use windows::Win32::Graphics::Direct3D11::{
    D3D11CreateDevice, D3D11_BIND_RENDER_TARGET, D3D11_BIND_SHADER_RESOURCE,
    D3D11_CPU_ACCESS_READ, D3D11_CREATE_DEVICE_FLAG, D3D11_MAP_READ,
    D3D11_MAPPED_SUBRESOURCE, D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX, D3D11_SDK_VERSION,
    D3D11_TEXTURE2D_DESC, D3D11_USAGE_DEFAULT, D3D11_USAGE_STAGING,
    ID3D11Device, ID3D11DeviceContext, ID3D11Texture2D,
};
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_SAMPLE_DESC};
use windows::Win32::Graphics::Dxgi::{IDXGIKeyedMutex, IDXGIResource};
use windows::Win32::System::Memory::{
    CreateFileMappingA, FILE_MAP_ALL_ACCESS, FILE_MAP_READ, MEMORY_BASIC_INFORMATION,
    MapViewOfFile, OpenFileMappingA, PAGE_READWRITE, UnmapViewOfFile, VirtualQuery,
};

// ---------------------------------------------------------------------------
// Spout2 shared-memory layout constants
// ---------------------------------------------------------------------------

/// Max bytes per sender name (including null terminator).
const SPOUT_MAX_NAME_LEN: usize = 256;

/// Default max senders (Spout2 reads this from the registry, fallback = 64).
const SPOUT_MAX_SENDERS: usize = 64;

/// Per-sender info struct — matches Spout2 SDK `SharedTextureInfo`.
///
/// ```text
/// offset  0: shareHandle  (u32)  — DXGI handle via HandleToLong()
/// offset  4: width        (u32)
/// offset  8: height       (u32)
/// offset 12: format       (u32)  — DXGI_FORMAT enum value
/// offset 16: usage        (u32)  — adapter index / usage
/// offset 20: description  [u8; 256] — sender description / exe path
/// offset 276: partnerId   (u32)
/// total: 280 bytes
/// ```
#[repr(C)]
struct SharedTextureInfo {
    share_handle: u32,
    width: u32,
    height: u32,
    format: u32,
    usage: u32,
    description: [u8; 256],
    partner_id: u32,
}

/// DXGI format constants for Spout2 protocol compatibility.
pub mod dxgi_format {
    pub const B8G8R8A8_UNORM: u32 = 87;
    pub const R8G8B8A8_UNORM: u32 = 28;
}

/// Convert pixel format to DirectX DXGI format
pub fn pixel_format_to_dxgi(format: PixelFormat) -> u32 {
    match format {
        PixelFormat::RGBA => dxgi_format::R8G8B8A8_UNORM,
        PixelFormat::BGRA => dxgi_format::B8G8R8A8_UNORM,
        PixelFormat::NV12 => 103,
        PixelFormat::RGB24 | PixelFormat::YUYV => dxgi_format::R8G8B8A8_UNORM,
    }
}

// ---------------------------------------------------------------------------
// Shared-memory helpers
// ---------------------------------------------------------------------------

/// Read width/height from the per-sender named shared memory block.
unsafe fn read_sender_dimensions(name: &str) -> (u32, u32) {
    let Ok(cname) = std::ffi::CString::new(name) else {
        return (0, 0);
    };
    let Ok(hmap) = OpenFileMappingA(
        FILE_MAP_READ.0,
        false,
        windows::core::PCSTR(cname.as_ptr() as *const u8),
    ) else {
        return (0, 0);
    };

    let view = MapViewOfFile(hmap, FILE_MAP_READ, 0, 0, 0);
    let result = if !view.Value.is_null() {
        let info = &*(view.Value as *const SharedTextureInfo);
        let dims = (info.width, info.height);
        UnmapViewOfFile(view).ok();
        dims
    } else {
        (0, 0)
    };
    CloseHandle(hmap).ok();
    result
}

/// Read share handle and dimensions from a sender's named shared-memory block.
unsafe fn read_sender_info(name: &str) -> Result<(HANDLE, u32, u32), IpcError> {
    let cname = std::ffi::CString::new(name)
        .map_err(|_| IpcError::NativeError(format!("Invalid sender name: {}", name)))?;
    let hmap = OpenFileMappingA(
        FILE_MAP_READ.0,
        false,
        windows::core::PCSTR(cname.as_ptr() as *const u8),
    )
    .map_err(|_| IpcError::ServerNotFound(name.to_string()))?;

    let view = MapViewOfFile(hmap, FILE_MAP_READ, 0, 0, 0);
    if view.Value.is_null() {
        CloseHandle(hmap).ok();
        return Err(IpcError::NativeError(format!(
            "MapViewOfFile failed for sender '{}'",
            name
        )));
    }

    let info = &*(view.Value as *const SharedTextureInfo);
    let handle = HANDLE(info.share_handle as usize as *mut _);
    let width = info.width;
    let height = info.height;

    log::debug!(
        "[Spout] Sender '{}': handle=0x{:08x}, {}x{}, fmt={}",
        name,
        info.share_handle,
        width,
        height,
        info.format
    );

    UnmapViewOfFile(view).ok();
    CloseHandle(hmap).ok();
    Ok((handle, width, height))
}

// ---------------------------------------------------------------------------
// SpoutInput — implements IpcInput
// ---------------------------------------------------------------------------

/// Spout input receiver using D3D11 shared textures.
///
/// Opens the sender's shared D3D11 texture via its DXGI handle, copies to a
/// staging texture each frame, and reads BGRA pixels into a CPU buffer.
pub struct SpoutInput {
    d3d_device: ID3D11Device,
    d3d_context: ID3D11DeviceContext,
    sender_name: Option<String>,
    shared_texture: Option<ID3D11Texture2D>,
    staging_texture: Option<ID3D11Texture2D>,
    resolution: (u32, u32),
    pixel_buffer: Vec<u8>,
}

// Safety: D3D11 device/context are thread-safe when used from one thread
unsafe impl Send for SpoutInput {}

impl SpoutInput {
    /// Create a new Spout input receiver with its own D3D11 device.
    pub fn new() -> Self {
        unsafe {
            let mut device = None;
            let mut context = None;
            D3D11CreateDevice(
                None,
                D3D_DRIVER_TYPE_HARDWARE,
                HMODULE::default(),
                D3D11_CREATE_DEVICE_FLAG(0),
                None,
                D3D11_SDK_VERSION,
                Some(&mut device),
                None,
                Some(&mut context),
            )
            .expect("[Spout] SpoutInput: D3D11CreateDevice failed");

            log::info!("[Spout] SpoutInput: D3D11 device created");
            Self {
                d3d_device: device.expect("D3D11 device"),
                d3d_context: context.expect("D3D11 context"),
                sender_name: None,
                shared_texture: None,
                staging_texture: None,
                resolution: (0, 0),
                pixel_buffer: Vec::new(),
            }
        }
    }

    /// Open (or re-open) the shared D3D11 texture for the connected sender.
    fn open_shared_texture(&mut self) -> IpcResult<()> {
        let sender_name = self
            .sender_name
            .as_deref()
            .ok_or(IpcError::NotInitialized)?
            .to_string();

        unsafe {
            let (handle, width, height) = read_sender_info(&sender_name)?;

            if width == 0 || height == 0 {
                return Err(IpcError::InvalidDimensions { width, height });
            }
            if handle.0.is_null() {
                return Err(IpcError::NativeError(format!(
                    "Sender '{}' has null share handle",
                    sender_name
                )));
            }

            // Open the shared texture on our D3D11 device
            let mut shared_tex: Option<ID3D11Texture2D> = None;
            self.d3d_device
                .OpenSharedResource(handle, &mut shared_tex)
                .map_err(|e| IpcError::NativeError(format!("OpenSharedResource: {:?}", e)))?;
            let shared_tex = shared_tex.ok_or_else(|| {
                IpcError::NativeError("OpenSharedResource returned None".into())
            })?;

            // Create a CPU-readable staging texture
            let staging_desc = D3D11_TEXTURE2D_DESC {
                Width: width,
                Height: height,
                MipLevels: 1,
                ArraySize: 1,
                Format: DXGI_FORMAT_B8G8R8A8_UNORM,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Usage: D3D11_USAGE_STAGING,
                BindFlags: 0,
                CPUAccessFlags: D3D11_CPU_ACCESS_READ.0 as u32,
                MiscFlags: 0,
            };
            let mut staging = None;
            self.d3d_device
                .CreateTexture2D(&staging_desc, None, Some(&mut staging))
                .map_err(|e| IpcError::NativeError(format!("CreateTexture2D staging: {:?}", e)))?;
            let staging = staging.ok_or_else(|| {
                IpcError::NativeError("CreateTexture2D staging returned None".into())
            })?;

            log::info!(
                "[Spout] Opened shared texture {}x{} from '{}' (handle={:?})",
                width, height, sender_name, handle
            );

            self.shared_texture = Some(shared_tex);
            self.staging_texture = Some(staging);
            self.resolution = (width, height);
        }
        Ok(())
    }

    /// Poll for a new frame: copy shared texture → staging → CPU buffer.
    fn try_receive(&mut self) -> bool {
        if self.sender_name.is_none() {
            return false;
        }

        if self.shared_texture.is_none() {
            if let Err(e) = self.open_shared_texture() {
                log::error!("[Spout Input] Failed to open texture: {}", e);
                return false;
            }
        }

        let (w, h) = self.resolution;
        if w == 0 || h == 0 {
            return false;
        }

        unsafe {
            let shared_tex = match self.shared_texture.as_ref() {
                Some(t) => t,
                None => return false,
            };
            let staging_tex = match self.staging_texture.as_ref() {
                Some(t) => t,
                None => return false,
            };

            // Copy under keyed mutex if present
            let use_keyed_mutex = match shared_tex.cast::<IDXGIKeyedMutex>() {
                Ok(keyed_mutex) => {
                    match keyed_mutex.AcquireSync(0, 1000) {
                        Ok(_) => {
                            self.d3d_context.CopyResource(staging_tex, shared_tex);
                            self.d3d_context.Flush();
                            keyed_mutex.ReleaseSync(0).ok();
                            true
                        }
                        Err(e) => {
                            log::warn!("[Spout Input] AcquireSync failed: {:?}", e);
                            false
                        }
                    }
                }
                Err(_) => false,
            };

            if !use_keyed_mutex {
                self.d3d_context.CopyResource(staging_tex, shared_tex);
                self.d3d_context.Flush();
            }

            // Map staging texture and read BGRA bytes
            let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
            if let Err(e) = self.d3d_context.Map(staging_tex, 0, D3D11_MAP_READ, 0, Some(&mut mapped)) {
                log::error!("[Spout Input] Map failed: {:?}", e);
                return false;
            }

            let needed = (w * h * 4) as usize;
            if self.pixel_buffer.len() != needed {
                self.pixel_buffer.resize(needed, 0);
            }

            let src = mapped.pData as *const u8;
            let row_pitch = mapped.RowPitch as usize;
            let dst_row_bytes = (w * 4) as usize;

            if row_pitch == dst_row_bytes {
                std::ptr::copy_nonoverlapping(src, self.pixel_buffer.as_mut_ptr(), needed);
            } else {
                for row in 0..h as usize {
                    let src_row =
                        std::slice::from_raw_parts(src.add(row * row_pitch), dst_row_bytes);
                    self.pixel_buffer[row * dst_row_bytes..(row + 1) * dst_row_bytes]
                        .copy_from_slice(src_row);
                }
            }

            self.d3d_context.Unmap(staging_tex, 0);
        }

        true
    }
}

impl Debug for SpoutInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpoutInput")
            .field("sender_name", &self.sender_name)
            .field("resolution", &self.resolution)
            .field("connected", &self.is_connected())
            .finish()
    }
}

impl IpcInput for SpoutInput {
    fn connect(&mut self, source: &str) -> IpcResult<()> {
        if self.is_connected() {
            return Err(IpcError::AlreadyConnected);
        }

        log::info!("[Spout] Connecting to sender: {}", source);
        self.sender_name = Some(source.to_string());
        self.open_shared_texture()?;
        log::info!("[Spout] Connected to sender: {}", source);
        Ok(())
    }

    fn disconnect(&mut self) {
        self.shared_texture = None;
        self.staging_texture = None;
        self.resolution = (0, 0);
        self.pixel_buffer.clear();
        if let Some(ref name) = self.sender_name {
            log::info!("[Spout] Disconnected from '{}'", name);
        }
        self.sender_name = None;
    }

    fn is_connected(&self) -> bool {
        self.sender_name.is_some()
    }

    fn receive_frame(&mut self) -> Option<IpcFrame> {
        if !self.is_connected() {
            return None;
        }

        if !self.try_receive() {
            return None;
        }

        let (w, h) = self.resolution;
        Some(IpcFrame::CpuBuffer {
            data: self.pixel_buffer.clone(),
            format: PixelFormat::BGRA,
            width: w,
            height: h,
        })
    }

    fn resolution(&self) -> Option<(u32, u32)> {
        if self.resolution.0 > 0 && self.resolution.1 > 0 {
            Some(self.resolution)
        } else {
            None
        }
    }

    fn source_name(&self) -> Option<&str> {
        self.sender_name.as_deref()
    }
}

impl Default for SpoutInput {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for SpoutInput {
    fn drop(&mut self) {
        self.disconnect();
    }
}

// ---------------------------------------------------------------------------
// SpoutOutput — implements IpcOutput
// ---------------------------------------------------------------------------

/// Spout output sender using D3D11 shared textures.
///
/// Creates a D3D11 shared texture with keyed mutex, registers it in
/// the Spout2 shared-memory maps, and updates it each frame with CPU bytes
/// from the async readback pool.
pub struct SpoutOutput {
    sender_name: Option<String>,
    d3d_device: Option<ID3D11Device>,
    d3d_context: Option<ID3D11DeviceContext>,
    shared_texture: Option<ID3D11Texture2D>,
    share_handle: HANDLE,
    width: u32,
    height: u32,
    /// Held open to keep the SpoutSenderNames mapping alive
    sender_names_map: HANDLE,
    /// Held open to keep the per-sender SharedTextureInfo mapping alive
    sender_info_map: HANDLE,
}

// Safety: D3D11 device/context are thread-safe when used from one thread
unsafe impl Send for SpoutOutput {}

impl SpoutOutput {
    pub fn new() -> Self {
        Self {
            sender_name: None,
            d3d_device: None,
            d3d_context: None,
            shared_texture: None,
            share_handle: HANDLE::default(),
            width: 0,
            height: 0,
            sender_names_map: HANDLE::default(),
            sender_info_map: HANDLE::default(),
        }
    }

    fn create_shared_texture(&mut self, width: u32, height: u32) -> IpcResult<()> {
        let d3d_device = self.d3d_device.as_ref().ok_or(IpcError::NotInitialized)?;

        unsafe {
            let desc = D3D11_TEXTURE2D_DESC {
                Width: width,
                Height: height,
                MipLevels: 1,
                ArraySize: 1,
                Format: DXGI_FORMAT_B8G8R8A8_UNORM,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Usage: D3D11_USAGE_DEFAULT,
                BindFlags: (D3D11_BIND_RENDER_TARGET.0 | D3D11_BIND_SHADER_RESOURCE.0) as u32,
                CPUAccessFlags: 0,
                MiscFlags: D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX.0 as u32,
            };

            let mut tex = None;
            d3d_device.CreateTexture2D(&desc, None, Some(&mut tex))
                .map_err(|e| IpcError::NativeError(format!("CreateTexture2D: {:?}", e)))?;
            let tex: ID3D11Texture2D = tex.ok_or_else(|| {
                IpcError::NativeError("CreateTexture2D returned None".into())
            })?;

            // Get the DXGI shared handle
            let dxgi_resource: IDXGIResource = tex.cast()
                .map_err(|e| IpcError::NativeError(format!("IDXGIResource cast: {:?}", e)))?;
            let handle = dxgi_resource.GetSharedHandle()
                .map_err(|e| IpcError::NativeError(format!("GetSharedHandle: {:?}", e)))?;

            // Initialize the keyed mutex in a known state
            if let Ok(keyed_mutex) = tex.cast::<IDXGIKeyedMutex>() {
                keyed_mutex.AcquireSync(0, 0xFFFFFFFF)
                    .map_err(|e| IpcError::NativeError(format!("AcquireSync init: {:?}", e)))?;
                keyed_mutex.ReleaseSync(0)
                    .map_err(|e| IpcError::NativeError(format!("ReleaseSync init: {:?}", e)))?;
            }

            log::info!(
                "[Spout] Shared texture {}x{} created, handle={:?}",
                width, height, handle
            );

            // Close old sender info handle before replacing
            if !self.sender_info_map.is_invalid() && !self.sender_info_map.0.is_null() {
                CloseHandle(self.sender_info_map).ok();
            }

            self.share_handle = handle;
            self.shared_texture = Some(tex);
            self.width = width;
            self.height = height;

            let (names_map, info_map) = self.register_spout_sender(width, height, handle)?;
            self.sender_names_map = names_map;
            self.sender_info_map = info_map;
        }
        Ok(())
    }

    /// Register this sender in the Spout2 shared-memory maps.
    unsafe fn register_spout_sender(
        &self,
        width: u32,
        height: u32,
        handle: HANDLE,
    ) -> IpcResult<(HANDLE, HANDLE)> {
        let sender_name = self.sender_name.as_deref().ok_or(IpcError::NotInitialized)?;

        // ── Global sender name list ─────────────────────────────────────
        let map_name = windows::core::s!("SpoutSenderNames");
        let map_size = (SPOUT_MAX_SENDERS * SPOUT_MAX_NAME_LEN) as u32;
        let hmap = CreateFileMappingA(
            INVALID_HANDLE_VALUE,
            None,
            PAGE_READWRITE,
            0,
            map_size,
            map_name,
        )
        .map_err(|e| IpcError::NativeError(format!("CreateFileMappingA names: {:?}", e)))?;

        let view = MapViewOfFile(hmap, FILE_MAP_ALL_ACCESS, 0, 0, 0);
        if view.Value.is_null() {
            CloseHandle(hmap).ok();
            return Err(IpcError::NativeError(
                "MapViewOfFile failed for SpoutSenderNames".into(),
            ));
        }

        {
            let base = view.Value as *mut u8;
            let name_bytes = sender_name.as_bytes();
            let mut already_present = false;

            for i in 0..SPOUT_MAX_SENDERS {
                let slot = base.add(i * SPOUT_MAX_NAME_LEN);
                if *slot == 0 {
                    break;
                }
                let mut len = 0usize;
                while len < SPOUT_MAX_NAME_LEN {
                    if *slot.add(len) == 0 {
                        break;
                    }
                    len += 1;
                }
                if len == name_bytes.len()
                    && std::slice::from_raw_parts(slot, len) == name_bytes
                {
                    already_present = true;
                    break;
                }
            }

            if !already_present {
                for i in 0..SPOUT_MAX_SENDERS {
                    let slot = base.add(i * SPOUT_MAX_NAME_LEN);
                    if *slot == 0 {
                        let copy_len = name_bytes.len().min(SPOUT_MAX_NAME_LEN - 1);
                        std::ptr::copy_nonoverlapping(name_bytes.as_ptr(), slot, copy_len);
                        *slot.add(copy_len) = 0;
                        if i + 1 < SPOUT_MAX_SENDERS {
                            *base.add((i + 1) * SPOUT_MAX_NAME_LEN) = 0;
                        }
                        log::info!(
                            "[Spout] Registered '{}' in SpoutSenderNames (slot {})",
                            sender_name, i
                        );
                        break;
                    }
                }
            }
        }

        UnmapViewOfFile(view).ok();

        // ── Per-sender info block ───────────────────────────────────────
        let sender_cstr = std::ffi::CString::new(sender_name)
            .map_err(|e| IpcError::NativeError(format!("Invalid sender name: {}", e)))?;

        let hmap2 = CreateFileMappingA(
            INVALID_HANDLE_VALUE,
            None,
            PAGE_READWRITE,
            0,
            std::mem::size_of::<SharedTextureInfo>() as u32,
            windows::core::PCSTR(sender_cstr.as_ptr() as *const u8),
        )
        .map_err(|e| IpcError::NativeError(format!("CreateFileMappingA sender: {:?}", e)))?;

        let view2 = MapViewOfFile(hmap2, FILE_MAP_ALL_ACCESS, 0, 0, 0);
        if view2.Value.is_null() {
            CloseHandle(hmap2).ok();
            return Err(IpcError::NativeError(format!(
                "MapViewOfFile failed for sender info '{}'",
                sender_name
            )));
        }

        let handle_u32 = handle.0 as i32 as u32;

        // Populate description with executable path
        let mut description = [0u8; 256];
        if let Ok(exe_path) = std::env::current_exe() {
            let path_str = exe_path.to_string_lossy();
            let path_bytes = path_str.as_bytes();
            let copy_len = path_bytes.len().min(255);
            description[..copy_len].copy_from_slice(&path_bytes[..copy_len]);
        }

        let info_ptr = view2.Value as *mut SharedTextureInfo;
        *info_ptr = SharedTextureInfo {
            share_handle: handle_u32,
            width,
            height,
            format: dxgi_format::B8G8R8A8_UNORM,
            usage: 0,
            description,
            partner_id: 0,
        };

        UnmapViewOfFile(view2).ok();

        log::info!(
            "[Spout] Sender info written for '{}' {}x{} (handle=0x{:08x})",
            sender_name, width, height, handle_u32,
        );
        Ok((hmap, hmap2))
    }

    /// Remove this sender from the global SpoutSenderNames list.
    unsafe fn unregister_spout_sender(&self) {
        let sender_name = match self.sender_name.as_deref() {
            Some(n) => n,
            None => return,
        };

        let map_name = windows::core::s!("SpoutSenderNames");
        let Ok(hmap) = CreateFileMappingA(
            INVALID_HANDLE_VALUE,
            None,
            PAGE_READWRITE,
            0,
            (SPOUT_MAX_SENDERS * SPOUT_MAX_NAME_LEN) as u32,
            map_name,
        ) else {
            return;
        };

        let view = MapViewOfFile(hmap, FILE_MAP_ALL_ACCESS, 0, 0, 0);
        if !view.Value.is_null() {
            let base = view.Value as *mut u8;
            let name_bytes = sender_name.as_bytes();

            let mut found_idx: Option<usize> = None;
            let mut total_count = 0usize;
            for i in 0..SPOUT_MAX_SENDERS {
                let slot = base.add(i * SPOUT_MAX_NAME_LEN);
                if *slot == 0 {
                    total_count = i;
                    break;
                }
                let mut len = 0usize;
                while len < SPOUT_MAX_NAME_LEN {
                    if *slot.add(len) == 0 {
                        break;
                    }
                    len += 1;
                }
                if found_idx.is_none()
                    && len == name_bytes.len()
                    && std::slice::from_raw_parts(slot, len) == name_bytes
                {
                    found_idx = Some(i);
                }
                if i == SPOUT_MAX_SENDERS - 1 {
                    total_count = SPOUT_MAX_SENDERS;
                }
            }

            if let Some(idx) = found_idx {
                let remaining = total_count.saturating_sub(idx + 1);
                if remaining > 0 {
                    std::ptr::copy(
                        base.add((idx + 1) * SPOUT_MAX_NAME_LEN),
                        base.add(idx * SPOUT_MAX_NAME_LEN),
                        remaining * SPOUT_MAX_NAME_LEN,
                    );
                }
                let last = if total_count > 0 { total_count - 1 } else { 0 };
                std::ptr::write_bytes(
                    base.add(last * SPOUT_MAX_NAME_LEN),
                    0,
                    SPOUT_MAX_NAME_LEN,
                );
                log::info!(
                    "[Spout] Unregistered '{}' from SpoutSenderNames",
                    sender_name
                );
            }

            UnmapViewOfFile(view).ok();
        }
        CloseHandle(hmap).ok();
    }
}

impl Debug for SpoutOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpoutOutput")
            .field("sender_name", &self.sender_name)
            .field("dimensions", &(self.width, self.height))
            .field("active", &self.is_active())
            .finish()
    }
}

impl IpcOutput for SpoutOutput {
    fn create_server(&mut self, name: &str, width: u32, height: u32) -> IpcResult<()> {
        if self.is_active() {
            self.destroy_server();
        }

        if width == 0 || height == 0 {
            return Err(IpcError::InvalidDimensions { width, height });
        }

        log::info!("[Spout] Creating sender '{}' at {}x{}", name, width, height);

        // Create D3D11 device
        unsafe {
            let mut d3d_device = None;
            let mut d3d_context = None;

            D3D11CreateDevice(
                None,
                D3D_DRIVER_TYPE_HARDWARE,
                HMODULE::default(),
                D3D11_CREATE_DEVICE_FLAG(0),
                None,
                D3D11_SDK_VERSION,
                Some(&mut d3d_device),
                None,
                Some(&mut d3d_context),
            )
            .map_err(|e| IpcError::NativeError(format!("D3D11CreateDevice: {:?}", e)))?;

            self.d3d_device = Some(d3d_device
                .ok_or_else(|| IpcError::NativeError("D3D11 device None".into()))?);
            self.d3d_context = Some(d3d_context
                .ok_or_else(|| IpcError::NativeError("D3D11 context None".into()))?);
        }

        self.sender_name = Some(name.to_string());

        // Early registration with placeholder texture
        self.create_shared_texture(64, 64)?;
        log::info!("[Spout] Sender '{}' registered early (placeholder 64x64)", name);

        // Now create at actual size
        self.create_shared_texture(width, height)?;

        Ok(())
    }

    fn destroy_server(&mut self) {
        if !self.is_active() {
            return;
        }

        log::info!("[Spout] Destroying sender: {:?}", self.sender_name);

        unsafe {
            self.unregister_spout_sender();
            if !self.sender_info_map.is_invalid() && !self.sender_info_map.0.is_null() {
                CloseHandle(self.sender_info_map).ok();
            }
            if !self.sender_names_map.is_invalid() && !self.sender_names_map.0.is_null() {
                CloseHandle(self.sender_names_map).ok();
            }
        }

        self.shared_texture = None;
        self.d3d_device = None;
        self.d3d_context = None;
        self.sender_name = None;
        self.width = 0;
        self.height = 0;
        self.share_handle = HANDLE::default();
        self.sender_names_map = HANDLE::default();
        self.sender_info_map = HANDLE::default();
    }

    fn is_active(&self) -> bool {
        self.sender_name.is_some() && self.d3d_device.is_some()
    }

    fn send_texture(
        &mut self,
        _texture: &wgpu::Texture,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
    ) -> IpcResult<()> {
        // GPU zero-copy path not implemented — use send_buffer() via readback pool
        Err(IpcError::SharingNotAvailable)
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

        // Validate buffer size
        let expected_size = format.buffer_size(width, height);
        if data.len() < expected_size {
            return Err(IpcError::NativeError(
                format!("Buffer too small: {} < {}", data.len(), expected_size),
            ));
        }

        // (Re-)create shared texture when dimensions change
        if self.shared_texture.is_none() || self.width != width || self.height != height {
            self.shared_texture = None;
            self.create_shared_texture(width, height)?;
        }

        unsafe {
            let d3d_tex = self.shared_texture.as_ref().unwrap();
            let d3d_context = self.d3d_context.as_ref().unwrap();
            let keyed_mutex: IDXGIKeyedMutex = d3d_tex.cast()
                .map_err(|e| IpcError::NativeError(format!("KeyedMutex cast: {:?}", e)))?;

            keyed_mutex.AcquireSync(0, 0xFFFFFFFF)
                .map_err(|e| IpcError::NativeError(format!("AcquireSync: {:?}", e)))?;

            let row_pitch = width * 4;
            d3d_context.UpdateSubresource(
                d3d_tex,
                0,
                None,
                data.as_ptr() as *const _,
                row_pitch,
                0,
            );

            keyed_mutex.ReleaseSync(0)
                .map_err(|e| IpcError::NativeError(format!("ReleaseSync: {:?}", e)))?;
        }

        Ok(())
    }

    fn server_name(&self) -> Option<&str> {
        self.sender_name.as_deref()
    }

    fn dimensions(&self) -> Option<(u32, u32)> {
        if self.width > 0 && self.height > 0 {
            Some((self.width, self.height))
        } else {
            None
        }
    }
}

impl Default for SpoutOutput {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for SpoutOutput {
    fn drop(&mut self) {
        self.destroy_server();
    }
}

// ---------------------------------------------------------------------------
// SpoutDiscovery — implements IpcDiscovery
// ---------------------------------------------------------------------------

/// Discovers active Spout senders by reading the SpoutSenderNames shared memory.
pub struct SpoutDiscovery {
    sources: Vec<IpcSourceInfo>,
}

impl SpoutDiscovery {
    pub fn new() -> Self {
        Self {
            sources: Vec::new(),
        }
    }
}

impl Default for SpoutDiscovery {
    fn default() -> Self {
        Self::new()
    }
}

impl IpcDiscovery for SpoutDiscovery {
    fn discover_sources(&mut self, _timeout_ms: u32) -> Vec<IpcSourceInfo> {
        self.sources.clear();

        unsafe {
            let map_name = windows::core::s!("SpoutSenderNames");
            let Ok(hmap) = OpenFileMappingA(FILE_MAP_READ.0, false, map_name) else {
                return self.sources.clone();
            };

            let view = MapViewOfFile(hmap, FILE_MAP_READ, 0, 0, 0);
            if view.Value.is_null() {
                CloseHandle(hmap).ok();
                return self.sources.clone();
            }

            // Determine mapped region size
            let mut mbi = MEMORY_BASIC_INFORMATION::default();
            let mbi_size = std::mem::size_of::<MEMORY_BASIC_INFORMATION>();
            let queried = VirtualQuery(Some(view.Value), &mut mbi, mbi_size);
            let mapped_size: usize = if queried == mbi_size {
                mbi.RegionSize
            } else {
                SPOUT_MAX_SENDERS * SPOUT_MAX_NAME_LEN
            };

            let base = view.Value as *const u8;
            let max_slots = (mapped_size / SPOUT_MAX_NAME_LEN).min(SPOUT_MAX_SENDERS);

            for i in 0..max_slots {
                let slot_offset = i * SPOUT_MAX_NAME_LEN;
                if slot_offset + SPOUT_MAX_NAME_LEN > mapped_size {
                    break;
                }
                let slot =
                    std::slice::from_raw_parts(base.add(slot_offset), SPOUT_MAX_NAME_LEN);
                if slot[0] == 0 {
                    break;
                }
                let null_pos = slot.iter().position(|&b| b == 0).unwrap_or(SPOUT_MAX_NAME_LEN);
                let name = String::from_utf8_lossy(&slot[..null_pos]).into_owned();
                if name.is_empty() {
                    continue;
                }
                let (width, height) = read_sender_dimensions(&name);
                log::debug!("[Spout] sender[{}]: '{}' {}x{}", i, name, width, height);
                self.sources.push(IpcSourceInfo {
                    name: name.clone(),
                    app_name: String::new(),
                    dimensions: if width > 0 && height > 0 {
                        Some((width, height))
                    } else {
                        None
                    },
                    handle: name,
                });
            }

            UnmapViewOfFile(view).ok();
            CloseHandle(hmap).ok();
        }

        log::debug!("[Spout] Discovery: {} sender(s)", self.sources.len());
        self.sources.clone()
    }

    fn is_source_available(&self, name: &str) -> bool {
        self.sources.iter().any(|s| s.name == name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spout_output_creation() {
        let output = SpoutOutput::new();
        assert!(!output.is_active());
        assert_eq!(output.server_name(), None);
    }

    #[test]
    fn test_pixel_format_conversion() {
        assert_eq!(pixel_format_to_dxgi(PixelFormat::RGBA), 28);
        assert_eq!(pixel_format_to_dxgi(PixelFormat::BGRA), 87);
    }

    #[test]
    fn test_discovery_creation() {
        let discovery = SpoutDiscovery::new();
        assert!(discovery.sources.is_empty());
    }
}
