//! # Webcam Capture Module
//!
//! Handles webcam capture using nokhwa library.

use anyhow::{anyhow, Result};
use nokhwa::{
    utils::{ApiBackend, CameraIndex, RequestedFormat, RequestedFormatType, Resolution, FrameFormat},
    Camera,
};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;
use std::io::Cursor;

/// A captured webcam frame
#[derive(Debug, Clone)]
pub struct WebcamFrame {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>, // BGRA pixel data
    pub timestamp: std::time::Instant,
}

/// Webcam capture handler
pub struct WebcamCapture {
    device_index: usize,
    width: u32,
    height: u32,
    fps: u32,
    capture_thread: Option<thread::JoinHandle<()>>,
    stop_signal: Option<mpsc::Sender<()>>,
}

impl WebcamCapture {
    /// Create a new webcam capture (doesn't start yet)
    pub fn new(device_index: usize, width: u32, height: u32, fps: u32) -> Result<Self> {
        Ok(Self {
            device_index,
            width,
            height,
            fps,
            capture_thread: None,
            stop_signal: None,
        })
    }
    
    /// Start capturing frames
    pub fn start(&mut self) -> Result<mpsc::Receiver<WebcamFrame>> {
        // Create channel for frame communication
        let (frame_sender, frame_receiver) = mpsc::channel::<WebcamFrame>();
        let (stop_sender, stop_receiver) = mpsc::channel::<()>();
        
        let device_index = self.device_index;
        let width = self.width;
        let height = self.height;
        let fps = self.fps;
        
        // Spawn capture thread
        let handle = thread::spawn(move || {
            if let Err(e) = capture_thread(
                device_index,
                width,
                height,
                fps,
                frame_sender,
                stop_receiver,
            ) {
                log::error!("Webcam capture thread error: {:?}", e);
            }
        });
        
        self.capture_thread = Some(handle);
        self.stop_signal = Some(stop_sender);
        
        Ok(frame_receiver)
    }
    
    /// Stop capturing
    pub fn stop(&mut self) -> Result<()> {
        // Send stop signal
        if let Some(sender) = self.stop_signal.take() {
            let _ = sender.send(());
        }
        
        // Wait for thread to finish
        if let Some(handle) = self.capture_thread.take() {
            let _ = handle.join();
        }
        
        Ok(())
    }
}

impl Drop for WebcamCapture {
    fn drop(&mut self) {
        let _ = self.stop();
    }
}

/// Capture thread function
/// Try to open camera with multiple fallback strategies
/// This handles virtual cameras (OBS, etc.) that may use non-standard formats
/// 
/// Environment variable workarounds:
/// - `RUSTJAY_FORCE_YUYV=1` - Force YUYV format
/// - `RUSTJAY_FORCE_MJPEG=1` - Force MJPEG format
/// - `RUSTJAY_FORCE_640=1` - Force 640x480 resolution
/// - `RUSTJAY_WEBCAM_DEBUG=1` - Enable verbose debugging
fn try_open_camera(camera_index: CameraIndex) -> Result<Camera> {
    // Check for environment variable overrides (for troubleshooting)
    let force_yuyv = std::env::var("RUSTJAY_FORCE_YUYV").is_ok();
    let force_mjpeg = std::env::var("RUSTJAY_FORCE_MJPEG").is_ok();
    let force_640 = std::env::var("RUSTJAY_FORCE_640").is_ok();
    let debug_mode = std::env::var("RUSTJAY_WEBCAM_DEBUG").is_ok();
    
    // First, verify the camera index exists by querying available cameras
    if let Some(backend) = nokhwa::native_api_backend() {
        match nokhwa::query(backend) {
            Ok(cameras) => {
                let idx = match &camera_index {
                    CameraIndex::Index(i) => *i as usize,
                    _ => 0,
                };
                if idx >= cameras.len() {
                    return Err(anyhow!(
                        "Camera index {} is out of range. Only {} camera(s) available.",
                        idx, cameras.len()
                    ));
                }
                if debug_mode {
                    log::info!("[WEBCAM] Camera {} info: {:?}", idx, cameras[idx]);
                }
            }
            Err(e) => {
                log::warn!("[WEBCAM] Could not query camera list: {:?}", e);
            }
        }
    }
    
    if force_yuyv || force_mjpeg || force_640 {
        log::info!("[WEBCAM] Environment override: YUYV={}, MJPEG={}, 640x480={}", 
            force_yuyv, force_mjpeg, force_640);
    }
    
    // Environment override: Force YUYV format
    if force_yuyv {
        log::info!("[WEBCAM] Trying forced YUYV format (RUSTJAY_FORCE_YUYV set)...");
        let res = if force_640 { Resolution::new(640, 480) } else { Resolution::new(1280, 720) };
        let requested_format = RequestedFormat::new::<nokhwa::pixel_format::YuyvFormat>(
            RequestedFormatType::Closest(
                nokhwa::utils::CameraFormat::new(res, FrameFormat::YUYV, 30)
            )
        );
        match try_create_camera(&camera_index, requested_format) {
            Ok(camera) => {
                log::info!("[WEBCAM] Success with forced YUYV");
                return Ok(camera);
            }
            Err(e) => log::warn!("[WEBCAM] Forced YUYV failed: {:?}", e),
        }
    }
    
    // Environment override: Force MJPEG format
    if force_mjpeg {
        log::info!("[WEBCAM] Trying forced MJPEG format (RUSTJAY_FORCE_MJPEG set)...");
        let res = if force_640 { Resolution::new(640, 480) } else { Resolution::new(1280, 720) };
        let requested_format = RequestedFormat::new::<nokhwa::pixel_format::RgbFormat>(
            RequestedFormatType::Closest(
                nokhwa::utils::CameraFormat::new(res, FrameFormat::MJPEG, 30)
            )
        );
        match try_create_camera(&camera_index, requested_format) {
            Ok(camera) => {
                log::info!("[WEBCAM] Success with forced MJPEG");
                return Ok(camera);
            }
            Err(e) => log::warn!("[WEBCAM] Forced MJPEG failed: {:?}", e),
        }
    }
    
    // Strategy 1: Try with HighestResolution preference (default behavior)
    log::info!("[WEBCAM] Trying HighestResolution strategy...");
    let requested_format = RequestedFormat::new::<nokhwa::pixel_format::RgbFormat>(
        RequestedFormatType::HighestResolution(Resolution::new(1280, 720))
    );
    
    match try_create_camera(&camera_index, requested_format) {
        Ok(camera) => {
            log::info!("[WEBCAM] Success with HighestResolution");
            return Ok(camera);
        }
        Err(e) => {
            log::warn!("[WEBCAM] HighestResolution failed: {:?}", e);
        }
    }
    
    // Strategy 2: Try with no format preference (let camera decide)
    log::info!("[WEBCAM] Trying NoPreference strategy...");
    let requested_format = RequestedFormat::new::<nokhwa::pixel_format::RgbFormat>(
        RequestedFormatType::None
    );
    
    match try_create_camera(&camera_index, requested_format) {
        Ok(camera) => {
            log::info!("[WEBCAM] Success with NoPreference");
            return Ok(camera);
        }
        Err(e) => {
            log::warn!("[WEBCAM] NoPreference failed: {:?}", e);
        }
    }
    
    // Strategy 3: Try with a specific format (workaround for OBS Virtual Camera)
    // OBS Virtual Camera often uses YUYV or custom formats at 1280x720@30fps
    log::info!("[WEBCAM] Trying specific format (YUYV 1280x720@30fps)...");
    let requested_format = RequestedFormat::new::<nokhwa::pixel_format::YuyvFormat>(
        RequestedFormatType::Closest(
            nokhwa::utils::CameraFormat::new(
                Resolution::new(1280, 720),
                FrameFormat::YUYV,
                30
            )
        )
    );
    
    match try_create_camera(&camera_index, requested_format) {
        Ok(camera) => {
            log::info!("[WEBCAM] Success with YUYV format");
            return Ok(camera);
        }
        Err(e) => {
            log::warn!("[WEBCAM] YUYV format failed: {:?}", e);
        }
    }
    
    // Strategy 4: Try with MJPEG format (another common virtual camera format)
    log::info!("[WEBCAM] Trying specific format (MJPEG 1280x720@30fps)...");
    let requested_format = RequestedFormat::new::<nokhwa::pixel_format::RgbFormat>(
        RequestedFormatType::Closest(
            nokhwa::utils::CameraFormat::new(
                Resolution::new(1280, 720),
                FrameFormat::MJPEG,
                30
            )
        )
    );
    
    match try_create_camera(&camera_index, requested_format) {
        Ok(camera) => {
            log::info!("[WEBCAM] Success with MJPEG format");
            return Ok(camera);
        }
        Err(e) => {
            log::warn!("[WEBCAM] MJPEG format failed: {:?}", e);
        }
    }
    
    // Strategy 5: Lower resolution fallback (640x480 is almost universally supported)
    log::info!("[WEBCAM] Trying lower resolution (640x480)...");
    let requested_format = RequestedFormat::new::<nokhwa::pixel_format::RgbFormat>(
        RequestedFormatType::Closest(
            nokhwa::utils::CameraFormat::new(
                Resolution::new(640, 480),
                FrameFormat::YUYV,
                30
            )
        )
    );
    
    match try_create_camera(&camera_index, requested_format) {
        Ok(camera) => {
            log::info!("[WEBCAM] Success with 640x480");
            return Ok(camera);
        }
        Err(e) => {
            log::warn!("[WEBCAM] 640x480 failed: {:?}", e);
        }
    }
    
    // All strategies failed
    log::error!("[WEBCAM] All camera opening strategies failed for device {:?}", camera_index);
    log::error!("[WEBCAM] OBS Virtual Camera and some other virtual cameras are not supported on macOS due to format incompatibility.");
    log::error!("[WEBCAM] Workarounds: 1) Use NDI output from OBS instead, 2) Use a physical webcam, 3) Try OBS on Windows/Linux");
    Err(anyhow!("Failed to open camera. OBS Virtual Camera is not supported on macOS. Try using NDI output from OBS instead, or use a physical webcam."))
}

/// Helper to create camera with panic handling
fn try_create_camera(camera_index: &CameraIndex, requested_format: RequestedFormat) -> Result<Camera> {
    log::debug!("[WEBCAM] Creating camera with index {:?}", camera_index);
    
    match std::panic::catch_unwind(|| {
        Camera::new(camera_index.clone(), requested_format)
    }) {
        Ok(Ok(camera)) => {
            log::debug!("[WEBCAM] Camera created successfully");
            Ok(camera)
        }
        Ok(Err(e)) => {
            log::debug!("[WEBCAM] Camera creation failed: {:?}", e);
            Err(anyhow!("Camera creation error: {:?}", e))
        }
        Err(panic_info) => {
            let msg = if let Some(s) = panic_info.downcast_ref::<&str>() {
                s.to_string()
            } else if let Some(s) = panic_info.downcast_ref::<String>() {
                s.clone()
            } else {
                "Unknown panic".to_string()
            };
            log::debug!("[WEBCAM] Camera creation panicked: {}", msg);
            Err(anyhow!("Camera creation panicked: {}", msg))
        }
    }
}

fn capture_thread(
    device_index: usize,
    _width: u32,
    _height: u32,
    _fps: u32,
    frame_sender: mpsc::Sender<WebcamFrame>,
    stop_receiver: mpsc::Receiver<()>,
) -> Result<()> {
    log::info!("[WEBCAM] Starting webcam capture thread for device {}", device_index);
    
    // Create camera index
    let camera_index = CameraIndex::Index(device_index as u32);
    
    // Try multiple approaches to open the camera
    // OBS Virtual Camera and other virtual cams may use formats that nokhwa can't auto-detect
    let mut camera = try_open_camera(camera_index.clone())?;
    
    camera.open_stream()
        .map_err(|e| anyhow!("Failed to open camera stream: {:?}", e))?;
    
    log::info!("[WEBCAM] Camera opened successfully");
    
    // Get actual format being used
    let actual_format = camera.camera_format();
    log::info!("[WEBCAM] Camera format: {:?}", actual_format);
    
    let actual_width = actual_format.resolution().width_x;
    let actual_height = actual_format.resolution().height_y;
    let actual_fps = actual_format.frame_rate();
    
    log::info!("[WEBCAM] Actual resolution: {}x{} @ {}fps", actual_width, actual_height, actual_fps);
    
    let mut frame_count = 0u64;
    let mut last_log_time = std::time::Instant::now();
    
    let frame_duration = Duration::from_millis(1000 / actual_fps.max(1) as u64);
    let mut last_frame_time = std::time::Instant::now();
    
    loop {
        // Check for stop signal (non-blocking)
        if stop_receiver.try_recv().is_ok() {
            log::info!("Stopping webcam capture thread");
            break;
        }
        
        // Capture frame
        match camera.frame() {
            Ok(buffer) => {
                frame_count += 1;
                
                // Log frame capture rate every 5 seconds
                let now = std::time::Instant::now();
                if now.duration_since(last_log_time).as_secs() >= 5 {
                    let fps = frame_count as f64 / now.duration_since(last_log_time).as_secs_f64();
                    log::info!("[WEBCAM] Captured {} frames in 5s ({:.1} fps)", frame_count, fps);
                    frame_count = 0;
                    last_log_time = now;
                }
                
                // Convert to RGBA based on actual format
                let frame_data = buffer.buffer();
                let reported_width = actual_width;
                let reported_height = actual_height;
                let frame_format = actual_format.format();
                let expected_rgb_size = (reported_width * reported_height * 3) as usize;
                let expected_rgba_size = (reported_width * reported_height * 4) as usize;
                let expected_yuyv_size = (reported_width * reported_height * 2) as usize;
                
                if frame_count == 1 {
                    log::info!("[WEBCAM] First frame: {} bytes, format: {:?}, expected RGB={}/RGBA={}/YUYV={}", 
                        frame_data.len(), frame_format, expected_rgb_size, expected_rgba_size, expected_yuyv_size);
                }
                
                // Detect actual format based on buffer size - the camera may return different format than reported
                // Common camera formats: 640x480, 960x540, 1280x720, 1920x1080
                let (bgra_data, actual_width, actual_height) = if frame_format == FrameFormat::MJPEG || 
                                                                  (frame_data.len() < expected_yuyv_size && frame_data.len() > 10000) {
                    // MJPEG format - needs to be decoded
                    match decode_mjpeg_to_rgba(frame_data) {
                        Some((bgra, w, h)) => {
                            if frame_count <= 5 {
                                log::info!("[WEBCAM] Decoded MJPEG frame: {}x{} from {} bytes", w, h, frame_data.len());
                            }
                            (bgra, w, h)
                        }
                        None => {
                            log::error!("[WEBCAM] Failed to decode MJPEG frame ({} bytes), skipping", frame_data.len());
                            continue;
                        }
                    }
                } else if frame_data.len() == expected_yuyv_size {
                    // YUYV format (2 bytes per pixel) at reported resolution
                    (convert_yuyv_to_rgba(frame_data, reported_width as usize, reported_height as usize), reported_width, reported_height)
                } else if frame_data.len() == expected_rgba_size {
                    // Already 4 bytes/pixel — on macOS, cameras deliver BGRA natively
                    (frame_data.to_vec(), reported_width, reported_height)
                } else if frame_data.len() == expected_rgb_size {
                    // RGB format (3 bytes per pixel)
                    (convert_rgb_to_rgba(frame_data, reported_width as usize, reported_height as usize), reported_width, reported_height)
                } else {
                    // Unknown format - try common resolutions
                    // Try to detect based on common YUYV frame sizes
                    let detected = match frame_data.len() {
                        // 640x480 YUYV = 614400 bytes
                        614400 => (640u32, 480u32, "YUYV 640x480"),
                        // 960x540 YUYV = 1036800 bytes  
                        1036800 => (960u32, 540u32, "YUYV 960x540"),
                        // 1280x720 YUYV = 1843200 bytes
                        1843200 => (1280u32, 720u32, "YUYV 1280x720"),
                        // 1920x1080 YUYV = 4147200 bytes
                        4147200 => (1920u32, 1080u32, "YUYV 1920x1080"),
                        // Unknown size - try to infer
                        _ => {
                            // Try assuming YUYV and calculate height from width
                            if frame_data.len() % (reported_width as usize * 2) == 0 {
                                let h = frame_data.len() / (reported_width as usize * 2);
                                (reported_width, h as u32, "inferred YUYV")
                            } else if frame_data.len() % (reported_width as usize * 3) == 0 {
                                let h = frame_data.len() / (reported_width as usize * 3);
                                (reported_width, h as u32, "inferred RGB")
                            } else {
                                // Last resort: try common 960x425-like sizes
                                // 816000 bytes = 960 x 425 x 2 (YUYV)
                                if frame_data.len() == 816000 {
                                    (960, 425, "detected 960x425 YUYV")
                                } else {
                                    (0, 0, "unknown")
                                }
                            }
                        }
                    };
                    
                    if detected.0 == 0 {
                        log::error!("[WEBCAM] Cannot determine frame format: {} bytes, skipping", frame_data.len());
                        continue;
                    }
                    
                    if frame_count <= 5 {
                        log::warn!("[WEBCAM] Detected {} (reported {}x{}), converting from {} bytes",
                            detected.2, reported_width, reported_height, frame_data.len());
                    }
                    
                    let (w, h) = (detected.0, detected.1);
                    let rgba = convert_yuyv_to_rgba(frame_data, w as usize, h as usize);
                    (rgba, w, h)
                };
                
                let frame = WebcamFrame {
                    width: actual_width,
                    height: actual_height,
                    data: bgra_data,
                    timestamp: std::time::Instant::now(),
                };
                
                // Send frame (ignore if receiver dropped)
                if frame_sender.send(frame).is_err() {
                    log::error!("[WEBCAM] Frame receiver dropped, stopping capture");
                    break;
                }
                
                last_frame_time = std::time::Instant::now();
            }
            Err(e) => {
                log::error!("[WEBCAM] Frame capture error: {:?}", e);
                // Small delay before retry
                thread::sleep(Duration::from_millis(10));
            }
        }
        
        // Frame rate limiting
        let elapsed = last_frame_time.elapsed();
        if elapsed < frame_duration {
            thread::sleep(frame_duration - elapsed);
        }
    }
    
    // Cleanup
    drop(camera);
    log::info!("[WEBCAM] Capture thread ended. Total frames: {}", frame_count);
    
    Ok(())
}

/// Convert YUYV data to BGRA.
/// YUYV is a YUV 4:2:2 format with 2 bytes per pixel: Y0 U Y1 V.
/// Most webcam feeds use limited-range BT.601, so expand luma before RGB conversion.
fn convert_yuyv_to_rgba(yuyv: &[u8], width: usize, height: usize) -> Vec<u8> {
    let expected_size = width * height * 2; // YUYV is 2 bytes per pixel
    if yuyv.len() != expected_size {
        log::warn!("Unexpected YUYV buffer size: {} vs expected {}", yuyv.len(), expected_size);
    }
    
    let mut rgba = Vec::with_capacity(width * height * 4);
    
    // Process 4 bytes (2 pixels) at a time
    for i in (0..yuyv.len().saturating_sub(3)).step_by(4) {
        let y0 = (yuyv[i] as f32 - 16.0).max(0.0);
        let u = yuyv[i + 1] as f32 - 128.0;
        let y1 = (yuyv[i + 2] as f32 - 16.0).max(0.0);
        let v = yuyv[i + 3] as f32 - 128.0;
        
        // Convert first pixel (Y0, U, V)
        let r0 = (1.164383 * y0 + 1.596027 * v).clamp(0.0, 255.0) as u8;
        let g0 = (1.164383 * y0 - 0.391762 * u - 0.812968 * v).clamp(0.0, 255.0) as u8;
        let b0 = (1.164383 * y0 + 2.017232 * u).clamp(0.0, 255.0) as u8;
        
        rgba.push(b0);
        rgba.push(g0);
        rgba.push(r0);
        rgba.push(255); // Alpha

        // Convert second pixel (Y1, U, V)
        let r1 = (1.164383 * y1 + 1.596027 * v).clamp(0.0, 255.0) as u8;
        let g1 = (1.164383 * y1 - 0.391762 * u - 0.812968 * v).clamp(0.0, 255.0) as u8;
        let b1 = (1.164383 * y1 + 2.017232 * u).clamp(0.0, 255.0) as u8;

        rgba.push(b1);
        rgba.push(g1);
        rgba.push(r1);
        rgba.push(255); // Alpha
    }
    
    rgba
}

/// Convert RGB data to RGBA (adds alpha channel, fully opaque)
fn convert_rgb_to_rgba(rgb: &[u8], width: usize, height: usize) -> Vec<u8> {
    let expected_size = width * height * 3;
    if rgb.len() != expected_size {
        log::warn!("Unexpected RGB buffer size: {} vs expected {}, attempting YUYV conversion", rgb.len(), expected_size);
        // Try YUYV conversion if size matches YUYV format
        if rgb.len() == width * height * 2 {
            return convert_yuyv_to_rgba(rgb, width, height);
        }
    }
    
    let pixel_count = rgb.len() / 3;
    let mut rgba = Vec::with_capacity(pixel_count * 4);
    
    for i in 0..pixel_count.min(rgb.len() / 3) {
        let idx = i * 3;
        rgba.push(rgb[idx + 2]); // B
        rgba.push(rgb[idx + 1]); // G
        rgba.push(rgb[idx]);     // R
        rgba.push(255);          // A (fully opaque)
    }
    
    rgba
}

/// Decode MJPEG data to RGBA
/// Returns (rgba_data, width, height) if successful
fn decode_mjpeg_to_rgba(mjpeg_data: &[u8]) -> Option<(Vec<u8>, u32, u32)> {
    // Use image crate to decode JPEG
    let cursor = Cursor::new(mjpeg_data);
    match image::load(cursor, image::ImageFormat::Jpeg) {
        Ok(dynamic_image) => {
            let rgba_image = dynamic_image.to_rgba8();
            let (width, height) = rgba_image.dimensions();
            // Convert RGBA → BGRA (swap R and B) for Bgra8Unorm upload
            let bgra_data: Vec<u8> = rgba_image.into_raw().chunks_exact(4)
                .flat_map(|p| [p[2], p[1], p[0], p[3]])
                .collect();
            Some((bgra_data, width, height))
        }
        Err(e) => {
            log::debug!("[WEBCAM] MJPEG decode error: {:?}", e);
            None
        }
    }
}

/// List available camera devices (safe wrapper)
pub fn list_cameras() -> Vec<String> {
    // Use catch_unwind to prevent panics from crashing the app
    match std::panic::catch_unwind(|| list_cameras_internal()) {
        Ok(result) => result,
        Err(_) => {
            log::error!("Camera enumeration panicked - camera access may be denied");
            Vec::new()
        }
    }
}

/// Internal function that may panic
fn list_cameras_internal() -> Vec<String> {
    let mut devices = Vec::new();
    
    // Use the native API backend for current platform
    let backend = match nokhwa::native_api_backend() {
        Some(b) => b,
        None => {
            log::warn!("No native API backend available for this platform");
            return devices;
        }
    };
    
    match nokhwa::query(backend) {
        Ok(camera_infos) => {
            for (idx, info) in camera_infos.iter().enumerate() {
                let name = info.human_name();
                let desc = info.description();
                
                // Detect virtual cameras
                let is_virtual = name.to_lowercase().contains("virtual") 
                    || name.to_lowercase().contains("obs")
                    || name.to_lowercase().contains("mmhmm")
                    || name.to_lowercase().contains("snap")
                    || desc.to_lowercase().contains("virtual");
                
                let display_name = if is_virtual {
                    format!("{}: {} (virtual)", idx, name)
                } else {
                    format!("{}: {}", idx, name)
                };
                
                devices.push(display_name);
                log::info!("Found camera {}: {} (desc: {}, misc: {}, virtual: {})", 
                    idx,
                    name, 
                    desc,
                    info.misc(),
                    is_virtual
                );
            }
        }
        Err(e) => {
            log::error!("Failed to list cameras: {:?}", e);
        }
    }
    
    devices
}

/// Get information about a specific camera
pub fn get_camera_info(device_index: usize) -> Result<String> {
    let backend = nokhwa::native_api_backend()
        .unwrap_or(ApiBackend::Auto);
    
    let cameras = nokhwa::query(backend)
        .map_err(|e| anyhow!("Failed to get camera list: {:?}", e))?;
    
    if let Some(info) = cameras.get(device_index) {
        Ok(format!(
            "Camera {}: {} ({})",
            device_index,
            info.human_name(),
            info.description()
        ))
    } else {
        Err(anyhow!("Camera index {} not found", device_index))
    }
}

/// Get supported formats for a camera (simplified)
pub fn get_camera_formats(_device_index: usize) -> Result<Vec<(u32, u32, u32)>> {
    // This would require opening the camera and querying compatible formats
    // For now, return common formats
    Ok(vec![
        (640, 480, 30),
        (1280, 720, 30),
        (1920, 1080, 30),
    ])
}
