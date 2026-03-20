//! # MIDI Input Handler
//!
//! Handles MIDI input using the `midir` crate with support for:
//! - Multiple input devices
//! - Hot-plugging (device connection/disconnection)
//! - Async message processing via channels
//! - 14-bit CC support

use midir::{Ignore, MidiInput, MidiInputConnection, MidiInputPort};
use std::collections::HashMap;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

use super::{MidiChannel, MidiController, MidiEvent, MidiValue};

/// A MIDI message
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MidiMessage {
    /// Control Change: channel, controller, value
    ControlChange {
        channel: MidiChannel,
        controller: MidiController,
        value: u8,
    },
    /// Note On: channel, note, velocity
    NoteOn {
        channel: MidiChannel,
        note: MidiController,
        velocity: u8,
    },
    /// Note Off: channel, note, velocity
    NoteOff {
        channel: MidiChannel,
        note: MidiController,
        velocity: u8,
    },
    /// Polyphonic Aftertouch: channel, note, pressure
    PolyAftertouch {
        channel: MidiChannel,
        note: MidiController,
        pressure: u8,
    },
    /// Channel Aftertouch: channel, pressure
    ChannelAftertouch {
        channel: MidiChannel,
        pressure: u8,
    },
    /// Pitch Bend: channel, value (-8192 to 8191)
    PitchBend {
        channel: MidiChannel,
        value: i16,
    },
    /// Program Change: channel, program
    ProgramChange {
        channel: MidiChannel,
        program: MidiController,
    },
}

impl MidiMessage {
    /// Get the message type
    pub fn message_type(&self) -> super::MidiMessageType {
        match self {
            MidiMessage::ControlChange { .. } => super::MidiMessageType::ControlChange,
            MidiMessage::NoteOn { .. } => super::MidiMessageType::NoteOn,
            MidiMessage::NoteOff { .. } => super::MidiMessageType::NoteOff,
            MidiMessage::PolyAftertouch { .. } => super::MidiMessageType::PolyAftertouch,
            MidiMessage::ChannelAftertouch { .. } => super::MidiMessageType::ChannelAftertouch,
            MidiMessage::PitchBend { .. } => super::MidiMessageType::PitchBend,
            MidiMessage::ProgramChange { .. } => super::MidiMessageType::ProgramChange,
        }
    }

    /// Get the MIDI channel (1-16, or 0 if not applicable)
    pub fn channel(&self) -> MidiChannel {
        match self {
            MidiMessage::ControlChange { channel, .. } => *channel,
            MidiMessage::NoteOn { channel, .. } => *channel,
            MidiMessage::NoteOff { channel, .. } => *channel,
            MidiMessage::PolyAftertouch { channel, .. } => *channel,
            MidiMessage::ChannelAftertouch { channel, .. } => *channel,
            MidiMessage::PitchBend { channel, .. } => *channel,
            MidiMessage::ProgramChange { channel, .. } => *channel,
        }
    }

    /// Get the value (normalized to parameter range 0-16383 for high-res)
    pub fn value(&self) -> MidiValue {
        match self {
            MidiMessage::ControlChange { value, .. } => *value as u16,
            MidiMessage::NoteOn { velocity, .. } => *velocity as u16,
            MidiMessage::NoteOff { velocity, .. } => *velocity as u16,
            MidiMessage::PolyAftertouch { pressure, .. } => *pressure as u16,
            MidiMessage::ChannelAftertouch { pressure, .. } => *pressure as u16,
            MidiMessage::PitchBend { value, .. } => {
                // Convert -8192..8191 to 0..16383
                (*value + 8192) as u16
            }
            MidiMessage::ProgramChange { program, .. } => *program as u16,
        }
    }

    /// Parse a raw MIDI message
    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        if data.is_empty() {
            return None;
        }

        let status = data[0];
        
        // Handle running status (message without status byte)
        if status < 0x80 {
            return None; // Running status not handled in this simple parser
        }

        let message_type = status & 0xF0;
        let channel = (status & 0x0F) + 1; // Convert to 1-16 range

        match message_type {
            0x80 => {
                // Note Off
                if data.len() >= 3 {
                    Some(MidiMessage::NoteOff {
                        channel,
                        note: data[1] & 0x7F,
                        velocity: data[2] & 0x7F,
                    })
                } else {
                    None
                }
            }
            0x90 => {
                // Note On
                if data.len() >= 3 {
                    let velocity = data[2] & 0x7F;
                    if velocity == 0 {
                        // Note On with velocity 0 is treated as Note Off
                        Some(MidiMessage::NoteOff {
                            channel,
                            note: data[1] & 0x7F,
                            velocity: 0,
                        })
                    } else {
                        Some(MidiMessage::NoteOn {
                            channel,
                            note: data[1] & 0x7F,
                            velocity,
                        })
                    }
                } else {
                    None
                }
            }
            0xA0 => {
                // Polyphonic Aftertouch
                if data.len() >= 3 {
                    Some(MidiMessage::PolyAftertouch {
                        channel,
                        note: data[1] & 0x7F,
                        pressure: data[2] & 0x7F,
                    })
                } else {
                    None
                }
            }
            0xB0 => {
                // Control Change
                if data.len() >= 3 {
                    Some(MidiMessage::ControlChange {
                        channel,
                        controller: data[1] & 0x7F,
                        value: data[2] & 0x7F,
                    })
                } else {
                    None
                }
            }
            0xC0 => {
                // Program Change
                if data.len() >= 2 {
                    Some(MidiMessage::ProgramChange {
                        channel,
                        program: data[1] & 0x7F,
                    })
                } else {
                    None
                }
            }
            0xD0 => {
                // Channel Aftertouch
                if data.len() >= 2 {
                    Some(MidiMessage::ChannelAftertouch {
                        channel,
                        pressure: data[1] & 0x7F,
                    })
                } else {
                    None
                }
            }
            0xE0 => {
                // Pitch Bend
                if data.len() >= 3 {
                    let lsb = (data[1] & 0x7F) as i16;
                    let msb = (data[2] & 0x7F) as i16;
                    let value = (msb << 7) | lsb;
                    // Convert to signed (-8192 to 8191)
                    let signed_value = value as i16 - 8192;
                    Some(MidiMessage::PitchBend {
                        channel,
                        value: signed_value,
                    })
                } else {
                    None
                }
            }
            _ => None, // System messages not handled
        }
    }
}

/// Connection info for a single MIDI input device
struct DeviceConnection {
    name: String,
    _connection: MidiInputConnection<Sender<MidiEvent>>,
}

/// Handles MIDI input from multiple devices
pub struct MidiInputHandler {
    /// Active connections (each has its own MidiInput, stored separately)
    connections: HashMap<String, DeviceConnection>,
    /// Channel receiver for MIDI events
    receiver: Receiver<MidiEvent>,
    /// Channel sender (shared with callbacks)
    sender: Sender<MidiEvent>,
    /// Whether 14-bit CC is enabled
    high_resolution_cc: bool,
    /// Pending MSB values for 14-bit CC (device -> controller -> value)
    pending_msb: HashMap<String, HashMap<MidiController, u8>>,
    /// Last update time for device scan
    last_device_scan: Instant,
    /// Device scan interval
    scan_interval: std::time::Duration,
    /// Upper bound for idle scan backoff
    max_scan_interval: std::time::Duration,
}

impl MidiInputHandler {
    /// Create a new MIDI input handler
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let (sender, receiver) = channel();

        Ok(Self {
            connections: HashMap::new(),
            receiver,
            sender,
            high_resolution_cc: true,
            pending_msb: HashMap::new(),
            last_device_scan: Instant::now(),
            scan_interval: std::time::Duration::from_secs(10),
            max_scan_interval: std::time::Duration::from_secs(60),
        })
    }

    /// List available MIDI input ports
    pub fn list_ports(&self) -> Vec<(String, MidiInputPort)> {
        // Create a temporary MidiInput for listing ports
        match MidiInput::new("RustJay Waaaves MIDI Input - Port Lister") {
            Ok(midi_input) => {
                let ports = midi_input.ports();
                ports
                    .into_iter()
                    .filter_map(|port| {
                        midi_input.port_name(&port)
                            .ok()
                            .map(|name| (name, port))
                    })
                    .collect()
            }
            Err(e) => {
                log::warn!("Failed to create MIDI input for port listing: {}", e);
                Vec::new()
            }
        }
    }

    /// Connect to a specific MIDI device
    pub fn connect_device(&mut self, name: &str, port: MidiInputPort) -> Result<(), Box<dyn std::error::Error>> {
        if self.connections.contains_key(name) {
            log::debug!("MIDI device '{}' already connected", name);
            return Ok(());
        }

        let device_name = name.to_string();
        let sender = self.sender.clone();
        let name_clone = name.to_string();

        // Create a new MidiInput for this connection
        let midi_input = MidiInput::new(&format!("RustJay Waaaves - {}", name))?;

        // Create connection with callback
        let connection = midi_input.connect(
            &port,
            name,
            move |timestamp, message, sender| {
                // Parse the MIDI message
                if let Some(msg) = MidiMessage::from_bytes(message) {
                    // DEBUG: Log aftertouch messages
                    match &msg {
                        MidiMessage::ChannelAftertouch { channel, pressure } => {
                            log::info!("[MIDI RX] Channel Aftertouch ch={} pressure={}", channel, pressure);
                        }
                        MidiMessage::PolyAftertouch { channel, note, pressure } => {
                            log::info!("[MIDI RX] Poly Aftertouch ch={} note={} pressure={}", channel, note, pressure);
                        }
                        MidiMessage::NoteOn { channel, note, velocity } => {
                            log::info!("[MIDI RX] Note On ch={} note={} vel={}", channel, note, velocity);
                        }
                        _ => {}
                    }
                    
                    // Handle 14-bit CC if needed
                    let event = MidiEvent {
                        device_id: name_clone.clone(),
                        message: msg,
                        timestamp,
                    };
                    
                    // Send to main thread
                    let _ = sender.send(event);
                }
            },
            sender,
        )?;

        log::info!("Connected to MIDI device: {}", name);
        
        self.connections.insert(
            name.to_string(),
            DeviceConnection {
                name: name.to_string(),
                _connection: connection,
            },
        );

        Ok(())
    }

    /// Connect to all available MIDI devices
    pub fn connect_all(&mut self) -> usize {
        let ports = self.list_ports();
        let mut connected = 0;

        for (name, port) in ports {
            if !self.connections.contains_key(&name) {
                if let Err(e) = self.connect_device(&name, port) {
                    log::warn!("Failed to connect to MIDI device '{}': {}", name, e);
                } else {
                    connected += 1;
                }
            }
        }

        if connected > 0 {
            log::info!("Connected to {} new MIDI device(s)", connected);
        }

        connected
    }

    /// Disconnect from a specific device
    pub fn disconnect_device(&mut self, name: &str) {
        if self.connections.remove(name).is_some() {
            log::info!("Disconnected from MIDI device: {}", name);
        }
    }

    /// Disconnect from all devices
    pub fn disconnect_all(&mut self) {
        self.connections.clear();
        log::info!("Disconnected from all MIDI devices");
    }

    /// Get list of connected device names
    pub fn connected_devices(&self) -> Vec<String> {
        self.connections.keys().cloned().collect()
    }

    /// Check for new devices and connect to them
    pub fn scan_and_connect(&mut self) -> usize {
        self.connect_all()
    }

    /// Poll for new MIDI events (call this regularly from main thread)
    pub fn poll_events(&mut self) -> Vec<MidiEvent> {
        let mut events = Vec::new();
        
        // Check for device changes periodically
        if self.last_device_scan.elapsed() > self.scan_interval {
            let connected = self.scan_and_connect();
            self.last_device_scan = Instant::now();
            if connected > 0 {
                self.scan_interval = std::time::Duration::from_secs(10);
            } else {
                let next = (self.scan_interval.as_secs().max(10) * 2)
                    .min(self.max_scan_interval.as_secs());
                self.scan_interval = std::time::Duration::from_secs(next);
            }
        }
        
        // Collect all pending events
        while let Ok(event) = self.receiver.try_recv() {
            // Process 14-bit CC if enabled
            let processed_event = if self.high_resolution_cc {
                self.process_high_resolution_cc(event)
            } else {
                event
            };
            events.push(processed_event);
        }
        
        events
    }

    /// Process a MIDI event for 14-bit CC support
    /// CC 0-31 are MSB, CC 32-63 are LSB for the same controller
    fn process_high_resolution_cc(&mut self, event: MidiEvent) -> MidiEvent {
        match &event.message {
            MidiMessage::ControlChange { channel, controller, value } => {
                let device_entry = self.pending_msb.entry(event.device_id.clone()).or_default();
                
                if *controller < 32 {
                    // MSB - store it and return the event with current value
                    device_entry.insert(*controller, *value);
                    event
                } else if *controller < 64 {
                    // LSB - combine with stored MSB
                    let msb_controller = *controller - 32;
                    if let Some(&msb) = device_entry.get(&msb_controller) {
                        // Create a high-resolution message
                        let high_res_value = ((msb as u16) << 7) | (*value as u16);
                        // Convert to 0-127 for consistency, or keep 14-bit?
                        // For now, let's keep the original message but you could emit a special high-res event
                        event
                    } else {
                        event
                    }
                } else {
                    event
                }
            }
            _ => event,
        }
    }

    /// Enable/disable high-resolution CC
    pub fn set_high_resolution_cc(&mut self, enabled: bool) {
        self.high_resolution_cc = enabled;
        if !enabled {
            self.pending_msb.clear();
        }
    }

    /// Get the number of connected devices
    pub fn device_count(&self) -> usize {
        self.connections.len()
    }

    /// Check if any devices are connected
    pub fn has_devices(&self) -> bool {
        !self.connections.is_empty()
    }
}

impl Drop for MidiInputHandler {
    fn drop(&mut self) {
        self.disconnect_all();
    }
}

/// Thread-safe wrapper for MIDI input handler
pub struct SharedMidiInput {
    inner: Arc<Mutex<MidiInputHandler>>,
}

impl SharedMidiInput {
    /// Create a new shared MIDI input handler
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let handler = MidiInputHandler::new()?;
        Ok(Self {
            inner: Arc::new(Mutex::new(handler)),
        })
    }

    /// Get connected devices
    pub fn connected_devices(&self) -> Vec<String> {
        self.inner.lock().unwrap().connected_devices()
    }

    /// Poll for MIDI events
    pub fn poll_events(&self) -> Vec<MidiEvent> {
        self.inner.lock().unwrap().poll_events()
    }

    /// Connect to all available devices
    pub fn connect_all(&self) -> usize {
        self.inner.lock().unwrap().connect_all()
    }

    /// Scan for and connect to new devices
    pub fn scan_and_connect(&self) -> usize {
        self.inner.lock().unwrap().scan_and_connect()
    }

    /// Check if any devices are connected
    pub fn has_devices(&self) -> bool {
        self.inner.lock().unwrap().has_devices()
    }

    /// Get the number of connected devices
    pub fn device_count(&self) -> usize {
        self.inner.lock().unwrap().device_count()
    }

    /// Enable/disable high-resolution CC
    pub fn set_high_resolution_cc(&self, enabled: bool) {
        self.inner.lock().unwrap().set_high_resolution_cc(enabled);
    }
}

impl Clone for SharedMidiInput {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}
