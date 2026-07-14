use crate::vocab::{Note, DRUM_PROGRAM};
use std::collections::HashMap;

/// Write MIDI file bytes from notes.
/// Uses raw MIDI byte construction for reliability.
pub fn notes_to_midi(notes: &[Note], velocity: u8, tempo_bpm: u16) -> Vec<u8> {
    let ticks_per_beat: u16 = 480;
    let tempo_us: u32 = 60_000_000 / tempo_bpm as u32;

    // Build sorted note-on/off events: (time_sec, pitch, program, is_on)
    let mut midi_events: Vec<(f64, u8, u8, bool)> = Vec::new();
    for note in notes {
        if note.is_drum {
            midi_events.push((note.onset, note.pitch, DRUM_PROGRAM, true));
            midi_events.push((note.offset, note.pitch, DRUM_PROGRAM, false));
        } else {
            midi_events.push((note.onset, note.pitch, note.program, true));
            midi_events.push((note.offset, note.pitch, note.program, false));
        }
    }
    midi_events.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // Convert to MIDI ticks
    let tick_from_sec = |sec: f64| -> u32 {
        (sec * 1_000_000.0 / tempo_us as f64 * ticks_per_beat as f64) as u32
    };

    // We'll construct MIDI as raw bytes
    let mut buf = Vec::new();

    // Header: MThd, length=6, format=1, ntrks=1, division
    let division: u16 = ticks_per_beat;
    write_midi_header(&mut buf, 1, 1, division);

    // Track chunk
    let mut track_data = Vec::new();

    // Tempo meta event
    write_vlq(&mut track_data, 0); // delta
    track_data.extend_from_slice(&[0xFF, 0x51, 0x03]);
    track_data.extend_from_slice(&[(tempo_us >> 16) as u8, (tempo_us >> 8) as u8, tempo_us as u8]);

    let mut program_to_channel: HashMap<u8, u8> = HashMap::new();
    let mut available_channels: Vec<u8> = (0..9).chain(10..16).collect();
    let mut drums_initialized = false;
    let mut last_tick: u32 = 0;

    for (time, pitch, program, is_on) in &midi_events {
        let tick = tick_from_sec(*time);
        // The delta belongs to whichever message is emitted first at this time.
        // If a program-change is written here it consumes the delta, so the
        // note that follows must use delta 0 — otherwise the time is counted
        // twice and every later event is offset by the first note's timestamp.
        let mut delta = tick.saturating_sub(last_tick);
        last_tick = tick;

        let channel: u8;
        if *program == DRUM_PROGRAM {
            if !drums_initialized {
                write_vlq(&mut track_data, delta);
                track_data.extend_from_slice(&[0xC9, 0x00]); // program change ch 9, prog 0
                drums_initialized = true;
                delta = 0;
            }
            channel = 9;
        } else if let Some(&ch) = program_to_channel.get(program) {
            channel = ch;
        } else {
            let ch = if available_channels.is_empty() { 15 } else { available_channels.remove(0) };
            write_vlq(&mut track_data, delta);
            track_data.push(0xC0 | ch); // program change
            track_data.push(*program);
            program_to_channel.insert(*program, ch);
            channel = ch;
            delta = 0;
        }

        write_vlq(&mut track_data, delta);
        if *is_on {
            track_data.push(0x90 | channel);
            track_data.push(*pitch);
            track_data.push(velocity);
        } else {
            track_data.push(0x80 | channel);
            track_data.push(*pitch);
            track_data.push(0);
        }
    }

    // End of track
    write_vlq(&mut track_data, 0);
    track_data.extend_from_slice(&[0xFF, 0x2F, 0x00]);

    write_midi_track(&mut buf, &track_data);
    buf
}

fn write_vlq(buf: &mut Vec<u8>, mut value: u32) {
    // Build in reverse order
    let mut bytes = Vec::new();
    bytes.push((value & 0x7F) as u8);
    value >>= 7;
    while value > 0 {
        bytes.push(0x80 | ((value & 0x7F) as u8));
        value >>= 7;
    }
    bytes.reverse();
    buf.extend_from_slice(&bytes);
}

fn write_midi_header(buf: &mut Vec<u8>, format: u16, ntrks: u16, division: u16) {
    buf.extend_from_slice(b"MThd");
    buf.extend_from_slice(&6u32.to_be_bytes());
    buf.extend_from_slice(&format.to_be_bytes());
    buf.extend_from_slice(&ntrks.to_be_bytes());
    buf.extend_from_slice(&division.to_be_bytes());
}

fn write_midi_track(buf: &mut Vec<u8>, data: &[u8]) {
    buf.extend_from_slice(b"MTrk");
    buf.extend_from_slice(&(data.len() as u32).to_be_bytes());
    buf.extend_from_slice(data);
}
