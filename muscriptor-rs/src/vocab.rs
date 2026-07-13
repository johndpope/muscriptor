use std::collections::{HashMap, HashSet};

const SPECIAL_TOKENS: &[&str] = &["PAD", "EOS", "UNK"];

#[derive(Debug, Clone)]
pub struct Event {
    pub etype: String,
    pub value: i32,
}

pub fn build_event_vocab(max_shift_steps: usize) -> Vec<Event> {
    let mut vocab = Vec::new();
    for token in SPECIAL_TOKENS {
        vocab.push(Event { etype: token.to_string(), value: 0 });
    }
    for v in 0..max_shift_steps {
        vocab.push(Event { etype: "shift".to_string(), value: v as i32 });
    }
    for v in 0..128 {
        vocab.push(Event { etype: "pitch".to_string(), value: v });
    }
    for v in 0..2 {
        vocab.push(Event { etype: "velocity".to_string(), value: v });
    }
    vocab.push(Event { etype: "tie".to_string(), value: 0 });
    for v in 0..130 {
        vocab.push(Event { etype: "program".to_string(), value: v });
    }
    for v in 0..128 {
        vocab.push(Event { etype: "drum".to_string(), value: v });
    }
    vocab
}

pub fn eos_id() -> usize { 1 }

/// Map instrument group name to a space-separated list of group IDs.
pub fn instrument_group_from_names(names: &[String]) -> Option<String> {
    if names.is_empty() {
        return None;
    }
    let groups: HashMap<&str, i64> = [
        ("acoustic_piano", 0), ("electric_piano", 1),
        ("chromatic_percussion", 2), ("organ", 3),
        ("acoustic_guitar", 4), ("clean_electric_guitar", 5),
        ("distorted_electric_guitar", 6), ("acoustic_bass", 7),
        ("electric_bass", 8), ("violin", 9), ("viola", 10),
        ("cello", 11), ("contrabass", 12), ("orchestral_harp", 13),
        ("timpani", 14), ("string_ensemble", 15), ("synth_strings", 16),
        ("voice", 17), ("orchestra_hit", 18), ("trumpet", 19),
        ("trombone", 20), ("tuba", 21), ("french_horn", 22),
        ("brass_section", 23), ("soprano_and_alto_sax", 24),
        ("tenor_sax", 25), ("baritone_sax", 26), ("oboe", 27),
        ("english_horn", 28), ("bassoon", 29), ("clarinet", 30),
        ("flutes", 31), ("synth_lead", 32), ("synth_pad", 33),
        ("drums", 36),
    ].iter().cloned().collect();

    let ids: Vec<String> = names.iter()
        .filter_map(|n| groups.get(n.as_str()).map(|id| id.to_string()))
        .collect();
    if ids.is_empty() { None } else { Some(ids.join(" ")) }
}

#[derive(Debug, Clone)]
pub enum TranscriptionEvent {
    Progress { completed: usize, total: usize },
    NoteStart { pitch: u8, start_time: f64, index: usize, instrument: String },
    NoteEnd { end_time: f64, start_event_index: usize },
}

#[derive(Debug, Clone)]
pub struct Note {
    pub is_drum: bool,
    pub program: u8,
    pub onset: f64,
    pub offset: f64,
    pub pitch: u8,
}

pub const DRUM_PROGRAM: u8 = 128;
pub const MINIMUM_NOTE_DURATION: f64 = 0.01;
pub const FRAME_RATE: usize = 100;

pub fn instrument_for_program(program: u8) -> String {
    let groups: HashMap<u8, &str> = [
        (0, "acoustic_piano"), (2, "electric_piano"),
        (8, "chromatic_percussion"), (16, "organ"),
        (24, "acoustic_guitar"), (26, "clean_electric_guitar"),
        (29, "distorted_electric_guitar"), (32, "acoustic_bass"),
        (40, "electric_bass"), (41, "violin"), (42, "viola"),
        (43, "cello"), (44, "contrabass"), (46, "orchestral_harp"),
        (47, "timpani"), (48, "string_ensemble"), (50, "synth_strings"),
        (52, "voice"), (55, "orchestra_hit"), (56, "trumpet"),
        (57, "trombone"), (58, "tuba"), (60, "french_horn"),
        (61, "brass_section"), (64, "soprano_and_alto_sax"),
        (66, "tenor_sax"), (67, "baritone_sax"), (68, "oboe"),
        (69, "english_horn"), (70, "bassoon"), (71, "clarinet"),
        (72, "flutes"), (80, "synth_lead"), (88, "synth_pad"),
    ].iter().cloned().collect();
    groups.get(&program).map(|s| s.to_string()).unwrap_or_else(|| format!("program_{}", program))
}

pub fn decode_tokens(
    tokens: &[i64],
    vocab: &[Event],
    seek_time: f64,
    next_seek_time: Option<f64>,
) -> Vec<TranscriptionEvent> {
    let mut events = Vec::new();
    let mut next_index: usize = 0;
    let mut open_notes: HashMap<(u8, u8), usize> = HashMap::new();
    let mut open_note_info: HashMap<usize, (u8, f64)> = HashMap::new();

    let start_tick = (seek_time * FRAME_RATE as f64).round() as i32;
    let mut tick_state = start_tick;
    let mut program_state: Option<u8> = None;
    let mut velocity_state: Option<u8> = None;
    let mut in_prologue = true;
    let mut tie_set: HashSet<(u8, u8)> = HashSet::new();
    let mut skip_rest = false;

    for &token_id in tokens {
        if skip_rest { break; }
        if token_id < 0 || token_id as usize >= vocab.len() { continue; }
        let event = &vocab[token_id as usize];
        match event.etype.as_str() {
            "PAD" | "EOS" | "UNK" => continue,
            _ => {}
        }

        if in_prologue {
            match event.etype.as_str() {
                "tie" => {
                    in_prologue = false;
                    velocity_state = None;
                    let to_close: Vec<(u8, u8)> = open_notes.keys()
                        .filter(|k| !tie_set.contains(k)).copied().collect();
                    for key in to_close {
                        if let Some(idx) = open_notes.remove(&key) {
                            events.push(TranscriptionEvent::NoteEnd {
                                end_time: seek_time, start_event_index: idx,
                            });
                            open_note_info.remove(&idx);
                        }
                    }
                }
                "shift" => {
                    in_prologue = false;
                    skip_rest = true;
                    for (_, idx) in open_notes.drain() {
                        events.push(TranscriptionEvent::NoteEnd {
                            end_time: seek_time, start_event_index: idx,
                        });
                    }
                }
                "program" => program_state = Some(event.value as u8),
                "pitch" => {
                    if let Some(prog) = program_state {
                        tie_set.insert((prog, event.value as u8));
                    }
                }
                _ => {}
            }
            continue;
        }

        match event.etype.as_str() {
            "shift" => { if event.value > 0 { tick_state = start_tick + event.value; } }
            "program" => program_state = Some(event.value as u8),
            "velocity" => velocity_state = Some(event.value as u8),
            "drum" => {
                let time = tick_state as f64 / FRAME_RATE as f64;
                if next_seek_time.map_or(true, |nst| time < nst) {
                    let idx = next_index; next_index += 1;
                    events.push(TranscriptionEvent::NoteStart {
                        pitch: event.value as u8, start_time: time, index: idx,
                        instrument: "drums".to_string(),
                    });
                    events.push(TranscriptionEvent::NoteEnd {
                        end_time: time + MINIMUM_NOTE_DURATION, start_event_index: idx,
                    });
                }
            }
            "pitch" => {
                let (prog, vel) = match (program_state, velocity_state) {
                    (Some(p), Some(v)) => (p, v),
                    _ => continue,
                };
                let time = tick_state as f64 / FRAME_RATE as f64;
                if next_seek_time.map_or(false, |nst| time >= nst) { continue; }
                let key = (prog, event.value as u8);
                if let Some(&prev_idx) = open_notes.get(&key) {
                    events.push(TranscriptionEvent::NoteEnd {
                        end_time: time, start_event_index: prev_idx,
                    });
                    open_note_info.remove(&prev_idx);
                    open_notes.remove(&key);
                }
                if vel > 0 {
                    let idx = next_index; next_index += 1;
                    events.push(TranscriptionEvent::NoteStart {
                        pitch: event.value as u8, start_time: time, index: idx,
                        instrument: instrument_for_program(prog),
                    });
                    open_notes.insert(key, idx);
                    open_note_info.insert(idx, (prog, time));
                }
            }
            _ => {}
        }
    }

    // Close remaining open notes
    let end_time = next_seek_time.unwrap_or(seek_time + 5.0);
    for (_, idx) in open_notes.drain() {
        let t = open_note_info.get(&idx)
            .map(|&(_, st)| st + MINIMUM_NOTE_DURATION)
            .unwrap_or(end_time);
        events.push(TranscriptionEvent::NoteEnd { end_time: t, start_event_index: idx });
    }

    events
}

pub fn events_to_notes(events: &[TranscriptionEvent]) -> Vec<Note> {
    let mut notes = Vec::new();
    let mut open: HashMap<usize, Note> = HashMap::new();
    for ev in events {
        match ev {
            TranscriptionEvent::Progress { .. } => {}
            TranscriptionEvent::NoteStart { pitch, start_time, index, instrument } => {
                let is_drum = instrument == "drums";
                let program = if is_drum { DRUM_PROGRAM } else {
                    // map instrument name back to program
                    let groups: HashMap<&str, u8> = [
                        ("acoustic_piano", 0), ("electric_piano", 2),
                        ("chromatic_percussion", 8), ("organ", 16),
                        ("acoustic_guitar", 24), ("clean_electric_guitar", 26),
                        ("distorted_electric_guitar", 29), ("acoustic_bass", 32),
                        ("electric_bass", 40), ("violin", 41), ("viola", 42),
                        ("cello", 43), ("contrabass", 44), ("orchestral_harp", 46),
                        ("timpani", 47), ("string_ensemble", 48),
                        ("synth_strings", 50), ("voice", 52),
                        ("orchestra_hit", 55), ("trumpet", 56),
                        ("trombone", 57), ("tuba", 58), ("french_horn", 60),
                        ("brass_section", 61), ("soprano_and_alto_sax", 64),
                        ("tenor_sax", 66), ("baritone_sax", 67), ("oboe", 68),
                        ("english_horn", 69), ("bassoon", 70), ("clarinet", 71),
                        ("flutes", 72), ("synth_lead", 80), ("synth_pad", 88),
                    ].iter().cloned().collect();
                    *groups.get(instrument.as_str()).unwrap_or(&0)
                };
                open.insert(*index, Note {
                    is_drum, program, onset: *start_time,
                    offset: *start_time, pitch: *pitch,
                });
            }
            TranscriptionEvent::NoteEnd { end_time, start_event_index } => {
                if let Some(mut note) = open.remove(start_event_index) {
                    note.offset = *end_time;
                    notes.push(note);
                }
            }
        }
    }
    // Validate and trim overlapping notes
    notes = validate_notes(notes);
    notes = trim_overlapping(notes);
    notes
}

fn validate_notes(notes: Vec<Note>) -> Vec<Note> {
    notes.into_iter().filter(|n| {
        n.onset.is_finite() && n.offset.is_finite()
    }).map(|mut n| {
        if n.offset - n.onset < MINIMUM_NOTE_DURATION {
            n.offset = n.onset + MINIMUM_NOTE_DURATION;
        }
        n
    }).collect()
}

fn trim_overlapping(notes: Vec<Note>) -> Vec<Note> {
    if notes.len() <= 1 { return notes; }
    let mut by_key: HashMap<(u8, u8, bool), Vec<Note>> = HashMap::new();
    for n in notes {
        by_key.entry((n.program, n.pitch, n.is_drum)).or_default().push(n);
    }
    let mut result = Vec::new();
    for (_key, mut channel_notes) in by_key {
        channel_notes.sort_by(|a, b| a.onset.partial_cmp(&b.onset).unwrap());
        for i in 1..channel_notes.len() {
            if channel_notes[i - 1].offset > channel_notes[i].onset {
                channel_notes[i - 1].offset = channel_notes[i].onset;
            }
        }
        channel_notes.retain(|n| n.onset < n.offset);
        result.extend(channel_notes);
    }
    result.sort_by(|a, b| a.onset.partial_cmp(&b.onset).unwrap());
    result
}
