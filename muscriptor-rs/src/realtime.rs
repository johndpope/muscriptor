//! Real-time microphone capture → streaming transcription.
//! Captures audio from the default mic, buffers into overlapping chunks, and
//! runs the model on each chunk to emit detected notes on stdout.

use std::collections::VecDeque;
use std::io::Write;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use crossbeam_channel::{bounded, Sender};

use crate::model::{GenerateOptions, Model, FRAME_RATE, SAMPLE_RATE, SEGMENT_DURATION};
use crate::tokenizer::{self, Note};

const CHUNK_DURATION_SECS: f64 = 4.0;
const OVERLAP_SECS: f64 = 1.0;

pub struct RealtimeTranscriber {
    model: Model,
    inst_ids: Option<Vec<u32>>,
    opts: GenerateOptions,
}

impl RealtimeTranscriber {
    pub fn new(model: Model, inst_ids: Option<Vec<u32>>, opts: GenerateOptions) -> Self {
        Self { model, inst_ids, opts }
    }

    /// Transcribe one audio chunk (16 kHz mono f32) into notes, onsets shifted
    /// to absolute `chunk_start_time`.
    pub fn transcribe_chunk(
        &mut self,
        chunk_audio: &[f32],
        chunk_start_time: f64,
    ) -> Result<Vec<Note>, Box<dyn std::error::Error>> {
        if chunk_audio.is_empty() {
            return Ok(vec![]);
        }
        // The model works on fixed 5-second segments; zero-pad the mic window.
        let segment_samples = (SEGMENT_DURATION * SAMPLE_RATE as f64) as usize;
        let mut segment = chunk_audio.to_vec();
        segment.resize(segment_samples, 0.0);

        let prefix = self.model.build_prefix(&[segment], self.inst_ids.as_deref())?;
        let rows = self.model.generate(&prefix, &self.opts, |_| Ok(()))?;
        let tokens = rows.into_iter().next().unwrap_or_default();

        // Decode this single chunk in isolation.
        let mut decoder = tokenizer::TokenDecoder::new(FRAME_RATE);
        let mut events = Vec::new();
        decoder.start_chunk(0.0, None, &mut events);
        for tok in tokens {
            decoder.push(tok, &mut events);
        }
        decoder.finish(&mut events);
        let notes = tokenizer::events_to_notes(&events);

        // Keep onsets inside the fresh (non-overlapped) part of the window and
        // shift to absolute time.
        let keep_from = if chunk_start_time > 0.0 { CHUNK_DURATION_SECS - OVERLAP_SECS } else { 0.0 };
        Ok(notes
            .into_iter()
            .filter(|n| n.onset >= keep_from && n.onset < CHUNK_DURATION_SECS)
            .map(|mut n| {
                let shift = chunk_start_time - keep_from;
                n.onset += shift;
                n.offset += shift;
                n
            })
            .collect())
    }
}

/// Entry point for `--mic`: capture from the default input device and print
/// notes as they are detected.
pub fn run_realtime(
    model: Model,
    inst_ids: Option<Vec<u32>>,
    opts: GenerateOptions,
    json: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut transcriber = RealtimeTranscriber::new(model, inst_ids, opts);

    log::info!("Starting realtime mic capture...");
    let (_stream, rx) = start_mic_capture()?;
    // Status/banner goes to stderr so stdout stays a clean machine-readable
    // stream (TSV, or JSONL for a UI).
    eprintln!("🎤 Listening... play music!");
    if !json {
        eprintln!("(stdout columns: start_time\tpitch\tduration\tprogram)");
    }
    for (chunk_audio, start_time) in rx {
        match transcriber.transcribe_chunk(&chunk_audio, start_time) {
            Ok(notes) => {
                for n in &notes {
                    if json {
                        // One JSON object per line (JSONL) — instrument is a
                        // fixed-vocabulary identifier, so no escaping needed.
                        println!(
                            "{{\"start_time\":{:.3},\"duration\":{:.3},\"pitch\":{},\"program\":{},\"instrument\":\"{}\",\"is_drum\":{}}}",
                            n.onset,
                            n.offset - n.onset,
                            n.pitch,
                            n.program,
                            tokenizer::instrument_for_program(n.program),
                            n.is_drum,
                        );
                    } else {
                        println!("{:.3}\t{}\t{:.3}\t{}", n.onset, n.pitch, n.offset - n.onset, n.program);
                    }
                }
                std::io::stdout().flush().ok();
            }
            Err(e) => eprintln!("Transcription error: {e}"),
        }
    }
    Ok(())
}

/// Start microphone capture. Returns a receiver yielding (audio_chunk, start_time).
pub fn start_mic_capture(
) -> Result<
    (cpal::Stream, crossbeam_channel::Receiver<(Vec<f32>, f64)>),
    Box<dyn std::error::Error>,
> {
    let host = cpal::default_host();
    let input_device = host
        .default_input_device()
        .ok_or("No input device available")?;
    let config = input_device.default_input_config()?;
    let channels = config.channels() as usize;
    let device_sr = config.sample_rate().0;

    // The ring buffer holds mono samples at the device's native rate; chunks
    // are resampled to 16 kHz before being sent to the model.
    let capacity = device_sr as usize * 30;
    let ring: Arc<Mutex<VecDeque<f32>>> =
        Arc::new(Mutex::new(VecDeque::with_capacity(capacity)));
    let ring_clone = ring.clone();
    let err_fn = |err| eprintln!("Audio stream error: {err}");

    let stream = input_device.build_input_stream(
        &config.into(),
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            let mut buf = ring_clone.lock().unwrap();
            if channels > 1 {
                for ch in data.chunks(channels) {
                    let mono: f32 = ch.iter().sum::<f32>() / channels as f32;
                    if buf.len() >= capacity {
                        buf.pop_front();
                    }
                    buf.push_back(mono);
                }
            } else {
                for &s in data {
                    if buf.len() >= capacity {
                        buf.pop_front();
                    }
                    buf.push_back(s);
                }
            }
        },
        err_fn,
        None,
    )?;

    let (tx, rx): (Sender<(Vec<f32>, f64)>, _) = bounded(10);
    let chunk_samples = (CHUNK_DURATION_SECS * device_sr as f64) as usize;

    thread::spawn(move || {
        let mut last_time = 0.0f64;
        loop {
            thread::sleep(Duration::from_millis(100));
            let mut read_buf = vec![0.0f32; chunk_samples];
            let filled = {
                let buf = ring.lock().unwrap();
                if buf.len() >= chunk_samples {
                    let start = buf.len() - chunk_samples;
                    for (i, s) in buf.range(start..).enumerate() {
                        read_buf[i] = *s;
                    }
                    chunk_samples
                } else {
                    0
                }
            };
            if filled > 0 {
                // Resample the native-rate window to the model's 16 kHz.
                let chunk16 = crate::audio::resample(&read_buf, device_sr, SAMPLE_RATE as u32);
                let t = last_time;
                last_time += CHUNK_DURATION_SECS - OVERLAP_SECS;
                if tx.send((chunk16, t)).is_err() {
                    break;
                }
            }
        }
    });

    stream.play()?;
    Ok((stream, rx))
}
