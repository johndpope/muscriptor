mod config;
mod audio;
mod mel;
mod model;
mod vocab;
mod midi;
mod download;
#[cfg(feature = "realtime")]
mod realtime;

use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use candle_core::{Device, Tensor};

use config::{ModelConfig, DEFAULT_CONFIGS};
use vocab::{build_event_vocab, decode_tokens, events_to_notes, instrument_group_from_names,
    tie_section_token_ids, OpenNoteTracker, FRAME_RATE};

/// MuScriptor-rs: Audio-to-MIDI transcription in Rust
#[derive(Parser)]
#[command(name = "muscriptor-rs", version)]
struct Cli {
    /// Path to input audio file (WAV); omit with --mic for live capture
    #[arg(short, long)]
    input: Option<PathBuf>,

    /// Path to output MIDI file (default: <input>.mid); with --mic, notes go to stdout
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Live microphone capture mode
    #[arg(long)]
    mic: bool,

    /// Model size: small, medium, large (default: medium)
    #[arg(short = 'm', long, default_value = "medium")]
    model: String,

    /// Path to local model.safetensors (overrides --model)
    #[arg(long)]
    weights: Option<PathBuf>,

    /// Instrument names to condition on (comma-separated)
    #[arg(short = 'I', long)]
    instruments: Option<String>,

    /// Use sampling instead of greedy decoding
    #[arg(long)]
    sampling: bool,

    /// Sampling temperature
    #[arg(long, default_value = "1.0")]
    temperature: f64,

    /// Top-k sampling
    #[arg(long, default_value = "0")]
    top_k: usize,

    /// Top-p sampling
    #[arg(long, default_value = "0.0")]
    top_p: f64,

    /// Use CPU only
    #[arg(long)]
    cpu: bool,

    /// Maximum generation length per chunk
    #[arg(long, default_value = "2000")]
    max_gen_len: usize,

    /// Disable prelude forcing (teacher-forcing each chunk's tie prologue from
    /// the previous chunk's still-open notes, so chunks can't restart sustained
    /// notes with the wrong instrument). On by default.
    #[arg(long = "no-prelude-forcing", action = clap::ArgAction::SetTrue)]
    no_prelude_forcing: bool,
}

fn load_device(cpu: bool) -> Device {
    if cpu {
        log::info!("Using CPU device");
        return Device::Cpu;
    }
    // Try CUDA first, then Metal. Only the backend compiled in via cargo
    // features can succeed; the others report as unavailable and fall through.
    let dev = Device::cuda_if_available(0)
        .or_else(|_| Device::metal_if_available(0))
        .unwrap_or(Device::Cpu);
    match &dev {
        Device::Cuda(_) => log::info!("Using CUDA device 0"),
        Device::Metal(_) => log::info!("Using Metal device 0"),
        Device::Cpu => log::warn!("No GPU backend available; falling back to CPU"),
    }
    dev
}

fn resolve_config(model_size: &str, weights: &Option<PathBuf>) -> Result<ModelConfig, Box<dyn std::error::Error>> {
    if let Some(ref wpath) = weights {
        let cfg_path = wpath.parent().unwrap_or(std::path::Path::new(".")).join("config.json");
        if cfg_path.exists() {
            let cfg_str = std::fs::read_to_string(&cfg_path)?;
            let cfg_json: serde_json::Value = serde_json::from_str(&cfg_str)?;
            return Ok(ModelConfig {
                dim: cfg_json["dim"].as_u64().unwrap_or(1024) as usize,
                num_heads: cfg_json["num_heads"].as_u64().unwrap_or(16) as usize,
                num_layers: cfg_json["num_layers"].as_u64().unwrap_or(24) as usize,
                card: cfg_json["card"].as_u64().unwrap_or(1395) as usize,
                hidden_scale: 4,
                max_period: 10000.0,
            });
        }
        let fname = wpath.to_string_lossy();
        if let Some((_, cfg)) = DEFAULT_CONFIGS.iter().find(|(name, _)| fname.contains(name)) {
            return Ok(cfg.clone());
        }
        log::warn!("Unknown model, using medium defaults");
    }
    DEFAULT_CONFIGS
        .iter()
        .find(|(n, _)| *n == model_size)
        .map(|(_, c)| c.clone())
        .ok_or_else(|| format!("Unknown model size: {model_size}. Use small, medium, or large.").into())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let cli = Cli::parse();

    let device = load_device(cli.cpu);
    log::info!("Using device: {:?}", device);

    let cfg = resolve_config(&cli.model, &cli.weights)?;
    log::info!("Model config: {} layers, {} dim, {} heads, {} card",
        cfg.num_layers, cfg.dim, cfg.num_heads, cfg.card);

    let weights_path = if let Some(ref wp) = cli.weights {
        wp.clone()
    } else {
        let url = format!("hf://MuScriptor/muscriptor-{}/model.safetensors", cli.model);
        download::download_weights(&url)?
    };
    log::info!("Weights: {}", weights_path.display());

    log::info!("Loading model...");
    let t0 = Instant::now();
    let model = model::LMModel::load(&weights_path, &cfg, &device)?;
    log::info!("Model loaded in {:.2}s", t0.elapsed().as_secs_f64());

    let inst_names: Option<Vec<String>> = cli.instruments.as_ref().map(|s| {
        s.split(',').map(|s| s.trim().to_string()).collect()
    });

    if cli.mic {
        #[cfg(feature = "realtime")]
        {
            run_realtime(model, &cli, inst_names)?;
        }
        #[cfg(not(feature = "realtime"))]
        {
            let _ = (model, inst_names);
            return Err("--mic requires building with the `realtime` feature \
                        (cargo build --release --features realtime,cuda)".into());
        }
    } else {
        run_file(model, &cli, inst_names)?;
    }

    Ok(())
}

fn run_file(
    model: model::LMModel,
    cli: &Cli,
    inst_names: Option<Vec<String>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let input = cli.input.as_ref().ok_or("--input required for file mode")?;
    let max_shift = 1001;
    let vocab = build_event_vocab(max_shift);
    // Mel front-end with window/filterbank loaded from the checkpoint.
    let mel_spec = model.mel();

    let inst_tokens = inst_names.as_ref().map(|names| instrument_group_from_names(names).unwrap_or_default());

    log::info!("Loading audio: {}", input.display());
    let t0 = Instant::now();
    let audio = audio::load_audio(input, 16000)?;
    let total_secs = audio.len() as f64 / 16000.0;
    log::info!("Audio loaded: {:.1}s ({:.2}ms)", total_secs, t0.elapsed().as_secs_f64());

    let segment_dur = 5.0f64;
    let segment_samples = (segment_dur * 16000.0) as usize;
    let num_chunks = (audio.len() + segment_samples - 1) / segment_samples;
    log::info!("Processing {} chunks of {}s", num_chunks, segment_dur);

    let mut all_tokens: Vec<(usize, Vec<i64>)> = Vec::new();

    let inst_t = model.ic.tokenize(
        &[inst_tokens.as_ref().map(|s| {
            s.split_whitespace().filter_map(|v| v.parse::<i64>().ok()).collect()
        })],
        &model.device,
    )?;
    let ds_t = model.dc.tokenize(&[None], &model.device)?;

    // Prelude forcing: teacher-force each chunk's tie prologue from the notes
    // the previous chunk left open, so a chunk can't restart sustained notes
    // with the wrong instrument. Chunks are generated strictly in order here,
    // so it is always applicable.
    let prelude_forcing = !cli.no_prelude_forcing;
    if prelude_forcing {
        log::info!("Prelude forcing enabled");
    }
    let mut tracker = OpenNoteTracker::new(&vocab, FRAME_RATE);

    for chunk_idx in 0..num_chunks {
        log::info!("Chunk {}/{}", chunk_idx + 1, num_chunks);
        let start = chunk_idx * segment_samples;
        let end = (start + segment_samples).min(audio.len());
        let mut chunk_audio: Vec<f32> = audio[start..end].to_vec();
        chunk_audio.resize(segment_samples, 0.0f32);

        let seek_time = chunk_idx as f64 * segment_dur;
        let next_seek_time = if chunk_idx + 1 < num_chunks {
            Some((chunk_idx + 1) as f64 * segment_dur)
        } else {
            None
        };
        // Settle the tracker on the boundary before reading its open notes.
        tracker.feed_boundary(seek_time, next_seek_time);
        let prompt: Vec<i64> = if prelude_forcing && chunk_idx > 0 {
            tie_section_token_ids(&tracker.open_keys(), &vocab)
        } else {
            Vec::new()
        };

        let t0 = Instant::now();
        let chunk_mel_raw = mel_spec.compute(&chunk_audio);
        let chunk_log_mel = mel_spec.log_mel(&chunk_mel_raw, 1e-6);
        let t_frames = chunk_log_mel.len();
        let mel_flat: Vec<f32> = chunk_log_mel.into_iter().flatten().collect();
        let mel_t = Tensor::from_vec(mel_flat, (1, t_frames, 512), &model.device)?;
        log::info!("  mel: {} frames ({:.2}ms)", t_frames, t0.elapsed().as_secs_f64());

        let t0 = Instant::now();
        let tokens = model.generate(
            &mel_t, &inst_t, &ds_t,
            cli.max_gen_len, cli.sampling, cli.temperature,
            cli.top_k, cli.top_p,
            &prompt,
        )?;
        log::info!("  -> {} tokens ({:.2}s)", tokens.len(), t0.elapsed().as_secs_f64());

        // Feed the chunk's tokens (forced prologue included) so the tracker's
        // open-note set is ready for the next chunk's prologue.
        for &t in &tokens {
            tracker.feed_token(t);
        }
        all_tokens.push((chunk_idx, tokens));
    }

    log::info!("Decoding tokens...");
    let mut all_events = Vec::new();
    for (chunk_idx, tokens) in &all_tokens {
        let seek_time = *chunk_idx as f64 * segment_dur;
        let next_seek_time = if chunk_idx + 1 < num_chunks {
            Some((chunk_idx + 1) as f64 * segment_dur)
        } else {
            None
        };
        let events = decode_tokens(tokens, &vocab, seek_time, next_seek_time);
        all_events.extend(events);
    }

    let notes = events_to_notes(&all_events);
    log::info!("Transcribed {} notes", notes.len());

    let output_path = cli.output.clone().unwrap_or_else(|| {
        let mut p = input.clone();
        p.set_extension("mid");
        p
    });
    let midi_bytes = midi::notes_to_midi(&notes, 100, 120);
    std::fs::write(&output_path, &midi_bytes)?;
    log::info!("MIDI written to {}", output_path.display());
    Ok(())
}

#[cfg(feature = "realtime")]
fn run_realtime(
    model: model::LMModel,
    cli: &Cli,
    inst_names: Option<Vec<String>>,
) -> Result<(), Box<dyn std::error::Error>> {
    log::info!("Starting realtime mic capture...");

    let transcriber = realtime::RealtimeTranscriber::new(
        model,
        inst_names,
        cli.max_gen_len,
        cli.sampling,
        cli.temperature,
        cli.top_k,
        cli.top_p,
    );

    let (_stream, rx) = realtime::start_mic_capture()?;
    println!("🎤 Listening... Play music! Notes appear below.");
    println!("format: start_time\\tpitch\\tduration\\tprogram");

    for (chunk_audio, start_time) in rx {
        match transcriber.transcribe_chunk(&chunk_audio, start_time) {
            Ok(notes) => {
                for note in &notes {
                    let dur = note.offset - note.onset;
                    println!("{:.3}\t{}\t{:.3}\t{}",
                        note.onset, note.pitch, dur, note.program);
                }
                // Flush stdout so the Flutter subprocess can read lines
                use std::io::Write;
                std::io::stdout().flush().ok();
            }
            Err(e) => eprintln!("Transcription error: {e}"),
        }
    }

    Ok(())
}
