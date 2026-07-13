mod config;
mod audio;
mod mel;
mod model;
mod vocab;
mod midi;
mod download;

use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use candle_core::{Device, Tensor};

use config::{ModelConfig, DEFAULT_CONFIGS};
use vocab::{build_event_vocab, instrument_group_from_names, decode_tokens, events_to_notes};

/// MuScriptor-rs: Audio-to-MIDI transcription in Rust
#[derive(Parser)]
#[command(name = "muscriptor-rs", version)]
struct Cli {
    /// Path to input audio file (WAV)
    #[arg(short, long)]
    input: PathBuf,

    /// Path to output MIDI file (default: <input>.mid)
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Model size: small, medium, large (default: medium)
    #[arg(short = 'm', long, default_value = "medium")]
    model: String,

    /// Path to local model.safetensors (overrides --model)
    #[arg(long)]
    weights: Option<PathBuf>,

    /// Instrument names to condition on (comma-separated)
    #[arg(short, long)]
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
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let cli = Cli::parse();

    // ---- Device ----
    let device = if cli.cpu {
        Device::Cpu
    } else {
        Device::cuda_if_available(0).unwrap_or(Device::Cpu)
    };
    log::info!("Using device: {:?}", device);

    // ---- Config ----
    let cfg = if let Some(ref wpath) = cli.weights {
        let cfg_path = wpath.parent().unwrap_or(std::path::Path::new(".")).join("config.json");
        if cfg_path.exists() {
            let cfg_str = std::fs::read_to_string(&cfg_path)?;
            let cfg_json: serde_json::Value = serde_json::from_str(&cfg_str)?;
            ModelConfig {
                dim: cfg_json["dim"].as_u64().unwrap_or(1024) as usize,
                num_heads: cfg_json["num_heads"].as_u64().unwrap_or(16) as usize,
                num_layers: cfg_json["num_layers"].as_u64().unwrap_or(24) as usize,
                card: cfg_json["card"].as_u64().unwrap_or(1395) as usize,
                hidden_scale: 4,
                max_period: 10000.0,
            }
        } else {
            let fname = wpath.to_string_lossy();
            DEFAULT_CONFIGS.iter()
                .find(|(name, _)| fname.contains(name))
                .map(|(_, cfg)| cfg.clone())
                .unwrap_or_else(|| {
                    log::warn!("Unknown model, using medium defaults");
                    DEFAULT_CONFIGS.iter().find(|(n, _)| *n == "medium").unwrap().1.clone()
                })
        }
    } else {
        let size = cli.model.as_str();
        DEFAULT_CONFIGS.iter()
            .find(|(n, _)| *n == size)
            .map(|(_, cfg)| cfg.clone())
            .ok_or_else(|| format!("Unknown model size: {}. Use small, medium, or large.", size))?
    };

    log::info!("Model config: {} layers, {} dim, {} heads, {} card",
        cfg.num_layers, cfg.dim, cfg.num_heads, cfg.card);

    // ---- Download / find weights ----
    let weights_path = if let Some(ref wp) = cli.weights {
        wp.clone()
    } else {
        let url = format!("hf://MuScriptor/muscriptor-{}/model.safetensors", cli.model);
        download::download_weights(&url)?
    };
    log::info!("Weights: {}", weights_path.display());

    // ---- Load model ----
    log::info!("Loading model...");
    let t0 = Instant::now();
    let model = model::LMModel::load(&weights_path, &cfg, &device)?;
    log::info!("Model loaded in {:.2}s", t0.elapsed().as_secs_f64());

    // ---- Build vocab ----
    let max_shift = 1001;
    let vocab = build_event_vocab(max_shift);
    log::info!("Vocabulary size: {}", vocab.len());

    // ---- Load audio ----
    log::info!("Loading audio: {}", cli.input.display());
    let t0 = Instant::now();
    let audio = audio::load_audio(&cli.input, 16000)?;
    let total_secs = audio.len() as f64 / 16000.0;
    log::info!("Audio loaded: {:.1}s ({:.2}ms)", total_secs, t0.elapsed().as_secs_f64());

    // ---- Mel spectrogram setup ----
    let mel_spec = mel::MelSpectrogram::new(16000, 2048, 160, 512);

    // ---- Instrument conditioning ----
    let inst_tokens = if let Some(ref inst_str) = cli.instruments {
        let names: Vec<String> = inst_str.split(',').map(|s| s.trim().to_string()).collect();
        instrument_group_from_names(&names)
    } else {
        None
    };

    // ---- Chunk & generate ----
    let sample_rate = 16000usize;
    let segment_dur = 5.0f64;
    let frame_rate = 100usize;
    let segment_samples = (segment_dur * sample_rate as f64) as usize;
    let num_chunks = (audio.len() + segment_samples - 1) / segment_samples;

    log::info!("Processing {} chunks of {}s", num_chunks, segment_dur);

    let mut all_tokens: Vec<(usize, Vec<i64>)> = Vec::new();

    for chunk_idx in 0..num_chunks {
        log::info!("Chunk {}/{}", chunk_idx + 1, num_chunks);

        let start = chunk_idx * segment_samples;
        let end = (start + segment_samples).min(audio.len());
        let mut chunk_audio: Vec<f32> = audio[start..end].to_vec();
        chunk_audio.resize(segment_samples, 0.0f32);

        // Compute mel for this chunk
        let t0 = Instant::now();
        let chunk_mel_raw = mel_spec.compute(&chunk_audio);
        let chunk_log_mel = mel_spec.log_mel(&chunk_mel_raw, 1e-6);
        // chunk_log_mel: Vec<Vec<f32>> where [T][512]
        let t_frames = chunk_log_mel.len();
        let mel_flat: Vec<f32> = chunk_log_mel.into_iter().flatten().collect();
        let mel_t = Tensor::from_vec(mel_flat, (1, t_frames, 512), &device)?;
        log::info!("  mel: {} frames ({:.2}ms)", t_frames, t0.elapsed().as_secs_f64());

        let inst_t = model.ic.tokenize(
            &[inst_tokens.as_ref().map(|s| {
                s.split_whitespace().filter_map(|v| v.parse::<i64>().ok()).collect()
            })],
            &device,
        )?;
        let ds_t = model.dc.tokenize(&[None], &device)?;

        let t0 = Instant::now();
        let tokens = model.generate(
            &mel_t, &inst_t, &ds_t,
            cli.max_gen_len, cli.sampling, cli.temperature,
            cli.top_k, cli.top_p,
        )?;
        log::info!("  -> {} tokens generated ({:.2}s)", tokens.len(), t0.elapsed().as_secs_f64());

        all_tokens.push((chunk_idx, tokens));
    }

    // ---- Decode tokens to events ----
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

    // ---- Convert events to notes and write MIDI ----
    let notes = events_to_notes(&all_events);
    log::info!("Transcribed {} notes", notes.len());

    let output_path = cli.output.unwrap_or_else(|| {
        let mut p = cli.input.clone();
        p.set_extension("mid");
        p
    });

    let midi_bytes = midi::notes_to_midi(&notes, 100, 120);
    std::fs::write(&output_path, &midi_bytes)?;
    log::info!("MIDI written to {}", output_path.display());

    Ok(())
}
