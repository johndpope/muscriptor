mod audio;
mod config;
mod download;
mod mel;
mod midi;
mod model;
mod sampling;
mod tokenizer;
#[cfg(feature = "realtime")]
mod realtime;

use std::path::PathBuf;
use std::time::Instant;

use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use clap::Parser;

use config::{ModelConfig, DEFAULT_CONFIGS};
use model::{Config, GenerateOptions, Model, SAMPLE_RATE, SEGMENT_DURATION, FRAME_RATE};
use sampling::Sampling;

/// MuScriptor-rs: audio-to-MIDI transcription in Rust (candle port).
#[derive(Parser)]
#[command(name = "muscriptor-rs", version)]
struct Cli {
    /// Input audio file (WAV). Omit with --mic for live capture.
    #[arg(short, long)]
    input: Option<PathBuf>,

    /// Output MIDI file (default: <input>.mid). With --mic, notes go to stdout.
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Live microphone capture mode.
    #[arg(long)]
    mic: bool,

    /// Model size: small, medium, large (default: medium).
    #[arg(short = 'm', long, default_value = "medium")]
    model: String,

    /// Path to local model.safetensors (overrides --model).
    #[arg(long)]
    weights: Option<PathBuf>,

    /// Instrument names to condition on (comma-separated, e.g. acoustic_piano,drums).
    #[arg(short = 'I', long)]
    instruments: Option<String>,

    /// Use temperature sampling instead of greedy decoding.
    #[arg(long)]
    sampling: bool,

    /// Sampling temperature (only with --sampling).
    #[arg(short = 't', long, default_value = "1.0")]
    temperature: f64,

    /// Top-k sampling (only with --sampling).
    #[arg(long)]
    top_k: Option<usize>,

    /// Top-p / nucleus sampling (only with --sampling; wins over top-k).
    #[arg(long)]
    top_p: Option<f64>,

    /// RNG seed for --sampling.
    #[arg(long, default_value = "299792458")]
    seed: u64,

    /// Use CPU only.
    #[arg(long)]
    cpu: bool,

    /// Transformer compute dtype: f32, f16 or bf16. Defaults to f16 on Metal
    /// (decoding is bandwidth-bound there), f32 elsewhere. Conditioners always
    /// run in f32.
    #[arg(long)]
    dtype: Option<String>,

    /// Maximum generated tokens per chunk.
    #[arg(long, default_value = "2000")]
    max_gen_len: usize,

    /// Chunks transcribed per forward pass. Default: 4 on GPU, 1 on CPU.
    #[arg(long)]
    batch_size: Option<usize>,

    /// Print decoded note events to stderr.
    #[arg(long)]
    notes: bool,
}

fn load_device(cpu: bool) -> Device {
    if cpu {
        log::info!("Using CPU device");
        return Device::Cpu;
    }
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

fn build_options(cli: &Cli) -> GenerateOptions {
    let sampling = if !cli.sampling || cli.temperature <= 0.0 {
        None
    } else {
        let temperature = cli.temperature;
        Some(match (cli.top_p.filter(|&p| p > 0.0), cli.top_k.filter(|&k| k > 0)) {
            (Some(p), _) => Sampling::TopP { p, temperature },
            (None, Some(k)) => Sampling::TopK { k, temperature },
            (None, None) => Sampling::All { temperature },
        })
    };
    GenerateOptions { max_gen_len: cli.max_gen_len, sampling, seed: cli.seed }
}

/// Load weights + config and build the model (f32; conditioners always f32).
fn load_model(cli: &Cli, device: &Device) -> Result<Model, Box<dyn std::error::Error>> {
    let mcfg = resolve_config(&cli.model, &cli.weights)?;
    log::info!(
        "Model config: {} layers, {} dim, {} heads, {} card",
        mcfg.num_layers, mcfg.dim, mcfg.num_heads, mcfg.card
    );
    let config = Config {
        dim: mcfg.dim,
        num_heads: mcfg.num_heads,
        num_layers: mcfg.num_layers,
        card: mcfg.card,
    };

    let weights_path = if let Some(ref wp) = cli.weights {
        wp.clone()
    } else {
        let url = format!("hf://MuScriptor/muscriptor-{}/model.safetensors", cli.model);
        download::download_weights(&url)?
    };
    log::info!("Weights: {}", weights_path.display());

    // Transformer dtype: explicit --dtype, else f16 on Metal (bandwidth-bound
    // decoding) and f32 everywhere else. Conditioners always run in f32.
    let dtype = match cli.dtype.as_deref() {
        Some("f32") | Some("float32") => DType::F32,
        Some("f16") | Some("float16") => DType::F16,
        Some("bf16") | Some("bfloat16") => DType::BF16,
        Some(other) => return Err(format!("unsupported dtype '{other}' (use f32, f16 or bf16)").into()),
        None if device.is_metal() => DType::F16,
        None => DType::F32,
    };
    log::info!("Transformer dtype: {:?}", dtype);

    let t0 = Instant::now();
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&weights_path], dtype, device)? };
    let vb_f32 = unsafe { VarBuilder::from_mmaped_safetensors(&[&weights_path], DType::F32, device)? };
    let model = Model::new(&config, vb, vb_f32)?;
    log::info!("Model loaded in {:.2}s", t0.elapsed().as_secs_f64());
    Ok(model)
}

fn instrument_ids(cli: &Cli) -> Result<Option<Vec<u32>>, Box<dyn std::error::Error>> {
    match &cli.instruments {
        None => Ok(None),
        Some(s) => {
            let names: Vec<&str> = s.split(',').map(|n| n.trim()).filter(|n| !n.is_empty()).collect();
            if names.is_empty() {
                return Ok(None);
            }
            let ids = tokenizer::instrument_class_ids(&names).map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
            log::info!("Instruments: {}", names.join(", "));
            Ok(Some(ids))
        }
    }
}

fn print_event(ev: &tokenizer::NoteEvent) {
    match ev {
        tokenizer::NoteEvent::Start(s) => eprintln!(
            "NoteStart(pitch={}, t={:.2}, idx={}, instr={})",
            s.pitch, s.start_time, s.index, s.instrument
        ),
        tokenizer::NoteEvent::End { end_time, start_index } => {
            eprintln!("NoteEnd(t={end_time:.2}, start_index={start_index})")
        }
    }
}

fn run_file(mut model: Model, cli: &Cli) -> Result<(), Box<dyn std::error::Error>> {
    let input = cli.input.as_ref().ok_or("--input required for file mode")?;
    let inst_ids = instrument_ids(cli)?;
    let opts = build_options(cli);

    log::info!("Loading audio: {}", input.display());
    let pcm = audio::load_audio(input, SAMPLE_RATE as u32)?;
    let segment_samples = (SEGMENT_DURATION * SAMPLE_RATE as f64) as usize;
    let num_chunks = pcm.len().div_ceil(segment_samples).max(1);
    log::info!(
        "Audio: {:.1}s -> {} chunk(s) of {}s",
        pcm.len() as f64 / SAMPLE_RATE as f64, num_chunks, SEGMENT_DURATION
    );
    let chunks: Vec<Vec<f32>> = (0..num_chunks)
        .map(|i| {
            let start = i * segment_samples;
            let mut c = pcm[start..(start + segment_samples).min(pcm.len())].to_vec();
            c.resize(segment_samples, 0.0);
            c
        })
        .collect();

    let batch_size = cli.batch_size.unwrap_or(if model.device().is_cpu() { 1 } else { 4 }).max(1);
    log::info!("Batch size {}", batch_size);

    let seek_time = |i: usize| i as f64 * SEGMENT_DURATION;
    let next_seek = |i: usize| (i + 1 < num_chunks).then(|| seek_time(i + 1));

    let mut decoder = tokenizer::TokenDecoder::new(FRAME_RATE);
    let mut all_events: Vec<tokenizer::NoteEvent> = Vec::new();
    let mut emitted = 0usize; // index into all_events already printed

    let gen_start = Instant::now();
    for batch_start in (0..num_chunks).step_by(batch_size) {
        let batch = &chunks[batch_start..(batch_start + batch_size).min(num_chunks)];
        let n = batch.len();
        let prefix = model.build_prefix(batch, inst_ids.as_deref())?;

        // The model emits one token per chunk per step; the decoder consumes
        // whole chunks in order, so the first chunk streams live while later
        // ones buffer until every earlier chunk has hit EOS.
        let mut buffers: Vec<Vec<u32>> = vec![Vec::new(); n];
        let mut done = vec![false; n];
        let mut active = 0usize;
        decoder.start_chunk(seek_time(batch_start), next_seek(batch_start), &mut all_events);

        model.generate(&prefix, &opts, |tokens| {
            for (j, &tok) in tokens.iter().enumerate() {
                if done[j] {
                    continue;
                }
                if tok == tokenizer::EOS_ID {
                    done[j] = true;
                } else if j == active {
                    decoder.push(tok, &mut all_events);
                } else {
                    buffers[j].push(tok);
                }
            }
            while active < n && done[active] {
                active += 1;
                if active < n {
                    decoder.start_chunk(seek_time(batch_start + active), next_seek(batch_start + active), &mut all_events);
                    let buffered = std::mem::take(&mut buffers[active]);
                    for tok in buffered {
                        decoder.push(tok, &mut all_events);
                    }
                }
            }
            Ok(())
        })?;

        // Chunks after `active` never got their turn (an earlier chunk hit the
        // token cap without EOS): start and flush them in order.
        for j in (active + 1)..n {
            decoder.start_chunk(seek_time(batch_start + j), next_seek(batch_start + j), &mut all_events);
            let buffered = std::mem::take(&mut buffers[j]);
            for tok in buffered {
                decoder.push(tok, &mut all_events);
            }
        }
        for (j, d) in done.iter().enumerate() {
            if !d {
                log::warn!("chunk {} did not emit EOS within {} tokens", batch_start + j, cli.max_gen_len);
            }
        }
        if cli.notes {
            for ev in &all_events[emitted..] {
                print_event(ev);
            }
            emitted = all_events.len();
        }
        log::info!("chunks {}/{} ({:.2}s)", batch_start + n, num_chunks, gen_start.elapsed().as_secs_f64());
    }
    decoder.finish(&mut all_events);
    if cli.notes {
        for ev in &all_events[emitted..] {
            print_event(ev);
        }
    }

    let notes = tokenizer::events_to_notes(&all_events);
    log::info!("Transcribed {} notes ({:.2}s total)", notes.len(), gen_start.elapsed().as_secs_f64());

    let midi_bytes = midi::notes_to_midi_bytes(&notes);
    let output = cli.output.clone().unwrap_or_else(|| input.with_extension("mid"));
    std::fs::write(&output, &midi_bytes)?;
    log::info!("Wrote {} note(s) to {}", notes.len(), output.display());
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let cli = Cli::parse();

    let device = load_device(cli.cpu);
    let model = load_model(&cli, &device)?;

    if cli.mic {
        #[cfg(feature = "realtime")]
        {
            let inst_ids = instrument_ids(&cli)?;
            let opts = build_options(&cli);
            realtime::run_realtime(model, inst_ids, opts)?;
        }
        #[cfg(not(feature = "realtime"))]
        {
            let _ = model;
            return Err("--mic requires building with the `realtime` feature \
                        (cargo build --release --features realtime,cuda)".into());
        }
    } else {
        run_file(model, &cli)?;
    }
    Ok(())
}
