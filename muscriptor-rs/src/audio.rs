use std::path::Path;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum AudioError {
    #[error("WAV error: {0}")]
    Hound(#[from] hound::Error),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Load a WAV file and return mono f32 samples at 16kHz.
/// Handles PCM WAV via hound. Converts to mono, resamples to 16kHz.
pub fn load_audio(path: &Path, target_sr: u32) -> Result<Vec<f32>, AudioError> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let orig_sr = spec.sample_rate;
    let channels = spec.channels as usize;

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => {
            reader.samples::<f32>().map(|s| s.unwrap_or(0.0)).collect()
        }
        hound::SampleFormat::Int => {
        let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
        reader
            .samples::<i32>()
            .map(|s| s.unwrap_or(0) as f32 / max)
                .collect()
        }
    };

    // Convert to mono by averaging channels
    let mono: Vec<f32> = if channels > 1 {
        samples
            .chunks(channels)
            .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
            .collect()
    } else {
        samples
    };

    // Resample to target_sr
    if orig_sr != target_sr {
        Ok(resample(&mono, orig_sr, target_sr))
    } else {
        Ok(mono)
    }
}

/// Simple linear interpolation resampler
pub fn resample(input: &[f32], orig_sr: u32, target_sr: u32) -> Vec<f32> {
    if orig_sr == target_sr {
        return input.to_vec();
    }
    let ratio = orig_sr as f64 / target_sr as f64;
    let output_len = (input.len() as f64 / ratio).ceil() as usize;
    let mut output = Vec::with_capacity(output_len);

    for i in 0..output_len {
        let src_pos = i as f64 * ratio;
        let src_idx = src_pos as usize;
        let frac = src_pos - src_idx as f64;
        if src_idx + 1 < input.len() {
            let a = input[src_idx] as f64;
            let b = input[src_idx + 1] as f64;
            output.push((a + (b - a) * frac) as f32);
        } else {
            output.push(input[input.len() - 1]);
        }
    }
    output
}
