//! Minimal logits sampler — a local stand-in for
//! `candle_transformers::generation` (not vendored so the crate stays free of
//! the candle-transformers dependency). Supports the greedy default plus
//! temperature / top-k / top-p sampling.

use candle_core::{Result, Tensor};

#[derive(Debug, Clone)]
pub enum Sampling {
    /// Plain temperature sampling over the full distribution.
    All { temperature: f64 },
    /// Sample from the `k` most likely tokens.
    TopK { k: usize, temperature: f64 },
    /// Nucleus sampling: smallest set whose cumulative prob exceeds `p`.
    TopP { p: f64, temperature: f64 },
}

pub struct LogitsProcessor {
    rng: fastrand::Rng,
    sampling: Sampling,
}

impl LogitsProcessor {
    pub fn from_sampling(seed: u64, sampling: Sampling) -> Self {
        Self {
            rng: fastrand::Rng::with_seed(seed),
            sampling,
        }
    }

    /// Sample one token id from a 1-D `[vocab]` logits tensor.
    pub fn sample(&mut self, logits: &Tensor) -> Result<u32> {
        let logits: Vec<f32> = logits.to_dtype(candle_core::DType::F32)?.to_vec1()?;
        let (temperature, top_k, top_p) = match self.sampling {
            Sampling::All { temperature } => (temperature, 0usize, 1.0f64),
            Sampling::TopK { k, temperature } => (temperature, k, 1.0),
            Sampling::TopP { p, temperature } => (temperature, 0, p),
        };
        let temperature = temperature.max(1e-6) as f32;

        // Softmax with temperature.
        let max = logits.iter().cloned().fold(f32::MIN, f32::max);
        let mut probs: Vec<f32> = logits.iter().map(|&l| ((l - max) / temperature).exp()).collect();
        let sum: f32 = probs.iter().sum();
        for p in probs.iter_mut() {
            *p /= sum;
        }

        // Rank indices by descending probability.
        let mut idx: Vec<usize> = (0..probs.len()).collect();
        idx.sort_unstable_by(|&a, &b| probs[b].total_cmp(&probs[a]));

        // Restrict to the top-k / top-p nucleus.
        let mut keep = idx.len();
        if top_k > 0 {
            keep = keep.min(top_k);
        }
        if top_p < 1.0 {
            let mut cum = 0.0f32;
            for (n, &i) in idx.iter().enumerate() {
                cum += probs[i];
                if cum >= top_p as f32 {
                    keep = keep.min(n + 1);
                    break;
                }
            }
        }
        let kept = &idx[..keep.max(1)];

        // Renormalize over the kept set and sample.
        let kept_sum: f32 = kept.iter().map(|&i| probs[i]).sum();
        let r = self.rng.f32() * kept_sum;
        let mut acc = 0.0f32;
        for &i in kept {
            acc += probs[i];
            if r < acc {
                return Ok(i as u32);
            }
        }
        Ok(*kept.last().unwrap() as u32)
    }
}
