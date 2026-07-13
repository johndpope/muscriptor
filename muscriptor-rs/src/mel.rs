use std::sync::Arc;
use rustfft::{Fft, FftPlanner};
use rustfft::num_complex::Complex;

/// Mel spectrogram matching the Python implementation (HTK scale, power=1.0).
pub struct MelSpectrogram {
    n_fft: usize,
    hop_length: usize,
    n_mels: usize,
    sample_rate: u32,
    window: Vec<f64>,
    fb: Vec<Vec<f64>>,   // mel filterbank [n_freqs, n_mels]
    fft: Arc<dyn Fft<f64>>,
}

impl MelSpectrogram {
    pub fn new(sample_rate: u32, n_fft: usize, hop_length: usize, n_mels: usize) -> Self {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(n_fft);

        // torch.hann_window(n_fft) is *periodic* (denominator n_fft), not the
        // symmetric window (denominator n_fft-1).
        let window: Vec<f64> = (0..n_fft)
            .map(|i| {
                0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / n_fft as f64).cos())
            })
            .collect();

        let fb = melscale_fbanks(n_fft / 2 + 1, 0.0, sample_rate as f64 / 2.0, n_mels, sample_rate);

        MelSpectrogram {
            n_fft,
            hop_length,
            n_mels,
            sample_rate,
            window,
            fb,
            fft,
        }
    }

    /// Compute mel spectrogram for mono audio at the model's sample rate.
    /// Returns [n_mels, n_frames] as f32.
    ///
    /// Matches `torch.stft(center=True, pad_mode="reflect")`: the signal is
    /// reflect-padded by n_fft/2 on each side, so frame `m` is centred at
    /// `m * hop` in the original signal and the frame count is `n/hop + 1`.
    pub fn compute(&self, audio: &[f32]) -> Vec<Vec<f32>> {
        let n = audio.len();
        let pad = self.n_fft / 2;
        // Reflect-pad the signal (no edge repetition, matching torch 'reflect').
        let padded_len = n + 2 * pad;
        let mut sig = vec![0.0f64; padded_len];
        for (i, s) in sig.iter_mut().enumerate() {
            let idx = i as isize - pad as isize;
            *s = audio[reflect_index(idx, n)] as f64;
        }

        let n_frames = if n == 0 { 1 } else { n / self.hop_length + 1 };
        let mut mel_spec = vec![vec![0.0f32; n_frames]; self.n_mels];

        for frame in 0..n_frames {
            let start = frame * self.hop_length;

            // Apply Hann window and compute FFT over the padded signal.
            let mut fft_in: Vec<Complex<f64>> = (0..self.n_fft)
                .map(|i| {
                    let s = sig.get(start + i).copied().unwrap_or(0.0);
                    Complex::new(s * self.window[i], 0.0)
                })
                .collect();

            self.fft.process(&mut fft_in);

            // Magnitude spectrum (power=1.0)
            let mag: Vec<f64> = fft_in[..self.n_fft / 2 + 1]
                .iter()
                .map(|c| c.norm())
                .collect();

            // Apply mel filterbank
            for m in 0..self.n_mels {
                let mut val = 0.0f64;
                for f in 0..mag.len() {
                    val += mag[f] * self.fb[f][m];
                }
                mel_spec[m][frame] = val as f32;
            }
        }

        mel_spec
    }

    /// Apply log and rearrange to [frames, n_mels]
    pub fn log_mel(&self, raw: &[Vec<f32>], eps: f32) -> Vec<Vec<f32>> {
        let n_frames = raw[0].len();
        let mut out = vec![vec![0.0f32; self.n_mels]; n_frames];
        for t in 0..n_frames {
            for m in 0..self.n_mels {
                out[t][m] = (raw[m][t] + eps).ln();
            }
        }
        out
    }
}

/// Reflect an index into [0, n) with torch 'reflect' semantics (the boundary
/// samples are not repeated), i.e. a triangle wave of period 2*(n-1).
fn reflect_index(idx: isize, n: usize) -> usize {
    if n <= 1 {
        return 0;
    }
    let period = 2 * (n as isize - 1);
    let mut r = idx.rem_euclid(period);
    if r >= n as isize {
        r = period - r;
    }
    r as usize
}

fn hz_to_mel_htk(freq: f64) -> f64 {
    2595.0 * (1.0 + freq / 700.0).log10()
}

fn mel_to_hz_htk(mel: f64) -> f64 {
    700.0 * (10.0_f64.powf(mel / 2595.0) - 1.0)
}

fn melscale_fbanks(
    n_freqs: usize,
    f_min: f64,
    f_max: f64,
    n_mels: usize,
    _sample_rate: u32,
) -> Vec<Vec<f64>> {
    let m_min = hz_to_mel_htk(f_min);
    let m_max = hz_to_mel_htk(f_max);
    let m_pts: Vec<f64> = (0..n_mels + 2)
        .map(|i| m_min + (m_max - m_min) * i as f64 / (n_mels + 1) as f64)
        .collect();
    let f_pts: Vec<f64> = m_pts.iter().map(|&m| mel_to_hz_htk(m)).collect();

    let all_freqs: Vec<f64> = (0..n_freqs)
        .map(|i| i as f64 * f_max / (n_freqs - 1) as f64)
        .collect();

    let mut fb = vec![vec![0.0f64; n_mels]; n_freqs];

    for f in 0..n_freqs {
        for m in 0..n_mels {
            let up = (all_freqs[f] - f_pts[m]) / (f_pts[m + 1] - f_pts[m]);
            let down = (f_pts[m + 2] - all_freqs[f]) / (f_pts[m + 2] - f_pts[m + 1]);
            fb[f][m] = up.min(down).max(0.0);
        }
    }

    fb
}
