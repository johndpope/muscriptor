# muscriptor-rs

Audio-to-MIDI transcription using a transformer language model (Rust port, built on [candle](https://github.com/huggingface/candle)).

The model, mel front-end and MT3 token decoder are adapted from the
reference-grade candle implementation ([huggingface/candle#3738](https://github.com/huggingface/candle/pull/3738)):
the mel window/filterbank are loaded from the checkpoint buffers so the output
matches the PyTorch reference, a preallocated KV cache and fused attention keep
decoding fast, and chunks are batched per forward pass. This crate adds a
standalone CLI, HuggingFace weight download, and a real-time microphone mode.

## Building

The GPU backend is selected at compile time via cargo features. Pick the one
that matches your hardware:

```bash
# NVIDIA GPU (Linux / Windows) — requires the CUDA toolkit (nvcc) on PATH
cargo build --release --features cuda

# NVIDIA with cuDNN acceleration (optional, requires cuDNN installed)
cargo build --release --features cudnn

# Apple Silicon (macOS only)
cargo build --release --features metal

# CPU only (portable, no accelerator toolkit needed)
cargo build --release
```

> The `metal` crate is macOS-only and must **not** be enabled on Linux/Windows.
> `cuda` and `metal` are mutually exclusive — enable at most one.

### CUDA build gotchas (Linux)

The CUDA kernels are compiled by `nvcc` at build time. Two things commonly bite:

1. **Host compiler too new.** `nvcc` (CUDA ≤12.x) refuses GCC newer than 14 with
   `unsupported GNU version! gcc versions later than 14 are not supported`. If
   your default `gcc` is 15+, install an older toolchain and point `nvcc` at it
   via candle's `NVCC_CCBIN` env var:

   ```bash
   sudo apt-get install g++-13        # or g++-14
   NVCC_CCBIN=/usr/bin/g++-13 cargo build --release --features cuda
   ```

2. **`nvcc` not found.** Ensure the CUDA toolkit's `bin` is on `PATH`
   (e.g. `export PATH=/usr/local/cuda/bin:$PATH`). Verify with `nvcc --version`.

Verified on: RTX 3090, driver 580.x / CUDA 13, `nvcc` 12.8, `g++-13`.

At runtime the binary automatically uses whichever GPU backend was compiled in,
falling back to CPU if none is available. Pass `--cpu` to force CPU. The selected
device is logged at startup (run with `RUST_LOG=info`).

## Running

```bash
RUST_LOG=info ./target/release/muscriptor-rs --help
```
