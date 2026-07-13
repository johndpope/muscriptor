# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

MuScriptor is a multi-instrument audio→MIDI transcription model. It transcribes
audio (WAV/mp3/flac/…) into a stream of note events, or into a MIDI file.

The repo contains **two independent implementations of the same model**:

- **`muscriptor/`** — the primary Python/PyTorch package (library + CLI + FastAPI
  server). This is what ships on PyPI and where nearly all logic lives.
- **`muscriptor-rs/`** — a standalone Rust port built on
  [candle](https://github.com/huggingface/candle). It reuses the same
  `model.safetensors` weights and `config.json`. It is a separate Cargo project
  (not a workspace member) with its own README covering GPU build flags.

`web/` is the browser client (Vite + TypeScript) for the Python server.

Model weights are **not** in the repo — they download from HuggingFace
(`hf://MuScriptor/muscriptor-<small|medium|large>`) on first use, gated behind a
CC BY-NC license that requires `hf auth login` or `HF_TOKEN`, and cache under
`~/.cache/muscriptor/`.

## Commands

### Python (primary)

```bash
uv sync                              # install deps + dev group (pytest, pre-commit)
uv run pytest                        # run all tests
uv run pytest tests/test_midi.py     # single file
uv run pytest tests/test_midi.py::test_name   # single test
uv run pytest -m integration         # integration tests (need weights; see below)
uv run ruff check --fix && uv run ruff format   # lint + format (also via pre-commit)
uv run muscriptor transcribe audio.wav -o out.mid
uv run muscriptor serve --model medium --device cuda   # web server on :8222
uv run muscriptor list-instruments   # valid --instruments names
```

Most unit tests run without weights. Tests marked `integration` (and the
`transcription_model` fixture) **skip unless a weights file exists** — see
`tests/conftest.py`, which looks for `muscriptor_weights_*.safetensors` at repo
root and a hardcoded local audio path. Don't expect those to run in CI/clean
checkouts.

### Web UI (for the Python server)

```bash
cd web && pnpm install && pnpm run build   # outputs to muscriptor/web_dist/
pnpm run dev                               # Vite dev server (proxies to backend)
```

`pnpm run build` must be run once so `muscriptor serve` can mount the UI from
`muscriptor/web_dist/` (that dir is gitignored but shipped in the wheel via
`[tool.hatch.build] artifacts`).

### Rust port

See `muscriptor-rs/README.md`. Key point: GPU backend is a **compile-time cargo
feature** (`cuda` / `cudnn` / `metal`, mutually exclusive; plain build = CPU).
On Linux with GCC >14, point nvcc at an older compiler:
`NVCC_CCBIN=/usr/bin/g++-13 cargo build --release --features cuda`.

## Architecture

### Transcription pipeline (`muscriptor/`)

The model is a **transformer decoder** that autoregressively generates MT3-style
MIDI tokens conditioned on an audio mel-spectrogram. The end-to-end flow:

1. **`TranscriptionModel`** (`transcription_model.py`) — the single user-facing
   entry point and the orchestrator. `load_model()` resolves a size keyword /
   path / URL to weights, reads architecture from the repo's `config.json`, and
   builds the `LMModel`. `transcribe()` splits audio into fixed **5-second
   chunks** (`_SEGMENT_DURATION`, 16 kHz), runs generation per chunk (optionally
   batched — 1 on CPU, 4 on GPU by default), decodes tokens back to notes, and
   **yields events as a generator** while work proceeds.
2. **Conditioning** (`modules/conditioners.py`) — each chunk's mel-spectrogram
   (`modules/mel_spectrogram.py`) plus optional instrument/dataset class
   conditions are packaged as `ConditioningAttributes` and fed to the model.
   `--instruments` names resolve through `tokenizer/mt3.py`.
3. **Generation** (`models/lm.py` + `modules/transformer.py` +
   `modules/streaming.py`) — `LMModel` wraps a `StreamingTransformer` with a KV
   cache (streaming state), doing greedy / temperature-sampling / beam-search
   decoding. `utils/sampling.py` holds the sampling logic.
4. **Token → note decoding** — the transformer emits MT3 event tokens;
   `tokenizer/notes.py` and `tokenizer/mt3.py` define the vocabulary and convert
   token sequences into `Note` objects (onset/offset/pitch/program; velocity is
   **not** preserved). `events.py`'s `decode_model_tokens` turns these into the
   public `NoteStartEvent`/`NoteEndEvent`/`ProgressEvent` stream.
5. **Output** — `utils/midi.py` builds a MIDI file from notes;
   `utils/auralization.py` renders a stereo check-mix (L=original, R=synthesized
   via fluidsynth). `main.py` (Typer CLI) and `server.py` (FastAPI, streams
   events as SSE) are thin consumers of the generator.

### Event-stream contract (important invariant)

`transcribe()` returns a generator with guarantees consumers rely on: every
`NoteStartEvent` is followed by exactly one matching `NoteEndEvent` (same
`index`) later in the stream; events within a chunk are emitted in temporal
order and all of chunk N precedes any of chunk N+1; drum hits are a start
immediately followed by its end. `ProgressEvent`s are advisory chunk-completion
anchors that pure note consumers can ignore. See `events.py` and the README's
type definitions before changing event ordering or fields.

### Lineage

The model/tokenizer code is adapted from **audiocraft** (`models/lm.py`,
transformer) and **YourMT3+** (`tokenizer/notes.py`, `tokenizer/mt3.py`); some
weight/embedding quirks (e.g. `ScaledEmbedding`'s zero index) exist to stay
checkpoint-compatible with those. The Rust port in `muscriptor-rs/src/` mirrors
this same structure (model, mel, vocab, midi, download) against the same weights.

## Conventions

- Python dependency and env management is **uv** (not pip/poetry directly);
  `uv.lock` is committed and kept in sync via a pre-commit hook.
- Lint/format is **ruff** (check + format), wired through `.pre-commit-config.yaml`.
- CLI convention: all chatty progress/timing goes to **stderr**; stdout is
  reserved for actual output so `-o -` piping works.
