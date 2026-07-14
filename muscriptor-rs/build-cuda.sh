#!/usr/bin/env bash
#
# build-cuda.sh — build muscriptor-rs on Linux/Windows with the NVIDIA (CUDA)
# GPU backend.
#
# Usage:
#   ./build-cuda.sh                 # release build with the CUDA backend
#   ./build-cuda.sh --cudnn         # CUDA + cuDNN acceleration
#   ./build-cuda.sh --realtime      # also enable the live-mic (--mic) feature
#   ./build-cuda.sh --cpu           # CPU-only build (no CUDA toolkit needed)
#   ./build-cuda.sh --debug         # debug profile instead of --release
#   ./build-cuda.sh --run song.wav  # build, then transcribe song.wav to song.mid
#
# Flags combine, e.g. `./build-cuda.sh --cudnn --realtime --run song.wav`.
#
# nvcc rejects GCC newer than it supports; this script auto-points nvcc at the
# newest compatible g++ (g++-13/14) via NVCC_CCBIN when the default is too new.
# Override by exporting NVCC_CCBIN yourself before running.
set -euo pipefail

cd "$(dirname "$0")"

profile="release"
features=("cuda")
run_input=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cudnn)    features+=("cudnn") ;;
    --realtime) features+=("realtime") ;;
    --cpu)      features=("${features[@]/cuda/}") ;;   # drop cuda → plain CPU
    --debug)    profile="debug" ;;
    --run)      run_input="${2:-}"; shift ;;
    -h|--help)  sed -n '3,16p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

want_cuda=false
printf '%s\n' "${features[@]}" | grep -qx cuda && want_cuda=true

# --- sanity checks -----------------------------------------------------------
if ! command -v cargo >/dev/null 2>&1; then
  echo "error: cargo not found. Install Rust from https://rustup.rs and re-run." >&2
  exit 1
fi

if $want_cuda; then
  if ! command -v nvcc >/dev/null 2>&1; then
    echo "error: nvcc not found. Install the CUDA toolkit and put it on PATH," >&2
    echo "       e.g. export PATH=/usr/local/cuda/bin:\$PATH  (or use --cpu)." >&2
    exit 1
  fi
  # nvcc caps the host compiler version; if the default g++ is newer, point
  # NVCC_CCBIN at the newest supported g++ we can find.
  if [[ -z "${NVCC_CCBIN:-}" ]]; then
    max_gcc="$(nvcc --version 2>/dev/null | grep -oiE 'gcc versions? later than [0-9]+' | grep -oE '[0-9]+' | head -1 || true)"
    gpp_default="$(g++ -dumpversion 2>/dev/null | cut -d. -f1 || echo 0)"
    # If nvcc didn't advertise a cap, assume the common CUDA 12.x limit of 14.
    cap="${max_gcc:-14}"
    if [[ "${gpp_default:-0}" -gt "$cap" ]]; then
      for v in 14 13 12; do
        if [[ "$v" -le "$cap" ]] && command -v "g++-$v" >/dev/null 2>&1; then
          export NVCC_CCBIN="$(command -v "g++-$v")"
          echo "note: default g++ $gpp_default > nvcc cap $cap; using NVCC_CCBIN=$NVCC_CCBIN"
          break
        fi
      done
      if [[ -z "${NVCC_CCBIN:-}" ]]; then
        echo "warning: default g++ ($gpp_default) may be too new for nvcc (cap $cap) and no" >&2
        echo "         compatible g++-<=$cap found. Install one, e.g. sudo apt install g++-13." >&2
      fi
    fi
  else
    echo "note: using preset NVCC_CCBIN=$NVCC_CCBIN"
  fi
fi

# --- assemble cargo args -----------------------------------------------------
feat_csv="$(printf '%s,' "${features[@]}" | sed 's/,,*/,/g; s/^,//; s/,$//')"
args=(build)
[[ "$profile" == "release" ]] && args+=(--release)
[[ -n "$feat_csv" ]] && args+=(--features "$feat_csv")

echo "==> cargo ${args[*]}"
cargo "${args[@]}"

bin="target/${profile}/muscriptor-rs"
echo "==> built: $(cd "$(dirname "$bin")" && pwd)/$(basename "$bin")"

# --- optional run ------------------------------------------------------------
if [[ -n "$run_input" ]]; then
  if [[ ! -f "$run_input" ]]; then
    echo "error: --run file not found: $run_input" >&2
    exit 1
  fi
  out="${run_input%.*}.mid"
  echo "==> transcribing $run_input -> $out"
  RUST_LOG=info "$bin" --input "$run_input" -o "$out"
fi

echo "Done. Try:  $bin --help"
echo "Live mic:   $bin --mic --model small        (needs a --realtime build)"
