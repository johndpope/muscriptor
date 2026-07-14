#!/usr/bin/env bash
#
# build-mac.sh — build muscriptor-rs on macOS (Apple Silicon → Metal).
#
# Usage:
#   ./build-mac.sh                 # release build with the Metal GPU backend
#   ./build-mac.sh --realtime      # also enable the live-mic (--mic) feature
#   ./build-mac.sh --cpu           # CPU-only build (no Metal; works on Intel too)
#   ./build-mac.sh --debug         # debug profile instead of --release
#   ./build-mac.sh --run song.wav  # build, then transcribe song.wav to song.mid
#
# Flags combine, e.g. `./build-mac.sh --realtime --run song.wav`.
set -euo pipefail

cd "$(dirname "$0")"

profile="release"
features=("metal")
run_input=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --realtime) features+=("realtime") ;;
    --cpu)      features=("${features[@]/metal/}") ;;   # drop metal → plain CPU
    --debug)    profile="debug" ;;
    --run)      run_input="${2:-}"; shift ;;
    -h|--help)  sed -n '3,14p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

# --- sanity checks -----------------------------------------------------------
if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "warning: not macOS ($(uname -s)); use 'cargo build --features cuda' on Linux+NVIDIA." >&2
fi
if ! command -v cargo >/dev/null 2>&1; then
  echo "error: cargo not found. Install Rust from https://rustup.rs and re-run." >&2
  exit 1
fi
# Metal shader compilation needs the Xcode command-line tools.
if printf '%s\n' "${features[@]}" | grep -qx metal && ! xcode-select -p >/dev/null 2>&1; then
  echo "error: Xcode command-line tools are required for the Metal backend." >&2
  echo "       run: xcode-select --install" >&2
  exit 1
fi

# --- assemble cargo args -----------------------------------------------------
# Collapse the features array (may contain an empty string after --cpu).
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
