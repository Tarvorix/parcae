#!/usr/bin/env bash
# One-shot runner for the Rust + ONNX stack.
# Usage:
#   ./run_rust.sh            # start server with ONNX
#   ./run_rust.sh --ui       # start server + wasm UI (opens browser)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
MODEL="${PARCAE_MODEL_PATH:-$ROOT/models/rust/parcae_model.onnx}"
PTH_DEFAULT="$ROOT/models/python/parcae_model.pth"
PTH_CHECKPOINT="$ROOT/checkpoints/parcae_latest.pt"
EXPORT_SCRIPT="$ROOT/scripts/export_onnx.py"
SERVER_TARGET_DIR="$ROOT/rust/target_server_local"
UI_TARGET_DIR="$ROOT/rust/target_wasm_local"

want_ui=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ui) want_ui=true; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

use_onnx=true

# Ensure we have an ONNX model; auto-export if a .pth exists.
if [[ ! -f "$MODEL" ]]; then
  SRC=""
  if [[ -f "$PTH_DEFAULT" ]]; then
    SRC="$PTH_DEFAULT"
  elif [[ -f "$PTH_CHECKPOINT" ]]; then
    SRC="$PTH_CHECKPOINT"
  fi

  if [[ -z "$SRC" ]]; then
    echo "No ONNX model found at $MODEL and no compatible checkpoint source found."
    echo "Continuing with heuristic AI fallback."
    use_onnx=false
  else
    echo "Exporting ONNX from $SRC -> $MODEL ..."
    if ! PYTHONPATH="$ROOT/src${PYTHONPATH+:$PYTHONPATH}" \
      python3 "$EXPORT_SCRIPT" --checkpoint "$SRC" --out "$MODEL"; then
      echo "ONNX export failed. Continuing with heuristic AI fallback."
      use_onnx=false
    fi
  fi
fi

if [[ "$use_onnx" == true && -f "$MODEL" ]]; then
  export PARCAE_MODEL_PATH="$MODEL"
  echo "Using ONNX model: $PARCAE_MODEL_PATH"
else
  unset PARCAE_MODEL_PATH || true
  echo "Starting Rust server without ONNX model (heuristic AI fallback)."
fi
mkdir -p "$SERVER_TARGET_DIR" "$UI_TARGET_DIR"
echo "Using server cargo target dir: $SERVER_TARGET_DIR"
echo "Using UI cargo target dir: $UI_TARGET_DIR"

cd "$ROOT/rust"
# Start the backend server with ONNX (AI feature lives in ai crate via dependency).
CARGO_TARGET_DIR="$SERVER_TARGET_DIR" cargo run -p server &
SERVER_PID=$!
echo "Server PID: $SERVER_PID (http://127.0.0.1:8000)"

UI_PID=""
if $want_ui; then
  cd "$ROOT/rust/wasm-ui"
  if ! command -v trunk >/dev/null 2>&1; then
    echo "Missing trunk (install once: cargo install trunk)" >&2
    kill "$SERVER_PID" 2>/dev/null || true
    exit 1
  fi
  export API_BASE="${API_BASE:-http://127.0.0.1:8000}"
  # Use a fresh dist dir to avoid macOS-protected files from previous builds.
  DIST_DIR="$ROOT/rust/wasm-ui/dist_local"
  CARGO_TARGET_DIR="$UI_TARGET_DIR" trunk serve --dist "$DIST_DIR" --open &
  UI_PID=$!
  echo "UI PID: $UI_PID (serving via trunk)"
fi

trap '[[ -n "$UI_PID" ]] && kill "$UI_PID" 2>/dev/null; kill "$SERVER_PID" 2>/dev/null' INT TERM
wait
