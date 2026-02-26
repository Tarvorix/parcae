#!/usr/bin/env bash
# One-shot runner for the Rust + ONNX stack.
# Usage:
#   ./run_rust.sh            # start server with ONNX
#   ./run_rust.sh --ui       # start server + wasm UI (opens browser)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
MODEL="${PARCAE_MODEL_PATH:-$ROOT/models/rust/parcae_model.onnx}"
BOOK_MODEL="${PARCAE_BOOK_PATH:-$ROOT/models/rust/book/centurion_book_v1.bin}"
TB_MODEL="${PARCAE_TB_PATH:-$ROOT/models/rust/tablebases/centurion_tb_4pc_v1.bin}"
NNUE_MODEL="${PARCAE_NNUE_PATH:-$ROOT/models/rust/nnue/centurion_nnue_v1.bin}"
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

if [[ -f "$BOOK_MODEL" ]]; then
  export PARCAE_BOOK_PATH="$BOOK_MODEL"
  echo "Using Centurion book: $PARCAE_BOOK_PATH"
else
  unset PARCAE_BOOK_PATH || true
  echo "No Centurion book file found."
fi

if [[ -f "$TB_MODEL" ]]; then
  export PARCAE_TB_PATH="$TB_MODEL"
  echo "Using Centurion tablebase: $PARCAE_TB_PATH"
else
  unset PARCAE_TB_PATH || true
  echo "No Centurion tablebase file found."
fi

if [[ -f "$NNUE_MODEL" ]]; then
  export PARCAE_NNUE_PATH="$NNUE_MODEL"
  echo "Using Centurion NNUE: $PARCAE_NNUE_PATH"
else
  unset PARCAE_NNUE_PATH || true
  echo "No Centurion NNUE file found (Centurion will use handcrafted eval blend fallback)."
fi

# Help ONNX runtime locate dylib on macOS if installed by Homebrew.
if [[ -d "/opt/homebrew/opt/onnxruntime/lib" ]]; then
  export DYLD_LIBRARY_PATH="/opt/homebrew/opt/onnxruntime/lib${DYLD_LIBRARY_PATH+:$DYLD_LIBRARY_PATH}"
  echo "Using ONNX runtime lib: /opt/homebrew/opt/onnxruntime/lib"
fi
mkdir -p "$SERVER_TARGET_DIR" "$UI_TARGET_DIR"
echo "Using server cargo target dir: $SERVER_TARGET_DIR"
echo "Using UI cargo target dir: $UI_TARGET_DIR"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
export HOST PORT
echo "Server bind target: $HOST:$PORT"

cd "$ROOT/rust"
# Build first so startup failures are visible before backgrounding.
echo "Building Rust server binary..."
CARGO_TARGET_DIR="$SERVER_TARGET_DIR" cargo build -p server
SERVER_BIN="$SERVER_TARGET_DIR/debug/server"
if [[ ! -x "$SERVER_BIN" ]]; then
  echo "Server binary missing after build: $SERVER_BIN" >&2
  exit 1
fi

# Start the backend server binary.
"$SERVER_BIN" &
SERVER_PID=$!
echo "Server PID: $SERVER_PID (http://$HOST:$PORT)"

# Readiness check so ERR_CONNECTION_REFUSED is caught early.
server_ready=false
for _ in $(seq 1 80); do
  if curl -fsS "http://$HOST:$PORT/health" >/dev/null 2>&1; then
    server_ready=true
    break
  fi
  sleep 0.25
done

if [[ "$server_ready" != true ]]; then
  echo "Server failed to become ready on http://$HOST:$PORT/health" >&2
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
  exit 1
fi
echo "Server is ready."

UI_PID=""
if $want_ui; then
  cd "$ROOT/rust/wasm-ui"
  if ! command -v trunk >/dev/null 2>&1; then
    echo "Missing trunk (install once: cargo install trunk)" >&2
    kill "$SERVER_PID" 2>/dev/null || true
    exit 1
  fi
  export API_BASE="${API_BASE:-http://$HOST:$PORT}"
  # Use a fresh dist dir to avoid macOS-protected files from previous builds.
  DIST_DIR="$ROOT/rust/wasm-ui/dist_local"
  CARGO_TARGET_DIR="$UI_TARGET_DIR" trunk serve --dist "$DIST_DIR" --open &
  UI_PID=$!
  echo "UI PID: $UI_PID (serving via trunk)"
fi

trap '[[ -n "$UI_PID" ]] && kill "$UI_PID" 2>/dev/null; kill "$SERVER_PID" 2>/dev/null' INT TERM
wait
