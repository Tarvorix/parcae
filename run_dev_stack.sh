#!/usr/bin/env bash
# One-command local stack runner:
# - Rust API (127.0.0.1:8000)
# - React/Vite web (127.0.0.1:5173)
#
# Usage:
#   ./run_dev_stack.sh
#
# Optional env:
#   HOST=127.0.0.1 PORT=8000 WEB_PORT=5173 ./run_dev_stack.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
WEB_PORT="${WEB_PORT:-5173}"
API_BASE="http://$HOST:$PORT"
WEB_URL="http://$HOST:$WEB_PORT"
SERVER_LOG="$ROOT/.parcae_server.log"
MODEL="$ROOT/models/rust/parcae_model.onnx"
BOOK="$ROOT/models/rust/book/centurion_book_v1.bin"
TB="$ROOT/models/rust/tablebases/centurion_tb_4pc_v1.bin"
NNUE="$ROOT/models/rust/nnue/centurion_nnue_v1.bin"
PARCAE_REQUIRE_ONNX="${PARCAE_REQUIRE_ONNX:-0}"
PARCAE_CENTURION_STRICT="${PARCAE_CENTURION_STRICT:-1}"
PARCAE_CENTURION_REQUIRE_BOOK="${PARCAE_CENTURION_REQUIRE_BOOK:-$PARCAE_CENTURION_STRICT}"
PARCAE_CENTURION_REQUIRE_TB="${PARCAE_CENTURION_REQUIRE_TB:-$PARCAE_CENTURION_STRICT}"
PARCAE_CENTURION_REQUIRE_NNUE="${PARCAE_CENTURION_REQUIRE_NNUE:-$PARCAE_CENTURION_STRICT}"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "Building Rust server..."
cd "$ROOT/rust"
cargo build -p server
SERVER_BIN="$ROOT/rust/target/debug/server"
if [[ ! -x "$SERVER_BIN" ]]; then
  echo "Server binary missing: $SERVER_BIN" >&2
  exit 1
fi

echo "Starting Rust API at $API_BASE ..."
SERVER_ENV=(
  "HOST=$HOST"
  "PORT=$PORT"
  "PARCAE_CENTURION_STRICT=$PARCAE_CENTURION_STRICT"
  "PARCAE_CENTURION_REQUIRE_BOOK=$PARCAE_CENTURION_REQUIRE_BOOK"
  "PARCAE_CENTURION_REQUIRE_TB=$PARCAE_CENTURION_REQUIRE_TB"
  "PARCAE_CENTURION_REQUIRE_NNUE=$PARCAE_CENTURION_REQUIRE_NNUE"
)

if [[ -f "$MODEL" ]]; then
  SERVER_ENV+=("PARCAE_MODEL_PATH=$MODEL")
  echo "Abaddon model: $MODEL"
else
  if [[ "$PARCAE_REQUIRE_ONNX" == "1" ]]; then
    echo "Missing required Abaddon model: $MODEL" >&2
    exit 1
  fi
  echo "Abaddon model missing (backend disabled): $MODEL"
fi

if [[ -f "$BOOK" ]]; then
  SERVER_ENV+=("PARCAE_BOOK_PATH=$BOOK")
  echo "Centurion book: $BOOK"
else
  if [[ "$PARCAE_CENTURION_REQUIRE_BOOK" == "1" ]]; then
    echo "Missing required Centurion book: $BOOK" >&2
    exit 1
  fi
  echo "Centurion book missing (allowed by config): $BOOK"
fi

if [[ -f "$TB" ]]; then
  SERVER_ENV+=("PARCAE_TB_PATH=$TB")
  echo "Centurion tablebase: $TB"
else
  if [[ "$PARCAE_CENTURION_REQUIRE_TB" == "1" ]]; then
    echo "Missing required Centurion tablebase: $TB" >&2
    exit 1
  fi
  echo "Centurion tablebase missing (allowed by config): $TB"
fi

if [[ -f "$NNUE" ]]; then
  SERVER_ENV+=("PARCAE_NNUE_PATH=$NNUE")
  echo "Centurion NNUE: $NNUE"
else
  if [[ "$PARCAE_CENTURION_REQUIRE_NNUE" == "1" ]]; then
    echo "Missing required Centurion NNUE: $NNUE" >&2
    exit 1
  fi
  echo "Centurion NNUE missing (allowed by config): $NNUE"
fi

echo "Centurion strict: $PARCAE_CENTURION_STRICT (book=$PARCAE_CENTURION_REQUIRE_BOOK tb=$PARCAE_CENTURION_REQUIRE_TB nnue=$PARCAE_CENTURION_REQUIRE_NNUE)"

# Help ONNX runtime locate dylib on macOS if installed by Homebrew.
ORT_LIB=""
if [[ -d "/opt/homebrew/opt/onnxruntime/lib" ]]; then
  ORT_LIB="/opt/homebrew/opt/onnxruntime/lib"
else
  ORT_LIB="$(ls -d /opt/homebrew/Cellar/onnxruntime/*/lib 2>/dev/null | tail -n 1 || true)"
fi
if [[ -n "$ORT_LIB" && -d "$ORT_LIB" ]]; then
  SERVER_ENV+=("DYLD_LIBRARY_PATH=$ORT_LIB${DYLD_LIBRARY_PATH+:$DYLD_LIBRARY_PATH}")
  echo "ONNX runtime lib: $ORT_LIB"
fi

env "${SERVER_ENV[@]}" "$SERVER_BIN" >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

server_ready=false
for _ in $(seq 1 80); do
  if curl -fsS "$API_BASE/health" >/dev/null 2>&1; then
    server_ready=true
    break
  fi
  sleep 0.25
done

if [[ "$server_ready" != true ]]; then
  echo "Rust API failed to start at $API_BASE" >&2
  echo "--- server log tail ---" >&2
  tail -n 60 "$SERVER_LOG" >&2 || true
  exit 1
fi

echo "Rust API ready: $API_BASE"
echo "Verifying AI capabilities endpoint..."
curl -fsS "$API_BASE/ai/capabilities" >/dev/null
echo "AI endpoint ready."

cd "$ROOT/web"
if [[ ! -d node_modules ]]; then
  echo "Installing web dependencies..."
  npm install
fi

echo "Starting web dev server at $WEB_URL (API -> $API_BASE)"
VITE_API_URL="$API_BASE" npm run dev -- --host "$HOST" --port "$WEB_PORT"
