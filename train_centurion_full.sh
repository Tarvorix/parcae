#!/usr/bin/env bash
# One-command Centurion overnight training pipeline:
# 1) Opening book
# 2) Tablebase bootstrap artifact
# 3) NNUE weights
#
# Usage:
#   ./train_centurion_full.sh
#
# Optional overrides:
#   OUT_TAG=v1 THREADS=8 ./train_centurion_full.sh
#   BOOK_GAMES=80000 TB_GAMES=180000 NNUE_SAMPLES=350000 ./train_centurion_full.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
RUST_DIR="$ROOT/rust"
LOG_DIR="$ROOT/models/rust/logs"
mkdir -p "$LOG_DIR" "$ROOT/models/rust/book" "$ROOT/models/rust/tablebases" "$ROOT/models/rust/nnue"

default_threads() {
  if command -v sysctl >/dev/null 2>&1; then
    sysctl -n hw.ncpu 2>/dev/null || true
    return
  fi
  if command -v nproc >/dev/null 2>&1; then
    nproc 2>/dev/null || true
    return
  fi
  echo 4
}

THREADS="${THREADS:-$(default_threads)}"
THREADS="${THREADS:-4}"
OUT_TAG="${OUT_TAG:-v1}"
RUN_ID="$(date +%Y%m%d-%H%M%S)"
LOG_FILE="$LOG_DIR/centurion_train_${RUN_ID}.log"
DRY_RUN="${DRY_RUN:-0}"

# Book defaults
BOOK_GAMES="${BOOK_GAMES:-50000}"
BOOK_MAX_PLIES="${BOOK_MAX_PLIES:-260}"
BOOK_BOOK_PLIES="${BOOK_BOOK_PLIES:-44}"
BOOK_EXPLORE_PLIES="${BOOK_EXPLORE_PLIES:-20}"
BOOK_EPSILON="${BOOK_EPSILON:-0.30}"
BOOK_PROGRESS_EVERY="${BOOK_PROGRESS_EVERY:-1000}"

# Tablebase defaults
TB_GAMES="${TB_GAMES:-120000}"
TB_MAX_PLIES="${TB_MAX_PLIES:-360}"
TB_MAX_PIECES="${TB_MAX_PIECES:-4}"
TB_SYNTHETIC="${TB_SYNTHETIC:-150000}"
TB_EXPLORE_PLIES="${TB_EXPLORE_PLIES:-36}"
TB_EPSILON="${TB_EPSILON:-0.40}"
TB_PROGRESS_EVERY="${TB_PROGRESS_EVERY:-2000}"
TB_PROBE_PROGRESS_EVERY="${TB_PROBE_PROGRESS_EVERY:-2000}"

# NNUE defaults
NNUE_SAMPLES="${NNUE_SAMPLES:-250000}"
NNUE_EPOCHS="${NNUE_EPOCHS:-20}"
NNUE_HIDDEN="${NNUE_HIDDEN:-256}"
NNUE_MAX_PLIES="${NNUE_MAX_PLIES:-220}"
NNUE_LR="${NNUE_LR:-0.0006}"
NNUE_EPSILON="${NNUE_EPSILON:-0.35}"
NNUE_PROGRESS_EVERY="${NNUE_PROGRESS_EVERY:-5000}"

BOOK_OUT="$ROOT/models/rust/book/centurion_book_${OUT_TAG}.bin"
TB_OUT="$ROOT/models/rust/tablebases/centurion_tb_4pc_${OUT_TAG}.bin"
NNUE_OUT="$ROOT/models/rust/nnue/centurion_nnue_${OUT_TAG}.bin"

run_step() {
  local label="$1"
  shift
  {
    echo
    echo "== $label =="
    echo "cmd: $*"
    if [[ "$DRY_RUN" == "1" ]]; then
      return 0
    fi
    "$@"
  } 2>&1 | tee -a "$LOG_FILE"
}

{
  echo "Centurion full training run"
  echo "run_id=$RUN_ID"
  echo "threads=$THREADS"
  echo "dry_run=$DRY_RUN"
  echo "book_progress_every=$BOOK_PROGRESS_EVERY"
  echo "tb_progress_every=$TB_PROGRESS_EVERY"
  echo "tb_probe_progress_every=$TB_PROBE_PROGRESS_EVERY"
  echo "nnue_progress_every=$NNUE_PROGRESS_EVERY"
  echo "book_out=$BOOK_OUT"
  echo "tb_out=$TB_OUT"
  echo "nnue_out=$NNUE_OUT"
  echo "log_file=$LOG_FILE"
} | tee -a "$LOG_FILE"

cd "$RUST_DIR"

run_step "Build release binaries" \
  cargo build --release -p ai --bins

run_step "Train opening book" \
  cargo run --release -p ai --bin centurion_bookgen -- \
    --games "$BOOK_GAMES" \
    --max-plies "$BOOK_MAX_PLIES" \
    --book-plies "$BOOK_BOOK_PLIES" \
    --explore-plies "$BOOK_EXPLORE_PLIES" \
    --epsilon "$BOOK_EPSILON" \
    --progress-every "$BOOK_PROGRESS_EVERY" \
    --threads "$THREADS" \
    --out "$BOOK_OUT"

run_step "Generate tablebase bootstrap" \
  cargo run --release -p ai --bin centurion_tbgen -- \
    --games "$TB_GAMES" \
    --max-plies "$TB_MAX_PLIES" \
    --max-pieces "$TB_MAX_PIECES" \
    --synthetic "$TB_SYNTHETIC" \
    --explore-plies "$TB_EXPLORE_PLIES" \
    --epsilon "$TB_EPSILON" \
    --progress-every "$TB_PROGRESS_EVERY" \
    --probe-progress-every "$TB_PROBE_PROGRESS_EVERY" \
    --threads "$THREADS" \
    --out "$TB_OUT"

run_step "Train NNUE" \
  cargo run --release -p ai --bin centurion_nnue_train -- \
    --samples "$NNUE_SAMPLES" \
    --epochs "$NNUE_EPOCHS" \
    --hidden "$NNUE_HIDDEN" \
    --max-plies "$NNUE_MAX_PLIES" \
    --lr "$NNUE_LR" \
    --epsilon "$NNUE_EPSILON" \
    --progress-every "$NNUE_PROGRESS_EVERY" \
    --threads "$THREADS" \
    --out "$NNUE_OUT"

echo
{
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "Dry run complete."
  else
    echo "Done."
  fi
  echo "Artifacts:"
  echo "  $BOOK_OUT"
  echo "  $TB_OUT"
  echo "  $NNUE_OUT"
  echo
  echo "Use these env vars to run with this training set:"
  echo "  export PARCAE_BOOK_PATH=$BOOK_OUT"
  echo "  export PARCAE_TB_PATH=$TB_OUT"
  echo "  export PARCAE_NNUE_PATH=$NNUE_OUT"
} | tee -a "$LOG_FILE"
