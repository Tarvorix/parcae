# ParcaeStrategy

Kowalski/Stanway Ludus Latrunculorum implementation with:

- Rust production server + shared Rust rules engine (`rust/server`, `rust/engine`)
- Rust multi-backend AI runtime (`heuristic`, `centurion`, `abaddon`) in `rust/ai`
- Python headless tooling for training/export/MCP (`src/parcaestrategy`)
- React/Vite TypeScript client in `web/`

## Rust production quickstart

```bash
./run_rust.sh
```

This starts the Rust API on `http://127.0.0.1:8000`.
`run_rust.sh` now defaults to strict Centurion startup (`book/tb/nnue` required) and fails fast if required artifacts are missing.

- `PARCAE_MODEL_PATH` controls Abaddon ONNX path.
- `PARCAE_AI_DEFAULT_BACKEND` controls default backend (`heuristic|centurion|abaddon`).
- `PARCAE_STOCKFISH_DEFAULT_TIME_MS` and `PARCAE_STOCKFISH_DEFAULT_HASH_MB` control Centurion defaults.
- `PARCAE_BOOK_PATH` and `PARCAE_TB_PATH` control Centurion book/tablebase artifacts.
- `PARCAE_NNUE_PATH` controls optional Centurion NNUE weights.
- `PARCAE_CENTURION_THREADS` controls Centurion root-parallel thread count.
- `PARCAE_CENTURION_STRICT=1` forces startup failure when required Centurion assets are missing.
- `PARCAE_CENTURION_REQUIRE_BOOK|PARCAE_CENTURION_REQUIRE_TB|PARCAE_CENTURION_REQUIRE_NNUE` select strict required assets.
- `PARCAE_REQUIRE_ONNX=1` fails startup if Abaddon ONNX cannot be loaded/exported.

## Python / API quickstart

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .[dev]        # installs FastAPI/uvicorn + pytest
parcaestrategy --serve --host 0.0.0.0 --port 8000
# fallback if your shell has not refreshed entry points:
# python -m parcaestrategy.cli --serve --host 0.0.0.0 --port 8000
```

Hit `http://localhost:8000/health` or create a match:

```bash
curl -X POST http://localhost:8000/match -d '{"mode":"pva"}' -H "Content-Type: application/json"
```

Match modes:

- `pva`: player vs built-in AI (white=player, black=ai)
- `pvg`: player vs MCP agent (white=player, black=agent)
- `gvg`: MCP agent vs MCP agent (white=agent, black=agent)
- `gva`: MCP agent vs built-in AI (white=agent, black=ai)
- `ava`: built-in AI vs built-in AI

AI capabilities endpoint:

```bash
curl "http://localhost:8000/ai/capabilities"
```

Per-side AI backend selection on match create:

```bash
curl -X POST http://localhost:8000/match \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "gva",
    "ai_profiles": {
      "black": {
        "backend": "centurion",
        "move_time_ms": 500,
        "max_depth": 32,
        "hash_mb": 128,
        "use_book": true,
        "use_tb": true,
        "skill": 12,
        "threads": 2
      }
    }
  }'
```

Legal move helper endpoint:

```bash
curl "http://localhost:8000/match/<id>/legal"               # all legal moves for side to play
curl "http://localhost:8000/match/<id>/legal?origin=A1"     # legal moves for one piece
```

Run tests:

```bash
pytest
```

## Web client quickstart

```bash
cd web
npm install   # or pnpm/yarn if preferred
npm run dev   # Vite dev server on 5173
```

The client expects the API at `http://localhost:8000` (override with `VITE_API_URL`).
Use the mode buttons to start `PvAI`, `PvAgent`, `Agent vs Agent`, `Agent vs AI`, or `AI vs AI`.

## Render deployment

This repo includes a Render Blueprint at `render.yaml` that creates:

- `parcae-api` (Rust Axum backend)
- `parcae-web` (static Vite frontend)

Deploy flow:

```bash
git push origin main
```

Then in Render:

1. New + -> Blueprint
2. Select this repo
3. Deploy `render.yaml`

Notes:

- `parcae-web` uses `VITE_API_URL` from the backend `RENDER_EXTERNAL_URL`.
- Backend health check is `/health`.
- Backend binds `HOST=0.0.0.0` and uses Render `PORT`.
- Blueprint pins strict Centurion startup and required assets so deployments fail fast if artifacts are missing/corrupt.
- To enable full Abaddon/Centurion artifacts, set:
  - `PARCAE_MODEL_PATH`
  - `PARCAE_BOOK_PATH`
  - `PARCAE_TB_PATH`
  - `PARCAE_NNUE_PATH`
  using paths that exist in the deployed environment.

Post-deploy smoke check:

```bash
./scripts/render_smoke.sh https://<your-api>.onrender.com https://<your-web>.onrender.com
```

## MCP quickstart (agent control)

Install MCP extras and run the MCP server over stdio:

```bash
python -m pip install -e .[mcp]
PARCAE_API_URL=http://127.0.0.1:8000 parcaestrategy-mcp
```

Expose this command in your MCP client config, then use tools:

- `create_match(mode="pvg")`
- `create_match(mode="gva", ai_profiles={"black": {"backend": "centurion"}})`
- `get_match(match_id)`
- `legal_moves(match_id, origin?)`
- `play_move(match_id, origin, target)`
- `step_ai(match_id)`

## AI (learning bot)

- The AI stack uses PyTorch (install with `pip install torch` or `python -m pip install -e .[ai]`).
- Quick self-play training and checkpoint write:

```bash
parcaestrategy --train-ai \
  --ai-games 64 \
  --ai-simulations 128 \
  --ai-epochs 2 \
  --ai-checkpoint checkpoints/parcae_latest.pt
```

- Useful knobs:
  - `--ai-batch-size` controls SGD batch size.
  - `--ai-max-plies` caps game length to avoid endless loops.
  - `--ai-temp-drop-plies` switches from exploratory to greedy play late in game.
  - `--ai-d-model --ai-layers --ai-heads --ai-ffn-dim --ai-dropout` configure Abaddon architecture.
  - `--ai-lr --ai-warmup-steps` tune optimization schedule.

- The training stack writes strict Abaddon transformer checkpoints (with metadata `arch=abaddon_transformer`).
- Python `AIAgent` now fail-hard errors when Abaddon is unavailable; no implicit heuristic fallback inside Abaddon paths.
- For Rust runtime, selecting backend `abaddon` with missing/invalid ONNX returns explicit API error, while `centurion` and `heuristic` are unaffected.
- Example model path:

```bash
export PARCAE_MODEL_PATH=checkpoints/parcae_latest.pt
parcaestrategy --serve
```

- `models/python/parcae_model.pth` may be from an older architecture and is incompatible with Abaddon checkpoints. Train a fresh checkpoint first, then export ONNX.

Abaddon evaluation workflow:

```bash
PYTHONPATH=src python3 scripts/eval_models.py \
  --challenger checkpoints/parcae_latest.pt \
  --baseline checkpoints/abaddon_baseline.pt \
  --games 40 \
  --simulations 128
```

Colab-oriented 12-hour profile:

```bash
cat scripts/abaddon_colab_12h.json
```

Centurion artifact generators:

```bash
cd rust
cargo run --release -p ai --bin centurion_bookgen -- --games 5000 --max-plies 220 --explore-plies 14 --epsilon 0.35 --threads 8 --out ../models/rust/book/centurion_book_v1.bin
cargo run --release -p ai --bin centurion_tbgen -- --games 20000 --max-plies 320 --max-pieces 4 --synthetic 40000 --explore-plies 28 --epsilon 0.45 --threads 8 --out ../models/rust/tablebases/centurion_tb_4pc_v1.bin
cargo run --release -p ai --bin centurion_nnue_train -- --samples 60000 --epochs 10 --hidden 128 --threads 8 --out ../models/rust/nnue/centurion_nnue_v1.bin
```

One-command full Centurion overnight run:

```bash
./train_centurion_full.sh
```

Optional tuning:

```bash
OUT_TAG=v2 THREADS=8 BOOK_GAMES=80000 TB_GAMES=180000 NNUE_SAMPLES=350000 ./train_centurion_full.sh
```

Progress output tuning (prints periodic status during long runs):

```bash
BOOK_PROGRESS_EVERY=1000 TB_PROGRESS_EVERY=2000 TB_PROBE_PROGRESS_EVERY=2000 NNUE_PROGRESS_EVERY=5000 ./train_centurion_full.sh
```

## Notes

- Rust API is the production authority; Python API remains useful for local/dev and training workflows.
- Centurion is a Stockfish-style backend (iterative deepening alpha-beta + TT + quiescence + move ordering).
- Rules engine is deterministic and shared; see `src/parcaestrategy/engine.py` for move/capture logic.
