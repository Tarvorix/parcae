# ParcaeStrategy

Kowalski/Stanway Ludus Latrunculorum implementation scaffold with:

- Python rules engine + FastAPI backend (`parcaestrategy.engine`, `parcaestrategy.api`)
- PyTorch-ready AI stub hooks
- React/Vite TypeScript client in `web/`

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

## MCP quickstart (agent control)

Install MCP extras and run the MCP server over stdio:

```bash
python -m pip install -e .[mcp]
PARCAE_API_URL=http://127.0.0.1:8000 parcaestrategy-mcp
```

Expose this command in your MCP client config, then use tools:

- `create_match(mode="pvg")`
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

- The API will load `PARCAE_MODEL_PATH` if set and compatible; otherwise it falls back to a heuristic bot. Example:

```bash
export PARCAE_MODEL_PATH=checkpoints/parcae_latest.pt
parcaestrategy --serve
```

- `models/python/parcae_model.pth` may be from an older architecture and can be incompatible with the current code. Train a fresh checkpoint first, then export ONNX if needed.

## Notes

- AI agent defaults to the learning model if PyTorch + checkpoint are available; otherwise it falls back to a capture-aware heuristic.
- Rules engine is deterministic and shared; see `src/parcaestrategy/engine.py` for move/capture logic.
