"""Simple CLI entrypoint for parcaestrategy."""
import argparse

from . import __version__, greet


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="parcaestrategy")
    parser.add_argument("--version", action="store_true", help="Show version")
    parser.add_argument("--name", default="world", help="Name to greet")
    parser.add_argument("--serve", action="store_true", help="Run FastAPI server")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument(
        "--train-ai",
        action="store_true",
        help="Run a quick self-play training loop to produce a checkpoint.",
    )
    parser.add_argument(
        "--ai-games",
        type=int,
        default=4,
        help="Number of self-play games when training AI (default: 4).",
    )
    parser.add_argument(
        "--ai-checkpoint",
        default="checkpoints/parcae_latest.pt",
        help="Where to write the trained model checkpoint.",
    )
    parser.add_argument(
        "--ai-batch-size",
        type=int,
        default=32,
        help="Training batch size for AI self-play updates (default: 32).",
    )
    parser.add_argument(
        "--ai-epochs",
        type=int,
        default=2,
        help="Gradient epochs per self-play game (default: 2).",
    )
    parser.add_argument(
        "--ai-simulations",
        type=int,
        default=64,
        help="MCTS simulations per move during self-play (default: 64).",
    )
    parser.add_argument(
        "--ai-temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for early self-play plies (default: 1.0).",
    )
    parser.add_argument(
        "--ai-max-plies",
        type=int,
        default=256,
        help="Maximum plies per self-play game before score-based cutoff (default: 256).",
    )
    parser.add_argument(
        "--ai-temp-drop-plies",
        type=int,
        default=20,
        help="After this many plies, self-play switches to greedy move selection (default: 20).",
    )
    args = parser.parse_args(argv)

    if args.version:
        print(__version__)
        return 0

    if args.train_ai:
        try:
            from parcaestrategy.ai import train_self_play
        except ImportError:
            print("PyTorch is required for training. Install with `pip install torch`.")
            return 1
        path = train_self_play(
            games=args.ai_games,
            batch_size=args.ai_batch_size,
            epochs=args.ai_epochs,
            simulations=args.ai_simulations,
            self_play_temperature=args.ai_temperature,
            self_play_max_plies=args.ai_max_plies,
            temperature_drop_plies=args.ai_temp_drop_plies,
            out_path=args.ai_checkpoint,
        )
        print(f"Checkpoint written to {path}")
        return 0

    if args.serve:
        try:
            from uvicorn import run
            from parcaestrategy.api import create_app
        except ImportError:
            print("uvicorn and fastapi are required to serve the API. Install extras.")
            return 1

        run(create_app(), host=args.host, port=args.port, reload=False)
        return 0

    print(greet(args.name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
