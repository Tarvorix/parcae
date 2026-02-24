#!/usr/bin/env python
"""
Export the Parcae PyTorch checkpoint to ONNX for Rust inference.

Usage:
    python scripts/export_onnx.py --checkpoint models/python/parcae_model.pth --out models/rust/parcae_model.onnx
"""

import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from parcaestrategy.ai.agent import MOVE_TO_INDEX
from parcaestrategy.ai.model import ParcaeNet, load_checkpoint_into_model
from parcaestrategy.engine import BOARD_HEIGHT, BOARD_WIDTH


def export(checkpoint: str, out_path: str) -> None:
    device = torch.device("cpu")
    model = ParcaeNet(channels=64, blocks=4, move_space=len(MOVE_TO_INDEX)).to(device)
    loaded, error = load_checkpoint_into_model(model, checkpoint, device=str(device))
    if not loaded:
        raise RuntimeError(
            "Checkpoint is incompatible with current ParcaeNet architecture. "
            "Train a new checkpoint with the current code before ONNX export. "
            f"Details: {error}"
        ) from None
    model.eval()

    dummy = torch.zeros((1, 5, BOARD_HEIGHT, BOARD_WIDTH), dtype=torch.float32, device=device)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    export_kwargs = {
        "input_names": ["input"],
        "output_names": ["policy", "value"],
        "opset_version": 18,
        "dynamic_axes": None,
    }
    try:
        # Prefer legacy exporter path for compatibility with Python 3.14 toolchains.
        torch.onnx.export(
            model,
            dummy,
            out_path,
            dynamo=False,
            **export_kwargs,
        )
    except TypeError:
        # Older PyTorch versions do not support the `dynamo` argument.
        torch.onnx.export(model, dummy, out_path, **export_kwargs)
    print(f"Exported ONNX to {out_path}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/python/parcae_model.pth")
    parser.add_argument("--out", default="models/rust/parcae_model.onnx")
    args = parser.parse_args(argv)
    export(args.checkpoint, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
