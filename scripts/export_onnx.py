#!/usr/bin/env python
"""
Export the Parcae PyTorch checkpoint to ONNX for Rust inference.

Usage:
    python scripts/export_onnx.py --checkpoint checkpoints/parcae_latest.pt --out models/rust/parcae_model.onnx
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
from parcaestrategy.ai.model import ABADDON_ARCH, load_model_from_checkpoint
from parcaestrategy.engine import BOARD_HEIGHT, BOARD_WIDTH


def _verify_torch_output_shapes(model: "torch.nn.Module") -> None:
    with torch.no_grad():
        sample = torch.zeros((1, 5, BOARD_HEIGHT, BOARD_WIDTH), dtype=torch.float32)
        policy, value = model(sample)
    if list(policy.shape) != [1, len(MOVE_TO_INDEX)]:
        raise RuntimeError(
            f"Invalid PyTorch policy output shape {list(policy.shape)}; "
            f"expected [1, {len(MOVE_TO_INDEX)}]."
        )
    if list(value.shape) != [1]:
        raise RuntimeError(f"Invalid PyTorch value output shape {list(value.shape)}; expected [1].")


def _verify_export_shapes(onnx_path: str) -> None:
    try:
        import onnxruntime as ort
    except ImportError:  # pragma: no cover - optional runtime dep
        print("Warning: onnxruntime not installed; skipping ONNX shape validation.")
        return

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    dummy = torch.zeros((1, 5, BOARD_HEIGHT, BOARD_WIDTH), dtype=torch.float32).numpy()
    outputs = session.run(["policy", "value"], {"input": dummy})
    policy, value = outputs[0], outputs[1]
    if list(policy.shape) != [1, len(MOVE_TO_INDEX)]:
        raise RuntimeError(
            f"Invalid ONNX policy output shape {policy.shape}; expected (1, {len(MOVE_TO_INDEX)})."
        )
    if list(value.shape) not in ([1], [1, 1]):
        raise RuntimeError(f"Invalid ONNX value output shape {value.shape}; expected (1,) or (1,1).")


def _verify_parity(
    model: "torch.nn.Module",
    onnx_path: str,
    samples: int = 3,
    atol: float = 1e-4,
    rtol: float = 1e-3,
) -> None:
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - optional runtime dep
        raise RuntimeError(
            "onnxruntime is required for parity checks. Install with `pip install onnxruntime`."
        ) from exc

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    with torch.no_grad():
        for _ in range(samples):
            sample = torch.randn((1, 5, BOARD_HEIGHT, BOARD_WIDTH), dtype=torch.float32)
            policy_pt, value_pt = model(sample)
            policy_onnx, value_onnx = session.run(["policy", "value"], {"input": sample.numpy()})
            policy_ref = policy_pt.cpu()
            value_ref = value_pt.view(-1).cpu()
            policy_cmp = torch.from_numpy(policy_onnx)
            value_cmp = torch.from_numpy(value_onnx).view(-1)
            if not torch.allclose(policy_ref, policy_cmp, atol=atol, rtol=rtol):
                raise RuntimeError("ONNX parity check failed for policy logits.")
            if not torch.allclose(value_ref, value_cmp, atol=atol, rtol=rtol):
                raise RuntimeError("ONNX parity check failed for value output.")


def export(checkpoint: str, out_path: str, verify_parity: bool = False) -> None:
    device = torch.device("cpu")
    model, meta, error = load_model_from_checkpoint(
        checkpoint_path=checkpoint,
        move_space=len(MOVE_TO_INDEX),
        device=str(device),
    )
    if model is None:
        raise RuntimeError(
            "Checkpoint is incompatible with Abaddon transformer architecture. "
            f"Details: {error}"
        ) from None
    if not isinstance(meta, dict) or meta.get("arch") != ABADDON_ARCH:
        raise RuntimeError(
            f"Checkpoint metadata missing or invalid; expected arch `{ABADDON_ARCH}`."
        )
    model.eval()
    _verify_torch_output_shapes(model)

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
    _verify_export_shapes(out_path)
    if verify_parity:
        _verify_parity(model, out_path)
    print(f"Exported ONNX to {out_path}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/parcae_latest.pt")
    parser.add_argument("--out", default="models/rust/parcae_model.onnx")
    parser.add_argument(
        "--verify-parity",
        action="store_true",
        help="Run PyTorch-vs-ONNX numeric parity checks after export.",
    )
    args = parser.parse_args(argv)
    export(args.checkpoint, args.out, verify_parity=args.verify_parity)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
