from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from parcaestrategy.ai.agent import MOVE_TO_INDEX
from parcaestrategy.ai.model import (
    AbaddonConfig,
    build_checkpoint_payload,
    create_abaddon_model,
    load_model_from_checkpoint,
)
from parcaestrategy.engine import BOARD_HEIGHT, BOARD_WIDTH


def test_abaddon_forward_shapes() -> None:
    model = create_abaddon_model(move_space=len(MOVE_TO_INDEX), config=AbaddonConfig())
    x = torch.zeros((2, 5, BOARD_HEIGHT, BOARD_WIDTH), dtype=torch.float32)
    policy, value = model(x)
    assert list(policy.shape) == [2, len(MOVE_TO_INDEX)]
    assert list(value.shape) == [2]
    assert torch.all(value <= 1.0)
    assert torch.all(value >= -1.0)


def test_abaddon_checkpoint_round_trip(tmp_path: Path) -> None:
    cfg = AbaddonConfig(d_model=64, layers=2, heads=4, ffn_dim=128, dropout=0.0)
    model = create_abaddon_model(move_space=len(MOVE_TO_INDEX), config=cfg)
    path = tmp_path / "abaddon.pt"
    torch.save(build_checkpoint_payload(model, cfg), path)

    loaded, meta, error = load_model_from_checkpoint(
        checkpoint_path=str(path),
        move_space=len(MOVE_TO_INDEX),
        device="cpu",
    )
    assert error is None
    assert loaded is not None
    assert isinstance(meta, dict)
    assert meta["arch"] == "abaddon_transformer"
