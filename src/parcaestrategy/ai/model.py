from __future__ import annotations

from typing import Mapping, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - torch optional in CI
    torch = None
    nn = None
    F = None

from parcaestrategy.engine import BOARD_HEIGHT, BOARD_WIDTH, Color, GameState, PieceType


def _ensure_torch():
    if torch is None:
        raise ImportError("PyTorch is not installed. Install with `pip install torch`.")


def _unwrap_checkpoint_state(raw_state: object) -> Mapping[str, "torch.Tensor"]:
    if isinstance(raw_state, Mapping) and "model" in raw_state and isinstance(
        raw_state["model"], Mapping
    ):
        return raw_state["model"]
    if isinstance(raw_state, Mapping):
        return raw_state
    raise TypeError("Unsupported checkpoint format: expected a mapping/state_dict.")


def load_checkpoint_into_model(
    model: "nn.Module", checkpoint_path: str, device: str = "cpu"
) -> Tuple[bool, Optional[str]]:
    """Load model weights from checkpoint path.

    Returns a (success, error_message) tuple so callers can gracefully fall back
    when checkpoints are from an incompatible architecture.
    """
    _ensure_torch()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = _unwrap_checkpoint_state(checkpoint)
    try:
        model.load_state_dict(state_dict)
    except Exception as exc:
        return False, str(exc)
    return True, None


def state_to_tensor(state: GameState, device: str = "cpu") -> "torch.Tensor":
    """Encode game state into tensor (C,H,W). Channels:
    0: white soldiers
    1: black soldiers
    2: white dux
    3: black dux
    4: to-play (all ones if white to move)
    """
    _ensure_torch()
    planes = torch.zeros((5, BOARD_HEIGHT, BOARD_WIDTH), dtype=torch.float32, device=device)
    for (file_idx, rank_idx), piece in state.board.items():
        if piece.color is Color.WHITE and piece.kind is PieceType.SOLDIER:
            planes[0, rank_idx, file_idx] = 1.0
        elif piece.color is Color.BLACK and piece.kind is PieceType.SOLDIER:
            planes[1, rank_idx, file_idx] = 1.0
        elif piece.color is Color.WHITE and piece.kind is PieceType.DUX:
            planes[2, rank_idx, file_idx] = 1.0
        elif piece.color is Color.BLACK and piece.kind is PieceType.DUX:
            planes[3, rank_idx, file_idx] = 1.0
    if state.turn is Color.WHITE:
        planes[4, :, :] = 1.0
    return planes


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":  # type: ignore[name-defined]
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class ParcaeNet(nn.Module):
    """Small policy/value network for 12x8 board."""

    def __init__(self, channels: int, blocks: int, move_space: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(5, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.Sequential(*(ResidualBlock(channels) for _ in range(blocks)))
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2 * BOARD_HEIGHT * BOARD_WIDTH, move_space),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(BOARD_HEIGHT * BOARD_WIDTH, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, 1),
            nn.Tanh(),
        )

    def forward(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:  # type: ignore[name-defined]
        feat = self.stem(x)
        feat = self.res_blocks(feat)
        policy = self.policy_head(feat)
        value = self.value_head(feat)
        return policy, value.squeeze(-1)
