from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - torch optional in CI
    torch = None
    nn = None
    F = None

from parcaestrategy.engine import BOARD_HEIGHT, BOARD_WIDTH, Color, GameState, PieceType

ABADDON_ARCH = "abaddon_transformer"
CHECKPOINT_VERSION = 1


def _ensure_torch() -> None:
    if torch is None or nn is None or F is None:
        raise ImportError("PyTorch is not installed. Install with `pip install torch`.")


@dataclass(frozen=True)
class AbaddonConfig:
    d_model: int = 128
    layers: int = 8
    heads: int = 4
    ffn_dim: int = 384
    dropout: float = 0.1

    @classmethod
    def from_mapping(cls, data: Optional[Mapping[str, object]]) -> "AbaddonConfig":
        if data is None:
            return cls()
        return cls(
            d_model=int(data.get("d_model", cls.d_model)),
            layers=int(data.get("layers", cls.layers)),
            heads=int(data.get("heads", cls.heads)),
            ffn_dim=int(data.get("ffn_dim", cls.ffn_dim)),
            dropout=float(data.get("dropout", cls.dropout)),
        )

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


def _build_move_token_indices() -> Tuple[List[int], List[int]]:
    directions: Tuple[Tuple[int, int], ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))
    origin_indices: List[int] = []
    target_indices: List[int] = []
    for file_idx in range(BOARD_WIDTH):
        for rank_idx in range(BOARD_HEIGHT):
            origin_flat = rank_idx * BOARD_WIDTH + file_idx
            for dx, dy in directions:
                step = 1
                while True:
                    tx = file_idx + dx * step
                    ty = rank_idx + dy * step
                    if not (0 <= tx < BOARD_WIDTH and 0 <= ty < BOARD_HEIGHT):
                        break
                    target_flat = ty * BOARD_WIDTH + tx
                    origin_indices.append(origin_flat)
                    target_indices.append(target_flat)
                    step += 1
    return origin_indices, target_indices


def _extract_checkpoint(raw_state: object) -> Tuple[Mapping[str, "torch.Tensor"], Mapping[str, object]]:
    if not isinstance(raw_state, Mapping):
        raise TypeError("Unsupported checkpoint format: expected a mapping.")
    model_state = raw_state.get("model")
    if not isinstance(model_state, Mapping):
        raise TypeError("Checkpoint missing `model` state_dict.")
    meta = raw_state.get("meta")
    if not isinstance(meta, Mapping):
        raise TypeError("Checkpoint missing `meta` metadata block.")
    arch = meta.get("arch")
    if arch != ABADDON_ARCH:
        raise TypeError(
            f"Checkpoint arch mismatch: expected `{ABADDON_ARCH}`, got `{arch}`."
        )
    return model_state, meta


def load_checkpoint_into_model(
    model: "nn.Module", checkpoint_path: str, device: str = "cpu"
) -> Tuple[bool, Optional[str], Optional[Mapping[str, object]]]:
    _ensure_torch()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    try:
        state_dict, meta = _extract_checkpoint(checkpoint)
        model.load_state_dict(state_dict)
    except Exception as exc:
        return False, str(exc), None
    return True, None, meta


def load_model_from_checkpoint(
    checkpoint_path: str,
    move_space: int,
    device: str = "cpu",
) -> Tuple[Optional["AbaddonTransformer"], Optional[Mapping[str, object]], Optional[str]]:
    _ensure_torch()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    try:
        state_dict, meta = _extract_checkpoint(checkpoint)
        config = AbaddonConfig.from_mapping(meta.get("spec"))  # type: ignore[arg-type]
        model = AbaddonTransformer(config=config, move_space=move_space).to(device)
        model.load_state_dict(state_dict)
        model.eval()
    except Exception as exc:
        return None, None, str(exc)
    return model, meta, None


def build_checkpoint_payload(
    model: "nn.Module",
    config: AbaddonConfig,
) -> Dict[str, object]:
    return {
        "model": model.state_dict(),
        "meta": {
            "version": CHECKPOINT_VERSION,
            "arch": ABADDON_ARCH,
            "spec": config.as_dict(),
        },
    }


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


class TransformerBlock(nn.Module):
    def __init__(self, config: AbaddonConfig, token_count: int) -> None:
        super().__init__()
        if config.d_model % config.heads != 0:
            raise ValueError("d_model must be divisible by heads.")
        self.d_model = config.d_model
        self.heads = config.heads
        self.head_dim = config.d_model // config.heads
        self.scale = self.head_dim**-0.5

        self.norm1 = nn.LayerNorm(config.d_model)
        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.norm2 = nn.LayerNorm(config.d_model)
        self.ffn_gate = nn.Linear(config.d_model, config.ffn_dim)
        self.ffn_value = nn.Linear(config.d_model, config.ffn_dim)
        self.ffn_out = nn.Linear(config.ffn_dim, config.d_model)

        self.row_bias = nn.Embedding(2 * BOARD_HEIGHT - 1, config.heads)
        self.col_bias = nn.Embedding(2 * BOARD_WIDTH - 1, config.heads)
        row_delta, col_delta = self._build_delta_tables(token_count)
        self.register_buffer("row_delta", row_delta, persistent=False)
        self.register_buffer("col_delta", col_delta, persistent=False)

    def _build_delta_tables(self, token_count: int) -> Tuple["torch.Tensor", "torch.Tensor"]:
        rows: List[int] = []
        cols: List[int] = []
        for idx in range(token_count):
            rows.append(idx // BOARD_WIDTH)
            cols.append(idx % BOARD_WIDTH)
        row_delta = []
        col_delta = []
        for r_i, c_i in zip(rows, cols):
            row_row = []
            col_row = []
            for r_j, c_j in zip(rows, cols):
                row_row.append(r_i - r_j + (BOARD_HEIGHT - 1))
                col_row.append(c_i - c_j + (BOARD_WIDTH - 1))
            row_delta.append(row_row)
            col_delta.append(col_row)
        return torch.tensor(row_delta, dtype=torch.long), torch.tensor(col_delta, dtype=torch.long)

    def _relative_bias(self) -> "torch.Tensor":
        row_bias = self.row_bias(self.row_delta)
        col_bias = self.col_bias(self.col_delta)
        return (row_bias + col_bias).permute(2, 0, 1).contiguous()

    def _attention(self, x: "torch.Tensor") -> "torch.Tensor":
        batch, tokens, _dim = x.shape
        q = self.q_proj(x).view(batch, tokens, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, tokens, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, tokens, self.heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        scores = scores + self._relative_bias().unsqueeze(0)
        probs = torch.softmax(scores, dim=-1)
        probs = self.attn_dropout(probs)
        attended = torch.matmul(probs, v)
        attended = attended.transpose(1, 2).contiguous().view(batch, tokens, self.d_model)
        return self.out_proj(attended)

    def _mlp(self, x: "torch.Tensor") -> "torch.Tensor":
        gated = F.gelu(self.ffn_gate(x))
        value = self.ffn_value(x)
        return self.ffn_out(gated * value)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        x = x + self.resid_dropout(self._attention(self.norm1(x)))
        x = x + self.resid_dropout(self._mlp(self.norm2(x)))
        return x


class AbaddonTransformer(nn.Module):
    """Transformer-only policy/value network for the 12x8 board."""

    def __init__(self, config: AbaddonConfig, move_space: int) -> None:
        super().__init__()
        _ensure_torch()
        self.config = config
        self.move_space = move_space

        token_count = BOARD_HEIGHT * BOARD_WIDTH
        origin_indices, target_indices = _build_move_token_indices()
        if len(origin_indices) != move_space:
            raise ValueError(
                f"Move-space mismatch: expected {len(origin_indices)} entries, got {move_space}."
            )

        token_rows = []
        token_cols = []
        for idx in range(token_count):
            token_rows.append(idx // BOARD_WIDTH)
            token_cols.append(idx % BOARD_WIDTH)

        self.register_buffer(
            "move_origin_index",
            torch.tensor(origin_indices, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "move_target_index",
            torch.tensor(target_indices, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer("token_row", torch.tensor(token_rows, dtype=torch.long), persistent=False)
        self.register_buffer("token_col", torch.tensor(token_cols, dtype=torch.long), persistent=False)

        self.input_proj = nn.Conv2d(5, config.d_model, kernel_size=1)
        self.row_embedding = nn.Embedding(BOARD_HEIGHT, config.d_model)
        self.col_embedding = nn.Embedding(BOARD_WIDTH, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            TransformerBlock(config=config, token_count=token_count) for _ in range(config.layers)
        )
        self.final_norm = nn.LayerNorm(config.d_model)

        self.policy_head = nn.Sequential(
            nn.LayerNorm(config.d_model * 4),
            nn.Linear(config.d_model * 4, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, 1),
        )

        self.value_head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, 1),
            nn.Tanh(),
        )

    def forward(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
        tokens = self.input_proj(x).flatten(2).transpose(1, 2)
        pos = self.row_embedding(self.token_row) + self.col_embedding(self.token_col)
        tokens = self.dropout(tokens + pos.unsqueeze(0))
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.final_norm(tokens)

        origin = tokens.index_select(1, self.move_origin_index)
        target = tokens.index_select(1, self.move_target_index)
        pair_features = torch.cat(
            [origin, target, origin - target, origin * target],
            dim=-1,
        )
        policy = self.policy_head(pair_features).squeeze(-1)

        pooled = tokens.mean(dim=1)
        value = self.value_head(pooled).squeeze(-1)
        return policy, value


def create_abaddon_model(
    move_space: int,
    config: Optional[AbaddonConfig] = None,
) -> "AbaddonTransformer":
    return AbaddonTransformer(config=config or AbaddonConfig(), move_space=move_space)
