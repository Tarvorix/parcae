from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import List

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:  # pragma: no cover - torch optional
    torch = None
    nn = None
    optim = None

from .agent import AIAgent, MOVE_TO_INDEX
from .model import AbaddonConfig, build_checkpoint_payload, create_abaddon_model
from .selfplay import Experience, self_play_game


@dataclass
class ReplayBuffer:
    capacity: int
    items: List[Experience] = field(default_factory=list)

    def add(self, batch: List[Experience]) -> None:
        self.items.extend(batch)
        if len(self.items) > self.capacity:
            self.items = self.items[-self.capacity :]

    def sample(self, batch_size: int) -> List[Experience]:
        import random

        return random.sample(self.items, min(batch_size, len(self.items)))


def train_self_play(
    games: int = 4,
    batch_size: int = 32,
    epochs: int = 2,
    device: str = "cpu",
    simulations: int = 64,
    self_play_temperature: float = 1.0,
    self_play_max_plies: int = 256,
    temperature_drop_plies: int = 20,
    out_path: str = "checkpoints/parcae_latest.pt",
    d_model: int = 128,
    layers: int = 8,
    heads: int = 4,
    ffn_dim: int = 384,
    dropout: float = 0.1,
    lr: float = 1e-3,
    warmup_steps: int = 50,
    progress_every: int = 10,
    checkpoint_every: int = 0,
) -> str:
    """Run self-play + training loop and write a checkpoint."""
    if torch is None:
        raise ImportError("PyTorch required for training. Install with `pip install torch`.") from None

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    buffer = ReplayBuffer(capacity=10_000)
    config = AbaddonConfig(
        d_model=d_model,
        layers=layers,
        heads=heads,
        ffn_dim=ffn_dim,
        dropout=dropout,
    )
    model = create_abaddon_model(move_space=len(MOVE_TO_INDEX), config=config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_value = nn.MSELoss()
    expected_updates = max(1, games * epochs)

    def lr_schedule(step: int) -> float:
        warmup = max(0, warmup_steps)
        if warmup > 0 and step < warmup:
            return float(step + 1) / float(warmup)
        if expected_updates <= warmup:
            return 1.0
        progress = float(step - warmup) / float(expected_updates - warmup)
        progress = min(1.0, max(0.0, progress))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule)

    agent = AIAgent(model_path=None, device=device, simulations=simulations)
    # Replace the randomly initialized model inside agent
    agent.model = model
    agent.available = True
    agent.load_error = None

    started = time.time()
    print(
        "Training start:",
        {
            "games": games,
            "batch_size": batch_size,
            "epochs": epochs,
            "device": device,
            "simulations": simulations,
            "progress_every": progress_every,
            "checkpoint_every": checkpoint_every,
        },
        flush=True,
    )

    for game_idx in range(games):
        model.eval()
        experiences, game_result = self_play_game(
            agent,
            temperature=self_play_temperature,
            max_plies=self_play_max_plies,
            temperature_drop_plies=temperature_drop_plies,
        )
        buffer.add(experiences)

        model.train()
        for _ in range(epochs):
            batch = buffer.sample(batch_size)
            if not batch:
                continue
            states = torch.stack([exp.tensor for exp in batch]).to(device)
            target_pi = torch.stack([exp.pi for exp in batch]).to(device)
            target_z = torch.tensor([exp.z for exp in batch], dtype=torch.float32, device=device)

            optimizer.zero_grad()
            policy_logits, values = model(states)
            log_probs = torch.log_softmax(policy_logits, dim=1)
            lp = -(target_pi * log_probs).sum(dim=1).mean()
            lv = loss_value(values.view(-1), target_z)
            loss = lp + lv
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

        should_log = progress_every > 0 and (
            (game_idx + 1) % progress_every == 0 or (game_idx + 1) == games
        )
        if should_log:
            elapsed = time.time() - started
            avg_per_game = elapsed / float(game_idx + 1)
            eta = avg_per_game * float(games - (game_idx + 1))
            winner_text = game_result.winner.value if game_result.winner is not None else "draw"
            print(
                f"[train] game {game_idx + 1}/{games} "
                f"winner={winner_text} "
                f"plies={game_result.plies} "
                f"white_caps={game_result.white_captures} "
                f"black_caps={game_result.black_captures} "
                f"end={game_result.end_reason} "
                f"experiences={len(experiences)} "
                f"buffer={len(buffer.items)} "
                f"elapsed={elapsed:.1f}s "
                f"eta={eta:.1f}s",
                flush=True,
            )

        should_checkpoint = checkpoint_every > 0 and (game_idx + 1) % checkpoint_every == 0
        if should_checkpoint and (game_idx + 1) < games:
            stem, ext = os.path.splitext(out_path)
            interval_path = f"{stem}_g{game_idx + 1:05d}{ext or '.pt'}"
            torch.save(build_checkpoint_payload(model, config), interval_path)
            print(f"[train] checkpoint saved: {interval_path}", flush=True)

    torch.save(build_checkpoint_payload(model, config), out_path)
    total_elapsed = time.time() - started
    print(f"Training complete in {total_elapsed:.1f}s. Checkpoint: {out_path}", flush=True)
    return out_path
