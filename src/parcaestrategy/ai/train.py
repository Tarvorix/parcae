from __future__ import annotations

import os
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
from .model import ParcaeNet
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
) -> str:
    """Run self-play + training loop and write a checkpoint."""
    if torch is None:
        raise ImportError("PyTorch required for training. Install with `pip install torch`.") from None

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    buffer = ReplayBuffer(capacity=10_000)
    model = ParcaeNet(channels=64, blocks=4, move_space=len(MOVE_TO_INDEX)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_value = nn.MSELoss()

    agent = AIAgent(model_path=None, device=device, simulations=simulations)
    # Replace the randomly initialized model inside agent
    agent.model = model
    agent.available = True

    for _ in range(games):
        model.eval()
        experiences, _winner = self_play_game(
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
            optimizer.step()

    torch.save({"model": model.state_dict()}, out_path)
    return out_path
