from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

try:
    import torch
except ImportError:  # pragma: no cover - torch optional
    torch = None

from parcaestrategy.engine import Color, GameState, apply_move, initial_state

from .agent import AIAgent, MOVE_TO_INDEX, list_legal_moves
from .model import state_to_tensor


@dataclass
class Experience:
    tensor: "torch.Tensor"  # type: ignore[name-defined]
    pi: "torch.Tensor"  # probability over ALL_MOVES
    z: float  # outcome from perspective of state.turn


def self_play_game(
    agent: AIAgent,
    temperature: float = 1.0,
    max_plies: int = 256,
    temperature_drop_plies: int = 20,
) -> Tuple[List[Experience], Optional[Color]]:
    """Play one self-play game, returning experiences and winner."""
    if torch is None:
        raise ImportError("PyTorch required for self-play. Install with `pip install torch`.") from None

    state = initial_state()
    experiences: List[Experience] = []
    # Track which player perspective each state was from
    players: List[Color] = []
    plies = 0

    while state.winner is None and plies < max_plies:
        legal = list_legal_moves(state)
        if not legal:
            state.winner = state.turn.opponent()
            state.summary = "No legal moves."
            break

        current_temperature = temperature if plies < temperature_drop_plies else 0.0
        move, pi = agent.select_move_with_policy(
            state,
            temperature=current_temperature,
            add_root_noise=True,
        )
        if move is None:
            state.winner = state.turn.opponent()
            state.summary = "AI resigned (no moves)."
            break

        if pi is None:
            pi = torch.zeros(len(MOVE_TO_INDEX), dtype=torch.float32)
            pi[MOVE_TO_INDEX[(move.origin, move.target)]] = 1.0

        experiences.append(Experience(tensor=state_to_tensor(state), pi=pi, z=0.0))
        players.append(state.turn)
        state = apply_move(state, move)
        plies += 1

    winner: Optional[Color] = state.winner
    if winner is None and plies >= max_plies:
        white_score = state.captures[Color.WHITE]
        black_score = state.captures[Color.BLACK]
        if white_score > black_score:
            winner = Color.WHITE
        elif black_score > white_score:
            winner = Color.BLACK

    for exp, player in zip(experiences, players):
        if winner is None:
            exp.z = 0.0
        else:
            exp.z = 1.0 if winner == player else -1.0

    return experiences, winner
