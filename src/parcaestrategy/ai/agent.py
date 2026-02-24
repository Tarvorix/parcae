from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - torch optional in CI
    torch = None
    F = None

from parcaestrategy.engine import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    Color,
    GameState,
    Move,
    apply_move,
    legal_moves,
)

from .model import ParcaeNet, load_checkpoint_into_model, state_to_tensor

# Precompute all geometric moves on the empty board for policy indexing.
DIRECTIONS: Tuple[Tuple[int, int], ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))
ALL_MOVES: List[Move] = []
MOVE_TO_INDEX: Dict[Tuple[Tuple[int, int], Tuple[int, int]], int] = {}
for file_idx in range(BOARD_WIDTH):
    for rank_idx in range(BOARD_HEIGHT):
        origin = (file_idx, rank_idx)
        for dx, dy in DIRECTIONS:
            step = 1
            while True:
                target = (file_idx + dx * step, rank_idx + dy * step)
                if not (0 <= target[0] < BOARD_WIDTH and 0 <= target[1] < BOARD_HEIGHT):
                    break
                mv = Move(origin=origin, target=target)
                MOVE_TO_INDEX[(origin, target)] = len(ALL_MOVES)
                ALL_MOVES.append(mv)
                step += 1


def list_legal_moves(state: GameState) -> List[Move]:
    moves: List[Move] = []
    for coord, piece in state.board.items():
        if piece.color is state.turn:
            for target in legal_moves(state, coord):
                moves.append(Move(origin=coord, target=target))
    return moves


def _heuristic_move(state: GameState) -> Optional[Move]:
    """Fallback: prefer winning moves, then capturing, else first legal."""
    legal = list_legal_moves(state)
    if not legal:
        return None
    best_move = legal[0]
    best_score = -1_000_000
    for mv in legal:
        next_state = apply_move(state, mv)
        score = 0
        if next_state.winner == state.turn:
            score = 1_000_000
        gained = next_state.captures[state.turn] - state.captures[state.turn]
        score += gained * 100
        score += len(legal_moves(next_state, mv.target))
        if score > best_score:
            best_score = score
            best_move = mv
    return best_move


@dataclass
class Node:
    state: GameState
    to_play: Color
    prior: float = 0.0
    visits: int = 0
    value_sum: float = 0.0
    children: Dict[int, "Node"] = None  # key is move index

    def q_value(self) -> float:
        return 0.0 if self.visits == 0 else self.value_sum / self.visits

    def expanded(self) -> bool:
        return bool(self.children)


class AIAgent:
    """MCTS + policy/value net. Falls back to heuristic if torch/model missing."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        simulations: int = 64,
        c_puct: float = 1.25,
    ) -> None:
        self.device = device
        self.simulations = simulations
        self.c_puct = c_puct
        self.model_path = model_path or os.environ.get("PARCAE_MODEL_PATH")
        self.model: Optional[ParcaeNet] = None
        self.available = False
        self.load_error: Optional[str] = None
        if torch is not None and self.model_path:
            self._load_model()

    def _load_model(self) -> None:
        if self.model_path is None:
            self.available = False
            self.load_error = "PARCAE_MODEL_PATH is not set."
            return
        if not os.path.exists(self.model_path):
            self.available = False
            self.load_error = f"Model path does not exist: {self.model_path}"
            return

        self.model = ParcaeNet(channels=64, blocks=4, move_space=len(ALL_MOVES)).to(self.device)
        loaded, error = load_checkpoint_into_model(self.model, self.model_path, device=self.device)
        if not loaded:
            self.available = False
            self.model = None
            self.load_error = error
            short_error = error.splitlines()[0] if error else "unknown checkpoint error"
            warnings.warn(
                f"Failed to load checkpoint at {self.model_path}; using heuristic fallback. {short_error}",
                RuntimeWarning,
            )
            return
        self.model.eval()
        self.available = True
        self.load_error = None

    def select_move(self, state: GameState) -> Optional[Move]:
        if not self.available:
            return _heuristic_move(state)
        move, _ = self._mcts_action(state, temperature=0.0, add_root_noise=False)
        return move

    def select_move_with_policy(
        self,
        state: GameState,
        temperature: float = 1.0,
        add_root_noise: bool = True,
    ) -> Tuple[Optional[Move], Optional["torch.Tensor"]]:  # type: ignore[name-defined]
        """Return selected move and full policy target over ALL_MOVES.

        This is intended for self-play training, where the policy target should
        come from MCTS visit counts (or one-hot heuristic fallback).
        """
        if not self.available:
            move = _heuristic_move(state)
            if move is None or torch is None:
                return move, None
            pi = torch.zeros(len(ALL_MOVES), dtype=torch.float32)
            pi[MOVE_TO_INDEX[(move.origin, move.target)]] = 1.0
            return move, pi
        return self._mcts_action(
            state,
            temperature=max(0.0, temperature),
            add_root_noise=add_root_noise,
        )

    # --- MCTS internals ---
    def _mcts_action(
        self,
        state: GameState,
        temperature: float = 0.0,
        add_root_noise: bool = False,
    ) -> Tuple[Optional[Move], Optional["torch.Tensor"]]:  # type: ignore[name-defined]
        root = Node(state=state, to_play=state.turn, prior=1.0, children={})
        legal_root = list_legal_moves(state)
        if not legal_root:
            return None, None
        policy, _value = self._evaluate(state)
        self._expand(root, policy, legal_root)
        if add_root_noise:
            self._add_dirichlet_noise(root, legal_root)

        for _ in range(self.simulations):
            node = root
            search_path = [node]
            # Selection
            while node.expanded():
                move_idx, node = self._select_child(node)
                search_path.append(node)

            leaf = search_path[-1]
            if leaf.state.winner is not None:
                leaf_value = 1.0 if leaf.state.winner == leaf.state.turn else -1.0
            else:
                legal_moves = list_legal_moves(leaf.state)
                if not legal_moves:
                    leaf_value = -1.0
                else:
                    policy, value = self._evaluate(leaf.state)
                    self._expand(leaf, policy, legal_moves)
                    leaf_value = value

            # Backpropagate (flip value each ply)
            v = leaf_value
            for node in reversed(search_path):
                node.visits += 1
                node.value_sum += v
                v = -v

        if torch is None:
            best_move_idx = max(root.children.items(), key=lambda kv: kv[1].visits)[0]
            return ALL_MOVES[best_move_idx], None

        legal_indices = [MOVE_TO_INDEX[(mv.origin, mv.target)] for mv in legal_root]
        visit_counts = torch.zeros(len(ALL_MOVES), dtype=torch.float32)
        for idx, child in root.children.items():
            visit_counts[idx] = float(child.visits)

        policy_target = torch.zeros(len(ALL_MOVES), dtype=torch.float32)
        total_visits = float(visit_counts.sum().item())
        if total_visits > 0.0:
            policy_target = visit_counts / total_visits
        else:
            uniform = 1.0 / len(legal_indices)
            for idx in legal_indices:
                policy_target[idx] = uniform

        if temperature <= 1e-6:
            best_move_idx = max(root.children.items(), key=lambda kv: kv[1].visits)[0]
            return ALL_MOVES[best_move_idx], policy_target

        legal_counts = visit_counts[legal_indices]
        if float(legal_counts.sum().item()) <= 0.0:
            legal_probs = torch.full((len(legal_indices),), 1.0 / len(legal_indices))
        else:
            powered = legal_counts.pow(1.0 / temperature)
            powered_sum = float(powered.sum().item())
            if powered_sum <= 0.0:
                legal_probs = torch.full((len(legal_indices),), 1.0 / len(legal_indices))
            else:
                legal_probs = powered / powered_sum
        sampled_offset = int(torch.multinomial(legal_probs, num_samples=1).item())
        best_move_idx = legal_indices[sampled_offset]
        return ALL_MOVES[best_move_idx], policy_target

    def _select_child(self, node: Node) -> Tuple[int, Node]:
        assert node.children is not None
        total_visits = sum(child.visits for child in node.children.values())

        def ucb(child: Node) -> float:
            prior = child.prior
            # Child value is from child.to_play perspective, so negate for parent.
            q = -child.q_value()
            u = self.c_puct * prior * math.sqrt(total_visits + 1e-8) / (1 + child.visits)
            return q + u

        move_idx, child = max(node.children.items(), key=lambda kv: ucb(kv[1]))
        return move_idx, child

    def _add_dirichlet_noise(
        self,
        node: Node,
        legal_moves: List[Move],
        alpha: float = 0.3,
        epsilon: float = 0.25,
    ) -> None:
        if torch is None or not node.children or len(legal_moves) <= 1:
            return
        concentration = torch.full((len(legal_moves),), alpha, dtype=torch.float32)
        noise = torch.distributions.Dirichlet(concentration).sample().tolist()
        for mv, noise_value in zip(legal_moves, noise):
            idx = MOVE_TO_INDEX[(mv.origin, mv.target)]
            child = node.children.get(idx)
            if child is None:
                continue
            child.prior = (1.0 - epsilon) * child.prior + epsilon * float(noise_value)

    def _expand(self, node: Node, policy_logits: "torch.Tensor", legal_moves: List[Move]) -> None:  # type: ignore[name-defined]
        node.children = {}
        if policy_logits is None or torch is None:
            probs = [1.0 / len(legal_moves)] * len(legal_moves)
        else:
            mask = torch.full((len(ALL_MOVES),), -1e9, device=policy_logits.device)
            for mv in legal_moves:
                idx = MOVE_TO_INDEX[(mv.origin, mv.target)]
                mask[idx] = policy_logits[idx]
            probs = F.softmax(mask, dim=0)
        for mv in legal_moves:
            idx = MOVE_TO_INDEX[(mv.origin, mv.target)]
            prior = float(probs[idx]) if torch is not None else 1.0 / len(legal_moves)
            next_state = apply_move(node.state, mv)
            node.children[idx] = Node(
                state=next_state,
                to_play=next_state.turn,
                prior=prior,
                children=None,
            )

    def _evaluate(self, state: GameState) -> Tuple["torch.Tensor", float]:  # type: ignore[name-defined]
        if self.model is None:
            raise RuntimeError("Model is unavailable for evaluation.")
        if torch is None:
            raise RuntimeError("PyTorch is unavailable for evaluation.")
        with torch.no_grad():
            tensor = state_to_tensor(state, device=self.device).unsqueeze(0)
            policy, value = self.model(tensor)
            return policy.squeeze(0), float(value.item())
