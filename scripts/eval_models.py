#!/usr/bin/env python3
"""
Head-to-head evaluator for Abaddon checkpoints.

Example:
  PYTHONPATH=src python3 scripts/eval_models.py \
    --challenger checkpoints/abaddon_new.pt \
    --baseline checkpoints/abaddon_prev.pt \
    --games 40 \
    --simulations 128
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from parcaestrategy.ai.agent import AIAgent, list_legal_moves
from parcaestrategy.engine import Color, GameState, apply_move, initial_state


@dataclass
class MatchResult:
    winner: Optional[Color]
    plies: int


def play_game(
    white: AIAgent,
    black: AIAgent,
    max_plies: int,
) -> MatchResult:
    state: GameState = initial_state()
    plies = 0
    while state.winner is None and plies < max_plies:
        legal = list_legal_moves(state)
        if not legal:
            state.winner = state.turn.opponent()
            state.summary = "No legal moves."
            break
        agent = white if state.turn is Color.WHITE else black
        move = agent.select_move(state)
        if move is None:
            state.winner = state.turn.opponent()
            state.summary = "No move selected."
            break
        state = apply_move(state, move)
        plies += 1

    winner = state.winner
    if winner is None and plies >= max_plies:
        white_score = state.captures[Color.WHITE]
        black_score = state.captures[Color.BLACK]
        if white_score > black_score:
            winner = Color.WHITE
        elif black_score > white_score:
            winner = Color.BLACK
    return MatchResult(winner=winner, plies=plies)


def elo_from_score(score: float) -> float:
    score = min(0.9999, max(0.0001, score))
    return -400.0 * math.log10((1.0 / score) - 1.0)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate Abaddon checkpoint vs baseline.")
    parser.add_argument("--challenger", required=True, help="Path to challenger checkpoint.")
    parser.add_argument("--baseline", required=True, help="Path to baseline checkpoint.")
    parser.add_argument("--games", type=int, default=40, help="Number of games (default: 40).")
    parser.add_argument(
        "--simulations",
        type=int,
        default=128,
        help="MCTS simulations per move for both sides (default: 128).",
    )
    parser.add_argument(
        "--max-plies",
        type=int,
        default=256,
        help="Maximum plies per game before score tiebreak (default: 256).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    args = parser.parse_args(argv)

    random.seed(args.seed)

    challenger = AIAgent(model_path=args.challenger, simulations=args.simulations)
    baseline = AIAgent(model_path=args.baseline, simulations=args.simulations)
    if not challenger.available:
        raise RuntimeError(f"Challenger unavailable: {challenger.load_error}")
    if not baseline.available:
        raise RuntimeError(f"Baseline unavailable: {baseline.load_error}")

    wins = 0
    losses = 0
    draws = 0

    for game_idx in range(args.games):
        if game_idx % 2 == 0:
            white, black = challenger, baseline
            challenger_is_white = True
        else:
            white, black = baseline, challenger
            challenger_is_white = False
        result = play_game(white, black, max_plies=args.max_plies)
        if result.winner is None:
            draws += 1
            continue
        challenger_won = (result.winner is Color.WHITE and challenger_is_white) or (
            result.winner is Color.BLACK and not challenger_is_white
        )
        if challenger_won:
            wins += 1
        else:
            losses += 1

    total = wins + losses + draws
    score = (wins + 0.5 * draws) / max(1, total)
    elo = elo_from_score(score)
    variance = score * (1.0 - score) / max(1, total)
    ci = 1.96 * math.sqrt(variance)
    score_lo = max(0.0001, score - ci)
    score_hi = min(0.9999, score + ci)
    elo_lo = elo_from_score(score_lo)
    elo_hi = elo_from_score(score_hi)

    print("Abaddon head-to-head results")
    print(f"Games: {total}  Wins: {wins}  Losses: {losses}  Draws: {draws}")
    print(f"Score: {score:.4f}")
    print(f"Elo estimate: {elo:+.1f} (95% CI: {elo_lo:+.1f} .. {elo_hi:+.1f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
