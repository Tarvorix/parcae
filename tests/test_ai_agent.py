from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")

from parcaestrategy.ai.agent import AIAgent, Node
from parcaestrategy.ai.selfplay import self_play_game
from parcaestrategy.engine import Color, initial_state


def test_agent_without_checkpoint_uses_heuristic() -> None:
    agent = AIAgent(model_path=None)
    state = initial_state()
    move = agent.select_move(state)
    assert not agent.available
    assert move is not None


def test_incompatible_checkpoint_falls_back_to_heuristic() -> None:
    checkpoint = Path("models/python/parcae_model.pth")
    if not checkpoint.exists():
        pytest.skip("Bundled checkpoint not present in this environment.")

    with pytest.warns(RuntimeWarning, match="using heuristic fallback"):
        agent = AIAgent(model_path=str(checkpoint))
    state = initial_state()
    move = agent.select_move(state)
    assert not agent.available
    assert agent.load_error
    assert move is not None


def test_ucb_selection_uses_parent_perspective() -> None:
    state = initial_state()
    agent = AIAgent(model_path=None, c_puct=0.0)
    root = Node(state=state, to_play=Color.WHITE, prior=1.0, children={})
    # q_value is stored from each child node's own perspective.
    # For parent selection, a high child q should be treated as bad for parent.
    bad_for_parent = Node(
        state=state,
        to_play=Color.BLACK,
        prior=0.5,
        visits=10,
        value_sum=9.0,  # +0.9 for child player
        children=None,
    )
    good_for_parent = Node(
        state=state,
        to_play=Color.BLACK,
        prior=0.5,
        visits=10,
        value_sum=-9.0,  # -0.9 for child player, therefore good for parent
        children=None,
    )
    root.children = {0: bad_for_parent, 1: good_for_parent}
    selected_idx, _ = agent._select_child(root)
    assert selected_idx == 1


def test_self_play_fallback_policy_target_is_one_hot() -> None:
    agent = AIAgent(model_path=None)
    experiences, _winner = self_play_game(agent, max_plies=1)
    assert experiences
    pi = experiences[0].pi
    assert float(pi.sum().item()) == pytest.approx(1.0)
    assert int((pi > 0).sum().item()) == 1
    assert float(pi.max().item()) == pytest.approx(1.0)
