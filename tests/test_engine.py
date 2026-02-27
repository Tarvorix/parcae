import pytest

from parcaestrategy.engine import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    Color,
    GameState,
    Move,
    Piece,
    PieceType,
    apply_move,
    initial_state,
    legal_moves,
)


def test_initial_state_counts():
    state = initial_state()
    assert len(state.board) == 24
    white_pieces = [p for p in state.board.values() if p.color is Color.WHITE]
    black_pieces = [p for p in state.board.values() if p.color is Color.BLACK]
    assert len(white_pieces) == 12
    assert len(black_pieces) == 12
    assert state.turn is Color.WHITE


def test_dux_has_clear_vertical_moves_from_start():
    state = initial_state(dux_file=5)
    dux_origin = (5, 0)
    targets = legal_moves(state, dux_origin)
    assert len(targets) == BOARD_HEIGHT - 2  # blocked by black dux on far rank
    assert (5, BOARD_HEIGHT - 2) in targets


def test_custodial_capture_single():
    # White moves to flank a black soldier.
    board = {
        (0, 0): Piece(Color.WHITE, PieceType.SOLDIER),
        (1, 0): Piece(Color.BLACK, PieceType.SOLDIER),
        (3, 0): Piece(Color.WHITE, PieceType.SOLDIER),
    }
    state = GameState(board=board, turn=Color.WHITE)
    new_state = apply_move(state, Move(origin=(3, 0), target=(2, 0)))
    assert (1, 0) not in new_state.board  # black captured
    assert new_state.captures[Color.WHITE] == 1


def test_custodial_double_capture():
    board = {
        (0, 0): Piece(Color.WHITE, PieceType.SOLDIER),
        (1, 0): Piece(Color.BLACK, PieceType.SOLDIER),
        (3, 0): Piece(Color.BLACK, PieceType.SOLDIER),
        (4, 0): Piece(Color.WHITE, PieceType.SOLDIER),
        (2, 3): Piece(Color.WHITE, PieceType.SOLDIER),
    }
    state = GameState(board=board, turn=Color.WHITE)
    new_state = apply_move(state, Move(origin=(2, 3), target=(2, 0)))
    assert (1, 0) not in new_state.board
    assert (3, 0) not in new_state.board
    assert new_state.captures[Color.WHITE] == 2


def test_corner_capture():
    board = {
        (0, 0): Piece(Color.BLACK, PieceType.SOLDIER),
        (0, 1): Piece(Color.WHITE, PieceType.SOLDIER),
        (1, 3): Piece(Color.WHITE, PieceType.SOLDIER),
    }
    state = GameState(board=board, turn=Color.WHITE)
    new_state = apply_move(state, Move(origin=(1, 3), target=(1, 0)))
    assert (0, 0) not in new_state.board
    assert new_state.captures[Color.WHITE] == 1


def test_suicide_move_not_captured():
    # White may move between two black pieces; no auto-capture happens to white.
    board = {
        (1, 0): Piece(Color.BLACK, PieceType.SOLDIER),
        (3, 0): Piece(Color.BLACK, PieceType.SOLDIER),
        (2, 3): Piece(Color.WHITE, PieceType.SOLDIER),
    }
    state = GameState(board=board, turn=Color.WHITE)
    new_state = apply_move(state, Move(origin=(2, 3), target=(2, 0)))
    assert (2, 0) in new_state.board
    assert new_state.board[(2, 0)].color is Color.WHITE


def test_dux_immobilization_is_victory():
    board = {
        (0, 0): Piece(Color.BLACK, PieceType.DUX),
        (0, 2): Piece(Color.WHITE, PieceType.SOLDIER),
        (1, 0): Piece(Color.WHITE, PieceType.SOLDIER),
    }
    state = GameState(board=board, turn=Color.WHITE)
    result = apply_move(state, Move(origin=(0, 2), target=(0, 1)))
    assert result.winner is Color.WHITE
    assert result.summary and "Dux" in result.summary


def test_dux_blocked_by_own_side_is_not_immobilization_loss():
    state = initial_state(dux_file=5)
    result = apply_move(state, Move(origin=(5, 0), target=(5, 6)))
    assert result.winner is None
