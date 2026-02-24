"""Core rules engine for Parcae Strategy (Ludus Latrunculorum - Kowalski/Stanway).

The engine is deterministic and UI-agnostic so it can be shared by the server
and clients. Coordinates are zero-based tuples (file, rank) but helpers for
algebraic-style strings (e.g. "A1") are provided for API friendliness.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

BOARD_WIDTH = 12
BOARD_HEIGHT = 8
FILES = "ABCDEFGHIJKL"
RANKS = list(range(1, BOARD_HEIGHT + 1))
Coord = Tuple[int, int]  # (file, rank), zero-based


class Color(str, Enum):
    WHITE = "white"
    BLACK = "black"

    def opponent(self) -> "Color":
        return Color.BLACK if self is Color.WHITE else Color.WHITE


class PieceType(str, Enum):
    SOLDIER = "soldier"
    DUX = "dux"


@dataclass(frozen=True)
class Piece:
    color: Color
    kind: PieceType


@dataclass(frozen=True)
class Move:
    origin: Coord
    target: Coord


@dataclass
class GameState:
    board: Dict[Coord, Piece]
    turn: Color = Color.WHITE
    winner: Optional[Color] = None
    captures: Dict[Color, int] = field(
        default_factory=lambda: {Color.WHITE: 0, Color.BLACK: 0}
    )
    history: List[Move] = field(default_factory=list)
    summary: Optional[str] = None

    @property
    def active(self) -> bool:
        return self.winner is None


DIRECTIONS: Sequence[Coord] = ((1, 0), (-1, 0), (0, 1), (0, -1))
CORNERS: Sequence[Coord] = (
    (0, 0),
    (BOARD_WIDTH - 1, 0),
    (0, BOARD_HEIGHT - 1),
    (BOARD_WIDTH - 1, BOARD_HEIGHT - 1),
)


def coord_in_bounds(coord: Coord) -> bool:
    file_idx, rank_idx = coord
    return 0 <= file_idx < BOARD_WIDTH and 0 <= rank_idx < BOARD_HEIGHT


def coord_to_notation(coord: Coord) -> str:
    file_idx, rank_idx = coord
    return f"{FILES[file_idx]}{rank_idx + 1}"


def notation_to_coord(token: str) -> Coord:
    if len(token) < 2:
        raise ValueError(f"Invalid coordinate token: {token}")
    file_char, rank_str = token[0].upper(), token[1:]
    if file_char not in FILES:
        raise ValueError(f"Invalid file: {file_char}")
    rank = int(rank_str) - 1
    file_idx = FILES.index(file_char)
    coord = (file_idx, rank)
    if not coord_in_bounds(coord):
        raise ValueError(f"Out of bounds coordinate: {token}")
    return coord


def initial_state(dux_file: int = 5) -> GameState:
    """Create the default starting position. dux_file is zero-based index."""
    if not 0 <= dux_file < BOARD_WIDTH:
        raise ValueError("dux_file must be within board files")
    board: Dict[Coord, Piece] = {}
    for file_idx in range(BOARD_WIDTH):
        board[(file_idx, 0)] = Piece(
            Color.WHITE,
            PieceType.DUX if file_idx == dux_file else PieceType.SOLDIER,
        )
        board[(file_idx, BOARD_HEIGHT - 1)] = Piece(
            Color.BLACK,
            PieceType.DUX if file_idx == dux_file else PieceType.SOLDIER,
        )
    return GameState(board=board, turn=Color.WHITE)


def legal_moves(state: GameState, origin: Coord) -> List[Coord]:
    """Return empty target squares the piece at origin may slide to."""
    piece = state.board.get(origin)
    if piece is None:
        return []
    moves: List[Coord] = []
    for dx, dy in DIRECTIONS:
        step = 1
        while True:
            target = (origin[0] + dx * step, origin[1] + dy * step)
            if not coord_in_bounds(target):
                break
            occupant = state.board.get(target)
            if occupant:
                # Cannot land on occupied squares; sliding stops before them.
                break
            moves.append(target)
            step += 1
    return moves


def apply_move(state: GameState, move: Move) -> GameState:
    if not state.active:
        raise ValueError("Game already finished")
    piece = state.board.get(move.origin)
    if piece is None:
        raise ValueError("No piece at origin")
    if piece.color is not state.turn:
        raise ValueError("Not this piece's turn")
    if move.target not in legal_moves(state, move.origin):
        raise ValueError("Illegal move target")

    board: Dict[Coord, Piece] = dict(state.board)
    board.pop(move.origin, None)
    board[move.target] = piece

    captured_positions = _resolve_captures(board, piece.color, move.target)
    for pos in captured_positions:
        board.pop(pos, None)

    captures = dict(state.captures)
    captures[piece.color] += len(captured_positions)

    history = list(state.history)
    history.append(move)

    next_turn = state.turn.opponent()
    winner, summary = _determine_winner(board, piece.color, next_turn, captures)

    return GameState(
        board=board,
        turn=next_turn if winner is None else next_turn,
        winner=winner,
        captures=captures,
        history=history,
        summary=summary,
    )


def _resolve_captures(board: Dict[Coord, Piece], mover: Color, moved_to: Coord) -> Set[Coord]:
    """Compute captured enemy soldiers after mover placed at moved_to."""
    captured: Set[Coord] = set()
    for dx, dy in DIRECTIONS:
        adjacent = (moved_to[0] + dx, moved_to[1] + dy)
        if not coord_in_bounds(adjacent):
            continue
        target_piece = board.get(adjacent)
        if target_piece is None or target_piece.color is mover:
            continue
        beyond = (adjacent[0] + dx, adjacent[1] + dy)
        if coord_in_bounds(beyond):
            guardian = board.get(beyond)
            if guardian and guardian.color is mover and target_piece.kind is PieceType.SOLDIER:
                captured.add(adjacent)

    # Corner capture: enemy in corner removed if mover now blocks both orthogonal adjacents.
    for corner in CORNERS:
        occupant = board.get(corner)
        if occupant is None or occupant.color is mover or occupant.kind is PieceType.DUX:
            continue
        adjacents = _orthogonal_adjacents(corner)
        if all(
            board.get(adj) and board[adj].color is mover for adj in adjacents if coord_in_bounds(adj)
        ):
            captured.add(corner)

    return captured


def _orthogonal_adjacents(coord: Coord) -> List[Coord]:
    x, y = coord
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]


def _find_piece(board: Dict[Coord, Piece], color: Color, kind: PieceType) -> Optional[Coord]:
    for coord, piece in board.items():
        if piece.color is color and piece.kind is kind:
            return coord
    return None


def _determine_winner(
    board: Dict[Coord, Piece],
    mover: Color,
    next_turn: Color,
    captures: Dict[Color, int],
) -> Tuple[Optional[Color], Optional[str]]:
    opponent = mover.opponent()
    opponent_dux = _find_piece(board, opponent, PieceType.DUX)
    if opponent_dux is not None and not _piece_has_legal_moves(board, opponent_dux):
        return mover, "Opponent Dux immobilized."

    if not any_legal_moves(board, opponent):
        return mover, "Opponent has no legal moves."

    opponent_soldiers = sum(
        1 for piece in board.values() if piece.color is opponent and piece.kind is PieceType.SOLDIER
    )
    if opponent_soldiers == 0:
        return mover, "All opposing soldiers removed (material victory)."

    return None, None


def _piece_has_legal_moves(board: Dict[Coord, Piece], origin: Coord) -> bool:
    piece = board.get(origin)
    if piece is None:
        return False
    dummy_state = GameState(board=board)
    return bool(legal_moves(dummy_state, origin))


def any_legal_moves(board: Dict[Coord, Piece], color: Color) -> bool:
    for coord, piece in board.items():
        if piece.color is color:
            if _piece_has_legal_moves(board, coord):
                return True
    return False


def serialize_state(state: GameState) -> Dict:
    """Serialize GameState to a JSON-friendly dict."""
    return {
        "turn": state.turn.value,
        "winner": state.winner.value if state.winner else None,
        "captures": {c.value: v for c, v in state.captures.items()},
        "history": [
            {"origin": coord_to_notation(m.origin), "target": coord_to_notation(m.target)}
            for m in state.history
        ],
        "board": [
            {
                "coord": coord_to_notation(coord),
                "piece": {"color": piece.color.value, "kind": piece.kind.value},
            }
            for coord, piece in sorted(state.board.items())
        ],
        "summary": state.summary,
    }


def deserialize_state(payload: Dict) -> GameState:
    board: Dict[Coord, Piece] = {}
    for item in payload.get("board", []):
        coord = notation_to_coord(item["coord"])
        piece = Piece(color=Color(item["piece"]["color"]), kind=PieceType(item["piece"]["kind"]))
        board[coord] = piece
    captures_payload = payload.get("captures") or {}
    captures = {
        Color.WHITE: captures_payload.get("white", 0),
        Color.BLACK: captures_payload.get("black", 0),
    }
    history_payload = payload.get("history") or []
    history = [
        Move(origin=notation_to_coord(m["origin"]), target=notation_to_coord(m["target"]))
        for m in history_payload
    ]
    return GameState(
        board=board,
        turn=Color(payload.get("turn", "white")),
        winner=Color(payload["winner"]) if payload.get("winner") else None,
        captures=captures,
        history=history,
        summary=payload.get("summary"),
    )
