"""parcaestrategy package."""

from .engine import (  # noqa: F401
    GameState,
    Move,
    Piece,
    PieceType,
    Color,
    initial_state,
    legal_moves,
    apply_move,
    serialize_state,
    deserialize_state,
)
from .ai import AIAgent, train_self_play  # noqa: F401
__all__ = [
    "__version__",
    "GameState",
    "Move",
    "Piece",
    "PieceType",
    "Color",
    "initial_state",
    "legal_moves",
    "apply_move",
    "serialize_state",
    "deserialize_state",
    "create_app",
    "AIAgent",
    "train_self_play",
]

__version__ = "0.1.0"


def greet(name: str = "world") -> str:
    return f"Hello, {name}!"


def create_app():
    """Lazy import to avoid requiring FastAPI unless requested."""
    from parcaestrategy.api import create_app as factory

    return factory()
