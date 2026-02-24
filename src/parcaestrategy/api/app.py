from __future__ import annotations

import asyncio
import uuid
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Set, Tuple

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from parcaestrategy.engine import (
    Color,
    GameState,
    Move,
    apply_move,
    coord_to_notation,
    initial_state,
    legal_moves,
    notation_to_coord,
    serialize_state,
)

try:
    from parcaestrategy.ai.agent import AIAgent
except Exception:  # pragma: no cover - AI optional
    AIAgent = None

Controller = Literal["player", "agent", "ai"]
MatchMode = Literal["pva", "ava", "pvg", "gvg", "gva"]

MODE_CONTROLLERS: Dict[MatchMode, Tuple[Controller, Controller]] = {
    "pva": ("player", "ai"),
    "ava": ("ai", "ai"),
    "pvg": ("player", "agent"),
    "gvg": ("agent", "agent"),
    "gva": ("agent", "ai"),
}


class CreateMatchRequest(BaseModel):
    mode: MatchMode = "pva"


class MoveRequest(BaseModel):
    origin: str
    target: str


class MatchResponse(BaseModel):
    id: str
    mode: MatchMode
    players: Dict[Literal["white", "black"], Controller]
    state: Dict


@dataclass
class Match:
    id: str
    mode: MatchMode
    white_controller: Controller
    black_controller: Controller
    state: GameState

    def controller_for(self, color: Color) -> Controller:
        return self.white_controller if color is Color.WHITE else self.black_controller

    @property
    def ai_colors(self) -> Set[Color]:
        colors: Set[Color] = set()
        if self.white_controller == "ai":
            colors.add(Color.WHITE)
        if self.black_controller == "ai":
            colors.add(Color.BLACK)
        return colors


class Hub:
    def __init__(self) -> None:
        self.connections: Dict[str, Set[WebSocket]] = defaultdict(set)
        self._lock = asyncio.Lock()

    async def connect(self, match_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self.connections[match_id].add(websocket)

    async def disconnect(self, match_id: str, websocket: WebSocket) -> None:
        async with self._lock:
            self.connections[match_id].discard(websocket)

    async def broadcast(self, match_id: str, payload: Dict) -> None:
        async with self._lock:
            recipients = list(self.connections.get(match_id, set()))
        for ws in recipients:
            try:
                await ws.send_json(payload)
            except WebSocketDisconnect:
                await self.disconnect(match_id, ws)
            except Exception:
                await self.disconnect(match_id, ws)


def create_app() -> FastAPI:
    app = FastAPI(title="Parcae Strategy API")
    hub = Hub()
    matches: Dict[str, Match] = {}
    ai_agent = AIAgent() if AIAgent is not None else None

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def serialize_match(match: Match) -> Dict:
        return {
            "id": match.id,
            "mode": match.mode,
            "players": {
                "white": match.white_controller,
                "black": match.black_controller,
            },
            "state": serialize_state(match.state),
        }

    def require_match(match_id: str) -> Match:
        match = matches.get(match_id)
        if match is None:
            raise HTTPException(status_code=404, detail="Match not found")
        return match

    def list_legal_moves_for_color(state: GameState, color: Color) -> List[Move]:
        moves: List[Move] = []
        for coord, piece in state.board.items():
            if piece.color is color:
                for target in legal_moves(state, coord):
                    moves.append(Move(origin=coord, target=target))
        return moves

    def evaluate_move(state: GameState, move: Move) -> Tuple[int, GameState]:
        """Return (score, next_state) where score favors immediate wins/captures."""
        next_state = apply_move(state, move)
        mover = state.board[move.target].color if move.target in state.board else state.turn
        score = 0
        if next_state.winner == mover:
            score += 1000
        capture_gain = next_state.captures[mover] - state.captures[mover]
        score += capture_gain * 10
        return score, next_state

    def choose_ai_move(state: GameState, color: Color) -> Optional[Move]:
        # If a learning agent is available, use it; otherwise fall back to heuristic.
        if ai_agent and ai_agent.available:
            return ai_agent.select_move(state)
        moves = list_legal_moves_for_color(state, color)
        if not moves:
            return None
        best_score = -10**9
        best_move = moves[0]
        for move in moves:
            score, _ = evaluate_move(state, move)
            if score > best_score:
                best_score = score
                best_move = move
        return best_move

    def run_ai_turns(match: Match, max_plies: Optional[int] = None) -> None:
        # Let AIs play until a limit or a human turn.
        plies = 0
        while match.state.active and match.state.turn in match.ai_colors:
            if max_plies is not None and plies >= max_plies:
                break
            move = choose_ai_move(match.state, match.state.turn)
            if move is None:
                # No legal move; opponent wins by immobilization.
                match.state.winner = match.state.turn.opponent()
                match.state.summary = "AI has no legal moves."
                break
            match.state = apply_move(match.state, move)
            plies += 1

    @app.get("/health")
    async def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.post("/match", response_model=MatchResponse)
    async def create_match(req: CreateMatchRequest) -> MatchResponse:
        match_id = uuid.uuid4().hex[:8]
        white_controller, black_controller = MODE_CONTROLLERS[req.mode]
        match = Match(
            id=match_id,
            mode=req.mode,
            white_controller=white_controller,
            black_controller=black_controller,
            state=initial_state(),
        )
        # For AI vs AI, make a single opening ply so spectators can watch from move 1.
        run_ai_turns(match, max_plies=1 if req.mode == "ava" else None)
        matches[match_id] = match
        payload = serialize_match(match)
        asyncio.create_task(hub.broadcast(match_id, payload))
        return MatchResponse(**payload)

    @app.get("/match/{match_id}", response_model=MatchResponse)
    async def get_match(match_id: str) -> MatchResponse:
        match = require_match(match_id)
        return MatchResponse(**serialize_match(match))

    @app.get("/match/{match_id}/legal")
    async def get_legal(match_id: str, origin: Optional[str] = None) -> Dict:
        match = require_match(match_id)
        moves: List[Dict[str, str]] = []
        if origin is not None:
            try:
                source = notation_to_coord(origin)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            piece = match.state.board.get(source)
            if piece is not None and piece.color is match.state.turn:
                moves = [
                    {"origin": coord_to_notation(source), "target": coord_to_notation(target)}
                    for target in legal_moves(match.state, source)
                ]
        else:
            for source, piece in match.state.board.items():
                if piece.color is not match.state.turn:
                    continue
                for target in legal_moves(match.state, source):
                    moves.append(
                        {"origin": coord_to_notation(source), "target": coord_to_notation(target)}
                    )
            moves.sort(key=lambda item: (item["origin"], item["target"]))

        return {
            "id": match.id,
            "turn": match.state.turn.value,
            "origin": origin.upper() if origin else None,
            "moves": moves,
        }

    @app.post("/match/{match_id}/move", response_model=MatchResponse)
    async def play_move(match_id: str, body: MoveRequest) -> MatchResponse:
        match = require_match(match_id)
        if not match.state.active:
            raise HTTPException(status_code=400, detail="Match already finished")
        origin = notation_to_coord(body.origin)
        target = notation_to_coord(body.target)
        move = Move(origin=origin, target=target)
        try:
            match.state = apply_move(match.state, move)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        run_ai_turns(match)
        payload = serialize_match(match)
        asyncio.create_task(hub.broadcast(match_id, payload))
        return MatchResponse(**payload)

    @app.post("/match/{match_id}/ai-step", response_model=MatchResponse)
    async def step_ai(match_id: str) -> MatchResponse:
        match = require_match(match_id)
        if match.state.turn not in match.ai_colors:
            raise HTTPException(status_code=400, detail="No AI is configured for the current turn")
        run_ai_turns(match, max_plies=1)
        payload = serialize_match(match)
        asyncio.create_task(hub.broadcast(match_id, payload))
        return MatchResponse(**payload)

    @app.websocket("/ws/match/{match_id}")
    async def ws_match(websocket: WebSocket, match_id: str) -> None:
        await hub.connect(match_id, websocket)
        try:
            match = matches.get(match_id)
            if match:
                await websocket.send_json(serialize_match(match))
            while True:
                # Keep connection open; we don't accept inbound commands yet.
                await websocket.receive_text()
        except WebSocketDisconnect:
            await hub.disconnect(match_id, websocket)
        except Exception:
            await hub.disconnect(match_id, websocket)

    return app
