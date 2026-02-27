"""MCP tool server for Parcae Strategy.

Expose match operations so an external MCP client can play as an "agent"
participant against a human player or built-in AI.
"""
from __future__ import annotations

import os
import socket
import threading
from typing import Any, Dict, Optional

import httpx
from mcp.server.fastmcp import FastMCP

API_BASE = os.getenv("PARCAE_API_URL", "http://127.0.0.1:8000").rstrip("/")
VALID_MODES = {"pva", "pvg", "gvg", "gva", "ava"}
AGENT_ID = os.getenv("PARCAE_AGENT_ID", f"mcp-{socket.gethostname()}-{os.getpid()}")
HEARTBEAT_SECONDS = max(2, int(os.getenv("PARCAE_AGENT_HEARTBEAT_SECONDS", "5")))

mcp = FastMCP("parcae-strategy")


def _request(
    method: str,
    path: str,
    json: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    with httpx.Client(timeout=20.0) as client:
        response = client.request(method, f"{API_BASE}{path}", json=json, params=params)
    if response.is_error:
        detail = response.text
        try:
            payload = response.json()
            detail = payload.get("detail", detail)
        except Exception:
            pass
        raise ValueError(f"{method} {path} failed ({response.status_code}): {detail}")
    return response.json()


@mcp.tool(description="Check whether the Parcae HTTP API is running.")
def health() -> Dict[str, Any]:
    return _request("GET", "/health")


@mcp.tool(
    description=(
        "Create a match. Modes: pva (player vs ai), pvg (player vs agent), "
        "gvg (agent vs agent), gva (agent vs ai), ava (ai vs ai). "
        "Optionally pass ai_profiles to select ai backend per side "
        "(heuristic|centurion|abaddon)."
    )
)
def create_match(mode: str = "pvg", ai_profiles: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    mode = mode.lower().strip()
    if mode not in VALID_MODES:
        raise ValueError(f"Unsupported mode '{mode}'. Valid modes: {sorted(VALID_MODES)}")
    payload: Dict[str, Any] = {"mode": mode}
    if ai_profiles is not None:
        payload["ai_profiles"] = ai_profiles
    return _request("POST", "/match", json=payload)


@mcp.tool(description="Fetch the current state of a match by id.")
def get_match(match_id: str) -> Dict[str, Any]:
    return _request("GET", f"/match/{match_id}")


@mcp.tool(
    description=(
        "List legal moves for the side to play. Optionally provide origin "
        "(e.g. A1) to restrict to one piece."
    )
)
def legal_moves(match_id: str, origin: Optional[str] = None) -> Dict[str, Any]:
    params = {"origin": origin} if origin else None
    return _request("GET", f"/match/{match_id}/legal", params=params)


@mcp.tool(description="Play a move in notation form, e.g. origin='A1' target='A4'.")
def play_move(match_id: str, origin: str, target: str) -> Dict[str, Any]:
    return _request(
        "POST",
        f"/match/{match_id}/move",
        json={"origin": origin.upper(), "target": target.upper()},
    )


@mcp.tool(description="Advance exactly one built-in AI ply when an AI side is on turn.")
def step_ai(match_id: str) -> Dict[str, Any]:
    return _request("POST", f"/match/{match_id}/ai-step")


def _heartbeat_loop(stop_event: threading.Event) -> None:
    with httpx.Client(timeout=5.0) as client:
        while not stop_event.is_set():
            try:
                client.post(
                    f"{API_BASE}/agent/heartbeat",
                    json={"agent_id": AGENT_ID},
                )
            except Exception:
                pass
            stop_event.wait(HEARTBEAT_SECONDS)


def main() -> None:
    stop_event = threading.Event()
    heartbeat_thread = threading.Thread(
        target=_heartbeat_loop,
        args=(stop_event,),
        daemon=True,
    )
    heartbeat_thread.start()
    try:
        mcp.run()
    finally:
        stop_event.set()
        heartbeat_thread.join(timeout=1.0)


if __name__ == "__main__":
    main()
