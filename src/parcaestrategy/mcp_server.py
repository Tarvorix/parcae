"""MCP tool server for Parcae Strategy.

Expose match operations so an external MCP client can play as an "agent"
participant against a human player or built-in AI.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import httpx
from mcp.server.fastmcp import FastMCP

API_BASE = os.getenv("PARCAE_API_URL", "http://127.0.0.1:8000").rstrip("/")
VALID_MODES = {"pva", "pvg", "gvg", "gva", "ava"}

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
        "gvg (agent vs agent), gva (agent vs ai), ava (ai vs ai)."
    )
)
def create_match(mode: str = "pvg") -> Dict[str, Any]:
    mode = mode.lower().strip()
    if mode not in VALID_MODES:
        raise ValueError(f"Unsupported mode '{mode}'. Valid modes: {sorted(VALID_MODES)}")
    return _request("POST", "/match", json={"mode": mode})


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


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
