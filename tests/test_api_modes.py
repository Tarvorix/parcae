from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from parcaestrategy.api import create_app


@pytest.fixture()
def client() -> TestClient:
    app = create_app()
    return TestClient(app)


def test_create_player_vs_agent_mode(client: TestClient) -> None:
    res = client.post("/match", json={"mode": "pvg"})
    assert res.status_code == 200
    body = res.json()
    assert body["mode"] == "pvg"
    assert body["players"] == {"white": "player", "black": "agent"}
    assert body["state"]["turn"] == "white"


def test_legal_moves_endpoint_returns_notation(client: TestClient) -> None:
    match = client.post("/match", json={"mode": "pvg"}).json()
    match_id = match["id"]

    res = client.get(f"/match/{match_id}/legal?origin=A1")
    assert res.status_code == 200
    body = res.json()
    assert body["id"] == match_id
    assert body["turn"] == "white"
    assert {"origin": "A1", "target": "A2"} in body["moves"]


def test_player_vs_agent_does_not_auto_move_black(client: TestClient) -> None:
    match = client.post("/match", json={"mode": "pvg"}).json()
    match_id = match["id"]

    res = client.post(f"/match/{match_id}/move", json={"origin": "A1", "target": "A2"})
    assert res.status_code == 200
    body = res.json()
    assert len(body["state"]["history"]) == 1
    assert body["state"]["turn"] == "black"


def test_agent_vs_ai_auto_responds_after_agent_turn(client: TestClient) -> None:
    match = client.post("/match", json={"mode": "gva"}).json()
    match_id = match["id"]

    res = client.post(f"/match/{match_id}/move", json={"origin": "A1", "target": "A2"})
    assert res.status_code == 200
    body = res.json()
    assert len(body["state"]["history"]) == 2
    assert body["state"]["turn"] == "white"


def test_agent_vs_agent_waits_for_external_moves(client: TestClient) -> None:
    match = client.post("/match", json={"mode": "gvg"}).json()
    match_id = match["id"]

    res = client.post(f"/match/{match_id}/move", json={"origin": "A1", "target": "A2"})
    assert res.status_code == 200
    body = res.json()
    assert len(body["state"]["history"]) == 1
    assert body["state"]["turn"] == "black"
