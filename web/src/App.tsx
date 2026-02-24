import React, { useEffect, useMemo, useState } from "react";
import "./styles.css";
import { Board } from "./components/Board";
import { boardToMap, legalMovesClient } from "./coords";
import { createMatch, postMove, stepAi } from "./api";
import type { MatchMode, MatchResponse, Piece } from "./types";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";
const MODE_LABELS: Record<MatchMode, string> = {
  pva: "Player vs AI",
  pvg: "Player vs Agent",
  gvg: "Agent vs Agent",
  gva: "Agent vs AI",
  ava: "AI vs AI",
};

function useMatch(mode: MatchMode, reloadKey: number) {
  const [match, setMatch] = useState<MatchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

  useEffect(() => {
    setMatch(null);
    setLoading(true);
    createMatch(mode)
      .then(setMatch)
      .catch((err) => setError(String(err)))
      .finally(() => setLoading(false));
  }, [mode, reloadKey]);

  return { match, setMatch, error, setError, loading, setLoading };
}

export const App: React.FC = () => {
  const [mode, setMode] = useState<MatchMode>("pva");
  const [reloadKey, setReloadKey] = useState<number>(0);
  const { match, setMatch, error, setError, loading, setLoading } = useMatch(
    mode,
    reloadKey,
  );
  const [selected, setSelected] = useState<string | null>(null);
  const [legalTargets, setLegalTargets] = useState<string[]>([]);
  const [socketError, setSocketError] = useState<string | null>(null);

  const boardMap = useMemo<Record<string, Piece>>(() => {
    if (!match) return {};
    return boardToMap(match.state.board);
  }, [match]);

  const isPlayersTurn = match ? match.players[match.state.turn] === "player" : false;
  const canStepAi =
    match != null &&
    !match.state.winner &&
    match.players[match.state.turn] === "ai";

  const resetMatch = (nextMode: MatchMode) => {
    setMode(nextMode);
    setReloadKey((v) => v + 1);
    setSelected(null);
    setLegalTargets([]);
  };

  const onCellClick = (coord: string) => {
    if (!match || match.state.winner) return;
    if (!isPlayersTurn) return;

    if (selected && legalTargets.includes(coord)) {
      setLoading(true);
      postMove(match.id, selected, coord)
        .then((res) => setMatch(res))
        .catch((err) => setError(String(err)))
        .finally(() => {
          setSelected(null);
          setLegalTargets([]);
          setLoading(false);
        });
      return;
    }

    const piece = boardMap[coord];
    if (piece && piece.color === match.state.turn) {
      setSelected(coord);
      setLegalTargets(legalMovesClient(boardMap, coord));
    } else {
      setSelected(null);
      setLegalTargets([]);
    }
  };

  const handleAiStep = () => {
    if (!match) return;
    setLoading(true);
    stepAi(match.id)
      .then((res) => setMatch(res))
      .catch((err) => setError(String(err)))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    if (!match) return;
    const wsUrl = `${API_BASE.replace(/^http/, "ws")}/ws/match/${match.id}`;
    const ws = new WebSocket(wsUrl);
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as MatchResponse;
        setMatch(data);
        setSocketError(null);
      } catch (e) {
        setSocketError("Failed to parse live update");
      }
    };
    ws.onerror = () => setSocketError("WebSocket error");
    return () => ws.close();
  }, [match?.id]);

  return (
    <div className="page">
      <header className="header">
        <div>
          <h1>Parcae Strategy</h1>
          <p className="subtitle">
            Kowalski/Stanway rules. Mix player, MCP agent, and built-in AI opponents.
          </p>
        </div>
        <div className="controls">
          <button onClick={() => resetMatch("pva")}>New PvAI</button>
          <button onClick={() => resetMatch("pvg")}>New PvAgent</button>
          <button onClick={() => resetMatch("gva")}>New Agent vs AI</button>
          <button onClick={() => resetMatch("gvg")}>New Agent vs Agent</button>
          <button onClick={() => resetMatch("ava")}>New AI vs AI</button>
          <button onClick={handleAiStep} disabled={loading || !canStepAi}>
              Step AI
          </button>
        </div>
      </header>

      {error && <div className="alert">Error: {error}</div>}
      {socketError && <div className="alert">Live update issue: {socketError}</div>}
      {loading && <div className="hint">Working...</div>}

      {!match && <div className="hint">Spawning match...</div>}

      {match && (
        <div className="content">
          <div className="board-column">
            <Board
              board={boardMap}
              selected={selected}
              legalMoves={legalTargets}
              onCellClick={onCellClick}
            />
          </div>
          <div className="sidebar">
            <div className="card">
              <div className="label">Match</div>
              <div className="value">#{match.id}</div>
              <div className="label">Mode</div>
              <div className="value">{MODE_LABELS[match.mode]}</div>
              <div className="label">Sides</div>
              <div className="value">
                White: {match.players.white} / Black: {match.players.black}
              </div>
            </div>

            <div className="card">
              <div className="label">Turn</div>
              <div className="value turn">
                {match.state.winner
                  ? "Game over"
                  : match.state.turn === "white"
                    ? "White"
                    : "Black"}
              </div>
              <div className="label">Turn Owner</div>
              <div className="value">{match.players[match.state.turn]}</div>
              <div className="label">Captures</div>
              <div className="value">
                White {match.state.captures.white} / Black {match.state.captures.black}
              </div>
            </div>

            <div className="card">
              <div className="label">History</div>
              <div className="history">
                {match.state.history.length === 0 && <span>None yet</span>}
                {match.state.history.map((move, idx) => (
                  <div key={`${move.origin}-${move.target}-${idx}`}>
                    {idx + 1}. {move.origin} → {move.target}
                  </div>
                ))}
              </div>
            </div>

            {match.state.winner && (
              <div className="card winner">
                <div className="label">Winner</div>
                <div className="value">{match.state.winner}</div>
                {match.state.summary && <div className="summary">{match.state.summary}</div>}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
