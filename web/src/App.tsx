import React, { useEffect, useMemo, useState } from "react";
import "./styles.css";
import { Board } from "./components/Board";
import { boardToMap, legalMovesClient } from "./coords";
import {
  createMatch,
  fetchLegalMoves,
  fetchAgentStatus,
  fetchAiCapabilities,
  postMove,
  stepAi,
} from "./api";
import type {
  AgentStatusResponse,
  AiBackend,
  AiCapabilities,
  Color,
  MatchAiProfiles,
  MatchMode,
  MatchResponse,
  Piece,
} from "./types";

const API_BASE = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";
const MODE_LABELS: Record<MatchMode, string> = {
  pva: "Player vs AI",
  pvg: "Player vs Agent",
  gvg: "Agent vs Agent",
  gva: "Agent vs AI",
  ava: "AI vs AI",
};

const BACKEND_OPTIONS: AiBackend[] = ["heuristic", "centurion", "abaddon"];
const BACKEND_LABELS: Record<AiBackend, string> = {
  heuristic: "Heuristic",
  centurion: "Centurion",
  abaddon: "Abaddon",
  stockfish: "Centurion",
};

function aiSidesForMode(mode: MatchMode): Color[] {
  switch (mode) {
    case "pva":
      return ["black"];
    case "gva":
      return ["black"];
    case "ava":
      return ["white", "black"];
    default:
      return [];
  }
}

function requiredAgentCount(mode: MatchMode): number {
  switch (mode) {
    case "pvg":
    case "gva":
      return 1;
    case "gvg":
      return 2;
    default:
      return 0;
  }
}

function normalizeBackend(backend: AiBackend): AiBackend {
  if (backend === "stockfish") {
    return "centurion";
  }
  return backend;
}

function prettyAge(lastSeenUnixMs: number): string {
  const ageSec = Math.max(0, Math.floor((Date.now() - lastSeenUnixMs) / 1000));
  if (ageSec < 2) {
    return "just now";
  }
  return `${ageSec}s ago`;
}

export const App: React.FC = () => {
  const [mode, setMode] = useState<MatchMode>("pva");
  const [whiteBackend, setWhiteBackend] = useState<AiBackend>("centurion");
  const [blackBackend, setBlackBackend] = useState<AiBackend>("centurion");
  const [match, setMatch] = useState<MatchResponse | null>(null);
  const [capabilities, setCapabilities] = useState<AiCapabilities | null>(null);
  const [agentStatus, setAgentStatus] = useState<AgentStatusResponse | null>(null);
  const [statusError, setStatusError] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [selected, setSelected] = useState<string | null>(null);
  const [legalTargets, setLegalTargets] = useState<string[]>([]);
  const [socketError, setSocketError] = useState<string | null>(null);
  const [capabilitiesError, setCapabilitiesError] = useState<string | null>(null);

  const aiSides = aiSidesForMode(mode);
  const agentCountRequired = requiredAgentCount(mode);
  const abaddonUnavailable =
    capabilities !== null && !capabilities.abaddon_available;
  const connectedAgentCount = agentStatus?.active_count ?? 0;
  const agentsReady = connectedAgentCount >= agentCountRequired;
  const requiresAgents = agentCountRequired > 0;
  const hasAgentTelemetry = agentStatus !== null;
  const blockStartForAgents = requiresAgents && hasAgentTelemetry && !agentsReady;

  useEffect(() => {
    let cancelled = false;
    const refreshCaps = async () => {
      try {
        const caps = await fetchAiCapabilities();
        if (!cancelled) {
          setCapabilities(caps);
          setCapabilitiesError(null);
        }
      } catch {
        if (!cancelled) {
          setCapabilitiesError("AI capability check failed");
        }
      }
    };
    void refreshCaps();
    const timer = window.setInterval(() => {
      void refreshCaps();
    }, 8000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    if (!requiresAgents || match) {
      setAgentStatus(null);
      setStatusError(null);
      return;
    }
    let cancelled = false;
    const refresh = async () => {
      try {
        const status = await fetchAgentStatus();
        if (!cancelled) {
          setAgentStatus(status);
          setStatusError(null);
        }
      } catch {
        if (!cancelled) {
          setStatusError("Agent status check failed");
        }
      }
    };

    void refresh();
    const timer = window.setInterval(() => {
      void refresh();
    }, 3000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [requiresAgents, match]);

  const boardMap = useMemo<Record<string, Piece>>(() => {
    if (!match) return {};
    return boardToMap(match.state.board);
  }, [match]);

  const isPlayersTurn = match ? match.players[match.state.turn] === "player" : false;
  const canStepAi =
    match != null &&
    !match.state.winner &&
    match.players[match.state.turn] === "ai";

  const isBackendDisabled = (backend: AiBackend): boolean =>
    backend === "abaddon" && abaddonUnavailable;

  useEffect(() => {
    if (!abaddonUnavailable) return;
    if (whiteBackend === "abaddon") {
      setWhiteBackend("centurion");
    }
    if (blackBackend === "abaddon") {
      setBlackBackend("centurion");
    }
  }, [abaddonUnavailable, whiteBackend, blackBackend]);

  const selectedProfiles: MatchAiProfiles = {};
  if (aiSides.includes("white")) {
    selectedProfiles.white = { backend: normalizeBackend(whiteBackend) };
  }
  if (aiSides.includes("black")) {
    selectedProfiles.black = { backend: normalizeBackend(blackBackend) };
  }

  const startMatch = () => {
    setLoading(true);
    setError(null);
    setSocketError(null);
    setSelected(null);
    setLegalTargets([]);
    createMatch(mode, selectedProfiles)
      .then((res) => setMatch(res))
      .catch((err) => setError(String(err)))
      .finally(() => setLoading(false));
  };

  const returnToSetup = () => {
    setMatch(null);
    setSelected(null);
    setLegalTargets([]);
    setError(null);
    setSocketError(null);
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
      fetchLegalMoves(match.id, coord)
        .then((res) => {
          const targets = res.moves.map((mv) => mv.target);
          setLegalTargets(targets);
        })
        .catch(() => setLegalTargets(legalMovesClient(boardMap, coord)));
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
      } catch {
        setSocketError("Failed to parse live update");
      }
    };
    ws.onerror = () => setSocketError("WebSocket connection failed");
    return () => ws.close();
  }, [match?.id]);

  return (
    <div className="page">
      <header className="header">
        <div>
          <h1>Parcae Strategy</h1>
          <p className="subtitle">
            Choose game mode, pick AI backend(s), verify agent connections, then start.
          </p>
        </div>
      </header>

      {error && <div className="alert">Error: {error}</div>}
      {socketError && <div className="alert">Live update issue: {socketError}</div>}
      {capabilitiesError && <div className="hint warning">Status checks: {capabilitiesError}</div>}
      {statusError && <div className="hint warning">Status checks: {statusError}</div>}
      {loading && <div className="hint">Working...</div>}

      {!match && (
        <div className="content single-column">
          <div className="card setup-card">
            <div className="label">Game Setup</div>
            <div className="setup-grid">
              <label className="field">
                <span>Game Type</span>
                <select value={mode} onChange={(e) => setMode(e.target.value as MatchMode)}>
                  {(Object.keys(MODE_LABELS) as MatchMode[]).map((key) => (
                    <option key={key} value={key}>
                      {MODE_LABELS[key]}
                    </option>
                  ))}
                </select>
              </label>

              {aiSides.includes("white") && (
                <label className="field">
                  <span>White AI</span>
                  <select
                    value={whiteBackend}
                    onChange={(e) => setWhiteBackend(e.target.value as AiBackend)}
                  >
                    {BACKEND_OPTIONS.map((backend) => (
                      <option key={backend} value={backend} disabled={isBackendDisabled(backend)}>
                        {BACKEND_LABELS[backend]}
                      </option>
                    ))}
                  </select>
                </label>
              )}

              {aiSides.includes("black") && (
                <label className="field">
                  <span>Black AI</span>
                  <select
                    value={blackBackend}
                    onChange={(e) => setBlackBackend(e.target.value as AiBackend)}
                  >
                    {BACKEND_OPTIONS.map((backend) => (
                      <option key={backend} value={backend} disabled={isBackendDisabled(backend)}>
                        {BACKEND_LABELS[backend]}
                      </option>
                    ))}
                  </select>
                </label>
              )}
            </div>
            {abaddonUnavailable && (
              <div className="hint warning">
                Abaddon is unavailable in this runtime (missing or invalid model artifact).
              </div>
            )}

            {capabilities && (
              <div className="status-row">
                <div>
                  Backends:{" "}
                  {capabilities.supported_backends.map((b) => BACKEND_LABELS[b]).join(", ")}
                </div>
                <div>
                  Centurion strict: {capabilities.centurion_strict_mode ? "on" : "off"}
                </div>
                <div>
                  Centurion assets: {capabilities.centurion_assets_ok ? "ok" : "invalid"}
                </div>
                <div>Book: {capabilities.centurion_book_loaded ? "loaded" : "missing"}</div>
                <div>Tablebase: {capabilities.centurion_tb_loaded ? "loaded" : "missing"}</div>
                <div>NNUE: {capabilities.centurion_nnue_loaded ? "loaded" : "missing"}</div>
                <div>Abaddon: {capabilities.abaddon_available ? "loaded" : "missing"}</div>
              </div>
            )}

            {requiresAgents && (
              <div
                className={`agent-box ${
                  hasAgentTelemetry ? (agentsReady ? "ready" : "not-ready") : "unknown"
                }`}
              >
                <div className="label">Agent Connectivity</div>
                <div>
                  Required: {agentCountRequired} | Connected:{" "}
                  {hasAgentTelemetry ? connectedAgentCount : "unknown"}
                </div>
                {agentStatus && agentStatus.agents.length > 0 && (
                  <div className="agent-list">
                    {agentStatus.agents.map((entry) => (
                      <div key={entry.agent_id}>
                        {entry.connected ? "Online" : "Offline"}: {entry.agent_id} (
                        {prettyAge(entry.last_seen_unix_ms)})
                      </div>
                    ))}
                  </div>
                )}
                {hasAgentTelemetry && !agentsReady && (
                  <div className="hint">
                    Start your MCP agent process first, then this will auto-update.
                  </div>
                )}
                {!hasAgentTelemetry && (
                  <div className="hint">
                    Agent status endpoint is unavailable. You can still start the match.
                  </div>
                )}
              </div>
            )}

            <div className="controls">
              <button
                onClick={startMatch}
                disabled={loading || blockStartForAgents}
              >
                Start Match
              </button>
            </div>
          </div>
        </div>
      )}

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
              <div className="controls compact">
                <button onClick={returnToSetup}>Back to Setup</button>
                <button onClick={handleAiStep} disabled={loading || !canStepAi}>
                  Step AI
                </button>
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

            {match.ai_profiles && (match.ai_profiles.white || match.ai_profiles.black) && (
              <div className="card">
                <div className="label">AI Profiles</div>
                {match.ai_profiles.white && (
                  <div className="value">White: {BACKEND_LABELS[match.ai_profiles.white.backend ?? "heuristic"]}</div>
                )}
                {match.ai_profiles.black && (
                  <div className="value">Black: {BACKEND_LABELS[match.ai_profiles.black.backend ?? "heuristic"]}</div>
                )}
              </div>
            )}

            {match.last_ai_meta && (
              <div className="card">
                <div className="label">Last AI Move</div>
                <div className="value">
                  {BACKEND_LABELS[match.last_ai_meta.backend]} via {match.last_ai_meta.source}
                </div>
                <div className="meta-line">Depth: {match.last_ai_meta.depth_reached}</div>
                <div className="meta-line">Nodes: {match.last_ai_meta.nodes}</div>
                <div className="meta-line">NPS: {match.last_ai_meta.nps}</div>
                <div className="meta-line">TT Hit: {(match.last_ai_meta.tt_hit_rate * 100).toFixed(1)}%</div>
                <div className="meta-line">Time: {match.last_ai_meta.time_ms}ms</div>
              </div>
            )}

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
