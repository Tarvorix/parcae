import type {
  AiCapabilities,
  AgentStatusResponse,
  MatchAiProfiles,
  MatchMode,
  MatchResponse,
} from "./types";

const API_BASE = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

async function handle<T>(promise: Promise<Response>): Promise<T> {
  const res = await promise;
  if (!res.ok) {
    let message = "Request failed";
    try {
      const payload = await res.json();
      if (typeof payload?.detail === "string" && payload.detail.length > 0) {
        message = payload.detail;
      } else {
        message = JSON.stringify(payload);
      }
    } catch {
      const text = await res.text();
      if (text.length > 0) {
        message = text;
      }
    }
    throw new Error(message);
  }
  return res.json() as Promise<T>;
}

export function createMatch(
  mode: MatchMode = "pva",
  aiProfiles?: MatchAiProfiles,
): Promise<MatchResponse> {
  const payload: { mode: MatchMode; ai_profiles?: MatchAiProfiles } = { mode };
  if (aiProfiles && (aiProfiles.white || aiProfiles.black)) {
    payload.ai_profiles = aiProfiles;
  }
  return handle(
    fetch(`${API_BASE}/match`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  );
}

export function fetchAiCapabilities(): Promise<AiCapabilities> {
  return handle(fetch(`${API_BASE}/ai/capabilities`));
}

export function fetchAgentStatus(): Promise<AgentStatusResponse> {
  return handle(fetch(`${API_BASE}/agent/status`));
}

export function fetchMatch(matchId: string): Promise<MatchResponse> {
  return handle(fetch(`${API_BASE}/match/${matchId}`));
}

export function fetchLegalMoves(
  matchId: string,
  origin?: string,
): Promise<{ moves: Array<{ origin: string; target: string }> }> {
  const query = origin ? `?origin=${encodeURIComponent(origin)}` : "";
  return handle(fetch(`${API_BASE}/match/${matchId}/legal${query}`));
}

export function postMove(
  matchId: string,
  origin: string,
  target: string,
): Promise<MatchResponse> {
  return handle(
    fetch(`${API_BASE}/match/${matchId}/move`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ origin, target }),
    }),
  );
}

export function stepAi(matchId: string): Promise<MatchResponse> {
  return handle(
    fetch(`${API_BASE}/match/${matchId}/ai-step`, {
      method: "POST",
    }),
  );
}
