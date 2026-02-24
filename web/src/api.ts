import type { MatchMode, MatchResponse } from "./types";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

async function handle<T>(promise: Promise<Response>): Promise<T> {
  const res = await promise;
  if (!res.ok) {
    const message = await res.text();
    throw new Error(message || "Request failed");
  }
  return res.json() as Promise<T>;
}

export function createMatch(mode: MatchMode = "pva"): Promise<MatchResponse> {
  return handle(
    fetch(`${API_BASE}/match`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode }),
    }),
  );
}

export function fetchMatch(matchId: string): Promise<MatchResponse> {
  return handle(fetch(`${API_BASE}/match/${matchId}`));
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
