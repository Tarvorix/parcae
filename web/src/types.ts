export type Color = "white" | "black";
export type PieceType = "soldier" | "dux";
export type Controller = "player" | "agent" | "ai";
export type AiBackend = "heuristic" | "centurion" | "alphazero" | "stockfish";
export type AiMoveSource = "heuristic" | "search" | "book" | "tb";

export type Piece = {
  color: Color;
  kind: PieceType;
};

export type BoardEntry = {
  coord: string;
  piece: Piece;
};

export type Move = {
  origin: string;
  target: string;
};

export type GameState = {
  turn: Color;
  winner: Color | null;
  captures: Record<Color, number>;
  history: Move[];
  board: BoardEntry[];
  summary?: string | null;
};

export type MatchMode = "pva" | "ava" | "pvg" | "gvg" | "gva";

export type AiProfile = {
  backend?: AiBackend;
  move_time_ms?: number;
  max_depth?: number;
  hash_mb?: number;
  use_book?: boolean;
  use_tb?: boolean;
  skill?: number;
};

export type MatchAiProfiles = {
  white?: AiProfile;
  black?: AiProfile;
};

export type AiMoveMeta = {
  backend: AiBackend;
  depth_reached: number;
  nodes: number;
  nps: number;
  tt_hit_rate: number;
  time_ms: number;
  source: AiMoveSource;
};

export type MatchResponse = {
  id: string;
  mode: MatchMode;
  players: Record<Color, Controller>;
  state: GameState;
  ai_profiles?: MatchAiProfiles;
  last_ai_meta?: AiMoveMeta | null;
};

export type AiCapabilities = {
  supported_backends: AiBackend[];
  default_backend: AiBackend;
  onnx_available: boolean;
  centurion_book_loaded: boolean;
  centurion_tb_loaded: boolean;
  centurion_nnue_loaded: boolean;
  default_profile: AiProfile;
};

export type AgentStatusEntry = {
  agent_id: string;
  connected: boolean;
  last_seen_unix_ms: number;
};

export type AgentStatusResponse = {
  connected: boolean;
  active_count: number;
  stale_after_seconds: number;
  agents: AgentStatusEntry[];
};
