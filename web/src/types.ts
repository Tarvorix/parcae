export type Color = "white" | "black";
export type PieceType = "soldier" | "dux";
export type Controller = "player" | "agent" | "ai";

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

export type MatchResponse = {
  id: string;
  mode: MatchMode;
  players: Record<Color, Controller>;
  state: GameState;
};
