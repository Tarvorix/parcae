import type { BoardEntry, Piece } from "./types";

export const FILES = "ABCDEFGHIJKL";
export const BOARD_WIDTH = 12;
export const BOARD_HEIGHT = 8;
const DIRECTIONS: Array<[number, number]> = [
  [1, 0],
  [-1, 0],
  [0, 1],
  [0, -1],
];

export function notationToCoord(token: string): [number, number] {
  const fileChar = token[0]?.toUpperCase();
  const rankStr = token.slice(1);
  const fileIdx = FILES.indexOf(fileChar);
  const rank = Number(rankStr) - 1;
  if (fileIdx < 0 || Number.isNaN(rank)) {
    throw new Error(`Invalid coordinate: ${token}`);
  }
  return [fileIdx, rank];
}

export function coordToNotation([file, rank]: [number, number]): string {
  return `${FILES[file]}${rank + 1}`;
}

export function boardToMap(board: BoardEntry[]): Record<string, Piece> {
  return board.reduce<Record<string, Piece>>((acc, entry) => {
    acc[entry.coord] = entry.piece;
    return acc;
  }, {});
}

export function legalMovesClient(
  board: Record<string, Piece>,
  origin: string,
): string[] {
  const [ox, oy] = notationToCoord(origin);
  const piece = board[origin];
  if (!piece) {
    return [];
  }
  const targets: string[] = [];
  for (const [dx, dy] of DIRECTIONS) {
    let step = 1;
    while (true) {
      const nx = ox + dx * step;
      const ny = oy + dy * step;
      if (nx < 0 || nx >= BOARD_WIDTH || ny < 0 || ny >= BOARD_HEIGHT) {
        break;
      }
      const target = coordToNotation([nx, ny]);
      if (board[target]) {
        break;
      }
      targets.push(target);
      step += 1;
    }
  }
  return targets;
}
