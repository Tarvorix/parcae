import React from "react";
import { BOARD_HEIGHT, BOARD_WIDTH, FILES, coordToNotation } from "../coords";
import type { Piece } from "../types";
import "./Board.css";

type Props = {
  board: Record<string, Piece>;
  selected?: string | null;
  legalMoves: string[];
  onCellClick: (coord: string) => void;
};

function pieceGlyph(piece: Piece | undefined): string {
  if (!piece) return "";
  if (piece.kind === "dux") {
    return piece.color === "white" ? "♔" : "♚";
  }
  return piece.color === "white" ? "⛂" : "⛀";
}

export const Board: React.FC<Props> = ({
  board,
  selected,
  legalMoves,
  onCellClick,
}) => {
  const legalSet = new Set(legalMoves);
  const rows = [];
  for (let rank = BOARD_HEIGHT - 1; rank >= 0; rank -= 1) {
    const cells = [];
    for (let file = 0; file < BOARD_WIDTH; file += 1) {
      const coord = coordToNotation([file, rank]);
      const piece = board[coord];
      const isLight = (file + rank) % 2 === 0;
      const isSelected = selected === coord;
      const isLegal = legalSet.has(coord);
      cells.push(
        <button
          key={coord}
          className={`cell ${isLight ? "light" : "dark"} ${
            isSelected ? "selected" : ""
          } ${isLegal ? "legal" : ""}`}
          onClick={() => onCellClick(coord)}
          aria-label={`Cell ${coord}${piece ? ` with ${piece.kind} ${piece.color}` : ""}`}
        >
          <span className="coord">{coord}</span>
          <span className={`piece ${piece ? piece.color : ""}`}>
            {pieceGlyph(piece)}
          </span>
        </button>,
      );
    }
    rows.push(
      <div className="row" key={`rank-${rank}`}>
        <div className="rank-label">{rank + 1}</div>
        {cells}
      </div>,
    );
  }

  const fileLabels = [];
  for (let file = 0; file < BOARD_WIDTH; file += 1) {
    fileLabels.push(
      <div className="file-label" key={`file-${file}`}>
        {FILES[file]}
      </div>,
    );
  }

  return (
    <div className="board-wrapper">
      <div className="board">{rows}</div>
      <div className="file-row">
        <div className="file-spacer" />
        {fileLabels}
      </div>
    </div>
  );
};
