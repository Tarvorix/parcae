//! Core Parcae Strategy (Kowalski/Stanway) rules engine.
//! Shared by native server and WASM UI for identical move logic.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

pub const BOARD_WIDTH: u8 = 12;
pub const BOARD_HEIGHT: u8 = 8;
pub const FILES: &str = "ABCDEFGHIJKL";

pub type Coord = (u8, u8); // (file, rank) zero-based

const DIRECTIONS: [(i8, i8); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
const CORNERS: [Coord; 4] = [
    (0, 0),
    (BOARD_WIDTH - 1, 0),
    (0, BOARD_HEIGHT - 1),
    (BOARD_WIDTH - 1, BOARD_HEIGHT - 1),
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Color {
    White,
    Black,
}

impl Color {
    pub fn opponent(self) -> Color {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PieceType {
    Soldier,
    Dux,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Piece {
    pub color: Color,
    pub kind: PieceType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Move {
    pub origin: Coord,
    pub target: Coord,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameState {
    pub board: HashMap<Coord, Piece>,
    pub turn: Color,
    pub winner: Option<Color>,
    pub captures: HashMap<Color, u32>,
    pub history: Vec<Move>,
    pub summary: Option<String>,
}

impl Default for GameState {
    fn default() -> Self {
        Self {
            board: HashMap::new(),
            turn: Color::White,
            winner: None,
            captures: HashMap::from([(Color::White, 0), (Color::Black, 0)]),
            history: Vec::new(),
            summary: None,
        }
    }
}

impl GameState {
    pub fn active(&self) -> bool {
        self.winner.is_none()
    }
}

#[derive(Debug, Error)]
pub enum EngineError {
    #[error("invalid coordinate token: {0}")]
    InvalidCoord(String),
    #[error("out of bounds coordinate: {0}")]
    OutOfBounds(String),
    #[error("invalid dux file index")]
    InvalidDuxFile,
    #[error("game already finished")]
    Finished,
    #[error("no piece at origin")]
    NoPiece,
    #[error("not this piece's turn")]
    WrongTurn,
    #[error("illegal move target")]
    IllegalMove,
}

pub fn coord_in_bounds(coord: Coord) -> bool {
    let (file_idx, rank_idx) = coord;
    file_idx < BOARD_WIDTH && rank_idx < BOARD_HEIGHT
}

pub fn coord_to_notation(coord: Coord) -> String {
    let (file_idx, rank_idx) = coord;
    let file_char = FILES
        .chars()
        .nth(file_idx as usize)
        .expect("file index in range");
    format!("{file_char}{}", rank_idx + 1)
}

pub fn notation_to_coord(token: &str) -> Result<Coord, EngineError> {
    if token.len() < 2 {
        return Err(EngineError::InvalidCoord(token.to_string()));
    }
    let (file_part, rank_part) = token.split_at(1);
    let file_char = file_part.chars().next().unwrap().to_ascii_uppercase();
    let file_idx = FILES
        .chars()
        .position(|c| c == file_char)
        .ok_or_else(|| EngineError::InvalidCoord(token.to_string()))?;
    let rank: i32 = rank_part
        .parse()
        .map_err(|_| EngineError::InvalidCoord(token.to_string()))?;
    let coord = (file_idx as u8, (rank - 1) as u8);
    if !coord_in_bounds(coord) {
        return Err(EngineError::OutOfBounds(token.to_string()));
    }
    Ok(coord)
}

pub fn initial_state(dux_file: u8) -> Result<GameState, EngineError> {
    if dux_file >= BOARD_WIDTH {
        return Err(EngineError::InvalidDuxFile);
    }
    let mut board = HashMap::new();
    for file_idx in 0..BOARD_WIDTH {
        board.insert(
            (file_idx, 0),
            Piece {
                color: Color::White,
                kind: if file_idx == dux_file {
                    PieceType::Dux
                } else {
                    PieceType::Soldier
                },
            },
        );
        board.insert(
            (file_idx, BOARD_HEIGHT - 1),
            Piece {
                color: Color::Black,
                kind: if file_idx == dux_file {
                    PieceType::Dux
                } else {
                    PieceType::Soldier
                },
            },
        );
    }
    Ok(GameState {
        board,
        ..GameState::default()
    })
}

pub fn legal_moves(state: &GameState, origin: Coord) -> Vec<Coord> {
    let piece = match state.board.get(&origin) {
        Some(p) => p,
        None => return Vec::new(),
    };
    let mut moves = Vec::new();
    for (dx, dy) in DIRECTIONS {
        let mut step: i32 = 1;
        loop {
            let nx = origin.0 as i32 + dx as i32 * step;
            let ny = origin.1 as i32 + dy as i32 * step;
            if nx < 0 || ny < 0 || nx >= BOARD_WIDTH as i32 || ny >= BOARD_HEIGHT as i32 {
                break;
            }
            let target = (nx as u8, ny as u8);
            if state.board.get(&target).is_some() {
                break;
            }
            moves.push(target);
            step += 1;
        }
    }
    // Keep API parity; in this variant all pieces share movement so piece unused.
    let _ = piece;
    moves
}

pub fn apply_move(state: &GameState, mv: Move) -> Result<GameState, EngineError> {
    if !state.active() {
        return Err(EngineError::Finished);
    }
    let piece = state.board.get(&mv.origin).ok_or(EngineError::NoPiece)?;
    if piece.color != state.turn {
        return Err(EngineError::WrongTurn);
    }
    if !legal_moves(state, mv.origin).contains(&mv.target) {
        return Err(EngineError::IllegalMove);
    }

    let mut board = state.board.clone();
    board.remove(&mv.origin);
    board.insert(mv.target, *piece);

    let captured_positions = resolve_captures(&board, piece.color, mv.target);
    for pos in &captured_positions {
        board.remove(pos);
    }

    let mut captures = state.captures.clone();
    let entry = captures.entry(piece.color).or_insert(0);
    *entry += captured_positions.len() as u32;

    let mut history = state.history.clone();
    history.push(mv);

    let next_turn = piece.color.opponent();
    let (winner, summary) = determine_winner(&board, piece.color, next_turn, &captures);

    Ok(GameState {
        board,
        turn: if winner.is_none() { next_turn } else { next_turn },
        winner,
        captures,
        history,
        summary,
    })
}

fn resolve_captures(board: &HashMap<Coord, Piece>, mover: Color, moved_to: Coord) -> Vec<Coord> {
    let mut captured = Vec::new();
    for (dx, dy) in DIRECTIONS {
        let adj_x = moved_to.0 as i32 + dx as i32;
        let adj_y = moved_to.1 as i32 + dy as i32;
        if adj_x < 0 || adj_y < 0 || adj_x >= BOARD_WIDTH as i32 || adj_y >= BOARD_HEIGHT as i32 {
            continue;
        }
        let adj = (adj_x as u8, adj_y as u8);
        let target_piece = match board.get(&adj) {
            Some(p) if p.color != mover => p,
            _ => continue,
        };
        let beyond_x = adj_x + dx as i32;
        let beyond_y = adj_y + dy as i32;
        if beyond_x >= 0
            && beyond_y >= 0
            && beyond_x < BOARD_WIDTH as i32
            && beyond_y < BOARD_HEIGHT as i32
        {
            let beyond = (beyond_x as u8, beyond_y as u8);
            if let Some(guardian) = board.get(&beyond) {
                if guardian.color == mover && target_piece.kind == PieceType::Soldier {
                    captured.push(adj);
                }
            }
        }
    }

    for &corner in &CORNERS {
        if let Some(occupant) = board.get(&corner) {
            if occupant.color != mover && occupant.kind == PieceType::Soldier {
                let adjacents: Vec<Coord> = DIRECTIONS
                    .iter()
                    .filter_map(|(dx, dy)| {
                        let nx = corner.0 as i16 + *dx as i16;
                        let ny = corner.1 as i16 + *dy as i16;
                        (nx >= 0
                            && ny >= 0
                            && nx < BOARD_WIDTH as i16
                            && ny < BOARD_HEIGHT as i16)
                            .then_some((nx as u8, ny as u8))
                    })
                    .collect();
                if adjacents
                    .iter()
                    .all(|c| board.get(c).map(|p| p.color == mover).unwrap_or(false))
                {
                    captured.push(corner);
                }
            }
        }
    }
    captured
}

fn find_piece(board: &HashMap<Coord, Piece>, color: Color, kind: PieceType) -> Option<Coord> {
    board
        .iter()
        .find_map(|(coord, piece)| (piece.color == color && piece.kind == kind).then_some(*coord))
}

fn determine_winner(
    board: &HashMap<Coord, Piece>,
    mover: Color,
    _next_turn: Color,
    captures: &HashMap<Color, u32>,
) -> (Option<Color>, Option<String>) {
    let opponent = mover.opponent();
    if let Some(opponent_dux) = find_piece(board, opponent, PieceType::Dux) {
        if dux_immobilized_by_enemy(board, opponent_dux, mover) {
            return (
                Some(mover),
                Some("Opponent Dux immobilized.".to_string()),
            );
        }
    }

    if !any_legal_moves(board, opponent) {
        return (
            Some(mover),
            Some("Opponent has no legal moves.".to_string()),
        );
    }

    let opponent_soldiers = board
        .values()
        .filter(|p| p.color == opponent && p.kind == PieceType::Soldier)
        .count();
    if opponent_soldiers == 0 {
        return (
            Some(mover),
            Some("All opposing soldiers removed (material victory).".to_string()),
        );
    }

    // Unused captures except summarized; return None
    let _ = captures;
    (None, None)
}

fn dux_immobilized_by_enemy(board: &HashMap<Coord, Piece>, dux: Coord, enemy: Color) -> bool {
    // A Dux is considered immobilized for victory only when every orthogonal
    // adjacency is either board edge or occupied by an enemy piece.
    // Friendly blockers do not count as an immobilization victory.
    for (dx, dy) in DIRECTIONS {
        let nx = dux.0 as i16 + dx as i16;
        let ny = dux.1 as i16 + dy as i16;
        if nx < 0 || ny < 0 || nx >= BOARD_WIDTH as i16 || ny >= BOARD_HEIGHT as i16 {
            continue;
        }
        let adj = (nx as u8, ny as u8);
        match board.get(&adj) {
            Some(p) if p.color == enemy => {}
            _ => return false,
        }
    }
    true
}

pub fn any_legal_moves(board: &HashMap<Coord, Piece>, color: Color) -> bool {
    board.iter().any(|(coord, piece)| {
        if piece.color == color {
            let dummy = GameState {
                board: board.clone(),
                ..GameState::default()
            };
            !legal_moves(&dummy, *coord).is_empty()
        } else {
            false
        }
    })
}

// --- Serialization ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedMove {
    pub origin: String,
    pub target: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedBoardEntry {
    pub coord: String,
    pub piece: Piece,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedState {
    pub turn: Color,
    pub winner: Option<Color>,
    pub captures: HashMap<Color, u32>,
    pub history: Vec<SerializedMove>,
    pub board: Vec<SerializedBoardEntry>,
    pub summary: Option<String>,
}

pub fn serialize_state(state: &GameState) -> SerializedState {
    let mut board_entries: Vec<_> = state
        .board
        .iter()
        .map(|(coord, piece)| SerializedBoardEntry {
            coord: coord_to_notation(*coord),
            piece: *piece,
        })
        .collect();
    board_entries.sort_by_key(|b| notation_to_coord(&b.coord).ok());

    SerializedState {
        turn: state.turn,
        winner: state.winner,
        captures: state.captures.clone(),
        history: state
            .history
            .iter()
            .map(|m| SerializedMove {
                origin: coord_to_notation(m.origin),
                target: coord_to_notation(m.target),
            })
            .collect(),
        board: board_entries,
        summary: state.summary.clone(),
    }
}

pub fn deserialize_state(payload: &SerializedState) -> Result<GameState, EngineError> {
    let mut board = HashMap::new();
    for entry in &payload.board {
        let coord = notation_to_coord(&entry.coord)?;
        board.insert(coord, entry.piece);
    }
    let mut captures = HashMap::new();
    captures.insert(Color::White, *payload.captures.get(&Color::White).unwrap_or(&0));
    captures.insert(Color::Black, *payload.captures.get(&Color::Black).unwrap_or(&0));

    let history = payload
        .history
        .iter()
        .map(|m| {
            Ok(Move {
                origin: notation_to_coord(&m.origin)?,
                target: notation_to_coord(&m.target)?,
            })
        })
        .collect::<Result<Vec<_>, EngineError>>()?;

    Ok(GameState {
        board,
        turn: payload.turn,
        winner: payload.winner,
        captures,
        history,
        summary: payload.summary.clone(),
    })
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_state_counts() {
        let state = initial_state(5).unwrap();
        assert_eq!(state.board.len(), 24);
        let white_pieces = state
            .board
            .values()
            .filter(|p| p.color == Color::White)
            .count();
        let black_pieces = state
            .board
            .values()
            .filter(|p| p.color == Color::Black)
            .count();
        assert_eq!(white_pieces, 12);
        assert_eq!(black_pieces, 12);
        assert_eq!(state.turn, Color::White);
    }

    #[test]
    fn dux_has_clear_vertical_moves_from_start() {
        let state = initial_state(5).unwrap();
        let dux_origin = (5, 0);
        let targets = legal_moves(&state, dux_origin);
        assert_eq!(targets.len() as u8, BOARD_HEIGHT - 2);
        assert!(targets.contains(&(5, BOARD_HEIGHT - 2)));
    }

    #[test]
    fn custodial_capture_single() {
        let mut board = HashMap::new();
        board.insert((0, 0), Piece { color: Color::White, kind: PieceType::Soldier });
        board.insert((1, 0), Piece { color: Color::Black, kind: PieceType::Soldier });
        board.insert((3, 0), Piece { color: Color::White, kind: PieceType::Soldier });
        let state = GameState { board, turn: Color::White, ..GameState::default() };
        let new_state = apply_move(&state, Move { origin: (3, 0), target: (2, 0) }).unwrap();
        assert!(!new_state.board.contains_key(&(1, 0)));
        assert_eq!(new_state.captures[&Color::White], 1);
    }

    #[test]
    fn custodial_double_capture() {
        let mut board = HashMap::new();
        board.insert((0, 0), Piece { color: Color::White, kind: PieceType::Soldier });
        board.insert((1, 0), Piece { color: Color::Black, kind: PieceType::Soldier });
        board.insert((3, 0), Piece { color: Color::Black, kind: PieceType::Soldier });
        board.insert((4, 0), Piece { color: Color::White, kind: PieceType::Soldier });
        board.insert((2, 3), Piece { color: Color::White, kind: PieceType::Soldier });
        let state = GameState { board, turn: Color::White, ..GameState::default() };
        let new_state = apply_move(&state, Move { origin: (2, 3), target: (2, 0) }).unwrap();
        assert!(!new_state.board.contains_key(&(1, 0)));
        assert!(!new_state.board.contains_key(&(3, 0)));
        assert_eq!(new_state.captures[&Color::White], 2);
    }

    #[test]
    fn corner_capture() {
        let mut board = HashMap::new();
        board.insert((0, 0), Piece { color: Color::Black, kind: PieceType::Soldier });
        board.insert((0, 1), Piece { color: Color::White, kind: PieceType::Soldier });
        board.insert((1, 3), Piece { color: Color::White, kind: PieceType::Soldier });
        let state = GameState { board, turn: Color::White, ..GameState::default() };
        let new_state = apply_move(&state, Move { origin: (1, 3), target: (1, 0) }).unwrap();
        assert!(!new_state.board.contains_key(&(0, 0)));
        assert_eq!(new_state.captures[&Color::White], 1);
    }

    #[test]
    fn suicide_move_not_captured() {
        let mut board = HashMap::new();
        board.insert((1, 0), Piece { color: Color::Black, kind: PieceType::Soldier });
        board.insert((3, 0), Piece { color: Color::Black, kind: PieceType::Soldier });
        board.insert((2, 3), Piece { color: Color::White, kind: PieceType::Soldier });
        let state = GameState { board, turn: Color::White, ..GameState::default() };
        let new_state = apply_move(&state, Move { origin: (2, 3), target: (2, 0) }).unwrap();
        assert!(new_state.board.contains_key(&(2, 0)));
        assert_eq!(new_state.board.get(&(2, 0)).unwrap().color, Color::White);
    }

    #[test]
    fn dux_immobilization_is_victory() {
        let mut board = HashMap::new();
        board.insert((0, 0), Piece { color: Color::Black, kind: PieceType::Dux });
        board.insert((0, 2), Piece { color: Color::White, kind: PieceType::Soldier });
        board.insert((1, 0), Piece { color: Color::White, kind: PieceType::Soldier });
        let state = GameState { board, turn: Color::White, ..GameState::default() };
        let result = apply_move(&state, Move { origin: (0, 2), target: (0, 1) }).unwrap();
        assert_eq!(result.winner, Some(Color::White));
        assert!(result.summary.as_ref().unwrap().contains("Dux"));
    }

    #[test]
    fn dux_blocked_by_own_side_is_not_immobilization_loss() {
        let state = initial_state(5).unwrap();
        let result = apply_move(&state, Move { origin: (5, 0), target: (5, 6) }).unwrap();
        assert_eq!(result.winner, None);
        assert_eq!(result.turn, Color::Black);
    }
}
