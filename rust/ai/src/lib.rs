//! Parcae AI: MCTS + optional ONNX policy/value inference.

use engine::{apply_move, legal_moves, GameState, Move, BOARD_HEIGHT, BOARD_WIDTH};
#[cfg(feature = "onnx")]
use engine::Color;
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::fmt;
use std::path::PathBuf;
use thiserror::Error;

#[cfg(feature = "onnx")]
use crate::onnx::OnnxEvaluator;
#[cfg(feature = "onnx")]
use std::path::Path;

#[derive(Debug, Error)]
pub enum AiError {
    #[error("no legal moves")]
    NoLegalMoves,
    #[error("onnx error: {0}")]
    Onnx(String),
}

const DIRECTIONS: [(i8, i8); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];

// Precompute all geometric moves on an empty board.
pub static ALL_MOVES: Lazy<Vec<Move>> = Lazy::new(|| {
    let mut moves = Vec::new();
    for file_idx in 0..BOARD_WIDTH {
        for rank_idx in 0..BOARD_HEIGHT {
            let origin = (file_idx, rank_idx);
            for (dx, dy) in DIRECTIONS {
                let mut step = 1;
                loop {
                    let nx = origin.0 as i32 + dx as i32 * step;
                    let ny = origin.1 as i32 + dy as i32 * step;
                    if nx < 0 || ny < 0 || nx >= BOARD_WIDTH as i32 || ny >= BOARD_HEIGHT as i32 {
                        break;
                    }
                    moves.push(Move {
                        origin,
                        target: (nx as u8, ny as u8),
                    });
                    step += 1;
                }
            }
        }
    }
    moves
});

pub static MOVE_TO_INDEX: Lazy<HashMap<(MoveCoord, MoveCoord), usize>> = Lazy::new(|| {
    let mut map = HashMap::new();
    for (idx, mv) in ALL_MOVES.iter().enumerate() {
        map.insert((mv.origin, mv.target), idx);
    }
    map
});

pub type MoveCoord = (u8, u8);

pub struct AIAgent {
    pub simulations: usize,
    pub c_puct: f32,
    #[cfg(feature = "onnx")]
    evaluator: Option<OnnxEvaluator>,
}

impl fmt::Debug for AIAgent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AIAgent")
            .field("simulations", &self.simulations)
            .field("c_puct", &self.c_puct)
            .field(
                "onnx_available",
                &{
                    #[cfg(feature = "onnx")]
                    {
                        self.evaluator.is_some()
                    }
                    #[cfg(not(feature = "onnx"))]
                    {
                        false
                    }
                },
            )
            .finish()
    }
}

impl Default for AIAgent {
    fn default() -> Self {
        Self::new(None::<PathBuf>, 64, 1.25)
    }
}

impl AIAgent {
    pub fn new(_model_path: Option<std::path::PathBuf>, simulations: usize, c_puct: f32) -> Self {
        #[cfg(feature = "onnx")]
        let evaluator = _model_path.as_ref().and_then(|p| {
            // ONNX Runtime dynamic loading can panic when the shared library is missing.
            // Recover to heuristic mode instead of crashing the whole server.
            match std::panic::catch_unwind(|| OnnxEvaluator::load(p.as_path())) {
                Ok(Ok(eval)) => Some(eval),
                Ok(Err(err)) => {
                    eprintln!(
                        "Failed to initialize ONNX evaluator from {}: {err}. Falling back to heuristic AI.",
                        p.display()
                    );
                    None
                }
                Err(_) => {
                    eprintln!(
                        "ONNX runtime initialization panicked for {}. Falling back to heuristic AI.",
                        p.display()
                    );
                    None
                }
            }
        });

        AIAgent {
            simulations,
            c_puct,
            #[cfg(feature = "onnx")]
            evaluator,
        }
    }

    pub fn available(&self) -> bool {
        #[cfg(feature = "onnx")]
        {
            self.evaluator.is_some()
        }
        #[cfg(not(feature = "onnx"))]
        {
            false
        }
    }

    pub fn select_move(&self, state: &GameState) -> Result<Option<Move>, AiError> {
        let legal = list_legal_moves(state);
        if legal.is_empty() {
            return Err(AiError::NoLegalMoves);
        }
        #[cfg(feature = "onnx")]
        {
            if self.evaluator.is_some() {
                return Ok(self.mcts_action(state));
            }
        }
        Ok(self.heuristic_move(state))
    }

    fn heuristic_move(&self, state: &GameState) -> Option<Move> {
        let legal = list_legal_moves(state);
        if legal.is_empty() {
            return None;
        }
        let mut best = legal[0];
        let mut best_score = f32::MIN;
        for mv in legal {
            let next_state = apply_move(state, mv).ok()?;
            let mut score = 0.0;
            if next_state.winner == Some(state.turn) {
                score += 1_000_000.0;
            }
            let gained = next_state.captures[&state.turn] as i32 - state.captures[&state.turn] as i32;
            score += gained as f32 * 100.0;
            score += legal_moves(&next_state, mv.target).len() as f32;
            if score > best_score {
                best_score = score;
                best = mv;
            }
        }
        Some(best)
    }

    #[cfg(feature = "onnx")]
    fn mcts_action(&self, state: &GameState) -> Option<Move> {
        let legal_root = list_legal_moves(state);
        if legal_root.is_empty() {
            return None;
        }
        let (policy, _value) = self.evaluate(state);
        let mut root = Node::new(state.clone(), state.turn, 1.0);
        root.expand(&policy, &legal_root);

        for _ in 0..self.simulations {
            let mut node = &mut root;
            let mut path: Vec<*mut Node> = vec![node];
            while node.expanded() {
                let (_idx, child_ptr) = node.select_child(self.c_puct);
                node = unsafe { &mut *child_ptr };
                path.push(node);
            }

            let leaf_value;
            if node.state.winner.is_some() {
                leaf_value = if node.state.winner == Some(node.state.turn) { 1.0 } else { -1.0 };
            } else {
                let legal = list_legal_moves(&node.state);
                if legal.is_empty() {
                    leaf_value = -1.0;
                } else {
                    let (p, v) = self.evaluate(&node.state);
                    node.expand(&p, &legal);
                    leaf_value = v;
                }
            }

            let mut value_to_backprop = leaf_value;
            for raw_ptr in path.into_iter().rev() {
                let n = unsafe { &mut *raw_ptr };
                n.visits += 1;
                n.value_sum += value_to_backprop;
                value_to_backprop = -value_to_backprop;
            }
        }

        let best = root
            .children
            .iter()
            .max_by_key(|(_, child)| child.visits)
            .map(|(idx, _)| *idx)?;
        Some(ALL_MOVES[best])
    }

    #[cfg(feature = "onnx")]
    fn evaluate(&self, state: &GameState) -> (Vec<f32>, f32) {
        if let Some(eval) = &self.evaluator {
            match eval.policy_value(state) {
                Ok((p, v)) => (p, v),
                Err(_) => (vec![1.0; ALL_MOVES.len()], 0.0),
            }
        } else {
            (vec![1.0; ALL_MOVES.len()], 0.0)
        }
    }
}

pub fn list_legal_moves(state: &GameState) -> Vec<Move> {
    let mut moves = Vec::new();
    for (coord, piece) in &state.board {
        if piece.color == state.turn {
            for target in legal_moves(state, *coord) {
                moves.push(Move {
                    origin: *coord,
                    target,
                });
            }
        }
    }
    moves
}

#[cfg(feature = "onnx")]
mod onnx {
    use super::*;
    use engine::{Color, GameState, PieceType};
    use ndarray;
    use ort::session::builder::GraphOptimizationLevel;
    use ort::session::Session;
    use ort::value::Tensor;
    use parking_lot::Mutex;

    pub struct OnnxEvaluator {
        session: Mutex<Session>,
    }

    impl OnnxEvaluator {
        pub fn load(path: &Path) -> anyhow::Result<Self> {
            let session = Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_intra_threads(1)?
                .commit_from_file(path)?;
            Ok(Self {
                session: Mutex::new(session),
            })
        }

        pub fn policy_value(&self, state: &GameState) -> anyhow::Result<(Vec<f32>, f32)> {
            let mut session = self.session.lock();
            let tensor = state_to_tensor(state);
            let input = Tensor::from_array(ndarray::Array::from_shape_vec(
                (1, 5, BOARD_HEIGHT as usize, BOARD_WIDTH as usize),
                tensor,
            )?)?;
            let inputs = ort::inputs![input];
            let outputs = session
                .run(inputs)
                .map_err(|e| anyhow::anyhow!(e.to_string()))?;

            let policy = outputs[0]
                .try_extract_tensor::<f32>()
                .map_err(|e| anyhow::anyhow!(e.to_string()))?;
            let value = outputs[1]
                .try_extract_tensor::<f32>()
                .map_err(|e| anyhow::anyhow!(e.to_string()))?;

            let policy_vec = policy.1.to_vec();
            let val = *value.1.first().unwrap_or(&0.0);
            Ok((policy_vec, val))
        }
    }

    fn state_to_tensor(state: &GameState) -> Vec<f32> {
        // Shape (C,H,W) flattened; channels: white soldiers, black soldiers, white dux, black dux, to-play.
        let plane_size = BOARD_HEIGHT as usize * BOARD_WIDTH as usize;
        let mut planes = vec![0.0f32; 5 * plane_size];
        for ((file, rank), piece) in &state.board {
            let idx = *rank as usize * BOARD_WIDTH as usize + *file as usize;
            match (piece.color, piece.kind) {
                (Color::White, PieceType::Soldier) => planes[idx] = 1.0,
                (Color::Black, PieceType::Soldier) => planes[plane_size + idx] = 1.0,
                (Color::White, PieceType::Dux) => planes[2 * plane_size + idx] = 1.0,
                (Color::Black, PieceType::Dux) => planes[3 * plane_size + idx] = 1.0,
            }
        }
        if state.turn == Color::White {
            for v in planes.iter_mut().skip(4 * plane_size) {
                *v = 1.0;
            }
        }
        planes
    }
}

// --- MCTS internals ---

#[cfg(feature = "onnx")]
#[derive(Debug, Clone)]
struct Node {
    state: GameState,
    _to_play: Color,
    prior: f32,
    visits: u32,
    value_sum: f32,
    children: HashMap<usize, Box<Node>>,
}

#[cfg(feature = "onnx")]
impl Node {
    fn new(state: GameState, to_play: Color, prior: f32) -> Self {
        Self {
            state,
            _to_play: to_play,
            prior,
            visits: 0,
            value_sum: 0.0,
            children: HashMap::new(),
        }
    }

    fn expanded(&self) -> bool {
        !self.children.is_empty()
    }

    fn q_value(&self) -> f32 {
        if self.visits == 0 {
            0.0
        } else {
            self.value_sum / self.visits as f32
        }
    }

    #[cfg(feature = "onnx")]
    fn expand(&mut self, policy_logits: &[f32], legal_moves: &[Move]) {
        self.children.clear();
        let mut probs = vec![0.0f32; legal_moves.len()];
        if policy_logits.len() == ALL_MOVES.len() {
            // mask
            let mut logits = Vec::with_capacity(legal_moves.len());
            for mv in legal_moves {
                let idx = MOVE_TO_INDEX[&(mv.origin, mv.target)];
                logits.push(policy_logits[idx]);
            }
            let softmaxed = softmax(&logits);
            probs.copy_from_slice(&softmaxed);
        } else {
            probs.fill(1.0 / legal_moves.len() as f32);
        }
        for (mv, prior) in legal_moves.iter().zip(probs.iter()) {
            let idx = MOVE_TO_INDEX[&(mv.origin, mv.target)];
            let next_state = apply_move(&self.state, *mv).expect("legal move");
            let to_play = next_state.turn;
            self.children.insert(
                idx,
                Box::new(Node::new(next_state, to_play, *prior)),
            );
        }
    }

    #[cfg(feature = "onnx")]
    fn select_child(&mut self, c_puct: f32) -> (usize, *mut Node) {
        let total_visits: f32 = self.children.values().map(|c| c.visits as f32).sum();
        let (idx, child) = self
            .children
            .iter_mut()
            .max_by(|a, b| {
                let ua = ucb(a.1, c_puct, total_visits);
                let ub = ucb(b.1, c_puct, total_visits);
                ua.partial_cmp(&ub).unwrap()
            })
            .expect("child exists");
        (*idx, &mut **child)
    }
}

#[cfg(feature = "onnx")]
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    exp.iter().map(|v| v / sum).collect()
}

#[cfg(feature = "onnx")]
fn ucb(child: &Node, c_puct: f32, total_visits: f32) -> f32 {
    // Child Q is from child-to-play perspective, so negate for parent choice.
    let q = -child.q_value();
    let u = c_puct * child.prior * (total_visits + 1e-8).sqrt() / (1.0 + child.visits as f32);
    q + u
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;
    use engine::initial_state;

    #[test]
    fn heuristic_returns_move() {
        let state = initial_state(5).unwrap();
        let agent = AIAgent::new(None::<PathBuf>, 8, 1.25);
        let mv = agent.select_move(&state).unwrap();
        assert!(mv.is_some());
    }

    #[test]
    fn legal_move_generation_matches_engine() {
        let state = initial_state(5).unwrap();
        let moves = list_legal_moves(&state);
        assert!(moves.iter().all(|mv| mv.origin != mv.target));
    }

    #[test]
    fn move_indexing_covers_all_targets() {
        let state = initial_state(5).unwrap();
        let moves = list_legal_moves(&state);
        for mv in moves {
            let idx = MOVE_TO_INDEX.get(&(mv.origin, mv.target)).copied();
            assert!(idx.is_some());
            assert!(idx.unwrap() < ALL_MOVES.len());
        }
    }
}
