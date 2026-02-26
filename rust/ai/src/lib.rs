//! Parcae AI runtime: heuristic, AlphaZero (ONNX+MCTS), and Centurion (alpha-beta).

use engine::{
    apply_move, legal_moves, Color, Coord, GameState, Move, PieceType, BOARD_HEIGHT, BOARD_WIDTH,
};
use once_cell::sync::Lazy;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;

#[cfg(feature = "onnx")]
use crate::onnx::OnnxEvaluator;
#[cfg(feature = "onnx")]
use std::path::Path as StdPath;

#[derive(Debug, Error)]
pub enum AiError {
    #[error("no legal moves")]
    NoLegalMoves,
    #[error("alpha-zero model unavailable")]
    ModelUnavailable,
    #[error("invalid ai profile: {0}")]
    InvalidProfile(String),
    #[error("io error: {0}")]
    Io(String),
    #[error("search aborted")]
    SearchAborted,
}

const DIRECTIONS: [(i8, i8); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
const MATE_SCORE: i32 = 30_000;
const MAX_SEARCH_PLY: usize = 96;

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

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum AiBackend {
    Heuristic,
    Alphazero,
    #[serde(rename = "centurion", alias = "stockfish")]
    Centurion,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum AiMoveSource {
    Heuristic,
    Search,
    Book,
    Tb,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiProfile {
    #[serde(default = "default_backend")]
    pub backend: AiBackend,
    #[serde(default = "default_move_time_ms")]
    pub move_time_ms: u64,
    #[serde(default = "default_max_depth")]
    pub max_depth: u8,
    #[serde(default = "default_hash_mb")]
    pub hash_mb: usize,
    #[serde(default = "default_use_book")]
    pub use_book: bool,
    #[serde(default = "default_use_tb")]
    pub use_tb: bool,
    #[serde(default = "default_skill")]
    pub skill: u8,
}

fn default_backend() -> AiBackend {
    AiBackend::Heuristic
}
fn default_move_time_ms() -> u64 {
    500
}
fn default_max_depth() -> u8 {
    32
}
fn default_hash_mb() -> usize {
    128
}
fn default_use_book() -> bool {
    true
}
fn default_use_tb() -> bool {
    true
}
fn default_skill() -> u8 {
    12
}

impl Default for AiProfile {
    fn default() -> Self {
        Self {
            backend: default_backend(),
            move_time_ms: default_move_time_ms(),
            max_depth: default_max_depth(),
            hash_mb: default_hash_mb(),
            use_book: default_use_book(),
            use_tb: default_use_tb(),
            skill: default_skill(),
        }
    }
}

impl AiProfile {
    pub fn normalized_with_default(&self, defaults: &AiProfile) -> Self {
        let mut out = self.clone();
        if out.move_time_ms == 0 {
            out.move_time_ms = defaults.move_time_ms;
        }
        if out.max_depth == 0 {
            out.max_depth = defaults.max_depth;
        }
        if out.hash_mb == 0 {
            out.hash_mb = defaults.hash_mb;
        }
        out.skill = out.skill.clamp(1, 20);
        out
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiMoveMeta {
    pub backend: AiBackend,
    pub depth_reached: u8,
    pub nodes: u64,
    pub nps: u64,
    pub tt_hit_rate: f32,
    pub time_ms: u64,
    pub source: AiMoveSource,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiSelection {
    pub mv: Option<Move>,
    pub meta: AiMoveMeta,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiCapabilities {
    pub supported_backends: Vec<AiBackend>,
    pub default_backend: AiBackend,
    pub onnx_available: bool,
    pub centurion_book_loaded: bool,
    pub centurion_tb_loaded: bool,
    pub centurion_nnue_loaded: bool,
    pub default_profile: AiProfile,
}

pub trait MoveSelector {
    fn select_move(
        &self,
        state: &GameState,
        profile: &AiProfile,
        deadline: Instant,
    ) -> Result<AiSelection, AiError>;
}

pub struct AIAgent {
    default_profile: AiProfile,
    heuristic: HeuristicAgent,
    alphazero: AlphaZeroAgent,
    centurion: CenturionAgent,
}

impl fmt::Debug for AIAgent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AIAgent")
            .field("default_profile", &self.default_profile)
            .field("alphazero_available", &self.alphazero.available())
            .field(
                "centurion_book_loaded",
                &self.centurion.opening_book.is_some(),
            )
            .field("centurion_tb_loaded", &self.centurion.tablebase.is_some())
            .field("centurion_nnue_loaded", &self.centurion.nnue.is_some())
            .finish()
    }
}

impl Default for AIAgent {
    fn default() -> Self {
        Self::new(None::<PathBuf>, 64, 1.25)
    }
}

impl AIAgent {
    pub fn new(model_path: Option<PathBuf>, simulations: usize, c_puct: f32) -> Self {
        let default_backend = parse_default_backend(
            std::env::var("PARCAE_AI_DEFAULT_BACKEND")
                .unwrap_or_else(|_| "heuristic".to_string())
                .as_str(),
        );
        let mut default_profile = AiProfile::default();
        default_profile.backend = default_backend;
        default_profile.move_time_ms = std::env::var("PARCAE_STOCKFISH_DEFAULT_TIME_MS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(default_profile.move_time_ms)
            .max(1);
        default_profile.hash_mb = std::env::var("PARCAE_STOCKFISH_DEFAULT_HASH_MB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(default_profile.hash_mb)
            .max(1);

        let model_path =
            model_path.or_else(|| std::env::var("PARCAE_MODEL_PATH").ok().map(PathBuf::from));
        let book_path = std::env::var("PARCAE_BOOK_PATH")
            .ok()
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("models/rust/book/centurion_book_v1.bin"));
        let tb_path = std::env::var("PARCAE_TB_PATH")
            .ok()
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("models/rust/tablebases/centurion_tb_4pc_v1.bin"));
        let nnue_path = std::env::var("PARCAE_NNUE_PATH")
            .ok()
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("models/rust/nnue/centurion_nnue_v1.bin"));

        let opening_book = OpeningBook::load(&book_path).ok();
        let tablebase = Tablebase::load(&tb_path).ok();
        let nnue = load_nnue_weights(&nnue_path)
            .ok()
            .map(|weights| NnueEvaluator::new(Arc::new(weights)));

        Self {
            default_profile,
            heuristic: HeuristicAgent,
            alphazero: AlphaZeroAgent::new(model_path, simulations, c_puct),
            centurion: CenturionAgent {
                opening_book,
                tablebase,
                nnue,
            },
        }
    }

    pub fn default_profile(&self) -> AiProfile {
        self.default_profile.clone()
    }

    pub fn available(&self) -> bool {
        self.alphazero.available()
    }

    pub fn capabilities(&self) -> AiCapabilities {
        AiCapabilities {
            supported_backends: vec![
                AiBackend::Heuristic,
                AiBackend::Centurion,
                AiBackend::Alphazero,
            ],
            default_backend: self.default_profile.backend,
            onnx_available: self.alphazero.available(),
            centurion_book_loaded: self.centurion.opening_book.is_some(),
            centurion_tb_loaded: self.centurion.tablebase.is_some(),
            centurion_nnue_loaded: self.centurion.nnue.is_some(),
            default_profile: self.default_profile.clone(),
        }
    }

    pub fn select_move(&self, state: &GameState) -> Result<Option<Move>, AiError> {
        let selection = self.select_move_with_profile(state, &self.default_profile)?;
        Ok(selection.mv)
    }

    pub fn select_move_with_profile(
        &self,
        state: &GameState,
        requested_profile: &AiProfile,
    ) -> Result<AiSelection, AiError> {
        let profile = requested_profile.normalized_with_default(&self.default_profile);
        let now = Instant::now();
        let effective_time_ms = scaled_move_time_ms(profile.move_time_ms, profile.skill);
        let deadline = now + Duration::from_millis(effective_time_ms);

        match profile.backend {
            AiBackend::Heuristic => self.heuristic.select_move(state, &profile, deadline),
            AiBackend::Centurion => self.centurion.select_move(state, &profile, deadline),
            AiBackend::Alphazero => {
                if self.alphazero.available() {
                    self.alphazero.select_move(state, &profile, deadline)
                } else {
                    Err(AiError::ModelUnavailable)
                }
            }
        }
    }
}

fn parse_default_backend(token: &str) -> AiBackend {
    match token.to_ascii_lowercase().as_str() {
        "alphazero" | "alpha_zero" | "az" => AiBackend::Alphazero,
        "centurion" | "stockfish" => AiBackend::Centurion,
        _ => AiBackend::Heuristic,
    }
}

fn scaled_move_time_ms(base_ms: u64, skill: u8) -> u64 {
    let base = base_ms.max(1) as f32;
    let mult = 0.35 + (skill.clamp(1, 20) as f32 / 20.0) * 1.05;
    (base * mult).round().max(1.0) as u64
}

fn effective_max_depth(profile: &AiProfile) -> u8 {
    let scale = 0.45 + (profile.skill.clamp(1, 20) as f32 / 20.0) * 0.95;
    ((profile.max_depth as f32 * scale).round() as u8).clamp(2, profile.max_depth.max(2))
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

fn move_index(mv: Move) -> Option<usize> {
    MOVE_TO_INDEX.get(&(mv.origin, mv.target)).copied()
}

fn move_from_index(idx: usize) -> Option<Move> {
    ALL_MOVES.get(idx).copied()
}

pub fn move_to_index(mv: Move) -> Option<usize> {
    move_index(mv)
}

pub fn move_from_index_value(idx: usize) -> Option<Move> {
    move_from_index(idx)
}

fn move_capture_gain(state: &GameState, mv: Move) -> i32 {
    if let Ok(next) = apply_move(state, mv) {
        next.captures[&state.turn] as i32 - state.captures[&state.turn] as i32
    } else {
        0
    }
}

fn is_winning_move(state: &GameState, mv: Move) -> bool {
    apply_move(state, mv)
        .map(|n| n.winner == Some(state.turn))
        .unwrap_or(false)
}

// ---------------- Heuristic backend ----------------

struct HeuristicAgent;

impl MoveSelector for HeuristicAgent {
    fn select_move(
        &self,
        state: &GameState,
        _profile: &AiProfile,
        _deadline: Instant,
    ) -> Result<AiSelection, AiError> {
        let started = Instant::now();
        let mv = heuristic_move(state);
        let elapsed = started.elapsed();
        Ok(AiSelection {
            mv,
            meta: AiMoveMeta {
                backend: AiBackend::Heuristic,
                depth_reached: 1,
                nodes: 0,
                nps: 0,
                tt_hit_rate: 0.0,
                time_ms: elapsed.as_millis() as u64,
                source: AiMoveSource::Heuristic,
            },
        })
    }
}

fn heuristic_move(state: &GameState) -> Option<Move> {
    let legal = list_legal_moves(state);
    if legal.is_empty() {
        return None;
    }
    let mut best_move = legal[0];
    let mut best_score = i32::MIN;
    for mv in legal {
        let next_state = apply_move(state, mv).ok()?;
        let mut score = 0;
        if next_state.winner == Some(state.turn) {
            score += 1_000_000;
        }
        let gained = next_state.captures[&state.turn] as i32 - state.captures[&state.turn] as i32;
        score += gained * 100;
        score += legal_moves(&next_state, mv.target).len() as i32;
        if score > best_score {
            best_score = score;
            best_move = mv;
        }
    }
    Some(best_move)
}

// ---------------- AlphaZero backend ----------------

#[derive(Debug, Clone)]
struct MctsNode {
    state: GameState,
    prior: f32,
    visits: u32,
    value_sum: f32,
    children: HashMap<usize, Box<MctsNode>>,
}

impl MctsNode {
    fn new(state: GameState, prior: f32) -> Self {
        Self {
            state,
            prior,
            visits: 0,
            value_sum: 0.0,
            children: HashMap::new(),
        }
    }

    fn q_value(&self) -> f32 {
        if self.visits == 0 {
            0.0
        } else {
            self.value_sum / self.visits as f32
        }
    }

    fn expanded(&self) -> bool {
        !self.children.is_empty()
    }

    fn expand(&mut self, policy_logits: &[f32], legal: &[Move]) {
        self.children.clear();
        let probs = if policy_logits.len() == ALL_MOVES.len() {
            let mut logits = Vec::with_capacity(legal.len());
            for mv in legal {
                let idx = MOVE_TO_INDEX[&(mv.origin, mv.target)];
                logits.push(policy_logits[idx]);
            }
            softmax(&logits)
        } else {
            vec![1.0 / legal.len() as f32; legal.len()]
        };
        for (mv, prior) in legal.iter().zip(probs.into_iter()) {
            let idx = MOVE_TO_INDEX[&(mv.origin, mv.target)];
            let next_state = apply_move(&self.state, *mv).expect("legal move");
            self.children
                .insert(idx, Box::new(MctsNode::new(next_state, prior)));
        }
    }

    fn select_child(&mut self, c_puct: f32) -> Option<(usize, *mut MctsNode)> {
        let total_visits: f32 = self.children.values().map(|c| c.visits as f32).sum();
        let (idx, child) = self.children.iter_mut().max_by(|a, b| {
            let sa = mcts_ucb(a.1, c_puct, total_visits);
            let sb = mcts_ucb(b.1, c_puct, total_visits);
            sa.partial_cmp(&sb).unwrap_or(Ordering::Equal)
        })?;
        Some((*idx, &mut **child))
    }
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    if sum <= 0.0 {
        vec![1.0 / logits.len() as f32; logits.len()]
    } else {
        exp.into_iter().map(|v| v / sum).collect()
    }
}

fn mcts_ucb(child: &MctsNode, c_puct: f32, total_visits: f32) -> f32 {
    let q = -child.q_value();
    let u = c_puct * child.prior * (total_visits + 1e-8).sqrt() / (1.0 + child.visits as f32);
    q + u
}

struct AlphaZeroAgent {
    simulations: usize,
    c_puct: f32,
    #[cfg(feature = "onnx")]
    evaluator: Option<OnnxEvaluator>,
}

impl fmt::Debug for AlphaZeroAgent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AlphaZeroAgent")
            .field("simulations", &self.simulations)
            .field("c_puct", &self.c_puct)
            .field("available", &self.available())
            .finish()
    }
}

impl AlphaZeroAgent {
    fn new(_model_path: Option<PathBuf>, simulations: usize, c_puct: f32) -> Self {
        #[cfg(feature = "onnx")]
        let evaluator = _model_path.as_ref().and_then(|p| {
            match std::panic::catch_unwind(|| OnnxEvaluator::load(p.as_path())) {
                Ok(Ok(eval)) => Some(eval),
                Ok(Err(err)) => {
                    eprintln!(
                        "Failed to initialize AlphaZero ONNX evaluator from {}: {err}",
                        p.display()
                    );
                    None
                }
                Err(_) => {
                    eprintln!(
                        "ONNX runtime initialization panicked for {}; AlphaZero disabled",
                        p.display()
                    );
                    None
                }
            }
        });

        Self {
            simulations,
            c_puct,
            #[cfg(feature = "onnx")]
            evaluator,
        }
    }

    fn available(&self) -> bool {
        #[cfg(feature = "onnx")]
        {
            self.evaluator.is_some()
        }
        #[cfg(not(feature = "onnx"))]
        {
            false
        }
    }

    fn evaluate(&self, _state: &GameState) -> (Vec<f32>, f32) {
        #[cfg(feature = "onnx")]
        {
            if let Some(eval) = &self.evaluator {
                if let Ok((p, v)) = eval.policy_value(_state) {
                    return (p, v);
                }
            }
        }
        (vec![1.0; ALL_MOVES.len()], 0.0)
    }
}

impl MoveSelector for AlphaZeroAgent {
    fn select_move(
        &self,
        state: &GameState,
        profile: &AiProfile,
        deadline: Instant,
    ) -> Result<AiSelection, AiError> {
        if !self.available() {
            return Err(AiError::ModelUnavailable);
        }

        let legal_root = list_legal_moves(state);
        if legal_root.is_empty() {
            return Err(AiError::NoLegalMoves);
        }

        let started = Instant::now();
        let sims_scale = (profile.skill.clamp(1, 20) as f32 / 12.0).max(0.4);
        let sim_budget = ((self.simulations as f32) * sims_scale).round().max(8.0) as usize;

        let (policy, _value) = self.evaluate(state);
        let mut root = MctsNode::new(state.clone(), 1.0);
        root.expand(&policy, &legal_root);

        let mut sim_count = 0usize;
        while sim_count < sim_budget && Instant::now() < deadline {
            sim_count += 1;
            let mut node = &mut root;
            let mut path: Vec<*mut MctsNode> = vec![node as *mut MctsNode];
            while node.expanded() {
                let Some((_idx, child_ptr)) = node.select_child(self.c_puct) else {
                    break;
                };
                node = unsafe { &mut *child_ptr };
                path.push(node as *mut MctsNode);
            }

            let leaf_value = if node.state.winner.is_some() {
                if node.state.winner == Some(node.state.turn) {
                    1.0
                } else {
                    -1.0
                }
            } else {
                let legal = list_legal_moves(&node.state);
                if legal.is_empty() {
                    -1.0
                } else {
                    let (p, v) = self.evaluate(&node.state);
                    node.expand(&p, &legal);
                    v
                }
            };

            let mut value = leaf_value;
            for ptr in path.into_iter().rev() {
                let n = unsafe { &mut *ptr };
                n.visits += 1;
                n.value_sum += value;
                value = -value;
            }
        }

        let best_idx = root
            .children
            .iter()
            .max_by_key(|(_, child)| child.visits)
            .map(|(idx, _)| *idx)
            .ok_or(AiError::NoLegalMoves)?;

        let elapsed = started.elapsed();
        let nodes = sim_count as u64;
        let nps = if elapsed.as_secs_f64() > 0.0 {
            (nodes as f64 / elapsed.as_secs_f64()) as u64
        } else {
            nodes
        };

        Ok(AiSelection {
            mv: move_from_index(best_idx),
            meta: AiMoveMeta {
                backend: AiBackend::Alphazero,
                depth_reached: 1,
                nodes,
                nps,
                tt_hit_rate: 0.0,
                time_ms: elapsed.as_millis() as u64,
                source: AiMoveSource::Search,
            },
        })
    }
}

// ---------------- Centurion backend ----------------

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BookArtifact {
    pub version: u32,
    pub entries: Vec<BookEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BookEntry {
    pub hash: u64,
    pub moves: Vec<BookMoveStat>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BookMoveStat {
    pub move_index: usize,
    pub plays: u32,
    pub wins: u32,
    pub draws: u32,
    pub losses: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TablebaseArtifact {
    pub version: u32,
    pub max_pieces: u8,
    pub entries: Vec<TablebaseEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TablebaseEntry {
    pub hash: u64,
    pub move_index: usize,
    pub wdl: i8,
    pub dtm: Option<i16>,
}

pub fn save_book_artifact(path: &Path, artifact: &BookArtifact) -> Result<(), AiError> {
    let bytes = bincode::serialize(artifact).map_err(|e| AiError::Io(e.to_string()))?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| AiError::Io(e.to_string()))?;
    }
    fs::write(path, bytes).map_err(|e| AiError::Io(e.to_string()))
}

pub fn load_book_artifact(path: &Path) -> Result<BookArtifact, AiError> {
    let bytes = fs::read(path).map_err(|e| AiError::Io(e.to_string()))?;
    bincode::deserialize(&bytes).map_err(|e| AiError::Io(e.to_string()))
}

pub fn save_tablebase_artifact(path: &Path, artifact: &TablebaseArtifact) -> Result<(), AiError> {
    let bytes = bincode::serialize(artifact).map_err(|e| AiError::Io(e.to_string()))?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| AiError::Io(e.to_string()))?;
    }
    fs::write(path, bytes).map_err(|e| AiError::Io(e.to_string()))
}

pub fn load_tablebase_artifact(path: &Path) -> Result<TablebaseArtifact, AiError> {
    let bytes = fs::read(path).map_err(|e| AiError::Io(e.to_string()))?;
    bincode::deserialize(&bytes).map_err(|e| AiError::Io(e.to_string()))
}

pub const NNUE_INPUT_SIZE: usize = (BOARD_WIDTH as usize * BOARD_HEIGHT as usize * 4) + 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NnueWeights {
    pub version: u32,
    pub input_size: usize,
    pub hidden_size: usize,
    pub w1: Vec<f32>,
    pub b1: Vec<f32>,
    pub w2: Vec<f32>,
    pub b2: f32,
    pub scale: f32,
}

impl NnueWeights {
    pub fn new_random(hidden_size: usize, seed: u64) -> Self {
        let input_size = NNUE_INPUT_SIZE;
        let mut rng = StdRng::seed_from_u64(seed);
        let w1_len = hidden_size * input_size;
        let mut w1 = vec![0.0; w1_len];
        let mut w2 = vec![0.0; hidden_size];
        for w in &mut w1 {
            *w = rng.gen_range(-0.02..0.02);
        }
        for w in &mut w2 {
            *w = rng.gen_range(-0.08..0.08);
        }
        let b1 = vec![0.0; hidden_size];
        Self {
            version: 1,
            input_size,
            hidden_size,
            w1,
            b1,
            w2,
            b2: 0.0,
            scale: 180.0,
        }
    }
}

pub fn nnue_active_features(state: &GameState) -> Vec<usize> {
    let plane_size = BOARD_WIDTH as usize * BOARD_HEIGHT as usize;
    let mut out = Vec::with_capacity(state.board.len() + 1);
    for ((x, y), piece) in &state.board {
        let base = match (piece.color, piece.kind) {
            (Color::White, PieceType::Soldier) => 0,
            (Color::Black, PieceType::Soldier) => plane_size,
            (Color::White, PieceType::Dux) => 2 * plane_size,
            (Color::Black, PieceType::Dux) => 3 * plane_size,
        };
        out.push(base + (*y as usize * BOARD_WIDTH as usize + *x as usize));
    }
    if state.turn == Color::White {
        out.push(4 * plane_size);
    }
    out
}

pub fn save_nnue_weights(path: &Path, weights: &NnueWeights) -> Result<(), AiError> {
    let bytes = bincode::serialize(weights).map_err(|e| AiError::Io(e.to_string()))?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| AiError::Io(e.to_string()))?;
    }
    fs::write(path, bytes).map_err(|e| AiError::Io(e.to_string()))
}

pub fn load_nnue_weights(path: &Path) -> Result<NnueWeights, AiError> {
    let bytes = fs::read(path).map_err(|e| AiError::Io(e.to_string()))?;
    let weights: NnueWeights = bincode::deserialize(&bytes).map_err(|e| AiError::Io(e.to_string()))?;
    if weights.input_size != NNUE_INPUT_SIZE {
        return Err(AiError::InvalidProfile(format!(
            "nnue input mismatch: got {}, expected {}",
            weights.input_size, NNUE_INPUT_SIZE
        )));
    }
    if weights.w1.len() != weights.hidden_size * weights.input_size
        || weights.b1.len() != weights.hidden_size
        || weights.w2.len() != weights.hidden_size
    {
        return Err(AiError::InvalidProfile(
            "nnue shape mismatch in weights file".to_string(),
        ));
    }
    Ok(weights)
}

#[derive(Debug, Clone)]
struct NnueEvaluator {
    weights: Arc<NnueWeights>,
}

impl NnueEvaluator {
    fn new(weights: Arc<NnueWeights>) -> Self {
        Self { weights }
    }

    fn evaluate_cp(&self, state: &GameState) -> i32 {
        let active = nnue_active_features(state);
        let w = &self.weights;
        let mut hidden = vec![0.0f32; w.hidden_size];

        for h in 0..w.hidden_size {
            let mut sum = w.b1[h];
            let row = h * w.input_size;
            for &idx in &active {
                sum += w.w1[row + idx];
            }
            hidden[h] = sum.clamp(0.0, 1.0);
        }

        let mut out = w.b2;
        for (h, a) in hidden.iter().enumerate() {
            out += w.w2[h] * *a;
        }
        (out * w.scale) as i32
    }
}

#[derive(Debug, Clone)]
struct OpeningBook {
    entries: HashMap<u64, Vec<BookMoveStat>>,
}

impl OpeningBook {
    fn load(path: &Path) -> Result<Self, AiError> {
        let artifact = load_book_artifact(path)?;
        let mut entries = HashMap::with_capacity(artifact.entries.len());
        for entry in artifact.entries {
            entries.insert(entry.hash, entry.moves);
        }
        Ok(Self { entries })
    }

    fn probe(&self, hash: u64, skill: u8) -> Option<usize> {
        let options = self.entries.get(&hash)?;
        if options.is_empty() {
            return None;
        }
        let mut weighted = Vec::new();
        for m in options {
            let score = m.wins as f32 + 0.5 * m.draws as f32 - m.losses as f32;
            let confidence = (m.plays.max(1) as f32).ln_1p();
            let skill_mult = 0.6 + (skill.clamp(1, 20) as f32 / 20.0) * 0.9;
            let weight = ((score.max(0.5) + 1.0) * confidence * skill_mult).max(0.1);
            weighted.push((m.move_index, weight));
        }
        let seed = hash ^ ((skill as u64) << 48) ^ 0xC311_7001;
        let mut rng = StdRng::seed_from_u64(seed);
        sample_weighted_index(&weighted, &mut rng)
    }
}

#[derive(Debug, Clone)]
struct Tablebase {
    max_pieces: u8,
    entries: HashMap<u64, TablebaseEntry>,
}

impl Tablebase {
    fn load(path: &Path) -> Result<Self, AiError> {
        let artifact = load_tablebase_artifact(path)?;
        let mut entries = HashMap::with_capacity(artifact.entries.len());
        for entry in artifact.entries {
            entries.insert(entry.hash, entry);
        }
        Ok(Self {
            max_pieces: artifact.max_pieces,
            entries,
        })
    }

    fn probe(&self, state: &GameState, hash: u64) -> Option<&TablebaseEntry> {
        if state.board.len() > self.max_pieces as usize {
            return None;
        }
        self.entries.get(&hash)
    }
}

fn sample_weighted_index(weighted: &[(usize, f32)], rng: &mut StdRng) -> Option<usize> {
    if weighted.is_empty() {
        return None;
    }
    let total: f32 = weighted.iter().map(|(_, w)| w.max(0.0)).sum();
    if total <= 0.0 {
        return weighted.first().map(|(idx, _)| *idx);
    }
    let mut r = rng.gen_range(0.0..total);
    for (idx, w) in weighted {
        let ww = w.max(0.0);
        if r <= ww {
            return Some(*idx);
        }
        r -= ww;
    }
    weighted.last().map(|(idx, _)| *idx)
}

struct CenturionAgent {
    opening_book: Option<OpeningBook>,
    tablebase: Option<Tablebase>,
    nnue: Option<NnueEvaluator>,
}

#[derive(Debug, Clone, Copy)]
enum Bound {
    Exact,
    Lower,
    Upper,
}

#[derive(Debug, Clone, Copy)]
struct TtEntry {
    depth: i32,
    score: i32,
    bound: Bound,
    best_move_idx: Option<usize>,
}

#[derive(Debug, Default, Clone)]
struct SearchStats {
    nodes: u64,
    tt_probes: u64,
    tt_hits: u64,
    depth_reached: u8,
}

#[derive(Debug)]
struct SearchContext {
    tt: HashMap<u64, TtEntry>,
    history: HashMap<usize, i32>,
    killers: Vec<[Option<usize>; 2]>,
    stats: SearchStats,
    deadline: Instant,
    aborted: bool,
    nnue: Option<NnueEvaluator>,
}

#[derive(Debug, Clone, Copy)]
struct ScoredMove {
    mv: Move,
    idx: usize,
    score: i32,
    capture_gain: i32,
    winning: bool,
}

impl SearchContext {
    fn with_profile(profile: &AiProfile, deadline: Instant, nnue: Option<NnueEvaluator>) -> Self {
        let approx_entry_size = std::mem::size_of::<(u64, TtEntry)>().max(24);
        let capacity = (profile.hash_mb.max(1) * 1024 * 1024) / approx_entry_size;
        Self {
            tt: HashMap::with_capacity(capacity.max(1024)),
            history: HashMap::new(),
            killers: vec![[None, None]; MAX_SEARCH_PLY],
            stats: SearchStats::default(),
            deadline,
            aborted: false,
            nnue,
        }
    }

    fn timeout(&self) -> bool {
        Instant::now() >= self.deadline
    }
}

impl MoveSelector for CenturionAgent {
    fn select_move(
        &self,
        state: &GameState,
        profile: &AiProfile,
        deadline: Instant,
    ) -> Result<AiSelection, AiError> {
        let started = Instant::now();
        let legal = list_legal_moves(state);
        if legal.is_empty() {
            return Err(AiError::NoLegalMoves);
        }

        let hash = zobrist_hash(state);

        if profile.use_book {
            if let Some(book) = &self.opening_book {
                if let Some(idx) = book.probe(hash, profile.skill) {
                    if let Some(mv) = move_from_index(idx) {
                        if legal.iter().any(|m| *m == mv) {
                            let elapsed = started.elapsed();
                            return Ok(AiSelection {
                                mv: Some(mv),
                                meta: AiMoveMeta {
                                    backend: AiBackend::Centurion,
                                    depth_reached: 0,
                                    nodes: 0,
                                    nps: 0,
                                    tt_hit_rate: 0.0,
                                    time_ms: elapsed.as_millis() as u64,
                                    source: AiMoveSource::Book,
                                },
                            });
                        }
                    }
                }
            }
        }

        if profile.use_tb {
            if let Some(tb) = &self.tablebase {
                if let Some(entry) = tb.probe(state, hash) {
                    if let Some(mv) = move_from_index(entry.move_index) {
                        if legal.iter().any(|m| *m == mv) {
                            let elapsed = started.elapsed();
                            return Ok(AiSelection {
                                mv: Some(mv),
                                meta: AiMoveMeta {
                                    backend: AiBackend::Centurion,
                                    depth_reached: 0,
                                    nodes: 0,
                                    nps: 0,
                                    tt_hit_rate: 0.0,
                                    time_ms: elapsed.as_millis() as u64,
                                    source: AiMoveSource::Tb,
                                },
                            });
                        }
                    }
                }
            }
        }

        let mut ctx = SearchContext::with_profile(profile, deadline, self.nnue.clone());
        let max_depth = effective_max_depth(profile) as i32;

        let mut best_move: Option<Move> = None;
        let mut aspiration_center = 0;

        for depth in 1..=max_depth {
            if ctx.timeout() {
                break;
            }

            let (alpha_start, beta_start) = if depth >= 2 {
                let window = 80 + (depth * 12);
                (aspiration_center - window, aspiration_center + window)
            } else {
                (-MATE_SCORE, MATE_SCORE)
            };

            let mut alpha = alpha_start;
            let mut beta = beta_start;
            let mut resolved = false;

            for _ in 0..3 {
                if ctx.timeout() {
                    break;
                }
                match centurion_root_search(state, depth, alpha, beta, &mut ctx) {
                    Ok((score, mv)) => {
                        best_move = Some(mv);
                        aspiration_center = score;
                        resolved = true;
                        if score <= alpha {
                            alpha -= 200;
                            beta = ((alpha + beta) / 2).max(alpha + 1);
                            resolved = false;
                            continue;
                        }
                        if score >= beta {
                            beta += 200;
                            alpha = ((alpha + beta) / 2).min(beta - 1);
                            resolved = false;
                            continue;
                        }
                        break;
                    }
                    Err(AiError::SearchAborted) => break,
                    Err(err) => return Err(err),
                }
            }

            if resolved {
                ctx.stats.depth_reached = depth as u8;
            }

            if ctx.timeout() {
                break;
            }
        }

        let mv = best_move.or_else(|| heuristic_move(state));
        let elapsed = started.elapsed();
        let secs = elapsed.as_secs_f64();
        let nps = if secs > 0.0 {
            (ctx.stats.nodes as f64 / secs) as u64
        } else {
            0
        };
        let tt_hit_rate = if ctx.stats.tt_probes > 0 {
            ctx.stats.tt_hits as f32 / ctx.stats.tt_probes as f32
        } else {
            0.0
        };

        Ok(AiSelection {
            mv,
            meta: AiMoveMeta {
                backend: AiBackend::Centurion,
                depth_reached: ctx.stats.depth_reached,
                nodes: ctx.stats.nodes,
                nps,
                tt_hit_rate,
                time_ms: elapsed.as_millis() as u64,
                source: AiMoveSource::Search,
            },
        })
    }
}

fn centurion_root_search(
    state: &GameState,
    depth: i32,
    mut alpha: i32,
    beta: i32,
    ctx: &mut SearchContext,
) -> Result<(i32, Move), AiError> {
    if ctx.timeout() {
        return Err(AiError::SearchAborted);
    }
    let hash = zobrist_hash(state);
    let tt_move = ctx.tt.get(&hash).and_then(|e| e.best_move_idx);
    let mut moves = ordered_moves(state, tt_move, 0, ctx)?;
    if moves.is_empty() {
        return Err(AiError::NoLegalMoves);
    }

    let mut best_score = -MATE_SCORE;
    let mut best_move = moves[0].mv;

    for (i, sm) in moves.drain(..).enumerate() {
        if ctx.timeout() {
            return Err(AiError::SearchAborted);
        }
        let next = apply_move(state, sm.mv).map_err(|e| AiError::Io(e.to_string()))?;
        let score = if i == 0 {
            -centurion_negamax(&next, depth - 1, -beta, -alpha, 1, ctx)?
        } else {
            // PVS null-window search first.
            let mut s = -centurion_negamax(&next, depth - 1, -alpha - 1, -alpha, 1, ctx)?;
            if s > alpha && s < beta {
                s = -centurion_negamax(&next, depth - 1, -beta, -alpha, 1, ctx)?;
            }
            s
        };

        if score > best_score {
            best_score = score;
            best_move = sm.mv;
        }
        if score > alpha {
            alpha = score;
        }
        if alpha >= beta {
            break;
        }
    }

    ctx.tt.insert(
        hash,
        TtEntry {
            depth,
            score: best_score,
            bound: if best_score <= alpha {
                Bound::Upper
            } else if best_score >= beta {
                Bound::Lower
            } else {
                Bound::Exact
            },
            best_move_idx: move_index(best_move),
        },
    );

    Ok((best_score, best_move))
}

fn centurion_negamax(
    state: &GameState,
    depth: i32,
    mut alpha: i32,
    beta: i32,
    ply: usize,
    ctx: &mut SearchContext,
) -> Result<i32, AiError> {
    if ctx.timeout() {
        ctx.aborted = true;
        return Err(AiError::SearchAborted);
    }
    ctx.stats.nodes += 1;

    if let Some(winner) = state.winner {
        let score = if winner == state.turn {
            MATE_SCORE - ply as i32
        } else {
            -MATE_SCORE + ply as i32
        };
        return Ok(score);
    }

    if depth <= 0 {
        return centurion_quiescence(state, alpha, beta, ply, ctx);
    }

    let hash = zobrist_hash(state);
    let mut tt_move = None;
    if let Some(entry) = ctx.tt.get(&hash) {
        ctx.stats.tt_probes += 1;
        tt_move = entry.best_move_idx;
        if entry.depth >= depth {
            ctx.stats.tt_hits += 1;
            match entry.bound {
                Bound::Exact => return Ok(entry.score),
                Bound::Lower => {
                    if entry.score >= beta {
                        return Ok(entry.score);
                    }
                    alpha = alpha.max(entry.score);
                }
                Bound::Upper => {
                    if entry.score <= alpha {
                        return Ok(entry.score);
                    }
                }
            }
        }
    } else {
        ctx.stats.tt_probes += 1;
    }

    let mut moves = ordered_moves(state, tt_move, ply, ctx)?;
    if moves.is_empty() {
        return Ok(-MATE_SCORE + ply as i32);
    }

    let alpha_orig = alpha;
    let mut best_score = -MATE_SCORE;
    let mut best_move_idx = None;

    for (i, sm) in moves.drain(..).enumerate() {
        let next = apply_move(state, sm.mv).map_err(|e| AiError::Io(e.to_string()))?;
        let score = if i == 0 {
            -centurion_negamax(&next, depth - 1, -beta, -alpha, ply + 1, ctx)?
        } else {
            let mut s = -centurion_negamax(&next, depth - 1, -alpha - 1, -alpha, ply + 1, ctx)?;
            if s > alpha && s < beta {
                s = -centurion_negamax(&next, depth - 1, -beta, -alpha, ply + 1, ctx)?;
            }
            s
        };

        if score > best_score {
            best_score = score;
            best_move_idx = Some(sm.idx);
        }
        alpha = alpha.max(score);

        if alpha >= beta {
            // Update killer/history for quiet cutoffs.
            if sm.capture_gain <= 0 && !sm.winning {
                if let Some(slot) = ctx.killers.get_mut(ply.min(MAX_SEARCH_PLY - 1)) {
                    if slot[0] != Some(sm.idx) {
                        slot[1] = slot[0];
                        slot[0] = Some(sm.idx);
                    }
                }
                *ctx.history.entry(sm.idx).or_insert(0) += depth * depth;
            }
            break;
        }
    }

    let bound = if best_score <= alpha_orig {
        Bound::Upper
    } else if best_score >= beta {
        Bound::Lower
    } else {
        Bound::Exact
    };

    ctx.tt.insert(
        hash,
        TtEntry {
            depth,
            score: best_score,
            bound,
            best_move_idx,
        },
    );

    Ok(best_score)
}

fn centurion_quiescence(
    state: &GameState,
    mut alpha: i32,
    beta: i32,
    ply: usize,
    ctx: &mut SearchContext,
) -> Result<i32, AiError> {
    if ctx.timeout() {
        return Err(AiError::SearchAborted);
    }
    let stand_pat = blended_eval(state, ctx.nnue.as_ref());
    if stand_pat >= beta {
        return Ok(beta);
    }
    if stand_pat > alpha {
        alpha = stand_pat;
    }
    if ply >= MAX_SEARCH_PLY - 1 {
        return Ok(alpha);
    }

    let mut tactical: Vec<ScoredMove> = list_legal_moves(state)
        .into_iter()
        .filter_map(|mv| {
            let idx = move_index(mv)?;
            let gain = move_capture_gain(state, mv);
            let winning = is_winning_move(state, mv);
            if gain <= 0 && !winning {
                return None;
            }
            let score = if winning { 1_000_000 } else { gain * 10_000 };
            Some(ScoredMove {
                mv,
                idx,
                score,
                capture_gain: gain,
                winning,
            })
        })
        .collect();

    tactical.sort_by(|a, b| b.score.cmp(&a.score));

    for sm in tactical {
        let next = apply_move(state, sm.mv).map_err(|e| AiError::Io(e.to_string()))?;
        let score = -centurion_quiescence(&next, -beta, -alpha, ply + 1, ctx)?;
        if score >= beta {
            return Ok(beta);
        }
        if score > alpha {
            alpha = score;
        }
    }
    Ok(alpha)
}

fn ordered_moves(
    state: &GameState,
    tt_move: Option<usize>,
    ply: usize,
    ctx: &SearchContext,
) -> Result<Vec<ScoredMove>, AiError> {
    let legal = list_legal_moves(state);
    if legal.is_empty() {
        return Ok(Vec::new());
    }

    let killers = ctx
        .killers
        .get(ply.min(MAX_SEARCH_PLY - 1))
        .copied()
        .unwrap_or([None, None]);

    let mut scored = Vec::with_capacity(legal.len());
    for mv in legal {
        let idx = move_index(mv).ok_or_else(|| AiError::Io("move index missing".to_string()))?;
        let gain = move_capture_gain(state, mv);
        let winning = is_winning_move(state, mv);
        let mut score = 0;
        if Some(idx) == tt_move {
            score += 10_000_000;
        }
        if winning {
            score += 5_000_000;
        }
        if gain > 0 {
            score += 3_000_000 + gain * 50_000;
        }
        if Some(idx) == killers[0] {
            score += 100_000;
        } else if Some(idx) == killers[1] {
            score += 80_000;
        }
        score += ctx.history.get(&idx).copied().unwrap_or(0);

        scored.push(ScoredMove {
            mv,
            idx,
            score,
            capture_gain: gain,
            winning,
        });
    }

    scored.sort_by(|a, b| b.score.cmp(&a.score));
    Ok(scored)
}

fn terminal_eval(state: &GameState) -> Option<i32> {
    if let Some(winner) = state.winner {
        Some(if winner == state.turn {
            MATE_SCORE
        } else {
            -MATE_SCORE
        })
    } else {
        None
    }
}

fn blended_eval(state: &GameState, nnue: Option<&NnueEvaluator>) -> i32 {
    let heuristic = heuristic_static_eval(state);
    if let Some(model) = nnue {
        let nnue_cp = model.evaluate_cp(state);
        // Blend heuristic and NNUE to keep strategic terms while NNUE matures.
        ((heuristic * 35) + (nnue_cp * 65)) / 100
    } else {
        heuristic
    }
}

pub fn heuristic_static_eval(state: &GameState) -> i32 {
    if let Some(terminal) = terminal_eval(state) {
        return terminal;
    }

    let side = state.turn;
    let opp = side.opponent();

    let mat_side = material_score(state, side);
    let mat_opp = material_score(state, opp);

    let mob_side = total_mobility(state, side) as i32;
    let mob_opp = total_mobility(state, opp) as i32;

    let pressure_side = tactical_pressure(state, side) as i32;
    let pressure_opp = tactical_pressure(state, opp) as i32;

    let center_side = center_control(state, side);
    let center_opp = center_control(state, opp);

    let dux_side = dux_safety(state, side);
    let dux_opp = dux_safety(state, opp);

    let total_soldiers = state
        .board
        .values()
        .filter(|p| p.kind == PieceType::Soldier)
        .count() as f32;
    let endgame_factor = ((24.0 - total_soldiers) / 24.0).clamp(0.0, 1.0);

    let mut score = 0i32;
    score += (mat_side - mat_opp) * 10;
    score += (mob_side - mob_opp) * 3;
    score += (pressure_side - pressure_opp) * 10;
    score += (center_side - center_opp) * 2;
    score += (dux_side - dux_opp) * (20.0 + endgame_factor * 25.0) as i32;
    score += 12; // tempo

    score
}

fn material_score(state: &GameState, color: Color) -> i32 {
    state
        .board
        .values()
        .filter(|p| p.color == color)
        .map(|p| match p.kind {
            PieceType::Soldier => 100,
            PieceType::Dux => 1800,
        })
        .sum()
}

fn total_mobility(state: &GameState, color: Color) -> usize {
    state
        .board
        .iter()
        .filter(|(_, piece)| piece.color == color)
        .map(|(coord, _)| legal_moves(state, *coord).len())
        .sum()
}

fn tactical_pressure(state: &GameState, color: Color) -> usize {
    let mut count = 0usize;
    for (coord, piece) in &state.board {
        if piece.color != color {
            continue;
        }
        for target in legal_moves(state, *coord) {
            let mv = Move {
                origin: *coord,
                target,
            };
            if move_capture_gain(state, mv) > 0 || is_winning_move(state, mv) {
                count += 1;
            }
        }
    }
    count
}

fn center_control(state: &GameState, color: Color) -> i32 {
    let cx = (BOARD_WIDTH as i32 - 1) as f32 / 2.0;
    let cy = (BOARD_HEIGHT as i32 - 1) as f32 / 2.0;
    let mut score = 0.0;
    for ((x, y), piece) in &state.board {
        if piece.color != color {
            continue;
        }
        let dx = (*x as f32 - cx).abs();
        let dy = (*y as f32 - cy).abs();
        let dist = dx + dy;
        let piece_weight = if piece.kind == PieceType::Dux {
            1.3
        } else {
            1.0
        };
        score += (6.5 - dist).max(0.0) * piece_weight;
    }
    score.round() as i32
}

fn dux_safety(state: &GameState, color: Color) -> i32 {
    let Some(dux_coord) = find_piece(state, color, PieceType::Dux) else {
        return -200;
    };
    let mobility = legal_moves(state, dux_coord).len() as i32;
    let attackers = adjacent_enemy_count(state, dux_coord, color.opponent()) as i32;
    let defenders = adjacent_enemy_count(state, dux_coord, color) as i32;
    mobility * 6 + defenders * 8 - attackers * 15
}

fn adjacent_enemy_count(state: &GameState, coord: Coord, color: Color) -> usize {
    let mut n = 0usize;
    for (dx, dy) in DIRECTIONS {
        let nx = coord.0 as i16 + dx as i16;
        let ny = coord.1 as i16 + dy as i16;
        if nx < 0 || ny < 0 || nx >= BOARD_WIDTH as i16 || ny >= BOARD_HEIGHT as i16 {
            continue;
        }
        if let Some(piece) = state.board.get(&(nx as u8, ny as u8)) {
            if piece.color == color {
                n += 1;
            }
        }
    }
    n
}

fn find_piece(state: &GameState, color: Color, kind: PieceType) -> Option<Coord> {
    state
        .board
        .iter()
        .find_map(|(coord, piece)| (piece.color == color && piece.kind == kind).then_some(*coord))
}

// ---------------- Zobrist hashing ----------------

static ZOBRIST_BOARD: Lazy<[[[u64; 4]; BOARD_HEIGHT as usize]; BOARD_WIDTH as usize]> =
    Lazy::new(|| {
        let mut rng = StdRng::seed_from_u64(0xCE57_A110_0011u64.wrapping_mul(17));
        let mut arr = [[[0u64; 4]; BOARD_HEIGHT as usize]; BOARD_WIDTH as usize];
        for x in 0..BOARD_WIDTH as usize {
            for y in 0..BOARD_HEIGHT as usize {
                for slot in 0..4 {
                    arr[x][y][slot] = rng.gen::<u64>();
                }
            }
        }
        arr
    });

static ZOBRIST_TURN: Lazy<u64> = Lazy::new(|| {
    let mut rng = StdRng::seed_from_u64(0xCE57_A110_0055);
    rng.gen::<u64>()
});

fn piece_slot(color: Color, kind: PieceType) -> usize {
    match (color, kind) {
        (Color::White, PieceType::Soldier) => 0,
        (Color::Black, PieceType::Soldier) => 1,
        (Color::White, PieceType::Dux) => 2,
        (Color::Black, PieceType::Dux) => 3,
    }
}

pub fn zobrist_hash(state: &GameState) -> u64 {
    let mut h = 0u64;
    for ((x, y), piece) in &state.board {
        let slot = piece_slot(piece.color, piece.kind);
        h ^= ZOBRIST_BOARD[*x as usize][*y as usize][slot];
    }
    if state.turn == Color::White {
        h ^= *ZOBRIST_TURN;
    }
    h
}

#[cfg(feature = "onnx")]
mod onnx {
    use super::*;
    use engine::Color;
    use ndarray;
    use ort::session::builder::GraphOptimizationLevel;
    use ort::session::Session;
    use ort::value::Tensor;
    use parking_lot::Mutex;

    pub struct OnnxEvaluator {
        session: Mutex<Session>,
    }

    impl OnnxEvaluator {
        pub fn load(path: &StdPath) -> anyhow::Result<Self> {
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

// ---------------- Tests ----------------

#[cfg(test)]
mod tests {
    use super::*;
    use engine::initial_state;

    #[test]
    fn heuristic_returns_move() {
        let state = initial_state(5).unwrap();
        let agent = AIAgent::default();
        let mv = agent.select_move(&state).unwrap();
        assert!(mv.is_some());
    }

    #[test]
    fn zobrist_is_stable() {
        let state = initial_state(5).unwrap();
        let h1 = zobrist_hash(&state);
        let h2 = zobrist_hash(&state);
        assert_eq!(h1, h2);
    }

    #[test]
    fn move_indexing_covers_legal_moves() {
        let state = initial_state(5).unwrap();
        let moves = list_legal_moves(&state);
        for mv in moves {
            let idx = move_index(mv);
            assert!(idx.is_some());
            assert!(idx.unwrap() < ALL_MOVES.len());
        }
    }

    #[test]
    fn centurion_respects_small_time_budget() {
        let state = initial_state(5).unwrap();
        let mut profile = AiProfile::default();
        profile.backend = AiBackend::Centurion;
        profile.move_time_ms = 10;
        profile.max_depth = 16;
        profile.use_book = false;
        profile.use_tb = false;

        let agent = AIAgent::default();
        let started = Instant::now();
        let sel = agent.select_move_with_profile(&state, &profile).unwrap();
        assert!(sel.mv.is_some());
        assert!(started.elapsed() < Duration::from_millis(300));
    }

    #[test]
    fn centurion_replies_after_common_opening_move() {
        let state = initial_state(5).unwrap();
        let state = apply_move(
            &state,
            Move {
                origin: (0, 0),
                target: (0, 1),
            },
        )
        .unwrap();

        let mut profile = AiProfile::default();
        profile.backend = AiBackend::Centurion;
        profile.move_time_ms = 200;
        profile.max_depth = 24;

        let agent = AIAgent::default();
        let started = Instant::now();
        let sel = agent.select_move_with_profile(&state, &profile).unwrap();
        assert!(sel.mv.is_some());
        assert!(
            started.elapsed() < Duration::from_millis(2_500),
            "centurion exceeded expected reply window",
        );
    }

    #[test]
    fn book_artifact_roundtrip() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("parcae_book_test_{unique}.bin"));
        let artifact = BookArtifact {
            version: 1,
            entries: vec![BookEntry {
                hash: 123,
                moves: vec![BookMoveStat {
                    move_index: 7,
                    plays: 10,
                    wins: 6,
                    draws: 2,
                    losses: 2,
                }],
            }],
        };
        save_book_artifact(&path, &artifact).unwrap();
        let loaded = load_book_artifact(&path).unwrap();
        let _ = fs::remove_file(&path);
        assert_eq!(loaded.entries.len(), 1);
        assert_eq!(loaded.entries[0].hash, 123);
    }

    #[test]
    fn tablebase_artifact_roundtrip() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("parcae_tb_test_{unique}.bin"));
        let artifact = TablebaseArtifact {
            version: 1,
            max_pieces: 4,
            entries: vec![TablebaseEntry {
                hash: 555,
                move_index: 9,
                wdl: 1,
                dtm: Some(3),
            }],
        };
        save_tablebase_artifact(&path, &artifact).unwrap();
        let loaded = load_tablebase_artifact(&path).unwrap();
        let _ = fs::remove_file(&path);
        assert_eq!(loaded.max_pieces, 4);
        assert_eq!(loaded.entries[0].hash, 555);
    }

    #[test]
    fn nnue_artifact_roundtrip_and_eval() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("parcae_nnue_test_{unique}.bin"));
        let weights = NnueWeights::new_random(64, 12345);
        save_nnue_weights(&path, &weights).unwrap();
        let loaded = load_nnue_weights(&path).unwrap();
        let _ = fs::remove_file(&path);
        assert_eq!(loaded.input_size, NNUE_INPUT_SIZE);
        assert_eq!(loaded.hidden_size, 64);
        let state = initial_state(5).unwrap();
        let eval = NnueEvaluator::new(Arc::new(loaded)).evaluate_cp(&state);
        assert!(eval.abs() < 200_000);
    }
}
