use engine::{
    coord_to_notation, legal_moves, notation_to_coord, Color as EngineColor, GameState as EngineState,
    Piece as EnginePiece, PieceType as EnginePieceType, BOARD_HEIGHT, BOARD_WIDTH, FILES,
};
use futures_util::StreamExt;
use gloo_net::http::Request;
use gloo_net::websocket::futures::WebSocket;
use gloo_net::websocket::Message as WsMessage;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::spawn_local;
use yew::prelude::*;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

fn api_base() -> &'static str {
    option_env!("PARCAE_API_URL").unwrap_or("http://localhost:8000")
}

pub type Piece = EnginePiece;
pub type PieceType = EnginePieceType;
pub type Color = EngineColor;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MatchMode {
    Pva,
    Ava,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MoveDto {
    pub origin: String,
    pub target: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoardEntry {
    pub coord: String,
    pub piece: Piece,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GameState {
    pub turn: Color,
    pub winner: Option<Color>,
    pub captures: HashMap<Color, u32>,
    pub history: Vec<MoveDto>,
    pub board: Vec<BoardEntry>,
    pub summary: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MatchResponse {
    pub id: String,
    pub mode: MatchMode,
    pub state: GameState,
}

// --- API helpers ---

async fn api_create_match(mode: MatchMode) -> Result<MatchResponse, String> {
    Request::post(&format!("{}/match", api_base()))
        .header("Content-Type", "application/json")
        .body(
            serde_json::to_string(&serde_json::json!({
                "mode": match mode {
                    MatchMode::Pva => "pva",
                    MatchMode::Ava => "ava",
                }
            }))
            .unwrap(),
        )
        .map_err(|e| e.to_string())?
        .send()
        .await
        .map_err(|e| e.to_string())?
        .json::<MatchResponse>()
        .await
        .map_err(|e| e.to_string())
}

async fn api_post_move(id: &str, origin: &str, target: &str) -> Result<MatchResponse, String> {
    Request::post(&format!("{}/match/{id}/move", api_base()))
        .header("Content-Type", "application/json")
        .body(
            serde_json::to_string(&serde_json::json!({
                "origin": origin,
                "target": target,
            }))
            .unwrap(),
        )
        .map_err(|e| e.to_string())?
        .send()
        .await
        .map_err(|e| e.to_string())?
        .json::<MatchResponse>()
        .await
        .map_err(|e| e.to_string())
}

async fn api_step_ai(id: &str) -> Result<MatchResponse, String> {
    Request::post(&format!("{}/match/{id}/ai-step", api_base()))
        .send()
        .await
        .map_err(|e| e.to_string())?
        .json::<MatchResponse>()
        .await
        .map_err(|e| e.to_string())
}

// --- Board helpers ---

fn board_to_map(entries: &[BoardEntry]) -> HashMap<String, Piece> {
    entries.iter().fold(HashMap::new(), |mut acc, entry| {
        acc.insert(entry.coord.clone(), entry.piece);
        acc
    })
}

fn legal_moves_for(board: &[BoardEntry], turn: Color, origin: &str) -> Vec<String> {
    let mut board_map = HashMap::new();
    for entry in board {
        if let Ok(coord) = notation_to_coord(&entry.coord) {
            board_map.insert(coord, entry.piece);
        }
    }
    let mut captures = HashMap::new();
    captures.insert(Color::White, 0);
    captures.insert(Color::Black, 0);
    let state = EngineState {
        board: board_map,
        turn,
        winner: None,
        captures,
        history: Vec::new(),
        summary: None,
    };
    if let Ok(origin_coord) = notation_to_coord(origin) {
        legal_moves(&state, origin_coord)
            .into_iter()
            .map(coord_to_notation)
            .collect()
    } else {
        Vec::new()
    }
}

fn ws_base() -> String {
    let base = api_base();
    if let Some(stripped) = base.strip_prefix("http://") {
        format!("ws://{stripped}")
    } else if let Some(stripped) = base.strip_prefix("https://") {
        format!("wss://{stripped}")
    } else {
        base.replace("http", "ws")
    }
}

// --- Components ---

#[function_component(Board)]
fn board_view(
    props: &BoardProps,
) -> Html {
    let BoardProps {
        board,
        selected,
        legal_moves,
        on_cell_click,
    } = props;
    let legal_set: std::collections::HashSet<String> = legal_moves.iter().cloned().collect();
    let rows = (0..BOARD_HEIGHT).rev().map(|rank| {
        let cells = (0..BOARD_WIDTH).map(|file| {
            let coord = coord_to_notation((file, rank));
            let piece = board.get(&coord);
            let is_light = (file + rank) % 2 == 0;
            let selected_class = if Some(&coord) == selected.as_ref() { "selected" } else { "" };
            let legal_class = if legal_set.contains(&coord) { "legal" } else { "" };
            let glyph = piece_glyph(piece);
            let on_click = {
                let coord = coord.clone();
                let cb = on_cell_click.clone();
                Callback::from(move |_| cb.emit(coord.clone()))
            };
            html! {
                <button class={classes!("cell", if is_light { "light" } else { "dark" }, selected_class, legal_class)}
                    onclick={on_click}
                    aria-label={format!("Cell {coord}")}>
                    <span class="coord">{coord.clone()}</span>
                    <span class={classes!("piece", piece.map(|p| match p.color {
                        Color::White => "white",
                        Color::Black => "black",
                    }).unwrap_or(""))}>{glyph}</span>
                </button>
            }
        }).collect::<Html>();
        html! {
            <div class="row">
                <div class="rank-label">{rank + 1}</div>
                {cells}
            </div>
        }
    }).collect::<Html>();

    let file_labels = (0..BOARD_WIDTH)
        .map(|file| {
            let label = FILES.chars().nth(file as usize).unwrap_or('?');
            html! { <div class="file-label">{label}</div> }
        })
        .collect::<Html>();

    html! {
        <div class="board-wrapper">
            <div class="board">{rows}</div>
            <div class="file-row">
                <div class="file-spacer"></div>
                {file_labels}
            </div>
        </div>
    }
}

fn piece_glyph(piece: Option<&Piece>) -> String {
    match piece {
        None => "".into(),
        Some(p) => match p.kind {
            PieceType::Dux => {
                if p.color == Color::White {
                    "♔".into()
                } else {
                    "♚".into()
                }
            }
            PieceType::Soldier => {
                if p.color == Color::White {
                    "⛂".into()
                } else {
                    "⛀".into()
                }
            }
        },
    }
}

#[derive(Properties, PartialEq)]
struct BoardProps {
    board: HashMap<String, Piece>,
    selected: Option<String>,
    legal_moves: Vec<String>,
    on_cell_click: Callback<String>,
}

#[function_component(App)]
pub fn app() -> Html {
    let mode = use_state(|| MatchMode::Pva);
    let reload_key = use_state(|| 0u32);
    let current_match = use_state(|| Option::<MatchResponse>::None);
    let loading = use_state(|| false);
    let error = use_state(|| Option::<String>::None);
    let socket_error = use_state(|| Option::<String>::None);
    let selected = use_state(|| Option::<String>::None);
    let legal_targets = use_state(|| Vec::<String>::new());

    {
        let current_match = current_match.clone();
        let loading = loading.clone();
        let error = error.clone();
        let mode_dep = (*mode).clone();
        let reload_dep = *reload_key;
        use_effect_with(
            (mode_dep, reload_dep),
            move |(mode_value, _)| {
                loading.set(true);
                error.set(None);
                current_match.set(None);
                let current_match = current_match.clone();
                let loading = loading.clone();
                let error = error.clone();
                let mode_value = mode_value.clone();
                spawn_local(async move {
                    match api_create_match(mode_value).await {
                        Ok(res) => current_match.set(Some(res)),
                        Err(e) => error.set(Some(e)),
                    }
                    loading.set(false);
                });
                || ()
            },
        );
    }

    // WebSocket listener
    {
        let current_match = current_match.clone();
        let socket_error = socket_error.clone();
        let match_id_dep = (*current_match).as_ref().map(|m| m.id.clone());
        use_effect_with(
            match_id_dep,
            move |maybe_id| {
                if let Some(id) = maybe_id.clone() {
                    let ws_url = format!("{}/ws/match/{id}", ws_base());
                    let current_match = current_match.clone();
                    let socket_error = socket_error.clone();
                    spawn_local(async move {
                        match WebSocket::open(&ws_url) {
                            Ok(ws) => {
                                let (_, mut read) = ws.split();
                                while let Some(msg) = read.next().await {
                                    match msg {
                                        Ok(WsMessage::Text(txt)) => {
                                            if let Ok(parsed) = serde_json::from_str::<MatchResponse>(&txt) {
                                                current_match.set(Some(parsed));
                                                socket_error.set(None);
                                            } else {
                                                socket_error.set(Some("Failed to parse live update".into()));
                                            }
                                        }
                                        _ => {}
                                    }
                                }
                            }
                            Err(e) => socket_error.set(Some(format!("WebSocket error: {e}"))),
                        }
                    });
                }
                || ()
            },
        );
    }

    let board_map = {
        if let Some(m) = (*current_match).as_ref() {
            board_to_map(&m.state.board)
        } else {
            HashMap::new()
        }
    };

    let on_cell_click = {
        let current_match = current_match.clone();
        let selected = selected.clone();
        let legal_targets = legal_targets.clone();
        let loading = loading.clone();
        let error = error.clone();
        let board_map = board_map.clone();
        Callback::from(move |coord: String| {
            if let Some(mat) = (*current_match).as_ref() {
                if mat.state.winner.is_some() || !is_players_turn(mat) {
                    return;
                }
                if let Some(sel) = (*selected).clone() {
                    if legal_targets.contains(&coord) {
                        let mid = mat.id.clone();
                        let selected_coord = sel.clone();
                        let current_match = current_match.clone();
                        let selected = selected.clone();
                        let legal_targets = legal_targets.clone();
                        let loading = loading.clone();
                        let error = error.clone();
                        spawn_local(async move {
                            loading.set(true);
                            match api_post_move(&mid, &selected_coord, &coord).await {
                                Ok(res) => current_match.set(Some(res)),
                                Err(e) => error.set(Some(e)),
                            }
                            selected.set(None);
                            legal_targets.set(Vec::new());
                            loading.set(false);
                        });
                        return;
                    }
                }
                let piece = board_map.get(&coord);
                if let Some(p) = piece {
                    if p.color == mat.state.turn {
                        let legals = legal_moves_for(&mat.state.board, mat.state.turn, &coord);
                        selected.set(Some(coord.clone()));
                        legal_targets.set(legals);
                    }
                } else {
                    selected.set(None);
                    legal_targets.set(Vec::new());
                }
            }
        })
    };

    let handle_ai_step = {
        let current_match = current_match.clone();
        let loading = loading.clone();
        let error = error.clone();
        Callback::from(move |_| {
            if let Some(mat) = (*current_match).as_ref() {
                let id = mat.id.clone();
                let current_match = current_match.clone();
                let loading = loading.clone();
                let error = error.clone();
                spawn_local(async move {
                    loading.set(true);
                    match api_step_ai(&id).await {
                        Ok(res) => current_match.set(Some(res)),
                        Err(e) => error.set(Some(e)),
                    }
                    loading.set(false);
                });
            }
        })
    };

    let reset_match = {
        let reload_key = reload_key.clone();
        let mode = mode.clone();
        let selected = selected.clone();
        let legal_targets = legal_targets.clone();
        Callback::from(move |next_mode: MatchMode| {
            mode.set(next_mode);
            reload_key.set(*reload_key + 1);
            selected.set(None);
            legal_targets.set(Vec::new());
        })
    };

    html! {
        <div class="page">
            <header class="header">
                <div>
                    <h1>{"Parcae Strategy"}</h1>
                    <p class="subtitle">{"Kowalski/Stanway rules. Play vs AI or watch AI vs AI."}</p>
                </div>
                <div class="controls">
                    <button onclick={{
                        let reset_match = reset_match.clone();
                        Callback::from(move |_| reset_match.emit(MatchMode::Pva))
                    }}>{"New PvAI"}</button>
                    <button onclick={{
                        let reset_match = reset_match.clone();
                        Callback::from(move |_| reset_match.emit(MatchMode::Ava))
                    }}>{"New AI vs AI"}</button>
                    {
                        if let Some(mat) = (*current_match).as_ref() {
                            if mat.mode == MatchMode::Ava {
                                html! { <button onclick={handle_ai_step} disabled={*loading}>{"Step AI"}</button> }
                            } else {
                                html! {}
                            }
                        } else {
                            html! {}
                        }
                    }
                </div>
            </header>

            if let Some(err) = &*error {
                <div class="alert">{format!("Error: {err}")}</div>
            }
            if let Some(err) = &*socket_error {
                <div class="alert">{format!("Live update issue: {err}")}</div>
            }
            if *loading {
                <div class="hint">{"Working..."}</div>
            }
            if (*current_match).is_none() {
                <div class="hint">{"Spawning match..."}</div>
            }

            {
                if let Some(mat) = (*current_match).as_ref() {
                    html! {
                        <div class="content">
                            <div class="board-column">
                                <Board
                                    board={board_map.clone()}
                                    selected={(*selected).clone()}
                                    legal_moves={(*legal_targets).clone()}
                                    on_cell_click={on_cell_click.clone()}
                                />
                            </div>
                            <div class="sidebar">
                                <div class="card">
                                    <div class="label">{"Match"}</div>
                                    <div class="value">{format!("#{}", mat.id)}</div>
                                    <div class="label">{"Mode"}</div>
                                    <div class="value">{ match mat.mode { MatchMode::Pva => "Player vs AI", MatchMode::Ava => "AI vs AI" } }</div>
                                </div>

                                <div class="card">
                                    <div class="label">{"Turn"}</div>
                                    <div class="value turn">{ if mat.state.winner.is_some() { "Game over" } else if mat.state.turn == Color::White { "White" } else { "Black" } }</div>
                                    <div class="label">{"Captures"}</div>
                                    <div class="value">
                                        {format!("White {} / Black {}", mat.state.captures.get(&Color::White).unwrap_or(&0), mat.state.captures.get(&Color::Black).unwrap_or(&0))}
                                    </div>
                                </div>

                                <div class="card">
                                    <div class="label">{"History"}</div>
                                    <div class="history">
                                        {
                                            if mat.state.history.is_empty() {
                                                html! { <span>{"None yet"}</span> }
                                            } else {
                                                html! {
                                                    mat.state.history.iter().enumerate().map(|(idx, mv)| {
                                                        html! { <div key={format!("{}-{}", mv.origin, idx)}>{format!("{}. {} \u{2192} {}", idx + 1, mv.origin, mv.target)}</div> }
                                                    }).collect::<Html>()
                                                }
                                            }
                                        }
                                    </div>
                                </div>

                                {
                                    if let Some(winner) = mat.state.winner {
                                        html! {
                                            <div class="card winner">
                                                <div class="label">{"Winner"}</div>
                                                <div class="value">{ match winner { Color::White => "white", Color::Black => "black" } }</div>
                                                { mat.state.summary.as_ref().map(|s| html! { <div class="summary">{s.clone()}</div> }).unwrap_or_default() }
                                            </div>
                                        }
                                    } else {
                                        html! {}
                                    }
                                }
                            </div>
                        </div>
                    }
                } else {
                    html! {}
                }
            }
        </div>
    }
}

fn is_players_turn(mat: &MatchResponse) -> bool {
    match mat.mode {
        MatchMode::Pva => mat.state.turn == Color::White,
        MatchMode::Ava => true,
    }
}

#[wasm_bindgen(start)]
pub fn run_app() {
    wasm_logger::init(wasm_logger::Config::default());
    yew::Renderer::<App>::new().render();
}
