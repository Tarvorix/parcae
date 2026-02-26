use ai::{AIAgent, AiCapabilities, AiMoveMeta, AiProfile};
use axum::extract::{Path, Query, State, WebSocketUpgrade};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use engine::{
    apply_move, coord_to_notation, initial_state, legal_moves, serialize_state, Color, GameState,
    Move, SerializedState,
};
use futures_util::{SinkExt, StreamExt};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::broadcast;
use tower_http::cors::{Any, CorsLayer};
use tracing::info;
use uuid::Uuid;

const AGENT_STALE_AFTER_SECS: u64 = 15;
const AGENT_PRUNE_AFTER_SECS: u64 = 300;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum MatchMode {
    Pva,
    Pvg,
    Gvg,
    Gva,
    Ava,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum Controller {
    Player,
    Agent,
    Ai,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CreateMatchRequest {
    #[serde(default = "default_mode")]
    mode: MatchMode,
    #[serde(default)]
    ai_profiles: MatchAiProfiles,
}

fn default_mode() -> MatchMode {
    MatchMode::Pva
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MoveRequest {
    origin: String,
    target: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct MatchAiProfiles {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    white: Option<AiProfile>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    black: Option<AiProfile>,
}

impl MatchAiProfiles {
    fn for_color(&self, color: Color) -> Option<&AiProfile> {
        match color {
            Color::White => self.white.as_ref(),
            Color::Black => self.black.as_ref(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MatchResponse {
    id: String,
    mode: MatchMode,
    players: Players,
    state: SerializedState,
    ai_profiles: MatchAiProfiles,
    #[serde(skip_serializing_if = "Option::is_none")]
    last_ai_meta: Option<AiMoveMeta>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Players {
    white: Controller,
    black: Controller,
}

impl Default for Players {
    fn default() -> Self {
        Self {
            white: Controller::Player,
            black: Controller::Ai,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LegalQuery {
    origin: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LegalMoveResponse {
    origin: String,
    target: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LegalMovesResponse {
    id: String,
    turn: Color,
    origin: Option<String>,
    moves: Vec<LegalMoveResponse>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AgentHeartbeatRequest {
    #[serde(default = "default_agent_id")]
    agent_id: String,
}

fn default_agent_id() -> String {
    "default-agent".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AgentStatusEntry {
    agent_id: String,
    connected: bool,
    last_seen_unix_ms: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AgentStatusResponse {
    connected: bool,
    active_count: usize,
    stale_after_seconds: u64,
    agents: Vec<AgentStatusEntry>,
}

#[derive(Debug)]
struct MatchEntry {
    mode: MatchMode,
    players: Players,
    state: Mutex<GameState>,
    ai_profiles: MatchAiProfiles,
    last_ai_meta: Mutex<Option<AiMoveMeta>>,
}

#[derive(Clone)]
struct AppState {
    matches: Arc<dashmap::DashMap<String, Arc<MatchEntry>>>,
    hub: Arc<dashmap::DashMap<String, broadcast::Sender<MatchResponse>>>,
    agents: Arc<dashmap::DashMap<String, i64>>,
    ai: Arc<AIAgent>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();

    let ai_model_path = std::env::var("PARCAE_MODEL_PATH").ok().map(std::path::PathBuf::from);
    let ai = Arc::new(AIAgent::new(
        ai_model_path,
        64,
        1.25,
    ));
    let state = AppState {
        matches: Arc::new(dashmap::DashMap::new()),
        hub: Arc::new(dashmap::DashMap::new()),
        agents: Arc::new(dashmap::DashMap::new()),
        ai,
    };

    let app = Router::new()
        .route("/health", get(health))
        .route("/ai/capabilities", get(ai_capabilities))
        .route("/agent/status", get(agent_status))
        .route("/agent/heartbeat", post(agent_heartbeat))
        .route("/match", post(create_match))
        .route("/match/:id", get(get_match))
        .route("/match/:id/legal", get(get_legal))
        .route("/match/:id/move", post(play_move))
        .route("/match/:id/ai-step", post(step_ai))
        .route("/ws/match/:id", get(ws_match))
        .with_state(state)
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        );

    let host = std::env::var("HOST").unwrap_or_else(|_| "0.0.0.0".into());
    let port: u16 = std::env::var("PORT")
        .unwrap_or_else(|_| "8000".into())
        .parse()
        .unwrap_or(8000);
    let addr: SocketAddr = format!("{host}:{port}").parse()?;
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    info!("Starting server on http://{addr}");
    axum::serve(listener, app.into_make_service()).await?;
    Ok(())
}

async fn health() -> impl IntoResponse {
    Json(serde_json::json!({"status": "ok"}))
}

async fn ai_capabilities(State(app): State<AppState>) -> impl IntoResponse {
    let caps: AiCapabilities = app.ai.capabilities();
    Json(caps)
}

fn now_unix_ms() -> i64 {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(delta) => delta.as_millis() as i64,
        Err(_) => 0,
    }
}

fn build_agent_status(app: &AppState) -> AgentStatusResponse {
    let now = now_unix_ms();
    let stale_window = Duration::from_secs(AGENT_STALE_AFTER_SECS).as_millis() as i64;
    let prune_window = Duration::from_secs(AGENT_PRUNE_AFTER_SECS).as_millis() as i64;
    let mut stale_ids = Vec::new();
    let mut agents = Vec::new();

    for entry in app.agents.iter() {
        let agent_id = entry.key().clone();
        let last_seen = *entry.value();
        let age_ms = now.saturating_sub(last_seen);
        if age_ms > prune_window {
            stale_ids.push(agent_id.clone());
        }
        agents.push(AgentStatusEntry {
            agent_id,
            connected: age_ms <= stale_window,
            last_seen_unix_ms: last_seen,
        });
    }

    for stale in stale_ids {
        app.agents.remove(&stale);
    }

    agents.sort_by(|a, b| a.agent_id.cmp(&b.agent_id));
    let active_count = agents.iter().filter(|a| a.connected).count();
    AgentStatusResponse {
        connected: active_count > 0,
        active_count,
        stale_after_seconds: AGENT_STALE_AFTER_SECS,
        agents,
    }
}

async fn agent_status(State(app): State<AppState>) -> impl IntoResponse {
    Json(build_agent_status(&app))
}

async fn agent_heartbeat(
    State(app): State<AppState>,
    Json(req): Json<AgentHeartbeatRequest>,
) -> impl IntoResponse {
    let trimmed = req.agent_id.trim();
    let agent_id = if trimmed.is_empty() {
        default_agent_id()
    } else {
        trimmed.to_string()
    };
    app.agents.insert(agent_id, now_unix_ms());
    Json(build_agent_status(&app))
}

async fn create_match(
    State(app): State<AppState>,
    Json(req): Json<CreateMatchRequest>,
) -> Result<Json<MatchResponse>, ApiError> {
    let id = Uuid::new_v4().to_string()[..8].to_string();
    let mut state = initial_state(5).map_err(|e| ApiError::bad_request(e.to_string()))?;
    let players = players_for_mode(&req.mode);
    let ai_profiles = effective_ai_profiles(&players, req.ai_profiles, &app.ai.default_profile());

    let mut last_ai_meta = None;
    if req.mode == MatchMode::Ava {
        last_ai_meta = run_ai_turns(&app.ai, &mut state, &ai_profiles, Some(1))
            .map_err(|e| ApiError::server(e.to_string()))?;
    }

    let entry = Arc::new(MatchEntry {
        mode: req.mode.clone(),
        players: players.clone(),
        state: Mutex::new(state),
        ai_profiles: ai_profiles.clone(),
        last_ai_meta: Mutex::new(last_ai_meta),
    });
    app.matches.insert(id.clone(), entry);
    let payload = serialize_match(&id, &app);
    broadcast_match(&app, &id, &payload);
    Ok(Json(payload))
}

async fn get_match(
    State(app): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<MatchResponse>, ApiError> {
    if app.matches.get(&id).is_none() {
        return Err(ApiError::not_found("match not found"));
    }
    Ok(Json(serialize_match(&id, &app)))
}

async fn get_legal(
    State(app): State<AppState>,
    Path(id): Path<String>,
    Query(query): Query<LegalQuery>,
) -> Result<Json<LegalMovesResponse>, ApiError> {
    let entry = app
        .matches
        .get(&id)
        .ok_or_else(|| ApiError::not_found("match not found"))?;
    let state = entry.state.lock();
    let mut moves: Vec<LegalMoveResponse> = Vec::new();
    if let Some(origin_token) = query.origin.as_ref() {
        let origin = engine::notation_to_coord(origin_token)
            .map_err(|e| ApiError::bad_request(e.to_string()))?;
        if let Some(piece) = state.board.get(&origin) {
            if piece.color == state.turn {
                moves = legal_moves(&state, origin)
                    .into_iter()
                    .map(|target| LegalMoveResponse {
                        origin: coord_to_notation(origin),
                        target: coord_to_notation(target),
                    })
                    .collect();
            }
        }
    } else {
        for (origin, piece) in state.board.iter() {
            if piece.color != state.turn {
                continue;
            }
            for target in legal_moves(&state, *origin) {
                moves.push(LegalMoveResponse {
                    origin: coord_to_notation(*origin),
                    target: coord_to_notation(target),
                });
            }
        }
        moves.sort_by(|a, b| {
            a.origin
                .cmp(&b.origin)
                .then_with(|| a.target.cmp(&b.target))
        });
    }

    Ok(Json(LegalMovesResponse {
        id: id.clone(),
        turn: state.turn,
        origin: query.origin.as_ref().map(|v| v.to_ascii_uppercase()),
        moves,
    }))
}

async fn play_move(
    State(app): State<AppState>,
    Path(id): Path<String>,
    Json(body): Json<MoveRequest>,
) -> Result<Json<MatchResponse>, ApiError> {
    let entry = app
        .matches
        .get(&id)
        .ok_or_else(|| ApiError::not_found("match not found"))?;
    let latest_meta = {
        let mut state = entry.state.lock();
        if state.winner.is_some() {
            return Err(ApiError::bad_request("Match already finished"));
        }
        let origin = engine::notation_to_coord(&body.origin)
            .map_err(|e| ApiError::bad_request(e.to_string()))?;
        let target = engine::notation_to_coord(&body.target)
            .map_err(|e| ApiError::bad_request(e.to_string()))?;
        let mv = Move { origin, target };
        let next = apply_move(&state, mv).map_err(|e| ApiError::bad_request(e.to_string()))?;
        *state = next;

        run_ai_turns(&app.ai, &mut state, &entry.ai_profiles, None)
            .map_err(|e| ApiError::server(e.to_string()))?
    };

    if let Some(meta) = latest_meta {
        *entry.last_ai_meta.lock() = Some(meta);
    }

    let payload = serialize_match(&id, &app);
    broadcast_match(&app, &id, &payload);
    Ok(Json(payload))
}

async fn step_ai(
    State(app): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<MatchResponse>, ApiError> {
    let entry = app
        .matches
        .get(&id)
        .ok_or_else(|| ApiError::not_found("match not found"))?;
    let latest_meta = {
        let mut state = entry.state.lock();

        if entry.ai_profiles.for_color(state.turn).is_none() {
            return Err(ApiError::bad_request(
                "No AI is configured for the current turn",
            ));
        }

        run_ai_turns(&app.ai, &mut state, &entry.ai_profiles, Some(1))
            .map_err(|e| ApiError::server(e.to_string()))?
    };

    if let Some(meta) = latest_meta {
        *entry.last_ai_meta.lock() = Some(meta);
    }

    let payload = serialize_match(&id, &app);
    broadcast_match(&app, &id, &payload);
    Ok(Json(payload))
}

async fn ws_match(
    State(app): State<AppState>,
    Path(id): Path<String>,
    ws: WebSocketUpgrade,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_ws(app, id, socket))
}

async fn handle_ws(app: AppState, id: String, socket: axum::extract::ws::WebSocket) {
    use axum::extract::ws::Message;
    let (mut sender, mut receiver) = socket.split();
    let rx = {
        let entry = app.matches.get(&id);
        if entry.is_none() {
            let _ = sender.send(Message::Text("match not found".into())).await;
            return;
        }
        let hub = app
            .hub
            .entry(id.clone())
            .or_insert_with(|| {
                let (tx, _rx) = broadcast::channel(16);
                tx
            })
            .clone();
        let rx = hub.subscribe();
        let payload = serialize_match(&id, &app);
        let _ = sender
            .send(Message::Text(serde_json::to_string(&payload).unwrap()))
            .await;
        rx
    };

    let mut send_task = tokio::spawn(async move {
        let mut rx = rx;
        while let Ok(payload) = rx.recv().await {
            if sender
                .send(Message::Text(serde_json::to_string(&payload).unwrap()))
                .await
                .is_err()
            {
                break;
            }
        }
    });

    let mut recv_task = tokio::spawn(async move {
        while let Some(Ok(msg)) = receiver.next().await {
            if let Message::Close(_) = msg {
                break;
            }
        }
    });

    tokio::select! {
        _ = (&mut send_task) => recv_task.abort(),
        _ = (&mut recv_task) => send_task.abort(),
    }
}

fn players_for_mode(mode: &MatchMode) -> Players {
    match mode {
        MatchMode::Pva => Players {
            white: Controller::Player,
            black: Controller::Ai,
        },
        MatchMode::Pvg => Players {
            white: Controller::Player,
            black: Controller::Agent,
        },
        MatchMode::Gvg => Players {
            white: Controller::Agent,
            black: Controller::Agent,
        },
        MatchMode::Gva => Players {
            white: Controller::Agent,
            black: Controller::Ai,
        },
        MatchMode::Ava => Players {
            white: Controller::Ai,
            black: Controller::Ai,
        },
    }
}

fn effective_ai_profiles(
    players: &Players,
    requested: MatchAiProfiles,
    default_profile: &AiProfile,
) -> MatchAiProfiles {
    let mut out = MatchAiProfiles::default();

    if players.white == Controller::Ai {
        out.white = Some(requested.white.unwrap_or_else(|| default_profile.clone()));
    }
    if players.black == Controller::Ai {
        out.black = Some(requested.black.unwrap_or_else(|| default_profile.clone()));
    }

    out
}

fn run_ai_turns(
    ai: &AIAgent,
    state: &mut GameState,
    ai_profiles: &MatchAiProfiles,
    max_plies: Option<usize>,
) -> anyhow::Result<Option<AiMoveMeta>> {
    let mut plies = 0usize;
    let mut latest_meta: Option<AiMoveMeta> = None;

    while state.active() {
        if let Some(limit) = max_plies {
            if plies >= limit {
                break;
            }
        }

        let Some(profile) = ai_profiles.for_color(state.turn) else {
            break;
        };

        let selection = ai
            .select_move_with_profile(state, profile)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;

        latest_meta = Some(selection.meta.clone());

        let Some(mv) = selection.mv else {
            state.winner = Some(state.turn.opponent());
            state.summary = Some("AI has no legal moves.".into());
            break;
        };

        let next = apply_move(state, mv)?;
        *state = next;
        plies += 1;
    }

    Ok(latest_meta)
}

fn serialize_match(id: &str, app: &AppState) -> MatchResponse {
    if let Some(entry) = app.matches.get(id) {
        let state = entry.state.lock().clone();
        return MatchResponse {
            id: id.to_string(),
            mode: entry.mode.clone(),
            players: entry.players.clone(),
            state: serialize_state(&state),
            ai_profiles: entry.ai_profiles.clone(),
            last_ai_meta: entry.last_ai_meta.lock().clone(),
        };
    }

    let state = GameState::default();
    MatchResponse {
        id: id.to_string(),
        mode: MatchMode::Pva,
        players: Players::default(),
        state: serialize_state(&state),
        ai_profiles: MatchAiProfiles::default(),
        last_ai_meta: None,
    }
}

fn broadcast_match(app: &AppState, id: &str, payload: &MatchResponse) {
    if let Some(sender) = app.hub.get(id) {
        let _ = sender.send(payload.clone());
    } else {
        let (tx, _rx) = broadcast::channel(16);
        let _ = tx.send(payload.clone());
        app.hub.insert(id.to_string(), tx);
    }
}

#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    message: String,
}

impl ApiError {
    fn not_found(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            message: msg.into(),
        }
    }

    fn bad_request(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: msg.into(),
        }
    }

    fn server(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: msg.into(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        let body = serde_json::json!({ "detail": self.message });
        (self.status, Json(body)).into_response()
    }
}
