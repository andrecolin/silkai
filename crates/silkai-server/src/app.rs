use std::convert::Infallible;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::extract::{Query, State};
use axum::http::{header, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures_util::stream::{self, unfold};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use silkai_adapters::{ChatMessage, RunOptions};
use silkai_sched::JobId;
use tokio::sync::mpsc;

use crate::config::{load_from_path, AppConfig, ConfigError, UiConfig};
use crate::events::{self, Draft, EventLog};
use crate::runtime::{Runtime, RuntimeError};

pub(crate) struct AppState {
    pub(crate) config_path: Option<PathBuf>,
    pub(crate) runtime: tokio::sync::RwLock<Arc<Runtime>>,
    /// Shared across reloads so the history survives a config change.
    pub(crate) events: Arc<EventLog>,
    /// Fixed at startup; changing `[ui]` needs a restart.
    pub(crate) ui: UiConfig,
}

pub async fn app_from_config(cfg: AppConfig) -> Router {
    app_from_config_path(cfg, None).await
}

pub async fn app_from_path(path: impl AsRef<Path>) -> Result<Router, ConfigError> {
    let path = path.as_ref().to_path_buf();
    let cfg = load_from_path(&path)?;
    Ok(app_from_config_path(cfg, Some(path)).await)
}

pub async fn app_from_config_path(cfg: AppConfig, config_path: Option<PathBuf>) -> Router {
    let ui = cfg.ui.clone();
    router(config_path, build_runtime(cfg).await, ui)
}

#[cfg(feature = "test-util")]
pub async fn test_app() -> Router {
    app_from_config(clinic_cfg()).await
}

#[cfg(feature = "test-util")]
pub async fn test_app_with_disabled() -> Router {
    let mut cfg = clinic_cfg();
    cfg.disabled.push(too_big());
    app_from_config(cfg).await
}

#[cfg(feature = "test-util")]
pub async fn test_app_ws() -> Router {
    let mut cfg = clinic_cfg();
    for model in &mut cfg.enabled {
        if model.spec.name == "whisper" {
            model.transport = "websocket".into();
        }
    }
    app_from_config(cfg).await
}

#[cfg(feature = "test-util")]
pub async fn test_app_timeout_ms(ms: u64) -> Router {
    let mut cfg = clinic_cfg();
    cfg.request_timeout = std::time::Duration::from_millis(ms);
    app_from_config(cfg).await
}

/// The clinic app with the status page on or off and an optional token.
#[cfg(feature = "test-util")]
pub async fn test_app_ui(enabled: bool, token: Option<&str>) -> Router {
    let mut cfg = clinic_cfg();
    cfg.ui = UiConfig {
        enabled,
        token: token.map(str::to_string),
    };
    app_from_config(cfg).await
}

#[cfg(feature = "test-util")]
pub async fn test_app_llama_soap() -> Router {
    app_from_config(llama_soap_cfg()).await
}

#[cfg(feature = "test-util")]
fn llama_soap_cfg() -> AppConfig {
    let mut cfg = clinic_cfg();
    for model in &mut cfg.enabled {
        if model.spec.name == "soap" {
            model.engine = "llama.cpp".into();
        }
    }
    cfg
}

async fn build_runtime(cfg: AppConfig) -> Runtime {
    Runtime::new(cfg).await.expect("runtime")
}

fn router(config_path: Option<PathBuf>, rt: Runtime, ui: UiConfig) -> Router {
    let state = state_for(config_path, rt, ui);
    let open = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/status", get(status))
        .route("/v1/events", get(events_stream))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/session", get(crate::ws::session));
    let guarded = Router::new()
        .route("/ui", get(ui_page))
        .route("/metrics", get(metrics))
        .route("/admin/reload", post(reload))
        .route_layer(axum::middleware::from_fn_with_state(
            Arc::clone(&state),
            crate::auth::require_token,
        ));
    open.merge(guarded).with_state(state)
}

fn state_for(config_path: Option<PathBuf>, rt: Runtime, ui: UiConfig) -> Arc<AppState> {
    Arc::new(AppState {
        config_path,
        events: rt.events(),
        runtime: tokio::sync::RwLock::new(Arc::new(rt)),
        ui,
    })
}

const UI_PAGE: &str = include_str!("../ui/index.html");

async fn ui_page(State(state): State<Arc<AppState>>) -> Response {
    if !state.ui.enabled {
        return StatusCode::NOT_FOUND.into_response();
    }
    (
        [
            (header::CONTENT_TYPE, "text/html; charset=utf-8"),
            (header::CACHE_CONTROL, "no-store"),
        ],
        UI_PAGE,
    )
        .into_response()
}

async fn metrics(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let rt = runtime_of(&state).await;
    let body = crate::metrics::render(&rt.status(), &rt.counters());
    (
        [(
            header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        body,
    )
}

#[derive(Deserialize)]
struct EventsQuery {
    #[serde(default)]
    after: u64,
}

/// Replay the ring after `?after=<seq>`, then follow live. Subscribing
/// before reading the ring means nothing is lost in between; anything that
/// lands in both is dropped from the live side by sequence number.
async fn events_stream(
    State(state): State<Arc<AppState>>,
    Query(q): Query<EventsQuery>,
) -> impl IntoResponse {
    let log = Arc::clone(&state.events);
    let rx = log.subscribe();
    let replay = log.since(q.after);
    let last = replay.last().map(|e| e.seq).unwrap_or(q.after);
    let live = unfold((rx, last), |(mut rx, last)| async move {
        loop {
            match rx.recv().await {
                Ok(e) if e.seq <= last => continue,
                Ok(e) => {
                    let seq = e.seq;
                    return Some((sse_event(&e), (rx, seq)));
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {
                    return Some((Event::default().data(r#"{"kind":"lagged"}"#), (rx, last)));
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => return None,
            }
        }
    });
    let stream = stream::iter(replay.iter().map(sse_event).collect::<Vec<_>>())
        .chain(live)
        .map(Ok::<_, Infallible>);
    Sse::new(stream).keep_alive(KeepAlive::default())
}

fn sse_event(e: &events::Event) -> Event {
    Event::default()
        .id(e.seq.to_string())
        .data(serde_json::to_string(e).unwrap_or_default())
}

async fn health() -> &'static str {
    "ok"
}

async fn runtime_of(state: &AppState) -> Arc<Runtime> {
    state.runtime.read().await.clone()
}

async fn list_models(State(state): State<Arc<AppState>>) -> Json<ModelList> {
    let rt = runtime_of(&state).await;
    Json(ModelList::from_names(rt.configured_models()))
}

async fn status(State(state): State<Arc<AppState>>) -> Json<crate::status::Status> {
    Json(runtime_of(&state).await.status())
}

#[derive(Deserialize)]
struct ChatRequest {
    model: String,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    messages: Vec<WireMessage>,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    temperature: Option<f32>,
}

impl ChatRequest {
    fn options(&self) -> RunOptions {
        RunOptions {
            max_tokens: self.max_tokens,
            temperature: self.temperature,
        }
    }
}

/// One OpenAI-style message as clients send it. `content` is usually a
/// string; newer clients send a list of parts, of which the text parts are
/// joined. Anything else (images, tool calls) is dropped for now.
#[derive(Deserialize)]
struct WireMessage {
    #[serde(default = "default_role")]
    role: String,
    #[serde(default)]
    content: serde_json::Value,
}

fn default_role() -> String {
    "user".into()
}

impl WireMessage {
    fn into_chat(self) -> ChatMessage {
        ChatMessage::new(self.role, content_text(self.content))
    }
}

fn content_text(value: serde_json::Value) -> String {
    match value {
        serde_json::Value::String(s) => s,
        serde_json::Value::Array(parts) => parts
            .into_iter()
            .filter_map(|p| p.get("text").and_then(|t| t.as_str()).map(str::to_string))
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatRequest>,
) -> Response {
    let rt = runtime_of(&state).await;
    let opts = req.options();
    let model = req.model;
    let messages = messages_of(req.messages);
    match rt.submit_chat(&model, messages, opts).await {
        Ok((job, rx)) => {
            let meta = Meta::new(job, model);
            finish_chat(&rt, meta, rx, req.stream).await
        }
        Err(err) => chat_error(err).into_response(),
    }
}

/// The identifying fields every OpenAI-shaped reply carries.
#[derive(Clone)]
struct Meta {
    job: JobId,
    id: String,
    model: String,
    created: u64,
}

impl Meta {
    fn new(job: JobId, model: String) -> Self {
        Self {
            job,
            id: format!("chatcmpl-{}", job.0),
            model,
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        }
    }
}

async fn reload(State(state): State<Arc<AppState>>) -> StatusCode {
    let Some(path) = state.config_path.clone() else {
        return StatusCode::BAD_REQUEST;
    };
    match load_from_path(path) {
        Ok(cfg) => swap_if_idle(&state, cfg).await,
        Err(_) => StatusCode::BAD_REQUEST,
    }
}

async fn swap_if_idle(state: &AppState, cfg: AppConfig) -> StatusCode {
    let mut slot = state.runtime.write().await;
    if running_jobs(&slot) > 0 {
        return StatusCode::CONFLICT;
    }
    state.events.emit(Draft::new("reload"));
    match Runtime::rebuild(cfg, &slot).await {
        Ok(rt) => {
            *slot = Arc::new(rt);
            StatusCode::OK
        }
        Err(_) => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

fn running_jobs(rt: &Runtime) -> u32 {
    rt.status().models.iter().map(|m| m.running).sum()
}

fn messages_of(wire: Vec<WireMessage>) -> Vec<ChatMessage> {
    wire.into_iter().map(WireMessage::into_chat).collect()
}

async fn finish_chat(
    rt: &Runtime,
    meta: Meta,
    rx: mpsc::Receiver<String>,
    stream: bool,
) -> Response {
    if stream {
        stream_or_timeout(rt, meta, rx).await
    } else {
        json_or_timeout(rt, meta, rx).await
    }
}

async fn stream_or_timeout(rt: &Runtime, meta: Meta, mut rx: mpsc::Receiver<String>) -> Response {
    match tokio::time::timeout(rt.request_timeout(), rx.recv()).await {
        Ok(None) => match rt.take_rejection(meta.job) {
            Some(reason) => (StatusCode::BAD_REQUEST, reason).into_response(),
            None => sse_response(meta, None, rx),
        },
        Ok(first) => sse_response(meta, first, rx),
        Err(_) => timeout_drop(rt, meta.job).await,
    }
}

async fn json_or_timeout(rt: &Runtime, meta: Meta, rx: mpsc::Receiver<String>) -> Response {
    match tokio::time::timeout(rt.request_timeout(), collect_tokens(rx)).await {
        Ok(tokens) if tokens.is_empty() => match rt.take_rejection(meta.job) {
            Some(reason) => (StatusCode::BAD_REQUEST, reason).into_response(),
            None => json_completion(&meta, &tokens),
        },
        Ok(tokens) => json_completion(&meta, &tokens),
        Err(_) => timeout_drop(rt, meta.job).await,
    }
}

async fn timeout_drop(rt: &Runtime, job: JobId) -> Response {
    rt.drop_job(job).await;
    StatusCode::GATEWAY_TIMEOUT.into_response()
}

async fn collect_tokens(mut rx: mpsc::Receiver<String>) -> Vec<String> {
    let mut tokens = Vec::new();
    while let Some(token) = rx.recv().await {
        tokens.push(token);
    }
    tokens
}

/// The stream: a `: queued` comment, a role chunk, one chunk per token,
/// a closing chunk with `finish_reason: "stop"`, then `[DONE]`.
fn sse_response(meta: Meta, first: Option<String>, rx: mpsc::Receiver<String>) -> Response {
    let start = SsePhase::Comment { meta, first, rx };
    Sse::new(unfold(start, |phase| async move {
        let (event, next) = next_sse(phase).await?;
        Some((Ok::<_, Infallible>(event), next))
    }))
    .into_response()
}

enum SsePhase {
    Comment {
        meta: Meta,
        first: Option<String>,
        rx: mpsc::Receiver<String>,
    },
    Role {
        meta: Meta,
        pending: Option<String>,
        rx: mpsc::Receiver<String>,
    },
    Tokens {
        meta: Meta,
        pending: Option<String>,
        rx: mpsc::Receiver<String>,
    },
    Stop,
    Done,
}

async fn next_sse(phase: SsePhase) -> Option<(Event, SsePhase)> {
    match phase {
        SsePhase::Comment { meta, first, rx } => Some((
            Event::default().comment("queued"),
            SsePhase::Role {
                meta,
                pending: first,
                rx,
            },
        )),
        SsePhase::Role { meta, pending, rx } => Some((
            Event::default().data(chunk_json(&meta, Delta::Role, None)),
            SsePhase::Tokens { meta, pending, rx },
        )),
        SsePhase::Tokens { meta, pending, rx } => token_or_stop(meta, pending, rx).await,
        SsePhase::Stop => Some((Event::default().data("[DONE]"), SsePhase::Done)),
        SsePhase::Done => None,
    }
}

async fn token_or_stop(
    meta: Meta,
    pending: Option<String>,
    mut rx: mpsc::Receiver<String>,
) -> Option<(Event, SsePhase)> {
    let next = match pending {
        Some(token) => Some(token),
        None => rx.recv().await,
    };
    match next {
        Some(token) => Some((
            Event::default().data(chunk_json(&meta, Delta::Content(&token), None)),
            SsePhase::Tokens {
                meta,
                pending: None,
                rx,
            },
        )),
        None => Some((
            Event::default().data(chunk_json(&meta, Delta::Empty, Some("stop"))),
            SsePhase::Stop,
        )),
    }
}

enum Delta<'a> {
    Role,
    Content(&'a str),
    Empty,
}

fn chunk_json(meta: &Meta, delta: Delta<'_>, finish: Option<&str>) -> String {
    let delta = match delta {
        Delta::Role => serde_json::json!({"role": "assistant", "content": ""}),
        Delta::Content(text) => serde_json::json!({"content": text}),
        Delta::Empty => serde_json::json!({}),
    };
    serde_json::json!({
        "id": meta.id,
        "object": "chat.completion.chunk",
        "created": meta.created,
        "model": meta.model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish
        }]
    })
    .to_string()
}

fn json_completion(meta: &Meta, tokens: &[String]) -> Response {
    Json(serde_json::json!({
        "id": meta.id,
        "object": "chat.completion",
        "created": meta.created,
        "model": meta.model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": tokens.concat()
            },
            "finish_reason": "stop"
        }]
    }))
    .into_response()
}

/// Status plus the reason as the body, so a client sees why instead of a
/// bare code.
fn chat_error(err: RuntimeError) -> (StatusCode, String) {
    let status = match err {
        RuntimeError::Unknown => StatusCode::NOT_FOUND,
        RuntimeError::Disabled | RuntimeError::TooLarge => StatusCode::BAD_REQUEST,
        RuntimeError::Unavailable => StatusCode::SERVICE_UNAVAILABLE,
        RuntimeError::NoWebsocket => StatusCode::NOT_FOUND,
        RuntimeError::Interrupted | RuntimeError::Engine(_) => StatusCode::INTERNAL_SERVER_ERROR,
    };
    (status, err.to_string())
}

#[derive(Serialize)]
struct ModelList {
    object: &'static str,
    data: Vec<ModelEntry>,
}

#[derive(Serialize)]
struct ModelEntry {
    id: String,
    object: &'static str,
}

impl ModelList {
    fn from_names(names: Vec<String>) -> Self {
        Self {
            object: "list",
            data: names.into_iter().map(ModelEntry::new).collect(),
        }
    }
}

impl ModelEntry {
    fn new(id: String) -> Self {
        Self {
            id,
            object: "model",
        }
    }
}

/// The fixed "clinic" machine from the design spec: a 29 GB bench, a 96 GB
/// shelf, and three fake models. Every router test starts here.
#[cfg(feature = "test-util")]
fn clinic_cfg() -> AppConfig {
    use silkai_sched::clinic::{clinic_models, clinic_resources};
    AppConfig {
        listen: "127.0.0.1:0".into(),
        prefetch_on_start: true,
        request_timeout_secs: 600,
        request_timeout: std::time::Duration::from_secs(600),
        resources: clinic_resources(),
        enabled: clinic_models().into_iter().map(fake_model).collect(),
        disabled: vec![],
        ui: Default::default(),
    }
}

#[cfg(feature = "test-util")]
fn too_big() -> crate::config::ConfiguredModel {
    fake_model(silkai_sched::ModelSpec {
        name: "too-big".into(),
        vram_gb: 40.0,
        ram_gb: 40.0,
        priority: silkai_sched::Priority::Normal,
        exclusive: true,
        slots: 1,
        keep_warm: true,
        gpu: None,
        gpus: Vec::new(),
    })
}

#[cfg(feature = "test-util")]
fn fake_model(spec: silkai_sched::ModelSpec) -> crate::config::ConfiguredModel {
    crate::config::ConfiguredModel {
        engine: "fake".into(),
        path: format!("/models/{}.bin", spec.name),
        url: None,
        cmd: Vec::new(),
        transport: "http".into(),
        idle_timeout_secs: None,
        ctx_size: None,
        spec,
    }
}
