use std::convert::Infallible;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures_util::stream::unfold;
use serde::{Deserialize, Serialize};
use silkai_sched::clinic::{clinic_models, clinic_resources};
use silkai_sched::{JobId, ModelSpec, Priority, StatusSnapshot};
use tokio::sync::mpsc;

use crate::config::{load_from_path, AppConfig, ConfigError, ConfiguredModel};
use crate::runtime::{Runtime, RuntimeError};

pub(crate) struct AppState {
    pub(crate) config_path: Option<PathBuf>,
    pub(crate) runtime: tokio::sync::RwLock<Arc<Runtime>>,
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
    router(config_path, build_runtime(cfg).await)
}

pub async fn test_app() -> Router {
    app_from_config(clinic_cfg()).await
}

pub async fn test_app_with_disabled() -> Router {
    let mut cfg = clinic_cfg();
    cfg.disabled.push(too_big());
    app_from_config(cfg).await
}

pub async fn test_app_ws() -> Router {
    let mut cfg = clinic_cfg();
    for model in &mut cfg.enabled {
        if model.spec.name == "whisper" {
            model.transport = "websocket".into();
        }
    }
    app_from_config(cfg).await
}

pub async fn test_app_timeout_ms(ms: u64) -> Router {
    let mut cfg = clinic_cfg();
    cfg.request_timeout = Duration::from_millis(ms);
    app_from_config(cfg).await
}

pub async fn test_app_llama_soap() -> Router {
    app_from_config(llama_soap_cfg()).await
}

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

fn router(config_path: Option<PathBuf>, rt: Runtime) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/status", get(status))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/session", get(crate::ws::session))
        .route("/admin/reload", post(reload))
        .with_state(state_for(config_path, rt))
}

fn state_for(config_path: Option<PathBuf>, rt: Runtime) -> Arc<AppState> {
    Arc::new(AppState {
        config_path,
        runtime: tokio::sync::RwLock::new(Arc::new(rt)),
    })
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

async fn status(State(state): State<Arc<AppState>>) -> Json<StatusSnapshot> {
    Json(runtime_of(&state).await.status())
}

#[derive(Deserialize)]
struct ChatRequest {
    model: String,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    messages: Vec<ChatMessage>,
}

#[derive(Deserialize)]
struct ChatMessage {
    #[serde(default)]
    content: String,
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatRequest>,
) -> Response {
    let rt = runtime_of(&state).await;
    match rt.submit_chat(&req.model, prompt_of(&req)).await {
        Ok((job, rx)) => finish_chat(&rt, job, rx, req.stream).await,
        Err(err) => chat_error(err).into_response(),
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
    match Runtime::new(cfg).await {
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

fn prompt_of(req: &ChatRequest) -> &str {
    req.messages
        .last()
        .map(|m| m.content.as_str())
        .unwrap_or("")
}

async fn finish_chat(
    rt: &Runtime,
    job: JobId,
    rx: mpsc::Receiver<String>,
    stream: bool,
) -> Response {
    if stream {
        stream_or_timeout(rt, job, rx).await
    } else {
        json_or_timeout(rt, job, rx).await
    }
}

async fn stream_or_timeout(rt: &Runtime, job: JobId, mut rx: mpsc::Receiver<String>) -> Response {
    match tokio::time::timeout(rt.request_timeout(), rx.recv()).await {
        Ok(first) => sse_response(job, first, rx),
        Err(_) => timeout_drop(rt, job).await,
    }
}

async fn json_or_timeout(rt: &Runtime, job: JobId, rx: mpsc::Receiver<String>) -> Response {
    match tokio::time::timeout(rt.request_timeout(), collect_tokens(rx)).await {
        Ok(tokens) => json_completion(&tokens),
        Err(_) => timeout_drop(rt, job).await,
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

fn sse_response(job: JobId, first: Option<String>, rx: mpsc::Receiver<String>) -> Response {
    let start = SsePhase::Comment { job, first, rx };
    Sse::new(unfold(start, |phase| async move {
        let (event, next) = next_sse(phase).await?;
        Some((Ok::<_, Infallible>(event), next))
    }))
    .into_response()
}

enum SsePhase {
    Comment {
        job: JobId,
        first: Option<String>,
        rx: mpsc::Receiver<String>,
    },
    Tokens {
        job: JobId,
        pending: Option<String>,
        rx: mpsc::Receiver<String>,
    },
    Done,
}

async fn next_sse(phase: SsePhase) -> Option<(Event, SsePhase)> {
    match phase {
        SsePhase::Comment { job, first, rx } => Some(queued_then(job, first, rx)),
        SsePhase::Tokens { job, pending, rx } => token_or_done(job, pending, rx).await,
        SsePhase::Done => None,
    }
}

fn queued_then(job: JobId, first: Option<String>, rx: mpsc::Receiver<String>) -> (Event, SsePhase) {
    (
        Event::default().comment("queued"),
        SsePhase::Tokens {
            job,
            pending: first,
            rx,
        },
    )
}

async fn token_or_done(
    job: JobId,
    pending: Option<String>,
    mut rx: mpsc::Receiver<String>,
) -> Option<(Event, SsePhase)> {
    match pending {
        Some(token) => Some(data_then_rest(job, token, rx)),
        None => match rx.recv().await {
            Some(token) => Some(data_then_rest(job, token, rx)),
            None => Some((Event::default().data("[DONE]"), SsePhase::Done)),
        },
    }
}

fn data_then_rest(job: JobId, token: String, rx: mpsc::Receiver<String>) -> (Event, SsePhase) {
    (
        data_event(job, &token),
        SsePhase::Tokens {
            job,
            pending: None,
            rx,
        },
    )
}

fn data_event(job: JobId, token: &str) -> Event {
    Event::default().data(chunk_json(job, token))
}

fn chunk_json(job: JobId, token: &str) -> String {
    serde_json::json!({
        "id": format!("job-{}", job.0),
        "object": "chat.completion.chunk",
        "choices": [{
            "index": 0,
            "delta": {"content": token},
            "finish_reason": null
        }]
    })
    .to_string()
}

fn json_completion(tokens: &[String]) -> Response {
    Json(serde_json::json!({
        "choices": [{
            "message": {
                "role": "assistant",
                "content": tokens.concat()
            }
        }]
    }))
    .into_response()
}

fn chat_error(err: RuntimeError) -> StatusCode {
    match err {
        RuntimeError::Unknown => StatusCode::NOT_FOUND,
        RuntimeError::Disabled | RuntimeError::TooLarge => StatusCode::BAD_REQUEST,
        RuntimeError::Unavailable => StatusCode::SERVICE_UNAVAILABLE,
        RuntimeError::NoWebsocket => StatusCode::NOT_FOUND,
        RuntimeError::Engine(_) => StatusCode::INTERNAL_SERVER_ERROR,
    }
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

fn clinic_cfg() -> AppConfig {
    AppConfig {
        listen: "127.0.0.1:0".into(),
        prefetch_on_start: true,
        request_timeout_secs: 600,
        request_timeout: Duration::from_secs(600),
        resources: clinic_resources(),
        enabled: clinic_models().into_iter().map(fake_model).collect(),
        disabled: vec![],
    }
}

fn too_big() -> ConfiguredModel {
    fake_model(ModelSpec {
        name: "too-big".into(),
        vram_gb: 40.0,
        ram_gb: 40.0,
        priority: Priority::Normal,
        exclusive: true,
        slots: 1,
        keep_warm: true,
        gpu: None,
        gpus: Vec::new(),
    })
}

fn fake_model(spec: ModelSpec) -> ConfiguredModel {
    ConfiguredModel {
        engine: "fake".into(),
        path: format!("/models/{}.bin", spec.name),
        url: None,
        cmd: Vec::new(),
        transport: "http".into(),
        idle_timeout_secs: None,
        spec,
    }
}
