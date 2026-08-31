use std::sync::Arc;
use std::time::Duration;

use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use silkai_sched::clinic::{clinic_models, clinic_resources};
use silkai_sched::{JobId, ModelSpec, Priority, StatusSnapshot};
use tokio::sync::mpsc;

use crate::config::{AppConfig, ConfiguredModel};
use crate::runtime::{Runtime, RuntimeError};

pub async fn app_from_config(cfg: AppConfig) -> Router {
    let rt = Runtime::new(cfg).await.expect("runtime");
    router(Arc::new(rt))
}

pub async fn test_app() -> Router {
    app_from_config(clinic_cfg()).await
}

pub async fn test_app_with_disabled() -> Router {
    let mut cfg = clinic_cfg();
    cfg.disabled.push(too_big());
    app_from_config(cfg).await
}

pub async fn test_app_timeout_ms(ms: u64) -> Router {
    let mut cfg = clinic_cfg();
    cfg.request_timeout = Duration::from_millis(ms);
    app_from_config(cfg).await
}

fn router(rt: Arc<Runtime>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/status", get(status))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(rt)
}

async fn health() -> &'static str {
    "ok"
}

async fn list_models(State(rt): State<Arc<Runtime>>) -> Json<ModelList> {
    Json(ModelList::from_names(rt.configured_models()))
}

async fn status(State(rt): State<Arc<Runtime>>) -> Json<StatusSnapshot> {
    Json(rt.status())
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
    State(rt): State<Arc<Runtime>>,
    Json(req): Json<ChatRequest>,
) -> Response {
    match rt.submit_chat(&req.model, prompt_of(&req)).await {
        Ok((job, rx)) => finish_chat(&rt, job, rx, req.stream).await,
        Err(err) => chat_error(err).into_response(),
    }
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
    match timed_tokens(rt, rx).await {
        Ok(tokens) if stream => sse_response(job, &tokens),
        Ok(tokens) => json_completion(&tokens),
        Err(status) => status.into_response(),
    }
}

async fn timed_tokens(rt: &Runtime, rx: mpsc::Receiver<String>) -> Result<Vec<String>, StatusCode> {
    tokio::time::timeout(rt.request_timeout(), collect_tokens(rx))
        .await
        .map_err(|_| StatusCode::GATEWAY_TIMEOUT)
}

async fn collect_tokens(mut rx: mpsc::Receiver<String>) -> Vec<String> {
    let mut tokens = Vec::new();
    while let Some(token) = rx.recv().await {
        tokens.push(token);
    }
    tokens
}

fn sse_response(job: JobId, tokens: &[String]) -> Response {
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "text/event-stream")],
        sse_body(job, tokens),
    )
        .into_response()
}

fn sse_body(job: JobId, tokens: &[String]) -> String {
    let mut body = String::from(": queued\n\n");
    for token in tokens {
        body.push_str("data: ");
        body.push_str(&chunk_json(job, token));
        body.push_str("\n\n");
    }
    body.push_str("data: [DONE]\n\n");
    body
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
    })
}

fn fake_model(spec: ModelSpec) -> ConfiguredModel {
    ConfiguredModel {
        engine: "fake".into(),
        path: format!("/models/{}.bin", spec.name),
        transport: "http".into(),
        idle_timeout_secs: None,
        spec,
    }
}
