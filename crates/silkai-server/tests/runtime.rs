use std::time::Duration;

use axum::http::{header, StatusCode};
use axum::response::IntoResponse;
use axum::routing::post;
use axum::Router;
use silkai_adapters::FakeEngine;
use silkai_sched::clinic::{clinic_models, clinic_resources};
use silkai_sched::{ModelSpec, Priority, Resources, Tier};
use silkai_server::config::{AppConfig, ConfiguredModel};
use silkai_server::runtime::{Runtime, RuntimeError};
use tokio::net::TcpListener;

fn clinic_cfg() -> AppConfig {
    let models = clinic_models()
        .into_iter()
        .map(|spec| ConfiguredModel {
            engine: "fake".into(),
            path: format!("/models/{}.bin", spec.name),
            url: None,
            cmd: Vec::new(),
            transport: "http".into(),
            idle_timeout_secs: None,
            spec,
        })
        .collect();
    AppConfig {
        listen: "127.0.0.1:0".into(),
        prefetch_on_start: true,
        request_timeout_secs: 600,
        request_timeout: Duration::from_secs(600),
        resources: clinic_resources(),
        enabled: models,
        disabled: vec![],
    }
}

fn too_big() -> ConfiguredModel {
    ConfiguredModel {
        engine: "fake".into(),
        path: "/models/too-big.bin".into(),
        url: None,
        cmd: Vec::new(),
        transport: "http".into(),
        idle_timeout_secs: None,
        spec: ModelSpec {
            name: "too-big".into(),
            vram_gb: 40.0,
            ram_gb: 40.0,
            priority: Priority::Normal,
            exclusive: true,
            slots: 1,
            keep_warm: true,
            gpu: None,
        },
    }
}

#[tokio::test]
async fn prefetch_then_soap_wakes_fake_engine() {
    let rt = Runtime::new(clinic_cfg()).await.unwrap();
    let (job, mut tokens) = rt.submit_chat("soap", "note").await.unwrap();
    let mut out = String::new();
    while let Some(t) = tokens.recv().await {
        out.push_str(&t);
    }
    rt.finished(job).await;
    assert_eq!(out, "note world");
    let st = rt.status();
    let soap = st.models.iter().find(|m| m.name == "soap").unwrap();
    assert_eq!(soap.running, 0);
}

#[tokio::test]
async fn unknown_model_errors() {
    let rt = Runtime::new(clinic_cfg()).await.unwrap();
    let err = rt.submit_chat("nope", "x").await.unwrap_err();
    assert!(matches!(err, RuntimeError::Unknown));
}

#[tokio::test]
async fn disabled_model_errors() {
    let mut cfg = clinic_cfg();
    cfg.disabled.push(too_big());
    let rt = Runtime::new(cfg).await.unwrap();
    let err = rt.submit_chat("too-big", "x").await.unwrap_err();
    assert!(matches!(err, RuntimeError::Disabled));
}

#[cfg(not(feature = "llama"))]
fn llama_soap_cfg() -> AppConfig {
    let mut cfg = clinic_cfg();
    for model in &mut cfg.enabled {
        if model.spec.name == "soap" {
            model.engine = "llama.cpp".into();
        }
    }
    cfg
}

#[cfg(not(feature = "llama"))]
#[tokio::test]
async fn llama_cpp_submit_unavailable_without_feature() {
    let rt = Runtime::new(llama_soap_cfg()).await.unwrap();
    let err = rt.submit_chat("soap", "x").await.unwrap_err();
    assert!(
        matches!(err, RuntimeError::Unavailable),
        "expected Unavailable, got {err:?}"
    );
}

fn nope_soap_cfg() -> AppConfig {
    let mut cfg = clinic_cfg();
    for model in &mut cfg.enabled {
        if model.spec.name == "soap" {
            model.engine = "nope".into();
        }
    }
    cfg
}

fn vllm_soap_cfg(url: Option<&str>) -> AppConfig {
    let mut cfg = clinic_cfg();
    for model in &mut cfg.enabled {
        if model.spec.name == "soap" {
            model.engine = "vllm".into();
            model.path = "Qwen/Qwen3-0.6B".into();
            model.url = url.map(str::to_string);
        }
    }
    cfg
}

#[tokio::test]
async fn vllm_engine_is_known() {
    let rt = Runtime::new(vllm_soap_cfg(Some("http://127.0.0.1:1")))
        .await
        .unwrap();
    let err = rt.submit_chat("soap", "x").await.unwrap_err();
    assert!(
        !matches!(err, RuntimeError::Unavailable | RuntimeError::Unknown),
        "expected an engine error, got {err:?}"
    );
}

#[tokio::test]
async fn vllm_submit_streams_from_http_engine() {
    let url = spawn_vllm_mock().await;
    let rt = Runtime::new(vllm_soap_cfg(Some(&url))).await.unwrap();
    let (job, mut tokens) = rt.submit_chat("soap", "note").await.unwrap();
    let mut out = String::new();
    while let Some(t) = tokens.recv().await {
        out.push_str(&t);
    }
    rt.finished(job).await;
    assert_eq!(out, "hello world");
}

fn process_soap_cfg(url: Option<&str>) -> AppConfig {
    let mut cfg = clinic_cfg();
    for model in &mut cfg.enabled {
        if model.spec.name == "soap" {
            model.engine = "process".into();
            model.path = "Qwen/Qwen3-0.6B".into();
            model.url = url.map(str::to_string);
            model.cmd = vec!["sleep".into(), "30".into()];
        }
    }
    cfg
}

#[tokio::test]
async fn process_submit_streams_from_spawned_http() {
    let url = spawn_vllm_mock().await;
    let rt = Runtime::new(process_soap_cfg(Some(&url))).await.unwrap();
    let (job, mut tokens) = rt.submit_chat("soap", "note").await.unwrap();
    let mut out = String::new();
    while let Some(t) = tokens.recv().await {
        out.push_str(&t);
    }
    rt.finished(job).await;
    assert_eq!(out, "hello world");
}

fn ollama_soap_cfg(url: Option<&str>) -> AppConfig {
    let mut cfg = clinic_cfg();
    for model in &mut cfg.enabled {
        if model.spec.name == "soap" {
            model.engine = "ollama".into();
            model.path = "llama3.2".into();
            model.url = url.map(str::to_string);
        }
    }
    cfg
}

#[tokio::test]
async fn ollama_engine_is_known() {
    let rt = Runtime::new(ollama_soap_cfg(Some("http://127.0.0.1:1")))
        .await
        .unwrap();
    let err = rt.submit_chat("soap", "x").await.unwrap_err();
    assert!(
        !matches!(err, RuntimeError::Unavailable | RuntimeError::Unknown),
        "expected an engine error, got {err:?}"
    );
}

#[tokio::test]
async fn ollama_submit_streams_from_http_engine() {
    let url = spawn_ollama_mock().await;
    let rt = Runtime::new(ollama_soap_cfg(Some(&url))).await.unwrap();
    let (job, mut tokens) = rt.submit_chat("soap", "note").await.unwrap();
    let mut out = String::new();
    while let Some(t) = tokens.recv().await {
        out.push_str(&t);
    }
    rt.finished(job).await;
    assert_eq!(out, "hello world");
}

fn crashy_cfg(name: &str) -> AppConfig {
    AppConfig {
        listen: "127.0.0.1:0".into(),
        prefetch_on_start: false,
        request_timeout_secs: 600,
        request_timeout: Duration::from_secs(600),
        resources: Resources::single(29.0, 96.0),
        enabled: vec![ConfiguredModel {
            engine: "fake".into(),
            path: format!("/models/{name}.bin"),
            url: None,
            cmd: Vec::new(),
            transport: "http".into(),
            idle_timeout_secs: None,
            spec: ModelSpec {
                name: name.into(),
                vram_gb: 8.0,
                ram_gb: 8.0,
                priority: Priority::Normal,
                exclusive: true,
                slots: 1,
                keep_warm: true,
                gpu: None,
            },
        }],
        disabled: vec![],
    }
}

fn model_status(rt: &Runtime, name: &str) -> (Tier, u32) {
    let m = rt
        .status()
        .models
        .into_iter()
        .find(|m| m.name == name)
        .unwrap();
    (m.tier, m.running)
}

async fn collect_chat(rt: &Runtime, model: &str, prompt: &str) -> String {
    let (job, mut tokens) = rt.submit_chat(model, prompt).await.unwrap();
    let mut out = String::new();
    while let Some(t) = tokens.recv().await {
        out.push_str(&t);
    }
    rt.finished(job).await;
    out
}

#[tokio::test]
async fn load_failure_unloads_and_next_submit_retries() {
    FakeEngine::fail_next_load("crashy-load");
    let rt = Runtime::new(crashy_cfg("crashy-load")).await.unwrap();
    let err = rt.submit_chat("crashy-load", "x").await.unwrap_err();
    assert!(
        matches!(err, RuntimeError::Engine(_)),
        "expected engine error, got {err:?}"
    );
    assert_eq!(model_status(&rt, "crashy-load"), (Tier::Cupboard, 0));
    let out = tokio::time::timeout(
        Duration::from_secs(2),
        collect_chat(&rt, "crashy-load", "hi"),
    )
    .await
    .expect("retry submit after load fault should finish");
    assert_eq!(out, "hi world");
}

#[tokio::test]
async fn run_failure_unloads_and_next_submit_retries() {
    FakeEngine::fail_next_run("crashy-run");
    let rt = Runtime::new(crashy_cfg("crashy-run")).await.unwrap();
    let err = rt.submit_chat("crashy-run", "x").await.unwrap_err();
    assert!(
        matches!(err, RuntimeError::Engine(_)),
        "expected engine error, got {err:?}"
    );
    assert_eq!(model_status(&rt, "crashy-run"), (Tier::Cupboard, 0));
    let out = tokio::time::timeout(
        Duration::from_secs(2),
        collect_chat(&rt, "crashy-run", "hi"),
    )
    .await
    .expect("retry submit after run fault should finish");
    assert_eq!(out, "hi world");
}

#[tokio::test]
async fn unknown_engine_submit_unavailable() {
    let rt = Runtime::new(nope_soap_cfg()).await.unwrap();
    let err = rt.submit_chat("soap", "x").await.unwrap_err();
    assert!(
        matches!(err, RuntimeError::Unavailable),
        "expected Unavailable, got {err:?}"
    );
}

#[tokio::test]
async fn stream_end_then_live_submit_does_not_panic() {
    let rt = Runtime::new(clinic_cfg()).await.unwrap();
    let (soap_job, mut tokens) = rt.submit_chat("soap", "note").await.unwrap();
    while tokens.recv().await.is_some() {}
    // do NOT call finished(soap) yet
    let (w_job, mut wtok) = rt.submit_chat("whisper", "hi").await.unwrap();
    while wtok.recv().await.is_some() {}
    rt.finished(soap_job).await; // must not panic; should be idempotent
    rt.finished(w_job).await; // must not panic
}

#[tokio::test]
async fn preempted_soap_does_not_replay_streamed_tokens() {
    let rt = Runtime::new(clinic_cfg()).await.unwrap();
    let (soap_job, mut soap_rx) = rt.submit_chat("soap", "note").await.unwrap();
    let first = soap_rx.recv().await.expect("first soap token");
    assert_eq!(first, "note");
    let (w_job, mut w_rx) = rt.submit_chat("whisper", "hi").await.unwrap();
    while w_rx.recv().await.is_some() {}
    rt.finished(w_job).await;
    let mut got = vec![first];
    while let Some(t) = soap_rx.recv().await {
        got.push(t);
    }
    rt.finished(soap_job).await;
    assert_eq!(got, vec!["note".to_string(), " world".to_string()]);
}

fn ws_whisper_cfg() -> AppConfig {
    let mut cfg = clinic_cfg();
    for model in &mut cfg.enabled {
        if model.spec.name == "whisper" {
            model.transport = "websocket".into();
        }
    }
    cfg
}

#[tokio::test]
async fn http_transport_rejects_websocket_session() {
    let rt = Runtime::new(clinic_cfg()).await.unwrap();
    let err = rt.begin_session("soap").await.unwrap_err();
    assert!(matches!(err, RuntimeError::NoWebsocket));
}

#[tokio::test]
async fn websocket_session_holds_slot_until_end() {
    let rt = Runtime::new(ws_whisper_cfg()).await.unwrap();
    let job = rt.begin_session("whisper").await.unwrap();
    let whisper = rt
        .status()
        .models
        .into_iter()
        .find(|m| m.name == "whisper")
        .unwrap();
    assert_eq!(whisper.running, 1);
    let mut rx = rt.session_prompt(job, "whisper", "hello").await.unwrap();
    let mut out = String::new();
    while let Some(t) = rx.recv().await {
        out.push_str(&t);
    }
    assert_eq!(out, "hello world");
    let still = rt
        .status()
        .models
        .into_iter()
        .find(|m| m.name == "whisper")
        .unwrap();
    assert_eq!(still.running, 1);
    rt.end_session(job).await;
    let after = rt
        .status()
        .models
        .into_iter()
        .find(|m| m.name == "whisper")
        .unwrap();
    assert_eq!(after.running, 0);
}

async fn spawn_vllm_mock() -> String {
    let app = Router::new()
        .route("/sleep", post(vllm_ok))
        .route("/wake_up", post(vllm_ok))
        .route("/v1/chat/completions", post(vllm_chat_sse));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://{addr}")
}

async fn vllm_ok() -> StatusCode {
    StatusCode::OK
}

async fn vllm_chat_sse() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "text/event-stream")],
        concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n\n",
            "data: [DONE]\n\n",
        ),
    )
}

async fn spawn_ollama_mock() -> String {
    let app = Router::new()
        .route("/api/generate", post(ollama_generate))
        .route("/api/chat", post(ollama_chat));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://{addr}")
}

async fn ollama_generate() -> impl IntoResponse {
    axum::Json(serde_json::json!({"done": true, "response": ""}))
}

async fn ollama_chat() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "application/x-ndjson")],
        concat!(
            "{\"message\":{\"content\":\"hello\"},\"done\":false}\n",
            "{\"message\":{\"content\":\" world\"},\"done\":false}\n",
            "{\"message\":{\"content\":\"\"},\"done\":true}\n",
        ),
    )
}
