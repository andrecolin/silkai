//! `POST /v1/decision` through the router: the body reaches the engine
//! whole, the reply comes back whole, and the job releases its slot.

use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use silkai_adapters::FakeEngine;
use silkai_sched::{ModelSpec, Priority, Resources};
use silkai_server::app::test_app;
use silkai_server::config::{AppConfig, ConfiguredModel};
use tower::ServiceExt;

fn decision(body: serde_json::Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/decision")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

fn ticket(model: &str) -> serde_json::Value {
    serde_json::json!({
        "model": model,
        "instructions": "Answer each question about this support request.",
        "schema": {
            "category": {"type": "enum", "choices": ["billing", "technical", "other"]},
            "urgent": {"type": "boolean"}
        },
        "contexts": ["I was charged twice and need this fixed today."]
    })
}

async fn json_of(res: axum::response::Response) -> serde_json::Value {
    let bytes = axum::body::to_bytes(res.into_body(), usize::MAX)
        .await
        .unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

async fn text_of(res: axum::response::Response) -> String {
    let bytes = axum::body::to_bytes(res.into_body(), usize::MAX)
        .await
        .unwrap();
    String::from_utf8(bytes.to_vec()).unwrap()
}

/// Jobs running on `model`, as `/v1/status` reports it.
async fn running(app: &Router, model: &str) -> u64 {
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/status")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let status = json_of(res).await;
    status["models"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["name"] == model)
        .and_then(|m| m["running"].as_u64())
        .unwrap()
}

/// The reply is handed to the client before the job is booked as
/// finished, so the count may lag the response by a moment.
async fn wait_idle(app: &Router, model: &str) {
    for _ in 0..100 {
        if running(app, model).await == 0 {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("{model} still running");
}

/// An app with a single fake model under a name no other test uses: the
/// fake's fail/reject hooks are global and keyed by model name.
fn one_fake_cfg(name: &str) -> AppConfig {
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
            ctx_size: None,
            spec: ModelSpec {
                name: name.into(),
                vram_gb: 8.0,
                ram_gb: 8.0,
                priority: Priority::Normal,
                exclusive: true,
                slots: 1,
                keep_warm: true,
                gpu: None,
                gpus: Vec::new(),
            },
        }],
        disabled: vec![],
        ui: Default::default(),
    }
}

#[tokio::test]
async fn decision_answers_whole_and_releases_the_slot() {
    let app = test_app().await;
    let res = app.clone().oneshot(decision(ticket("soap"))).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let body = json_of(res).await;
    assert_eq!(body["object"], "decision");
    let result = &body["results"][0];
    assert_eq!(result["decision"]["category"], "billing");
    assert_eq!(result["decision"]["urgent"], true);
    assert_eq!(result["fields"]["category"]["probability"], 1.0);
    wait_idle(&app, "soap").await;
}

#[tokio::test]
async fn decision_loads_the_model_like_a_chat_would() {
    // The model starts in the cupboard; a decision has to bring it to the
    // card first, exactly as a chat does, and leave it there afterwards.
    let app = test_app().await;
    let res = app
        .clone()
        .oneshot(decision(ticket("whisper")))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    wait_idle(&app, "whisper").await;
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/status")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let status = json_of(res).await;
    let whisper = status["models"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["name"] == "whisper")
        .unwrap();
    assert_eq!(whisper["tier"], "bench", "{whisper}");
}

#[tokio::test]
async fn decision_then_chat_share_the_model() {
    let app = test_app().await;
    let res = app.clone().oneshot(decision(ticket("soap"))).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let chat = serde_json::json!({
        "model": "soap",
        "messages": [{"role": "user", "content": "hello"}]
    });
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(chat.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let body = json_of(res).await;
    assert_eq!(body["choices"][0]["message"]["content"], "hello world");
    wait_idle(&app, "soap").await;
}

#[tokio::test]
async fn decision_unknown_model_404() {
    let app = test_app().await;
    let res = app.oneshot(decision(ticket("nope"))).await.unwrap();
    assert_eq!(res.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn decision_without_model_400() {
    let app = test_app().await;
    let body = serde_json::json!({"schema": {}, "contexts": ["x"]});
    let res = app.clone().oneshot(decision(body)).await.unwrap();
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);
    assert_eq!(text_of(res).await, "model is required");
    // A body that is not an object at all is the same mistake.
    let res = app
        .oneshot(decision(serde_json::json!(["soap"])))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn decision_refused_by_engine_400_with_reason() {
    FakeEngine::reject_next_run("picky-decider");
    let app = silkai_server::app::app_from_config(one_fake_cfg("picky-decider")).await;
    let res = app
        .clone()
        .oneshot(decision(ticket("picky-decider")))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);
    assert_eq!(text_of(res).await, "schema too wide for the fake");
    // A refusal ends the job; the model is still there for the next one.
    wait_idle(&app, "picky-decider").await;
    let res = app
        .clone()
        .oneshot(decision(ticket("picky-decider")))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
}

#[tokio::test]
async fn decision_engine_fault_returns_500_health_stays_ok() {
    FakeEngine::fail_next_load("crashy-decider");
    let app = silkai_server::app::app_from_config(one_fake_cfg("crashy-decider")).await;
    let res = app
        .clone()
        .oneshot(decision(ticket("crashy-decider")))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::INTERNAL_SERVER_ERROR);
    let res = app
        .clone()
        .oneshot(decision(ticket("crashy-decider")))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
}

#[cfg(not(feature = "llama"))]
#[tokio::test]
async fn decision_llama_cpp_without_feature_returns_503() {
    let app = silkai_server::app::test_app_llama_soap().await;
    let res = app.oneshot(decision(ticket("soap"))).await.unwrap();
    assert_eq!(res.status(), StatusCode::SERVICE_UNAVAILABLE);
}
