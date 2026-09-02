use axum::body::Body;
use axum::http::{Request, StatusCode};
use silkai_adapters::FakeEngine;
use silkai_sched::{ModelSpec, Priority, Resources};
use silkai_server::app::test_app;
use silkai_server::config::{AppConfig, ConfiguredModel};
use tower::ServiceExt;

fn chat(model: &str, stream: bool) -> Request<Body> {
    let body = serde_json::json!({
        "model": model,
        "stream": stream,
        "messages": [{"role": "user", "content": "hello"}]
    });
    Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

fn crashy_http_cfg() -> AppConfig {
    AppConfig {
        listen: "127.0.0.1:0".into(),
        prefetch_on_start: false,
        request_timeout_secs: 600,
        request_timeout: std::time::Duration::from_secs(600),
        resources: Resources::single(29.0, 96.0),
        enabled: vec![ConfiguredModel {
            engine: "fake".into(),
            path: "/models/crashy-http.bin".into(),
            url: None,
            cmd: Vec::new(),
            transport: "http".into(),
            idle_timeout_secs: None,
            spec: ModelSpec {
                name: "crashy-http".into(),
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
    }
}

#[tokio::test]
async fn engine_fault_returns_500_health_stays_ok() {
    FakeEngine::fail_next_load("crashy-http");
    let app = silkai_server::app::app_from_config(crashy_http_cfg()).await;
    let res = app
        .clone()
        .oneshot(chat("crashy-http", false))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::INTERNAL_SERVER_ERROR);
    let health = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(health.status(), StatusCode::OK);
    let res = app.oneshot(chat("crashy-http", false)).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
}

#[tokio::test]
async fn unknown_model_404() {
    let app = test_app().await;
    let res = app.oneshot(chat("nope", false)).await.unwrap();
    assert_eq!(res.status(), StatusCode::NOT_FOUND);
}

#[cfg(not(feature = "llama"))]
#[tokio::test]
async fn llama_cpp_without_feature_returns_503() {
    let app = silkai_server::app::test_app_llama_soap().await;
    let res = app.oneshot(chat("soap", false)).await.unwrap();
    assert_eq!(res.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn disabled_model_400() {
    let app = silkai_server::app::test_app_with_disabled().await;
    let res = app.oneshot(chat("too-big", false)).await.unwrap();
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn streaming_soap_returns_sse_data_lines() {
    let app = test_app().await;
    let res = app.oneshot(chat("soap", true)).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(res.into_body(), usize::MAX)
        .await
        .unwrap();
    let text = String::from_utf8(bytes.to_vec()).unwrap();
    assert!(text.contains("data:"));
    assert!(text.contains("[DONE]"));
}

#[tokio::test]
async fn queued_stream_sends_comment_before_tokens() {
    let app = test_app().await;
    let whisper_app = app.clone();
    let whisper =
        tokio::spawn(async move { whisper_app.oneshot(chat("whisper", true)).await.unwrap() });
    tokio::time::sleep(std::time::Duration::from_millis(30)).await;
    let res = app.oneshot(chat("soap", true)).await.unwrap();
    let bytes = axum::body::to_bytes(res.into_body(), usize::MAX)
        .await
        .unwrap();
    let text = String::from_utf8(bytes.to_vec()).unwrap();
    assert!(
        text.contains(": queued") || text.contains("queued"),
        "expected a queued SSE comment, got {text:?}"
    );
    let _ = whisper.await;
}

#[tokio::test]
async fn timeout_while_queued_returns_504() {
    let app = silkai_server::app::test_app_timeout_ms(1).await;
    let whisper_app = app.clone();
    let _whisper = tokio::spawn(async move {
        let _ = whisper_app.oneshot(chat("whisper", true)).await;
    });
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    let res = app.oneshot(chat("soap", true)).await.unwrap();
    assert_eq!(res.status(), StatusCode::GATEWAY_TIMEOUT);
}

#[tokio::test]
async fn timeout_does_not_leave_soap_queued() {
    let app = silkai_server::app::test_app_timeout_ms(1).await;
    let whisper_app = app.clone();
    let whisper = tokio::spawn(async move {
        let _ = whisper_app.oneshot(chat("whisper", true)).await;
    });
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    let res = app.clone().oneshot(chat("soap", true)).await.unwrap();
    assert_eq!(res.status(), StatusCode::GATEWAY_TIMEOUT);
    let status = app
        .oneshot(
            Request::builder()
                .uri("/v1/status")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let bytes = axum::body::to_bytes(status.into_body(), usize::MAX)
        .await
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let soap = v["models"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["name"] == "soap")
        .unwrap();
    assert_eq!(soap["queued"], 0);
    assert_eq!(soap["running"], 0);
    let _ = whisper.await;
}
