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
    one_fake_cfg("crashy-http")
}

/// An app with a single fake model under a name no other test uses. The
/// fake engine's fail/reject hooks are global and keyed by model name, and
/// tests in one binary run in parallel, so a shared name lets one test
/// consume another's hook.
fn one_fake_cfg(name: &str) -> AppConfig {
    AppConfig {
        listen: "127.0.0.1:0".into(),
        prefetch_on_start: false,
        request_timeout_secs: 600,
        request_timeout: std::time::Duration::from_secs(600),
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

fn chat_body(messages: serde_json::Value) -> Request<Body> {
    let body = serde_json::json!({"model": "soap", "messages": messages});
    Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

async fn answer(res: axum::response::Response) -> String {
    let bytes = axum::body::to_bytes(res.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    json["choices"][0]["message"]["content"]
        .as_str()
        .unwrap()
        .to_string()
}

/// The fake engine echoes the last message; a system prompt ahead of it
/// must not displace the user turn.
#[tokio::test]
async fn system_prompt_keeps_user_turn_last() {
    let app = test_app().await;
    let res = app
        .oneshot(chat_body(serde_json::json!([
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hello"}
        ])))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    assert_eq!(answer(res).await, "hello world");
}

/// Newer clients send `content` as a list of parts; the text parts are joined.
#[tokio::test]
async fn content_parts_are_joined() {
    let app = test_app().await;
    let res = app
        .oneshot(chat_body(serde_json::json!([
            {"role": "user", "content": [
                {"type": "text", "text": "hel"},
                {"type": "image_url", "image_url": {"url": "data:,"}},
                {"type": "text", "text": "lo"}
            ]}
        ])))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    assert_eq!(answer(res).await, "hello world");
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

/// A request the engine cannot take (a prompt beyond its window) is a 400
/// with the reason, and the model is still resident for the next one.
#[tokio::test]
async fn rejected_prompt_is_400_and_model_stays_up() {
    let app = silkai_server::app::app_from_config(one_fake_cfg("reject-json")).await;
    let warm = app
        .clone()
        .oneshot(chat("reject-json", false))
        .await
        .unwrap();
    assert_eq!(warm.status(), StatusCode::OK);
    let _ = axum::body::to_bytes(warm.into_body(), usize::MAX).await;

    FakeEngine::reject_next_run("reject-json");
    let res = app
        .clone()
        .oneshot(chat("reject-json", false))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);
    let body = axum::body::to_bytes(res.into_body(), usize::MAX)
        .await
        .unwrap();
    assert!(std::str::from_utf8(&body).unwrap().contains("too long"));

    let status = app
        .clone()
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
    let model = v["models"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["name"] == "reject-json")
        .unwrap();
    assert_eq!(model["state"], "bench", "{model}");
    assert_eq!(model["running"], 0);

    let again = app.oneshot(chat("reject-json", false)).await.unwrap();
    assert_eq!(again.status(), StatusCode::OK);
}

#[tokio::test]
async fn rejected_prompt_streaming_is_400_too() {
    let app = silkai_server::app::app_from_config(one_fake_cfg("reject-stream")).await;
    FakeEngine::reject_next_run("reject-stream");
    let res = app.oneshot(chat("reject-stream", true)).await.unwrap();
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);
}

/// The fields the OpenAI SDKs and their imitators read.
#[tokio::test]
async fn completion_carries_openai_fields() {
    let app = test_app().await;
    let res = app.oneshot(chat("soap", false)).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(res.into_body(), usize::MAX)
        .await
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert!(v["id"].as_str().unwrap().starts_with("chatcmpl-"));
    assert_eq!(v["object"], "chat.completion");
    assert!(v["created"].as_u64().unwrap() > 1_700_000_000);
    assert_eq!(v["model"], "soap");
    assert_eq!(v["choices"][0]["index"], 0);
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
    assert_eq!(v["choices"][0]["message"]["role"], "assistant");
    assert_eq!(v["choices"][0]["message"]["content"], "hello world");
}

#[tokio::test]
async fn stream_has_role_chunk_then_stop_then_done() {
    let app = test_app().await;
    let res = app.oneshot(chat("soap", true)).await.unwrap();
    let bytes = axum::body::to_bytes(res.into_body(), usize::MAX)
        .await
        .unwrap();
    let text = String::from_utf8(bytes.to_vec()).unwrap();
    let chunks: Vec<serde_json::Value> = text
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter(|d| *d != "[DONE]")
        .map(|d| serde_json::from_str(d).unwrap())
        .collect();
    assert!(chunks.len() >= 3, "{text}");
    assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
    assert_eq!(chunks[0]["model"], "soap");
    assert_eq!(chunks[0]["object"], "chat.completion.chunk");
    let last = chunks.last().unwrap();
    assert_eq!(last["choices"][0]["finish_reason"], "stop");
    assert!(last["choices"][0]["delta"].as_object().unwrap().is_empty());
    let content: String = chunks[1..chunks.len() - 1]
        .iter()
        .map(|c| c["choices"][0]["delta"]["content"].as_str().unwrap_or(""))
        .collect();
    assert_eq!(content, "hello world");
    assert!(text.trim_end().ends_with("data: [DONE]"));
    assert!(chunks.iter().all(|c| c["id"] == chunks[0]["id"]));
}
