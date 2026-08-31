use axum::body::Body;
use axum::http::{Request, StatusCode};
use silkai_server::app::test_app;
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

#[tokio::test]
async fn unknown_model_404() {
    let app = test_app().await;
    let res = app.oneshot(chat("nope", false)).await.unwrap();
    assert_eq!(res.status(), StatusCode::NOT_FOUND);
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
