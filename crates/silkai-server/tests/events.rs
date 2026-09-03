use std::time::Duration;

use axum::body::Body;
use axum::http::Request;
use futures_util::StreamExt;
use silkai_adapters::ChatMessage;
use silkai_server::app::test_app;
use silkai_server::runtime::Runtime;
use tower::ServiceExt;

fn cfg() -> silkai_server::config::AppConfig {
    let toml = r#"
[resources]
gpu_total_gb = 32
gpu_headroom_gb = 3
ram_total_gb = 128
ram_headroom_gb = 32

[models.soap]
engine = "fake"
path = "/models/soap.bin"
vram_gb = 28
priority = "normal"
exclusive = true
"#;
    silkai_server::config::load_from_str(toml).unwrap()
}

#[tokio::test]
async fn a_chat_leaves_wake_start_finish_in_order() {
    let rt = Runtime::new(cfg()).await.unwrap();
    let (job, mut rx) = rt
        .submit_chat("soap", vec![ChatMessage::user("note")])
        .await
        .unwrap();
    while rx.recv().await.is_some() {}
    rt.finished(job).await;
    let kinds: Vec<&str> = rt
        .events()
        .since(0)
        .iter()
        .filter(|e| e.model.as_deref() == Some("soap"))
        .map(|e| e.kind)
        .collect();
    assert_eq!(kinds, vec!["warm", "wake", "start", "finish"], "{kinds:?}");
    let wake = rt
        .events()
        .since(0)
        .into_iter()
        .find(|e| e.kind == "wake")
        .unwrap();
    assert!(wake.ms.is_some());
    assert_eq!(wake.gpu, Some(0));
    assert!(wake.t.ends_with('Z'));
    let c = rt.counters();
    assert_eq!(c["soap"].wakes, 1);
    assert_eq!(c["soap"].loads, 0);
}

#[tokio::test]
async fn since_skips_replayed_events() {
    let rt = Runtime::new(cfg()).await.unwrap();
    let seq = rt.events().last_seq();
    assert!(seq >= 1);
    assert!(rt.events().since(seq).is_empty());
    assert_eq!(rt.events().since(seq - 1).len(), 1);
}

/// `/v1/events` replays the ring, then follows. Read frames until the
/// chat's `finish` shows up, with a deadline so a broken stream fails
/// instead of hanging.
#[tokio::test]
async fn sse_replays_then_streams_live_events() {
    let app = test_app().await;
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/events")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), 200);
    assert!(res
        .headers()
        .get("content-type")
        .unwrap()
        .to_str()
        .unwrap()
        .starts_with("text/event-stream"));
    let mut frames = res.into_body().into_data_stream();

    // Prefetch already put warm events in the ring: those come back first.
    let mut seen = String::new();
    let first = tokio::time::timeout(Duration::from_secs(2), frames.next())
        .await
        .expect("replay frame")
        .unwrap()
        .unwrap();
    seen.push_str(std::str::from_utf8(&first).unwrap());
    assert!(seen.contains("\"kind\":\"warm\""), "{seen}");

    // Now a live event.
    let body = serde_json::json!({"model":"soap","messages":[{"role":"user","content":"hi"}]});
    let chat = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(chat.status(), 200);
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    while !seen.contains("\"kind\":\"finish\"") {
        let left = deadline.saturating_duration_since(tokio::time::Instant::now());
        let frame = tokio::time::timeout(left, frames.next())
            .await
            .expect("finish event before deadline")
            .unwrap()
            .unwrap();
        seen.push_str(std::str::from_utf8(&frame).unwrap());
    }
    assert!(seen.contains("id: "));
    assert!(seen.contains("\"kind\":\"start\""));
}
