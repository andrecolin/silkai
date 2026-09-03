use futures_util::{SinkExt, StreamExt};
use silkai_server::app::{test_app, test_app_ws};
use tokio::net::TcpListener;
use tokio_tungstenite::tungstenite::Message;

async fn serve_ws() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = test_app_ws().await;
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("ws://{addr}")
}

#[tokio::test]
async fn http_only_model_cannot_open_session() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = test_app().await;
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let url = format!("ws://{addr}/v1/session?model=soap");
    let err = tokio_tungstenite::connect_async(&url).await.unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("404") || msg.contains("Not Found") || msg.contains("status code"),
        "{msg}"
    );
}

#[tokio::test]
async fn any_websocket_model_can_prompt() {
    let base = serve_ws().await;
    let url = format!("{base}/v1/session?model=whisper");
    let (mut ws, _) = tokio_tungstenite::connect_async(&url).await.unwrap();
    let queued = ws.next().await.unwrap().unwrap();
    assert!(queued.to_string().contains("queued"));
    let live = ws.next().await.unwrap().unwrap();
    assert!(live.to_string().contains("live"));
    ws.send(Message::Text(
        r#"{"type":"prompt","content":"hello"}"#.into(),
    ))
    .await
    .unwrap();
    let mut body = String::new();
    loop {
        let msg = ws.next().await.unwrap().unwrap();
        let text = msg.to_string();
        if text.contains("done") {
            break;
        }
        if text.contains("token") {
            body.push_str(&text);
        }
    }
    assert!(body.contains("hello") || body.contains("world"));
    ws.close(None).await.unwrap();
}

/// A session that pins a model while the app talks to the engine directly
/// sends no prompts. Pings must hold it open, not close it: Python's
/// `websockets` sends one every 20 s by default.
#[tokio::test]
async fn keepalives_hold_the_session_open() {
    let base = serve_ws().await;
    let url = format!("{base}/v1/session?model=whisper");
    let (mut ws, _) = tokio_tungstenite::connect_async(&url).await.unwrap();
    let queued = ws.next().await.unwrap().unwrap();
    assert!(queued.to_string().contains("queued"));
    let live = ws.next().await.unwrap().unwrap();
    assert!(live.to_string().contains("live"));

    ws.send(Message::Ping(Vec::new().into())).await.unwrap();
    ws.send(Message::Text(r#"{"type":"ping"}"#.into()))
        .await
        .unwrap();

    // The session is still usable: a prompt after the keepalives still runs.
    ws.send(Message::Text(
        r#"{"type":"prompt","content":"hello"}"#.into(),
    ))
    .await
    .unwrap();
    let mut saw_done = false;
    for _ in 0..40 {
        let msg = ws.next().await.unwrap().unwrap();
        if matches!(msg, Message::Pong(_) | Message::Ping(_)) {
            continue;
        }
        let text = msg.to_string();
        assert!(
            !text.contains("idle_close"),
            "keepalive closed the session: {text}"
        );
        if text.contains("done") {
            saw_done = true;
            break;
        }
    }
    assert!(saw_done, "prompt after keepalives never completed");
    ws.close(None).await.unwrap();
}
