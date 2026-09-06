use std::sync::{Arc, Mutex};

use silkai_adapters::{ChatMessage, Chunk, Engine, EngineError, RunOptions, Usage, VllmEngine};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;

#[tokio::test]
async fn vllm_load_posts_wake_up() {
    let (url, log) = spawn_mock().await;
    let e = VllmEngine::new("write", 28.0, &url);
    e.load("Qwen/Qwen3-0.6B", 1).await.unwrap();
    assert_eq!(e.gpu(), Some(1));
    assert!(logged(&log, "POST /wake_up"));
}

#[tokio::test]
async fn vllm_sleep_posts_level_1() {
    let (url, log) = spawn_mock().await;
    let e = VllmEngine::new("write", 28.0, &url);
    e.load("Qwen/Qwen3-0.6B", 0).await.unwrap();
    e.sleep().await.unwrap();
    assert!(logged(&log, "POST /sleep?level=1"));
}

#[tokio::test]
async fn vllm_wake_posts_wake_up() {
    let (url, log) = spawn_mock().await;
    let e = VllmEngine::new("write", 28.0, &url);
    e.load("Qwen/Qwen3-0.6B", 0).await.unwrap();
    e.sleep().await.unwrap();
    e.wake(0).await.unwrap();
    let hits = count_logged(&log, "POST /wake_up");
    assert_eq!(hits, 2);
}

#[tokio::test]
async fn vllm_run_forwards_reasoning_apart_from_content() {
    let (url, _log) = spawn_mock_serving(REASONING_SSE).await;
    let e = VllmEngine::new("write", 28.0, &url);
    e.load("Qwen/Qwen3-0.6B", 0).await.unwrap();
    let mut rx = e
        .run(
            &[ChatMessage::user("hello")],
            "",
            &RunOptions::default(),
            CancellationToken::new(),
        )
        .await
        .unwrap();
    let mut reasoning = String::new();
    let mut content = String::new();
    while let Some(c) = rx.recv().await {
        match c {
            Chunk::Reasoning(t) => reasoning.push_str(&t),
            Chunk::Token(t) => content.push_str(&t),
            Chunk::End(_) | Chunk::Reject(_) => {}
        }
    }
    // The trace is delivered, and it never contaminates the answer.
    assert_eq!(reasoning, "thinking");
    assert_eq!(content, "hello");
}

#[tokio::test]
async fn vllm_run_streams_sse_content() {
    let (url, log) = spawn_mock().await;
    let e = VllmEngine::new("write", 28.0, &url);
    e.load("Qwen/Qwen3-0.6B", 0).await.unwrap();
    let mut rx = e
        .run(
            &[ChatMessage::user("hello")],
            "",
            &RunOptions::default(),
            CancellationToken::new(),
        )
        .await
        .unwrap();
    let mut got = Vec::new();
    while let Some(c) = rx.recv().await {
        if let Some(t) = c.text() {
            got.push(t.to_string());
        }
    }
    assert_eq!(got, vec!["hello".to_string(), " world".to_string()]);
    assert!(logged(&log, "POST /v1/chat/completions"));
    assert!(logged(&log, "Qwen/Qwen3-0.6B"));
}

#[tokio::test]
async fn vllm_forwards_every_message_and_prefix() {
    let (url, log) = spawn_mock().await;
    let e = VllmEngine::new("write", 28.0, &url);
    e.load("Qwen/Qwen3-0.6B", 0).await.unwrap();
    let chat = [
        ChatMessage::system("You are terse."),
        ChatMessage::user("hello"),
    ];
    let opts = RunOptions {
        max_tokens: Some(64),
        temperature: Some(0.2),
    };
    let mut rx = e
        .run(&chat, "hel", &opts, CancellationToken::new())
        .await
        .unwrap();
    while rx.recv().await.is_some() {}
    let body = log
        .lock()
        .unwrap()
        .iter()
        .find(|l| l.contains("/v1/chat/completions"))
        .cloned()
        .expect("chat request logged");
    let json: serde_json::Value = serde_json::from_str(
        body.split_once(' ')
            .and_then(|(_, r)| r.split_once(' '))
            .map(|(_, b)| b)
            .unwrap(),
    )
    .unwrap();
    let messages = json["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 3);
    assert_eq!(messages[0]["role"], "system");
    assert_eq!(messages[0]["content"], "You are terse.");
    assert_eq!(messages[1]["role"], "user");
    assert_eq!(messages[2]["role"], "assistant");
    assert_eq!(messages[2]["content"], "hel");
    assert_eq!(json["max_tokens"], 64);
    assert!((json["temperature"].as_f64().unwrap() - 0.2).abs() < 1e-6);
}

/// A content list reaches the engine as a list. SilkAI used to keep the text
/// parts and drop the rest, which handed a vision model a prompt with no
/// image and got back a confident guess instead of an answer.
#[tokio::test]
async fn vllm_forwards_image_parts() {
    let (url, log) = spawn_mock().await;
    let e = VllmEngine::new("write", 28.0, &url);
    e.load("Qwen/Qwen3-VL", 0).await.unwrap();
    let parts = vec![
        serde_json::json!({"type": "text", "text": "what colour?"}),
        serde_json::json!({"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}),
    ];
    let mut rx = e
        .run(
            &[ChatMessage::user(parts)],
            "",
            &RunOptions::default(),
            CancellationToken::new(),
        )
        .await
        .unwrap();
    while rx.recv().await.is_some() {}
    assert!(logged(&log, "what colour?"));
    assert!(logged(&log, "image_url"));
    assert!(logged(&log, "data:image/png;base64,AAAA"));
}

/// The reason the run stopped and the tokens it cost come back with it.
/// SilkAI used to discard both and report every reply as a clean "stop"
/// with no counts, so a truncated answer was indistinguishable from a
/// complete one.
#[tokio::test]
async fn vllm_reports_finish_reason_and_usage() {
    let (url, log) = spawn_mock().await;
    let e = VllmEngine::new("write", 28.0, &url);
    e.load("Qwen/Qwen3-0.6B", 0).await.unwrap();
    let mut rx = e
        .run(
            &[ChatMessage::user("hello")],
            "",
            &RunOptions::default(),
            CancellationToken::new(),
        )
        .await
        .unwrap();
    let mut text = String::new();
    let mut end = None;
    while let Some(chunk) = rx.recv().await {
        match chunk {
            Chunk::Token(t) => text.push_str(&t),
            Chunk::Reasoning(_) => {}
            Chunk::End(e) => end = Some(e),
            Chunk::Reject(_) => {}
        }
    }
    assert_eq!(text, "hello world");
    let end = end.expect("an end chunk");
    assert_eq!(end.finish_reason.as_deref(), Some("length"));
    assert_eq!(
        end.usage,
        Some(Usage {
            prompt_tokens: 11,
            completion_tokens: 2,
            total_tokens: 13,
        })
    );
    // A server only sends usage when it is asked to.
    assert!(logged(&log, "stream_options"));
}

#[tokio::test]
async fn vllm_run_without_load_is_not_loaded() {
    let (url, _) = spawn_mock().await;
    let e = VllmEngine::new("write", 28.0, &url);
    let err = e
        .run(
            &[ChatMessage::user("hello")],
            "",
            &RunOptions::default(),
            CancellationToken::new(),
        )
        .await
        .unwrap_err();
    assert!(matches!(err, EngineError::NotLoaded));
}

#[tokio::test]
async fn vllm_run_after_sleep_is_not_loaded() {
    let (url, _) = spawn_mock().await;
    let e = VllmEngine::new("write", 28.0, &url);
    e.load("Qwen/Qwen3-0.6B", 0).await.unwrap();
    e.sleep().await.unwrap();
    let err = e
        .run(
            &[ChatMessage::user("hello")],
            "",
            &RunOptions::default(),
            CancellationToken::new(),
        )
        .await
        .unwrap_err();
    assert!(matches!(err, EngineError::NotLoaded));
}

#[tokio::test]
async fn vllm_run_stops_on_cancel() {
    let (url, _) = spawn_mock().await;
    let e = VllmEngine::new("write", 28.0, &url);
    e.load("Qwen/Qwen3-0.6B", 0).await.unwrap();
    let cancel = CancellationToken::new();
    cancel.cancel();
    let mut rx = e
        .run(
            &[ChatMessage::user("hello")],
            "",
            &RunOptions::default(),
            cancel,
        )
        .await
        .unwrap();
    assert!(rx.recv().await.is_none());
}

#[tokio::test]
async fn vllm_load_errors_when_server_down() {
    let e = VllmEngine::new("write", 28.0, "http://127.0.0.1:1");
    let err = e.load("Qwen/Qwen3-0.6B", 0).await.unwrap_err();
    match err {
        EngineError::Other(msg) => assert!(!msg.is_empty()),
        other => panic!("expected Other, got {other:?}"),
    }
}

#[tokio::test]
async fn vllm_warm_does_not_hit_http() {
    let (url, log) = spawn_mock().await;
    let e = VllmEngine::new("write", 28.0, &url);
    e.warm("Qwen/Qwen3-0.6B").await.unwrap();
    assert!(log.lock().expect("log").is_empty());
    assert_eq!(e.measured_vram_gb(), 28.0);
}

async fn spawn_mock() -> (String, Arc<Mutex<Vec<String>>>) {
    spawn_mock_serving(CHAT_SSE).await
}

/// A mock that answers `/v1/chat/completions` with `chat`, so a test can pick
/// the stream shape its engine is supposed to read.
async fn spawn_mock_serving(chat: &'static str) -> (String, Arc<Mutex<Vec<String>>>) {
    let log = Arc::new(Mutex::new(Vec::new()));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let state = log.clone();
    tokio::spawn(async move {
        loop {
            let Ok((mut sock, _)) = listener.accept().await else {
                break;
            };
            let state = state.clone();
            tokio::spawn(async move {
                handle_conn(&mut sock, &state, chat).await;
            });
        }
    });
    (format!("http://{addr}"), log)
}

async fn handle_conn(
    sock: &mut tokio::net::TcpStream,
    log: &Mutex<Vec<String>>,
    chat: &'static str,
) {
    let Some((method, path, body)) = read_request(sock).await else {
        return;
    };
    {
        let mut line = format!("{method} {path}");
        if !body.is_empty() {
            line.push(' ');
            line.push_str(&body);
        }
        log.lock().expect("log").push(line);
    }
    let resp = if path.starts_with("/v1/chat/completions") {
        chat
    } else {
        OK_EMPTY
    };
    let _ = sock.write_all(resp.as_bytes()).await;
}

async fn read_request(sock: &mut tokio::net::TcpStream) -> Option<(String, String, String)> {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 1024];
    loop {
        let n = sock.read(&mut tmp).await.ok()?;
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&tmp[..n]);
        if let Some(idx) = find_headers_end(&buf) {
            let headers = std::str::from_utf8(&buf[..idx]).ok()?;
            let mut lines = headers.split("\r\n");
            let req = lines.next()?;
            let mut parts = req.split_whitespace();
            let method = parts.next()?.to_string();
            let path = parts.next()?.to_string();
            let mut content_len = 0usize;
            for line in lines {
                let lower = line.to_ascii_lowercase();
                if let Some(v) = lower.strip_prefix("content-length:") {
                    content_len = v.trim().parse().unwrap_or(0);
                }
            }
            let mut body = buf[idx + 4..].to_vec();
            while body.len() < content_len {
                let n = sock.read(&mut tmp).await.ok()?;
                if n == 0 {
                    break;
                }
                body.extend_from_slice(&tmp[..n]);
            }
            body.truncate(content_len);
            let body = String::from_utf8_lossy(&body).into_owned();
            return Some((method, path, body));
        }
        if buf.len() > 64 * 1024 {
            return None;
        }
    }
    None
}

fn find_headers_end(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n")
}

fn logged(log: &Mutex<Vec<String>>, needle: &str) -> bool {
    log.lock().expect("log").iter().any(|l| l.contains(needle))
}

fn count_logged(log: &Mutex<Vec<String>>, needle: &str) -> usize {
    log.lock()
        .expect("log")
        .iter()
        .filter(|l| l.contains(needle))
        .count()
}

const OK_EMPTY: &str = "HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";

/// What llama.cpp sends under `--reasoning-format deepseek`: the trace in
/// `reasoning_content` deltas, then the answer in `content` deltas.
const REASONING_SSE: &str = concat!(
    "HTTP/1.1 200 OK\r\n",
    "Content-Type: text/event-stream\r\n",
    "Connection: close\r\n",
    "\r\n",
    "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"think\"}}]}\n\n",
    "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"ing\"}}]}\n\n",
    "data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n",
    "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
    "data: [DONE]\n\n",
);

const CHAT_SSE: &str = concat!(
    "HTTP/1.1 200 OK\r\n",
    "Content-Type: text/event-stream\r\n",
    "Connection: close\r\n",
    "\r\n",
    "data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n",
    "data: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n\n",
    "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"length\"}]}\n\n",
    "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":11,\"completion_tokens\":2,\"total_tokens\":13}}\n\n",
    "data: [DONE]\n\n",
);
