use std::sync::{Arc, Mutex};

use silkai_adapters::{Engine, EngineError, VllmEngine};
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
async fn vllm_run_streams_sse_content() {
    let (url, log) = spawn_mock().await;
    let e = VllmEngine::new("write", 28.0, &url);
    e.load("Qwen/Qwen3-0.6B", 0).await.unwrap();
    let mut rx = e.run("hello", CancellationToken::new()).await.unwrap();
    let mut got = Vec::new();
    while let Some(t) = rx.recv().await {
        got.push(t);
    }
    assert_eq!(got, vec!["hello".to_string(), " world".to_string()]);
    assert!(logged(&log, "POST /v1/chat/completions"));
    assert!(logged(&log, "Qwen/Qwen3-0.6B"));
}

#[tokio::test]
async fn vllm_run_without_load_is_not_loaded() {
    let (url, _) = spawn_mock().await;
    let e = VllmEngine::new("write", 28.0, &url);
    let err = e.run("hello", CancellationToken::new()).await.unwrap_err();
    assert!(matches!(err, EngineError::NotLoaded));
}

#[tokio::test]
async fn vllm_run_after_sleep_is_not_loaded() {
    let (url, _) = spawn_mock().await;
    let e = VllmEngine::new("write", 28.0, &url);
    e.load("Qwen/Qwen3-0.6B", 0).await.unwrap();
    e.sleep().await.unwrap();
    let err = e.run("hello", CancellationToken::new()).await.unwrap_err();
    assert!(matches!(err, EngineError::NotLoaded));
}

#[tokio::test]
async fn vllm_run_stops_on_cancel() {
    let (url, _) = spawn_mock().await;
    let e = VllmEngine::new("write", 28.0, &url);
    e.load("Qwen/Qwen3-0.6B", 0).await.unwrap();
    let cancel = CancellationToken::new();
    cancel.cancel();
    let mut rx = e.run("hello", cancel).await.unwrap();
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
                handle_conn(&mut sock, &state).await;
            });
        }
    });
    (format!("http://{addr}"), log)
}

async fn handle_conn(sock: &mut tokio::net::TcpStream, log: &Mutex<Vec<String>>) {
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
        CHAT_SSE
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

const CHAT_SSE: &str = concat!(
    "HTTP/1.1 200 OK\r\n",
    "Content-Type: text/event-stream\r\n",
    "Connection: close\r\n",
    "\r\n",
    "data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n",
    "data: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n\n",
    "data: [DONE]\n\n",
);
