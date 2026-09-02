use std::sync::{Arc, Mutex};

use silkai_adapters::{Engine, ProcessEngine};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;

#[tokio::test]
async fn process_load_spawns_child_sleep_kills_it() {
    let (url, _) = spawn_mock().await;
    let e = ProcessEngine::new("write", 28.0, &url, vec!["sleep".into(), "30".into()]);
    e.load("Qwen/Qwen3-0.6B", 0).await.unwrap();
    assert!(e.alive());
    e.sleep().await.unwrap();
    assert!(!e.alive());
}

#[tokio::test]
async fn process_run_streams_after_spawn() {
    let (url, _) = spawn_mock().await;
    let e = ProcessEngine::new("write", 28.0, &url, vec!["sleep".into(), "30".into()]);
    e.load("Qwen/Qwen3-0.6B", 0).await.unwrap();
    let mut rx = e.run("hello", "", CancellationToken::new()).await.unwrap();
    let mut got = Vec::new();
    while let Some(t) = rx.recv().await {
        got.push(t);
    }
    e.sleep().await.unwrap();
    assert_eq!(got, vec!["hello".to_string(), " world".to_string()]);
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
