use std::sync::{Arc, Mutex};
use std::time::Duration;

use silkai_adapters::{Engine, ProcessEngine};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;

fn sleep_cmd() -> Vec<String> {
    vec!["sleep".into(), "30".into()]
}

#[tokio::test]
async fn process_load_spawns_child_sleep_kills_it() {
    let (url, _) = spawn_mock(false).await;
    let e = ProcessEngine::new("write", 28.0, &url, sleep_cmd());
    e.load("Qwen/Qwen3-0.6B", 0).await.unwrap();
    assert!(e.alive());
    e.sleep().await.unwrap();
    assert!(!e.alive());
}

#[tokio::test]
async fn process_run_streams_after_spawn() {
    let (url, _) = spawn_mock(false).await;
    let e = ProcessEngine::new("write", 28.0, &url, sleep_cmd());
    e.load("Qwen/Qwen3-0.6B", 0).await.unwrap();
    let mut rx = e.run("hello", "", CancellationToken::new()).await.unwrap();
    let mut got = Vec::new();
    while let Some(t) = rx.recv().await {
        got.push(t);
    }
    e.sleep().await.unwrap();
    assert_eq!(got, vec!["hello".to_string(), " world".to_string()]);
}

#[tokio::test]
async fn process_bad_cmd_fails_and_is_not_alive() {
    let (url, _) = spawn_mock(false).await;
    let e = ProcessEngine::new("write", 28.0, &url, vec!["silkai-no-such-bin-xyz".into()]);
    assert!(e.load("Qwen/Qwen3-0.6B", 0).await.is_err());
    assert!(!e.alive());
}

#[tokio::test]
async fn process_empty_cmd_fails_and_is_not_alive() {
    let (url, _) = spawn_mock(false).await;
    let e = ProcessEngine::new("write", 28.0, &url, Vec::new());
    assert!(e.load("Qwen/Qwen3-0.6B", 0).await.is_err());
    assert!(!e.alive());
}

#[tokio::test]
async fn process_wake_up_error_kills_child() {
    let (url, _) = spawn_mock(true).await;
    let e = ProcessEngine::new("write", 28.0, &url, sleep_cmd());
    assert!(e.load("Qwen/Qwen3-0.6B", 0).await.is_err());
    assert!(!e.alive());
}

#[tokio::test]
async fn process_wake_after_sleep_respawns() {
    let (url, _) = spawn_mock(false).await;
    let e = ProcessEngine::new("write", 28.0, &url, sleep_cmd());
    e.load("Qwen/Qwen3-0.6B", 0).await.unwrap();
    e.sleep().await.unwrap();
    assert!(!e.alive());
    e.wake(0).await.unwrap();
    assert!(e.alive());
    e.discard().await.unwrap();
    assert!(!e.alive());
}

#[tokio::test]
async fn process_load_waits_until_http_ready() {
    let bound = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = bound.local_addr().unwrap();
    drop(bound);
    let url = format!("http://{addr}");
    let e = Arc::new(ProcessEngine::new("write", 28.0, &url, sleep_cmd()));
    let loading = e.clone();
    let handle = tokio::spawn(async move { loading.load("Qwen/Qwen3-0.6B", 0).await });
    tokio::time::sleep(Duration::from_millis(150)).await;
    let listener = TcpListener::bind(addr).await.unwrap();
    serve_mock(listener, false, Arc::new(Mutex::new(Vec::new())));
    handle.await.unwrap().unwrap();
    assert!(e.alive());
    e.sleep().await.unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn process_sleep_kills_process_group() {
    let (url, _) = spawn_mock(false).await;
    let e = ProcessEngine::new(
        "write",
        28.0,
        &url,
        vec!["sh".into(), "-c".into(), "sleep 300 & wait".into()],
    );
    e.load("Qwen/Qwen3-0.6B", 0).await.unwrap();
    let pid = e.child_id().expect("child pid");
    // `sh` forks `sleep` after load() returns, so wait for the group to fill.
    let members = await_group(pid, 2).await;
    assert!(
        members.len() >= 2,
        "expected sh and sleep in group, got {members:?}"
    );
    e.sleep().await.unwrap();
    assert!(!e.alive());
    for member in members {
        assert!(
            !pid_exists(member),
            "process {member} still running after sleep"
        );
    }
}

async fn spawn_mock(fail_wake: bool) -> (String, Arc<Mutex<Vec<String>>>) {
    let log = Arc::new(Mutex::new(Vec::new()));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    serve_mock(listener, fail_wake, log.clone());
    (format!("http://{addr}"), log)
}

fn serve_mock(listener: TcpListener, fail_wake: bool, log: Arc<Mutex<Vec<String>>>) {
    tokio::spawn(async move {
        loop {
            let Ok((mut sock, _)) = listener.accept().await else {
                break;
            };
            let state = log.clone();
            tokio::spawn(async move {
                handle_conn(&mut sock, &state, fail_wake).await;
            });
        }
    });
}

#[cfg(unix)]
async fn await_group(pgid: u32, want: usize) -> Vec<u32> {
    let mut members = pids_in_group(pgid);
    for _ in 0..100 {
        if members.len() >= want {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
        members = pids_in_group(pgid);
    }
    members
}

#[cfg(unix)]
fn pids_in_group(pgid: u32) -> Vec<u32> {
    let out = std::process::Command::new("ps")
        .args(["-axo", "pid=,pgid="])
        .output()
        .expect("ps");
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|line| {
            let mut bits = line.split_whitespace();
            let pid: u32 = bits.next()?.parse().ok()?;
            let group: u32 = bits.next()?.parse().ok()?;
            (group == pgid).then_some(pid)
        })
        .collect()
}

#[cfg(unix)]
fn pid_exists(pid: u32) -> bool {
    std::process::Command::new("kill")
        .args(["-0", &pid.to_string()])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

async fn handle_conn(sock: &mut tokio::net::TcpStream, log: &Mutex<Vec<String>>, fail_wake: bool) {
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
    let resp = if fail_wake && path.starts_with("/wake_up") {
        FAIL_WAKE
    } else if path.starts_with("/v1/chat/completions") {
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

const FAIL_WAKE: &str =
    "HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";

const CHAT_SSE: &str = concat!(
    "HTTP/1.1 200 OK\r\n",
    "Content-Type: text/event-stream\r\n",
    "Connection: close\r\n",
    "\r\n",
    "data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n",
    "data: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n\n",
    "data: [DONE]\n\n",
);
