//! The sd-server engine against a mock: readiness on `/sdcpp/v1/capabilities`,
//! a prompt submitted as a `vid_gen` job, the job polled to completion, the
//! clip written to the output directory, and the reply linking to it.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use silkai_adapters::{ChatMessage, Chunk, Engine, RunOptions, SdcppEngine, SdcppOutput};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;

fn sleep_cmd() -> Vec<String> {
    vec!["sleep".into(), "30".into()]
}

fn engine(url: &str, dir: &std::path::Path, link_base: &str) -> SdcppEngine {
    SdcppEngine::new(
        "h3",
        14.0,
        url,
        sleep_cmd(),
        SdcppOutput {
            dir: dir.to_path_buf(),
            link_base: link_base.into(),
        },
        serde_json::json!({"width": 640, "video_frames": 25}),
    )
}

#[tokio::test]
async fn load_waits_for_capabilities_and_sleep_kills_the_child() {
    let (url, log) = spawn_mock(Mock::Completes).await;
    let dir = tempfile::tempdir().unwrap();
    let e = engine(&url, dir.path(), "");
    e.load("h3", 0).await.unwrap();
    assert!(e.alive());
    assert_eq!(e.gpu(), Some(0));
    assert!(log
        .lock()
        .unwrap()
        .iter()
        .any(|l| l.starts_with("GET /sdcpp/v1/capabilities")));
    e.sleep().await.unwrap();
    assert!(!e.alive());
    assert_eq!(e.gpu(), None);
}

#[tokio::test]
async fn run_before_load_is_not_loaded() {
    let (url, _) = spawn_mock(Mock::Completes).await;
    let dir = tempfile::tempdir().unwrap();
    let e = engine(&url, dir.path(), "");
    let err = e
        .run(
            &[ChatMessage::user("a fox")],
            "",
            &RunOptions::default(),
            CancellationToken::new(),
        )
        .await
        .err()
        .unwrap();
    assert!(matches!(err, silkai_adapters::EngineError::NotLoaded));
}

#[tokio::test]
async fn run_submits_prompt_with_params_and_links_the_clip() {
    let (url, log) = spawn_mock(Mock::Completes).await;
    let dir = tempfile::tempdir().unwrap();
    let e = engine(&url, dir.path(), "https://clinic.example/silkai/");
    e.load("h3", 0).await.unwrap();
    let messages = vec![
        ChatMessage::system("you make videos"),
        ChatMessage::user("a fox trotting through snow"),
    ];
    let mut rx = e
        .run(
            &messages,
            "",
            &RunOptions::default(),
            CancellationToken::new(),
        )
        .await
        .unwrap();
    let mut chunks = Vec::new();
    while let Some(c) = rx.recv().await {
        chunks.push(c);
    }
    e.discard().await.unwrap();

    let submit = log
        .lock()
        .unwrap()
        .iter()
        .find(|l| l.starts_with("POST /sdcpp/v1/vid_gen"))
        .cloned()
        .expect("vid_gen submitted");
    let body: serde_json::Value =
        serde_json::from_str(submit.splitn(3, ' ').nth(2).unwrap()).unwrap();
    assert_eq!(body["prompt"], "a fox trotting through snow");
    assert_eq!(body["width"], 640);
    assert_eq!(body["video_frames"], 25);

    let reasoning: Vec<&str> = chunks
        .iter()
        .filter_map(|c| match c {
            Chunk::Reasoning(t) => Some(t.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(reasoning, vec!["queued, 1 ahead\n", "generating\n"]);
    let text: String = chunks.iter().filter_map(|c| c.text()).collect();
    let link = "https://clinic.example/silkai/v1/files/h3/job_01TEST.webm";
    assert_eq!(
        text,
        format!("<video controls src=\"{link}\"></video>\n\n[job_01TEST.webm]({link})")
    );
    assert!(
        matches!(chunks.last(), Some(Chunk::End(end)) if end.finish_reason.as_deref() == Some("stop"))
    );
    let clip = std::fs::read(dir.path().join("job_01TEST.webm")).unwrap();
    assert_eq!(clip, b"WEBM-BYTES");
}

#[tokio::test]
async fn failed_job_is_a_rejection_with_the_servers_reason() {
    let (url, _) = spawn_mock(Mock::Fails).await;
    let dir = tempfile::tempdir().unwrap();
    let e = engine(&url, dir.path(), "");
    e.load("h3", 0).await.unwrap();
    let mut rx = e
        .run(
            &[ChatMessage::user("a fox")],
            "",
            &RunOptions::default(),
            CancellationToken::new(),
        )
        .await
        .unwrap();
    let mut chunks = Vec::new();
    while let Some(c) = rx.recv().await {
        chunks.push(c);
    }
    e.discard().await.unwrap();
    assert!(
        matches!(chunks.last(), Some(Chunk::Reject(r)) if r == "generate_video returned no results")
    );
    assert!(std::fs::read_dir(dir.path()).unwrap().next().is_none());
}

#[tokio::test]
async fn rejected_submission_carries_the_servers_error() {
    let (url, _) = spawn_mock(Mock::RefusesSubmit).await;
    let dir = tempfile::tempdir().unwrap();
    let e = engine(&url, dir.path(), "");
    e.load("h3", 0).await.unwrap();
    let mut rx = e
        .run(
            &[ChatMessage::user("a fox")],
            "",
            &RunOptions::default(),
            CancellationToken::new(),
        )
        .await
        .unwrap();
    let first = rx.recv().await;
    e.discard().await.unwrap();
    assert!(matches!(first, Some(Chunk::Reject(r)) if r == "video generation is not supported"));
}

#[tokio::test]
async fn no_user_turn_is_rejected_without_a_request() {
    let (url, log) = spawn_mock(Mock::Completes).await;
    let dir = tempfile::tempdir().unwrap();
    let e = engine(&url, dir.path(), "");
    e.load("h3", 0).await.unwrap();
    let mut rx = e
        .run(
            &[ChatMessage::system("only a system turn")],
            "",
            &RunOptions::default(),
            CancellationToken::new(),
        )
        .await
        .unwrap();
    let first = rx.recv().await;
    e.discard().await.unwrap();
    assert!(matches!(first, Some(Chunk::Reject(_))));
    assert!(!log
        .lock()
        .unwrap()
        .iter()
        .any(|l| l.starts_with("POST /sdcpp/v1/vid_gen")));
}

#[tokio::test]
async fn cancel_while_generating_cancels_the_job() {
    let (url, log) = spawn_mock(Mock::Hangs).await;
    let dir = tempfile::tempdir().unwrap();
    let e = engine(&url, dir.path(), "");
    e.load("h3", 0).await.unwrap();
    let cancel = CancellationToken::new();
    let mut rx = e
        .run(
            &[ChatMessage::user("a fox")],
            "",
            &RunOptions::default(),
            cancel.clone(),
        )
        .await
        .unwrap();
    // The first poll answers `generating`; only then is there a job to cancel.
    assert!(matches!(rx.recv().await, Some(Chunk::Reasoning(_))));
    cancel.cancel();
    assert!(rx.recv().await.is_none());
    for _ in 0..50 {
        if log
            .lock()
            .unwrap()
            .iter()
            .any(|l| l.starts_with("POST /sdcpp/v1/jobs/job_01TEST/cancel"))
        {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    e.discard().await.unwrap();
    assert!(log
        .lock()
        .unwrap()
        .iter()
        .any(|l| l.starts_with("POST /sdcpp/v1/jobs/job_01TEST/cancel")));
}

#[tokio::test]
async fn resumed_run_only_ends() {
    let (url, log) = spawn_mock(Mock::Completes).await;
    let dir = tempfile::tempdir().unwrap();
    let e = engine(&url, dir.path(), "");
    e.load("h3", 0).await.unwrap();
    let mut rx = e
        .run(
            &[ChatMessage::user("a fox")],
            "<video controls src=\"/v1/files/h3/x.webm\"></video>",
            &RunOptions::default(),
            CancellationToken::new(),
        )
        .await
        .unwrap();
    let first = rx.recv().await;
    e.discard().await.unwrap();
    assert!(matches!(first, Some(Chunk::End(_))));
    assert!(!log
        .lock()
        .unwrap()
        .iter()
        .any(|l| l.starts_with("POST /sdcpp/v1/vid_gen")));
}

#[derive(Clone, Copy)]
enum Mock {
    /// queued -> generating -> completed
    Completes,
    /// queued -> failed
    Fails,
    /// vid_gen answers 400
    RefusesSubmit,
    /// generating forever
    Hangs,
}

struct State {
    mock: Mock,
    polls: usize,
}

async fn spawn_mock(mock: Mock) -> (String, Arc<Mutex<Vec<String>>>) {
    let log = Arc::new(Mutex::new(Vec::new()));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let state = Arc::new(Mutex::new(State { mock, polls: 0 }));
    let log2 = log.clone();
    tokio::spawn(async move {
        loop {
            let Ok((mut sock, _)) = listener.accept().await else {
                break;
            };
            let log = log2.clone();
            let state = state.clone();
            tokio::spawn(async move {
                handle_conn(&mut sock, &log, &state).await;
            });
        }
    });
    (format!("http://{addr}"), log)
}

async fn handle_conn(
    sock: &mut tokio::net::TcpStream,
    log: &Mutex<Vec<String>>,
    state: &Mutex<State>,
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
    let resp = respond(&method, &path, state);
    let _ = sock.write_all(resp.as_bytes()).await;
}

fn respond(method: &str, path: &str, state: &Mutex<State>) -> String {
    let mut st = state.lock().unwrap();
    match (method, path) {
        ("GET", "/sdcpp/v1/capabilities") => json(200, r#"{"supported_modes":["vid_gen"]}"#),
        ("POST", "/sdcpp/v1/vid_gen") => match st.mock {
            Mock::RefusesSubmit => json(400, r#"{"error":"video generation is not supported"}"#),
            _ => json(
                202,
                r#"{"id":"job_01TEST","kind":"vid_gen","status":"queued","poll_url":"/sdcpp/v1/jobs/job_01TEST"}"#,
            ),
        },
        ("GET", "/sdcpp/v1/jobs/job_01TEST") => {
            st.polls += 1;
            let body = match (st.mock, st.polls) {
                (Mock::Hangs, _) => job("generating", 0, "null", "null"),
                (Mock::Fails, 1) => job("queued", 0, "null", "null"),
                (Mock::Fails, _) => job(
                    "failed",
                    0,
                    "null",
                    r#"{"code":"generation_failed","message":"generate_video returned no results"}"#,
                ),
                (_, 1) => job("queued", 1, "null", "null"),
                (_, 2) => job("generating", 0, "null", "null"),
                (_, _) => job(
                    "completed",
                    0,
                    r#"{"output_format":"webm","mime_type":"video/webm","fps":24,"frame_count":25,"b64_json":"V0VCTS1CWVRFUw=="}"#,
                    "null",
                ),
            };
            json(200, &body)
        }
        ("POST", "/sdcpp/v1/jobs/job_01TEST/cancel") => json(200, r#"{"status":"cancelled"}"#),
        _ => json(404, r#"{"error":"no such route"}"#),
    }
}

fn job(status: &str, queue: u64, result: &str, error: &str) -> String {
    format!(
        r#"{{"id":"job_01TEST","kind":"vid_gen","status":"{status}","queue_position":{queue},"result":{result},"error":{error}}}"#
    )
}

fn json(status: u16, body: &str) -> String {
    let reason = match status {
        200 => "OK",
        202 => "Accepted",
        400 => "Bad Request",
        _ => "Not Found",
    };
    format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
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
        if let Some(idx) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
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
            return Some((method, path, String::from_utf8_lossy(&body).into_owned()));
        }
        if buf.len() > 64 * 1024 {
            return None;
        }
    }
    None
}
