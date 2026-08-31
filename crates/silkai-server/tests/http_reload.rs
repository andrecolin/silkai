use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;

const VALID_TOML: &str = r#"
listen = "127.0.0.1:0"

[resources]
gpu_total_gb = 32
gpu_headroom_gb = 3
ram_total_gb = 128
ram_headroom_gb = 32
prefetch_on_start = true
request_timeout_secs = 600

[models.soap]
engine = "fake"
path = "/models/soap.bin"
vram_gb = 28
priority = "normal"
exclusive = true
slots = 1
keep_warm = true
transport = "http"

[models.whisper]
engine = "fake"
path = "/models/whisper.bin"
vram_gb = 12
priority = "live"
exclusive = false
slots = 2
keep_warm = true
transport = "http"
"#;

fn reload_req() -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/admin/reload")
        .body(Body::empty())
        .unwrap()
}

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

fn write_config(toml: &str) -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("c.toml");
    std::fs::write(&path, toml).unwrap();
    (dir, path)
}

#[tokio::test]
async fn reload_keeps_inflight_and_returns_ok() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("c.toml");
    std::fs::write(&path, VALID_TOML).unwrap();
    let app = silkai_server::app::app_from_path(&path).await.unwrap();
    let res = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/admin/reload")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
}

#[tokio::test]
async fn reload_bad_file_keeps_old_config() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("c.toml");
    std::fs::write(&path, VALID_TOML).unwrap();
    let app = silkai_server::app::app_from_path(&path).await.unwrap();
    std::fs::write(&path, "listen = ").unwrap();
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/admin/reload")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);
    let res = app
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let bytes = axum::body::to_bytes(res.into_body(), usize::MAX)
        .await
        .unwrap();
    let text = String::from_utf8(bytes.to_vec()).unwrap();
    assert!(text.contains("soap"));
}

#[tokio::test]
async fn reload_while_whisper_running_returns_conflict() {
    let (_dir, path) = write_config(VALID_TOML);
    let app = silkai_server::app::app_from_path(&path).await.unwrap();
    let whisper_app = app.clone();
    let whisper =
        tokio::spawn(async move { whisper_app.oneshot(chat("whisper", true)).await.unwrap() });
    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    let res = app.oneshot(reload_req()).await.unwrap();
    assert_eq!(res.status(), StatusCode::CONFLICT);
    let _ = whisper.await;
}

#[tokio::test]
async fn reload_idle_rebuilds_runtime_from_file() {
    let (_dir, path) = write_config(VALID_TOML);
    let app = silkai_server::app::app_from_path(&path).await.unwrap();
    std::fs::write(&path, soap_only_toml()).unwrap();
    let res = app.clone().oneshot(reload_req()).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let res = app
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let bytes = axum::body::to_bytes(res.into_body(), usize::MAX)
        .await
        .unwrap();
    let text = String::from_utf8(bytes.to_vec()).unwrap();
    assert!(text.contains("soap"));
    assert!(!text.contains("whisper"));
}

fn soap_only_toml() -> String {
    VALID_TOML
        .lines()
        .take_while(|line| !line.contains("models.whisper"))
        .collect::<Vec<_>>()
        .join("\n")
}

#[tokio::test]
async fn reload_via_serve_listener_idle_ok() {
    let (_dir, path) = write_config(VALID_TOML);
    let cfg = silkai_server::config::load_from_path(&path).unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        silkai_server::serve_listener(listener, cfg, Some(path))
            .await
            .unwrap();
    });
    let res = reqwest::Client::new()
        .post(format!("http://{addr}/admin/reload"))
        .send()
        .await
        .unwrap();
    assert_eq!(res.status(), reqwest::StatusCode::OK);
}
