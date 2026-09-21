//! `/v1/files/{model}/{name}` hands out what a file-producing engine wrote,
//! and nothing else.

use axum::body::Body;
use axum::http::{header, Request, StatusCode};
use silkai_server::app::app_from_config;
use silkai_server::config::load_from_str;
use tower::ServiceExt;

async fn app_with_output(dir: &std::path::Path) -> axum::Router {
    let t = format!(
        r#"
listen = "127.0.0.1:8080"
[resources]
gpu_total_gb = 32
ram_total_gb = 128
prefetch_on_start = false
[models.h3]
engine = "sdcpp"
path = "h3"
url = "http://127.0.0.1:1"
cmd = ["sleep", "30"]
output_dir = "{}"
vram_gb = 14
priority = "normal"
keep_warm = false
[models.soap]
engine = "fake"
path = "soap"
vram_gb = 4
priority = "normal"
"#,
        dir.display()
    );
    app_from_config(load_from_str(&t).unwrap()).await
}

async fn get(app: axum::Router, uri: &str) -> (StatusCode, Option<String>, Vec<u8>) {
    let res = app
        .oneshot(Request::builder().uri(uri).body(Body::empty()).unwrap())
        .await
        .unwrap();
    let status = res.status();
    let ctype = res
        .headers()
        .get(header::CONTENT_TYPE)
        .map(|v| v.to_str().unwrap().to_string());
    let bytes = axum::body::to_bytes(res.into_body(), usize::MAX)
        .await
        .unwrap();
    (status, ctype, bytes.to_vec())
}

#[tokio::test]
async fn serves_a_clip_from_the_models_output_dir() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("job_01.webm"), b"WEBM").unwrap();
    let app = app_with_output(dir.path()).await;
    let (status, ctype, body) = get(app, "/v1/files/h3/job_01.webm").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(ctype.as_deref(), Some("video/webm"));
    assert_eq!(body, b"WEBM");
}

#[tokio::test]
async fn missing_file_unknown_model_and_engine_without_output_are_404() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("job_01.webm"), b"WEBM").unwrap();
    for uri in [
        "/v1/files/h3/job_02.webm",
        "/v1/files/nope/job_01.webm",
        "/v1/files/soap/job_01.webm",
    ] {
        let app = app_with_output(dir.path()).await;
        assert_eq!(get(app, uri).await.0, StatusCode::NOT_FOUND, "{uri}");
    }
}

#[tokio::test]
async fn names_that_could_leave_the_directory_are_404() {
    let dir = tempfile::tempdir().unwrap();
    let secret = dir.path().join("secret.txt");
    std::fs::write(&secret, b"no").unwrap();
    let sub = dir.path().join("h3");
    std::fs::create_dir(&sub).unwrap();
    std::fs::write(sub.join(".hidden"), b"no").unwrap();
    for uri in [
        "/v1/files/h3/..%2Fsecret.txt",
        "/v1/files/h3/%2E%2E%2Fsecret.txt",
        "/v1/files/h3/.hidden",
    ] {
        let app = app_with_output(&sub).await;
        assert_eq!(get(app, uri).await.0, StatusCode::NOT_FOUND, "{uri}");
    }
}
