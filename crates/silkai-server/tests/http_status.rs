use axum::body::Body;
use axum::http::{Request, StatusCode};
use silkai_server::app::test_app;
use tower::ServiceExt;

#[tokio::test]
async fn health_ok() {
    let app = test_app().await;
    let res = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
}

#[tokio::test]
async fn lists_configured_models() {
    let app = test_app().await;
    let res = app
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(res.into_body(), usize::MAX)
        .await
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let ids: Vec<&str> = v["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["id"].as_str().unwrap())
        .collect();
    assert!(ids.contains(&"whisper"));
    assert!(ids.contains(&"soap"));
}

#[tokio::test]
async fn status_json_has_tiers() {
    let app = test_app().await;
    let res = app
        .oneshot(
            Request::builder()
                .uri("/v1/status")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(res.into_body(), usize::MAX)
        .await
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert!(v.get("models").is_some());
    assert!(v.get("gpu_used_gb").is_some());
    let gpus = v["gpus"].as_array().expect("gpus");
    assert_eq!(gpus.len(), 1);
    assert_eq!(gpus[0]["id"], 0);
    assert_eq!(gpus[0]["used_gb"], 0.0);
    assert_eq!(gpus[0]["schedulable_gb"], 29.0);
}
