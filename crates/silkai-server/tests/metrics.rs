use axum::body::Body;
use axum::http::Request;
use silkai_server::app::test_app;
use tower::ServiceExt;

async fn get(app: axum::Router, uri: &str) -> (u16, String) {
    let res = app
        .oneshot(Request::builder().uri(uri).body(Body::empty()).unwrap())
        .await
        .unwrap();
    let status = res.status().as_u16();
    let bytes = axum::body::to_bytes(res.into_body(), usize::MAX)
        .await
        .unwrap();
    (status, String::from_utf8(bytes.to_vec()).unwrap())
}

fn chat(model: &str) -> Request<Body> {
    let body = serde_json::json!({"model": model, "messages": [{"role":"user","content":"hi"}]});
    Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

/// Every sample line is `name{labels} value` or `name value`; comments are
/// `# HELP` / `# TYPE`. No crate needed to check that much.
fn well_formed(text: &str) -> bool {
    text.lines().all(|l| {
        if l.starts_with("# HELP ") || l.starts_with("# TYPE ") {
            return true;
        }
        let Some((name, value)) = l.rsplit_once(' ') else {
            return false;
        };
        let bare = name.split('{').next().unwrap_or("");
        bare.chars().all(|c| c.is_ascii_lowercase() || c == '_')
            && value.parse::<f64>().is_ok()
            && (!name.contains('{') || name.ends_with('}'))
    })
}

#[tokio::test]
async fn metrics_reflect_state_and_counters() {
    let app = test_app().await;
    let (status, before) = get(app.clone(), "/metrics").await;
    assert_eq!(status, 200);
    assert!(well_formed(&before), "{before}");
    assert!(before.contains("silkai_model_state{model=\"soap\",state=\"shelf\"} 1"));
    assert!(before.contains("silkai_gpu_schedulable_gb{gpu=\"0\"} 29"));

    let res = app.clone().oneshot(chat("soap")).await.unwrap();
    assert_eq!(res.status(), 200);
    let _ = axum::body::to_bytes(res.into_body(), usize::MAX).await;

    let (_, after) = get(app, "/metrics").await;
    assert!(well_formed(&after), "{after}");
    assert!(
        after.contains("silkai_wakes_total{model=\"soap\"} 1"),
        "{after}"
    );
    assert!(after.contains("silkai_model_state{model=\"soap\",state=\"bench\"} 1"));
    assert!(after.contains("# TYPE silkai_load_seconds_sum counter"));
}
