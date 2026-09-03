use axum::body::Body;
use axum::http::{Request, StatusCode};
use silkai_server::app::test_app_ui;
use silkai_server::config::load_from_str;
use tower::ServiceExt;

fn get(uri: &str, auth: Option<&str>) -> Request<Body> {
    let mut b = Request::builder().uri(uri);
    if let Some(a) = auth {
        b = b.header("authorization", a);
    }
    b.body(Body::empty()).unwrap()
}

#[tokio::test]
async fn page_is_404_unless_enabled() {
    let off = test_app_ui(false, None).await;
    let res = off.oneshot(get("/ui", None)).await.unwrap();
    assert_eq!(res.status(), StatusCode::NOT_FOUND);

    let on = test_app_ui(true, None).await;
    let res = on.oneshot(get("/ui", None)).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    assert_eq!(res.headers()["cache-control"], "no-store");
    let body = axum::body::to_bytes(res.into_body(), usize::MAX)
        .await
        .unwrap();
    let html = std::str::from_utf8(&body).unwrap();
    assert!(html.contains("<title>SilkAI</title>"));
    assert!(
        !html.contains("http://") && !html.contains("https://"),
        "page must not load anything external"
    );
}

#[tokio::test]
async fn token_guards_page_metrics_and_admin_but_not_the_api() {
    let app = test_app_ui(true, Some("s3cret")).await;
    for uri in ["/ui", "/metrics"] {
        let res = app.clone().oneshot(get(uri, None)).await.unwrap();
        assert_eq!(res.status(), StatusCode::UNAUTHORIZED, "{uri}");
        assert!(res.headers()["www-authenticate"]
            .to_str()
            .unwrap()
            .starts_with("Basic"));
        let res = app
            .clone()
            .oneshot(get(uri, Some("Bearer s3cret")))
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK, "{uri} bearer");
        // "silkai:s3cret"
        let res = app
            .clone()
            .oneshot(get(uri, Some("Basic c2lsa2FpOnMzY3JldA==")))
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK, "{uri} basic");
        let res = app
            .clone()
            .oneshot(get(uri, Some("Bearer wrong")))
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::UNAUTHORIZED, "{uri} wrong");
    }
    let reload = Request::builder()
        .method("POST")
        .uri("/admin/reload")
        .body(Body::empty())
        .unwrap();
    let res = app.clone().oneshot(reload).await.unwrap();
    assert_eq!(res.status(), StatusCode::UNAUTHORIZED);
    for uri in ["/v1/status", "/v1/models", "/health"] {
        let res = app.clone().oneshot(get(uri, None)).await.unwrap();
        assert_eq!(res.status(), StatusCode::OK, "{uri} must stay open");
    }
}

#[tokio::test]
async fn ui_section_parses_and_empty_token_means_none() {
    let base = r#"
[resources]
gpu_total_gb = 32
ram_total_gb = 128
"#;
    let cfg = load_from_str(base).unwrap();
    assert!(!cfg.ui.enabled);
    assert_eq!(cfg.ui.token, None);

    let cfg = load_from_str(&format!("{base}\n[ui]\nenabled = true\ntoken = \"\"\n")).unwrap();
    assert!(cfg.ui.enabled);
    assert_eq!(cfg.ui.token, None);

    let cfg = load_from_str(&format!("{base}\n[ui]\nenabled = true\ntoken = \"abc\"\n")).unwrap();
    assert_eq!(cfg.ui.token.as_deref(), Some("abc"));
}
