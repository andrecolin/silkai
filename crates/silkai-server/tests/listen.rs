use silkai_sched::clinic::{clinic_models, clinic_resources};
use silkai_server::config::{AppConfig, ConfiguredModel};
use silkai_server::serve_listener;

fn clinic_cfg() -> AppConfig {
    AppConfig {
        listen: "127.0.0.1:0".into(),
        prefetch_on_start: true,
        request_timeout_secs: 600,
        request_timeout: std::time::Duration::from_secs(600),
        resources: clinic_resources(),
        enabled: clinic_models()
            .into_iter()
            .map(|spec| ConfiguredModel {
                engine: "fake".into(),
                path: format!("/models/{}.bin", spec.name),
                transport: "http".into(),
                idle_timeout_secs: None,
                spec,
            })
            .collect(),
        disabled: vec![],
    }
}

#[tokio::test]
async fn health_via_tcp() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let cfg = clinic_cfg();
    tokio::spawn(async move {
        serve_listener(listener, cfg).await.unwrap();
    });
    let body = reqwest::get(format!("http://{addr}/health"))
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert_eq!(body, "ok");
}
