use silkai_sched::clinic::{clinic_models, clinic_resources};
use silkai_sched::{ModelSpec, Priority};
use silkai_server::config::{AppConfig, ConfiguredModel};
use silkai_server::runtime::{Runtime, RuntimeError};

fn clinic_cfg() -> AppConfig {
    let models = clinic_models()
        .into_iter()
        .map(|spec| ConfiguredModel {
            engine: "fake".into(),
            path: format!("/models/{}.bin", spec.name),
            transport: "http".into(),
            idle_timeout_secs: None,
            spec,
        })
        .collect();
    AppConfig {
        listen: "127.0.0.1:0".into(),
        prefetch_on_start: true,
        request_timeout_secs: 600,
        resources: clinic_resources(),
        enabled: models,
        disabled: vec![],
    }
}

fn too_big() -> ConfiguredModel {
    ConfiguredModel {
        engine: "fake".into(),
        path: "/models/too-big.bin".into(),
        transport: "http".into(),
        idle_timeout_secs: None,
        spec: ModelSpec {
            name: "too-big".into(),
            vram_gb: 40.0,
            ram_gb: 40.0,
            priority: Priority::Normal,
            exclusive: true,
            slots: 1,
            keep_warm: true,
        },
    }
}

#[tokio::test]
async fn prefetch_then_soap_wakes_fake_engine() {
    let rt = Runtime::new(clinic_cfg()).await.unwrap();
    let (job, mut tokens) = rt.submit_chat("soap", "note").await.unwrap();
    let mut out = String::new();
    while let Some(t) = tokens.recv().await {
        out.push_str(&t);
    }
    rt.finished(job).await;
    assert_eq!(out, "note world");
    let st = rt.status();
    let soap = st.models.iter().find(|m| m.name == "soap").unwrap();
    assert_eq!(soap.running, 0);
}

#[tokio::test]
async fn unknown_model_errors() {
    let rt = Runtime::new(clinic_cfg()).await.unwrap();
    let err = rt.submit_chat("nope", "x").await.unwrap_err();
    assert!(matches!(err, RuntimeError::Unknown));
}

#[tokio::test]
async fn disabled_model_errors() {
    let mut cfg = clinic_cfg();
    cfg.disabled.push(too_big());
    let rt = Runtime::new(cfg).await.unwrap();
    let err = rt.submit_chat("too-big", "x").await.unwrap_err();
    assert!(matches!(err, RuntimeError::Disabled));
}
