use std::time::Duration;

use silkai_adapters::{ChatMessage, FakeEngine};
use silkai_sched::{ModelSpec, Priority, Resources};
use silkai_server::config::{AppConfig, ConfiguredModel};
use silkai_server::runtime::Runtime;
use silkai_server::sampler::{GpuSample, Sample};

fn model(name: &str, engine: &str, vram: f64) -> ConfiguredModel {
    ConfiguredModel {
        engine: engine.into(),
        path: format!("/models/{name}"),
        url: Some("http://127.0.0.1:1".into()),
        cmd: Vec::new(),
        transport: "http".into(),
        idle_timeout_secs: None,
        spec: ModelSpec {
            name: name.into(),
            vram_gb: vram,
            ram_gb: vram,
            priority: Priority::Normal,
            exclusive: false,
            slots: 1,
            keep_warm: true,
            gpu: None,
            gpus: Vec::new(),
        },
    }
}

fn cfg(models: Vec<ConfiguredModel>) -> AppConfig {
    AppConfig {
        listen: "127.0.0.1:0".into(),
        prefetch_on_start: true,
        request_timeout_secs: 600,
        request_timeout: Duration::from_secs(600),
        resources: Resources::single(29.0, 96.0),
        enabled: models,
        disabled: vec![],
        ui: Default::default(),
    }
}

async fn state_of(rt: &Runtime, name: &str) -> String {
    rt.status()
        .models
        .into_iter()
        .find(|m| m.name == name)
        .map(|m| m.state)
        .unwrap_or_default()
}

async fn wait_for(rt: &Runtime, name: &str, want: &str) {
    for _ in 0..200 {
        if state_of(rt, name).await == want {
            return;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    panic!(
        "{name} never reached {want}, at {}",
        state_of(rt, name).await
    );
}

#[tokio::test]
async fn loading_state_shows_while_load_is_in_flight() {
    let rt = Runtime::new(cfg(vec![model("held", "fake", 8.0)]))
        .await
        .unwrap();
    assert_eq!(state_of(&rt, "held").await, "shelf");
    let gate = FakeEngine::hold_next_load("held");
    let submit = {
        let rt = rt.clone();
        tokio::spawn(async move {
            rt.submit_chat("held", vec![ChatMessage::user("hi")])
                .await
                .unwrap()
        })
    };
    wait_for(&rt, "held", "loading").await;
    gate.notify_one();
    let (job, mut rx) = submit.await.unwrap();
    wait_for(&rt, "held", "bench").await;
    while rx.recv().await.is_some() {}
    rt.finished(job).await;
}

#[tokio::test]
async fn ram_counts_only_engines_with_a_shelf() {
    let fake = Runtime::new(cfg(vec![
        model("a", "fake", 12.0),
        model("b", "fake", 10.0),
    ]))
    .await
    .unwrap();
    assert_eq!(fake.status().ram_used_gb, 22.0);
    let http = Runtime::new(cfg(vec![
        model("a", "vllm", 12.0),
        model("b", "vllm", 10.0),
    ]))
    .await
    .unwrap();
    let st = http.status();
    assert!(st.models.iter().all(|m| m.state == "shelf"));
    assert_eq!(st.ram_used_gb, 0.0);
}

#[tokio::test]
async fn status_carries_engine_and_budget_facts() {
    let rt = Runtime::new(cfg(vec![model("a", "vllm", 12.0)]))
        .await
        .unwrap();
    let st = rt.status();
    let a = &st.models[0];
    assert_eq!(a.engine, "vllm");
    assert_eq!(a.budget_gb, 12.0);
    assert_eq!(a.priority, "normal");
    assert_eq!(a.slots, 1);
    assert_eq!(a.sessions, 0);
    assert_eq!(a.measured_gb, None);
    assert!(!st.measured);
}

#[tokio::test]
async fn injected_sample_fills_measured_fields() {
    let rt = Runtime::new(cfg(vec![model("a", "fake", 12.0)]))
        .await
        .unwrap();
    rt.inject_sample(Some(Sample {
        gpus: vec![GpuSample {
            id: 0,
            used_gb: 3.5,
            total_gb: 32.0,
        }],
        apps: vec![(std::process::id(), 3.5)],
    }));
    let st = rt.status();
    assert!(st.measured);
    assert_eq!(st.gpus[0].measured_used_gb, Some(3.5));
    assert_eq!(st.gpus[0].total_gb, Some(32.0));
    // The fake engine has no pid, so nothing is attributed to it.
    assert_eq!(st.models[0].measured_gb, None);
}

#[tokio::test]
async fn no_negative_zero_in_json() {
    let rt = Runtime::new(cfg(vec![model("a", "fake", 12.0)]))
        .await
        .unwrap();
    let json = serde_json::to_string(&rt.status()).unwrap();
    assert!(!json.contains("-0.0"), "{json}");
}
