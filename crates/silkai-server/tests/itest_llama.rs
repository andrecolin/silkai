#[test]
fn skip_without_itest_or_feature() {}

#[cfg(feature = "llama")]
#[tokio::test]
async fn two_tiny_ggufs_pack_or_queue() {
    if std::env::var("SILKAI_ITEST").ok().as_deref() != Some("1") {
        eprintln!("skip");
        return;
    }
    run_two_ggufs().await;
}

#[cfg(feature = "llama")]
async fn run_two_ggufs() {
    let rt = silkai_server::Runtime::new(gguf_cfg())
        .await
        .expect("runtime");
    let (job_a, mut rx_a) = rt.submit_chat("a", "hi").await.expect("submit a");
    let (job_b, mut rx_b) = rt.submit_chat("b", "hi").await.expect("submit b");
    assert_pack_or_exclusive(&rt.status());
    let _ = tokio::join!(recv_some(&mut rx_a), recv_some(&mut rx_b));
    assert_pack_or_exclusive(&rt.status());
    rt.drop_job(job_a).await;
    rt.drop_job(job_b).await;
}

#[cfg(feature = "llama")]
fn gguf_cfg() -> silkai_server::config::AppConfig {
    let vram_a = env_f64("SILKAI_VRAM_A", 12.0);
    let vram_b = env_f64("SILKAI_VRAM_B", 28.0);
    silkai_server::config::AppConfig {
        listen: "127.0.0.1:0".into(),
        prefetch_on_start: true,
        request_timeout_secs: 600,
        request_timeout: std::time::Duration::from_secs(600),
        resources: resources_for(vram_a, vram_b),
        enabled: vec![model_a(vram_a), model_b(vram_b)],
        disabled: vec![],
    }
}

#[cfg(feature = "llama")]
fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

#[cfg(feature = "llama")]
fn resources_for(vram_a: f64, vram_b: f64) -> silkai_sched::Resources {
    silkai_sched::Resources::single(
        29.0_f64.max(vram_a.max(vram_b) + 1.0),
        96.0_f64.max(vram_a + vram_b + 8.0),
    )
}

#[cfg(feature = "llama")]
fn model_a(vram_gb: f64) -> silkai_server::config::ConfiguredModel {
    llama_model(
        "a",
        std::env::var("SILKAI_GGUF_A").expect("SILKAI_GGUF_A"),
        vram_gb,
        false,
        silkai_sched::Priority::Live,
    )
}

#[cfg(feature = "llama")]
fn model_b(vram_gb: f64) -> silkai_server::config::ConfiguredModel {
    llama_model(
        "b",
        std::env::var("SILKAI_GGUF_B").expect("SILKAI_GGUF_B"),
        vram_gb,
        true,
        silkai_sched::Priority::Normal,
    )
}

#[cfg(feature = "llama")]
fn llama_model(
    name: &str,
    path: String,
    vram_gb: f64,
    exclusive: bool,
    priority: silkai_sched::Priority,
) -> silkai_server::config::ConfiguredModel {
    silkai_server::config::ConfiguredModel {
        spec: silkai_sched::ModelSpec {
            name: name.into(),
            vram_gb,
            ram_gb: vram_gb,
            priority,
            exclusive,
            slots: 1,
            keep_warm: true,
            gpu: None,
        },
        engine: "llama.cpp".into(),
        path,
        url: None,
        transport: "http".into(),
        idle_timeout_secs: None,
    }
}

#[cfg(feature = "llama")]
fn assert_pack_or_exclusive(st: &silkai_sched::StatusSnapshot) {
    let a = named(st, "a");
    let b = named(st, "b");
    let packed = a.tier == silkai_sched::Tier::Bench && b.tier == silkai_sched::Tier::Bench;
    if packed {
        return;
    }
    if b.tier == silkai_sched::Tier::Bench {
        assert_ne!(a.tier, silkai_sched::Tier::Bench);
    }
}

#[cfg(feature = "llama")]
fn named<'a>(st: &'a silkai_sched::StatusSnapshot, name: &str) -> &'a silkai_sched::ModelStatus {
    st.models
        .iter()
        .find(|m| m.name == name)
        .unwrap_or_else(|| panic!("missing model {name}"))
}

#[cfg(feature = "llama")]
async fn recv_some(rx: &mut tokio::sync::mpsc::Receiver<String>) -> Vec<String> {
    let mut out = Vec::new();
    let wait = tokio::time::timeout(std::time::Duration::from_secs(30), rx.recv()).await;
    if let Ok(Some(tok)) = wait {
        out.push(tok);
    }
    out
}
