//! A live request must not wait behind a long load of a lower-priority
//! model, and that load's job must come back after, not fail.

use std::time::{Duration, Instant};

use silkai_adapters::{ChatMessage, FakeEngine};
use silkai_server::runtime::Runtime;

fn cfg() -> silkai_server::config::AppConfig {
    silkai_server::config::load_from_str(
        r#"
[resources]
gpu_total_gb = 32
gpu_headroom_gb = 3
ram_total_gb = 128
ram_headroom_gb = 32

[models.whisper]
engine = "fake"
path = "/models/whisper.bin"
vram_gb = 12
priority = "live"
slots = 2

[models.soap]
engine = "fake"
path = "/models/soap.bin"
vram_gb = 28
priority = "normal"
exclusive = true
"#,
    )
    .unwrap()
}

async fn state_of(rt: &Runtime, name: &str) -> String {
    rt.status()
        .models
        .into_iter()
        .find(|m| m.name == name)
        .map(|m| m.state)
        .unwrap()
}

async fn wait_state(rt: &Runtime, name: &str, want: &str) {
    for _ in 0..400 {
        if state_of(rt, name).await == want {
            return;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    panic!("{name} never reached {want}");
}

async fn collect(mut rx: tokio::sync::mpsc::Receiver<String>) -> String {
    let mut out = String::new();
    while let Some(t) = rx.recv().await {
        out.push_str(&t);
    }
    out
}

#[tokio::test]
async fn live_request_interrupts_a_load_and_the_job_comes_back() {
    let rt = Runtime::new(cfg()).await.unwrap();
    let _gate = FakeEngine::hold_next_load("soap");
    let soap = {
        let rt = rt.clone();
        tokio::spawn(async move {
            rt.submit_chat("soap", vec![ChatMessage::user("note")], Default::default())
                .await
                .unwrap()
        })
    };
    wait_state(&rt, "soap", "loading").await;

    // The live request must be accepted while soap is still loading, and
    // must not wait for that load to finish (the gate is never released).
    let t0 = Instant::now();
    let (w_job, w_rx) = rt
        .submit_chat("whisper", vec![ChatMessage::user("hi")], Default::default())
        .await
        .unwrap();
    assert!(
        t0.elapsed() < Duration::from_secs(2),
        "live submit waited {:?}",
        t0.elapsed()
    );
    assert_eq!(collect(w_rx).await, "hi world");
    rt.finished(w_job).await;

    // The interrupted soap job was re-queued, not failed: its submit
    // returns Ok, and once whisper is done its tokens arrive.
    let (s_job, s_rx) = soap.await.unwrap();
    assert_eq!(collect(s_rx).await, "note world");
    rt.finished(s_job).await;

    let kinds: Vec<String> = rt
        .events()
        .since(0)
        .into_iter()
        .filter(|e| e.model.as_deref() == Some("soap"))
        .map(|e| format!("{}{}", e.kind, if e.error.is_some() { "!" } else { "" }))
        .collect();
    // One interrupted wake, one kill, then exactly one clean wake and run.
    // The interrupted wake and the kill's sleep event race each other.
    assert_eq!(kinds[0], "warm", "{kinds:?}");
    assert_eq!(kinds[1], "preempt", "{kinds:?}");
    assert_eq!(
        kinds.iter().filter(|k| *k == "wake!").count(),
        1,
        "{kinds:?}"
    );
    assert_eq!(
        kinds.iter().filter(|k| *k == "sleep").count(),
        1,
        "{kinds:?}"
    );
    assert_eq!(
        kinds.iter().filter(|k| *k == "start").count(),
        1,
        "{kinds:?}"
    );
    assert_eq!(
        &kinds[kinds.len() - 3..],
        ["wake", "start", "finish"],
        "{kinds:?}"
    );
    assert_eq!(rt.counters()["soap"].faults, 0);
}
