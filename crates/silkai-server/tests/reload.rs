//! A config reload keeps untouched models resident and only rebuilds what
//! changed.

use silkai_adapters::ChatMessage;
use silkai_server::config::load_from_str;
use silkai_server::runtime::Runtime;

const BASE: &str = r#"
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
"#;

fn with_soap(vram: u32) -> silkai_server::config::AppConfig {
    load_from_str(&format!(
        "{BASE}\n[models.soap]\nengine = \"fake\"\npath = \"/models/soap.bin\"\nvram_gb = {vram}\npriority = \"normal\"\nexclusive = true\n"
    ))
    .unwrap()
}

fn kinds(rt: &Runtime, model: &str, after: u64) -> Vec<&'static str> {
    rt.events()
        .since(after)
        .into_iter()
        .filter(|e| e.model.as_deref() == Some(model))
        .map(|e| e.kind)
        .collect()
}

fn tier_of(rt: &Runtime, model: &str) -> (String, Option<u32>) {
    let st = rt.status();
    let m = st.models.iter().find(|m| m.name == model).unwrap();
    (m.state.clone(), m.gpu)
}

#[tokio::test]
async fn unchanged_models_stay_where_they_are() {
    let rt = Runtime::new(with_soap(28)).await.unwrap();
    let (job, mut rx) = rt
        .submit_chat("soap", vec![ChatMessage::user("note")], Default::default())
        .await
        .unwrap();
    while rx.recv().await.is_some() {}
    rt.finished(job).await;
    assert_eq!(tier_of(&rt, "soap"), ("bench".into(), Some(0)));
    assert_eq!(tier_of(&rt, "whisper").0, "shelf");
    let mark = rt.events().last_seq();

    let again = Runtime::rebuild(with_soap(28), &rt).await.unwrap();
    assert_eq!(tier_of(&again, "soap"), ("bench".into(), Some(0)));
    assert_eq!(tier_of(&again, "whisper").0, "shelf");
    // No engine work at all for either model: no warm, load, wake, discard.
    assert!(
        kinds(&again, "soap", mark).is_empty(),
        "{:?}",
        kinds(&again, "soap", mark)
    );
    assert!(kinds(&again, "whisper", mark).is_empty());
    // Same event log, same sequence numbers: the history carried over.
    assert!(again.events().last_seq() >= mark);

    // The adopted model answers without a reload.
    let (job, mut rx) = again
        .submit_chat("soap", vec![ChatMessage::user("again")], Default::default())
        .await
        .unwrap();
    let mut out = String::new();
    while let Some(t) = rx.recv().await {
        out.push_str(&t);
    }
    again.finished(job).await;
    assert_eq!(out, "again world");
    assert_eq!(kinds(&again, "soap", mark), vec!["start", "finish"]);
}

#[tokio::test]
async fn changed_model_is_discarded_and_rewarmed_others_untouched() {
    let rt = Runtime::new(with_soap(28)).await.unwrap();
    let mark = rt.events().last_seq();
    let again = Runtime::rebuild(with_soap(20), &rt).await.unwrap();
    assert_eq!(kinds(&again, "soap", mark), vec!["discard", "warm"]);
    assert!(kinds(&again, "whisper", mark).is_empty());
    assert_eq!(tier_of(&again, "soap").0, "shelf");
}

#[tokio::test]
async fn removed_model_is_discarded_added_model_is_warmed() {
    let rt = Runtime::new(with_soap(28)).await.unwrap();
    let mark = rt.events().last_seq();
    let only_whisper = load_from_str(BASE).unwrap();
    let again = Runtime::rebuild(only_whisper, &rt).await.unwrap();
    assert_eq!(kinds(&again, "soap", mark), vec!["discard"]);
    assert!(again.status().models.iter().all(|m| m.name != "soap"));

    let mark = again.events().last_seq();
    let back = Runtime::rebuild(with_soap(28), &again).await.unwrap();
    assert_eq!(kinds(&back, "soap", mark), vec!["warm"]);
    assert!(kinds(&back, "whisper", mark).is_empty());
}
