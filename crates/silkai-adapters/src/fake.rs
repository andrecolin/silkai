use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::time::Duration;

use async_trait::async_trait;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::{last_content, ChatMessage, Chunk, Engine, EngineError, RunEnd, RunOptions, Usage};

fn fail_next() -> &'static Mutex<HashMap<String, FailNext>> {
    static MAP: OnceLock<Mutex<HashMap<String, FailNext>>> = OnceLock::new();
    MAP.get_or_init(|| Mutex::new(HashMap::new()))
}

#[derive(Default)]
struct FailNext {
    load: u32,
    run: u32,
    reject_run: u32,
    /// Hold the next load until this is notified, so tests can observe the
    /// runtime while a load is in flight.
    hold_load: Option<Arc<tokio::sync::Notify>>,
}

fn take_fail(name: &str, which: fn(&mut FailNext) -> &mut u32) -> bool {
    let mut map = fail_next().lock().expect("fail-next mutex");
    let Some(slot) = map.get_mut(name) else {
        return false;
    };
    let count = which(slot);
    if *count == 0 {
        return false;
    }
    *count -= 1;
    true
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tier {
    Cupboard,
    Shelf,
    Bench,
}

struct Inner {
    log: Vec<String>,
    tier: Tier,
    gpu: Option<u32>,
    /// A load or wake currently parked on `hold_next_load`.
    holding: Option<Arc<tokio::sync::Notify>>,
}

pub struct FakeEngine {
    name: String,
    vram_gb: f64,
    inner: Mutex<Inner>,
}

impl FakeEngine {
    pub fn new(name: &str, vram_gb: f64) -> Self {
        Self {
            name: name.to_string(),
            vram_gb,
            inner: Mutex::new(Inner {
                log: Vec::new(),
                tier: Tier::Cupboard,
                gpu: None,
                holding: None,
            }),
        }
    }

    pub fn fail_next_load(name: &str) {
        fail_next()
            .lock()
            .expect("fail-next mutex")
            .entry(name.to_string())
            .or_default()
            .load += 1;
    }

    /// Make the next `load` or `wake` of `name` wait until the returned
    /// handle is notified. Use `notify_one()` to release it.
    pub fn hold_next_load(name: &str) -> Arc<tokio::sync::Notify> {
        let gate = Arc::new(tokio::sync::Notify::new());
        fail_next()
            .lock()
            .expect("fail-next mutex")
            .entry(name.to_string())
            .or_default()
            .hold_load = Some(Arc::clone(&gate));
        gate
    }

    fn take_hold(name: &str) -> Option<Arc<tokio::sync::Notify>> {
        fail_next()
            .lock()
            .expect("fail-next mutex")
            .get_mut(name)
            .and_then(|f| f.hold_load.take())
    }

    /// Make the next `run` of `name` refuse the request (as a too-long
    /// prompt would) without faulting the engine.
    pub fn reject_next_run(name: &str) {
        fail_next()
            .lock()
            .expect("fail-next mutex")
            .entry(name.to_string())
            .or_default()
            .reject_run += 1;
    }

    pub fn fail_next_run(name: &str) {
        fail_next()
            .lock()
            .expect("fail-next mutex")
            .entry(name.to_string())
            .or_default()
            .run += 1;
    }

    pub fn log(&self) -> Vec<String> {
        self.lock().log.clone()
    }

    pub fn gpu(&self) -> Option<u32> {
        self.lock().gpu
    }

    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.inner.lock().expect("fake engine mutex")
    }

    fn record(&self, op: &str, tier: Tier) {
        let mut inner = self.lock();
        inner.log.push(op.to_string());
        inner.tier = tier;
    }

    /// Park until the test releases the gate, or until `sleep`/`discard`
    /// interrupts the load the way killing a child would.
    async fn hold_if_asked(&self) {
        let Some(gate) = Self::take_hold(&self.name) else {
            return;
        };
        self.lock().holding = Some(Arc::clone(&gate));
        gate.notified().await;
        self.lock().holding = None;
    }

    fn release_held(&self) {
        if let Some(gate) = self.lock().holding.take() {
            gate.notify_one();
        }
    }

    fn tier(&self) -> Tier {
        self.lock().tier
    }
}

#[async_trait]
impl Engine for FakeEngine {
    async fn warm(&self, _path: &str) -> Result<(), EngineError> {
        self.record("warm", Tier::Shelf);
        Ok(())
    }

    async fn load(&self, _path: &str, gpu: u32) -> Result<(), EngineError> {
        if take_fail(&self.name, |f| &mut f.load) {
            return Err(EngineError::Other("load failed".into()));
        }
        self.hold_if_asked().await;
        self.lock().gpu = Some(gpu);
        self.record("load", Tier::Bench);
        Ok(())
    }

    async fn wake(&self, gpu: u32) -> Result<(), EngineError> {
        self.hold_if_asked().await;
        self.lock().gpu = Some(gpu);
        self.record("wake", Tier::Bench);
        Ok(())
    }

    async fn sleep(&self) -> Result<(), EngineError> {
        self.release_held();
        self.record("sleep", Tier::Shelf);
        Ok(())
    }

    async fn discard(&self) -> Result<(), EngineError> {
        self.release_held();
        self.record("discard", Tier::Cupboard);
        Ok(())
    }

    async fn run(
        &self,
        messages: &[ChatMessage],
        prefix: &str,
        _opts: &RunOptions,
        cancel: CancellationToken,
    ) -> Result<mpsc::Receiver<Chunk>, EngineError> {
        if take_fail(&self.name, |f| &mut f.run) {
            return Err(EngineError::Other("run failed".into()));
        }
        if take_fail(&self.name, |f| &mut f.reject_run) {
            return Err(EngineError::Rejected("prompt too long for the fake".into()));
        }
        if self.tier() != Tier::Bench {
            return Err(EngineError::NotLoaded);
        }
        let prompt = last_content(messages).to_string();
        Ok(spawn_chunks(prompt, prefix.to_string(), cancel))
    }

    /// The first allowed value of every field at full confidence, once per
    /// context, in the shape llama-server's `/v1/decision` answers. Enough
    /// to see a whole request go out and a whole reply come back through
    /// the router. `reject_next_run` refuses a decision as it does a chat.
    async fn decide(
        &self,
        body: &serde_json::Value,
        _cancel: CancellationToken,
    ) -> Result<serde_json::Value, EngineError> {
        if take_fail(&self.name, |f| &mut f.reject_run) {
            return Err(EngineError::Rejected("schema too wide for the fake".into()));
        }
        if self.tier() != Tier::Bench {
            return Err(EngineError::NotLoaded);
        }
        Ok(fake_decision(body))
    }

    fn measured_vram_gb(&self) -> f64 {
        self.vram_gb
    }

    fn has_shelf(&self) -> bool {
        true
    }
}

fn fake_decision(body: &serde_json::Value) -> serde_json::Value {
    let schema = body.get("schema").and_then(serde_json::Value::as_object);
    let fields: serde_json::Map<String, serde_json::Value> = schema
        .into_iter()
        .flatten()
        .map(|(name, spec)| {
            let value = first_allowed(spec);
            let field = serde_json::json!({"value": value, "probability": 1.0});
            (name.clone(), field)
        })
        .collect();
    let decision: serde_json::Map<String, serde_json::Value> = fields
        .iter()
        .map(|(name, field)| (name.clone(), field["value"].clone()))
        .collect();
    let contexts = body
        .get("contexts")
        .and_then(serde_json::Value::as_array)
        .map(Vec::len)
        .unwrap_or(1);
    let results: Vec<serde_json::Value> = (0..contexts)
        .map(|_| serde_json::json!({"decision": decision, "fields": fields}))
        .collect();
    serde_json::json!({
        "object": "decision",
        "results": results,
        "usage": {"prompt_tokens": 7, "cached_tokens": 0},
    })
}

/// `true` for a boolean, the minimum for a number, the first choice for an
/// enum, and `null` for a field the fake does not understand.
fn first_allowed(spec: &serde_json::Value) -> serde_json::Value {
    match spec.get("type").and_then(serde_json::Value::as_str) {
        Some("boolean") => serde_json::json!(true),
        Some("integer") | Some("number") => {
            spec.get("minimum").cloned().unwrap_or(serde_json::json!(0))
        }
        _ => spec
            .get("choices")
            .or_else(|| spec.get("enum"))
            .and_then(|c| c.get(0))
            .cloned()
            .unwrap_or(serde_json::Value::Null),
    }
}

fn spawn_chunks(
    prompt: String,
    prefix: String,
    cancel: CancellationToken,
) -> mpsc::Receiver<Chunk> {
    let (tx, rx) = mpsc::channel(2);
    tokio::spawn(async move {
        stream_chunks(tx, prompt, prefix, cancel).await;
    });
    rx
}

async fn stream_chunks(
    tx: mpsc::Sender<Chunk>,
    prompt: String,
    prefix: String,
    cancel: CancellationToken,
) {
    let chunks = [prompt, " world".to_string()];
    let last = chunks.len() - 1;
    let mut seen = String::new();
    for (i, chunk) in chunks.into_iter().enumerate() {
        if cancel.is_cancelled() {
            return;
        }
        let emit = leftover(&mut seen, &prefix, &chunk);
        if !emit.is_empty() && tx.send(Chunk::Token(emit)).await.is_err() {
            return;
        }
        if i == last {
            break;
        }
        tokio::select! {
            _ = cancel.cancelled() => return,
            _ = tokio::time::sleep(Duration::from_millis(80)) => {}
        }
    }
    // A real engine says how it stopped and what it counted; the fake does
    // too, so the scheduler tests cover that path with GB numbers alone.
    let _ = tx
        .send(Chunk::End(RunEnd {
            finish_reason: Some("stop".into()),
            usage: Some(Usage {
                prompt_tokens: 7,
                completion_tokens: 3,
                total_tokens: 10,
            }),
        }))
        .await;
}

fn leftover(seen: &mut String, prefix: &str, chunk: &str) -> String {
    if seen.len() >= prefix.len() {
        seen.push_str(chunk);
        return chunk.to_string();
    }
    let already = prefix.len() - seen.len();
    seen.push_str(chunk);
    if chunk.len() <= already || !chunk.is_char_boundary(already) {
        String::new()
    } else {
        chunk[already..].to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Engine;
    use tokio_util::sync::CancellationToken;

    #[tokio::test]
    async fn fake_decide_answers_every_field_per_context() {
        let e = FakeEngine::new("decider", 1.0);
        e.load("/x", 0).await.unwrap();
        let body = serde_json::json!({
            "schema": {
                "category": {"type": "enum", "choices": ["billing", "technical"]},
                "urgent": {"type": "boolean"},
                "priority": {"type": "integer", "minimum": 1, "maximum": 5}
            },
            "contexts": ["charged twice", "app crashes"]
        });
        let out = e.decide(&body, CancellationToken::new()).await.unwrap();
        assert_eq!(out["object"], "decision");
        let results = out["results"].as_array().unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[1]["decision"]["category"], "billing");
        assert_eq!(results[1]["decision"]["urgent"], true);
        assert_eq!(results[1]["decision"]["priority"], 1);
        assert_eq!(results[0]["fields"]["category"]["probability"], 1.0);
    }

    #[tokio::test]
    async fn fake_decide_needs_the_bench() {
        let e = FakeEngine::new("unloaded-decider", 1.0);
        let err = e
            .decide(&serde_json::json!({}), CancellationToken::new())
            .await
            .unwrap_err();
        assert!(matches!(err, EngineError::NotLoaded));
    }

    #[tokio::test]
    async fn fake_load_sleep_wake_records_order() {
        let e = FakeEngine::new("soap", 28.0);
        e.load("/models/soap.gguf", 0).await.unwrap();
        e.sleep().await.unwrap();
        e.wake(1).await.unwrap();
        assert_eq!(e.log(), vec!["load", "sleep", "wake"]);
        assert_eq!(e.gpu(), Some(1));
        assert_eq!(e.measured_vram_gb(), 28.0);
    }

    #[tokio::test]
    async fn fake_run_streams_two_chunks_then_done() {
        let e = FakeEngine::new("soap", 28.0);
        e.load("/x", 0).await.unwrap();
        let cancel = CancellationToken::new();
        let mut rx = e
            .run(
                &[ChatMessage::user("hello")],
                "",
                &RunOptions::default(),
                cancel,
            )
            .await
            .unwrap();
        let mut got = Vec::new();
        while let Some(c) = rx.recv().await {
            if let Some(t) = c.text() {
                got.push(t.to_string());
            }
        }
        assert_eq!(got, vec!["hello".to_string(), " world".to_string()]);
    }

    #[tokio::test]
    async fn fake_run_stops_on_cancel() {
        let e = FakeEngine::new("soap", 28.0);
        e.load("/x", 0).await.unwrap();
        let cancel = CancellationToken::new();
        cancel.cancel();
        let mut rx = e
            .run(
                &[ChatMessage::user("hello")],
                "",
                &RunOptions::default(),
                cancel,
            )
            .await
            .unwrap();
        assert!(rx.recv().await.is_none());
    }

    #[tokio::test]
    async fn fake_run_skips_prefix_already_streamed() {
        let e = FakeEngine::new("soap", 28.0);
        e.load("/x", 0).await.unwrap();
        let mut rx = e
            .run(
                &[ChatMessage::user("hello")],
                "hello",
                &RunOptions::default(),
                CancellationToken::new(),
            )
            .await
            .unwrap();
        let mut got = Vec::new();
        while let Some(c) = rx.recv().await {
            if let Some(t) = c.text() {
                got.push(t.to_string());
            }
        }
        assert_eq!(got, vec![" world".to_string()]);
    }
}
