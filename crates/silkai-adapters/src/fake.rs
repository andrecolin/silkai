use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::time::Duration;

use async_trait::async_trait;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::{last_content, ChatMessage, Engine, EngineError, RunOptions};

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
        if let Some(gate) = Self::take_hold(&self.name) {
            gate.notified().await;
        }
        self.lock().gpu = Some(gpu);
        self.record("load", Tier::Bench);
        Ok(())
    }

    async fn wake(&self, gpu: u32) -> Result<(), EngineError> {
        if let Some(gate) = Self::take_hold(&self.name) {
            gate.notified().await;
        }
        self.lock().gpu = Some(gpu);
        self.record("wake", Tier::Bench);
        Ok(())
    }

    async fn sleep(&self) -> Result<(), EngineError> {
        self.record("sleep", Tier::Shelf);
        Ok(())
    }

    async fn discard(&self) -> Result<(), EngineError> {
        self.record("discard", Tier::Cupboard);
        Ok(())
    }

    async fn run(
        &self,
        messages: &[ChatMessage],
        prefix: &str,
        _opts: &RunOptions,
        cancel: CancellationToken,
    ) -> Result<mpsc::Receiver<String>, EngineError> {
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

    fn measured_vram_gb(&self) -> f64 {
        self.vram_gb
    }

    fn has_shelf(&self) -> bool {
        true
    }
}

fn spawn_chunks(
    prompt: String,
    prefix: String,
    cancel: CancellationToken,
) -> mpsc::Receiver<String> {
    let (tx, rx) = mpsc::channel(2);
    tokio::spawn(async move {
        stream_chunks(tx, prompt, prefix, cancel).await;
    });
    rx
}

async fn stream_chunks(
    tx: mpsc::Sender<String>,
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
        if !emit.is_empty() && tx.send(emit).await.is_err() {
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
        while let Some(t) = rx.recv().await {
            got.push(t);
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
        while let Some(t) = rx.recv().await {
            got.push(t);
        }
        assert_eq!(got, vec![" world".to_string()]);
    }
}
