use std::sync::{Mutex, MutexGuard};
use std::time::Duration;

use async_trait::async_trait;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::{Engine, EngineError};

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
    vram_gb: f64,
    inner: Mutex<Inner>,
}

impl FakeEngine {
    pub fn new(_name: &str, vram_gb: f64) -> Self {
        Self {
            vram_gb,
            inner: Mutex::new(Inner {
                log: Vec::new(),
                tier: Tier::Cupboard,
                gpu: None,
            }),
        }
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
        self.lock().gpu = Some(gpu);
        self.record("load", Tier::Bench);
        Ok(())
    }

    async fn wake(&self, gpu: u32) -> Result<(), EngineError> {
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
        prompt: &str,
        cancel: CancellationToken,
    ) -> Result<mpsc::Receiver<String>, EngineError> {
        if self.tier() != Tier::Bench {
            return Err(EngineError::NotLoaded);
        }
        Ok(spawn_chunks(prompt.to_string(), cancel))
    }

    fn measured_vram_gb(&self) -> f64 {
        self.vram_gb
    }
}

fn spawn_chunks(prompt: String, cancel: CancellationToken) -> mpsc::Receiver<String> {
    let (tx, rx) = mpsc::channel(2);
    tokio::spawn(async move {
        stream_chunks(tx, prompt, cancel).await;
    });
    rx
}

async fn stream_chunks(tx: mpsc::Sender<String>, prompt: String, cancel: CancellationToken) {
    if cancel.is_cancelled() {
        return;
    }
    if tx.send(prompt).await.is_err() {
        return;
    }
    tokio::select! {
        _ = cancel.cancelled() => return,
        _ = tokio::time::sleep(Duration::from_millis(80)) => {}
    }
    if cancel.is_cancelled() {
        return;
    }
    let _ = tx.send(" world".to_string()).await;
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
        let mut rx = e.run("hello", cancel).await.unwrap();
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
        let mut rx = e.run("hello", cancel).await.unwrap();
        assert!(rx.recv().await.is_none());
    }
}
