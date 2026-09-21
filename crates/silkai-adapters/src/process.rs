use async_trait::async_trait;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::child::ManagedChild;
use crate::vllm::VllmEngine;
use crate::{ChatMessage, Chunk, Engine, EngineError, RunOptions};

pub struct ProcessEngine {
    http: VllmEngine,
    url: String,
    child: ManagedChild,
}

impl ProcessEngine {
    pub fn new(name: &str, vram_gb: f64, url: impl AsRef<str>, cmd: Vec<String>) -> Self {
        let url = url.as_ref().trim_end_matches('/').to_string();
        Self {
            http: VllmEngine::new(name, vram_gb, &url),
            url,
            child: ManagedChild::new(cmd),
        }
    }

    pub fn alive(&self) -> bool {
        self.child.alive()
    }

    pub fn child_id(&self) -> Option<u32> {
        self.child.id()
    }

    /// Spawn and poll `GET /health` until the child answers 200. vLLM and
    /// llama-server both expose it.
    async fn spawn_ready(&self, gpu: u32) -> Result<(), EngineError> {
        let ready = format!("{}/health", self.url);
        self.child.spawn_ready(gpu, &ready).await
    }
}

#[async_trait]
impl Engine for ProcessEngine {
    async fn warm(&self, path: &str) -> Result<(), EngineError> {
        self.http.warm(path).await
    }

    async fn load(&self, path: &str, gpu: u32) -> Result<(), EngineError> {
        self.http.warm(path).await?;
        self.wake(gpu).await
    }

    /// A freshly spawned child starts awake, so there is no `/wake_up` here:
    /// once `/health` is green the model is on the bench.
    async fn wake(&self, gpu: u32) -> Result<(), EngineError> {
        self.spawn_ready(gpu).await?;
        self.http.mark_on_bench(gpu);
        Ok(())
    }

    async fn sleep(&self) -> Result<(), EngineError> {
        self.child.kill().await?;
        self.http.discard().await
    }

    async fn discard(&self) -> Result<(), EngineError> {
        self.child.kill().await?;
        self.http.discard().await
    }

    async fn run(
        &self,
        messages: &[ChatMessage],
        prefix: &str,
        opts: &RunOptions,
        cancel: CancellationToken,
    ) -> Result<mpsc::Receiver<Chunk>, EngineError> {
        self.http.run(messages, prefix, opts, cancel).await
    }

    fn measured_vram_gb(&self) -> f64 {
        self.http.measured_vram_gb()
    }

    fn pid(&self) -> Option<u32> {
        self.child_id()
    }
}
