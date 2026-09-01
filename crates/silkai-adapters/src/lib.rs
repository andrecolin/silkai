mod fake;
mod llama;
mod ollama;
mod vllm;
pub use fake::FakeEngine;
pub use llama::LlamaEngine;
pub use ollama::OllamaEngine;
pub use vllm::VllmEngine;

use async_trait::async_trait;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("not loaded")]
    NotLoaded,
    #[error("{0}")]
    Other(String),
}

#[async_trait]
pub trait Engine: Send + Sync {
    async fn warm(&self, path: &str) -> Result<(), EngineError>;
    async fn load(&self, path: &str, gpu: u32) -> Result<(), EngineError>;
    async fn wake(&self, gpu: u32) -> Result<(), EngineError>;
    async fn sleep(&self) -> Result<(), EngineError>;
    async fn discard(&self) -> Result<(), EngineError>;
    async fn run(
        &self,
        prompt: &str,
        prefix: &str,
        cancel: CancellationToken,
    ) -> Result<mpsc::Receiver<String>, EngineError>;
    fn measured_vram_gb(&self) -> f64;
}
