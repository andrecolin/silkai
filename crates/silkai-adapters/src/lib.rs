mod fake;
pub use fake::FakeEngine;

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
    async fn load(&self, path: &str) -> Result<(), EngineError>;
    async fn wake(&self) -> Result<(), EngineError>;
    async fn sleep(&self) -> Result<(), EngineError>;
    async fn discard(&self) -> Result<(), EngineError>;
    async fn run(
        &self,
        prompt: &str,
        cancel: CancellationToken,
    ) -> Result<mpsc::Receiver<String>, EngineError>;
    fn measured_vram_gb(&self) -> f64;
}
