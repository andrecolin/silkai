mod fake;
mod llama;
mod ollama;
mod process;
mod vllm;
pub use fake::FakeEngine;
pub use llama::LlamaEngine;
pub use ollama::OllamaEngine;
pub use process::ProcessEngine;
pub use vllm::VllmEngine;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

/// One turn of an OpenAI-style chat: `system`, `user`, `assistant`, or
/// whatever role the engine's template understands. The whole list reaches
/// the engine; SilkAI never collapses it to a single string.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self::new("system", content)
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::new("user", content)
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new("assistant", content)
    }
}

/// The text a plain completion engine sees: the last message's content.
/// Used by engines that have no chat template of their own.
pub fn last_content(messages: &[ChatMessage]) -> &str {
    messages.last().map(|m| m.content.as_str()).unwrap_or("")
}

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
    /// Generate for `messages`. `prefix` is text already streamed to the
    /// client by an earlier, preempted run; the engine continues after it
    /// and must not emit it again.
    async fn run(
        &self,
        messages: &[ChatMessage],
        prefix: &str,
        cancel: CancellationToken,
    ) -> Result<mpsc::Receiver<String>, EngineError>;
    fn measured_vram_gb(&self) -> f64;

    /// Whether `sleep` keeps a copy in host RAM that `wake` restores without
    /// touching disk. Engines that kill a child or re-read the file say no,
    /// so status does not report RAM that is not held.
    fn has_shelf(&self) -> bool {
        false
    }

    /// The OS process holding this model's VRAM, if any, so the sampler can
    /// attribute what the card measures.
    fn pid(&self) -> Option<u32> {
        None
    }
}
