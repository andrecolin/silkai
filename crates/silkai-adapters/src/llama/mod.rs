use std::sync::{Arc, Mutex, MutexGuard};

use async_trait::async_trait;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::{ChatMessage, Engine, EngineError, RunOptions};

#[cfg(feature = "llama")]
mod cpp;

#[cfg(not(feature = "llama"))]
const NO_FEATURE: &str = "built without feature llama";

/// Context window when the config does not set `ctx_size`.
pub const DEFAULT_CTX_SIZE: u32 = 4096;

pub struct LlamaEngine {
    vram_gb: f64,
    inner: Arc<Mutex<Inner>>,
}

pub(crate) struct Inner {
    pub on_bench: bool,
    pub path: Option<String>,
    /// Tokens of context per request; prompt plus answer must fit.
    #[cfg_attr(not(feature = "llama"), allow(dead_code))]
    pub ctx_size: u32,
    #[cfg(feature = "llama")]
    pub model: Option<llama_cpp_2::model::LlamaModel>,
}

impl LlamaEngine {
    pub fn new(_name: &str, vram_gb: f64, ctx_size: Option<u32>) -> Self {
        Self {
            vram_gb,
            inner: Arc::new(Mutex::new(Inner {
                on_bench: false,
                path: None,
                ctx_size: ctx_size.unwrap_or(DEFAULT_CTX_SIZE),
                #[cfg(feature = "llama")]
                model: None,
            })),
        }
    }

    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.inner.lock().expect("llama engine mutex")
    }

    fn stored_path(&self) -> Result<String, EngineError> {
        self.lock().path.clone().ok_or(EngineError::NotLoaded)
    }
}

#[async_trait]
impl Engine for LlamaEngine {
    async fn warm(&self, path: &str) -> Result<(), EngineError> {
        place_model(self, path, false, 0).await
    }

    async fn load(&self, path: &str, gpu: u32) -> Result<(), EngineError> {
        place_model(self, path, true, gpu).await
    }

    async fn wake(&self, gpu: u32) -> Result<(), EngineError> {
        let path = self.stored_path()?;
        place_model(self, &path, true, gpu).await
    }

    async fn sleep(&self) -> Result<(), EngineError> {
        let path = self.stored_path()?;
        place_model(self, &path, false, 0).await
    }

    async fn discard(&self) -> Result<(), EngineError> {
        let mut inner = self.lock();
        inner.on_bench = false;
        #[cfg(feature = "llama")]
        {
            inner.model = None;
        }
        Ok(())
    }

    async fn run(
        &self,
        messages: &[ChatMessage],
        prefix: &str,
        opts: &RunOptions,
        cancel: CancellationToken,
    ) -> Result<mpsc::Receiver<String>, EngineError> {
        start_run(self, messages, prefix, opts, cancel)
    }

    fn measured_vram_gb(&self) -> f64 {
        self.vram_gb
    }

    // No `has_shelf`: sleep re-reads the GGUF with zero GPU layers, which
    // mmaps it. The page cache does the caching, not this engine.

    fn pid(&self) -> Option<u32> {
        self.lock().on_bench.then(std::process::id)
    }
}

#[cfg(feature = "llama")]
async fn place_model(
    engine: &LlamaEngine,
    path: &str,
    bench: bool,
    gpu: u32,
) -> Result<(), EngineError> {
    cpp::place(Arc::clone(&engine.inner), path.to_string(), bench, gpu).await
}

#[cfg(not(feature = "llama"))]
async fn place_model(
    engine: &LlamaEngine,
    path: &str,
    bench: bool,
    gpu: u32,
) -> Result<(), EngineError> {
    let _ = (engine, path, gpu);
    if bench {
        refuse_without_feature()
    } else {
        Ok(())
    }
}

#[cfg(feature = "llama")]
fn start_run(
    engine: &LlamaEngine,
    messages: &[ChatMessage],
    prefix: &str,
    opts: &RunOptions,
    cancel: CancellationToken,
) -> Result<mpsc::Receiver<String>, EngineError> {
    cpp::start_run(
        Arc::clone(&engine.inner),
        messages.to_vec(),
        prefix.to_string(),
        opts.clone(),
        cancel,
    )
}

#[cfg(not(feature = "llama"))]
fn start_run(
    engine: &LlamaEngine,
    messages: &[ChatMessage],
    prefix: &str,
    opts: &RunOptions,
    cancel: CancellationToken,
) -> Result<mpsc::Receiver<String>, EngineError> {
    let _ = (engine, messages, prefix, opts, cancel);
    refuse_without_feature()
}

#[cfg(not(feature = "llama"))]
fn refuse_without_feature<T>() -> Result<T, EngineError> {
    Err(EngineError::Other(NO_FEATURE.into()))
}

#[cfg(all(test, feature = "llama"))]
mod llama_feature_tests {
    use super::*;
    use crate::Engine;

    #[tokio::test]
    async fn llama_rejects_missing_file() {
        let e = LlamaEngine::new("soap", 1.0, None);
        let err = e.load("/no/such/model.gguf", 0).await.unwrap_err();
        assert!(matches!(err, EngineError::Other(_)));
    }
}

#[cfg(all(test, not(feature = "llama")))]
mod llama_stub_tests {
    use super::*;
    use crate::Engine;
    use tokio_util::sync::CancellationToken;

    #[tokio::test]
    async fn load_fails_without_feature() {
        let e = LlamaEngine::new("soap", 1.0, None);
        let err = e.load("/no/such/model.gguf", 1).await.unwrap_err();
        match err {
            EngineError::Other(msg) => assert!(msg.contains("llama")),
            other => panic!("expected Other, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn run_fails_without_feature() {
        let e = LlamaEngine::new("soap", 1.0, None);
        let err = e
            .run(
                &[ChatMessage::user("hello")],
                "",
                &RunOptions::default(),
                CancellationToken::new(),
            )
            .await
            .unwrap_err();
        match err {
            EngineError::Other(msg) => assert!(msg.contains("llama")),
            other => panic!("expected Other, got {other:?}"),
        }
    }
}
