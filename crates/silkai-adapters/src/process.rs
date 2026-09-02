use std::process::Stdio;
use std::sync::Mutex;

use async_trait::async_trait;
use tokio::process::{Child, Command};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::vllm::VllmEngine;
use crate::{Engine, EngineError};

pub struct ProcessEngine {
    http: VllmEngine,
    cmd: Vec<String>,
    child: Mutex<Option<Child>>,
}

impl ProcessEngine {
    pub fn new(name: &str, vram_gb: f64, url: impl AsRef<str>, cmd: Vec<String>) -> Self {
        Self {
            http: VllmEngine::new(name, vram_gb, url),
            cmd,
            child: Mutex::new(None),
        }
    }

    pub fn alive(&self) -> bool {
        let mut slot = self.child.lock().expect("process child mutex");
        match slot.as_mut() {
            Some(child) => match child.try_wait() {
                Ok(None) => true,
                Ok(Some(_)) => {
                    *slot = None;
                    false
                }
                Err(_) => false,
            },
            None => false,
        }
    }

    async fn spawn(&self) -> Result<(), EngineError> {
        if self.alive() {
            return Ok(());
        }
        let prog = self
            .cmd
            .first()
            .ok_or_else(|| EngineError::Other("process engine missing cmd".into()))?;
        let child = Command::new(prog)
            .args(&self.cmd[1..])
            .kill_on_drop(true)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| EngineError::Other(e.to_string()))?;
        *self.child.lock().expect("process child mutex") = Some(child);
        Ok(())
    }

    async fn kill(&self) -> Result<(), EngineError> {
        let child = self.child.lock().expect("process child mutex").take();
        if let Some(mut child) = child {
            let _ = child.kill().await;
            let _ = child.wait().await;
        }
        Ok(())
    }
}

#[async_trait]
impl Engine for ProcessEngine {
    async fn warm(&self, path: &str) -> Result<(), EngineError> {
        self.http.warm(path).await
    }

    async fn load(&self, path: &str, gpu: u32) -> Result<(), EngineError> {
        self.spawn().await?;
        match self.http.load(path, gpu).await {
            Ok(()) => Ok(()),
            Err(err) => {
                let _ = self.kill().await;
                Err(err)
            }
        }
    }

    async fn wake(&self, gpu: u32) -> Result<(), EngineError> {
        self.spawn().await?;
        match self.http.wake(gpu).await {
            Ok(()) => Ok(()),
            Err(err) => {
                let _ = self.kill().await;
                Err(err)
            }
        }
    }

    async fn sleep(&self) -> Result<(), EngineError> {
        self.kill().await?;
        self.http.discard().await
    }

    async fn discard(&self) -> Result<(), EngineError> {
        self.kill().await?;
        self.http.discard().await
    }

    async fn run(
        &self,
        prompt: &str,
        prefix: &str,
        cancel: CancellationToken,
    ) -> Result<mpsc::Receiver<String>, EngineError> {
        self.http.run(prompt, prefix, cancel).await
    }

    fn measured_vram_gb(&self) -> f64 {
        self.http.measured_vram_gb()
    }
}
