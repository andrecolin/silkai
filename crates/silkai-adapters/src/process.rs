use std::process::Stdio;
use std::sync::Mutex;
use std::time::Duration;

use async_trait::async_trait;
use tokio::process::{Child, Command};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::vllm::VllmEngine;
use crate::{Engine, EngineError};

const READY_POLL: Duration = Duration::from_millis(50);
const READY_TIMEOUT: Duration = Duration::from_secs(60);

pub struct ProcessEngine {
    http: VllmEngine,
    url: String,
    cmd: Vec<String>,
    child: Mutex<Option<Child>>,
}

impl ProcessEngine {
    pub fn new(name: &str, vram_gb: f64, url: impl AsRef<str>, cmd: Vec<String>) -> Self {
        let url = url.as_ref().trim_end_matches('/').to_string();
        Self {
            http: VllmEngine::new(name, vram_gb, &url),
            url,
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

    pub fn child_id(&self) -> Option<u32> {
        self.child
            .lock()
            .expect("process child mutex")
            .as_ref()
            .and_then(|c| c.id())
    }

    async fn spawn(&self) -> Result<(), EngineError> {
        if self.alive() {
            return Ok(());
        }
        let prog = self
            .cmd
            .first()
            .ok_or_else(|| EngineError::Other("process engine missing cmd".into()))?;
        let mut command = Command::new(prog);
        command
            .args(&self.cmd[1..])
            .kill_on_drop(true)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        #[cfg(unix)]
        {
            command.process_group(0);
        }
        let child = command
            .spawn()
            .map_err(|e| EngineError::Other(e.to_string()))?;
        *self.child.lock().expect("process child mutex") = Some(child);
        Ok(())
    }

    async fn spawn_ready(&self) -> Result<(), EngineError> {
        self.spawn().await?;
        if let Err(err) = self.wait_ready().await {
            let _ = self.kill().await;
            return Err(err);
        }
        Ok(())
    }

    async fn wait_ready(&self) -> Result<(), EngineError> {
        let url = format!("{}/wake_up", self.url);
        let client = reqwest::Client::new();
        let deadline = tokio::time::Instant::now() + READY_TIMEOUT;
        loop {
            match client.post(&url).send().await {
                Ok(resp) if resp.status().is_success() => return Ok(()),
                Ok(resp) => {
                    return Err(EngineError::Other(format!("vllm {url} {}", resp.status())));
                }
                Err(err) if err.is_connect() && tokio::time::Instant::now() < deadline => {
                    tokio::time::sleep(READY_POLL).await;
                }
                Err(err) => return Err(EngineError::Other(err.to_string())),
            }
        }
    }

    async fn kill(&self) -> Result<(), EngineError> {
        let child = self.child.lock().expect("process child mutex").take();
        if let Some(mut child) = child {
            if let Some(pid) = child.id() {
                kill_group(pid);
            }
            let _ = child.kill().await;
            let _ = child.wait().await;
        }
        Ok(())
    }
}

/// SIGKILL the child's whole process group, so an engine that forks workers
/// (vLLM does) does not leave them behind holding VRAM.
///
/// `spawn` gives the child its own group with `process_group(0)`, so the
/// child's pid is the group id. This calls `killpg(2)` directly rather than
/// shelling out to a `kill` binary: the BSD and util-linux front ends do not
/// agree on how to spell a negative pid, and the failure was silent.
#[cfg(unix)]
fn kill_group(pid: u32) {
    // Safety: `killpg` takes a group id and a signal, touches no memory, and
    // only fails with ESRCH/EPERM, which we cannot act on here.
    unsafe {
        libc::killpg(pid as libc::pid_t, libc::SIGKILL);
    }
}

#[cfg(not(unix))]
fn kill_group(_pid: u32) {}

#[async_trait]
impl Engine for ProcessEngine {
    async fn warm(&self, path: &str) -> Result<(), EngineError> {
        self.http.warm(path).await
    }

    async fn load(&self, path: &str, gpu: u32) -> Result<(), EngineError> {
        self.spawn_ready().await?;
        match self.http.load(path, gpu).await {
            Ok(()) => Ok(()),
            Err(err) => {
                let _ = self.kill().await;
                Err(err)
            }
        }
    }

    async fn wake(&self, gpu: u32) -> Result<(), EngineError> {
        self.spawn_ready().await?;
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
