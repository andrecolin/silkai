//! A child process an engine starts and stops: llama-server, vLLM,
//! sd-server. Shared by the engines that manage one, so they agree on how
//! the child is pinned to a card, how readiness is polled, and how the
//! whole process group is killed.

use std::process::Stdio;
use std::sync::Mutex;
use std::time::Duration;

use tokio::process::{Child, Command};

use crate::EngineError;

const READY_POLL: Duration = Duration::from_millis(100);
/// A 20-plus GB GGUF read from disk and pushed to the card can take a few
/// minutes cold; llama-server answers `/health` 503 the whole time.
const READY_TIMEOUT: Duration = Duration::from_secs(300);

pub(crate) struct ManagedChild {
    cmd: Vec<String>,
    child: Mutex<Option<Child>>,
}

impl ManagedChild {
    pub(crate) fn new(cmd: Vec<String>) -> Self {
        Self {
            cmd,
            child: Mutex::new(None),
        }
    }

    pub(crate) fn alive(&self) -> bool {
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

    pub(crate) fn id(&self) -> Option<u32> {
        self.child
            .lock()
            .expect("process child mutex")
            .as_ref()
            .and_then(|c| c.id())
    }

    pub(crate) fn spawn(&self, gpu: u32) -> Result<(), EngineError> {
        if self.alive() {
            return Ok(());
        }
        let prog = self
            .cmd
            .first()
            .ok_or_else(|| EngineError::Other("process engine missing cmd".into()))?;
        let mut command = Command::new(prog);
        // The child inherits stderr so its load and error logs land in the
        // daemon's journal. CUDA_VISIBLE_DEVICES pins it to the bench the
        // scheduler chose; other backends ignore the variable.
        command
            .args(&self.cmd[1..])
            .env("CUDA_VISIBLE_DEVICES", gpu.to_string())
            .kill_on_drop(true)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::inherit());
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

    /// Spawn, then poll `ready_url` until it answers 200. A child that
    /// never gets there is killed so it does not hold the card.
    pub(crate) async fn spawn_ready(&self, gpu: u32, ready_url: &str) -> Result<(), EngineError> {
        self.spawn(gpu)?;
        if let Err(err) = self.wait_ready(ready_url).await {
            let _ = self.kill().await;
            return Err(err);
        }
        Ok(())
    }

    /// Poll `ready_url` until it answers 200. llama-server returns 503 while
    /// the model is still loading, so 503 and connection refused both mean
    /// "not yet"; any other status means the child is up but wrong.
    async fn wait_ready(&self, ready_url: &str) -> Result<(), EngineError> {
        let client = reqwest::Client::new();
        let deadline = tokio::time::Instant::now() + READY_TIMEOUT;
        loop {
            let not_yet = match client.get(ready_url).send().await {
                Ok(resp) if resp.status().is_success() => return Ok(()),
                Ok(resp) if resp.status() == reqwest::StatusCode::SERVICE_UNAVAILABLE => {
                    format!("{ready_url} 503")
                }
                Ok(resp) => {
                    return Err(EngineError::Other(format!(
                        "process {ready_url} {}",
                        resp.status()
                    )));
                }
                Err(err) if err.is_connect() => err.to_string(),
                Err(err) => return Err(EngineError::Other(err.to_string())),
            };
            if !self.alive() {
                return Err(EngineError::Other("process exited before ready".into()));
            }
            if tokio::time::Instant::now() >= deadline {
                return Err(EngineError::Other(format!("process not ready: {not_yet}")));
            }
            tokio::time::sleep(READY_POLL).await;
        }
    }

    pub(crate) async fn kill(&self) -> Result<(), EngineError> {
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
