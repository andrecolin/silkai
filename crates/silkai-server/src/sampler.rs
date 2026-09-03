//! What the card measures. A background task runs `nvidia-smi` every two
//! seconds and keeps the last answer; requests read the cached sample and
//! never wait on the binary. Boxes without nvidia-smi simply report nothing.

use std::sync::{Arc, Mutex};
use std::time::Duration;

const INTERVAL: Duration = Duration::from_secs(2);
const MIB: f64 = 1024.0;

#[derive(Debug, Clone, PartialEq)]
pub struct GpuSample {
    pub id: u32,
    pub used_gb: f64,
    pub total_gb: f64,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Sample {
    pub gpus: Vec<GpuSample>,
    /// (pid, GB) for every compute process the driver lists.
    pub apps: Vec<(u32, f64)>,
}

impl Sample {
    pub fn gpu(&self, id: u32) -> Option<&GpuSample> {
        self.gpus.iter().find(|g| g.id == id)
    }

    pub fn used_by(&self, pid: u32) -> Option<f64> {
        let total: f64 = self
            .apps
            .iter()
            .filter(|(p, _)| *p == pid)
            .map(|(_, gb)| gb)
            .sum();
        (total > 0.0).then_some(total)
    }
}

pub type Shared = Arc<Mutex<Option<Sample>>>;

/// Parse `--query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits`.
pub fn parse_gpus(text: &str) -> Vec<GpuSample> {
    text.lines()
        .filter_map(|line| {
            let mut parts = line.split(',').map(str::trim);
            let id = parts.next()?.parse().ok()?;
            let used: f64 = parts.next()?.parse().ok()?;
            let total: f64 = parts.next()?.parse().ok()?;
            Some(GpuSample {
                id,
                used_gb: used / MIB,
                total_gb: total / MIB,
            })
        })
        .collect()
}

/// Parse `--query-compute-apps=pid,used_memory --format=csv,noheader,nounits`.
pub fn parse_apps(text: &str) -> Vec<(u32, f64)> {
    text.lines()
        .filter_map(|line| {
            let mut parts = line.split(',').map(str::trim);
            let pid = parts.next()?.parse().ok()?;
            let used: f64 = parts.next()?.parse().ok()?;
            Some((pid, used / MIB))
        })
        .collect()
}

/// Start sampling if `nvidia-smi` is on PATH. Returns the shared slot either
/// way; it stays `None` when there is nothing to sample.
pub fn start() -> Shared {
    let shared: Shared = Arc::new(Mutex::new(None));
    if !nvidia_smi_present() {
        return shared;
    }
    let slot = Arc::clone(&shared);
    tokio::spawn(async move {
        loop {
            if let Some(sample) = sample_once().await {
                *slot.lock().expect("sampler mutex") = Some(sample);
            }
            tokio::time::sleep(INTERVAL).await;
        }
    });
    shared
}

fn nvidia_smi_present() -> bool {
    std::process::Command::new("nvidia-smi")
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

async fn sample_once() -> Option<Sample> {
    let gpus = run(&[
        "--query-gpu=index,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ])
    .await?;
    let apps = run(&[
        "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader,nounits",
    ])
    .await
    .unwrap_or_default();
    Some(Sample {
        gpus: parse_gpus(&gpus),
        apps: parse_apps(&apps),
    })
}

async fn run(args: &[&str]) -> Option<String> {
    let out = tokio::process::Command::new("nvidia-smi")
        .args(args)
        .output()
        .await
        .ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_gpu_lines() {
        let got = parse_gpus("0, 22667, 32768\n1, 0, 32768\n");
        assert_eq!(got.len(), 2);
        assert!((got[0].used_gb - 22.14).abs() < 0.01);
        assert_eq!(got[1].total_gb, 32.0);
    }

    #[test]
    fn parses_apps_and_sums_per_pid() {
        let s = Sample {
            gpus: vec![],
            apps: parse_apps("4375, 4\n488277, 27638\n488277, 1024\n"),
        };
        assert_eq!(s.used_by(4375), Some(4.0 / 1024.0));
        assert!((s.used_by(488277).unwrap() - 28.0).abs() < 0.01);
        assert_eq!(s.used_by(1), None);
    }

    #[test]
    fn ignores_garbage() {
        assert!(parse_gpus("No devices were found\n").is_empty());
        assert!(parse_apps("").is_empty());
    }
}
