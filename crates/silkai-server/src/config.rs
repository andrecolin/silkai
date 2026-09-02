use std::collections::HashMap;
use std::path::Path;
use std::time::Duration;

use serde::Deserialize;
use silkai_sched::{GpuBudget, ModelSpec, Priority, Resources};

#[derive(Debug, Clone, PartialEq)]
pub struct AppConfig {
    pub listen: String,
    pub prefetch_on_start: bool,
    pub request_timeout_secs: u64,
    pub request_timeout: Duration,
    pub resources: Resources,
    pub enabled: Vec<ConfiguredModel>,
    pub disabled: Vec<ConfiguredModel>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConfiguredModel {
    pub spec: ModelSpec,
    pub engine: String,
    pub path: String,
    pub url: Option<String>,
    pub transport: String,
    pub idle_timeout_secs: Option<u64>,
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error(transparent)]
    Toml(#[from] toml::de::Error),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("{0}")]
    Invalid(String),
}

#[derive(Deserialize)]
struct FileConfig {
    #[serde(default = "default_listen")]
    listen: String,
    resources: FileResources,
    #[serde(default)]
    models: HashMap<String, FileModel>,
}

#[derive(Deserialize)]
struct FileResources {
    gpu_total_gb: Option<f64>,
    #[serde(default)]
    gpu_headroom_gb: f64,
    ram_total_gb: f64,
    #[serde(default)]
    ram_headroom_gb: f64,
    #[serde(default = "default_prefetch")]
    prefetch_on_start: bool,
    #[serde(default = "default_timeout")]
    request_timeout_secs: u64,
    #[serde(default)]
    gpus: Vec<FileGpu>,
}

#[derive(Deserialize)]
struct FileGpu {
    id: u32,
    total_gb: f64,
    #[serde(default)]
    headroom_gb: f64,
}

#[derive(Deserialize)]
struct FileModel {
    engine: String,
    path: String,
    vram_gb: f64,
    ram_gb: Option<f64>,
    priority: String,
    #[serde(default)]
    exclusive: bool,
    #[serde(default = "default_slots")]
    slots: u32,
    #[serde(default = "default_keep_warm")]
    keep_warm: bool,
    #[serde(default = "default_transport")]
    transport: String,
    idle_timeout_secs: Option<u64>,
    #[serde(default)]
    gpu: Option<u32>,
    #[serde(default)]
    url: Option<String>,
}

pub fn load_from_str(s: &str) -> Result<AppConfig, ConfigError> {
    load(s, None)
}

pub fn load_from_str_probed(s: &str, probed: Vec<(u32, f64)>) -> Result<AppConfig, ConfigError> {
    load(s, Some(probed))
}

pub fn load_from_path(path: impl AsRef<Path>) -> Result<AppConfig, ConfigError> {
    load(&std::fs::read_to_string(path)?, probe_nvidia())
}

fn load(s: &str, probed: Option<Vec<(u32, f64)>>) -> Result<AppConfig, ConfigError> {
    let file: FileConfig = toml::from_str(s)?;
    app_config(file, probed)
}

fn app_config(file: FileConfig, probed: Option<Vec<(u32, f64)>>) -> Result<AppConfig, ConfigError> {
    let resources = sched_resources(&file.resources, probed)?;
    let (enabled, disabled) = split_models(file.models, resources.max_schedulable())?;
    Ok(AppConfig {
        listen: file.listen,
        prefetch_on_start: file.resources.prefetch_on_start,
        request_timeout_secs: file.resources.request_timeout_secs,
        request_timeout: Duration::from_secs(file.resources.request_timeout_secs),
        resources,
        enabled,
        disabled,
    })
}

fn sched_resources(
    r: &FileResources,
    probed: Option<Vec<(u32, f64)>>,
) -> Result<Resources, ConfigError> {
    let ram_shelf_gb = r.ram_total_gb - r.ram_headroom_gb;
    if !r.gpus.is_empty() {
        return Ok(resources_from_file_gpus(&r.gpus, ram_shelf_gb));
    }
    if let Some(total) = r.gpu_total_gb {
        return Ok(Resources::single(total - r.gpu_headroom_gb, ram_shelf_gb));
    }
    if let Some(probed) = probed.filter(|g| !g.is_empty()) {
        return Ok(resources_from_probe(
            probed,
            r.gpu_headroom_gb,
            ram_shelf_gb,
        ));
    }
    Err(ConfigError::Invalid(
        "resources.gpu_total_gb or resources.gpus is required (GPU probe found nothing)".into(),
    ))
}

fn resources_from_file_gpus(gpus: &[FileGpu], ram_shelf_gb: f64) -> Resources {
    let gpus: Vec<GpuBudget> = gpus
        .iter()
        .map(|g| GpuBudget {
            id: g.id,
            schedulable_gb: g.total_gb - g.headroom_gb,
        })
        .collect();
    resources_with_gpus(gpus, ram_shelf_gb)
}

fn resources_from_probe(probed: Vec<(u32, f64)>, headroom_gb: f64, ram_shelf_gb: f64) -> Resources {
    let gpus: Vec<GpuBudget> = probed
        .into_iter()
        .map(|(id, total_gb)| GpuBudget {
            id,
            schedulable_gb: total_gb - headroom_gb,
        })
        .collect();
    resources_with_gpus(gpus, ram_shelf_gb)
}

fn resources_with_gpus(gpus: Vec<GpuBudget>, ram_shelf_gb: f64) -> Resources {
    let gpu_schedulable_gb = gpus.iter().map(|g| g.schedulable_gb).fold(0.0, f64::max);
    Resources {
        gpu_schedulable_gb,
        ram_shelf_gb,
        gpus,
    }
}

pub fn parse_nvidia_smi(text: &str) -> Result<Vec<(u32, f64)>, ConfigError> {
    let mut gpus = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut parts = line.split(',');
        let id = parts
            .next()
            .and_then(|s| s.trim().parse().ok())
            .ok_or_else(|| ConfigError::Invalid(format!("bad nvidia-smi line: {line}")))?;
        let mem = parts
            .next()
            .and_then(|s| s.split_whitespace().next())
            .and_then(|s| s.parse::<f64>().ok())
            .ok_or_else(|| ConfigError::Invalid(format!("bad nvidia-smi line: {line}")))?;
        gpus.push((id, mem / 1024.0));
    }
    if gpus.is_empty() {
        return Err(ConfigError::Invalid("nvidia-smi listed no GPUs".into()));
    }
    Ok(gpus)
}

fn probe_nvidia() -> Option<Vec<(u32, f64)>> {
    let out = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    parse_nvidia_smi(&String::from_utf8_lossy(&out.stdout)).ok()
}

fn split_models(
    models: HashMap<String, FileModel>,
    gpu_schedulable_gb: f64,
) -> Result<(Vec<ConfiguredModel>, Vec<ConfiguredModel>), ConfigError> {
    let mut enabled = Vec::new();
    let mut disabled = Vec::new();
    for (name, model) in models {
        let configured = configured_model(name, model)?;
        if configured.spec.vram_gb > gpu_schedulable_gb {
            disabled.push(configured);
        } else {
            enabled.push(configured);
        }
    }
    Ok((enabled, disabled))
}

fn configured_model(name: String, m: FileModel) -> Result<ConfiguredModel, ConfigError> {
    Ok(ConfiguredModel {
        spec: ModelSpec {
            name,
            vram_gb: m.vram_gb,
            ram_gb: m.ram_gb.unwrap_or(m.vram_gb),
            priority: parse_priority(&m.priority)?,
            exclusive: m.exclusive,
            slots: m.slots,
            keep_warm: m.keep_warm,
            gpu: m.gpu,
        },
        engine: m.engine,
        path: m.path,
        url: m.url,
        transport: m.transport,
        idle_timeout_secs: m.idle_timeout_secs,
    })
}

fn parse_priority(s: &str) -> Result<Priority, ConfigError> {
    match s.to_ascii_lowercase().as_str() {
        "live" => Ok(Priority::Live),
        "normal" => Ok(Priority::Normal),
        "background" => Ok(Priority::Background),
        other => Err(ConfigError::Invalid(format!("unknown priority: {other}"))),
    }
}

fn default_listen() -> String {
    "127.0.0.1:8080".into()
}

fn default_prefetch() -> bool {
    true
}

fn default_timeout() -> u64 {
    600
}

fn default_slots() -> u32 {
    1
}

fn default_keep_warm() -> bool {
    true
}

fn default_transport() -> String {
    "http".into()
}
