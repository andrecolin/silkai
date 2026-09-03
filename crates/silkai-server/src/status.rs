//! The status the daemon reports: the scheduler's view plus what the runtime
//! knows on top of it (a load in flight, open sessions, what the card
//! measures). `silkai-sched` stays pure; everything here is an overlay.

use serde::Serialize;
use silkai_sched::{Priority, StatusSnapshot, Tier};

use crate::sampler::Sample;

#[derive(Debug, Clone, Serialize)]
pub struct Status {
    pub models: Vec<ModelStatus>,
    /// Sum of `budget_gb` for models on a card.
    pub gpu_used_gb: f64,
    /// Sum of `ram_gb` for shelved models whose engine really holds a copy.
    pub ram_used_gb: f64,
    pub gpus: Vec<GpuStatus>,
    /// Whether `measured_*` fields are being filled (nvidia-smi found).
    pub measured: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelStatus {
    pub name: String,
    /// The scheduler's tier. Kept for compatibility; `state` is finer.
    pub tier: Tier,
    /// `cupboard`, `shelf`, `loading`, `bench`, or `sleeping`.
    pub state: String,
    pub engine: String,
    pub budget_gb: f64,
    pub measured_gb: Option<f64>,
    pub priority: String,
    pub exclusive: bool,
    pub slots: u32,
    pub running: u32,
    pub queued: u32,
    pub gpu: Option<u32>,
    pub sessions: u32,
}

#[derive(Debug, Clone, Serialize)]
pub struct GpuStatus {
    pub id: u32,
    pub used_gb: f64,
    pub schedulable_gb: f64,
    pub measured_used_gb: Option<f64>,
    pub total_gb: Option<f64>,
}

/// What the runtime knows about one model beyond the scheduler.
pub(crate) struct ModelFacts {
    pub engine: String,
    pub budget_gb: f64,
    pub ram_gb: f64,
    pub priority: Priority,
    pub exclusive: bool,
    pub slots: u32,
    pub overlay: Option<&'static str>,
    pub sessions: u32,
    pub has_shelf: bool,
    pub pid: Option<u32>,
}

pub(crate) fn assemble(
    snap: &StatusSnapshot,
    facts: impl Fn(&str) -> Option<ModelFacts>,
    sample: Option<&Sample>,
) -> Status {
    let mut ram_used_gb = 0.0;
    let models = snap
        .models
        .iter()
        .map(|m| {
            let f = facts(&m.name);
            if m.tier == Tier::Shelf && f.as_ref().is_some_and(|f| f.has_shelf) {
                ram_used_gb += f.as_ref().map(|f| f.ram_gb).unwrap_or(0.0);
            }
            model_status(m, f, sample)
        })
        .collect();
    Status {
        models,
        gpu_used_gb: clean(snap.gpu_used_gb),
        ram_used_gb: clean(ram_used_gb),
        gpus: snap.gpus.iter().map(|g| gpu_status(g, sample)).collect(),
        measured: sample.is_some(),
    }
}

fn model_status(
    m: &silkai_sched::ModelStatus,
    facts: Option<ModelFacts>,
    sample: Option<&Sample>,
) -> ModelStatus {
    let f = facts.unwrap_or_else(unknown_facts);
    ModelStatus {
        name: m.name.clone(),
        tier: m.tier,
        state: f.overlay.unwrap_or(tier_name(m.tier)).to_string(),
        engine: f.engine,
        budget_gb: f.budget_gb,
        measured_gb: f.pid.and_then(|pid| sample.and_then(|s| s.used_by(pid))),
        priority: priority_name(f.priority).to_string(),
        exclusive: f.exclusive,
        slots: f.slots,
        running: m.running,
        queued: m.queued,
        gpu: m.gpu,
        sessions: f.sessions,
    }
}

fn gpu_status(g: &silkai_sched::GpuStatus, sample: Option<&Sample>) -> GpuStatus {
    let measured = sample.and_then(|s| s.gpu(g.id));
    GpuStatus {
        id: g.id,
        used_gb: clean(g.used_gb),
        schedulable_gb: g.schedulable_gb,
        measured_used_gb: measured.map(|m| m.used_gb),
        total_gb: measured.map(|m| m.total_gb),
    }
}

fn unknown_facts() -> ModelFacts {
    ModelFacts {
        engine: "unknown".into(),
        budget_gb: 0.0,
        ram_gb: 0.0,
        priority: Priority::Normal,
        exclusive: false,
        slots: 0,
        overlay: None,
        sessions: 0,
        has_shelf: false,
        pid: None,
    }
}

pub(crate) fn tier_name(tier: Tier) -> &'static str {
    match tier {
        Tier::Cupboard => "cupboard",
        Tier::Shelf => "shelf",
        Tier::Bench => "bench",
    }
}

pub(crate) fn priority_name(p: Priority) -> &'static str {
    match p {
        Priority::Live => "live",
        Priority::Normal => "normal",
        Priority::Background => "background",
    }
}

/// `-0.0` prints as `-0.0` in JSON and looks like a bug. It is not one.
fn clean(v: f64) -> f64 {
    if v == 0.0 {
        0.0
    } else {
        v
    }
}
