#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Priority {
    Background,
    Normal,
    Live,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Tier {
    Cupboard,
    Shelf,
    Bench,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct JobId(pub u64);

#[derive(Debug, Clone, PartialEq)]
pub struct GpuBudget {
    pub id: u32,
    pub schedulable_gb: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Resources {
    pub gpu_schedulable_gb: f64,
    pub ram_shelf_gb: f64,
    pub gpus: Vec<GpuBudget>,
}

impl Resources {
    pub fn single(gpu_schedulable_gb: f64, ram_shelf_gb: f64) -> Self {
        Self {
            gpu_schedulable_gb,
            ram_shelf_gb,
            gpus: Vec::new(),
        }
    }

    pub fn benches(&self) -> Vec<GpuBudget> {
        if self.gpus.is_empty() {
            vec![GpuBudget {
                id: 0,
                schedulable_gb: self.gpu_schedulable_gb,
            }]
        } else {
            self.gpus.clone()
        }
    }

    pub fn max_schedulable(&self) -> f64 {
        self.benches()
            .iter()
            .map(|g| g.schedulable_gb)
            .fold(0.0, f64::max)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModelSpec {
    pub name: String,
    pub vram_gb: f64,
    pub ram_gb: f64,
    pub priority: Priority,
    pub exclusive: bool,
    pub slots: u32,
    pub keep_warm: bool,
    pub gpu: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Action {
    Warm { model: String },
    Load { model: String },
    Wake { model: String },
    Sleep { model: String },
    Discard { model: String },
    Start { job_id: JobId, model: String },
    Preempt { job_id: JobId },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RejectReason {
    UnknownModel,
    TooLarge,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SubmitResult {
    Rejected { reason: RejectReason },
    Accepted { job_id: JobId, actions: Vec<Action> },
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct ModelStatus {
    pub name: String,
    pub tier: Tier,
    pub running: u32,
    pub queued: u32,
    pub gpu: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct StatusSnapshot {
    pub models: Vec<ModelStatus>,
    pub gpu_used_gb: f64,
    pub ram_used_gb: f64,
}
