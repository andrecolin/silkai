use std::collections::{HashMap, VecDeque};

use crate::types::{Action, JobId, ModelSpec, RejectReason, Resources, SubmitResult, Tier};

#[derive(Debug)]
pub enum SchedError {
    DuplicateModel(String),
    InvalidSlots,
}

#[derive(Debug)]
pub struct Scheduler {
    resources: Resources,
    models: HashMap<String, ModelSpec>,
    tiers: HashMap<String, Tier>,
    running: HashMap<String, u32>,
    queue: VecDeque<(JobId, String)>,
    next_id: u64,
}

impl Scheduler {
    pub fn new(resources: Resources, models: Vec<ModelSpec>) -> Result<Self, SchedError> {
        let models = index_models(models)?;
        let tiers = models
            .keys()
            .cloned()
            .map(|n| (n, Tier::Cupboard))
            .collect();
        let running = models.keys().cloned().map(|n| (n, 0)).collect();
        Ok(Self {
            resources,
            models,
            tiers,
            running,
            queue: VecDeque::new(),
            next_id: 1,
        })
    }

    pub fn gpu_used_gb(&self) -> f64 {
        self.models
            .values()
            .filter(|m| self.tier(&m.name) == Tier::Bench)
            .map(|m| m.vram_gb)
            .sum()
    }

    pub fn tier(&self, model: &str) -> Tier {
        self.tiers.get(model).copied().unwrap_or(Tier::Cupboard)
    }

    pub fn running(&self, model: &str) -> u32 {
        self.running.get(model).copied().unwrap_or(0)
    }

    pub fn queued(&self, model: &str) -> u32 {
        self.queue.iter().filter(|(_, name)| name == model).count() as u32
    }

    pub fn submit(&mut self, model: &str) -> SubmitResult {
        let Some(spec) = self.models.get(model).cloned() else {
            return rejected(RejectReason::UnknownModel);
        };
        if spec.vram_gb > self.resources.gpu_schedulable_gb {
            return rejected(RejectReason::TooLarge);
        }
        let job_id = self.alloc_id();
        if let Some(actions) = self.try_place(job_id, &spec) {
            SubmitResult::Accepted { job_id, actions }
        } else {
            self.queue.push_back((job_id, spec.name));
            SubmitResult::Accepted {
                job_id,
                actions: vec![],
            }
        }
    }

    fn alloc_id(&mut self) -> JobId {
        let id = JobId(self.next_id);
        self.next_id += 1;
        id
    }

    fn try_place(&mut self, job_id: JobId, spec: &ModelSpec) -> Option<Vec<Action>> {
        if self.tier(&spec.name) == Tier::Bench {
            return self.start_if_slot_free(job_id, spec);
        }
        if !self.can_place(spec) {
            return None;
        }
        Some(self.bring_to_bench_and_start(job_id, spec))
    }

    fn start_if_slot_free(&mut self, job_id: JobId, spec: &ModelSpec) -> Option<Vec<Action>> {
        if self.running(&spec.name) >= spec.slots {
            return None;
        }
        self.incr_running(&spec.name);
        Some(vec![start_action(job_id, &spec.name)])
    }

    fn bring_to_bench_and_start(&mut self, job_id: JobId, spec: &ModelSpec) -> Vec<Action> {
        let action = self.place_on_bench(spec);
        self.incr_running(&spec.name);
        vec![action, start_action(job_id, &spec.name)]
    }

    fn place_on_bench(&mut self, spec: &ModelSpec) -> Action {
        let from = self.tier(&spec.name);
        self.tiers.insert(spec.name.clone(), Tier::Bench);
        match from {
            Tier::Shelf => Action::Wake {
                model: spec.name.clone(),
            },
            _ => Action::Load {
                model: spec.name.clone(),
            },
        }
    }

    fn can_place(&self, spec: &ModelSpec) -> bool {
        self.exclusive_ok(spec) && self.fits(spec)
    }

    fn exclusive_ok(&self, spec: &ModelSpec) -> bool {
        let others = self.bench_except(&spec.name);
        if spec.exclusive {
            return others.is_empty();
        }
        others.iter().all(|name| !self.is_exclusive(name))
    }

    fn bench_except(&self, model: &str) -> Vec<&str> {
        self.tiers
            .iter()
            .filter(|(name, tier)| **tier == Tier::Bench && *name != model)
            .map(|(name, _)| name.as_str())
            .collect()
    }

    fn is_exclusive(&self, name: &str) -> bool {
        self.models.get(name).map(|m| m.exclusive).unwrap_or(false)
    }

    fn fits(&self, spec: &ModelSpec) -> bool {
        self.gpu_used_gb() + spec.vram_gb <= self.resources.gpu_schedulable_gb
    }

    fn incr_running(&mut self, model: &str) {
        if let Some(n) = self.running.get_mut(model) {
            *n += 1;
        }
    }
}

fn index_models(models: Vec<ModelSpec>) -> Result<HashMap<String, ModelSpec>, SchedError> {
    let mut specs = HashMap::new();
    for spec in models {
        if spec.slots == 0 {
            return Err(SchedError::InvalidSlots);
        }
        if specs.contains_key(&spec.name) {
            return Err(SchedError::DuplicateModel(spec.name));
        }
        specs.insert(spec.name.clone(), spec);
    }
    Ok(specs)
}

fn rejected(reason: RejectReason) -> SubmitResult {
    SubmitResult::Rejected { reason }
}

fn start_action(job_id: JobId, model: &str) -> Action {
    Action::Start {
        job_id,
        model: model.to_string(),
    }
}
