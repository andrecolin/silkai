use std::collections::{HashMap, VecDeque};

use crate::types::{
    Action, JobId, ModelSpec, Priority, RejectReason, Resources, SubmitResult, Tier,
};

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
    jobs: HashMap<JobId, String>,
    queue: VecDeque<(JobId, String)>,
    idle_since: HashMap<String, u64>,
    idle_tick: u64,
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
            jobs: HashMap::new(),
            queue: VecDeque::new(),
            idle_since: HashMap::new(),
            idle_tick: 0,
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
        self.accept(job_id, spec)
    }

    fn accept(&mut self, job_id: JobId, spec: ModelSpec) -> SubmitResult {
        if let Some(actions) = self.place_or_preempt(job_id, &spec) {
            return accepted(job_id, actions);
        }
        self.queue.push_back((job_id, spec.name));
        accepted(job_id, vec![])
    }

    pub fn finish(&mut self, job_id: JobId) -> Vec<Action> {
        if self.release_running(job_id).is_none() {
            return Vec::new();
        }
        self.admit_from_queue()
    }

    fn alloc_id(&mut self) -> JobId {
        let id = JobId(self.next_id);
        self.next_id += 1;
        id
    }

    fn release_running(&mut self, job_id: JobId) -> Option<String> {
        let model = self.jobs.remove(&job_id)?;
        self.decr_running(&model);
        Some(model)
    }

    fn admit_from_queue(&mut self) -> Vec<Action> {
        let mut actions = Vec::new();
        while let Some(more) = self.admit_next() {
            actions.extend(more);
        }
        actions
    }

    fn admit_next(&mut self) -> Option<Vec<Action>> {
        for i in 0..self.queue.len() {
            let (job_id, name) = self.queue[i].clone();
            let Some(spec) = self.models.get(&name).cloned() else {
                continue;
            };
            let Some(acts) = self.place_or_preempt(job_id, &spec) else {
                continue;
            };
            self.remove_queued(job_id);
            return Some(acts);
        }
        None
    }

    fn place_or_preempt(&mut self, job_id: JobId, spec: &ModelSpec) -> Option<Vec<Action>> {
        self.try_place(job_id, spec)
            .or_else(|| self.preempt_and_place(job_id, spec))
    }

    fn remove_queued(&mut self, job_id: JobId) {
        if let Some(i) = self.queue.iter().position(|(id, _)| *id == job_id) {
            self.queue.remove(i);
        }
    }

    fn preempt_and_place(&mut self, job_id: JobId, spec: &ModelSpec) -> Option<Vec<Action>> {
        if !self.should_preempt(spec) {
            return None;
        }
        let victims = self.preempt_victim_models(&spec.name);
        let mut actions = self.preempt_models(&victims);
        actions.extend(self.try_place(job_id, spec)?);
        Some(actions)
    }

    fn should_preempt(&self, spec: &ModelSpec) -> bool {
        self.may_preempt(spec)
            && !self.preempt_victim_models(&spec.name).is_empty()
            && self.can_place_after_preempt(spec)
    }

    fn may_preempt(&self, spec: &ModelSpec) -> bool {
        match spec.priority {
            Priority::Live => self.has_slot(spec),
            Priority::Background => false,
            Priority::Normal => spec.exclusive && !self.has_running_live() && self.has_slot(spec),
        }
    }

    fn has_slot(&self, spec: &ModelSpec) -> bool {
        self.tier(&spec.name) != Tier::Bench || self.running(&spec.name) < spec.slots
    }

    fn has_running_live(&self) -> bool {
        self.models
            .values()
            .any(|m| m.priority == Priority::Live && self.running(&m.name) > 0)
    }

    fn can_place_after_preempt(&self, spec: &ModelSpec) -> bool {
        let stuck = self.immovable_bench(&spec.name);
        if spec.exclusive && !stuck.is_empty() {
            return false;
        }
        if stuck.iter().any(|n| self.is_exclusive(n)) {
            return false;
        }
        let used: f64 = stuck.iter().map(|n| self.vram_gb(n)).sum();
        used + self.added_vram(spec) <= self.resources.gpu_schedulable_gb
    }

    fn added_vram(&self, spec: &ModelSpec) -> f64 {
        if self.tier(&spec.name) == Tier::Bench {
            0.0
        } else {
            spec.vram_gb
        }
    }

    fn immovable_bench(&self, incoming: &str) -> Vec<String> {
        self.bench_names_except(incoming)
            .into_iter()
            .filter(|n| self.running(n) > 0 && self.priority_of(n) == Priority::Live)
            .collect()
    }

    fn preempt_victim_models(&self, incoming: &str) -> Vec<String> {
        let mut names: Vec<String> = self
            .bench_names_except(incoming)
            .into_iter()
            .filter(|n| self.running(n) > 0 && self.priority_of(n) != Priority::Live)
            .collect();
        names.sort();
        names
    }

    fn preempt_models(&mut self, models: &[String]) -> Vec<Action> {
        let mut actions = Vec::new();
        for name in models {
            actions.extend(self.preempt_model(name));
        }
        actions
    }

    fn preempt_model(&mut self, name: &str) -> Vec<Action> {
        let ids = self.running_job_ids(name);
        let mut actions = self.take_running_jobs(&ids);
        self.requeue_front(name, &ids);
        actions.push(self.sleep_or_discard(name));
        actions
    }

    fn take_running_jobs(&mut self, ids: &[JobId]) -> Vec<Action> {
        ids.iter()
            .map(|&job_id| {
                self.jobs.remove(&job_id);
                Action::Preempt { job_id }
            })
            .collect()
    }

    fn requeue_front(&mut self, name: &str, ids: &[JobId]) {
        for &job_id in ids.iter().rev() {
            self.queue.push_front((job_id, name.to_string()));
        }
        if let Some(n) = self.running.get_mut(name) {
            *n = 0;
        }
    }

    fn running_job_ids(&self, name: &str) -> Vec<JobId> {
        let mut ids: Vec<JobId> = self
            .jobs
            .iter()
            .filter(|(_, model)| *model == name)
            .map(|(id, _)| *id)
            .collect();
        ids.sort_by_key(|id| id.0);
        ids
    }

    fn try_place(&mut self, job_id: JobId, spec: &ModelSpec) -> Option<Vec<Action>> {
        if self.tier(&spec.name) == Tier::Bench {
            return self.start_if_slot_free(job_id, spec);
        }
        let victims = self.idle_eviction_plan(spec)?;
        let mut actions = self.evict_idle(&victims);
        actions.extend(self.bring_to_bench_and_start(job_id, spec));
        Some(actions)
    }

    fn start_if_slot_free(&mut self, job_id: JobId, spec: &ModelSpec) -> Option<Vec<Action>> {
        if self.running(&spec.name) >= spec.slots {
            return None;
        }
        self.mark_running(job_id, spec);
        Some(vec![start_action(job_id, &spec.name)])
    }

    fn bring_to_bench_and_start(&mut self, job_id: JobId, spec: &ModelSpec) -> Vec<Action> {
        let action = self.place_on_bench(spec);
        self.mark_running(job_id, spec);
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

    fn idle_eviction_plan(&self, spec: &ModelSpec) -> Option<Vec<String>> {
        if self.can_place(spec) {
            return Some(Vec::new());
        }
        if spec.exclusive {
            self.exclusive_idle_plan(spec)
        } else {
            self.share_idle_plan(spec)
        }
    }

    fn exclusive_idle_plan(&self, spec: &ModelSpec) -> Option<Vec<String>> {
        let others = self.bench_names_except(&spec.name);
        if others.iter().any(|n| self.running(n) > 0) {
            return None;
        }
        if !self.fits_after(&others, spec) {
            return None;
        }
        Some(self.sorted_idle(others))
    }

    fn share_idle_plan(&self, spec: &ModelSpec) -> Option<Vec<String>> {
        if self.has_running_exclusive_other(&spec.name) {
            return None;
        }
        let evict = self.idle_exclusives_except(&spec.name);
        if self.can_place_after(&evict, spec) {
            return Some(self.sorted_idle(evict));
        }
        self.extend_idle_until_fit(spec, evict)
    }

    fn extend_idle_until_fit(
        &self,
        spec: &ModelSpec,
        mut evict: Vec<String>,
    ) -> Option<Vec<String>> {
        for victim in self.sorted_idle(self.idle_bench_except(&spec.name)) {
            if evict.contains(&victim) {
                continue;
            }
            evict.push(victim);
            if self.can_place_after(&evict, spec) {
                return Some(self.sorted_idle(evict));
            }
        }
        None
    }

    fn evict_idle(&mut self, names: &[String]) -> Vec<Action> {
        names.iter().map(|n| self.sleep_or_discard(n)).collect()
    }

    fn sleep_or_discard(&mut self, name: &str) -> Action {
        let keep_warm = self.models.get(name).map(|m| m.keep_warm).unwrap_or(false);
        self.idle_since.remove(name);
        if keep_warm {
            self.tiers.insert(name.to_string(), Tier::Shelf);
            Action::Sleep {
                model: name.to_string(),
            }
        } else {
            self.tiers.insert(name.to_string(), Tier::Cupboard);
            Action::Discard {
                model: name.to_string(),
            }
        }
    }

    fn can_place(&self, spec: &ModelSpec) -> bool {
        self.exclusive_ok(spec) && self.fits(spec)
    }

    fn can_place_after(&self, evict: &[String], spec: &ModelSpec) -> bool {
        self.exclusive_ok_after(evict, spec) && self.fits_after(evict, spec)
    }

    fn exclusive_ok(&self, spec: &ModelSpec) -> bool {
        let others = self.bench_except(&spec.name);
        if spec.exclusive {
            return others.is_empty();
        }
        others.iter().all(|name| !self.is_exclusive(name))
    }

    fn exclusive_ok_after(&self, evict: &[String], spec: &ModelSpec) -> bool {
        let others: Vec<&str> = self
            .bench_except(&spec.name)
            .into_iter()
            .filter(|n| !evict.iter().any(|e| e == n))
            .collect();
        if spec.exclusive {
            others.is_empty()
        } else {
            others.iter().all(|n| !self.is_exclusive(n))
        }
    }

    fn bench_except(&self, model: &str) -> Vec<&str> {
        self.tiers
            .iter()
            .filter(|(name, tier)| **tier == Tier::Bench && *name != model)
            .map(|(name, _)| name.as_str())
            .collect()
    }

    fn bench_names_except(&self, model: &str) -> Vec<String> {
        self.bench_except(model)
            .into_iter()
            .map(|n| n.to_string())
            .collect()
    }

    fn idle_bench_except(&self, model: &str) -> Vec<String> {
        self.bench_names_except(model)
            .into_iter()
            .filter(|n| self.running(n) == 0)
            .collect()
    }

    fn idle_exclusives_except(&self, model: &str) -> Vec<String> {
        self.idle_bench_except(model)
            .into_iter()
            .filter(|n| self.is_exclusive(n))
            .collect()
    }

    fn has_running_exclusive_other(&self, model: &str) -> bool {
        self.bench_except(model)
            .iter()
            .any(|n| self.is_exclusive(n) && self.running(n) > 0)
    }

    fn sorted_idle(&self, mut names: Vec<String>) -> Vec<String> {
        names.sort_by(|a, b| {
            self.priority_of(a)
                .cmp(&self.priority_of(b))
                .then(self.idle_rank(a).cmp(&self.idle_rank(b)))
                .then(a.cmp(b))
        });
        names
    }

    fn priority_of(&self, name: &str) -> Priority {
        self.models
            .get(name)
            .map(|m| m.priority)
            .unwrap_or(Priority::Background)
    }

    fn idle_rank(&self, name: &str) -> u64 {
        self.idle_since.get(name).copied().unwrap_or(0)
    }

    fn is_exclusive(&self, name: &str) -> bool {
        self.models.get(name).map(|m| m.exclusive).unwrap_or(false)
    }

    fn fits(&self, spec: &ModelSpec) -> bool {
        self.gpu_used_gb() + spec.vram_gb <= self.resources.gpu_schedulable_gb
    }

    fn fits_after(&self, evict: &[String], spec: &ModelSpec) -> bool {
        let freed: f64 = evict.iter().map(|n| self.vram_gb(n)).sum();
        self.gpu_used_gb() - freed + spec.vram_gb <= self.resources.gpu_schedulable_gb
    }

    fn vram_gb(&self, name: &str) -> f64 {
        self.models.get(name).map(|m| m.vram_gb).unwrap_or(0.0)
    }

    fn mark_running(&mut self, job_id: JobId, spec: &ModelSpec) {
        self.jobs.insert(job_id, spec.name.clone());
        self.incr_running(&spec.name);
        self.idle_since.remove(&spec.name);
    }

    fn incr_running(&mut self, model: &str) {
        if let Some(n) = self.running.get_mut(model) {
            *n += 1;
        }
    }

    fn decr_running(&mut self, model: &str) {
        let Some(n) = self.running.get_mut(model) else {
            return;
        };
        *n = n.saturating_sub(1);
        if *n == 0 {
            self.idle_since.insert(model.to_string(), self.idle_tick);
            self.idle_tick += 1;
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

fn accepted(job_id: JobId, actions: Vec<Action>) -> SubmitResult {
    SubmitResult::Accepted { job_id, actions }
}

fn start_action(job_id: JobId, model: &str) -> Action {
    Action::Start {
        job_id,
        model: model.to_string(),
    }
}
