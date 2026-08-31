use std::collections::{HashMap, VecDeque};

use crate::types::{
    Action, JobId, ModelSpec, ModelStatus, Priority, RejectReason, Resources, StatusSnapshot,
    SubmitResult, Tier,
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
    on_gpu: HashMap<String, u32>,
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
            on_gpu: HashMap::new(),
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

    pub fn ram_used_gb(&self) -> f64 {
        self.models
            .values()
            .filter(|m| self.holds_ram(&m.name))
            .map(|m| m.ram_gb)
            .sum()
    }

    pub fn prefetch(&mut self) -> Vec<Action> {
        let mut actions = Vec::new();
        for name in self.prefetch_candidates() {
            if let Some(action) = self.warm_if_fits(&name) {
                actions.push(action);
            }
        }
        actions
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

    pub fn job_running(&self, job_id: JobId) -> bool {
        self.jobs.contains_key(&job_id)
    }

    pub fn status(&self) -> StatusSnapshot {
        StatusSnapshot {
            models: self.model_statuses(),
            gpu_used_gb: self.gpu_used_gb(),
            ram_used_gb: self.ram_used_gb(),
        }
    }

    fn model_statuses(&self) -> Vec<ModelStatus> {
        let mut models: Vec<ModelStatus> = self
            .models
            .keys()
            .map(|name| self.model_status(name))
            .collect();
        models.sort_by(|a, b| a.name.cmp(&b.name));
        models
    }

    fn model_status(&self, name: &str) -> ModelStatus {
        ModelStatus {
            name: name.to_string(),
            tier: self.tier(name),
            running: self.running(name),
            queued: self.queued(name),
            gpu: if self.tier(name) == Tier::Bench {
                self.gpu_of(name)
            } else {
                None
            },
        }
    }

    pub fn gpu_of(&self, model: &str) -> Option<u32> {
        self.on_gpu.get(model).copied()
    }

    pub fn submit(&mut self, model: &str) -> SubmitResult {
        let Some(spec) = self.models.get(model).cloned() else {
            return rejected(RejectReason::UnknownModel);
        };
        if spec.vram_gb > self.max_for(&spec) {
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

    pub fn drop_job(&mut self, job_id: JobId) -> Vec<Action> {
        if self.queue.iter().any(|(id, _)| *id == job_id) {
            self.remove_queued(job_id);
            return Vec::new();
        }
        self.finish(job_id)
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
        for gpu in self.candidate_gpus(spec) {
            if !self.should_preempt_on(gpu.id, spec) {
                continue;
            }
            let victims = self.preempt_victim_models_on(gpu.id, &spec.name);
            if victims.is_empty() {
                continue;
            }
            let mut actions = self.preempt_models(&victims);
            if let Some(rest) = self.try_place(job_id, spec) {
                actions.extend(rest);
                return Some(actions);
            }
        }
        None
    }

    fn should_preempt_on(&self, gpu: u32, spec: &ModelSpec) -> bool {
        self.may_preempt_on(gpu, spec)
            && !self.preempt_victim_models_on(gpu, &spec.name).is_empty()
            && self.can_place_after_preempt_on(gpu, spec)
    }

    fn may_preempt_on(&self, gpu: u32, spec: &ModelSpec) -> bool {
        match spec.priority {
            Priority::Live => self.has_slot(spec),
            Priority::Background => false,
            Priority::Normal => {
                spec.exclusive && !self.has_running_live_on(gpu) && self.has_slot(spec)
            }
        }
    }

    fn has_slot(&self, spec: &ModelSpec) -> bool {
        self.tier(&spec.name) != Tier::Bench || self.running(&spec.name) < spec.slots
    }

    fn has_running_live_on(&self, gpu: u32) -> bool {
        self.models.values().any(|m| {
            m.priority == Priority::Live
                && self.running(&m.name) > 0
                && self.gpu_of(&m.name) == Some(gpu)
        })
    }

    fn can_place_after_preempt_on(&self, gpu: u32, spec: &ModelSpec) -> bool {
        let stuck = self.immovable_on(gpu, &spec.name);
        if spec.exclusive && !stuck.is_empty() {
            return false;
        }
        if stuck.iter().any(|n| self.is_exclusive(n)) {
            return false;
        }
        let used: f64 = stuck.iter().map(|n| self.vram_gb(n)).sum();
        used + self.added_vram(spec) <= self.schedulable(gpu)
    }

    fn added_vram(&self, spec: &ModelSpec) -> f64 {
        if self.tier(&spec.name) == Tier::Bench {
            0.0
        } else {
            spec.vram_gb
        }
    }

    fn immovable_on(&self, gpu: u32, incoming: &str) -> Vec<String> {
        self.bench_on(gpu, incoming)
            .into_iter()
            .filter(|n| self.running(n) > 0 && self.priority_of(n) == Priority::Live)
            .collect()
    }

    fn preempt_victim_models_on(&self, gpu: u32, incoming: &str) -> Vec<String> {
        let mut names: Vec<String> = self
            .bench_on(gpu, incoming)
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
        for gpu in self.candidate_gpus(spec) {
            if spec.vram_gb > gpu.schedulable_gb {
                continue;
            }
            if let Some(victims) = self.idle_eviction_plan_on(gpu.id, spec) {
                let mut actions = self.evict_idle(&victims);
                actions.extend(self.bring_to_bench_and_start(job_id, spec, gpu.id));
                return Some(actions);
            }
        }
        None
    }

    fn start_if_slot_free(&mut self, job_id: JobId, spec: &ModelSpec) -> Option<Vec<Action>> {
        if self.running(&spec.name) >= spec.slots {
            return None;
        }
        self.mark_running(job_id, spec);
        Some(vec![start_action(job_id, &spec.name)])
    }

    fn bring_to_bench_and_start(
        &mut self,
        job_id: JobId,
        spec: &ModelSpec,
        gpu: u32,
    ) -> Vec<Action> {
        let mut actions = self.demote_for_ram(spec);
        actions.push(self.place_on_bench(spec, gpu));
        self.mark_running(job_id, spec);
        actions.push(start_action(job_id, &spec.name));
        actions
    }

    fn place_on_bench(&mut self, spec: &ModelSpec, gpu: u32) -> Action {
        let from = self.tier(&spec.name);
        self.tiers.insert(spec.name.clone(), Tier::Bench);
        self.on_gpu.insert(spec.name.clone(), gpu);
        match from {
            Tier::Shelf => Action::Wake {
                model: spec.name.clone(),
                gpu,
            },
            _ => Action::Load {
                model: spec.name.clone(),
                gpu,
            },
        }
    }

    fn idle_eviction_plan_on(&self, gpu: u32, spec: &ModelSpec) -> Option<Vec<String>> {
        if self.can_place_on(gpu, spec) {
            return Some(Vec::new());
        }
        if spec.exclusive {
            self.exclusive_idle_plan_on(gpu, spec)
        } else {
            self.share_idle_plan_on(gpu, spec)
        }
    }

    fn exclusive_idle_plan_on(&self, gpu: u32, spec: &ModelSpec) -> Option<Vec<String>> {
        let others = self.bench_on(gpu, &spec.name);
        if others.iter().any(|n| !self.is_idle(n)) {
            return None;
        }
        if !self.fits_after_on(gpu, &others, spec) {
            return None;
        }
        Some(self.sorted_idle(others))
    }

    fn share_idle_plan_on(&self, gpu: u32, spec: &ModelSpec) -> Option<Vec<String>> {
        if self.has_running_exclusive_on(gpu, &spec.name) {
            return None;
        }
        let evict = self.idle_exclusives_on(gpu, &spec.name);
        if self.can_place_after_on(gpu, &evict, spec) {
            return Some(self.sorted_idle(evict));
        }
        self.extend_idle_until_fit_on(gpu, spec, evict)
    }

    fn extend_idle_until_fit_on(
        &self,
        gpu: u32,
        spec: &ModelSpec,
        mut evict: Vec<String>,
    ) -> Option<Vec<String>> {
        for victim in self.sorted_idle(self.idle_bench_on(gpu, &spec.name)) {
            if evict.contains(&victim) {
                continue;
            }
            evict.push(victim);
            if self.can_place_after_on(gpu, &evict, spec) {
                return Some(self.sorted_idle(evict));
            }
        }
        None
    }

    fn evict_idle(&mut self, names: &[String]) -> Vec<Action> {
        names.iter().map(|n| self.sleep_or_discard(n)).collect()
    }

    fn sleep_or_discard(&mut self, name: &str) -> Action {
        self.on_gpu.remove(name);
        if self.is_keep_warm(name) {
            self.tiers.insert(name.to_string(), Tier::Shelf);
            self.stamp_idle(name);
            Action::Sleep {
                model: name.to_string(),
            }
        } else {
            self.idle_since.remove(name);
            self.tiers.insert(name.to_string(), Tier::Cupboard);
            Action::Discard {
                model: name.to_string(),
            }
        }
    }

    fn prefetch_candidates(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .models
            .values()
            .filter(|m| m.keep_warm && self.tier(&m.name) == Tier::Cupboard)
            .map(|m| m.name.clone())
            .collect();
        names.sort_by(|a, b| self.priority_of(b).cmp(&self.priority_of(a)).then(a.cmp(b)));
        names
    }

    fn warm_if_fits(&mut self, name: &str) -> Option<Action> {
        let spec = self.models.get(name)?.clone();
        if self.ram_over_budget(&spec) {
            return None;
        }
        Some(self.warm(&spec))
    }

    fn warm(&mut self, spec: &ModelSpec) -> Action {
        self.tiers.insert(spec.name.clone(), Tier::Shelf);
        self.stamp_idle(&spec.name);
        Action::Warm {
            model: spec.name.clone(),
        }
    }

    fn demote_for_ram(&mut self, spec: &ModelSpec) -> Vec<Action> {
        let mut actions = Vec::new();
        while self.ram_over_budget(spec) {
            let Some(victim) = self.lru_shelf_except(&spec.name) else {
                break;
            };
            actions.push(self.discard_shelf(&victim));
        }
        actions
    }

    fn ram_over_budget(&self, spec: &ModelSpec) -> bool {
        self.ram_used_gb() + self.added_ram(spec) > self.resources.ram_shelf_gb
    }

    fn added_ram(&self, spec: &ModelSpec) -> f64 {
        if spec.keep_warm && !self.holds_ram(&spec.name) {
            spec.ram_gb
        } else {
            0.0
        }
    }

    fn holds_ram(&self, name: &str) -> bool {
        match self.tier(name) {
            Tier::Shelf => true,
            Tier::Bench => self.is_keep_warm(name),
            Tier::Cupboard => false,
        }
    }

    fn lru_shelf_except(&self, incoming: &str) -> Option<String> {
        let mut names: Vec<String> = self
            .tiers
            .iter()
            .filter(|(n, t)| **t == Tier::Shelf && *n != incoming)
            .map(|(n, _)| n.clone())
            .collect();
        names.sort_by(|a, b| self.idle_rank(a).cmp(&self.idle_rank(b)).then(a.cmp(b)));
        names.into_iter().next()
    }

    fn discard_shelf(&mut self, name: &str) -> Action {
        self.idle_since.remove(name);
        self.tiers.insert(name.to_string(), Tier::Cupboard);
        Action::Discard {
            model: name.to_string(),
        }
    }

    fn is_keep_warm(&self, name: &str) -> bool {
        self.models.get(name).map(|m| m.keep_warm).unwrap_or(false)
    }

    fn can_place_on(&self, gpu: u32, spec: &ModelSpec) -> bool {
        self.exclusive_ok_on(gpu, spec) && self.fits_on(gpu, spec)
    }

    fn can_place_after_on(&self, gpu: u32, evict: &[String], spec: &ModelSpec) -> bool {
        self.exclusive_ok_after_on(gpu, evict, spec) && self.fits_after_on(gpu, evict, spec)
    }

    fn exclusive_ok_on(&self, gpu: u32, spec: &ModelSpec) -> bool {
        let others = self.bench_on(gpu, &spec.name);
        if spec.exclusive {
            return others.is_empty();
        }
        others.iter().all(|name| !self.is_exclusive(name))
    }

    fn exclusive_ok_after_on(&self, gpu: u32, evict: &[String], spec: &ModelSpec) -> bool {
        let others: Vec<String> = self
            .bench_on(gpu, &spec.name)
            .into_iter()
            .filter(|n| !evict.iter().any(|e| e == n))
            .collect();
        if spec.exclusive {
            others.is_empty()
        } else {
            others.iter().all(|n| !self.is_exclusive(n))
        }
    }

    fn bench_on(&self, gpu: u32, except: &str) -> Vec<String> {
        self.models
            .keys()
            .filter(|name| {
                *name != except
                    && self.tier(name) == Tier::Bench
                    && self.gpu_of(name) == Some(gpu)
            })
            .cloned()
            .collect()
    }

    fn is_idle(&self, name: &str) -> bool {
        self.running(name) == 0 && self.queued(name) == 0
    }

    fn idle_bench_on(&self, gpu: u32, model: &str) -> Vec<String> {
        self.bench_on(gpu, model)
            .into_iter()
            .filter(|n| self.is_idle(n))
            .collect()
    }

    fn idle_exclusives_on(&self, gpu: u32, model: &str) -> Vec<String> {
        self.idle_bench_on(gpu, model)
            .into_iter()
            .filter(|n| self.is_exclusive(n))
            .collect()
    }

    fn has_running_exclusive_on(&self, gpu: u32, model: &str) -> bool {
        self.bench_on(gpu, model)
            .iter()
            .any(|n| self.is_exclusive(n) && self.running(n) > 0)
    }

    fn candidate_gpus(&self, spec: &ModelSpec) -> Vec<crate::GpuBudget> {
        let all = self.resources.benches();
        match spec.gpu {
            Some(id) => all.into_iter().filter(|g| g.id == id).collect(),
            None => all,
        }
    }

    fn max_for(&self, spec: &ModelSpec) -> f64 {
        self.candidate_gpus(spec)
            .iter()
            .map(|g| g.schedulable_gb)
            .fold(0.0, f64::max)
    }

    fn schedulable(&self, gpu: u32) -> f64 {
        self.resources
            .benches()
            .into_iter()
            .find(|g| g.id == gpu)
            .map(|g| g.schedulable_gb)
            .unwrap_or(0.0)
    }

    fn gpu_used_on(&self, gpu: u32) -> f64 {
        self.models
            .values()
            .filter(|m| self.tier(&m.name) == Tier::Bench && self.gpu_of(&m.name) == Some(gpu))
            .map(|m| m.vram_gb)
            .sum()
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

    fn fits_on(&self, gpu: u32, spec: &ModelSpec) -> bool {
        self.gpu_used_on(gpu) + spec.vram_gb <= self.schedulable(gpu)
    }

    fn fits_after_on(&self, gpu: u32, evict: &[String], spec: &ModelSpec) -> bool {
        let freed: f64 = evict.iter().map(|n| self.vram_gb(n)).sum();
        self.gpu_used_on(gpu) - freed + spec.vram_gb <= self.schedulable(gpu)
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
            self.stamp_idle(model);
        }
    }

    fn stamp_idle(&mut self, name: &str) {
        self.idle_since.insert(name.to_string(), self.idle_tick);
        self.idle_tick += 1;
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
