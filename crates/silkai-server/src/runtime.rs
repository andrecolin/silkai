use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex as StdMutex};

use silkai_adapters::{Engine, EngineError, FakeEngine};
use silkai_sched::{
    Action, JobId, RejectReason, SchedError, Scheduler, StatusSnapshot, SubmitResult,
};
use tokio::sync::{mpsc, Mutex};
use tokio_util::sync::CancellationToken;

use crate::config::{AppConfig, ConfiguredModel};

#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    #[error("unknown model")]
    Unknown,
    #[error("model disabled")]
    Disabled,
    #[error("model too large")]
    TooLarge,
    #[error(transparent)]
    Engine(#[from] EngineError),
}

struct Waiter {
    prompt: String,
    tx: mpsc::Sender<String>,
}

struct Inner {
    scheduler: Mutex<Scheduler>,
    snapshot: StdMutex<StatusSnapshot>,
    engines: HashMap<String, Arc<dyn Engine>>,
    paths: HashMap<String, String>,
    disabled: HashSet<String>,
    waiters: Mutex<HashMap<JobId, Waiter>>,
    cancels: Mutex<HashMap<JobId, CancellationToken>>,
}

impl Inner {
    fn new(scheduler: Scheduler, cfg: &AppConfig) -> Self {
        Self {
            snapshot: StdMutex::new(scheduler.status()),
            scheduler: Mutex::new(scheduler),
            engines: engines_for(&cfg.enabled),
            paths: paths_for(&cfg.enabled),
            disabled: disabled_set(&cfg.disabled),
            waiters: Mutex::new(HashMap::new()),
            cancels: Mutex::new(HashMap::new()),
        }
    }
}

#[derive(Clone)]
pub struct Runtime {
    inner: Arc<Inner>,
}

impl Runtime {
    pub async fn new(cfg: AppConfig) -> Result<Self, RuntimeError> {
        let (rt, warmup) = Self::assemble(cfg)?;
        rt.apply_all(warmup).await?;
        Ok(rt)
    }

    pub async fn submit_chat(
        &self,
        model: &str,
        prompt: &str,
    ) -> Result<(JobId, mpsc::Receiver<String>), RuntimeError> {
        self.ensure_enabled(model)?;
        let (tx, rx) = mpsc::channel(16);
        let job_id = self.accept(model, prompt, tx).await?;
        Ok((job_id, rx))
    }

    pub async fn finished(&self, job_id: JobId) {
        self.complete(job_id).await.expect("runtime finish actions");
    }

    pub fn status(&self) -> StatusSnapshot {
        self.inner.snapshot.lock().expect("status mutex").clone()
    }
}

impl Runtime {
    fn assemble(cfg: AppConfig) -> Result<(Self, Vec<Action>), RuntimeError> {
        let specs = cfg.enabled.iter().map(|m| m.spec.clone()).collect();
        let mut scheduler = Scheduler::new(cfg.resources.clone(), specs).map_err(sched_err)?;
        let warmup = prefetch_actions(&mut scheduler, cfg.prefetch_on_start);
        let rt = Self {
            inner: Arc::new(Inner::new(scheduler, &cfg)),
        };
        Ok((rt, warmup))
    }

    fn ensure_enabled(&self, model: &str) -> Result<(), RuntimeError> {
        if self.inner.disabled.contains(model) {
            Err(RuntimeError::Disabled)
        } else {
            Ok(())
        }
    }

    async fn accept(
        &self,
        model: &str,
        prompt: &str,
        tx: mpsc::Sender<String>,
    ) -> Result<JobId, RuntimeError> {
        let mut sched = self.inner.scheduler.lock().await;
        let (job_id, actions) = match sched.submit(model) {
            SubmitResult::Accepted { job_id, actions } => (job_id, actions),
            SubmitResult::Rejected { reason } => return Err(reject_err(reason)),
        };
        self.store_waiter(job_id, prompt, tx).await;
        self.apply_all(actions).await?;
        self.record_status(&sched);
        Ok(job_id)
    }

    async fn complete(&self, job_id: JobId) -> Result<(), RuntimeError> {
        let mut sched = self.inner.scheduler.lock().await;
        self.forget(job_id).await;
        let actions = sched.finish(job_id);
        self.apply_all(actions).await?;
        self.record_status(&sched);
        Ok(())
    }

    fn record_status(&self, sched: &Scheduler) {
        *self.inner.snapshot.lock().expect("status mutex") = sched.status();
    }

    async fn store_waiter(&self, job_id: JobId, prompt: &str, tx: mpsc::Sender<String>) {
        let waiter = Waiter {
            prompt: prompt.to_string(),
            tx,
        };
        self.inner.waiters.lock().await.insert(job_id, waiter);
    }

    async fn waiter(&self, job_id: JobId) -> Option<(String, mpsc::Sender<String>)> {
        self.inner
            .waiters
            .lock()
            .await
            .get(&job_id)
            .map(|w| (w.prompt.clone(), w.tx.clone()))
    }

    async fn watch(&self, job_id: JobId) -> CancellationToken {
        let token = CancellationToken::new();
        self.inner
            .cancels
            .lock()
            .await
            .insert(job_id, token.clone());
        token
    }

    async fn forget(&self, job_id: JobId) {
        if let Some(token) = self.inner.cancels.lock().await.remove(&job_id) {
            token.cancel();
        }
        self.inner.waiters.lock().await.remove(&job_id);
    }

    async fn apply_all(&self, actions: Vec<Action>) -> Result<(), RuntimeError> {
        for action in actions {
            self.apply(action).await?;
        }
        Ok(())
    }

    async fn apply(&self, action: Action) -> Result<(), RuntimeError> {
        match action {
            Action::Warm { model } => self.warm(&model).await,
            Action::Load { model } => self.load(&model).await,
            Action::Wake { model } => self.wake(&model).await,
            Action::Sleep { model } => self.sleep(&model).await,
            Action::Discard { model } => self.discard(&model).await,
            Action::Preempt { job_id } => self.preempt(job_id).await,
            Action::Start { job_id, model } => self.start(job_id, &model).await,
        }
    }

    async fn warm(&self, model: &str) -> Result<(), RuntimeError> {
        let path = self.path(model)?;
        Ok(self.engine(model)?.warm(path).await?)
    }

    async fn load(&self, model: &str) -> Result<(), RuntimeError> {
        let path = self.path(model)?;
        Ok(self.engine(model)?.load(path).await?)
    }

    async fn wake(&self, model: &str) -> Result<(), RuntimeError> {
        Ok(self.engine(model)?.wake().await?)
    }

    async fn sleep(&self, model: &str) -> Result<(), RuntimeError> {
        Ok(self.engine(model)?.sleep().await?)
    }

    async fn discard(&self, model: &str) -> Result<(), RuntimeError> {
        Ok(self.engine(model)?.discard().await?)
    }

    async fn preempt(&self, job_id: JobId) -> Result<(), RuntimeError> {
        if let Some(token) = self.inner.cancels.lock().await.remove(&job_id) {
            token.cancel();
        }
        Ok(())
    }

    async fn start(&self, job_id: JobId, model: &str) -> Result<(), RuntimeError> {
        let (prompt, tx) = self.require_waiter(job_id).await?;
        let token = self.watch(job_id).await;
        let rx = self.engine(model)?.run(&prompt, token.clone()).await?;
        let inner = Arc::clone(&self.inner);
        tokio::spawn(forward_job(job_id, rx, tx, token, inner));
        Ok(())
    }

    async fn require_waiter(
        &self,
        job_id: JobId,
    ) -> Result<(String, mpsc::Sender<String>), RuntimeError> {
        self.waiter(job_id)
            .await
            .ok_or_else(|| RuntimeError::Engine(EngineError::Other("missing waiter".into())))
    }

    fn engine(&self, model: &str) -> Result<&Arc<dyn Engine>, RuntimeError> {
        self.inner
            .engines
            .get(model)
            .ok_or_else(|| missing("engine", model))
    }

    fn path(&self, model: &str) -> Result<&str, RuntimeError> {
        self.inner
            .paths
            .get(model)
            .map(String::as_str)
            .ok_or_else(|| missing("path", model))
    }
}

async fn forward_job(
    job_id: JobId,
    rx: mpsc::Receiver<String>,
    tx: mpsc::Sender<String>,
    token: CancellationToken,
    inner: Arc<Inner>,
) {
    pump_tokens(rx, tx, token.clone()).await;
    if token.is_cancelled() {
        return;
    }
    inner.waiters.lock().await.remove(&job_id);
}

async fn pump_tokens(
    mut rx: mpsc::Receiver<String>,
    tx: mpsc::Sender<String>,
    token: CancellationToken,
) {
    loop {
        tokio::select! {
            _ = token.cancelled() => return,
            next = rx.recv() => match next {
                Some(chunk) => {
                    if tx.send(chunk).await.is_err() {
                        return;
                    }
                }
                None => return,
            },
        }
    }
}

fn prefetch_actions(scheduler: &mut Scheduler, prefetch: bool) -> Vec<Action> {
    if prefetch {
        scheduler.prefetch()
    } else {
        Vec::new()
    }
}

fn engines_for(models: &[ConfiguredModel]) -> HashMap<String, Arc<dyn Engine>> {
    models
        .iter()
        .map(|m| {
            let engine = FakeEngine::new(&m.spec.name, m.spec.vram_gb);
            (m.spec.name.clone(), Arc::new(engine) as Arc<dyn Engine>)
        })
        .collect()
}

fn paths_for(models: &[ConfiguredModel]) -> HashMap<String, String> {
    models
        .iter()
        .map(|m| (m.spec.name.clone(), m.path.clone()))
        .collect()
}

fn disabled_set(models: &[ConfiguredModel]) -> HashSet<String> {
    models.iter().map(|m| m.spec.name.clone()).collect()
}

fn sched_err(err: SchedError) -> RuntimeError {
    RuntimeError::Engine(EngineError::Other(format!("{err:?}")))
}

fn reject_err(reason: RejectReason) -> RuntimeError {
    match reason {
        RejectReason::UnknownModel => RuntimeError::Unknown,
        RejectReason::TooLarge => RuntimeError::TooLarge,
    }
}

fn missing(kind: &str, model: &str) -> RuntimeError {
    RuntimeError::Engine(EngineError::Other(format!("no {kind}: {model}")))
}
