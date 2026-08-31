use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex as StdMutex, Weak};
use std::time::Duration;

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
    apply_tx: mpsc::UnboundedSender<Vec<Action>>,
    apply_rx: Mutex<Option<mpsc::UnboundedReceiver<Vec<Action>>>>,
    request_timeout: Duration,
}

impl Inner {
    fn new(scheduler: Scheduler, cfg: &AppConfig) -> Self {
        let (apply_tx, apply_rx) = mpsc::unbounded_channel();
        Self {
            snapshot: StdMutex::new(scheduler.status()),
            scheduler: Mutex::new(scheduler),
            engines: engines_for(&cfg.enabled),
            paths: paths_for(&cfg.enabled),
            disabled: disabled_set(&cfg.disabled),
            waiters: Mutex::new(HashMap::new()),
            cancels: Mutex::new(HashMap::new()),
            apply_tx,
            apply_rx: Mutex::new(Some(apply_rx)),
            request_timeout: cfg.request_timeout,
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
        rt.spawn_applier().await;
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

    pub fn request_timeout(&self) -> Duration {
        self.inner.request_timeout
    }

    pub fn configured_models(&self) -> Vec<String> {
        self.inner
            .engines
            .keys()
            .chain(self.inner.disabled.iter())
            .cloned()
            .collect()
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
        let actions = self.release_job(job_id).await?;
        self.apply_locked(actions).await
    }

    async fn release_job(&self, job_id: JobId) -> Result<Vec<Action>, RuntimeError> {
        let mut sched = self.inner.scheduler.lock().await;
        if !self.is_active(job_id).await {
            return Ok(Vec::new());
        }
        let actions = sched.finish(job_id);
        self.record_status(&sched);
        drop(sched);
        self.forget(job_id).await;
        Ok(actions)
    }

    async fn apply_locked(&self, actions: Vec<Action>) -> Result<(), RuntimeError> {
        if actions.is_empty() {
            return Ok(());
        }
        let sched = self.inner.scheduler.lock().await;
        let result = self.apply_all(actions).await;
        self.record_status(&sched);
        result
    }

    fn enqueue_apply(&self, actions: Vec<Action>) {
        if actions.is_empty() {
            return;
        }
        let _ = self.inner.apply_tx.send(actions);
    }

    async fn spawn_applier(&self) {
        let Some(rx) = self.inner.apply_rx.lock().await.take() else {
            return;
        };
        let weak = Arc::downgrade(&self.inner);
        tokio::spawn(async move {
            apply_loop(weak, rx).await;
        });
    }

    async fn is_active(&self, job_id: JobId) -> bool {
        self.inner.cancels.lock().await.contains_key(&job_id)
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
        let Some((prompt, tx)) = self.waiter(job_id).await else {
            return Ok(());
        };
        let token = self.watch(job_id).await;
        let engine = Arc::clone(self.engine(model)?);
        let rx = engine.run(&prompt, token.clone()).await?;
        let rt = self.clone();
        tokio::spawn(async move {
            forward_job(job_id, rx, tx, token, rt).await;
        });
        Ok(())
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
    rt: Runtime,
) {
    pump_tokens(rx, tx, token.clone()).await;
    if token.is_cancelled() {
        return;
    }
    if let Ok(actions) = rt.release_job(job_id).await {
        rt.enqueue_apply(actions);
    }
}

async fn apply_loop(weak: Weak<Inner>, mut rx: mpsc::UnboundedReceiver<Vec<Action>>) {
    while let Some(actions) = rx.recv().await {
        let Some(inner) = weak.upgrade() else {
            break;
        };
        let _ = Runtime { inner }.apply_locked(actions).await;
    }
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
