use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex as StdMutex, Weak};
use std::time::Duration;

use silkai_adapters::{
    ChatMessage, Engine, EngineError, FakeEngine, LlamaEngine, OllamaEngine, ProcessEngine,
    RunOptions, VllmEngine,
};
use silkai_sched::{
    Action, JobId, RejectReason, SchedError, Scheduler, StatusSnapshot, SubmitResult,
};

use crate::events::{Draft, EventLog};
use crate::metrics::{Counters, Metrics};
use crate::sampler::{self, Sample};
use crate::status::{self, ModelFacts, Status};
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
    #[error("engine unavailable")]
    Unavailable,
    #[error("websocket not enabled for model")]
    NoWebsocket,
    /// A load was abandoned because a higher-priority job took the card.
    /// The job is back in the queue; nothing is wrong.
    #[error("load interrupted")]
    Interrupted,
    #[error(transparent)]
    Engine(#[from] EngineError),
}

struct Waiter {
    model: String,
    messages: Vec<ChatMessage>,
    opts: RunOptions,
    emitted: String,
    tx: mpsc::Sender<String>,
}

struct Inner {
    scheduler: Mutex<Scheduler>,
    snapshot: StdMutex<StatusSnapshot>,
    engines: HashMap<String, Arc<dyn Engine>>,
    paths: HashMap<String, String>,
    disabled: HashSet<String>,
    unavailable: HashSet<String>,
    waiters: Mutex<HashMap<JobId, Waiter>>,
    cancels: Mutex<HashMap<JobId, CancellationToken>>,
    apply_tx: mpsc::UnboundedSender<Vec<Action>>,
    apply_rx: Mutex<Option<mpsc::UnboundedReceiver<Vec<Action>>>>,
    /// Why a job ended without tokens; read once by the HTTP handler.
    rejections: StdMutex<HashMap<JobId, String>>,
    /// One token per load or wake in flight, keyed by model. A preempt of
    /// the job that wanted the model fires it.
    loads: StdMutex<HashMap<String, CancellationToken>>,
    request_timeout: Duration,
    transports: HashMap<String, String>,
    idle_timeouts: HashMap<String, Duration>,
    /// Open session sockets and the model each one holds.
    sessions: StdMutex<HashMap<JobId, String>>,
    models: HashMap<String, ConfiguredModel>,
    /// `loading` or `sleeping` while an engine call is in flight.
    overlays: StdMutex<HashMap<String, &'static str>>,
    sample: sampler::Shared,
    events: Arc<EventLog>,
    metrics: Metrics,
}

impl Inner {
    fn new(scheduler: Scheduler, cfg: &AppConfig, events: Arc<EventLog>) -> Self {
        let (apply_tx, apply_rx) = mpsc::unbounded_channel();
        Self {
            snapshot: StdMutex::new(scheduler.status()),
            scheduler: Mutex::new(scheduler),
            engines: engines_for(&cfg.enabled),
            paths: paths_for(&cfg.enabled),
            disabled: disabled_set(&cfg.disabled),
            unavailable: unavailable_set(&cfg.enabled),
            waiters: Mutex::new(HashMap::new()),
            cancels: Mutex::new(HashMap::new()),
            apply_tx,
            apply_rx: Mutex::new(Some(apply_rx)),
            rejections: StdMutex::new(HashMap::new()),
            loads: StdMutex::new(HashMap::new()),
            request_timeout: cfg.request_timeout,
            transports: transports_for(&cfg.enabled),
            idle_timeouts: idle_timeouts_for(&cfg.enabled),
            sessions: StdMutex::new(HashMap::new()),
            models: cfg
                .enabled
                .iter()
                .map(|m| (m.spec.name.clone(), m.clone()))
                .collect(),
            overlays: StdMutex::new(HashMap::new()),
            sample: sampler::start(),
            events,
            metrics: Metrics::default(),
        }
    }
}

#[derive(Clone)]
pub struct Runtime {
    inner: Arc<Inner>,
}

impl Runtime {
    pub async fn new(cfg: AppConfig) -> Result<Self, RuntimeError> {
        Self::with_events(cfg, Arc::new(EventLog::new())).await
    }

    /// Build on an existing event log, so a reload keeps the history.
    pub async fn with_events(cfg: AppConfig, events: Arc<EventLog>) -> Result<Self, RuntimeError> {
        let (rt, warmup) = Self::assemble(cfg, events)?;
        rt.spawn_applier().await;
        rt.apply_all(warmup).await?;
        Ok(rt)
    }

    /// The reason a job produced no tokens, if it was refused. Consumed.
    pub fn take_rejection(&self, job_id: JobId) -> Option<String> {
        self.inner
            .rejections
            .lock()
            .expect("rejections mutex")
            .remove(&job_id)
    }

    pub fn events(&self) -> Arc<EventLog> {
        Arc::clone(&self.inner.events)
    }

    pub fn counters(&self) -> HashMap<String, Counters> {
        self.inner.metrics.snapshot()
    }

    fn emit(&self, draft: Draft) {
        self.inner.events.emit(draft);
    }

    pub async fn submit_chat(
        &self,
        model: &str,
        messages: Vec<ChatMessage>,
        opts: RunOptions,
    ) -> Result<(JobId, mpsc::Receiver<String>), RuntimeError> {
        self.ensure_enabled(model)?;
        self.ensure_available(model)?;
        let (tx, rx) = mpsc::channel(16);
        let job_id = self.accept(model, messages, opts, tx).await?;
        Ok((job_id, rx))
    }

    pub async fn finished(&self, job_id: JobId) {
        self.complete(job_id).await.expect("runtime finish actions");
    }

    pub fn allows_websocket(&self, model: &str) -> bool {
        matches!(
            self.inner
                .transports
                .get(model)
                .map(|s| s.to_ascii_lowercase())
                .as_deref(),
            Some("websocket") | Some("both")
        )
    }

    pub fn idle_timeout(&self, model: &str) -> Duration {
        self.inner
            .idle_timeouts
            .get(model)
            .copied()
            .unwrap_or(Duration::from_secs(45))
    }

    pub async fn begin_session(&self, model: &str) -> Result<JobId, RuntimeError> {
        self.ensure_enabled(model)?;
        self.ensure_available(model)?;
        if !self.allows_websocket(model) {
            return Err(RuntimeError::NoWebsocket);
        }
        let mut sched = self.inner.scheduler.lock().await;
        let (job_id, actions) = match sched.submit(model) {
            SubmitResult::Accepted { job_id, actions } => (job_id, actions),
            SubmitResult::Rejected { reason } => return Err(reject_err(reason)),
        };
        self.sessions().insert(job_id, model.to_string());
        self.record_status(&sched);
        drop(sched);
        self.emit(Draft::new("session_open").model(model).job(job_id.0));
        match self.apply_all(actions).await {
            Ok(()) | Err(RuntimeError::Interrupted) => {}
            Err(err) => {
                self.sessions().remove(&job_id);
                let mut sched = self.inner.scheduler.lock().await;
                self.isolate(&mut sched, job_id, Some(model), &err).await;
                return Err(err);
            }
        }
        let sched = self.inner.scheduler.lock().await;
        self.record_status(&sched);
        drop(sched);
        self.wait_until_running(job_id).await?;
        Ok(job_id)
    }

    pub async fn session_prompt(
        &self,
        job_id: JobId,
        model: &str,
        messages: &[ChatMessage],
        opts: &RunOptions,
    ) -> Result<mpsc::Receiver<String>, RuntimeError> {
        if !self.sessions().contains_key(&job_id) {
            return Err(RuntimeError::Unknown);
        }
        if let Some(old) = self.inner.cancels.lock().await.remove(&job_id) {
            old.cancel();
        }
        let token = self.watch(job_id).await;
        let engine = Arc::clone(self.engine(model)?);
        engine
            .run(messages, "", opts, token)
            .await
            .map_err(engine_err)
    }

    async fn wait_until_running(&self, job_id: JobId) -> Result<(), RuntimeError> {
        let deadline = tokio::time::Instant::now() + self.inner.request_timeout;
        loop {
            {
                let sched = self.inner.scheduler.lock().await;
                if sched.job_running(job_id) {
                    return Ok(());
                }
            }
            if tokio::time::Instant::now() >= deadline {
                self.end_session(job_id).await;
                return Err(RuntimeError::Unavailable);
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    pub async fn end_session(&self, job_id: JobId) {
        if let Some(model) = self.sessions().remove(&job_id) {
            self.emit(Draft::new("session_close").model(model).job(job_id.0));
        }
        self.drop_job(job_id).await;
    }

    pub async fn drop_job(&self, job_id: JobId) {
        self.forget(job_id).await;
        let actions = {
            let mut sched = self.inner.scheduler.lock().await;
            let actions = sched.drop_job(job_id);
            self.record_status(&sched);
            actions
        };
        self.enqueue_apply(actions);
    }

    pub fn status(&self) -> Status {
        let snap = self.inner.snapshot.lock().expect("status mutex").clone();
        let overlays = self.inner.overlays.lock().expect("overlay mutex").clone();
        let sessions = self.session_counts();
        let sample = self.inner.sample.lock().expect("sampler mutex").clone();
        status::assemble(
            &snap,
            |name| self.facts(name, &overlays, &sessions),
            sample.as_ref(),
        )
    }

    /// Replace the card sample, for tests that have no nvidia-smi.
    #[doc(hidden)]
    pub fn inject_sample(&self, sample: Option<Sample>) {
        *self.inner.sample.lock().expect("sampler mutex") = sample;
    }

    fn sessions(&self) -> std::sync::MutexGuard<'_, HashMap<JobId, String>> {
        self.inner.sessions.lock().expect("sessions mutex")
    }

    fn session_counts(&self) -> HashMap<String, u32> {
        let mut counts = HashMap::new();
        for model in self.sessions().values() {
            *counts.entry(model.clone()).or_insert(0) += 1;
        }
        counts
    }

    fn facts(
        &self,
        name: &str,
        overlays: &HashMap<String, &'static str>,
        sessions: &HashMap<String, u32>,
    ) -> Option<ModelFacts> {
        let m = self.inner.models.get(name)?;
        let engine = self.inner.engines.get(name)?;
        Some(ModelFacts {
            engine: m.engine.clone(),
            budget_gb: m.spec.vram_gb,
            ram_gb: m.spec.ram_gb,
            priority: m.spec.priority,
            exclusive: m.spec.exclusive,
            slots: m.spec.slots,
            overlay: overlays.get(name).copied(),
            sessions: sessions.get(name).copied().unwrap_or(0),
            has_shelf: engine.has_shelf(),
            pid: engine.pid(),
        })
    }

    fn set_overlay(&self, model: &str, state: &'static str) {
        self.inner
            .overlays
            .lock()
            .expect("overlay mutex")
            .insert(model.to_string(), state);
    }

    fn clear_overlay(&self, model: &str) {
        self.inner
            .overlays
            .lock()
            .expect("overlay mutex")
            .remove(model);
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
    fn assemble(
        cfg: AppConfig,
        events: Arc<EventLog>,
    ) -> Result<(Self, Vec<Action>), RuntimeError> {
        let specs = cfg
            .enabled
            .iter()
            .filter(|m| known_engine(&m.engine))
            .map(|m| m.spec.clone())
            .collect();
        let mut scheduler = Scheduler::new(cfg.resources.clone(), specs).map_err(sched_err)?;
        let warmup = prefetch_actions(&mut scheduler, cfg.prefetch_on_start);
        let rt = Self {
            inner: Arc::new(Inner::new(scheduler, &cfg, events)),
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

    fn ensure_available(&self, model: &str) -> Result<(), RuntimeError> {
        if self.inner.unavailable.contains(model) {
            Err(RuntimeError::Unavailable)
        } else {
            Ok(())
        }
    }

    async fn accept(
        &self,
        model: &str,
        messages: Vec<ChatMessage>,
        opts: RunOptions,
        tx: mpsc::Sender<String>,
    ) -> Result<JobId, RuntimeError> {
        let (job_id, actions) = {
            let mut sched = self.inner.scheduler.lock().await;
            let (job_id, actions) = match sched.submit(model) {
                SubmitResult::Accepted { job_id, actions } => (job_id, actions),
                SubmitResult::Rejected { reason } => return Err(reject_err(reason)),
            };
            self.store_waiter(job_id, model, messages, opts, tx).await;
            // The scheduler has already moved models in its own books;
            // publish that now so status is truthful while the engines
            // catch up.
            self.record_status(&sched);
            (job_id, actions)
        };
        // Engines run with the scheduler unlocked, so a live request can
        // get in while a long load is under way and preempt it.
        match self.apply_all(actions).await {
            Ok(()) => {}
            // The preemptor already re-queued this job; nothing to undo.
            Err(RuntimeError::Interrupted) => {}
            Err(err) => {
                let mut sched = self.inner.scheduler.lock().await;
                self.isolate(&mut sched, job_id, Some(model), &err).await;
                return Err(err);
            }
        }
        let sched = self.inner.scheduler.lock().await;
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
        let model = self.waiter_model(job_id).await;
        self.forget(job_id).await;
        let mut draft = Draft::new("finish").job(job_id.0);
        if let Some(model) = model {
            draft = draft.model(model);
        }
        self.emit(draft);
        Ok(actions)
    }

    async fn apply_locked(&self, actions: Vec<Action>) -> Result<(), RuntimeError> {
        if actions.is_empty() {
            return Ok(());
        }
        let sched = self.inner.scheduler.lock().await;
        self.record_status(&sched);
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

    async fn store_waiter(
        &self,
        job_id: JobId,
        model: &str,
        messages: Vec<ChatMessage>,
        opts: RunOptions,
        tx: mpsc::Sender<String>,
    ) {
        let waiter = Waiter {
            model: model.to_string(),
            messages,
            opts,
            emitted: String::new(),
            tx,
        };
        self.inner.waiters.lock().await.insert(job_id, waiter);
    }

    async fn waiter(
        &self,
        job_id: JobId,
    ) -> Option<(Vec<ChatMessage>, String, RunOptions, mpsc::Sender<String>)> {
        self.inner.waiters.lock().await.get(&job_id).map(|w| {
            (
                w.messages.clone(),
                w.emitted.clone(),
                w.opts.clone(),
                w.tx.clone(),
            )
        })
    }

    async fn waiter_model(&self, job_id: JobId) -> Option<String> {
        self.inner
            .waiters
            .lock()
            .await
            .get(&job_id)
            .map(|w| w.model.clone())
    }

    async fn append_emitted(&self, job_id: JobId, chunk: &str) {
        if let Some(w) = self.inner.waiters.lock().await.get_mut(&job_id) {
            w.emitted.push_str(chunk);
        }
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

    async fn isolate(
        &self,
        sched: &mut Scheduler,
        job_id: JobId,
        model: Option<&str>,
        err: &RuntimeError,
    ) {
        let mut draft = Draft::new("fault").job(job_id.0).error(err.to_string());
        if let Some(model) = model {
            draft = draft.model(model);
            self.inner.metrics.bump(model, |c| c.faults += 1);
        }
        self.emit(draft);
        let actions = sched.fault(job_id);
        self.record_status(sched);
        self.forget(job_id).await;
        let _ = self.apply_all(actions).await;
    }

    async fn apply_all(&self, actions: Vec<Action>) -> Result<(), RuntimeError> {
        // A model booked for the card is "loading" from the moment the batch
        // is decided, not only once its own engine call begins (it may be
        // waiting for another model to leave first).
        for action in &actions {
            if let Action::Load { model, .. } | Action::Wake { model, .. } = action {
                self.set_overlay(model, "loading");
            }
        }
        for action in actions {
            self.apply(action).await?;
        }
        Ok(())
    }

    async fn apply(&self, action: Action) -> Result<(), RuntimeError> {
        let started = std::time::Instant::now();
        let result = match &action {
            Action::Warm { model } => self.warm(model).await,
            Action::Load { model, gpu } => self.load(model, *gpu).await,
            Action::Wake { model, gpu } => self.wake(model, *gpu).await,
            Action::Sleep { model } => self.sleep(model).await,
            Action::Discard { model } => self.discard(model).await,
            Action::Preempt { job_id } => self.preempt(*job_id).await,
            Action::Start { job_id, model } => self.start(*job_id, model).await,
        };
        self.record_action(&action, started.elapsed(), &result)
            .await;
        result
    }

    async fn record_action(
        &self,
        action: &Action,
        took: Duration,
        result: &Result<(), RuntimeError>,
    ) {
        let ms = took.as_millis() as u64;
        let secs = took.as_secs_f64();
        let mut draft = match action {
            Action::Warm { model } => Draft::new("warm").model(model),
            Action::Load { model, gpu } => {
                self.inner.metrics.bump(model, |c| {
                    c.loads += 1;
                    c.load_secs += secs;
                });
                Draft::new("load").model(model).gpu(*gpu).ms(ms)
            }
            Action::Wake { model, gpu } => {
                self.inner.metrics.bump(model, |c| {
                    c.wakes += 1;
                    c.load_secs += secs;
                });
                Draft::new("wake").model(model).gpu(*gpu).ms(ms)
            }
            Action::Sleep { model } => {
                self.inner.metrics.bump(model, |c| c.sleeps += 1);
                Draft::new("sleep").model(model).ms(ms)
            }
            Action::Discard { model } => Draft::new("discard").model(model),
            Action::Preempt { job_id } => {
                let mut d = Draft::new("preempt").job(job_id.0);
                if let Some(model) = self.waiter_model(*job_id).await {
                    self.inner.metrics.bump(&model, |c| c.preempts += 1);
                    d = d.model(model);
                }
                d
            }
            Action::Start { job_id, model } => Draft::new("start").model(model).job(job_id.0),
        };
        if let Err(err) = result {
            draft = draft.error(err.to_string());
        }
        self.emit(draft);
    }

    async fn warm(&self, model: &str) -> Result<(), RuntimeError> {
        let path = self.path(model)?;
        self.engine(model)?.warm(path).await.map_err(engine_err)
    }

    async fn load(&self, model: &str, gpu: u32) -> Result<(), RuntimeError> {
        let path = self.path(model)?.to_string();
        let engine = Arc::clone(self.engine(model)?);
        self.interruptible(model, engine.clone(), async move {
            engine.load(&path, gpu).await
        })
        .await
    }

    async fn wake(&self, model: &str, gpu: u32) -> Result<(), RuntimeError> {
        let engine = Arc::clone(self.engine(model)?);
        self.interruptible(model, engine.clone(), async move { engine.wake(gpu).await })
            .await
    }

    /// Run a load or wake that a preempt can abandon. On interruption the
    /// engine is told to sleep (a child is killed, an HTTP server is put
    /// to sleep), the original call is allowed to finish so nothing is left
    /// half-placed, and it is slept once more in case it landed anyway.
    async fn interruptible(
        &self,
        model: &str,
        engine: Arc<dyn Engine>,
        call: impl std::future::Future<Output = Result<(), EngineError>>,
    ) -> Result<(), RuntimeError> {
        let token = CancellationToken::new();
        self.inner
            .loads
            .lock()
            .expect("loads mutex")
            .insert(model.to_string(), token.clone());
        self.set_overlay(model, "loading");
        tokio::pin!(call);
        let result = tokio::select! {
            r = &mut call => match r {
                // The call finished, but if the preempt already fired the
                // model no longer belongs on the card: undo the landing.
                Ok(()) if token.is_cancelled() => {
                    let _ = engine.sleep().await;
                    Err(RuntimeError::Interrupted)
                }
                r => r.map_err(engine_err),
            },
            _ = token.cancelled() => {
                let _ = engine.sleep().await;
                let landed = call.await.is_ok();
                if landed {
                    let _ = engine.sleep().await;
                }
                Err(RuntimeError::Interrupted)
            }
        };
        self.inner.loads.lock().expect("loads mutex").remove(model);
        self.clear_overlay(model);
        result
    }

    /// Fire the token of a load in flight for `model`, if any.
    fn interrupt_load(&self, model: &str) {
        if let Some(token) = self.inner.loads.lock().expect("loads mutex").get(model) {
            token.cancel();
        }
    }

    async fn sleep(&self, model: &str) -> Result<(), RuntimeError> {
        let engine = self.engine(model)?;
        self.set_overlay(model, "sleeping");
        let result = engine.sleep().await.map_err(engine_err);
        self.clear_overlay(model);
        result
    }

    async fn discard(&self, model: &str) -> Result<(), RuntimeError> {
        self.engine(model)?.discard().await.map_err(engine_err)
    }

    async fn preempt(&self, job_id: JobId) -> Result<(), RuntimeError> {
        if let Some(token) = self.inner.cancels.lock().await.remove(&job_id) {
            token.cancel();
        }
        // The job may not be generating yet: its model could still be on
        // its way to the card. Abandon that load; the job is re-queued.
        if let Some(model) = self.waiter_model(job_id).await {
            self.interrupt_load(&model);
        }
        Ok(())
    }

    async fn start(&self, job_id: JobId, model: &str) -> Result<(), RuntimeError> {
        let Some((messages, prefix, opts, tx)) = self.waiter(job_id).await else {
            return Ok(());
        };
        let token = self.watch(job_id).await;
        let engine = Arc::clone(self.engine(model)?);
        let rx = match engine.run(&messages, &prefix, &opts, token.clone()).await {
            Ok(rx) => rx,
            Err(EngineError::Rejected(reason)) => {
                self.reject(job_id, model, reason);
                return Ok(());
            }
            Err(err) => return Err(engine_err(err)),
        };
        let rt = self.clone();
        tokio::spawn(async move {
            forward_job(job_id, rx, tx, token, rt).await;
        });
        Ok(())
    }

    /// The engine refused this request but is healthy: record why, and end
    /// the job as if it had finished so the model stays resident. The
    /// scheduler lock may be held by our caller, so the finish runs after.
    fn reject(&self, job_id: JobId, model: &str, reason: String) {
        self.emit(
            Draft::new("reject")
                .model(model)
                .job(job_id.0)
                .error(reason.clone()),
        );
        self.inner
            .rejections
            .lock()
            .expect("rejections mutex")
            .insert(job_id, reason);
        let rt = self.clone();
        tokio::spawn(async move {
            if let Ok(actions) = rt.release_job(job_id).await {
                rt.enqueue_apply(actions);
            }
        });
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
    pump_tokens(job_id, rx, tx, token.clone(), &rt).await;
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
    job_id: JobId,
    mut rx: mpsc::Receiver<String>,
    tx: mpsc::Sender<String>,
    token: CancellationToken,
    rt: &Runtime,
) {
    loop {
        tokio::select! {
            _ = token.cancelled() => return,
            next = rx.recv() => match next {
                Some(chunk) => {
                    rt.append_emitted(job_id, &chunk).await;
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
        .filter_map(|m| Some((m.spec.name.clone(), arc_engine(m)?)))
        .collect()
}

fn arc_engine(model: &ConfiguredModel) -> Option<Arc<dyn Engine>> {
    match model.engine.as_str() {
        "fake" => Some(Arc::new(FakeEngine::new(
            &model.spec.name,
            model.spec.vram_gb,
        ))),
        "llama.cpp" => Some(llama_engine(model)),
        "vllm" => Some(vllm_engine(model)),
        "ollama" => Some(ollama_engine(model)),
        "process" => Some(process_engine(model)),
        _ => None,
    }
}

fn known_engine(engine: &str) -> bool {
    matches!(engine, "fake" | "llama.cpp" | "vllm" | "ollama" | "process")
}

fn transports_for(models: &[ConfiguredModel]) -> HashMap<String, String> {
    models
        .iter()
        .map(|m| (m.spec.name.clone(), m.transport.clone()))
        .collect()
}

fn idle_timeouts_for(models: &[ConfiguredModel]) -> HashMap<String, Duration> {
    models
        .iter()
        .filter_map(|m| {
            m.idle_timeout_secs
                .map(|s| (m.spec.name.clone(), Duration::from_secs(s)))
        })
        .collect()
}

fn llama_engine(model: &ConfiguredModel) -> Arc<dyn Engine> {
    warn_missing_llama();
    Arc::new(LlamaEngine::new(
        &model.spec.name,
        model.spec.vram_gb,
        model.ctx_size,
    ))
}

fn vllm_engine(model: &ConfiguredModel) -> Arc<dyn Engine> {
    let url = model
        .url
        .clone()
        .unwrap_or_else(|| "http://127.0.0.1:8000".into());
    Arc::new(VllmEngine::new(&model.spec.name, model.spec.vram_gb, url))
}

fn ollama_engine(model: &ConfiguredModel) -> Arc<dyn Engine> {
    let url = model
        .url
        .clone()
        .unwrap_or_else(|| "http://127.0.0.1:11434".into());
    Arc::new(OllamaEngine::new(&model.spec.name, model.spec.vram_gb, url))
}

fn process_engine(model: &ConfiguredModel) -> Arc<dyn Engine> {
    let url = model
        .url
        .clone()
        .unwrap_or_else(|| "http://127.0.0.1:8000".into());
    Arc::new(ProcessEngine::new(
        &model.spec.name,
        model.spec.vram_gb,
        url,
        model.cmd.clone(),
    ))
}

fn warn_missing_llama() {
    if cfg!(feature = "llama") {
        return;
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        eprintln!("llama.cpp requested but built without feature llama");
    });
}

fn unavailable_set(models: &[ConfiguredModel]) -> HashSet<String> {
    models
        .iter()
        .filter(|m| !engine_available(&m.engine))
        .map(|m| m.spec.name.clone())
        .collect()
}

fn engine_available(engine: &str) -> bool {
    match engine {
        "fake" | "vllm" | "ollama" | "process" => true,
        "llama.cpp" => cfg!(feature = "llama"),
        _ => false,
    }
}

fn engine_err(err: EngineError) -> RuntimeError {
    match err {
        EngineError::Other(msg) if is_unavailable(&msg) => RuntimeError::Unavailable,
        other => RuntimeError::Engine(other),
    }
}

fn is_unavailable(msg: &str) -> bool {
    msg.contains("built without feature llama") || msg.contains("native library not linked")
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
