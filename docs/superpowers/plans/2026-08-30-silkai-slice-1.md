# SilkAI Slice 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a Linux-first local daemon that packs models onto one GPU by VRAM, priority, exclusive vs share, and slots; parks unused weights in RAM; and serves OpenAI-shaped HTTP chat completions.

**Architecture:** `silkai-sched` is a pure scheduler (numeric GB, no GPU). `silkai-adapters` executes Load/Wake/Sleep/Discard/Start/Preempt. `silkai-server` loads TOML, serves HTTP, and applies scheduler actions. The `silkai` binary is the process. Slice 1 has a `fake` adapter (always, for tests) and a `llama.cpp` adapter (optional feature). No WebSocket, no whisper.cpp (slice 2).

**Tech Stack:** Rust 2021 workspace, Tokio, Axum 0.8, Serde, toml, llama-cpp-2 behind feature `llama`. Tests: `cargo test`. GPU integration: `SILKAI_ITEST=1`.

**Spec:** `docs/superpowers/specs/2026-08-30-silkai-gpu-scheduler-design.md`

---

## File structure

```
LICENSE
README.md
CONTRIBUTING.md
.gitignore
Cargo.toml                          workspace
crates/silkai-sched/Cargo.toml
crates/silkai-sched/src/lib.rs      re-exports
crates/silkai-sched/src/types.rs    Priority, Tier, JobId, ModelSpec, Resources, Action
crates/silkai-sched/src/scheduler.rs
crates/silkai-sched/src/clinic.rs   example 32/128 GB fixture used by tests
crates/silkai-adapters/Cargo.toml
crates/silkai-adapters/src/lib.rs   Engine trait
crates/silkai-adapters/src/fake.rs
crates/silkai-adapters/src/llama.rs feature = "llama"
crates/silkai-server/Cargo.toml
crates/silkai-server/src/lib.rs
crates/silkai-server/src/config.rs
crates/silkai-server/src/app.rs     axum router + state
crates/silkai-server/src/runtime.rs scheduler + engines
crates/silkai/Cargo.toml
crates/silkai/src/main.rs
examples/config.toml
```

Canonical types (do not rename later):

```rust
Priority { Background, Normal, Live }  // Background < Normal < Live
Tier { Cupboard, Shelf, Bench }
JobId(u64)
Resources { gpu_schedulable_gb: f64, ram_shelf_gb: f64 }
ModelSpec { name, vram_gb, ram_gb, priority, exclusive, slots, keep_warm }
Action::Load { model } | Wake { model } | Sleep { model } | Discard { model }
         | Warm { model } | Start { job_id, model } | Preempt { job_id }
RejectReason::UnknownModel | TooLarge
SubmitResult::Rejected { reason } | Accepted { job_id, actions }
Scheduler::new(resources, models)
Scheduler::submit(&mut self, model: &str) -> SubmitResult
Scheduler::finish(&mut self, job_id: JobId) -> Vec<Action>
Scheduler::prefetch(&mut self) -> Vec<Action>   // Warm keep_warm models; does not Bench
Scheduler::tier(&self, model: &str) -> Tier
Scheduler::running(&self, model: &str) -> u32
Scheduler::queued(&self, model: &str) -> u32
Scheduler::status(&self) -> StatusSnapshot
```

`submit` always accepts unknown vs too-large as `Rejected`. Everything else is `Accepted` (job may be queued: no `Start` in `actions`). Scheduler updates tiers **optimistically** when it emits Load/Wake/Sleep/Discard/Warm; the runtime runs actions in order before `Start`.

Clinic fixture (32 GB GPU − 3 headroom = 29; 128 RAM − 32 = 96):

- `whisper`: 12 GB, Live, share, slots=2, keep_warm
- `soap`: 28 GB, Normal, exclusive, slots=1, keep_warm
- `chart-scan`: 10 GB, Background, share, slots=1, keep_warm

TDD: every production function is preceded by a failing test. Config files and LICENSE are exempt. Commit after each task. If git is not initialized, `git init` once in Task 1.

---

### Task 1: Workspace, license, git

**Files:**
- Create: `LICENSE`
- Create: `.gitignore`
- Create: `Cargo.toml`
- Create: `crates/silkai-sched/Cargo.toml`
- Create: `crates/silkai-sched/src/lib.rs`

- [ ] **Step 1: Initialize git if needed**

```bash
cd /Users/acp/silkai
git init
```

- [ ] **Step 2: Write LICENSE (MIT)**

```
MIT License

Copyright (c) 2026 SilkAI contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

- [ ] **Step 3: Write `.gitignore`**

```
/target
**/*.rs.bk
.DS_Store
```

- [ ] **Step 4: Write workspace manifests**

Root `Cargo.toml`:

```toml
[workspace]
resolver = "2"
members = ["crates/silkai-sched"]

[workspace.package]
version = "0.1.0"
edition = "2021"
license = "MIT"
repository = "https://github.com/silkai/silkai"
```

`crates/silkai-sched/Cargo.toml`:

```toml
[package]
name = "silkai-sched"
version.workspace = true
edition.workspace = true
license.workspace = true
```

`crates/silkai-sched/src/lib.rs`:

```rust
//! GPU capacity scheduler. No GPU types — only GB numbers.
```

- [ ] **Step 5: Verify it builds**

Run: `cargo test -p silkai-sched`
Expected: compile succeeds, 0 tests.

- [ ] **Step 6: Commit**

```bash
git add LICENSE .gitignore Cargo.toml crates/silkai-sched
git commit -m "chore: MIT workspace and silkai-sched crate"
```

---

### Task 2: Types and clinic fixture

**Files:**
- Create: `crates/silkai-sched/src/types.rs`
- Create: `crates/silkai-sched/src/clinic.rs`
- Modify: `crates/silkai-sched/src/lib.rs`

- [ ] **Step 1: Write the failing test** in `crates/silkai-sched/src/clinic.rs` (tests at the bottom of the module) — actually put tests in `crates/silkai-sched/src/lib.rs` under `#[cfg(test)]` until modules exist. Prefer `crates/silkai-sched/tests/clinic.rs`:

```rust
use silkai_sched::clinic::{clinic_models, clinic_resources};
use silkai_sched::{Priority, Resources};

#[test]
fn clinic_budget_is_29_gb_gpu_and_96_gb_shelf() {
    let r: Resources = clinic_resources();
    assert_eq!(r.gpu_schedulable_gb, 29.0);
    assert_eq!(r.ram_shelf_gb, 96.0);
}

#[test]
fn clinic_models_match_spec() {
    let models = clinic_models();
    assert_eq!(models.len(), 3);
    let w = models.iter().find(|m| m.name == "whisper").unwrap();
    assert_eq!(w.vram_gb, 12.0);
    assert_eq!(w.priority, Priority::Live);
    assert!(!w.exclusive);
    assert_eq!(w.slots, 2);
    assert!(w.keep_warm);
    let s = models.iter().find(|m| m.name == "soap").unwrap();
    assert_eq!(s.vram_gb, 28.0);
    assert_eq!(s.priority, Priority::Normal);
    assert!(s.exclusive);
    assert_eq!(s.slots, 1);
    let c = models.iter().find(|m| m.name == "chart-scan").unwrap();
    assert_eq!(c.vram_gb, 10.0);
    assert_eq!(c.priority, Priority::Background);
}

#[test]
fn live_outranks_normal_outranks_background() {
    assert!(Priority::Background < Priority::Normal);
    assert!(Priority::Normal < Priority::Live);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p silkai-sched --test clinic`
Expected: FAIL compiling (`clinic` module does not exist).

- [ ] **Step 3: Write types, clinic, exports**

`crates/silkai-sched/src/types.rs`:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Priority {
    Background,
    Normal,
    Live,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    Cupboard,
    Shelf,
    Bench,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct JobId(pub u64);

#[derive(Debug, Clone, PartialEq)]
pub struct Resources {
    pub gpu_schedulable_gb: f64,
    pub ram_shelf_gb: f64,
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

#[derive(Debug, Clone, PartialEq)]
pub struct ModelStatus {
    pub name: String,
    pub tier: Tier,
    pub running: u32,
    pub queued: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StatusSnapshot {
    pub models: Vec<ModelStatus>,
    pub gpu_used_gb: f64,
    pub ram_used_gb: f64,
}
```

`crates/silkai-sched/src/clinic.rs`:

```rust
use crate::types::{ModelSpec, Priority, Resources};

pub fn clinic_resources() -> Resources {
    Resources {
        gpu_schedulable_gb: 29.0,
        ram_shelf_gb: 96.0,
    }
}

pub fn clinic_models() -> Vec<ModelSpec> {
    vec![
        ModelSpec {
            name: "whisper".into(),
            vram_gb: 12.0,
            ram_gb: 12.0,
            priority: Priority::Live,
            exclusive: false,
            slots: 2,
            keep_warm: true,
        },
        ModelSpec {
            name: "soap".into(),
            vram_gb: 28.0,
            ram_gb: 28.0,
            priority: Priority::Normal,
            exclusive: true,
            slots: 1,
            keep_warm: true,
        },
        ModelSpec {
            name: "chart-scan".into(),
            vram_gb: 10.0,
            ram_gb: 10.0,
            priority: Priority::Background,
            exclusive: false,
            slots: 1,
            keep_warm: true,
        },
    ]
}
```

`crates/silkai-sched/src/lib.rs`:

```rust
//! GPU capacity scheduler. No GPU types — only GB numbers.

pub mod clinic;
pub mod types;

pub use types::*;
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p silkai-sched --test clinic`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add crates/silkai-sched
git commit -m "feat(sched): types and clinic fixture"
```

---

### Task 3: Packing — 12+10 fit, 28 does not sit beside them

**Files:**
- Create: `crates/silkai-sched/src/scheduler.rs`
- Create: `crates/silkai-sched/tests/packing.rs`
- Modify: `crates/silkai-sched/src/lib.rs`

- [ ] **Step 1: Write the failing test**

`crates/silkai-sched/tests/packing.rs`:

```rust
use silkai_sched::clinic::{clinic_models, clinic_resources};
use silkai_sched::{Action, Scheduler, SubmitResult, Tier};

fn sched() -> Scheduler {
    Scheduler::new(clinic_resources(), clinic_models()).unwrap()
}

fn started(result: &SubmitResult) -> bool {
    match result {
        SubmitResult::Accepted { actions, .. } => {
            actions.iter().any(|a| matches!(a, Action::Start { .. }))
        }
        _ => false,
    }
}

#[test]
fn whisper_then_chart_scan_both_run() {
    let mut s = sched();
    let a = s.submit("whisper");
    assert!(started(&a));
    assert_eq!(s.tier("whisper"), Tier::Bench);
    let b = s.submit("chart-scan");
    assert!(started(&b));
    assert_eq!(s.tier("chart-scan"), Tier::Bench);
    assert_eq!(s.gpu_used_gb(), 22.0);
}

#[test]
fn soap_does_not_start_beside_whisper_and_scan() {
    let mut s = sched();
    s.submit("whisper");
    s.submit("chart-scan");
    let r = s.submit("soap");
    assert!(matches!(r, SubmitResult::Accepted { .. }));
    assert!(!started(&r));
    assert_eq!(s.tier("soap"), Tier::Cupboard);
    assert_eq!(s.queued("soap"), 1);
}

#[test]
fn unknown_model_is_rejected() {
    let mut s = sched();
    let r = s.submit("nope");
    assert!(matches!(
        r,
        SubmitResult::Rejected {
            reason: silkai_sched::RejectReason::UnknownModel
        }
    ));
}

#[test]
fn model_bigger_than_gpu_is_rejected() {
    let mut s = Scheduler::new(
        clinic_resources(),
        vec![silkai_sched::ModelSpec {
            name: "huge".into(),
            vram_gb: 40.0,
            ram_gb: 40.0,
            priority: silkai_sched::Priority::Normal,
            exclusive: true,
            slots: 1,
            keep_warm: true,
        }],
    )
    .unwrap();
    let r = s.submit("huge");
    assert!(matches!(
        r,
        SubmitResult::Rejected {
            reason: silkai_sched::RejectReason::TooLarge
        }
    ));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p silkai-sched --test packing`
Expected: FAIL (`Scheduler` not found).

- [ ] **Step 3: Minimal scheduler that packs and queues**

Implement `crates/silkai-sched/src/scheduler.rs` with:

- `HashMap<String, ModelSpec>`
- `HashMap<String, Tier>` default Cupboard
- `HashMap<String, u32>` running counts
- `VecDeque` of queued `(JobId, String)`
- `next_id: u64`
- `gpu_used_gb()` = sum of `vram_gb` of models on `Tier::Bench`
- `submit`: unknown → Rejected UnknownModel; `vram_gb > gpu_schedulable` → TooLarge; else allocate JobId, try `can_place(model)` without eviction yet. If already Bench and running < slots → Start. If can add to bench (`used + vram <= schedulable`, not blocked by exclusive rules): emit Load (from Cupboard) or Wake (from Shelf), set Bench, running += 1, Start. Else push queue, Accepted with empty/no Start.
- Exclusive rule: cannot place exclusive if any other model is Bench; cannot place non-exclusive if any exclusive is Bench.
- `can_place` for this task: no eviction/preempt yet. SOAP beside 12+10 fails exclusive (others on bench) AND size.

Export `Scheduler` from `lib.rs`: `pub mod scheduler; pub use scheduler::Scheduler;`

Also implement `tier`, `running`, `queued`, `gpu_used_gb` as methods on `Scheduler`.

`new` returns `Result<Scheduler, ()>` — use `Result<Self, SchedError>` with `SchedError::DuplicateModel`. Empty error type is fine: `pub type SchedError = String;` or a small enum:

```rust
#[derive(Debug)]
pub enum SchedError {
    DuplicateModel(String),
    InvalidSlots,
}
```

Reject `slots == 0` in `new`.

- [ ] **Step 4: Run tests**

Run: `cargo test -p silkai-sched --test packing`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/silkai-sched
git commit -m "feat(sched): pack models that fit, queue those that do not"
```

---

### Task 4: Slots — one load, many jobs

**Files:**
- Create: `crates/silkai-sched/tests/slots.rs`
- Modify: `crates/silkai-sched/src/scheduler.rs`

- [ ] **Step 1: Write the failing test**

```rust
use silkai_sched::clinic::{clinic_models, clinic_resources};
use silkai_sched::{Action, Scheduler, SubmitResult};

fn sched() -> Scheduler {
    Scheduler::new(clinic_resources(), clinic_models()).unwrap()
}

fn load_count(r: &SubmitResult) -> usize {
    match r {
        SubmitResult::Accepted { actions, .. } => actions
            .iter()
            .filter(|a| matches!(a, Action::Load { .. } | Action::Wake { .. }))
            .count(),
        _ => 0,
    }
}

fn started(r: &SubmitResult) -> bool {
    matches!(r, SubmitResult::Accepted { actions, .. } if actions.iter().any(|a| matches!(a, Action::Start { .. })))
}

#[test]
fn two_whisper_jobs_one_load() {
    let mut s = sched();
    let a = s.submit("whisper");
    assert_eq!(load_count(&a), 1);
    assert!(started(&a));
    let b = s.submit("whisper");
    assert_eq!(load_count(&b), 0);
    assert!(started(&b));
    assert_eq!(s.running("whisper"), 2);
}

#[test]
fn third_whisper_queues_at_two_slots() {
    let mut s = sched();
    s.submit("whisper");
    s.submit("whisper");
    let c = s.submit("whisper");
    assert!(started(&c) == false);
    assert_eq!(s.queued("whisper"), 1);
    assert_eq!(s.running("whisper"), 2);
}

#[test]
fn soap_second_job_queues_on_one_slot() {
    let mut s = sched();
    let a = s.submit("soap");
    assert!(started(&a));
    let b = s.submit("soap");
    assert!(!started(&b));
    assert_eq!(s.queued("soap"), 1);
    assert_eq!(s.running("soap"), 1);
    match (&a, &b) {
        (
            SubmitResult::Accepted { job_id: id_a, .. },
            SubmitResult::Accepted { job_id: id_b, .. },
        ) => assert_ne!(id_a, id_b),
        _ => panic!("both accepted"),
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p silkai-sched --test slots`
Expected: FAIL on third whisper or second soap if Task 3 always Start when model already on bench without checking slots.

- [ ] **Step 3: Enforce slots** — if Bench and `running >= slots`, queue (do not Load again). If Bench and `running < slots`, Start only.

- [ ] **Step 4: Run tests**

Run: `cargo test -p silkai-sched`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/silkai-sched
git commit -m "feat(sched): slot cap shares one resident copy"
```

---

### Task 5: finish() admits the queue

**Files:**
- Create: `crates/silkai-sched/tests/finish.rs`
- Modify: `crates/silkai-sched/src/scheduler.rs`

- [ ] **Step 1: Write the failing test**

```rust
use silkai_sched::clinic::{clinic_models, clinic_resources};
use silkai_sched::{Action, JobId, Scheduler, SubmitResult};

fn sched() -> Scheduler {
    Scheduler::new(clinic_resources(), clinic_models()).unwrap()
}

fn job_id(r: SubmitResult) -> JobId {
    match r {
        SubmitResult::Accepted { job_id, .. } => job_id,
        _ => panic!("expected accepted"),
    }
}

#[test]
fn finishing_soap_starts_queued_soap_without_second_load() {
    let mut s = sched();
    let first = job_id(s.submit("soap"));
    s.submit("soap");
    let actions = s.finish(first);
    assert!(actions.iter().any(|a| matches!(a, Action::Start { .. })));
    assert!(!actions
        .iter()
        .any(|a| matches!(a, Action::Load { .. } | Action::Wake { .. })));
    assert_eq!(s.running("soap"), 1);
    assert_eq!(s.queued("soap"), 0);
}

#[test]
fn finishing_whisper_and_scan_allows_queued_soap() {
    let mut s = sched();
    let w = job_id(s.submit("whisper"));
    let c = job_id(s.submit("chart-scan"));
    s.submit("soap");
    s.finish(w);
    assert_eq!(s.queued("soap"), 1);
    let actions = s.finish(c);
    assert!(actions.iter().any(|a| matches!(
        a,
        Action::Start { model, .. } if model == "soap"
    )));
    assert_eq!(s.tier("soap"), silkai_sched::Tier::Bench);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p silkai-sched --test finish`
Expected: FAIL (`finish` missing).

- [ ] **Step 3: Implement `finish`**

Decrement running (clamp at 0). If running hits 0, model **stays Bench** (lazy eviction). Then loop: peek queue in FIFO order; skip jobs whose model cannot place yet; if model's running < slots and model already Bench, Start it; if model can be placed (exclusive/size), emit Sleep/Load/Wake as needed then Start. Only one new exclusive at a time. Continue until no more queued jobs can start.

For `finishing_whisper_and_scan_allows_queued_soap`: after both idle, SOAP exclusive can evict idle neighbors — **eviction of idle models is required here**. Implement idle eviction as part of `place()`:

When placing M:
1. If already Bench and running < slots: just Start.
2. Try to free space by Sleep (keep_warm) or Discard (!keep_warm) of Bench models with running==0, lowest priority first, oldest idle first.
3. If M.exclusive, idle-evict **all** other Bench models with running==0.
4. If still blocked by running others, do not place (queue remains) — preemption is Task 6.
5. Load vs Wake based on current tier.

Track `idle_since` tick or monotonic counter when running hits 0 for oldest-idle order.

- [ ] **Step 4: Run tests**

Run: `cargo test -p silkai-sched`
Expected: PASS including packing (SOAP still queued while whisper+scan **running**).

- [ ] **Step 5: Commit**

```bash
git add crates/silkai-sched
git commit -m "feat(sched): finish admits queue and evicts idle residents"
```

---

### Task 6: Live never waits; live preempts running normal

**Files:**
- Create: `crates/silkai-sched/tests/preempt.rs`
- Modify: `crates/silkai-sched/src/scheduler.rs`

- [ ] **Step 1: Write the failing test**

```rust
use silkai_sched::clinic::{clinic_models, clinic_resources};
use silkai_sched::{Action, JobId, Scheduler, SubmitResult, Tier};

fn sched() -> Scheduler {
    Scheduler::new(clinic_resources(), clinic_models()).unwrap()
}

fn job_id(r: SubmitResult) -> JobId {
    match r {
        SubmitResult::Accepted { job_id, .. } => job_id,
        _ => panic!("expected accepted"),
    }
}

#[test]
fn live_preempts_running_soap_and_requeues_it_at_head() {
    let mut s = sched();
    let soap_id = job_id(s.submit("soap"));
    let r = s.submit("whisper");
    match r {
        SubmitResult::Accepted { actions, .. } => {
            assert!(actions.iter().any(|a| matches!(a, Action::Preempt { job_id } if *job_id == soap_id)));
            assert!(actions.iter().any(|a| matches!(a, Action::Sleep { model } if model == "soap")));
            assert!(actions.iter().any(|a| matches!(a, Action::Start { model, .. } if model == "whisper")));
        }
        _ => panic!("whisper should accept"),
    }
    assert_eq!(s.tier("whisper"), Tier::Bench);
    assert_eq!(s.tier("soap"), Tier::Shelf);
    assert_eq!(s.running("soap"), 0);
    assert_eq!(s.queued("soap"), 1);
    assert_eq!(s.running("whisper"), 1);
}

#[test]
fn soap_does_not_preempt_live_whisper() {
    let mut s = sched();
    s.submit("whisper");
    let r = s.submit("soap");
    match r {
        SubmitResult::Accepted { actions, job_id: _ } => {
            assert!(!actions.iter().any(|a| matches!(a, Action::Preempt { .. })));
            assert!(!actions.iter().any(|a| matches!(a, Action::Start { model, .. } if model == "soap")));
        }
        _ => panic!("soap queued"),
    }
    assert_eq!(s.tier("whisper"), Tier::Bench);
    assert_eq!(s.queued("soap"), 1);
}

#[test]
fn background_does_not_preempt_live() {
    let mut s = sched();
    s.submit("whisper");
    s.submit("whisper");
    // 24 GB used of 29 — chart-scan 10 does not fit; must queue, not preempt
    let r = s.submit("chart-scan");
    match r {
        SubmitResult::Accepted { actions, .. } => {
            assert!(!actions.iter().any(|a| matches!(a, Action::Preempt { .. })));
            assert!(!actions
                .iter()
                .any(|a| matches!(a, Action::Start { model, .. } if model == "chart-scan")));
        }
        _ => panic!("accepted queue"),
    }
    assert_eq!(s.running("whisper"), 2);
}

#[test]
fn exclusive_preempts_running_background() {
    let mut s = sched();
    let scan = job_id(s.submit("chart-scan"));
    let r = s.submit("soap");
    match r {
        SubmitResult::Accepted { actions, .. } => {
            assert!(actions
                .iter()
                .any(|a| matches!(a, Action::Preempt { job_id } if *job_id == scan)));
            assert!(actions
                .iter()
                .any(|a| matches!(a, Action::Start { model, .. } if model == "soap")));
        }
        _ => panic!("soap should start"),
    }
}

#[test]
fn preempted_soap_restarts_after_whisper_finishes() {
    let mut s = sched();
    let soap_id = job_id(s.submit("soap"));
    let w = job_id(s.submit("whisper"));
    let actions = s.finish(w);
    assert!(actions.iter().any(|a| matches!(
        a,
        Action::Start { model, job_id, .. } if model == "soap" && *job_id == soap_id
    )));
}
```

Note: after preemption, the same `JobId` stays queued at the **head** of that model’s queue (restart same job). `finish(whisper)` should Start that same soap `JobId`.

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p silkai-sched --test preempt`
Expected: FAIL (no Preempt actions).

- [ ] **Step 3: Implement preemption**

When `submit(M)` cannot place after idle eviction:

Incoming priority `Live`: preempt all running jobs on non-live Bench models (each job → `Preempt`, requeue that JobId at head of its model queue, running=0), then Sleep those models if keep_warm else Discard, then place M.

Incoming exclusive (and not blocked by **running live**): preempt running background and running normal (not live), same as above, then place M.

Incoming background: never preempt.

Never Preempt a Live running job.

Per-model queues: use `VecDeque<JobId>` plus `HashMap<JobId, String>` owner. Head requeue: `push_front`.

- [ ] **Step 4: Run tests**

Run: `cargo test -p silkai-sched`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/silkai-sched
git commit -m "feat(sched): live and exclusive preemption; never preempt live"
```

---

### Task 7: Shelf, prefetch, RAM demote

**Files:**
- Create: `crates/silkai-sched/tests/shelf.rs`
- Modify: `crates/silkai-sched/src/scheduler.rs`

- [ ] **Step 1: Write the failing test**

```rust
use silkai_sched::clinic::{clinic_models, clinic_resources};
use silkai_sched::{Action, JobId, ModelSpec, Priority, Resources, Scheduler, SubmitResult, Tier};

fn job_id(r: SubmitResult) -> JobId {
    match r {
        SubmitResult::Accepted { job_id, .. } => job_id,
        _ => panic!("expected accepted"),
    }
}

#[test]
fn prefetch_warms_shelf_not_bench() {
    let mut s = Scheduler::new(clinic_resources(), clinic_models()).unwrap();
    let actions = s.prefetch();
    assert!(actions.iter().any(|a| matches!(a, Action::Warm { model } if model == "whisper")));
    assert!(actions.iter().any(|a| matches!(a, Action::Warm { model } if model == "soap")));
    assert_eq!(s.tier("whisper"), Tier::Shelf);
    assert_eq!(s.tier("soap"), Tier::Shelf);
    assert_eq!(s.gpu_used_gb(), 0.0);
    assert_eq!(s.ram_used_gb(), 12.0 + 28.0 + 10.0);
}

#[test]
fn second_run_wakes_from_shelf_not_load() {
    let mut s = Scheduler::new(clinic_resources(), clinic_models()).unwrap();
    s.prefetch();
    let r = s.submit("soap");
    match r {
        SubmitResult::Accepted { actions, .. } => {
            assert!(actions.iter().any(|a| matches!(a, Action::Wake { model } if model == "soap")));
            assert!(!actions.iter().any(|a| matches!(a, Action::Load { .. })));
        }
        _ => panic!(),
    }
}

#[test]
fn ram_pressure_discards_lru_shelf() {
    let res = Resources {
        gpu_schedulable_gb: 29.0,
        ram_shelf_gb: 30.0, // cannot hold 12+28+10
    };
    let mut s = Scheduler::new(res, clinic_models()).unwrap();
    let actions = s.prefetch();
    assert!(actions.iter().any(|a| matches!(a, Action::Discard { .. }))
        || s.ram_used_gb() <= 30.0 + 1e-6);
    assert!(s.ram_used_gb() <= 30.0 + 1e-6);
    // At least one keep_warm model remains cupboard
    let cupboard = ["whisper", "soap", "chart-scan"]
        .iter()
        .filter(|n| s.tier(n) == Tier::Cupboard)
        .count();
    assert!(cupboard >= 1);
}

#[test]
fn keep_warm_false_goes_cupboard_on_evict() {
    let models = vec![ModelSpec {
        name: "temp".into(),
        vram_gb: 12.0,
        ram_gb: 12.0,
        priority: Priority::Normal,
        exclusive: false,
        slots: 1,
        keep_warm: false,
    }];
    let mut s = Scheduler::new(clinic_resources(), models).unwrap();
    let id = job_id(s.submit("temp"));
    assert_eq!(s.tier("temp"), Tier::Bench);
    s.finish(id);
    // still bench until someone needs space — place a 28 GB exclusive to force evict
    s.add_model(ModelSpec {
        name: "big".into(),
        vram_gb: 28.0,
        ram_gb: 28.0,
        priority: Priority::Normal,
        exclusive: true,
        slots: 1,
        keep_warm: true,
    })
    .unwrap();
    s.submit("big");
    assert_eq!(s.tier("temp"), Tier::Cupboard);
}
```

Avoid `add_model` if it is extra API: instead construct both models in `new`. Replace the last test with both models from the start; submit temp, finish, submit big.

```rust
#[test]
fn keep_warm_false_goes_cupboard_on_evict() {
    let models = vec![
        ModelSpec {
            name: "temp".into(),
            vram_gb: 12.0,
            ram_gb: 12.0,
            priority: Priority::Normal,
            exclusive: false,
            slots: 1,
            keep_warm: false,
        },
        ModelSpec {
            name: "big".into(),
            vram_gb: 28.0,
            ram_gb: 28.0,
            priority: Priority::Normal,
            exclusive: true,
            slots: 1,
            keep_warm: true,
        },
    ];
    let mut s = Scheduler::new(clinic_resources(), models).unwrap();
    let id = job_id(s.submit("temp"));
    s.finish(id);
    s.submit("big");
    assert_eq!(s.tier("temp"), Tier::Cupboard);
}
```

`ram_used_gb()`: every model on Shelf or Bench with `keep_warm` counts `ram_gb`. Bench + keep_warm duplicates RAM (weights kept). `!keep_warm` on Bench counts 0 RAM.

Prefetch: emit `Warm` until shelf budget fills; skip models that would exceed; if adding Warm exceeds, `Discard` LRU shelf first only if that helps a higher-priority keep_warm — simplest rule: Warm in Live, then Normal, then Background order; stop when next does not fit.

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p silkai-sched --test shelf`
Expected: FAIL (`prefetch` / `ram_used_gb` missing).

- [ ] **Step 3: Implement `prefetch`, `ram_used_gb`, Warm, Discard-on-evict for !keep_warm, Wake from Shelf on submit**

When sleeping keep_warm: Bench → Shelf. When sleeping !keep_warm: Bench → Cupboard (`Discard`).

If `prefetch` would exceed RAM, do not Warm that model (leave Cupboard). Demote: if a new Warm/keep_warm Bench copy exceeds RAM, Discard oldest Shelf model that is not also needed (not Bench). Bench models’ RAM copies cannot be discarded while on Bench.

- [ ] **Step 4: Run tests**

Run: `cargo test -p silkai-sched`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/silkai-sched
git commit -m "feat(sched): shelf prefetch, wake, and RAM demote"
```

---

### Task 8: Status snapshot

**Files:**
- Create: `crates/silkai-sched/tests/status.rs`
- Modify: `crates/silkai-sched/src/scheduler.rs`

- [ ] **Step 1: Write the failing test**

```rust
use silkai_sched::clinic::{clinic_models, clinic_resources};
use silkai_sched::{Scheduler, Tier};

#[test]
fn status_lists_each_model_tier_and_counts() {
    let mut s = Scheduler::new(clinic_resources(), clinic_models()).unwrap();
    s.submit("whisper");
    s.submit("whisper");
    s.submit("whisper"); // queued
    let snap = s.status();
    let w = snap.models.iter().find(|m| m.name == "whisper").unwrap();
    assert_eq!(w.tier, Tier::Bench);
    assert_eq!(w.running, 2);
    assert_eq!(w.queued, 1);
    assert_eq!(snap.gpu_used_gb, 12.0);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p silkai-sched --test status`
Expected: FAIL if `status` missing.

- [ ] **Step 3: Implement `status() -> StatusSnapshot`**

- [ ] **Step 4: Run tests** — `cargo test -p silkai-sched` PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/silkai-sched
git commit -m "feat(sched): status snapshot for /v1/status"
```

---

### Task 9: Fake engine adapter

**Files:**
- Create: `crates/silkai-adapters/Cargo.toml`
- Create: `crates/silkai-adapters/src/lib.rs`
- Create: `crates/silkai-adapters/src/fake.rs`
- Modify: `Cargo.toml` (add member)

- [ ] **Step 1: Write the failing test** in `crates/silkai-adapters/src/fake.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::Engine;
    use tokio_util::sync::CancellationToken;

    #[tokio::test]
    async fn fake_load_sleep_wake_records_order() {
        let e = FakeEngine::new("soap", 28.0);
        e.load("/models/soap.gguf").await.unwrap();
        e.sleep().await.unwrap();
        e.wake().await.unwrap();
        assert_eq!(e.log(), vec!["load", "sleep", "wake"]);
        assert_eq!(e.measured_vram_gb(), 28.0);
    }

    #[tokio::test]
    async fn fake_run_streams_two_chunks_then_done() {
        let e = FakeEngine::new("soap", 28.0);
        e.load("/x").await.unwrap();
        let cancel = CancellationToken::new();
        let mut rx = e.run("hello", cancel).await.unwrap();
        let mut got = Vec::new();
        while let Some(t) = rx.recv().await {
            got.push(t);
        }
        assert_eq!(got, vec!["hello".to_string(), " world".to_string()]);
    }

    #[tokio::test]
    async fn fake_run_stops_on_cancel() {
        let e = FakeEngine::new("soap", 28.0);
        e.load("/x").await.unwrap();
        let cancel = CancellationToken::new();
        cancel.cancel();
        let mut rx = e.run("hello", cancel).await.unwrap();
        assert!(rx.recv().await.is_none());
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Add member first so the crate exists, then run `cargo test -p silkai-adapters`.
Expected: FAIL missing types.

- [ ] **Step 3: Implement trait + FakeEngine**

Root workspace members include `crates/silkai-adapters`.

`crates/silkai-adapters/Cargo.toml`:

```toml
[package]
name = "silkai-adapters"
version.workspace = true
edition.workspace = true
license.workspace = true

[dependencies]
async-trait = "0.1"
tokio = { version = "1", features = ["rt", "macros", "sync", "time"] }
tokio-util = { version = "0.7", features = ["rt"] }
thiserror = "2"

[dev-dependencies]
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
```

`crates/silkai-adapters/src/lib.rs`:

```rust
mod fake;
pub use fake::FakeEngine;

use async_trait::async_trait;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("not loaded")]
    NotLoaded,
    #[error("{0}")]
    Other(String),
}

#[async_trait]
pub trait Engine: Send + Sync {
    async fn warm(&self, path: &str) -> Result<(), EngineError>;
    async fn load(&self, path: &str) -> Result<(), EngineError>;
    async fn wake(&self) -> Result<(), EngineError>;
    async fn sleep(&self) -> Result<(), EngineError>;
    async fn discard(&self) -> Result<(), EngineError>;
    async fn run(
        &self,
        prompt: &str,
        cancel: CancellationToken,
    ) -> Result<mpsc::Receiver<String>, EngineError>;
    fn measured_vram_gb(&self) -> f64;
}
```

`FakeEngine`: interior mutex log + state enum Cupboard/Shelf/Bench. `warm` pushes "warm" and sets shelf. `load` cupboard/shelf → bench. `sleep` bench → shelf. `discard` → cupboard. `run` only if bench; spawn task sending two tokens with 5ms sleep unless cancelled. `warm` may no-op as copy of path stored.

Default `warm` on trait: `self.load` then `self.sleep` is wrong (touches bench). Fake `warm` only sets shelf without bench.

- [ ] **Step 4: Run tests**

Run: `cargo test -p silkai-adapters`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml crates/silkai-adapters
git commit -m "feat(adapters): Engine trait and FakeEngine"
```

---

### Task 10: TOML config

**Files:**
- Create: `crates/silkai-server/Cargo.toml`
- Create: `crates/silkai-server/src/lib.rs`
- Create: `crates/silkai-server/src/config.rs`
- Modify: `Cargo.toml` members

- [ ] **Step 1: Write the failing test** at the bottom of `config.rs` in `#[cfg(test)]` — tests need the module, so put integration test `crates/silkai-server/tests/config.rs` after crate exists.

Test content:

```rust
use silkai_server::config::load_from_str;
use silkai_sched::Priority;

const TOML: &str = r#"
listen = "127.0.0.1:8080"

[resources]
gpu_total_gb = 32
gpu_headroom_gb = 3
ram_headroom_gb = 32
prefetch_on_start = true
request_timeout_secs = 600
ram_total_gb = 128

[models.whisper]
engine = "fake"
path = "/models/whisper.bin"
vram_gb = 12
priority = "live"
exclusive = false
slots = 2
keep_warm = true
transport = "websocket"
idle_timeout_secs = 45

[models.soap]
engine = "llama.cpp"
path = "/models/soap-q4.gguf"
vram_gb = 28
priority = "normal"
exclusive = true
slots = 1
keep_warm = true
transport = "http"

[models.too-big]
engine = "fake"
path = "/models/huge.gguf"
vram_gb = 40
priority = "normal"
exclusive = true
slots = 1
keep_warm = true
transport = "http"
"#;

#[test]
fn parses_clinic_and_disables_too_big() {
    let cfg = load_from_str(TOML).unwrap();
    assert_eq!(cfg.listen, "127.0.0.1:8080");
    assert_eq!(cfg.resources.gpu_schedulable_gb(), 29.0);
    assert_eq!(cfg.resources.ram_shelf_gb(), 96.0);
    assert!(cfg.prefetch_on_start);
    assert_eq!(cfg.request_timeout_secs, 600);
    let soap = cfg.enabled.iter().find(|m| m.spec.name == "soap").unwrap();
    assert_eq!(soap.spec.priority, Priority::Normal);
    assert!(soap.spec.exclusive);
    assert_eq!(soap.engine, "llama.cpp");
    assert!(cfg.disabled.iter().any(|m| m.spec.name == "too-big"));
}

#[test]
fn invalid_toml_errors() {
    assert!(load_from_str("listen = ").is_err());
}

#[test]
fn missing_gpu_total_errors() {
    let t = r#"
listen = "127.0.0.1:8080"
[resources]
gpu_headroom_gb = 3
ram_headroom_gb = 32
ram_total_gb = 128
"#;
    assert!(load_from_str(t).is_err());
}
```

`ram_total_gb` is required in slice 1 (no RAM probe). Spec said total RAM − headroom; probing RAM can be `sysinfo` later. Require `ram_total_gb` in TOML for determinism.

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p silkai-server --test config`
Expected: FAIL.

- [ ] **Step 3: Implement config**

Dependencies: `serde`, `toml`, `silkai-sched`, `thiserror`.

Types:

```rust
pub struct FileConfig {
    pub listen: String,
    pub resources: FileResources,
    pub prefetch_on_start: bool, // pulled from resources in TOML
    ...
}
```

Match spec TOML: `prefetch_on_start` and `request_timeout_secs` live under `[resources]`. Flatten into `AppConfig`:

```rust
pub struct AppConfig {
    pub listen: String,
    pub prefetch_on_start: bool,
    pub request_timeout_secs: u64,
    pub resources: silkai_sched::Resources,
    pub enabled: Vec<ConfiguredModel>,
    pub disabled: Vec<ConfiguredModel>,
}

pub struct ConfiguredModel {
    pub spec: silkai_sched::ModelSpec,
    pub engine: String,
    pub path: String,
    pub transport: String,
    pub idle_timeout_secs: Option<u64>,
}
```

`gpu_schedulable_gb = gpu_total_gb - gpu_headroom_gb`. If `vram_gb > gpu_schedulable`, move to `disabled`.
`ram_shelf_gb = ram_total_gb - ram_headroom_gb`.

Priority parse: `"live"|"normal"|"background"` case-insensitive.
Default listen `127.0.0.1:8080`. Default slots 1, keep_warm true, transport http, exclusive false.

- [ ] **Step 4: Run tests** — PASS.

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml crates/silkai-server
git commit -m "feat(server): TOML config with disabled oversized models"
```

---

### Task 11: Runtime applies actions to engines

**Files:**
- Create: `crates/silkai-server/src/runtime.rs`
- Create: `crates/silkai-server/tests/runtime.rs`

- [ ] **Step 1: Write the failing test**

```rust
use silkai_adapters::FakeEngine;
use silkai_server::config::{AppConfig, ConfiguredModel};
use silkai_server::runtime::Runtime;
use silkai_sched::clinic::{clinic_models, clinic_resources};
use std::sync::Arc;

fn clinic_cfg() -> AppConfig {
    let models = clinic_models()
        .into_iter()
        .map(|spec| ConfiguredModel {
            engine: "fake".into(),
            path: format!("/models/{}.bin", spec.name),
            transport: "http".into(),
            idle_timeout_secs: None,
            spec,
        })
        .collect();
    AppConfig {
        listen: "127.0.0.1:0".into(),
        prefetch_on_start: true,
        request_timeout_secs: 600,
        resources: clinic_resources(),
        enabled: models,
        disabled: vec![],
    }
}

#[tokio::test]
async fn prefetch_then_soap_wakes_fake_engine() {
    let rt = Runtime::new(clinic_cfg()).await.unwrap();
    let (job, mut tokens) = rt.submit_chat("soap", "note").await.unwrap();
    let mut out = String::new();
    while let Some(t) = tokens.recv().await {
        out.push_str(&t);
    }
    rt.finished(job).await;
    assert_eq!(out, "note world");
    let st = rt.status();
    let soap = st.models.iter().find(|m| m.name == "soap").unwrap();
    assert_eq!(soap.running, 0);
}
```

FakeEngine `run` yields `"hello"` + `" world"` if prompt is hello — for prompt `"note"` make FakeEngine echo `prompt` then `" world"` so this assertion works. Update FakeEngine in this task: first chunk is the prompt, second is `" world"`. Fix Task 9 test if needed (`"hello"` + `" world"` still holds).

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p silkai-server --test runtime`
Expected: FAIL (`Runtime` missing).

- [ ] **Step 3: Implement Runtime**

`Runtime` holds `Mutex<Scheduler>` and `HashMap<String, Arc<dyn Engine>>`.

`new`: for each enabled model, `Arc<FakeEngine::new(name, vram)>` (llama later). If `prefetch_on_start`, `scheduler.prefetch()` then for each `Warm { model }` call `engine.warm(path)`.

`submit_chat(model, prompt)`:
- if disabled (store disabled names on Runtime): return `RuntimeError::Disabled`
- unknown: `RuntimeError::UnknownModel`
- `scheduler.submit`
- for each action: Warm/Load/Wake/Sleep/Discard on that engine; Preempt: cancel token for that job_id; Start: `engine.run` with a new CancellationToken stored by JobId
- return `(JobId, Receiver<String>)` even if queued: if no Start yet, hold prompt in `pending: HashMap<JobId, String>` and a placeholder channel; when `finished` admits a Start, spawn run and forward — **too heavy**.

Simpler v1 runtime: **block inside submit_chat until the job is Started** (still async intake at HTTP layer with two connections). For the queued-SOAP-while-whisper-runs HTTP test, `submit_chat` must return immediately with a receiver that waits.

Implement: each Accepted job gets an `mpsc` pair. If Start in actions, spawn run and forward chunks. If queued, store `Waiter { prompt, tx }` keyed by JobId. `finished(job_id)` calls `scheduler.finish`, processes actions, on Start move waiter and spawn.

`CancellationToken` map for Preempt: cancel, then `finished` is **not** called for preempt (job stays queued). Runtime on Preempt: cancel run; do not `finish`; scheduler already requeued.

When run stream ends normally, caller (`app`) calls `rt.finished(job)`.

- [ ] **Step 4: Run tests** — PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/silkai-adapters crates/silkai-server
git commit -m "feat(server): runtime executes scheduler actions on engines"
```

---

### Task 12: HTTP health, models, status

**Files:**
- Create: `crates/silkai-server/src/app.rs`
- Create: `crates/silkai-server/tests/http_status.rs`

- [ ] **Step 1: Write the failing test**

```rust
use axum::body::Body;
use axum::http::{Request, StatusCode};
use silkai_server::app::test_app;
use tower::ServiceExt;

#[tokio::test]
async fn health_ok() {
    let app = test_app().await;
    let res = app
        .oneshot(Request::builder().uri("/health").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
}

#[tokio::test]
async fn lists_configured_models() {
    let app = test_app().await;
    let res = app
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(res.into_body(), 1 << 20).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let ids: Vec<&str> = v["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["id"].as_str().unwrap())
        .collect();
    assert!(ids.contains(&"whisper"));
    assert!(ids.contains(&"soap"));
}

#[tokio::test]
async fn status_json_has_tiers() {
    let app = test_app().await;
    let res = app
        .oneshot(
            Request::builder()
                .uri("/v1/status")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(res.into_body(), 1 << 20).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert!(v.get("models").is_some());
    assert!(v.get("gpu_used_gb").is_some());
}
```

`test_app()` builds clinic config + Runtime + router.

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p silkai-server --test http_status`
Expected: FAIL.

- [ ] **Step 3: Implement axum 0.8 router**

Dependencies: `axum`, `serde_json`, `tower`, `http-body-util` as needed.

```rust
GET /health -> "ok"
GET /v1/models -> { "object": "list", "data": [ { "id": name, "object": "model" } ] }  // enabled + disabled? spec: configured names. Include enabled only, plus disabled with a flag is extra — include both enabled and disabled ids so clients see soap even if too big. Spec: "Configured names". List enabled and disabled.
GET /v1/status -> StatusSnapshot as JSON
```

Make `pub async fn app_from_config(cfg: AppConfig) -> Router` and `pub async fn test_app() -> Router` public in `app.rs` (clinic fake config). Also `pub async fn test_app_with_disabled() -> Router` (includes `too-big` 40 GB) and `pub async fn test_app_timeout_ms(ms: u64) -> Router` (`request_timeout_secs` converted: use millis in AppConfig for tests — store `request_timeout: Duration` on Runtime instead of only seconds, built from `request_timeout_secs` in production and from ms in tests).

- [ ] **Step 4: Run tests** — PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/silkai-server
git commit -m "feat(server): /health /v1/models /v1/status"
```

---

### Task 13: POST /v1/chat/completions with queue SSE comments

**Files:**
- Modify: `crates/silkai-server/src/app.rs`
- Create: `crates/silkai-server/tests/http_chat.rs`

- [ ] **Step 1: Write the failing test**

```rust
use axum::body::Body;
use axum::http::{Request, StatusCode};
use silkai_server::app::test_app;
use tower::ServiceExt;

fn chat(model: &str, stream: bool) -> Request<Body> {
    let body = serde_json::json!({
        "model": model,
        "stream": stream,
        "messages": [{"role": "user", "content": "hello"}]
    });
    Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

#[tokio::test]
async fn unknown_model_404() {
    let app = test_app().await;
    let res = app.oneshot(chat("nope", false)).await.unwrap();
    assert_eq!(res.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn disabled_model_400() {
    // too-big only in full config; test_app clinic has no too-big.
    // Use a one-off router: skip if test_app has no disabled.
    // Instead: submit huge name not in clinic -> 404. Add test_app_with_disabled in app.rs.
    let app = silkai_server::app::test_app_with_disabled().await;
    let res = app.oneshot(chat("too-big", false)).await.unwrap();
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn streaming_soap_returns_sse_data_lines() {
    let app = test_app().await;
    let res = app.oneshot(chat("soap", true)).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(res.into_body(), 1 << 20).await.unwrap();
    let text = String::from_utf8(bytes.to_vec()).unwrap();
    assert!(text.contains("data:"));
    assert!(text.contains("[DONE]"));
}

#[tokio::test]
async fn queued_stream_sends_comment_before_tokens() {
    let app = test_app().await;
    let whisper_app = app.clone();
    let whisper = tokio::spawn(async move {
        whisper_app.oneshot(chat("whisper", true)).await.unwrap()
    });
    tokio::time::sleep(std::time::Duration::from_millis(30)).await;
    let res = app.oneshot(chat("soap", true)).await.unwrap();
    let bytes = axum::body::to_bytes(res.into_body(), 1 << 20).await.unwrap();
    let text = String::from_utf8(bytes.to_vec()).unwrap();
    assert!(
        text.contains(": queued") || text.contains("queued"),
        "expected a queued SSE comment, got {text:?}"
    );
    let _ = whisper.await;
}

#[tokio::test]
async fn timeout_while_queued_returns_504() {
    let app = silkai_server::app::test_app_timeout_ms(1).await;
    let whisper_app = app.clone();
    let _whisper = tokio::spawn(async move {
        let _ = whisper_app.oneshot(chat("whisper", true)).await;
    });
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    let res = app.oneshot(chat("soap", true)).await.unwrap();
    assert_eq!(res.status(), StatusCode::GATEWAY_TIMEOUT);
}
```

Clean this test file: one `test_app()`, clone router (`Router` is clone if state is `Arc`). FakeEngine should delay chunks ~50ms so whisper still running when soap POST happens.

Make FakeEngine `run` sleep 80ms before first token (tests only? use 20ms always — unit tests still pass).

SSE format:

```
: queued

data: {"id":"job-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"hello"},"finish_reason":null}]}

data: [DONE]

```

While waiting for first token, spawn interval 15s comments — for tests, send **one** `: queued\n\n` immediately if job is not started yet (check status queued > 0 or waiter’s still pending). Send immediately on enqueue so tests don’t wait 15s.

Timeout: if `request_timeout_secs` exceeded, 504. In tests leave 600.

Non-stream JSON completion also required: `{ choices: [{ message: { role: "assistant", content: "..." } }] }`.

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p silkai-server --test http_chat`
Expected: FAIL.

- [ ] **Step 3: Implement handler**

Parse `{ model, stream, messages[] }`. Prompt = last user content (if missing, empty string).

Map errors: Unknown → 404, Disabled/TooLarge → 400, timeout → 504, engine → 500.

After `submit_chat`, if stream: `Sse` stream of comments + data chunks + `[DONE]`, then `finished`. If not stream: collect tokens, `finished`, JSON body.

Do not allow request fields for priority/exclusive.

- [ ] **Step 4: Run tests** — PASS `cargo test -p silkai-server`.

- [ ] **Step 5: Commit**

```bash
git add crates/silkai-server crates/silkai-adapters
git commit -m "feat(server): OpenAI chat completions with queued SSE comments"
```

---

### Task 14: POST /admin/reload

**Files:**
- Modify: `crates/silkai-server/src/app.rs`
- Modify: `crates/silkai-server/src/runtime.rs`
- Create: `crates/silkai-server/tests/http_reload.rs`

- [ ] **Step 1: Write the failing test**

Reload from a temp file: start with soap+whisper fake; write new TOML removing whisper; POST `/admin/reload` with header `X-Silkai-Config: /tmp/...` **too ad hoc**.

Simpler: `Runtime::reload(AppConfig)` public. HTTP `/admin/reload` re-reads path stored at start (`config_path`). Test `Runtime::reload` at unit level and HTTP 200 when file still valid.

```rust
#[tokio::test]
async fn reload_keeps_inflight_and_returns_ok() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("c.toml");
    std::fs::write(&path, VALID_TOML).unwrap();
    let app = silkai_server::app::app_from_path(&path).await.unwrap();
    let res = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/admin/reload")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
}

#[tokio::test]
async fn reload_bad_file_keeps_old_config() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("c.toml");
    std::fs::write(&path, VALID_TOML).unwrap();
    let app = silkai_server::app::app_from_path(&path).await.unwrap();
    std::fs::write(&path, "listen = ").unwrap();
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/admin/reload")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);
    let res = app
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let bytes = axum::body::to_bytes(res.into_body(), 1 << 20).await.unwrap();
    let text = String::from_utf8(bytes.to_vec()).unwrap();
    assert!(text.contains("soap"));
}
```

`VALID_TOML` is the clinic fake TOML with `ram_total_gb = 128` and fake engines.

- [ ] **Step 2: Run test to verify it fails**

- [ ] **Step 3: Implement `load_from_path`, `app_from_path`, reload that on parse failure returns error and does not replace `Arc<Runtime>`**

In-flight jobs: replace scheduler only for **new** submits; easiest slice 1: reject reload if any running > 0 (`409 Conflict`), else swap Runtime. Spec said in-flight keep old record — 409 if busy is stricter but valid YAGNI. **Follow spec:** swap config for new jobs. Keep old Runtime for running JobIds is hard.

Slice 1 reload: if running==0, swap whole Runtime (re-prefetch). If running>0, `409`. Document in README. Closer to “new jobs use new config” without split-brain.

- [ ] **Step 4: Run tests** — PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/silkai-server
git commit -m "feat(server): /admin/reload keeps last good config"
```

---

### Task 15: `silkai` binary and example config

**Files:**
- Create: `crates/silkai/Cargo.toml`
- Create: `crates/silkai/src/main.rs`
- Create: `examples/config.toml`
- Modify: `Cargo.toml` members

- [ ] **Step 1: Write a failing test** — binary tests via `trycmd` are heavy. Test `silkai_server::run` bind: in `crates/silkai-server/tests/listen.rs` bind `127.0.0.1:0` and GET `/health` with `reqwest`.

```rust
#[tokio::test]
async fn binds_localhost_and_serves_health() {
    let cfg = /* clinic AppConfig listen 127.0.0.1:0 */;
    let handle = tokio::spawn(async move {
        silkai_server::serve(cfg).await.unwrap();
    });
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    // serve must return bound addr — change API:
}
```

Better API: `pub async fn bind_and_serve(cfg: AppConfig) -> anyhow::Result<(SocketAddr, impl Future)>` or `serve_with_ready`.

Simplest: `pub async fn serve(cfg: AppConfig) -> anyhow::Result<()>` and in test use `tokio::net::TcpListener::bind("127.0.0.1:0")` inside `serve` after parsing listen; if port 0, that's fine.

```rust
#[tokio::test]
async fn health_via_tcp() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let cfg = clinic_app_config();
    tokio::spawn(async move {
        silkai_server::serve_listener(listener, cfg).await.unwrap();
    });
    let body = reqwest::get(format!("http://{addr}/health"))
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert_eq!(body, "ok");
}
```

- [ ] **Step 2: Run test to verify it fails**

- [ ] **Step 3: Implement `serve_listener` + binary**

`main.rs`:

```rust
fn main() -> anyhow::Result<()> {
    let path = std::env::var("SILKAI_CONFIG")
        .unwrap_or_else(|_| default_config_path());
    let cfg = silkai_server::config::load_from_path(&path)?;
    tracing_subscriber::fmt::init();
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(silkai_server::serve(cfg))
}

fn default_config_path() -> String {
    let mut p = dirs::config_dir().unwrap_or_else(|| ".".into());
    p.push("silkai");
    p.push("config.toml");
    p.to_string_lossy().into_owned()
}
```

`serve`: bind `cfg.listen` (must be 127.0.0.1 host; if not, error — localhost only).

`examples/config.toml`: clinic-like with `engine = "fake"` and comment showing `llama.cpp`.

- [ ] **Step 4: Run `cargo test` all packages** PASS.

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml crates/silkai crates/silkai-server examples/config.toml
git commit -m "feat: silkai daemon binds localhost"
```

---

### Task 16: README and CONTRIBUTING

**Files:**
- Create: `README.md`
- Create: `CONTRIBUTING.md`

- [ ] **Step 1: Write README.md**

```markdown
# SilkAI

Local GPU capacity scheduler. One daemon owns the card; models pack, queue, or
park in RAM. Apps speak OpenAI-shaped HTTP.

MIT licensed. Optional coffee (uncomment when the public repo exists):

<!-- [Buy me a coffee](https://ko-fi.com/YOUR_PAGE) -->

## Requirements

- Linux (x86_64 or aarch64)
- Rust 1.80+
- Optional: a GGUF model and llama.cpp (feature `llama`)

## Run (fake engines, no GPU)

```bash
cp examples/config.toml ~/.config/silkai/config.toml
cargo run -p silkai
curl -s http://127.0.0.1:8080/health
```

## Config

See `examples/config.toml`. Priority, exclusive, slots, and VRAM are **only**
in this file — requests just send `"model"`.

## Development

```bash
cargo test
```

Scheduler tests do not need a GPU.
```

`CONTRIBUTING.md`: run `cargo test` before PR; do not put CUDA types in `silkai-sched`; no CLA.

- [ ] **Step 2: No test required (docs).**

- [ ] **Step 3: Commit**

```bash
git add README.md CONTRIBUTING.md
git commit -m "docs: README, coffee placeholder, contributing"
```

---

### Task 17: llama.cpp adapter (feature `llama`)

**Files:**
- Modify: `crates/silkai-adapters/Cargo.toml`
- Create: `crates/silkai-adapters/src/llama.rs`
- Modify: `crates/silkai-adapters/src/lib.rs`
- Modify: `crates/silkai-server/src/runtime.rs` to construct `LlamaEngine` when `engine == "llama.cpp"`
- Create: `crates/silkai-adapters/src/llama.rs` tests with a stub if no model

Implementation plan for `LlamaEngine` (feature `llama`):

Pin `llama-cpp-2 = "0.1"` (utilityai bindings). CPU build by default; `cuda`/`vulkan` features later.

`LlamaEngine` holds `Mutex<Inner>`: `backend: LlamaBackend`, `path`, optional `LlamaModel`, optional context.

```rust
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::context::params::LlamaContextParams;

// warm / sleep (shelf): GPU layers 0
let params = LlamaModelParams::default().with_n_gpu_layers(0);
let model = LlamaModel::load_from_file(&backend, path, &params)?;

// load / wake (bench): offload all layers
let params = LlamaModelParams::default().with_n_gpu_layers(1000);
let model = LlamaModel::load_from_file(&backend, path, &params)?;
let ctx = model.new_context(&backend, LlamaContextParams::default())?;
```

`run`: `spawn_blocking` tokenize with `model.str_to_token(prompt, AddBos::Always)`, decode batch, sample greedy, `token_to_str`, send each piece; if `cancel.is_cancelled()` return. Drop context on sleep; drop model on discard.

Load a missing path in `spawn_blocking` so the async trait does not block the runtime.

**Without a GGUF on disk, skip I/O tests:**

```rust
#[cfg(all(test, feature = "llama"))]
#[tokio::test]
async fn llama_rejects_missing_file() {
    let e = LlamaEngine::new("soap", 1.0);
    let err = e.load("/no/such/model.gguf").await.unwrap_err();
    assert!(matches!(err, EngineError::Other(_)));
}
```

- [ ] **Step 1: Write `llama_rejects_missing_file` first, run with `--features llama`, expect fail.**
- [ ] **Step 2: Implement `LlamaEngine`.**
- [ ] **Step 3: Wire Runtime: `engine == "llama.cpp"` → `LlamaEngine` if feature enabled, else `EngineError` mapped to 503 at submit (config warning at start: log `llama.cpp requested but built without feature llama`).**

Build default **without** `llama` so `cargo test` stays GPU-free.

`crates/silkai/Cargo.toml`:

```toml
[features]
default = []
llama = ["silkai-server/llama"]
```

Pass feature through server → adapters.

- [ ] **Step 4: `cargo test` (no feature) PASS; `cargo test -p silkai-adapters --features llama` PASS missing-file test.**
- [ ] **Step 5: Commit**

```bash
git add crates/silkai-adapters crates/silkai-server crates/silkai
git commit -m "feat(adapters): llama.cpp engine behind feature llama"
```

---

### Task 18: Optional GPU integration test

**Files:**
- Create: `crates/silkai-server/tests/itest_llama.rs`

- [ ] **Step 1: Write the test**

```rust
#[cfg(feature = "llama")]
#[tokio::test]
async fn two_tiny_ggufs_pack_or_queue() {
    if std::env::var("SILKAI_ITEST").ok().as_deref() != Some("1") {
        eprintln!("skip");
        return;
    }
    let a = std::env::var("SILKAI_GGUF_A").expect("SILKAI_GGUF_A");
    let b = std::env::var("SILKAI_GGUF_B").expect("SILKAI_GGUF_B");
    // Build AppConfig with two llama.cpp models vram 12 and 28 (or env-provided),
    // submit chat to both, assert /v1/status eventually has at most one exclusive
    // on bench, and both requests return 200 with content.
}
```

Use env `SILKAI_VRAM_A` default 12, `SILKAI_VRAM_B` default 28, exclusive true on B.

If env not set and SILKAI_ITEST!=1, return immediately (pass).

- [ ] **Step 2: This test passes as skip without env.**
- [ ] **Step 3: Document in README:**

```bash
SILKAI_ITEST=1 SILKAI_GGUF_A=/path/tiny.gguf SILKAI_GGUF_B=/path/small.gguf \
  cargo test -p silkai-server --features llama --test itest_llama -- --nocapture
```

- [ ] **Step 4: Commit**

```bash
git add crates/silkai-server README.md
git commit -m "test: optional llama GGUF integration gated by SILKAI_ITEST"
```

---

## Spec coverage

| Spec item | Task |
|---|---|
| Pack 12+10, not 28 beside them | 3 |
| Exclusive waits on live | 6 |
| slots=1 serial, one load | 4, 5 |
| slots=2 one load two running | 4 |
| live preempts normal, requeue head | 6 |
| background never preempts live | 6 |
| exclusive preempts background | 6 |
| prefetch not on bench | 7 |
| shelf demote | 7 |
| HTTP status/models/health | 12 |
| SSE queue comments | 13 |
| 404/400/504 | 13 (`timeout_while_queued_returns_504`) |
| localhost only | 15 |
| TOML policy, no request override | 10, 13 |
| fake + llama adapters | 9, 17 |
| MIT, README coffee comment | 1, 16 |
| reload last good | 14 |
| Warm sleep keep_warm | 5, 7, 9 |
| Preempted job same JobId restart | 6 |
| Disabled oversized model | 10, 13 |
| Engine missing 503 | 17 |
| WebSocket / whisper.cpp | **out of slice 1** (slice 2) |
| vLLM | slice 3 |
| SIGHUP | HTTP reload only in slice 1 |

Add to Task 13 if missing: a test `request_timeout_secs = 0` (or 1ms with a FakeEngine that waits 1s) expects 504.

## Type consistency

`SubmitResult::Accepted { job_id, actions }` — never `Queued` as a separate enum variant; queued = Accepted without `Start`.

`Action::Warm` cupboard→shelf; `Load` →bench from cupboard; `Wake` →bench from shelf.

`FakeEngine::run` yields `[prompt, " world"]`.

`AppConfig.enabled: Vec<ConfiguredModel>` with `spec: ModelSpec`, `engine`, `path`.

No `add_model` on Scheduler after Task 7 rewrite.

---

## Done when

`cargo test` is green without a GPU. Daemon serves chat on 127.0.0.1 with fake engines. Feature `llama` loads a missing file as an error. README + MIT present.
