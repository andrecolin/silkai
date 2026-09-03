# SilkAI Status Page and Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Truthful status (loading state, measured VRAM, honest RAM), an events stream, a Prometheus endpoint, and an optional embedded status page at `/ui`. Configuration from the page is phase 2 and waits on incremental reload.

**Architecture:** All changes live in `silkai-server` and `silkai-adapters`. `silkai-sched` does not change: `loading` and `sleeping` are runtime overlays on the scheduler's three tiers. A background sampler shells out to `nvidia-smi` every two seconds. Events are a ring buffer in the runtime, exposed as SSE and mirrored to `tracing`.

**Tech Stack:** Axum 0.8 (already present), SSE via `axum::response::sse`, `include_str!` for the page, hand-written Prometheus text. No new crates in phase 1. Phase 2 adds `toml_edit`.

**Spec:** `docs/superpowers/specs/2026-09-03-silkai-status-ui-design.md`

**Order matters.** Tasks 1 to 3 fix what the page would otherwise expose. Task 7 (the page) is last in phase 1 on purpose.

---

## File structure

```
crates/silkai-adapters/src/lib.rs          Engine::has_shelf() (default false)
crates/silkai-server/src/config.rs         [ui] section: enabled, token
crates/silkai-server/src/runtime.rs        overlays, events ring, counters
crates/silkai-server/src/status.rs         NEW: extended snapshot assembly
crates/silkai-server/src/sampler.rs        NEW: nvidia-smi sampler task
crates/silkai-server/src/events.rs         NEW: Event type, ring, SSE handler
crates/silkai-server/src/metrics.rs        NEW: /metrics text renderer
crates/silkai-server/src/auth.rs           NEW: bearer token layer
crates/silkai-server/src/app.rs            routes: /ui, /metrics, /v1/events
crates/silkai-server/ui/index.html         NEW: the page
crates/silkai-server/tests/status_ext.rs   NEW
crates/silkai-server/tests/events.rs       NEW
crates/silkai-server/tests/metrics.rs      NEW
crates/silkai-server/tests/ui.rs           NEW
examples/config.toml                       [ui] block, commented
README.md                                  Status page section
```

---

## Phase 1

### Task 1: Loading and sleeping overlays

**Why first:** the page would show a 22 GB load as "shelf, nothing happening".

- [ ] In `runtime.rs`, add `in_flight: StdMutex<HashMap<String, Overlay>>` where `Overlay` is `Loading | Sleeping`.
- [ ] In `apply()`, insert the overlay before calling `load`/`wake` (Loading) or `sleep` (Sleeping); remove it after the engine call returns, on success or error.
- [ ] Extend `ModelStatus` (in `silkai-sched`, additive, all new fields `#[serde(default)]`-friendly) with `state: String`, `engine`, `budget_gb`, `measured_gb: Option<f64>`, `priority`, `exclusive`, `slots`, `sessions`. Keep `tier`.
- [ ] `Runtime::status()` merges: tier from the scheduler snapshot, `state` = overlay if present else tier name, `sessions` from the sessions set.
- [ ] Normalise `-0.0` to `0.0` in `gpu_used_gb` and per-GPU `used_gb`.
- [ ] Test (`tests/status_ext.rs`): with a fake engine whose `load` blocks on a channel, submit a job, assert `state == "loading"` while blocked, `"bench"` after release. Test `-0.0` is gone.

### Task 2: Honest RAM

- [ ] Add `fn has_shelf(&self) -> bool { false }` to the `Engine` trait. Fake returns `true` (tests rely on shelf semantics). Process, vLLM, Ollama return `false`. In-process llama.cpp returns `false` with a comment explaining mmap and the page cache.
- [ ] `ram_used_gb` sums `ram_gb` only for models on the shelf whose engine has a shelf.
- [ ] Test: three process-engine models warmed at startup report `ram_used_gb == 0.0`; the same with fake engines reports the sum as before.

### Task 3: Measured VRAM sampler

- [ ] `sampler.rs`: a Tokio task started by `Runtime::new` when `nvidia-smi` is on PATH. Every 2 s run `--query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits` and `--query-compute-apps=pid,used_memory --format=csv,noheader,nounits`. Store in `StdMutex<Sample>`. Never block a request on it.
- [ ] Parsing functions are pure and unit-tested with captured output (`parse_gpu_line`, `parse_apps`). Reuse `config::parse_nvidia_smi` style.
- [ ] Attribution: `ProcessEngine::child_id()` for process models; `std::process::id()` for the in-process engine. Add `fn pid(&self) -> Option<u32>` to the trait with default `None`.
- [ ] `GpuStatus` gains `measured_used_gb: Option<f64>` and `total_gb: Option<f64>`. `ModelStatus.measured_gb` filled by pid match.
- [ ] When nvidia-smi is absent, fields are `null`; nothing else changes. Test both branches with an injected sampler (trait or closure), not the real binary.

### Task 4: Events ring and SSE

- [ ] `events.rs`: `Event { seq: u64, t: String (RFC 3339), kind: &'static str, model: Option<String>, job: Option<u64>, gpu: Option<u32>, ms: Option<u64>, error: Option<String> }`. Ring of 500 in `StdMutex<VecDeque<Event>>` plus a `tokio::sync::broadcast::Sender<Event>` (capacity 256).
- [ ] Emit from `apply()` for each `Action` with duration for load/wake/sleep; from `isolate()` as `fault`; from `begin_session`/`end_session`; from `reload`. Keep the existing `tracing` lines.
- [ ] `GET /v1/events`: replay ring (filtered by `?after=seq`), then forward broadcast. `data:` is the JSON event, `id:` is `seq`. Lagged receivers get a `{"kind":"lagged"}` event and continue.
- [ ] Test (`tests/events.rs`): submit a chat on the fake engine, read `/v1/events` until `finish`, assert order `load`, `start`, `finish` for that model, with `ms` present on `load`.

### Task 5: Metrics

- [ ] Counters in the runtime: `loads_total`, `load_seconds_sum`, `wakes_total`, `sleeps_total`, `preempts_total`, `faults_total`, all per model, plain `u64`/`f64` behind the existing status mutex.
- [ ] `metrics.rs`: render the snapshot plus counters as Prometheus text. Escape label values. One `# TYPE` line per family.
- [ ] `GET /metrics` returns `text/plain; version=0.0.4`.
- [ ] Test: after one chat on the fake engine, `/metrics` contains `silkai_loads_total{model="soap"} 1` and a `silkai_model_state{...,state="bench"} 1` line; the text parses with a tiny line-based checker (no crate).

### Task 6: `[ui]` config and bearer token

- [ ] `config.rs`: `#[serde(default)] ui: FileUi { enabled: bool (false), token: Option<String> }` → `AppConfig.ui`.
- [ ] `auth.rs`: an Axum middleware applied to `/ui`, `/metrics`, `/admin/*`. If `token` is set, require `Authorization: Bearer <token>`; compare with a constant-time equality (write a 10-line function; no crate). 401 with `WWW-Authenticate: Bearer` otherwise. If no token, pass through.
- [ ] Tests (`tests/config.rs` additions and `tests/ui.rs`): default off; token required on `/metrics` when set; `/v1/status` never requires it.
- [ ] `examples/config.toml` and `examples/llama-server.toml`: commented `[ui]` block.

### Task 7: The page

- [ ] `ui/index.html`: single file, no external requests, `prefers-color-scheme` aware, tabular numbers. Sections per spec: cards, shelf, models table, events. Polls `/v1/status` at 1 Hz, `EventSource('/v1/events')`. If the token is set, the page reads it from `localStorage` after a one-time prompt and sends it on `fetch`; `EventSource` cannot set headers, so accept `?token=` on `/v1/events` only when a token is configured.
- [ ] `GET /ui` serves it with `include_str!` when `ui.enabled`, else 404. `Cache-Control: no-store`.
- [ ] Draw the GPU bar with plain divs sized by percentage of `total_gb` (fallback `schedulable_gb + headroom` when total is unknown). Measured marker as a 2 px line. Exclusive models get a label, not a colour.
- [ ] Test (`tests/ui.rs`): enabled → 200 and body contains `<title>`; disabled → 404. No browser tests.
- [ ] Manual check on the GPU box with the llama-server config: watch a `write` request move `transcribe` off the card. Screenshot into the PR.

### Task 8: Docs

- [ ] README: a "Status page and metrics" section after "Run from a checkout": how to enable, what the page shows, `/metrics` sample, the token, and the reverse-proxy note for LAN use.
- [ ] CONTRIBUTING: the page is plain HTML on purpose; no framework PRs.

---

## Phase 2: narrow configuration (separate plan when phase 1 lands)

Prerequisite: incremental reload. `POST /admin/reload` must diff old and new `AppConfig`, keep untouched models resident, and only sleep/discard/re-create models whose fields changed. Then:

- `POST /admin/models/{name}` with `{enabled?, priority?, vram_gb?, keep_warm?}` edits the config file via `toml_edit` and triggers the incremental reload. 403 when no token is configured.
- Page controls for those four fields plus a reload button, hidden when writes are disabled.

---

## Effort

Rough, for one person familiar with the code:

| task | size |
|---|---|
| 1 overlays | half a day |
| 2 honest RAM | an hour |
| 3 sampler | half a day |
| 4 events + SSE | a day |
| 5 metrics | half a day |
| 6 config + token | half a day |
| 7 page | a day |
| 8 docs | an hour |

Phase 2 is roughly the same again, most of it in incremental reload.
