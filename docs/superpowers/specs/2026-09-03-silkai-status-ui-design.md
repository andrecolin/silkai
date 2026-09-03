# SilkAI Status Page and Metrics

A built-in, optional web page that shows what is on each GPU, what is
waiting, and what the card actually measures, plus a Prometheus endpoint for
the servers that will never open a browser. Configuration editing comes
later and stays narrow.

**Status:** design, not yet implemented. Plan:
`docs/superpowers/plans/2026-09-03-silkai-status-ui.md`.

## Problem

Today the only window into the scheduler is `GET /v1/status` and the daemon
log. Three things make that hard to use:

- A model that is loading shows as `shelf` with nothing running or queued.
  On a 22 GB writer that is a ten-second lie.
- `ram_used_gb` sums the budgets of "warm" models even for engines that hold
  no RAM copy (process children, and in practice the in-process llama.cpp
  engine, which mmaps the file and relies on the page cache).
- `used_gb` per GPU is the sum of configured `vram_gb`, not what the card
  reports. A user with the wrong `vram_gb` never finds out until something
  fails to load.

The project's pitch is a picture: a card as a bench, RAM as a shelf, models
moving between them. Nothing in the daemon draws that picture.

## Goals

- One page at `/ui`, embedded in the binary, no external assets, no build
  step, no JavaScript toolchain in the repo. Works on an offline server.
- Truthful status: a loading state, measured VRAM beside budgeted VRAM, RAM
  only for engines that hold a shelf copy.
- A `GET /metrics` endpoint in Prometheus text format so headless servers get
  the same numbers in Grafana without the page.
- Off by default. `[ui] enabled = true` turns the page on. The JSON and
  metrics endpoints are always on; they are the API.
- Optional bearer token for `/ui`, `/metrics`, and `/admin/*`. Binding stays
  loopback-only as today; LAN exposure goes through a reverse proxy.

## Non-goals

- A chat UI. Frontends own the workflow; the daemon only shows the card.
- Adding models through the browser. The command line for a `process` child
  belongs in the config file next to its comments and examples.
- TLS, users, roles. A single shared token is the whole auth story.
- Historical charts. The page shows now and the last few hundred events.
  Prometheus keeps history.

## Users and what they get

**Single-GPU home lab (the README's first reader).** Opens `localhost:8080/ui`
and watches the writer push the small model off the card. Learns the
product in ten seconds. Adjusts `vram_gb` because the page shows the budget
was 3 GB and the card says 3.5 GB.

**Small server operator (the stated primary target).** Never opens the page.
Scrapes `/metrics` into an existing Grafana. Alerts on queue depth and load
failures. Reads `/v1/events` from a script when something looks wrong.

**Integrator debugging a slow request.** Sees the request queued behind an
exclusive load in the events stream, with the load's duration, instead of
guessing from timing.

**Issue reporter.** Pastes a screenshot of the page or the JSON from
`/v1/status`. Maintainers see the tiers, budgets, and measured numbers in one
place instead of asking five questions.

## Configuration

```toml
[ui]
enabled = false          # default: off. Servers do not need the page.
token = ""               # optional. When set, /ui, /metrics and /admin/*
                         # require `Authorization: Bearer <token>`.
```

`enabled = false` means `/ui` returns 404. `/v1/*` and `/metrics` are always
served. The token, when set, also guards `/admin/reload`, which is
unauthenticated today.

## Status model changes

`ModelStatus` gains fields; existing ones keep their meaning.

| field | type | meaning |
|---|---|---|
| `state` | string | `cupboard`, `shelf`, `loading`, `bench`, `sleeping` |
| `engine` | string | `process`, `llama.cpp`, `vllm`, `ollama`, `fake` |
| `budget_gb` | number | the configured `vram_gb` |
| `measured_gb` | number or null | VRAM the card attributes to this model's process, if known |
| `priority` | string | `live`, `normal`, `background` |
| `exclusive` | bool | |
| `slots` | number | |
| `sessions` | number | open WebSocket sessions holding the model |

`tier` stays for compatibility and mirrors `state` collapsed to the three
scheduler tiers. `loading` and `sleeping` are runtime overlays: the runtime
records the model name when it begins applying `Load`/`Wake` or `Sleep` and
clears it when the engine call returns. The scheduler itself does not change.

`GpuStatus` gains `measured_used_gb` and `total_gb` from a sampler that runs
`nvidia-smi --query-gpu=memory.used,memory.total` and
`--query-compute-apps=pid,used_memory` every two seconds in a background
task. Process-engine children are matched by pid; the in-process engine is
matched by the daemon's own pid. When nvidia-smi is missing (macOS, Vulkan
boxes) the measured fields are null and the page says so.

`ram_used_gb` counts only models on the shelf whose engine reports
`has_shelf()`. Today no engine does, so the number becomes 0 and stops
implying a RAM copy that does not exist. The field stays for the day a real
shelf lands.

Negative zero in `gpu_used_gb` is normalised.

## Events

The runtime keeps a ring of the last 500 events:

```json
{"t":"2026-09-03T15:56:11.540Z","kind":"preempt","job":3,"model":"write"}
{"t":"...","kind":"load","model":"write","gpu":0,"ms":8210}
{"t":"...","kind":"sleep","model":"transcribe","ms":112}
{"t":"...","kind":"fault","model":"index","error":"process exited before ready"}
```

Kinds: `warm`, `load`, `wake`, `sleep`, `discard`, `start`, `finish`,
`preempt`, `fault`, `session_open`, `session_close`, `reload`.

`GET /v1/events` is Server-Sent Events: it replays the ring, then streams.
`?after=<seq>` skips the replay. Same events also go to `tracing` as today.

## Metrics

`GET /metrics`, Prometheus text format, hand-written (no crate):

```
silkai_gpu_schedulable_gb{gpu="0"} 28
silkai_gpu_budget_used_gb{gpu="0"} 26
silkai_gpu_measured_used_gb{gpu="0"} 22.1
silkai_model_state{model="write",state="bench"} 1
silkai_model_running{model="write"} 1
silkai_model_queued{model="write"} 0
silkai_model_sessions{model="transcribe"} 1
silkai_loads_total{model="write"} 4
silkai_load_seconds_sum{model="write"} 33.2
silkai_preempts_total{model="write"} 1
silkai_faults_total{model="index"} 0
```

Counters live in the runtime and reset on restart, which is what Prometheus
expects.

## The page

One HTML file, `crates/silkai-server/ui/index.html`, pulled in with
`include_str!`. Plain CSS and JavaScript, no framework, under 400 lines.
Respects the browser's light/dark preference. Polls `/v1/status` once a
second and holds one `EventSource` on `/v1/events`.

Layout, top to bottom:

1. **Cards.** One horizontal bar per GPU, drawn to `total_gb`. Segments for
   each model on the bench, sized by `budget_gb`, with a thin marker at
   `measured_gb` when known. Headroom hatched. Exclusive models labelled.
2. **Shelf.** The models not on a card, with state (`shelf`, `loading`,
   `sleeping`, `cupboard`), queue depth, and open sessions.
3. **Models table.** Name, engine, priority, budget, measured, slots,
   running, queued, transport, context size where the engine knows it. The
   in-process llama.cpp engine shows its 512 or configured context here, so
   its limits are visible rather than a README footnote.
4. **Events.** Newest first, the last 200, with load durations.

No configuration controls in this phase.

## Phase 2: narrow configuration

Only after reload is incremental. Today `POST /admin/reload` builds a new
runtime and drops the old one, which kills every child. Phase 2 needs a
reload that diffs the config and restarts only the models whose fields
changed.

Then, on the page, per model: enable/disable, priority, `vram_gb`,
`keep_warm`, and a reload button. Writes go to `POST /admin/models/{name}`
which edits the config file with `toml_edit` (comments preserved) and
triggers the incremental reload. Requires the bearer token to be set; with
no token configured the endpoints return 403 and the page hides the
controls.

## Security

- Loopback bind is unchanged. The page is reachable from the box only unless
  the operator proxies it.
- The token is compared in constant time. It guards `/ui`, `/metrics`, and
  `/admin/*`. It does not guard `/v1/chat/completions`; that is a separate
  decision and out of scope here.
- The page carries no external script or font. Nothing phones home.

## Alternatives considered

**Separate frontend repo (React, Vite).** Better looking, but adds a
JavaScript build to a project that ships via `cargo install`, and a second
thing for the community to keep working. Rejected for phase 1; a community
member can build one on the JSON and SSE endpoints, which is the point of
keeping those first-class.

**Metrics via the `prometheus` crate.** Fine, but the format is simple and
the crate adds dependencies to a daemon that wants a fast pure-Rust build.
Hand-write it; revisit if histograms are wanted.

**Feature flag for the page.** A cargo feature would keep the HTML out of
the binary, saving about 30 KB. Not worth a feature matrix entry; a config
toggle is enough.
