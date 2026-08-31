# SilkAI GPU Capacity Scheduler

A local daemon that owns a machine’s GPU memory — one card or several — and
runs multiple models by packing, priority, and warm RAM residency. Apps talk
HTTP or WebSocket. They never allocate GPU memory themselves.

## Problem

A small-business or power-user PC typically has one GPU (example: 32 GB) and
plenty of system RAM (example: 128 GB). Several models are needed in one
workflow — live dictation, a large specialist generator, a background scanner —
but they do not all fit on the GPU at once. Today each app loads its own model
and leaves it resident, so the next model OOMs or never starts.

Idle models still occupy VRAM. Cold reloads from disk take tens of seconds.
The machine’s desktop also needs a slice of VRAM or the display hitches.

## Goals

- One process on the machine is the only GPU owner for these models.
- If models fit **on a given GPU**, they run together. If they do not, they queue
  or evict by policy. Several GPUs are several benches, one shared RAM shelf.
- An ~80% VRAM model and an ~30% VRAM model run at the same time **when they
  land on different GPUs**. On the same GPU they still do not fit (110%).
- After first load, a model used again comes from RAM (about 1–3 seconds), not disk.
- Live dictation is never bumped while a session is open.
- Exclusive models (large SOAP-style generators) run alone.
- Any app can call in: OpenAI-shaped HTTP, WebSocket for live audio.
- Linux first; scheduler and API portable to other OS/CPU/GPU backends.

## Non-goals (this project)

- Transparent interception of foreign CUDA/Vulkan processes (Nixie-style).
- Replacing Whisper, llama.cpp, or vLLM kernels.
- Multi-machine / cluster scheduling (several GPUs in **one** box are in scope).
- Claiming 100% of VRAM (desktop always keeps a reserved slice).
- A kernel module or CUDA virtual-memory hypervisor.
- Windows/macOS installers in the first slice (design must not block them).
- License keys, paid tiers, or a hosted cloud service. Public release is free software with an optional donation link.

## Users

- Clinic / small-business box: two doctors dictating, SOAP notes after, chart
  scan in the gaps.
- Power user running several local models on one card, or a small server with
  two or more GPUs.

Example machine used throughout this spec (single GPU):

- GPU 32 GB, leave 3 GB for the desktop → **29 GB schedulable**
- RAM 128 GB, leave 32 GB for the OS → **96 GB shelf**
- Models: Whisper 12 GB live shareable 2 slots; SOAP 28 GB exclusive 1 slot;
  chart-scan 10 GB background shareable 1 slot

Example machine (two GPUs) — required, not yet implemented in slice 1:

- GPU 0 and GPU 1, 32 GB each, 3 GB headroom each → **29 GB schedulable per card**
- Same 128 GB RAM shelf for all warm copies
- Large writer ~80% of one card (~26 GB) on GPU 0; indexer ~30% (~10 GB) on
  GPU 1; both resident at once. Exclusive means **that card** is alone, not
  the whole machine.

## Alternatives

### A. Transparent GPU interposer

`LD_PRELOAD` on `cudaMalloc` / kernel launch. Existing Ollama/vLLM keep working
unmodified. Fragile across drivers, misses NVENC/Vulkan, NVIDIA-shaped.
Rejected for v1: we can require apps to call us.

### B. Process proxy only (llama-swap style)

Start/stop whole servers per model name. Fast to ship, weak packing (hard to
keep Whisper 12 GB + chart-scan 10 GB resident as one coherent schedule),
cold-ish process starts.

Useful later as an adapter for Ollama, not the core.

### C. Library + daemon + engine adapters (chosen)

Rust library is the scheduler. A daemon exposes HTTP/WebSocket and owns
adapters (`whisper.cpp`, `llama.cpp`, optional `vllm`). Config file is policy.
Apps send a model name only.

This is the only shape that gives packing, warm shelf, and “any app” without
owning every GPU API on the planet.

## Architecture

```
┌────────────┐  HTTP   ┌─────────────────────────────────┐
│ Clinic app │────────►│  silkai daemon                  │
│ Open WebUI │  WS     │  • HTTP + WebSocket frontends   │
│ scripts    │────────►│  • config                       │
└────────────┘         │  • scheduler (library)          │
                       │  • shelf (RAM copies)           │
                       │  • adapters ─────────────────┐  │
                       └──────────────────────────────│──┘
                                                      ▼
                                         whisper.cpp / llama.cpp / (vllm)
                                         CUDA | Vulkan | Metal | CPU
```

Four crates (or modules) with one job each:

| Unit | Does | Depends on |
|---|---|---|
| `silkai-sched` | Pack, priority, exclusive, slots, eviction, queue | Nothing GPU-specific |
| `silkai-adapters` | Load, sleep-to-RAM, run job, report VRAM | Engine SDKs |
| `silkai-server` | HTTP, WebSocket, config load, wire jobs to scheduler | both above |
| `silkai` bin | Process entry, logging, listen socket | `silkai-server` |

C is not the implementation language. A `cdylib` C ABI is out of scope until
a vendor app must embed us in-process.

### Portability

- Scheduler and API: OS/ISA/GPU agnostic. Linux x86_64 and aarch64 first.
- Default engines: whisper.cpp and llama.cpp (CUDA, Vulkan, Metal, CPU).
- vLLM: optional adapter, Linux + NVIDIA only.
- Bench/shelf *policy* is universal. *How* a copy happens is adapter-specific
  (PCIe copy on NVIDIA; near-noop on unified memory).

## Configuration

Admin-owned TOML. Apps cannot set priority, exclusive, slots, or VRAM on a
request. That would let a script steal the bench from dictation.

Path: `~/.config/silkai/config.toml`, override with `SILKAI_CONFIG`.

```toml
listen = "127.0.0.1:8080"

[resources]
gpu_total_gb = 32          # required if probe fails; daemon refuses to start if unknown
gpu_headroom_gb = 3
ram_headroom_gb = 32
prefetch_on_start = true
request_timeout_secs = 600 # queued+running HTTP jobs; then 504

[models.whisper]
engine = "whisper.cpp"
path = "/models/whisper-large-v3.bin"
vram_gb = 12
priority = "live"          # live | normal | background
exclusive = false
slots = 2
keep_warm = true
transport = "websocket"    # websocket | http | both
idle_timeout_secs = 45     # WS with no audio: drop slot

[models.soap]
engine = "llama.cpp"
path = "/models/soap-q4.gguf"
vram_gb = 28
priority = "normal"
exclusive = true
slots = 1
keep_warm = true
transport = "http"

[models.chart-scan]
engine = "llama.cpp"
path = "/models/chart-scan-q4.gguf"
vram_gb = 10
priority = "background"
exclusive = false
slots = 1
keep_warm = true
transport = "http"
```

### Field semantics

| Field | Rule |
|---|---|
| `name` | Request `"model"` value |
| `engine` | Adapter key |
| `path` | Cupboard (disk) |
| `vram_gb` | Cost of having this model **resident** with up to `slots` jobs. Trusted up front. First real load may log a warning if measured use differs by >10% |
| `priority` | `live` > `normal` > `background` |
| `exclusive` | Resident only when no other model is resident **on the same GPU** |
| `gpu` | Optional device pin (`0`, `1`, …). Omit = place on any GPU that has room |
| `slots` | Max concurrent jobs on that one resident copy. Extra jobs queue on the model (async accept, sequential or N-way run) |
| `keep_warm` | On GPU unload, keep weights in RAM (shelf) |
| `transport` | How clients should connect; server may still expose HTTP file-upload for a `websocket` model |
| `idle_timeout_secs` | WebSocket only; missing audio/heartbeat releases the slot |

`gpu_schedulable = gpu_total_gb - gpu_headroom_gb` on **each** GPU.

Slice 1 config keeps a single `[resources] gpu_total_gb` (one bench). Multi-GPU
config (later slice) lists devices:

```toml
[resources]
ram_total_gb = 128
ram_headroom_gb = 32

[[resources.gpus]]
id = 0
total_gb = 32
headroom_gb = 3

[[resources.gpus]]
id = 1
total_gb = 32
headroom_gb = 3

[models.write]
vram_gb = 26          # ~80% of 32 GB
exclusive = true      # GPU 0 alone; GPU 1 still free
# gpu = 0             # optional pin; default: first GPU that fits

[models.index]
vram_gb = 10          # ~30% of 32 GB
exclusive = false
```

Shelf budget = total RAM − `ram_headroom_gb`. If keep-warm copies would exceed
it, demote least-recently-used warm model to disk (cupboard).

Reload: `SIGHUP` or `POST /admin/reload` re-reads TOML. In-flight jobs keep
the old model record; new jobs use the new one. Changing `path`/`engine` on a
resident model forces sleep then load on next job.

## Scheduler

The scheduler is a pure function of: config, resident set, running jobs, queue.
Adapters execute decisions. Unit tests do not need a GPU: they use a fake
adapter and a numeric GB budget.

### Multiple GPUs (requirement)

Each physical GPU is its own **bench** with its own schedulable GB. System RAM
is still **one shelf** for all warm copies.

| Placement | Fits? |
|---|---|
| Model A 80% and model B 30% on **one** GPU | No (110%). Queue or evict as today. |
| Model A 80% on GPU 0, model B 30% on GPU 1 | Yes. Both resident. No swap. |
| Exclusive model on GPU 0 | GPU 0 has no neighbors. GPU 1 may still run others. |
| Live model pinned to a GPU | That card does not evict it while the session is open. Other cards may run writers. |

Default placement: pack onto the first GPU with enough free VRAM (and exclusive
rules). Optional `gpu = N` pins the model.

**Not in the first multi-GPU slice:** one model **split across** cards (llama.cpp
layer/tensor split occupying GPU 0 and 1 together). That is a later occupancy
type (“this job holds a set of devices”). Until then, each model maps to **one**
GPU.

Slice 1 implementation stays a single `gpu_schedulable_gb`. Multi-GPU packing
is a required later slice, not an optional extra.

### Tiers

| Tier | Meaning | Restore time (PCIe 4, process kept alive) |
|---|---|---|
| Bench | Weights on GPU | 0 |
| Shelf | Weights in RAM, not GPU | ~0.5–2 s for ~10–12 GB; ~1.5–4 s for ~28 GB |
| Cupboard | Disk only | ~3–20 s depending on size and NVMe |

With 128 GB RAM and three keep-warm models (~50 GB), all three stay on the
shelf after first load / prefetch. Switching is a bench↔shelf copy, not a
disk read.

Keeping warm means **duplicate** weights in RAM while on the bench, so eviction
off the GPU is free (unmap / free VRAM) and restore is RAM→GPU only.

### Job lifecycle

1. Frontend accepts the request immediately (HTTP connection stays open, or
   WebSocket accepts). That is asynchronous intake.
2. Job is queued on that model.
3. Scheduler tries to admit.
4. On admit: ensure model on bench (prefetch from shelf, else cupboard), then
   run on a free slot.
5. On completion or WS close/idle: release slot. If the model has no running
   jobs and someone else needs the bench, sleep to shelf (if `keep_warm`) or
   cupboard.

Two HTTP clients may POST `soap` at the same time. Both are accepted. With
`slots = 1` they run one after the other on the single resident SOAP. No
second 28 GB load.

### Admission and packing

Let `used` = sum of `vram_gb` of **resident** models (not per job).

Place model M on the bench if:

- `M.vram_gb <= gpu_schedulable`, else reject (model bigger than the machine).
- If `M.exclusive`: no other model may stay resident.
- If any resident model is `exclusive` and is not M: M cannot join; it must
  wait or that exclusive must leave.
- If not exclusive: `used + M.vram_gb <= gpu_schedulable` after evictions.

Same model already resident and `running < slots`: admit, no extra VRAM.

Same model already resident and `running == slots`: queue on M, do not load
another copy.

### Priority and preemption

| Event | Rule |
|---|---|
| `live` job arrives, M not resident, need space | Evict idle models, then **preempt running non-live** jobs (pause, requeue at the head of their model queue, sleep their model) |
| `live` job running | Never preempt. SOAP and chart-scan wait |
| `exclusive` job (SOAP) | Starts only when **no `live` slots are held**. May evict idle models and preempt running `background` or `normal` (not `live`) so it can have the bench alone |
| `background` | Only admitted if it fits beside current residents; first to evict |

Whisper WebSocket **holds** a live slot until disconnect or `idle_timeout_secs`.
Two doctors = two sockets = two slots on one Whisper.

If SOAP is generating and a doctor starts dictation: SOAP is **stopped between
tokens**, requeued at the head of the SOAP queue, Whisper takes the bench.
When dictation ends, SOAP is loaded from the shelf and the job **restarts from
the original prompt** (v1 does not snapshot KV). Adapters poll a pause/cancel
flag between tokens.

### Eviction order (when space is needed)

1. Resident models with zero running jobs, lowest priority first, oldest idle first.
2. Then running `background` jobs (preempt).
3. Then running `normal` jobs (preempt), only if the incoming job is `live`
   or the incoming job is `exclusive` and the running job is not `live`.
4. Never evict running `live`.

After eviction, if still no fit, the incoming job stays queued (except a model
larger than `gpu_schedulable`, which fails).

### Prefetch

If `prefetch_on_start`: for every `keep_warm` model, load weights into the
shelf (RAM) without necessarily placing them on the bench. First interactive
call still pays 1–3 s GPU copy, not a disk load.

Do not place all models on the bench at start. Bench starts empty (plus
headroom).

## Public release and license

After slice 1 is validated on a real box (packing, warm shelf, exclusive vs
share, HTTP), the project is published as a public GitHub repository.

**License: MIT.** One `LICENSE` file from the first commit so we never have to
relicense later. Anyone (including a clinic or a vendor) may use, copy, modify,
and sell products that include SilkAI, without paying us. Attribution stays
with the copyright notice.

MIT is the default because it matches “free tool, coffee if you want.”
Apache-2.0 is the alternative if we later want an explicit patent grant; it is
still free. We will not use GPL/AGPL: those would scare the small-business and
embedder audience this is for.

**Coffee:** the GitHub README (and the repo About blurb) include a voluntary
link — GitHub Sponsors and/or Ko-fi / “buy me a coffee.” The daemon does not
nag, does not disable features, and does not phone home. Paying is never
required to run models.

**Repo hygiene at first public commit:** `LICENSE`, `README.md` (what it is,
Linux install, config example, coffee link), `CONTRIBUTING.md` (tiny: tests
before PR, scheduler stays GPU-agnostic). No CLA.

## Frontends

### HTTP (OpenAI-shaped)

Listen on `listen`.

| Route | Use |
|---|---|
| `GET /health` | Process up |
| `GET /v1/models` | Configured names |
| `GET /v1/status` | Resident / shelf / queue (ops) |
| `POST /v1/chat/completions` | Chat/generate (`model` required) |
| `POST /v1/audio/transcriptions` | File upload to a whisper-capable model |
| `POST /admin/reload` | Re-read config (localhost only) |

Streaming: SSE / chunked as OpenAI does. The HTTP request is accepted at once;
if the job is queued, the stream opens with no tokens until the job admits
(or a `comment` / role-less chunk is sent so proxies do not time out — send
an SSE comment every 15 s while queued).

Errors: unknown model `404`; model larger than GPU `400`; timeout `504`.

### WebSocket

`GET /v1/audio/stream?model=whisper` upgrades to WebSocket.

Protocol (text JSON control, binary audio):

- Server → `{ "type": "queued" }` | `{ "type": "warming" }` | `{ "type": "live" }`
- Client → binary PCM chunks (config default: 16 kHz s16le mono) or JSON `{ "type": "stop" }`
- Server → `{ "type": "partial", "text": "..." }` / `{ "type": "final", "text": "..." }`
- Idle: no audio for `idle_timeout_secs` → `{ "type": "idle_close" }` and close

A live socket holds one slot. Heartbeat: client ping at least every 15 s or
send audio.

### Binding

Default `127.0.0.1` only. This is a local machine agent, not a public API.
TLS out of scope for v1.

## Adapters

Each adapter implements:

```
load(path) -> ResidentHandle      # cupboard or shelf → bench
sleep(handle) -> WarmHandle       # bench → shelf (keep RAM copy)
discard(warm)                     # shelf → cupboard
run(handle, job) -> stream        # must honor pause/cancel
measured_vram(handle) -> gb
```

v1 adapters:

1. **llama.cpp** — SOAP, chart-scan; GGUF; CUDA/Vulkan/CPU as built.
2. **whisper.cpp** — dictation; file HTTP + WebSocket streaming.

v1.1 optional: **vllm** if the binary is present and `engine = "vllm"`.

Ollama is not required. If we add it later, it is “start/stop child + HTTP”,
not the scheduler.

Adapter failures (engine crash): job errors, model marked not-resident, retry
load on next admit once. Do not take down the daemon.

## Example timelines

### Two doctors, then two SOAP notes (same SOAP model)

```
t0  Doctor A WS connect  → Whisper to bench, slot 1 live
t1  Doctor B WS connect  → Whisper slot 2 (12 GB still, not 24)
    SOAP POSTs from both → queued (exclusive + live held)
t2  Both WS close
    Whisper idle → sleep to shelf
    SOAP to bench (~2–3 s from shelf)
    SOAP note A runs, then note B (slots=1, same resident)
```

### Dictation during SOAP

```
SOAP running exclusive
Doctor A WS connect (live)
  → pause SOAP, sleep SOAP, Whisper to bench, dictation live
WS close
  → Whisper sleep, SOAP to bench, restart the note from the original prompt
```

v1 does not snapshot KV. A preempted generate is retried. KV resume is slice 3.

## Error handling

| Case | Behavior |
|---|---|
| Unknown model | 404 |
| `vram_gb` > schedulable | That model is disabled at config load (logged); request returns 400. Other models still run |
| Engine missing | Config load warning; that model’s requests 503. Daemon still starts |
| Queue wait too long | `request_timeout_secs` (default 600) then 504; job dropped |
| WS idle | Close, release slot |
| Adapter panic/crash | Isolate; 500 to that job; model unloaded |
| Config invalid | Daemon refuses to start / reload keeps last good config |
| RAM shelf full | Demote LRU warm to cupboard; log |

## Testing

Scheduler tests (no GPU, fake adapter, numeric GB):

- Pack 12+10 on 29 GB schedulable; reject placing 28 beside them.
- Exclusive model never admits while a `live` slot is held.
- Two jobs on `slots=1`: both accepted, serial run, one `load`.
- Two jobs on `slots=2`: one `load`, two running.
- `live` arrival preempts running `normal`; victim requeued at head.
- `background` never preempts `live`.
- Exclusive arrival evicts idle neighbors and preempts `background`.
- Shelf demote when warm copies exceed RAM budget.
- Prefetch does not occupy the bench.
- (Later slice) 80%+30% fail on one GPU budget; succeed when assigned to two
  GPU budgets. Exclusive on GPU 0 does not evict a resident on GPU 1.

Server tests (slice 1): HTTP SSE queue comments; `/v1/status` reflects resident/shelf.

Server tests (slice 2): WebSocket slot hold and idle timeout.

Optional GPU integration, gated by `SILKAI_ITEST=1`: tiny GGUF (slice 1); tiny whisper (slice 2).

## First slice vs later

### Slice 1 (next implementation plan — this is the first build)

Linux daemon, TOML config, `silkai-sched` with a fake adapter and a real GB
budget, llama.cpp adapter, HTTP `/v1/chat/completions` + `/v1/models` +
`/v1/status` + `/health`, warm sleep for `keep_warm`, priority + exclusive +
slots, live-preempts-normal using two GGUF models (no Whisper yet). Prove
packing, async queue, and 1–3 s shelf restore. Include `LICENSE` (MIT) and a
README stub from the first commit; the coffee link can stay commented until
the public repo exists.

### Slice 2

whisper.cpp + WebSocket dictation, live preemption, idle timeout.

### Slice 3

Multiple GPUs: N benches, one RAM shelf; per-GPU packing; optional `gpu` pin;
exclusive is per card. Prove 80%+30% colocated on two devices and rejected on
one. Status JSON lists resident GPU per model.

### Slice 4

Optional vLLM adapter; Vulkan build matrix; `/v1/audio/transcriptions` file path
if not done in slice 2; KV resume after preemption; one model spanning several
GPUs (layer/tensor split).

## Key decisions

1. **Daemon is the product; library is the brain.** Any-app integration is a URL,
   not a linked SDK.
2. **Policy in config, not in requests.** Prevents queue jumping.
3. **VRAM cost is per resident model, not per job.** Slots share one copy.
4. **Async intake, possibly serial run.** Two SOAP POSTs do not mean two 28 GB
   copies.
5. **WebSocket = lease on a live slot.** That is how dictation blocks SOAP.
6. **Warm RAM copies, not CUDA VMM.** Portable; 128 GB RAM makes 1–3 s switches.
7. **llama.cpp/whisper.cpp default, vLLM optional.** Portability over peak SOAP
   throughput on day one.
8. **Preempted generate restarts from the prompt in v1.** Avoids KV snapshot
   complexity.
9. **Localhost only.** This is a workstation agent.
10. **Desktop headroom is first-class.** Never schedule to 100% of a GPU.
11. **MIT, public GitHub, optional coffee.** Validated then released; no paid
    gate. Donation is a README link only.
12. **Several GPUs are several benches, one RAM shelf.** 80% + 30% on one card
    still fails; on two cards they both stay loaded. Exclusive is per GPU.
    Slice 1 is one bench; multi-GPU is a required follow-on, not a different
    product. One model split across cards is later than simple placement.

## Open questions (resolved in conversation)

- Integration: daemon + HTTP/WS, not library-only. **Resolved.**
- Whisper vs SOAP: live, not bumped while speaking; SOAP exclusive. **Resolved.**
- Two doctors: same Whisper, `slots=2`; same SOAP, `slots=1` async queue. **Resolved.**
- Transport: Whisper WebSocket, SOAP HTTP. **Resolved.**
- Linux first, portable scheduler. **Resolved.**
- 128 GB RAM → all sample models stay warm. **Resolved.**
- Multi-GPU: N benches, one shelf; 80%+30% on different cards. **Resolved
  (requirement). Slice 1 remains single-GPU; implement in slice 3.**
