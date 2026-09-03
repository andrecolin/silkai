# SilkAI

[![CI](https://github.com/andrecolin/silkai/actions/workflows/ci.yml/badge.svg)](https://github.com/andrecolin/silkai/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

On a server with a 32 GB GPU and 128 GB of RAM, you have three different
models that can stay in RAM all day. The GPU only holds whoever is working
right now. A little VRAM is left over for normal machine functions.

Several GPUs are several benches and the same RAM shelf. An ~80% model and an
~30% model run at the same time if they sit on **different** cards. On one
card they still do not fit. **Exclusive** means that card is alone, not the
whole machine. List cards under `[[resources.gpus]]` and optionally pin a
model with `gpu = 1`. A model that must occupy several cards at once uses
`gpus = [0, 1]`; `vram_gb` is split evenly across those benches.

![128 GB RAM keeps three models warm; the 32 GB GPU holds only the model that is working, plus a small slice for the desktop](docs/silkai-memory.svg)

SilkAI is not a new model runner and not a cloud cluster. We only decide who
is on the GPU, who waits, and who stays warm in RAM.

Apps use OpenAI-style HTTP and, if you turn it on, a WebSocket per model.
Rules (who can share, who must run alone, who is live) live in a config file,
not in each request. Your frontend owns the workflow — SilkAI only runs the
model you name.

[MIT](LICENSE) · [Ko-fi](https://ko-fi.com/andrecolin)

## A simple workflow

Your app does the talking. SilkAI only sees models and text.

1. Set `transport = "websocket"` (or `"both"`) on whichever model should take
   a live socket — a speech model, a chat model, anything.
2. Open `GET /v1/session?model=transcribe`. Speak in **your** UI; send
   `{"type":"prompt","content":"..."}` (the transcript or a question). Tokens
   come back as chat on that socket. The GPU holds that model until you hang
   up or go idle.
3. When you want notes, **the frontend** sends that text to another model —
   HTTP `POST /v1/chat/completions` with `"model":"write"`, or a second
   session socket. SilkAI does not chain models for you.
4. If the writer needs the whole card, the first model parks in RAM and comes
   back in a couple of seconds. Two note jobs on the same writer wait in line
   on one load.

Same idea if you code: socket to an autocomplete model while you type, then
your IDE posts the hard question to a larger model. One card, a queue, a warm
copy in RAM.

If two models fit, they run together. If they do not, the idle one parks in
ordinary RAM and comes back in a couple of seconds — not a full reload from
disk.

## What that gives you

- **Fit together → run together.** A huge model that needs the whole card waits,
  then runs alone.
- **Live work wins.** Captions or autocomplete are not interrupted. Background
  jobs fill leftover space and leave first.
- **One loaded model, many requests.** Two summaries share the writer. No second
  20-plus-GB load.
- **Warm RAM, not a disk reload.** After the first load, switching is about 1–3
  seconds if you have the RAM.
- **Policy in config.** Clients only send `"model"`. A random script cannot
  steal the GPU from something live.

## Status

Slice 1 is a working **Linux** daemon (`127.0.0.1` only): scheduler, HTTP chat
completions, fake engines (no GPU required), optional llama.cpp behind
`--features llama` (optional `cuda` / `vulkan` / `metal`), and HTTP adapters
for vLLM (`engine = "vllm"`) and Ollama (`engine = "ollama"`).

WebSocket is a **per-model** option (`transport = "websocket"` or `"both"`).
Any configured model can take a session. An open socket holds that model’s
slot until it closes or goes idle. Speech-in, notes-out, SOAP, search — that
routing stays in your frontend.

Still to come: more adapters. The scheduler and HTTP API are meant to stay
portable (x86_64 and ARM; CUDA / Vulkan / Metal via the engine, not the core).

## Install

Needs Rust 1.80+ ([rustup](https://rustup.rs)). Fake engines need no GPU.

```bash
git clone https://github.com/andrecolin/silkai
cd silkai
./scripts/install.sh
```

That puts `silkai` in `~/.local/bin`, writes `~/.config/silkai/config.toml` if
missing, and installs a **user** systemd unit on Linux:

```bash
systemctl --user enable --now silkai
curl -s http://127.0.0.1:8080/health
```

llama.cpp (real GGUFs). `llama` is CPU; add one GPU backend:

```bash
FEATURES=llama ./scripts/install.sh
FEATURES=llama,cuda ./scripts/install.sh      # NVIDIA, needs CUDA toolkit
FEATURES=llama,vulkan ./scripts/install.sh    # Linux/Windows, needs Vulkan SDK
FEATURES=llama,metal ./scripts/install.sh     # macOS (often already on by llama.cpp)
```

Pick **one** of `cuda`, `vulkan`, or `metal`. Do not pass `--all-features`.

The `llama` build compiles llama.cpp from source (10 minutes or more) and
needs cmake, a C++ compiler, and libclang headers for bindgen (on Ubuntu:
`libclang-common-18-dev`, or whichever LLVM version you have). For `cuda`
with Ubuntu's packaged toolkit the libraries live in
`/usr/lib/x86_64-linux-gnu`, not in a `lib64` directory, so point the crate
at a directory that has one: `mkdir -p ~/cuda && ln -s
/usr/lib/x86_64-linux-gnu ~/cuda/lib64 && CUDA_LIBRARY_PATH=~/cuda ...`.
A toolkit under `/usr/local/cuda` is found on its own. If you already have a
llama.cpp build, `engine = "process"` with `llama-server` (below) skips all
of this.

Without the script (from crates.io once 0.1.0 is published, or from a
checkout):

```bash
cargo install silkai --locked
cargo install --path crates/silkai --locked
# optional: --features llama
# GPU: --features llama,cuda | llama,vulkan | llama,metal
mkdir -p ~/.config/silkai
cp examples/config.toml ~/.config/silkai/config.toml
silkai
```

Config is `~/.config/silkai/config.toml`, or `SILKAI_CONFIG`. The daemon
listens on `127.0.0.1` only. If you omit `gpu_total_gb` and `[[resources.gpus]]`,
SilkAI runs `nvidia-smi` and uses those cards (minus `gpu_headroom_gb` each).
If you omit `ram_total_gb`, it reads `/proc/meminfo` (or `sysctl hw.memsize` on
macOS) and subtracts `ram_headroom_gb`. If a probe finds nothing, the daemon
refuses to start.

## Run from a checkout

```bash
cargo run -p silkai
curl -s http://127.0.0.1:8080/health
```

The example file names the three roles above `transcribe`, `write`, and `index`:

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"write","messages":[{"role":"user","content":"Summarize this meeting."}]}'
```

The whole `messages` list reaches the engine: system prompts, history,
and assistant turns are forwarded as sent (llama-server, vLLM, and Ollama
apply the model's chat template; the in-process llama.cpp engine applies
the GGUF's own template). `content` may be a string or a list of text parts.
`max_tokens` and `temperature` are forwarded too; anything else in the
body is ignored for now. Errors come back with the reason in the body.

Streaming: `"stream": true` (SSE). Also `GET /v1/models` and `GET /v1/status`
(models plus per-GPU `used_gb` / `schedulable_gb`). The daemon logs load, wake,
sleep, discard, and preempt. If an engine load or run fails, that job is 500,
the copy is marked not resident, and the next request loads again. The daemon
stays up. A preempted generate resumes from tokens already streamed; the
client does not see the prefix twice.

Any model with `transport = "websocket"` or `"both"`:

```text
GET /v1/session?model=transcribe   →  chat tokens on that model
# your app then:
POST /v1/chat/completions          { "model": "write", "messages": [...] }
```

Send `{"type":"prompt","content":"..."}` on the socket. SilkAI does not ingest
microphone audio; your app turns speech into text (or whatever) and chooses
the next model.

See `examples/config.toml` for VRAM, `priority` (`live` | `normal` | `background`),
`exclusive`, `slots`, and `keep_warm`.

Build llama.cpp support with `cargo run -p silkai --features llama` (CPU) or
`--features llama,cuda` / `llama,vulkan` / `llama,metal`, and set
`engine = "llama.cpp"` plus a GGUF `path` on that model. `ctx_size` sets
the context window (default 4096); a prompt that does not fit is refused
with a message rather than answered with nothing. This engine is the
experimental one: it re-reads the file on every wake instead of keeping a
RAM copy, and one request runs at a time per model. `engine = "process"`
with `llama-server` is the path to prefer.

vLLM is an HTTP adapter, not a built-in runner. Start vLLM yourself (sleep
mode needs `VLLM_SERVER_DEV_MODE=1` and `--enable-sleep-mode`), then:

```toml
engine = "vllm"
path = "Qwen/Qwen3-0.6B"          # model id sent to vLLM
url = "http://127.0.0.1:8000"     # optional; this is the default
gpu = 0                           # pin SilkAI's budget to the card vLLM uses
```

SilkAI posts `/wake_up`, `/sleep?level=1`, and streaming `/v1/chat/completions`.
It does not spawn or stop the vLLM process.

Ollama is the same idea: start Ollama yourself, then:

```toml
engine = "ollama"
path = "llama3.2"                 # Ollama model name
url = "http://127.0.0.1:11434"    # optional; this is the default
```

Load/wake POST `/api/generate` with `keep_alive = -1`. Sleep unloads with
`keep_alive = 0` (Ollama has no RAM shelf). Chat is streaming `/api/chat`.
SilkAI does not spawn or stop Ollama.

`engine = "process"` starts and stops a child for you. Anything that speaks
OpenAI chat and answers `GET /health` works: `llama-server`, `vllm serve`,
and similar. Load waits until `/health` returns 200 (llama-server says 503
while the GGUF is still loading; up to 5 minutes). Chat is streaming
`/v1/chat/completions`. Sleep kills the process group (no RAM shelf, the
next wake is a fresh start). The child gets `CUDA_VISIBLE_DEVICES` set to
the card the scheduler picked, and its stderr goes to SilkAI's log:

```toml
engine = "process"
path = "write"                    # model name sent in the request (--alias)
url = "http://127.0.0.1:8001"     # must match the child's port
cmd = ["llama-server", "--model", "/models/write-q4.gguf", "--alias", "write",
       "--port", "8001", "--n-gpu-layers", "999", "--jinja"]
```

`examples/llama-server.toml` is a complete three-model setup on this
pattern. It needs no `--features` on SilkAI; the GPU work is in llama.cpp.

## Status page and metrics

Every daemon serves `GET /v1/status` (tiers, a `loading` state while a
model is on its way to the card, the configured `vram_gb` budget beside
what nvidia-smi measures for that model's process, queue depth, open
sessions), `GET /v1/events` (Server-Sent Events: the last 500 scheduler
events replayed, then live; `?after=<seq>` skips the replay), and
`GET /metrics` in Prometheus text format:

```text
silkai_gpu_measured_used_gb{gpu="0"} 22.1
silkai_model_state{model="write",state="bench"} 1
silkai_model_queued{model="write"} 0
silkai_loads_total{model="write"} 4
silkai_load_seconds_sum{model="write"} 33.2
```

The page is optional and off by default. Turn it on and open
`http://127.0.0.1:8080/ui` to see each card drawn to scale, the shelf, and
the event log:

```toml
[ui]
enabled = true
token = "change-me"   # optional; guards /ui, /metrics and /admin/*
```

The token is sent as `Authorization: Bearer <token>` by tools, or typed as
the password when the browser asks (any user name). `/v1/*` never needs it.
The daemon still binds to loopback only; to reach the page from another
machine put a reverse proxy with TLS in front of it. The page loads nothing
from the internet. Changing `[ui]` needs a restart.

## Development

```bash
cargo test
cargo test --features llama
# GPU backends need the matching SDK; they are not part of `cargo test`.
```

Scheduler tests are numeric GB; they do not need a GPU.

```bash
SILKAI_ITEST=1 SILKAI_GGUF_A=/path/tiny.gguf SILKAI_GGUF_B=/path/small.gguf \
  cargo test -p silkai-server --features llama --test itest_llama -- --nocapture
```

PRs welcome; no CLA. `main` is protected — changes land through a pull
request with green CI. Keep CUDA (and other GPU SDK types) out of
`silkai-sched`. See [CONTRIBUTING.md](CONTRIBUTING.md),
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md), and [SECURITY.md](SECURITY.md).

## License and coffee

SilkAI is free software under the [MIT License](LICENSE). Use it, fork it, ship
it in a product. If it saves you a second GPU — or an afternoon of restarts —
[a coffee is appreciated](https://ko-fi.com/andrecolin). The daemon does not
nag, phone home, or lock features.
