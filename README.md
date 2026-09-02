# SilkAI

On a server with a 32 GB GPU and 128 GB of RAM, you have three different
models that can stay in RAM all day. The GPU only holds whoever is working
right now. A little VRAM is left over for normal machine functions.

Several GPUs are several benches and the same RAM shelf. An ~80% model and an
~30% model run at the same time if they sit on **different** cards. On one
card they still do not fit. **Exclusive** means that card is alone, not the
whole machine. List cards under `[[resources.gpus]]` and optionally pin a
model with `gpu = 1`.

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

Without the script:

```bash
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
If the probe finds nothing, the daemon refuses to start.

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
`engine = "llama.cpp"` plus a GGUF `path` on that model.

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

PRs welcome; no CLA. Keep CUDA (and other GPU SDK types) out of `silkai-sched`.
See [CONTRIBUTING.md](CONTRIBUTING.md).

## License and coffee

SilkAI is free software under the [MIT License](LICENSE). Use it, fork it, ship
it in a product. If it saves you a second GPU — or an afternoon of restarts —
[a coffee is appreciated](https://ko-fi.com/andrecolin). The daemon does not
nag, phone home, or lock features.
