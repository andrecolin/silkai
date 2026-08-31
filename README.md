# SilkAI

A local GPU **capacity scheduler**. One daemon owns the card. Models pack when
they fit, queue when they do not, and park in RAM so the next call is a copy,
not a cold load from disk.

Apps talk OpenAI-shaped HTTP. Policy lives in a config file, not in each
request.

[MIT](LICENSE) · [Ko-fi](https://ko-fi.com/andrecolin)

## Why

If you run models locally, each app grabs VRAM and keeps it. Whisper for
dictation, a large medical (or legal, or coding) model for the write-up, a
smaller scanner in the background — on a 32 GB card they cannot all sit on the
GPU at once. Today you close one program so another can start, or you wait on a
full reload.

That is a bad fit for a small clinic, a firm, or a power user with **one
machine**. The interesting work is a *chain*: transcribe, then generate notes,
then maybe scan a chart. Those jobs rarely need the same model at the same
instant. They need a fair owner of the workbench.

SilkAI is that owner.

It is not a new inference engine, and it is not a Kubernetes cluster stack.
Engines (llama.cpp, a fake in-process engine for tests, later Whisper) still do
the tokens. We decide **who is on the GPU**, **who waits**, and **who stays
warm in RAM**.

## What we contribute

- **Packing.** Two models that fit run together. A 28 GB exclusive job does not
  sit beside them; it waits, then takes the card alone.
- **Priority.** Live work (dictation) is never bumped while it is running.
  Background work fills the gaps and is first to leave.
- **Slots.** Two doctors can share one Whisper copy. Two SOAP notes on the same
  28 GB model are accepted at once and run one after the other — no second load.
- **Warm shelf.** With enough system RAM (e.g. 128 GB), weights stay in memory
  after they leave the GPU. Switching is about a second or two, not tens of
  seconds from disk. We leave a slice of VRAM for the desktop so the machine
  stays usable.
- **Config as policy.** Priority, exclusive, slots, and VRAM are in TOML.
  Clients only send `"model"`. A script cannot jump the queue.

A typical clinic box: 32 GB GPU, 128 GB RAM, Whisper live and shareable, SOAP
exclusive, a chart scanner in the background. Two people dictating at once;
notes generate after they stop talking.

## Status

Slice 1 is a working **Linux** daemon (`127.0.0.1` only): scheduler, HTTP chat
completions, fake engines (no GPU required), optional llama.cpp behind
`--features llama`.

Still to come: Whisper over WebSocket (a live socket holds a high-priority
slot), more adapters, and friendlier packaging. The scheduler and the HTTP
surface are meant to stay portable (x86_64 and ARM; CUDA / Vulkan / Metal via
the engine, not the core).

## Run

Requires Rust 1.80+. Fake engines need no GPU:

```bash
mkdir -p ~/.config/silkai
cp examples/config.toml ~/.config/silkai/config.toml
cargo run -p silkai
curl -s http://127.0.0.1:8080/health
```

Config is `~/.config/silkai/config.toml`, or `SILKAI_CONFIG`.

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"soap","messages":[{"role":"user","content":"hello"}]}'
```

Streaming: `"stream": true` (SSE). Also `GET /v1/models` and `GET /v1/status`.

See `examples/config.toml` for VRAM, `priority` (`live` | `normal` | `background`),
`exclusive`, `slots`, and `keep_warm`.

Build llama.cpp support with `cargo run -p silkai --features llama` and set
`engine = "llama.cpp"` plus a GGUF `path` on that model.

## Development

```bash
cargo test
cargo test --features llama
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
