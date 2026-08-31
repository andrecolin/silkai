# SilkAI

You have **one GPU** and **several local models**. They do not all fit at once.
Each app that loads a model tends to keep the memory even when it is idle, so
the next model never starts — or you sit through a long reload from disk.

SilkAI is a small daemon that owns the GPU for those models. If two fit, they
run together. If they do not, it parks the idle one in ordinary RAM and brings
it back in a couple of seconds. Apps use a normal OpenAI-style HTTP API. Rules
(who can share, who must run alone, who is live) live in a config file, not in
each request.

[MIT](LICENSE) · [Ko-fi](https://ko-fi.com/andrecolin)

## A simple workflow

Think of a meeting, not a hospital.

1. **While you talk**, a speech-to-text model stays on the GPU so the transcript
   keeps up. Two people on the same machine can share that one copy.
2. **When you stop**, SilkAI moves speech-to-text into RAM (still ready) and
   gives the **whole GPU** to a large writing model. That model turns the
   transcript into a summary or an email.
3. You can fire **two summaries at once**. They wait in line and reuse the
   writer that is already loaded — it does not load a second copy.
4. A **small** model that tags or searches files only runs when there is spare
   room. It never kicks you off in the middle of talking.

Same idea if you code: a small autocomplete model while you type, a large
“think hard” model when you ask a big question, embeddings for search in the
gaps. One card, a queue, a warm copy in RAM.

On a desktop with a 32 GB GPU and 128 GB of RAM, all three models can stay in
RAM all day. The GPU only holds whoever is working right now. A little VRAM is
left for the display so the machine still feels normal.

SilkAI is not a new model runner and not a cloud cluster. llama.cpp (and later
Whisper) still generate tokens. We only decide **who is on the GPU**, **who
waits**, and **who stays warm in RAM**.

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
`--features llama`.

Still to come: live speech-to-text over WebSocket (an open connection keeps
that model high-priority), more adapters, and easier install. The scheduler and
HTTP API are meant to stay portable (x86_64 and ARM; CUDA / Vulkan / Metal via
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

The example file names the three roles above `transcribe`, `write`, and `index`:

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"write","messages":[{"role":"user","content":"Summarize this meeting."}]}'
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
