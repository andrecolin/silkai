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
mkdir -p ~/.config/silkai
cp examples/config.toml ~/.config/silkai/config.toml
cargo run -p silkai
curl -s http://127.0.0.1:8080/health
```

Config defaults to `~/.config/silkai/config.toml`. Override with `SILKAI_CONFIG`.

Chat — requests send `"model"` only; policy lives in config:

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"whisper","messages":[{"role":"user","content":"hello"}]}'
```

Streaming uses SSE (`"stream": true`). Queued jobs keep the stream alive with
comments until they admit.

Also: `GET /v1/models`, `GET /v1/status`.

## Config

See `examples/config.toml`. Priority, exclusive, slots, and VRAM are **only**
in this file — requests just send `"model"`.

## Development

```bash
cargo test
cargo test --features llama
```

Scheduler tests do not need a GPU. `cargo test --features llama` builds the llama.cpp adapter.
