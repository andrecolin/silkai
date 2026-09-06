# Changelog

All notable changes to SilkAI. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[SemVer](https://semver.org/).

## [Unreleased]

### Fixed
- The systemd unit names its own `PATH`, with the install prefix first. It
  relied on whatever the user manager had inherited: a desktop login passes
  its session `PATH` down, so `~/.local/bin` — where `scripts/install.sh` puts
  `silkai`, and the natural place for a llama.cpp build — was on it. A manager
  started by lingering at boot gets `/etc/environment`, where that directory is
  normally absent, and a `process` engine's `cmd = ["llama-server", ...]` then
  does not resolve. Same behaviour either way now.
- `finish_reason` is the engine's own, and `usage` is reported. Every reply
  said `"stop"` no matter how it ended, and carried no counts at all: a reply
  cut off at `max_tokens` was indistinguishable from a complete one, and a
  client could not tell what a request cost. llama-server reported `"length"`
  on a truncated reply and SilkAI still said `"stop"`. Engines now send an
  end alongside their tokens — parsed from the OpenAI stream for `process`
  and `vllm` (which now ask for it with `stream_options`), from `done_reason`
  and the `prompt_eval_count` / `eval_count` pair for Ollama, and counted
  directly by the in-process llama.cpp engine. A job preempted mid-stream and
  resumed keeps its finish reason but reports no usage, since the engine
  counted only the run that finished.
- A `content` list now reaches the engine as a list. SilkAI kept the `text`
  parts and dropped everything else, so an image sent to a vision model was
  discarded on the way through and the model answered from the text alone —
  a confident wrong answer rather than an error. Asked the colour of a solid
  blue PNG, llama-server answers "Blue" directly and answered "Grey" through
  SilkAI. `ChatMessage::content` is now a `Content` enum, `Text` or `Parts`,
  serialized back exactly as it arrived. Engines whose wire format has no
  place for parts — the in-process llama.cpp engine, and Ollama, whose
  `/api/chat` carries images in a separate `images` field — project a list to
  its joined text, which is what every engine received before.

## [0.5.0] - 2026-09-03

First crates.io release: `silkai`, `silkai-server`, `silkai-adapters`,
`silkai-sched`.

### Added
- A session stays open on a WebSocket ping or `{"type":"ping"}`. Before this,
  every frame that was not a prompt or a stop closed the session, so a client
  sending standard keepalives (Python's `websockets` pings every 20 s by
  default) lost its pin, and a session that only held a model resident could
  not survive its own idle timeout at all.
- `engine = "process"` starts and stops any OpenAI-speaking child
  (llama-server, vLLM, and the like) and waits on `GET /health`. A complete
  three-model llama-server setup lives in `examples/llama-server.toml`.
- The whole `messages` list reaches the engine, with `max_tokens` and
  `temperature`; the in-process llama.cpp engine renders it through the
  GGUF's own chat template.
- `ctx_size` per model for the in-process engine (default 4096). A prompt
  that does not fit is refused with the reason instead of answered with
  nothing.
- Truthful status at `GET /v1/status`: a `loading` state, measured VRAM beside
  the configured budget (from nvidia-smi), open sessions, and RAM counted only
  for engines that hold a copy.
- `GET /v1/events`: the last 500 scheduler events replayed, then live over
  Server-Sent Events.
- `GET /metrics` in Prometheus text format.
- An optional embedded status page at `/ui` (`[ui] enabled = true`) and an
  optional bearer token for `/ui`, `/metrics`, and `/admin/*`.
- A live request that arrives during a long load abandons that load and
  re-queues the waiting job at the front.
- Chat responses carry `id`, `object`, `created`, `model`, and
  `finish_reason`; streams open with a role chunk and close with a stop
  chunk before `[DONE]`. Error responses carry the reason in the body.
- Clean shutdown on SIGTERM and Ctrl-C, taking process-engine children with
  the daemon.

### Fixed
- The process engine could not start llama-server (it waited on vLLM's
  `/wake_up`).
- Killing the daemon left process-engine children holding VRAM.
- The in-process llama.cpp engine returned an empty answer for prompts over
  256 tokens.

[0.5.0]: https://github.com/andrecolin/silkai/releases/tag/v0.5.0
