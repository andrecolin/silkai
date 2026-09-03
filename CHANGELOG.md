# Changelog

All notable changes to SilkAI. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[SemVer](https://semver.org/).

## [0.5.0] - 2026-09-03

First crates.io release: `silkai`, `silkai-server`, `silkai-adapters`,
`silkai-sched`.

### Added
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
