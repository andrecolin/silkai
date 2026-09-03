# Contributing

PRs are welcome. There is no CLA. Contributions are MIT, same as the rest of
the repo (see `LICENSE`). By taking part you agree to the
[Code of Conduct](CODE_OF_CONDUCT.md).

## How to send a change

`main` is protected: nobody pushes to it directly, and every change lands
through a pull request with green CI.

1. Fork the repo and branch off `main`.
2. Make the change. Keep the commit history readable — small commits with a
   subject line that says what changed, not what file you touched.
3. Run the checks below.
4. Open a pull request. The maintainer reviews and merges.

For anything large, open an issue first so we can agree on the shape before you
write it. SilkAI decides who is on the GPU, who waits, and who stays warm in
RAM — it is not a model runner and not a cluster manager. Changes that push
past that line are likely to be declined however good the code is.

## Checks

CI runs these on Linux and macOS. Run them locally first:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets
cargo test
```

Scheduler tests do not need a GPU. llama.cpp GPU backends are Cargo features
(`cuda`, `vulkan`, `metal`) on the `silkai` crate; they forward into
`llama-cpp-2`. Do not enable more than one GPU backend at once, and do not use
`--all-features`. CI does not build them — they need the matching SDK.

The GPU integration test is opt-in and needs real weights:

```bash
SILKAI_ITEST=1 SILKAI_GGUF_A=/path/tiny.gguf SILKAI_GGUF_B=/path/small.gguf \
  cargo test -p silkai-server --features llama --test itest_llama -- --nocapture
```

## Crate boundaries

`silkai-sched` is a pure scheduler: numeric GB, priority, exclusive, slots.
Do not put CUDA types (or any other GPU SDK types) in `silkai-sched`. Model
execution lives in `silkai-adapters`.

## Security

Do not open a public issue for a vulnerability. See [SECURITY.md](SECURITY.md).
