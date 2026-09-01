# Contributing

PRs are welcome. There is no CLA. Contributions are MIT, same as the rest of
the repo (see `LICENSE`).

## Tests

Run `cargo test` before opening a pull request. Scheduler tests do not need a
GPU. llama.cpp GPU backends are Cargo features (`cuda`, `vulkan`, `metal`)
on the `silkai` crate; they forward into `llama-cpp-2`. Do not enable more
than one GPU backend at once, and do not use `--all-features`.

## Crate boundaries

`silkai-sched` is a pure scheduler: numeric GB, priority, exclusive, slots.
Do not put CUDA types (or any other GPU SDK types) in `silkai-sched`. Model
execution lives in `silkai-adapters`.
