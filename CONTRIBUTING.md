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

## The status page

`crates/silkai-server/ui/index.html` is one file of plain HTML, CSS, and
JavaScript, embedded with `include_str!`. That is deliberate: no framework,
no bundler, no external assets, so `cargo install` is the whole build and the
page works on an offline server. Improve it in place; PRs that add a
JavaScript toolchain will be asked to build on `/v1/status` and
`/v1/events` in a separate repo instead.

## Crate boundaries

`silkai-sched` is a pure scheduler: numeric GB, priority, exclusive, slots.
Do not put CUDA types (or any other GPU SDK types) in `silkai-sched`. Model
execution lives in `silkai-adapters`.

## Releasing to crates.io

The four crates publish as one set, in dependency order: `silkai-sched`,
`silkai-adapters`, `silkai-server`, `silkai`. Cargo 1.90+ does the ordering:

```bash
# bump [workspace.package].version, commit, tag vX.Y.Z, then:
cargo publish --workspace --dry-run    # packages and verifies all four locally
cargo publish --workspace
```

If you dry-run, change code, and dry-run again at the same version, the
verify step can compile the *old* copy of a sibling crate: cargo treats
registry sources as immutable, keeps the copy it extracted the first time
under `~/.cargo/registry/src/*/silkai-*`, and reuses the earlier build.
Clear both, in this order, then dry-run again:

```bash
rm -rf ~/.cargo/registry/src/*/silkai-* target/package
cargo clean
cargo publish --workspace --dry-run
```

Every crate inherits its metadata from `[workspace.package]`, and the
inter-crate dependencies are declared once under `[workspace.dependencies]`
with both `path` and `version`, so a single version bump moves everything.
Test fixtures (`silkai_sched::clinic`, `silkai_server::app::test_app*`) sit
behind the `test-util` feature and are not part of the published API.

## Security

Do not open a public issue for a vulnerability. See [SECURITY.md](SECURITY.md).
