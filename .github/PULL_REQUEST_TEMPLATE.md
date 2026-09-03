## What this changes

<!-- One or two sentences. What behaviour is different after this merges? -->

## Why

<!-- The problem. Link an issue if there is one. -->

## Checklist

- [ ] `cargo test` passes
- [ ] `cargo fmt --all --check` is clean
- [ ] `cargo clippy --workspace --all-targets` is clean
- [ ] No GPU SDK types (CUDA or otherwise) added to `silkai-sched`
- [ ] Did not enable more than one GPU backend at once (no `--all-features`)
