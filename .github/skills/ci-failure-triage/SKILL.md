---
name: ci-failure-triage
description: 'Diagnose CI failures. Use when "CI failed", "build broken", "test failure in CI", or "triage CI".'
---

# CI Failure Triage

Systematic approach to diagnosing failures in the DiskANN CI pipeline.

## When to Use

- A CI workflow run has failed
- Tests pass locally but fail in CI
- Platform-specific failures (Linux vs Windows)

## Process

### Step 1: Identify Failure Type

Check which CI job failed:
- `format` — run `cargo fmt --all --check`
- `clippy` / `clippy-features` — run `cargo clippy --workspace --all-targets --config 'build.rustflags=["-Dwarnings"]'`
- `test` — run `cargo test`
- `sde-*` — architecture-specific issues (AVX-512, baseline x86-64)

### Step 2: Reproduce Locally

<!-- TODO: Add SDE setup instructions for architecture emulation -->
- For format/clippy/test: run `./scripts/verify`
- For feature-gated failures: check `DISKANN_FEATURES` env var in ci.yml
- For SDE failures: see diskann-wide/README.md for cross-platform validation

### Step 3: Fix and Verify

- Apply fix, then run `./scripts/verify`
- If SIMD-related, validate per diskann-wide/README.md

## Constraints

- Do not disable or weaken CI checks to fix failures
- Do not skip platform-specific test runs
