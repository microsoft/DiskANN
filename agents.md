# DiskANN Repository - Agent Onboarding Guide

**Last Updated**: 2026-02-11 (based on v0.45.0, Rust 1.92)

This guide helps coding agents understand how to work efficiently with the DiskANN repository.

---

## Table of Contents

1. [Repository Overview](#repository-overview)
2. [Repository Structure](#repository-structure)
3. [Testing](#testing)
4. [Code Quality & Linting](#code-quality--linting)

---

## Repository Overview

**DiskANN** is a Rust implementation of scalable approximate nearest neighbor (ANN) search algorithms. The project is a rewrite from C++ to Rust.

- **Language**: Rust (Edition 2021), toolchain version in [`rust-toolchain.toml`](rust-toolchain.toml)
- **License**: MIT (see [`LICENSE.txt`](LICENSE.txt))
- **Version**: See [`Cargo.toml`](Cargo.toml)
- **Architecture**: Cargo workspace with 15+ crates
- **Legacy Code**: Older C++ code is on the `cpp_main` branch (not maintained)

### Key Resources
- **Contributing**: See [`CONTRIBUTING.md`](CONTRIBUTING.md) (requires CLA)
- **Code of Conduct**: See [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md)

---

## Repository Structure

The repository uses a Cargo workspace with crates organized into functional tiers. See [`Cargo.toml`](Cargo.toml) for:
- Workspace members and their dependencies
- Shared dependency versions
- Build profiles (release, ci)
- Workspace-level lints

### Crate Organization

**Tier 1: Foundation**
- `diskann-wide/` - Low-level SIMD, bit manipulation, type width abstractions
- `diskann-vector/` - Vector primitives and operations
- `diskann-platform/` - Platform-specific utilities

**Tier 2: Core Libraries**
- `diskann-linalg/` - Linear algebra operations
- `diskann-utils/` - Shared utilities (Reborrow, MatrixView traits)
- `diskann-quantization/` - Vector quantization (PQ, SQ)

**Tier 3: Algorithm & Storage**
- `diskann/` - Core ANN graph algorithm and in-memory indexing (CENTRAL crate)
- `diskann-providers/` - Storage abstraction layer
- `diskann-disk/` - Disk-based indexing with io_uring support
- `diskann-label-filter/` - Inverted index for filtered search

**Tier 4: Infrastructure & Tools**
- `diskann-benchmark-runner/` - Test runner infrastructure
- `diskann-benchmark-core/` - Benchmark framework
- `diskann-benchmark-simd/` - SIMD-specific benchmarks
- `diskann-benchmark/` - Benchmark definitions and runners
- `diskann-tools/` - CLI utilities (autotuner, etc.)

---

## Dependencies

### Internal

- Tier 1 and Tier 2 crates may be added as dependencies of any internal crate
- `diskann` may be added as a dependency of any equal or higher tier internal crate except those below
- Do not add Tier 3 crates as dependencies of these Tier 4 crates:
  - `diskann-benchmark-runner`
  - `diskann-benchmark-core` (`diskann` is allowed)
  - `diskann-benchmark-simd`

---

## Testing

### Test Execution

```bash
# Run all tests
cargo test

# Run tests for specific crate
cargo test -p diskann

# Run specific test
cargo test -p diskann -- --exact test_name

# Run with CI profile (faster)
cargo test --profile ci

# Run doc tests
cargo test --doc
```

**Note**: CI uses `cargo-nextest` for running tests. See [`.cargo/nextest.toml`](.cargo/nextest.toml) for test configuration (timeouts, retries, etc.).

### Test Baseline Caching System

DiskANN uses a baseline caching system for regression detection. See [`diskann/README.md`](diskann/README.md) for a high-level overview of how the baseline system works. For implementation and API details, refer directly to:
- [`diskann/src/test/cache.rs`](diskann/src/test/cache.rs) — core baseline caching APIs
- [`diskann/src/test/cmp.rs`](diskann/src/test/cmp.rs) — `VerboseEq` and related helpers for better test error messages

### AVX-512, Aarch64, and multi-platform

When touching architecture-specific intrinsics, run cross-platform validation per `diskann-wide/README.md`:

* Testing AVX-512 code on non-AVX-512 capable x86-64 machines.
* Testing Aarch64 code on x86-64 machines.
* Testing code compiled for and running on the `x86-64` CPU (no AVX2) does not execute unsupported instructions.

---

## Code Quality & Linting

### Error Handling

There are three regimes of error handling and the strategy to use depends on the regime.

#### Low-Level

Low-level crates should use bespoke, precise, non-allocating error types. Use `thiserror` for boilerplate. Chain with `std::error::Error::source`.

`diskann::ANNError` is not a suitable low-level error type.

#### Mid-Level (diskann algorithms)

Use `diskann::ANNError` and its context machinery. This type:

* Has a small size and `Drop` implementation, so is efficient in function ABIs.
* Records stack trace of its first creation under `RUST_BACKTRACE=1`.
* Precisely records line numbers of creation.
* Has a context layering machinery to add additional information as an error is passed up the stack.

When converting to `ANNError`, use `#[track_caller]` for better source reporting.

Traits with associated error types should consider constraining with `diskann::error::ToRanked` instead of `Into<ANNError>` if non-critical errors should be supported.
`diskann::ANNError` should be used only for unrecoverable errors.

#### High Level (tooling)

At this level `anyhow::Error` is an appropriate type to use.

#### Do Not

Do not use a single crate-level error enum. Problems:

* Provides no documentation on how an individual function could fail
* Encourages **worse** error messages than bespoke types
* Generate large structs that blow up the stack
* Have branch-heavy `Drop` implementations which bloat code

### Formatting

**Note**: `rustfmt` is not installed by default. Run `rustup component add rustfmt` if needed.

```bash
# Check formatting (matches CI)
cargo fmt --all --check

# Apply formatting to all crates
cargo fmt --all
```

See [`rustfmt.toml`](rustfmt.toml) for formatting configuration.

### Clippy (Linting)

**Note**: `clippy` is not installed by default. Run `rustup component add clippy` if needed.

```bash
# Basic clippy check
cargo clippy --workspace --all-targets

# CI-style check (warnings as errors)
cargo clippy --workspace --all-targets --config 'build.rustflags=["-Dwarnings"]'

# Check with no default features (for specific crates)
cargo clippy -p diskann --no-default-features
```

See [`clippy.toml`](clippy.toml) for linting rules, including:
- Disallowed methods (rayon global thread pool, rand::thread_rng, etc.)
- Required documentation for unsafe blocks

### Code Coverage

Code coverage of changes is required for PRs. See [`.codecov.yml`](.codecov.yml) for coverage policy and thresholds.

### CI Pipeline

CI workflow is defined in [`.github/workflows/ci.yml`](.github/workflows/ci.yml). Key jobs include:
- Format and clippy checks
- Tests on multiple platforms (Linux, Windows)
- Code coverage
- Architecture compatibility (SDE)

### Test Patterns

**DO**:
- Look for existing setup/execution infrastructure
- Factor out common patterns

**DON'T**:
- Add tests for derived traits (Clone, Debug, PartialEq)
- Add tests for enums unless they have explicit functionality

---

## Pre-commit Checklist

Before committing changes, always run:

```bash
# Format all code
cargo fmt --all

# Run clippy with warnings as errors
cargo clippy --workspace --all-targets -- -D warnings
```

---

**End of Agent Onboarding Guide**

*This guide should be updated when major changes occur to the repository structure or development workflows.*
