# DiskANN Repository - Agent Onboarding Guide

**Last Updated**: 2026-10-29 (based on v0.61.0)

---

## Table of Contents

1. [Repository Overview](#repository-overview)
2. [Repository Structure](#repository-structure)
3. [Dependencies](#dependencies)
4. [Testing](#testing)
5. [Code Quality & Linting](#code-quality--linting)
6. [Pre-commit Checklist](#pre-commit-checklist)

---

## Repository Overview

**DiskANN** is a Rust implementation of scalable approximate nearest neighbor (ANN) search algorithms. The project is a rewrite from C++ to Rust.

- **Language**: Rust; edition in [`Cargo.toml`](Cargo.toml) (`workspace.package.edition`), toolchain version in [`rust-toolchain.toml`](rust-toolchain.toml)
- **License**: MIT (see [`LICENSE.txt`](LICENSE.txt))
- **Version**: See [`Cargo.toml`](Cargo.toml)
- **Architecture**: Cargo workspace; members are defined in [`Cargo.toml`](Cargo.toml)
- **Legacy Code**: Older C++ code is on the `cpp_main` branch (not maintained)

### Key Resources
- **Contributing**: See [`CONTRIBUTING.md`](CONTRIBUTING.md) (requires CLA)
- **Code of Conduct**: See [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md)

---

## Repository Structure

The repository uses a Cargo workspace with crates organized into functional tiers. See [`Cargo.toml`](Cargo.toml) for:
- Workspace members and their dependencies
- Shared dependency versions
- Build profiles (release, test)
- Workspace-level lints

### Crate Organization

**Tier 1: Foundation**
- `diskann-wide/` - Low-level SIMD, bit manipulation, type width abstractions
- `diskann-vector/` - Vector primitives and operations

**Tier 2: Core Libraries**
- `diskann-linalg/` - Linear algebra operations
- `diskann-utils/` - Shared utilities (Reborrow, Matrix traits)
- `diskann-quantization/` - Vector quantization (PQ, SQ)

**Tier 3: Algorithm & Storage**
- `diskann/` - Core ANN graph algorithm and in-memory indexing (CENTRAL crate)
- `diskann-providers/` - Storage abstraction layer
- `diskann-disk/` - Disk-based indexing with io_uring support
- `diskann-label-filter/` - Inverted index for filtered search
- `diskann-garnet/` - Garnet vector-set integration and FFI bindings
- `diskann-inmem/` - In-memory index provider
- `diskann-bftree/` - BfTree-backed storage provider for async indexes

**Tier 4: Infrastructure & Tools**
- `diskann-benchmark-runner/` - Test runner infrastructure
- `diskann-benchmark-core/` - Benchmark framework
- `diskann-benchmark-simd/` - SIMD-specific benchmarks
- `diskann-benchmark/` - Benchmark definitions and runners
- `diskann-record/` - Versioned persistence with manifests and binary artifacts
- `diskann-tools/` - CLI utilities (autotuner, etc.)

---

## Dependencies

- Tier 1 and Tier 2 crates may be added as dependencies of any internal crate
- `diskann` may be added as a dependency of any equal or higher tier internal crate except those below
- Do not add Tier 3 crates as dependencies of these Tier 4 crates:
  - `diskann-benchmark-runner`
  - `diskann-benchmark-core` (`diskann` is allowed)
  - `diskann-benchmark-simd`

---

## Testing

### Test Execution
- Bare `cargo test` covers only `workspace.default-members` in [`Cargo.toml`](Cargo.toml), not the entire workspace.
- Choose the package and feature scope affected by the change.
- For workspace-wide validation, follow the `test-workspace` and `test-workspace-features` jobs in [`.github/workflows/ci.yml`](.github/workflows/ci.yml), including their separate doc-test steps.
- CI uses `cargo-nextest`; [`.cargo/nextest.toml`](.cargo/nextest.toml) defines timeouts, retries, and other runner settings.

### Test Baseline Caching System

The `diskann` crate uses a baseline caching system for regression detection. See [`diskann/README.md`](diskann/README.md) for baseline usage and regeneration workflow.

### AVX-512, Aarch64, and multi-platform

When touching architecture-specific intrinsics, run cross-platform validation per `diskann-wide/README.md`:

- Testing AVX-512 code on non-AVX-512 capable x86-64 machines.
- Testing Aarch64 code on x86-64 machines.
- Testing code compiled for and running on the `x86-64` CPU (no AVX/AVX2) does not execute unsupported instructions.

---

## Code Quality & Linting

### License Headers

Add the [license header](.github/LICENSE_HEADER.txt) when creating Rust source files.

### Error Handling

There are three regimes of error handling and the strategy to use depends on the regime.

#### Low-Level Libraries

- Use focused, non-allocating error types. `diskann::ANNError` is not suitable at this layer.
- Use `thiserror` for boilerplate and `std::error::Error::source` for error chaining.

#### DiskANN Algorithms

- Use [`diskann::ANNError`](diskann/src/error/ann_error.rs) for unrecoverable errors and its context machinery when propagating them.
- When converting to `ANNError`, use `#[track_caller]` for better source reporting.

- For traits whose associated errors may be non-critical, consider [`ToRanked`](diskann/src/error/ranked.rs) instead of `Into<ANNError>`. This preserves the distinction between transient and hard failures.

#### High-Level Tooling

`anyhow::Error` is appropriate at this layer.

#### Do Not

Do not use a single crate-level error enum. Problems:

- Provides no documentation on how an individual function could fail
- Encourages **worse** error messages than bespoke types
- Generates large structs that blow up the stack
- Branch-heavy `Drop` implementations which bloat code

### Code Coverage

Code coverage of changes is required for PRs. See [`.codecov.yml`](.codecov.yml) for coverage policy and thresholds.

### CI workflow

CI workflow is defined in [`.github/workflows/ci.yml`](.github/workflows/ci.yml). Key jobs include:
- Format and Clippy checks
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

### Documentation

Less is more. Documentation goes out of date; rustdoc-generated output doesn't. Don't duplicate what rustdoc already provides for free (type listings, signatures, re-exports, intra-doc links).

**DO**:
- Document non-obvious behavior, errors, safety, and design intent
- Use `# Errors`, `# Safety`, `# Panics`, `# Example` sections
- Keep documentation for internal `pub(crate)` / `pub(super)` items concise and additive; critical items should still receive full documentation

**DON'T**:
- Maintain explicit lists of types/functions in module-level docs
- Restate what the signature already shows (e.g. "owns an `Arc<T>`" on a struct whose only field is `Arc<T>`)

---

## Pre-commit Checklist

Before committing changes, run:

```bash
# Format all code
cargo fmt --all

# Run Clippy with warnings as errors
cargo clippy --workspace --all-targets -- -D warnings
```

Repository-specific rules for the Clippy check:
- [`clippy.toml`](clippy.toml): disallowed methods and their alternatives, including explicit Rayon thread-pool use and repository random helpers.
- [`Cargo.toml`](Cargo.toml): workspace lints, including unsafe-block documentation.

---

*This guide should be updated when major changes occur to the repository structure or development workflows.*
