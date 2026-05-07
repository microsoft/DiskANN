# DiskANN Repository - Agent Onboarding Guide

**Last Updated**: 2026-05-04 (based on v0.50.1, Rust 1.92)


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
- `diskann-quantization/` - Vector quantization 

**Tier 3: Algorithm & Proivders**
- `diskann/` - Core indexing logic
- `diskann-providers/` - Hodge-podge of stuff, will be dismantled
- `diskann-disk/` - Disk-based provider for the index with io_uring support
- `diskann-label-filter/` - Inverted index for filtered search
- `diskann-garnet/` - Garnet (Redis-compatible) Provider and FFI endpoints for vector sets

**Tier 4: Infrastructure & Tools**
- `diskann-benchmark-runner/` - Test runner infrastructure
- `diskann-benchmark-core/` - Benchmark framework
- `diskann-benchmark-simd/` - SIMD-specific benchmarks
- `diskann-benchmark/` - Benchmark definitions and runners
- `diskann-tools/` - CLI utilities (autotuner, etc.)
- `vectorset/` - Garnet client for benchmarking vector set workloads

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

## Boundaries

### 🚫 Never

- Modify files in `diskann/tests/generated/` by hand — these are auto-generated baselines. Regenerate with `DISKANN_TEST=overwrite`.
- Modify `rust-toolchain.toml`, `.github/workflows/`, or `.codecov.yml` without explicit approval.
- Use the global Rayon thread pool — use `RayonThreadPool`/`RayonThreadPoolRef` (enforced by `clippy.toml` disallowed methods).
- Use `rand::thread_rng` — use the project's `random.rs` utilities instead (enforced by `clippy.toml`).
- Use `vfs::PhysicalFS::new` or `VirtualStorageProvider::new_physical()` in tests — use `VirtualStorageProvider::new_overlay()`.
- Remove or weaken existing tests without a strong, documented reason.
- Commit secrets, credentials, or API keys.

### ⚠️ Ask First

- Adding new workspace dependencies to `Cargo.toml` — justify the addition.
- Changing public API signatures in any `diskann-*` crate — requires SemVer analysis (may need a major version bump).
- Modifying tier dependency rules (e.g., adding a Tier 3 dependency to a Tier 4 benchmark crate).
- Changing `clippy.toml` or `rustfmt.toml` lint/formatting configuration.

### ✅ Always

- Include the MIT license header in every new source file.
- Run `cargo fmt --all` and `cargo clippy --workspace --all-targets -- -Dwarnings` before committing.
- Update doc comments when changing function signatures or removing parameters.
- Add a `// SAFETY:` comment above every `unsafe` block with specific, verifiable preconditions.

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

DiskANN uses a baseline caching system for regression detection. Test results are serialized as JSON into `diskann/tests/generated/` and compared against on subsequent runs. Any difference is flagged as a test failure.

- To regenerate baselines: run tests with `DISKANN_TEST=overwrite`
- Before checking in: delete `diskann/tests/generated/` first, then regenerate to prune unused baselines
- Regenerated JSON files should be inspected via `git diff` during review

The APIs are **`pub(crate)`** (internal to the `diskann` crate only):
- [`diskann/src/test/cache.rs`](diskann/src/test/cache.rs) — `get_or_save_test_results`, `TestRoot`, `TestPath`
- [`diskann/src/test/cmp.rs`](diskann/src/test/cmp.rs) — `VerboseEq` trait, `verbose_eq!` macro, `assert_eq_verbose!`

See [`diskann/README.md`](diskann/README.md) for additional details.

### AVX-512, Aarch64, and multi-platform

When touching architecture-specific intrinsics, run cross-platform validation per `diskann-wide/README.md`:

- Testing AVX-512 code on non-AVX-512 capable x86-64 machines.
- Testing Aarch64 code on x86-64 machines.
- Testing code compiled for and running on the `x86-64` CPU (no AVX/AVX2) does not execute unsupported instructions.

---

## Code Quality & Linting

### Error Handling

There are three regimes of error handling and the strategy to use depends on the regime.

#### Low-Level

Low-level crates should use bespoke, precise, non-allocating error types. Use `thiserror` for boilerplate. Chain with `std::error::Error::source`.

`diskann::ANNError` is not a suitable low-level error type.

```rust
// ✅ Good — bespoke error type with thiserror, uses #[from] for source chaining
#[derive(Debug, thiserror::Error)]
pub enum SgemmError {
    #[error("dimension overflow: {expected_rows}×{expected_cols} exceeds usize")]
    DimensionOverflow { expected_rows: usize, expected_cols: usize },

    #[error(transparent)]
    Allocation(#[from] AllocatorError),
}

// ❌ Bad — single crate-level enum, formats inner error in display string
#[derive(Debug, thiserror::Error)]
pub enum MyLibError {
    #[error("sgemm failed: {0}")]  // Don't format the inner error here
    Sgemm(#[source] SgemmError),
    #[error("io failed: {0}")]
    Io(#[from] std::io::Error),
    // ... 20 more variants — too broad
}
```

#### Mid-Level (diskann algorithms)

Use `diskann::ANNError` and its context machinery. This type:

- Is 16 bytes with niche optimization for `Option<ANNError>`, allowing return in registers.
- Records stack trace of its first construction under `RUST_BACKTRACE=1`.
- Precisely records file and line of creation via `#[track_caller]`.
- Has context layering machinery to add additional information as an error is passed up the stack.
- Is backed internally by `anyhow::Error`.

When converting to `ANNError`, use `#[track_caller]` for better source reporting.
Prefer `ANNError::new(ANNErrorKind::…, e)` over the old `log_*`-style constructors, which force eager string formatting and double-log errors.

```rust
// ✅ Good — deferred formatting, precise kind, track_caller
#[track_caller]
fn process_vectors(data: &[f32]) -> Result<(), ANNError> {
    validate(data).map_err(|e| ANNError::new(ANNErrorKind::IndexError, e))?;
    Ok(())
}

// ❌ Bad — eager string formatting, double-logs on creation
fn process_vectors(data: &[f32]) -> Result<(), ANNError> {
    validate(data).map_err(|e| ANNError::log_index_error(format!("validation failed: {e}")))?;
    Ok(())
}
```

Traits with associated error types should consider constraining with `diskann::error::ToRanked` instead of `Into<ANNError>` if non-critical errors should be supported.
`ANNError` is the mid-level propagated error type; use `ToRanked` and `RankedError` to distinguish transient/recoverable failures from fatal ones.

#### High Level (tooling)

At this level `anyhow::Error` is appropriate for binaries and CLI entry points. Note that some tooling helpers still use `ANNError` for compatibility.

#### Do Not

Do not use a single crate-level error enum. Problems:

- Provides no documentation on how an individual function could fail
- Encourages **worse** error messages than bespoke types
- Generates large structs that blow up the stack
- Branch-heavy `Drop` implementations which bloat code

### Formatting

See [`rustfmt.toml`](rustfmt.toml) for formatting configuration. Commands are in [Quick Reference](#quick-reference).

### Clippy (Linting)

```bash
# Check with no default features (for specific crates)
cargo clippy -p diskann --no-default-features
```

See [`clippy.toml`](clippy.toml) for linting rules, including:
- Disallowed methods (rayon global thread pool, rand::thread_rng, etc.)

The workspace-level lint in [`Cargo.toml`](Cargo.toml) enforces documentation for unsafe blocks:
- `undocumented_unsafe_blocks = "warn"`

```rust
// ✅ Good — specific, verifiable precondition
// SAFETY: `i + width <= len` ensures this read is in-bounds.
let val = unsafe { ptr.add(i).read() };

// ✅ Good — FFI with listed preconditions
// SAFETY: `a` and `b` are non-null, properly aligned, and do not alias.
// `m`, `n`, `k` match the actual matrix dimensions.
unsafe { ffi::sgemm(m, n, k, a.as_ptr(), b.as_ptr(), c.as_mut_ptr()) };

// ❌ Bad — vague, unverifiable
// SAFETY: this is safe
let val = unsafe { ptr.add(i).read() };
```

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

Before committing changes, always run the format and clippy commands from [Quick Reference](#quick-reference).

### Points to Add
how to create a new runtime
how to write error handling
how to configure providers
how to write a new benchmark

---

**End of Agent Onboarding Guide**

*This guide should be updated when major changes occur to the repository structure or development workflows.*
