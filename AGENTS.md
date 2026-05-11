# DiskANN Repository - Agent Onboarding Guide

**Last Updated**: 2026-05-04 (based on v0.50.1, Rust 1.92)


## Crate Organization

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

### Internal Dependencies


- Tier 1 and Tier 2 crates may be added as dependencies of any internal crate
- `diskann` may be added as a dependency of any equal or higher tier internal crate except those below
- Do not add Tier 3 crates as dependencies of these Tier 4 crates:
  - `diskann-benchmark-runner`
  - `diskann-benchmark-core` (`diskann` is allowed)
  - `diskann-benchmark-simd`

---

## Boundaries

### Never

- Modify files in `diskann/tests/generated/` by hand — these are auto-generated baselines. Regenerate with `DISKANN_TEST=overwrite`.
- Modify `rust-toolchain.toml`, `.github/workflows/`, or `.codecov.yml` without explicit approval.
- Use the global Rayon thread pool — use `RayonThreadPool`/`RayonThreadPoolRef` (enforced by `clippy.toml` disallowed methods).
- Use `rand::thread_rng` — use the project's `random.rs` utilities instead (enforced by `clippy.toml`).
- Use `vfs::PhysicalFS::new` or `VirtualStorageProvider::new_physical()` in tests — use `VirtualStorageProvider::new_overlay()`.
- Remove or weaken existing tests without a strong, documented reason.
- Commit secrets, credentials, or API keys.

### Ask First

- Adding new workspace dependencies to `Cargo.toml` — justify the addition.
- Changing public API signatures in any `diskann-*` crate — requires SemVer analysis (may need a major version bump).
- Modifying tier dependency rules (e.g., adding a Tier 3 dependency to a Tier 4 benchmark crate).
- Changing `clippy.toml` or `rustfmt.toml` lint/formatting configuration.

### Always

- Include the MIT license header in every new source file.
- Run `cargo fmt --all` and `cargo clippy --workspace --all-targets -- -Dwarnings` before committing.
- Update doc comments when changing function signatures or removing parameters.
- Add a `// SAFETY:` comment above every `unsafe` block with specific, verifiable preconditions.

---




## Error Handling

There are three regimes of error handling and the strategy to use depends on the regime.

### Low-Level

Low-level crates should use bespoke, precise, non-allocating error types. 
Use `thiserror` for boilerplate. Chain with `std::error::Error::source`.
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

### Mid-Level (diskann algorithms)

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

### High Level (tooling)

At this level `anyhow::Error` is appropriate for binaries and CLI entry points.
 Note that some tooling helpers still use `ANNError` for compatibility.

### Do Not

Do not use a single crate-level error enum. Problems:

- Provides no documentation on how an individual function could fail
- Encourages **worse** error messages than bespoke types
- Generates large structs that blow up the stack
- Branch-heavy `Drop` implementations which bloat code


## Document unsafe usage

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


## CI Pipeline

CI workflow is defined in [`.github/workflows/ci.yml`](.github/workflows/ci.yml). Key jobs include:
- Format and clippy checks
- Tests on multiple platforms (Linux, Windows)
- [Code coverage](.codecov.yml)
- Architecture compatibility (SDE)

**Note**: CI uses `cargo-nextest` for running tests. See [`.cargo/nextest.toml`](.cargo/nextest.toml) for test configuration (timeouts, retries, etc.).
