# AGENTS.md

## Verification

- Run: `cargo fmt --all --check && cargo clippy --workspace --all-targets --config 'build.rustflags=["-Dwarnings"]' && cargo test`
- Or: `./scripts/verify` (bash) / `./scripts/verify.ps1` (PowerShell)
- If it fails, fix the root cause and re-run before committing.

## Environment

- Rust edition 2021, toolchain version in `rust-toolchain.toml`
- Cargo workspace with 18 crates across 4 tiers (see `Cargo.toml`)
- CI uses `cargo-nextest` — see `.cargo/nextest.toml` for timeouts/retries
- `rustfmt` and `clippy` not installed by default — run `./scripts/setup` first

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

## Crate Dependency Rules

- Tier 1 and Tier 2 crates may be added as dependencies of any internal crate
- `diskann` may be added as a dependency of any equal or higher tier internal crate except those below
- Do not add Tier 3 crates as dependencies of these Tier 4 crates:
  - `diskann-benchmark-runner`
  - `diskann-benchmark-core` (`diskann` is allowed)
  - `diskann-benchmark-simd`

## Error Handling

Three regimes — use the one matching the crate's tier:

- **Low-level crates**: Bespoke, precise, non-allocating error types. Use `thiserror`. Chain with `std::error::Error::source`. Do not use `ANNError`.
- **Mid-level (diskann algorithms)**: Use `diskann::ANNError` with its context machinery. Use `#[track_caller]` when converting to ANNError. Use only for unrecoverable errors. Consider `diskann::error::ToRanked` for traits with associated error types.
- **High-level (tooling)**: Use `anyhow::Error`.
- **Never** use a single crate-level error enum — it provides no per-function failure docs, generates oversized structs, and bloats code.

## Guardrails

- Do not use `rand::thread_rng` — use functions from `random.rs`
- Do not use rayon global thread pool methods — use `*_in_pool` wrappers from `rayon_utils.rs` (see `clippy.toml`)
- Do not use `VirtualStorageProvider::new_physical()` in tests — use `new_overlay()`
- Do not use `vfs::PhysicalFS::new` in tests — use `VirtualStorageProvider::new_overlay()`
- When touching architecture-specific intrinsics, run cross-platform validation per `diskann-wide/README.md`

## Testing

- Baseline caching system for regression detection — see `diskann/src/test/cache.rs` and `diskann/src/test/cmp.rs`
- Do not add tests for derived traits (Clone, Debug, PartialEq)
- Do not add tests for enums unless they have explicit functionality
- Look for existing setup/execution infrastructure before creating new patterns
- Code coverage of changes is required for PRs

## AVX-512, Aarch64, and multi-platform

When touching architecture-specific intrinsics, run cross-platform validation per `diskann-wide/README.md`:

- Testing AVX-512 code on non-AVX-512 capable x86-64 machines.
- Testing Aarch64 code on x86-64 machines.
- Testing code compiled for and running on the `x86-64` CPU (no AVX/AVX2) does not execute unsupported instructions.

## Constraints

- Keep diffs minimal and scoped to the request
- Update or add tests for any behavior change
- Do not modify CI, dependency versions, or security settings unless asked
- Never print, log, or commit secrets

## Definition of Done

- `./scripts/verify` passes (fmt + clippy + tests)
- No new lint warnings introduced
- Changes are scoped to the request — no drive-by refactors

## Where to find more

- Path-specific rules: `.github/instructions/`
- Multi-step workflows: `.github/skills/*/SKILL.md`

## Best Practices for Writing an Effective AGENTS.md

- Keep AGENTS.md concise — ideally within 30–80 lines.
- Do not include structural details such as directory listings or README‑style content, as research shows these can degrade agent performance.