# DiskANN repository instructions

DiskANN is a Rust implementation of scalable approximate nearest neighbor (ANN) search, organized as
a Cargo workspace of ~17 crates (edition 2021). Toolchain is pinned in `rust-toolchain.toml`.

Deeper guidance lives in `AGENTS.md`. For reviewing pull requests, a dedicated skill exists at
`.github/skills/diskann-pr-review/` — prefer it over these instructions when assessing a diff, as it
carries the full rule catalog and the evidence bar for each kind of change.

## Crate tiers

Tier determines error handling, allocation tolerance, and dependency direction.

| Tier | Crates | Character |
|---|---|---|
| 1 | `diskann-wide`, `diskann-vector` | SIMD, bit manipulation, type-width abstractions |
| 2 | `diskann-linalg`, `diskann-utils`, `diskann-quantization` | Core libraries |
| 3 | `diskann`, `diskann-providers`, `diskann-disk`, `diskann-label-filter` | Algorithm and storage |
| 4 | `diskann-benchmark*`, `diskann-tools` | Infrastructure and tooling |

Tier 1 and 2 crates may be depended on by anything. `diskann` may be depended on by any equal or
higher tier crate, **except** that `diskann-benchmark-runner` and `diskann-benchmark-simd` must not
depend on any Tier 3 crate (`diskann-benchmark-core` may depend on `diskann` only).

## Error handling

Choose by tier. There is no single workspace error type.

- **Low level (Tier 1–2):** bespoke, precise, non-allocating error types. Use `thiserror` for
  boilerplate; chain with `std::error::Error::source`. `diskann::ANNError` is *not* a suitable
  low-level error type.
- **Mid level (`diskann` algorithms):** `diskann::ANNError` and its context machinery, for
  unrecoverable errors. Use `#[track_caller]` on conversions so the source location is useful.
  Traits with associated error types should consider `diskann::error::ToRanked` rather than
  `Into<ANNError>` when non-critical errors must be representable.
- **High level (tooling, benchmarks):** `anyhow::Error` is appropriate.

**Never introduce a single crate-level catch-all error enum.** It documents nothing about how an
individual function can fail, produces worse messages than bespoke types, inflates struct size, and
generates branch-heavy `Drop` implementations.

Do not add an error variant that discards a recovery payload — if a fallible conversion consumes an
owned buffer, the error should hand that buffer back rather than drop it.

## Unsafe code

- Every `unsafe` block needs a `// SAFETY:` comment naming the invariant being upheld. The workspace
  lints `undocumented_unsafe_blocks`.
- `unsafe` introduced for performance requires measured justification (roughly 10%+), a benchmark in
  CI to defend it, and a safe wrapper around it. See `rfcs/00109-unsafe-rust.md`.
- Architecture-specific intrinsics must be validated cross-platform (AVX-512 via SDE, aarch64 on
  x86-64) per `diskann-wide/README.md`. Check that the scalar fallback path is covered too.

## Testing

- Patch coverage on changed lines must be at least **90%**; this gate blocks merges
  (`.codecov.yml`, `informational: false`).
- Changes to algorithm behavior need a **baseline** regression test capturing both IDs and
  distances, plus invariant assertions — a baseline alone is insufficient, since a wrong baseline can
  be committed. See `diskann/src/test/cache.rs`.
- Concurrency changes need a stress test with enough threads and iterations to surface
  non-determinism. Document benign races and why they are acceptable.
- Do **not** add tests for derived traits (`Clone`, `Debug`, `PartialEq`), or for enums with no
  explicit functionality.
- Unit tests must not be removed without a stated, strong reason.

## API design

- Keep invariants in types: private fields with validating constructors, enums rather than `Option`
  where the set of cases may grow, checked conversions rather than `as`, failure at parse/load time
  rather than mid-run.
- Do not widen visibility (`pub(crate)` to `pub`) without a stated reason.
- Do not turn an infallible constructor into a panicking one; return an error or take a type that
  makes the invalid state unrepresentable (for example `NonZeroUsize`).
- A refactor described as mechanical must not silently drop a trait implementation or constructor.
- Avoid `unwrap()`, `expect()`, and `panic!` in non-test library code.

## Documentation

Less is more. Do not restate what the signature already shows, and do not maintain hand-written
lists of types or functions that rustdoc generates for free. Do document non-obvious behavior,
safety requirements, and design intent, using `# Errors`, `# Safety`, `# Panics`, and `# Example`
sections.

## Hygiene

- Every new file carries the license header:

  ```rust
  /*
   * Copyright (c) Microsoft Corporation.
   * Licensed under the MIT license.
   */
  ```

- New dependencies need a strong justification, including their transitive cost. Moving code to a
  higher tier often removes the need for a dependency entirely.
- Changes should not meaningfully increase build times.
- Before committing: `cargo fmt --all` and `cargo clippy --workspace --all-targets -- -D warnings`.

## When an RFC is expected

Cross-cutting changes, new crates, new cross-crate traits, new distance functions, storage layouts
or index formats, and anything with backward-compatibility implications belong in an RFC under
`rfcs/`. Routine single-crate API additions, bug fixes, and refactors do not.
