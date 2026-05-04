When performing a code review, check that:
- No unit tests were eliminated without a strong reason.
- Additional dependencies introduced have a strong justification.
- Changes are not likely to increase build times.
- Each file has a license header.
```
/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
```

## SemVer and API Compatibility

- The workspace obeys SemVer. Removing or changing public API signatures (functions, types, re-exports) is a breaking change and requires a major version bump or a deprecated compatibility shim.
- Re-exports are part of the public API surface — removing them is also a breaking change.
- If changing public behavior, explain migration impact in the PR description.

## Error Handling

- Do not introduce `panic!` paths for recoverable errors — propagate with `Result` instead.
- Keep error types small. Avoid large enums/structs that blow up the stack; look for ways to reduce field sizes (e.g., compute derivable fields, use enums instead of `&'static str`).
- Prefer `ANNError::new(ANNErrorKind::…, e)` over the old `log`-style constructors, which force eager string formatting.
- When using `thiserror`, rely on `#[from]` for automatic `Error::source` chaining — do not format the inner error in the `#[error("…")]` display string.
- Include relevant context values (e.g., the kind, key, or dimension) in error messages for debuggability.

## Documentation

- Doc comments and README examples must match actual API signatures and serialized shapes. Stale examples that fail to compile or deserialize are treated as bugs.
- Do not leave dead references to APIs that no longer exist.
- When changing a function signature or removing a parameter, update all doc comments that mention the old signature.

## Constants and Assumptions

- Do not hardcode magic values — make them configurable with sensible defaults and document the rationale.
- If using `wrapping_add` or other wrapping arithmetic, justify why overflow is expected or acceptable.
- Add assertions for invariants that callers or maintainers would otherwise have to discover by reading the implementation.

## SIMD and Platform Portability

- Do not assume specific SIMD lane widths (e.g., `f32s::LANES == 8`). Code must be correct on AVX2, AVX-512, and ARM/NEON.
- When touching architecture-specific intrinsics, verify cross-platform behavior per `diskann-wide/README.md`.

## Testing

- Keep test helpers close to the code they exercise, typically in a `mod tests` at the bottom of the file or in an adjacent test module, guarded with `#[cfg(test)]`.
- Do not add tests for derived traits (`Clone`, `Debug`, `PartialEq`) or enums unless they have explicit behavior beyond the derive.
- Test edge cases like empty inputs (e.g., empty iterators) to lock in defined behavior and prevent divide-by-zero or NaN results.

## Rayon and Parallelism

- Never use the global Rayon thread pool. Always execute parallel work within the provided `RayonThreadPool` or `RayonThreadPoolRef`.
- Preserve deadlock-avoidance intent when modifying nested parallel loops. Be aware that combining blocking synchronization (e.g., mutex acquisition) with Rayon work-stealing can cause deadlocks.

## Unsafe Code and Safety

- Every `unsafe` block must have a `// SAFETY:` comment directly above it explaining why the operation is sound. This is enforced by the `undocumented_unsafe_blocks = "warn"` workspace lint.
- Safety comments must be specific and verifiable — state the concrete precondition that makes the operation safe (e.g., `// SAFETY: i + width <= len ensures this read is in-bounds`). Do not use vague justifications like `// SAFETY: this is safe`.
- Safety contracts on `unsafe fn` signatures must be internally consistent — if the documented precondition says `scratch.len() >= n`, ensure the implementation does not write beyond `n` elements (e.g., due to rounding up to a panel/block size).
- Prefer safe abstractions over raw `unsafe` when possible. Use `unsafe` only when there is a measurable performance benefit or when interfacing with FFI/intrinsics.
- For pointer arithmetic, prefer `offset_from` to express bounds rather than `wrapping_add` unless wrapping behavior is intentionally needed — and document why.
- When calling SIMD intrinsics or FFI, list the specific preconditions being satisfied (alignment, length, non-null, valid initialization).

## Naming

- Use names that reflect the current architecture, not historical ones. Rename outdated terms when refactoring.
- Struct and type names should map clearly to their domain concepts for easier mental mapping.
