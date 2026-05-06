---
applyTo: '**/*.rs'
---
# Unsafe Code Conventions
**When to read:** Writing or modifying Rust source files that use `unsafe`.

- Document every `unsafe` block with a `// SAFETY:` comment explaining the invariant
- Prefer safe abstractions over raw unsafe code
- When touching architecture-specific intrinsics (SIMD), validate across platforms per diskann-wide/README.md
