# Implementation review notes

The active implementation follows the requested 16-point block-transposed, four
f32x8-accumulator scheme. Packing is owned by the low-level primitive; the IVF
adapter owns baseline fallback and the shared f64 state transitions. No timings,
RNG or IVF identifiers entered the primitive. Removed strategies have no active
Rust or CLI dispatch, while their historical evidence remains intact.

Public shape validation precedes all pointer use. Each 16-row block has exactly
16 times D readable floats, including zero-initialized tail padding. Stores mask
off padding; IDs are checked before shared sum indexing. Integer-oracle tests
poison tail lanes and guard output sentinels. Native scalar/V3/V4 paths passed;
AArch64 compiled. This does not substitute for unavailable Neon runtime/SDE.

The compiler artifact disassembly is assignment-disassembly.txt. The main L2
dimension loop (roughly offsets 0x220 through 0x2aa) uses broadcasts, contiguous
YMM loads, four chained FMA distances, and no vector accumulator stack spill in
that loop. It schedules two dimensions together while preserving each chain's
order. The dot loop (roughly 0x7d0 through 0x899) schedules four dimensions into
the same four dependency chains. These compiler unrolls are not independent U=2
or U=4 numeric candidates. XMM saves/restores in the prologue/epilogue implement
the Windows ABI and are not evidence of inner-loop accumulator spilling.

The explicit assignment fallback only addresses nonfinite scores. Finite
disagreements are part of this experiment's contract; they must stay visible.
The new diagnostics expose divergence at intermediate steps and final results,
not only at the final paired-L2 reclassification. The standalone cancellation
unit test and seeded workload divergences demonstrate why bit equality cannot
be claimed.

Metric integrity review: preparation, allocation and release are charged;
baseline timings come from the same binary and their own ABBA block; all queries
and all base rows are retained; no outlier is removed. Neither shorter builds
with changed partitions nor changed scan counts can alone establish an
arithmetic speedup. The memory-backed provider exposes logical scans, not SSD
traffic or Mimir write amplification. The existing NaN-initialization failure
is separately disclosed and has not been hidden by changing the test.

Skills consulted: code-simplifier for module boundaries and scoped cleanup;
pr-writer for preserving the repository PR template. No subagents were used.
