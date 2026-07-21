# Staged MaxSim kernel (`tiled_reduce_staged`) — design

> **Status: experimental POC.** This module validates a design hypothesis; it is
> not yet the production MaxSim path. See *Scope* below.

## Thesis

One generic, cache-tiled reduction driver — [`tiled_reduce_staged`](driver.rs) —
can compute multi-vector MaxSim/Chamfer for **any element type and any
quantization** by swapping small pluggable *stages*, instead of forking the tiled
loop nest once per datatype. This module proves that claim with two
instantiations:

- **f32** — bit-identical to, and on par with, the hand-fused V3 kernel.
- **4-bit MinMax-quantized `i8`** — a *new datatype* added by swapping two stages
  only, with a real speedup over the per-pair SIMD reference.

## Shape

The driver owns tiling/blocking and walks query×doc tiles. Per tile it calls four
pluggable stages, defined in [`mod.rs`](mod.rs):

| Stage | Trait | Role |
|-------|-------|------|
| A | `StagedKernel` | SIMD inner kernel; writes a per-pair accumulator `Acc` into a `partial` buffer |
| B | `Postprocess` | maps `Acc → Score` (e.g. dequantize) using optional per-call metadata (`scratch_len` + `apply`) |
| C | `Reducer` | folds per-doc scores into the running MaxSim (max) |
| — | `StagedConvert` | optional input conversion at tile load (identity for f32; future: on-the-fly quantize) |

The driver allocates **all** scratch (`partial`, `scored`, conversion buffers)
from a caller-supplied `ScopedAllocator`. Callers size nothing.

## What varies per axis (the generality proof)

| Axis | f32 | 4-bit MinMax (`i8`) |
|------|-----|---------------------|
| Stage A kernel | `StagedF32Kernel` ([`v3.rs`](v3.rs)) | `StagedI8Kernel` ([`i8.rs`](i8.rs)) |
| `Acc` type | `f32` | `i32` |
| Stage B postprocess | `Identity` — `scratch_len = 0`, returns acc ([`maxsim.rs`](maxsim.rs)) | `MinMaxPostprocess` — `a·x + b` dequant → `f32` ([`i8.rs`](i8.rs)) |
| Stage C reducer | `MaxReducer` ([`maxsim.rs`](maxsim.rs)) | `MaxReducer` *(shared, unchanged)* |
| Convert | identity | identity (codes are pre-quantized) |

Only **Stage A + the `Acc` type + Stage B** change between f32 and quantized; the
driver, tiling, and reducer are reused verbatim. That is the thesis.

## Evidence

- **f32 = parity.** The staged f32 path is bit-for-bit equal to the hand-fused V3
  kernel (it *is* the same math, restructured) and within ±1.7% throughput.
  - Tests: `staged_matches_fused_v3` ([`v3.rs`](v3.rs)),
    `staged_f32_arena_reuse` ([`../../factory.rs`](../factory.rs)).
  - Benches: `example/multi-vector-staged.json` (fused vs staged sweep),
    `example/multi-vector-3way.json` (reference vs fused vs staged).
- **Quantized = new datatype, real win.** 4-bit MinMax over `i8` codes was added
  with **no driver change**; it is correct and runs **1.5–4.1×** faster than the
  per-pair SIMD reference across a dim sweep.
  - Tests: `staged_i8_matches_minmax_reference`,
    `staged_i8_arena_reuse_across_calls`, `staged_i8_multi_tile_tiny_budget`
    ([`i8.rs`](i8.rs)).
  - Bench: `example/multi-vector-quant.json` (reference vs staged).
  - The reference (`MinMaxKernel`) is **not** scalar: its per-pair inner product
    over 4-bit codes is itself SIMD. The staged win comes from
    fusion / block-transposition / tiling, not from SIMD-vs-scalar.

## Scratch & allocation

Driver-owned scratch comes from a passed `ScopedAllocator`. For a
zero-allocation steady state, callers reuse a single-owner resettable bump arena,
[`ResettableArena`](arena.rs):

- the f32 kernel reuses one via `RefCell<F32StagedScratch>`
  (`PreparedStaged` in [`../../factory.rs`](../factory.rs));
- the quantized POC owns one in `QuantStagedQuery` ([`i8.rs`](i8.rs)) and `reset`s
  it per call.

`ResettableArena` is deliberately **not** `Clone`/`Sync`: `reset(&mut self)` is
sound only because the borrow checker forbids resetting while any
`ScopedAllocator` still borrows it. (The shared `BumpAllocator` is grow-only and
`Sync`, so it has no `reset`.)

## Scope

- The f32 staged kernel is wired end-to-end and selectable for A/B benchmarking as
  `MaxSimIsa::X86_64_V3_Staged` ([`../../isa.rs`](../isa.rs)); it coexists with the
  fused `X86_64_V3` path.
- The quantized path is a standalone POC entry (`QuantStagedQuery` /
  `QuantStagedDocs`), **not** yet behind a `MaxSimIsa` variant or a productized
  storage `Repr`.
- Deferred: a quantized storage `Repr`, folding the quantized path into
  `MaxSimIsa` / the factory, V4 (AVX-512) Stage-A kernels, and richer reducers
  (argmax / top-k).
