# Multi-vector benchmark — kernel-author workflow

The multi-vector benchmark dispatches through `diskann-quantization`'s
`build_max_sim_f32` / `build_max_sim_f16` factory. Selection is driven by a
non-exhaustive `MaxSimIsa` enum. To add a new in-tree experimental kernel,
extend the enum + factory + the benchmark's shadow enum.

## Steps

1. **Library: variant + factory arm.** In
   `diskann-quantization::multi_vector::distance`:
   - Add a new variant to `MaxSimIsa` (in `isa.rs`).
   - Implement `MaxSimKernel<T>` for your kernel struct (in `factory.rs`,
     next to `Prepared` and `ReferenceKernel`).
   - Add a matching arm to `build_max_sim_f32` and/or `build_max_sim_f16`
     that constructs your kernel and hands it to `erase.erase(...)`.

2. **Benchmark: matching shadow variant.** In
   `diskann-benchmark::inputs::multi_vector`:
   - Add the same variant to `BenchIsa`.
   - Add the matching arm to `From<BenchIsa> for MaxSimIsa`.

3. **Run.** Set `"isa": "your-variant"` in the JSON job; the existing
   `KernelF32` / `KernelF16` benchmark entries handle the rest. No new
   `Benchmark` registration required.

## Why two enums?

`MaxSimIsa` (library) and `BenchIsa` (benchmark) are kept separate so the
library doesn't pin its public API on a serde version or a particular JSON
shape. The benchmark owns its kebab-case JSON layout; the library is
serde-agnostic. Mirroring variant-for-variant is intentional — small price
for keeping the library boundary clean.

## Background

The factory follows the BYOTE ("Bring your own type erasure") pattern
described in [RFC #1068]. If you want your kernel packaged as something
other than `Box<dyn MaxSimKernel<T>>` (e.g. composed with chamfer summing,
or wrapped in a custom thin trait), implement your own `Erase<T>` and pass
it to the factory in place of `BoxErase`.

[RFC #1068]: https://github.com/microsoft/DiskANN/pull/1068
