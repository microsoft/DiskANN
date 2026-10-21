# Split reassignment SIMD measurements

For the full 1,087,932-vector Enron 384D FP32 follow-up, see
[Enron measurements](graph-ivf-enron.md). The results below use synthetic data.

Measured on 2026-10-21 against stack layer 2, commit
`0735fc3db269986613da9693c3ec269034ebd9b9` ([PR #1503](https://github.com/microsoft/DiskANN/pull/1503)).

## Scope

After fitting two children per split parent, every point still compares against
**all** fitted children and selects one owner. This is the current reassignment
algorithm's top-k with **k=1**. This layer does not add configurable k, replication,
neighborhood reassignment, or an IVF query-search API.

The assignment kernel reuses the paired squared-L2 kernel from layer 2, sharing
point loads between two child distances. Short candidate rows select immediately;
longer rows reuse one distance scratch row and test eight candidates against the
incumbent with SIMD. Only potentially improving groups enter scalar, stable
selection. Signed integer keys preserve `f32::total_cmp` ordering, including
infinity, signed zeros, and NaN payloads; equal keys retain the first ordinal.
The selector orders the distances it receives exactly; the distance kernel's
bitwise guarantee applies to **finite vector inputs**, including overflow to infinity.

Scratch is O(children), allocated once per split and reused for every borrowed
point. No points-by-children matrix, vector copy, candidate pruning, reduced
iteration count, altered seed, or approximate distance is introduced. Public
provider and index APIs are unchanged, with no new dependencies.

## Build benchmark

Windows, AMD Ryzen AI Max+ 395, Rust 1.97.1, release profile and the repository's
x86-64-v3 flags. Each process uses one thread and CPU affinity mask 16 (logical
processor 4). All timed processes run sequentially with no concurrent build/test
jobs started by this task.

Both versions use 65,536 FP32 vectors, corpus seed 42, 16 initial centroids,
split threshold 2,048, five Lloyd steps, build seed 7, and centroid graph degree
16 / L=64. The original batch-1,024 workload is retained. A batch-16,384 workload
also exercises larger simultaneous split working sets. Compare versions **within**
a workload: changing batch size changes list layouts and cannot establish a
quality-equivalent advantage between workloads.

For each workload and dimension: six AB/BA alternating rounds, 13 builds per
invocation, discarding the first build of every invocation, leave 72 samples
per version. These are pooled medians of total build time, including the final
centroid graph. A negative reduction denotes a slowdown.

| Batch size | Dimensions | Before (ms) | After (ms) | Reduction |
| ---: | ---: | ---: | ---: | ---: |
| 1,024 | 128 | 77.909 | 76.678 | 1.58% |
| 1,024 | 384 | 156.610 | 160.059 | -2.20% |
| 1,024 | 768 | 384.268 | 376.505 | 2.02% |
| 16,384 | 128 | 97.851 | 91.541 | 6.45% |
| 16,384 | 384 | 355.815 | 333.840 | 6.18% |
| 16,384 | 768 | 550.076 | 538.276 | 2.15% |

The larger-batch workload improves by 2.15–6.45% in these pooled measurements.
The original workload is mixed: 128/768 dimensions improve by about 1.6–2.0%,
while 384 dimensions is 2.20% slower. This host has substantial timing variation;
these results do **not** establish a universal build-time non-regression.

Per-round total-build medians, before / after in milliseconds:

| Batch / dimensions | 0 (AB) | 1 (BA) | 2 (AB) | 3 (BA) | 4 (AB) | 5 (BA) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 / 128 | 78.08 / 78.10 | 78.73 / 75.34 | 80.46 / 78.60 | 77.32 / 74.66 | 77.60 / 75.72 | 77.01 / 80.98 |
| 1,024 / 384 | 144.99 / 157.34 | 163.06 / 154.25 | 163.93 / 187.21 | 156.52 / 152.02 | 152.12 / 150.16 | 167.01 / 167.96 |
| 1,024 / 768 | 376.54 / 389.97 | 376.68 / 372.82 | 408.49 / 398.45 | 361.23 / 384.88 | 388.97 / 347.01 | 370.81 / 345.37 |
| 16,384 / 128 | 101.82 / 95.79 | 98.32 / 90.08 | 100.25 / 91.68 | 92.10 / 90.72 | 95.07 / 87.04 | 95.72 / 94.97 |
| 16,384 / 384 | 352.80 / 336.88 | 397.40 / 344.26 | 356.36 / 331.02 | 346.43 / 330.72 | 347.91 / 329.83 | 369.91 / 320.96 |
| 16,384 / 768 | 545.18 / 532.74 | 592.12 / 516.94 | 545.01 / 532.28 | 553.15 / 578.62 | 533.81 / 522.75 | 545.97 / 565.64 |

The batch-16,384 build can leave lists above the 2,048 split threshold (maximum
3,628 / 1,403 / 2,112 at 128 / 384 / 768 dimensions, respectively). This is the
existing single-split-per-operation policy, unchanged on both versions.

## Search quality and logical work

Each workload uses 64 independent queries, k=10, nprobe=4/16/64, search L=128,
one warmup and two measured passes after each of three builds. Two AB/BA rounds
produce 108 search passes per workload and version, 216 across both workloads.
Every corresponding result has **exactly equal** recall, returned-result count,
selected-list count, scanned-vector count, centroid comparisons, and build
membership/split statistics. The test suite separately checks centroid bits,
complete member IDs and their order, and final RNG state.

| Batch | Dimensions | nprobe | Recall@10, both | Vectors scanned/query, both | Median QPS change |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 128 | 4 | 0.24375 | 6233.484375 | -1.80% |
| 1,024 | 128 | 16 | 0.5890625 | 24403.328125 | -11.27% |
| 1,024 | 128 | 64 | 1 | 65536 | 4.96% |
| 1,024 | 384 | 4 | 0.146875 | 5623.875 | -0.55% |
| 1,024 | 384 | 16 | 0.4703125 | 22962.40625 | -7.63% |
| 1,024 | 384 | 64 | 1 | 65536 | -2.53% |
| 1,024 | 768 | 4 | 0.13125 | 5807.765625 | 14.06% |
| 1,024 | 768 | 16 | 0.4546875 | 23499.4375 | 9.04% |
| 1,024 | 768 | 64 | 1 | 65536 | 0.82% |
| 16,384 | 128 | 4 | 0.4 | 13189.5625 | -12.96% |
| 16,384 | 128 | 16 | 0.796875 | 52575 | -26.50% |
| 16,384 | 128 | 64 | 1 | 65536 | 6.27% |
| 16,384 | 384 | 4 | 0.1625 | 4764.90625 | -1.78% |
| 16,384 | 384 | 16 | 0.4703125 | 19070.1875 | -0.14% |
| 16,384 | 384 | 64 | 1 | 65536 | 0.82% |
| 16,384 | 768 | 4 | 0.1546875 | 7088.046875 | 8.47% |
| 16,384 | 768 | 16 | 0.509375 | 28360.15625 | -5.52% |
| 16,384 | 768 | 64 | 1 | 65536 | -0.35% |

This is the benchmark-local **in-memory provider**. Equal logical scan work is
not a measurement of physical SSD I/O, cache misses, bytes, or Mimir write
amplification. Search code is unchanged, but measured median QPS varies from
-26.50% to +14.06%; no search performance non-regression claim is made.
Synthetic uniform vectors and a small query set are a regression workload,
not a production relevance/recall evaluation.

## Reproduce

Save the release benchmark executable at the base commit as `before.exe`, then
build this layer identically as `after.exe`. Run each JSON input on both binaries
with distinct output files; alternate order across six build pairs and two
search pairs, pinning processes to the same CPU.

- Original build: [graph-ivf-two-means-build.json](perf_test_inputs/graph-ivf-two-means-build.json)
- Original search: [graph-ivf-two-means-search.json](perf_test_inputs/graph-ivf-two-means-search.json)
- Larger-batch build: [graph-ivf-reassignment-batched-build.json](perf_test_inputs/graph-ivf-reassignment-batched-build.json)
- Larger-batch search: [graph-ivf-reassignment-batched-search.json](perf_test_inputs/graph-ivf-reassignment-batched-search.json)

```powershell
cargo build --release -p diskann-benchmark --offline
./after.exe run --input-file diskann-benchmark/perf_test_inputs/graph-ivf-reassignment-batched-build.json --output-file target/after-build.json
./after.exe run --input-file diskann-benchmark/perf_test_inputs/graph-ivf-reassignment-batched-search.json --output-file target/after-search.json
```

An optional release-only kernel measurement compares the old independent-distance
selector and the optimized selector in the same process. It checks all generated
query answers before timing, alternates order, uses black-boxed inputs/results,
and has no timing assertion. It measures reassignment only, not full build/search:

```powershell
cargo test -p diskann --release --lib assignment_kernel_timing --offline -- --ignored --nocapture --test-threads=1
```

The benchmark implementation, provider, timing boundaries, and workload for each
comparison are identical. Generation, groundtruth, membership validation, and
recall evaluation remain outside the timers.

## Correctness and portability

- Nine clustering/assignment tests passed in test and release profiles. Coverage
  includes all SIMD/tail positions, candidate counts 1–35, multiple split parents,
  arbitrary/nonmonotonic IDs, first-on-tie/infinity behavior, and an independent
  exact-nearest oracle. The old selector remains test-only and independent.
- Total-order selector differential tests run on Scalar, V3, and V4; V4 was
  required with `WIDE_TEST_MIN_ARCH=x86-64-v4` and executed on this host.
- All five static build/search benchmark tests passed. Full IVF testing reports
  **15 passed / 1 failed / 1 manually ignored benchmark**. The failure is the
  pre-existing `invalid_initialization_and_uninitialized_insertion_fail` at
  `diskann/src/ivf/tests.rs:623`: initialization accepts a NaN centroid. The
  initialization path and test are unchanged from the base layer.
- Baseline `-C target-cpu=x86-64` build: all nine clustering/assignment tests
  passed on this host. AArch64 Windows library/test cross-compilation passed.
  Neon execution and old-x86 emulation were unavailable locally; cross-compilation
  is not runtime validation.
- Local LLVM line coverage: assignment 70/72 (97.22%), clustering 84/84 (100%).
  This is local source coverage, not a Codecov patch-status result.
- Workspace formatting and Clippy with warnings denied passed.
