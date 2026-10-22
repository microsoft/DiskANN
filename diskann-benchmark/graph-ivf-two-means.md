# Two-means paired-distance baseline

This is the original synthetic-corpus experiment. The later
[24-round arithmetic-mean study](graph-ivf-stages-means.md) found confidence
intervals spanning zero for the two-means gain at every tested dimension.
The [new strategy experiments](graph-ivf-two-means-experiments.md) extend the
comparison to packing, multi-point kernels and dot-product formulations.
The [FBV8 follow-up](graph-ivf-two-means-fbv8.md) includes a full corpus comparison.
Do not treat the original small-run speedups below as a general guarantee.

Measured on 2026-10-21 against the first stack layer, commit
`3661de1b1216bfb45ceccecce49183d2585340d0` (PR #1502).

`fit_two_means` now computes a point's two squared-L2 distances in one SIMD
loop. Full SIMD vectors share the point load; each distance retains its own
four accumulators, original combination tree, and architecture-specific tail.
The scalar implementation retains the existing two-distance path. The change
preserves f64 centroid sums, point order, seeds, iteration count, empty-cluster
handling, first-center ties, and final reassignment across all split children.

## Measurement

Windows, AMD Ryzen AI Max+ 395 (16 cores / 32 logical processors), Rust 1.97.1,
release profile, repository x86-64-v3 flags. Each benchmark process uses one
thread and affinity mask 16 (logical processor 4). Before/after processes run
sequentially; there are no concurrent test/build jobs during these measurements.

Both versions use the same synthetic corpus: 65,536 vectors, data seed 42,
16 initial centroids, batch size 1,024, split threshold 2,048, five Lloyd steps,
build seed 7, and centroid graph degree 16 / build L 64. Dimensions are
128, 384, and 768. No work, candidate set, precision, or quality target was
reduced in the optimized build.

The [build workload](perf_test_inputs/graph-ivf-two-means-build.json) uses
13 independent builds per dimension and invocation. Six before/after rounds
alternate AB / BA order. Discarding the first build of **every** invocation
leaves 72 samples per version and dimension. The table reports pooled medians
of total build time, including the final centroid graph.

| Dimensions | Before | After | Reduction | IVF stage reduction |
| --- | ---: | ---: | ---: | ---: |
| 128 | 95.839 ms | 85.858 ms | 10.41% | 10.30% |
| 384 | 185.421 ms | 169.683 ms | 8.49% | 8.48% |
| 768 | 332.808 ms | 310.014 ms | 6.85% | 6.86% |

Per-round total-build medians, in milliseconds, make the variation visible:

| Round / order | 128 before / after | 384 before / after | 768 before / after |
| --- | ---: | ---: | ---: |
| 0 / AB | 88.404 / 83.805 | 202.200 / 171.510 | 269.797 / 269.681 |
| 1 / BA | 102.676 / 78.390 | 197.467 / 148.063 | 313.286 / 300.850 |
| 2 / AB | 80.080 / 87.460 | 160.285 / 169.778 | 328.110 / 300.501 |
| 3 / BA | 96.516 / 84.556 | 194.419 / 188.506 | 339.166 / 281.299 |
| 4 / AB | 101.702 / 84.846 | 178.249 / 190.148 | 345.100 / 346.268 |
| 5 / BA | 97.266 / 91.593 | 183.408 / 165.518 | 343.473 / 342.699 |

This desktop run has substantial scheduling/frequency variation: not every
round improves. These are measured aggregate gains for this workload, not a
universal speedup or a statistical performance gate. No iteration early exit
or benchmark timing boundary changed.

## Recall and storage work

The [search workload](perf_test_inputs/graph-ivf-two-means-search.json) uses
64 independent queries, k=10, nprobe=4/16/64, search L=128, one warmup pass and
two measured passes after each of three builds. Two rounds alternate AB / BA,
giving 108 search passes per version over the three dimensions.

Recall, returned-result counts, selected-list counts, scanned-vector counts,
centroid comparisons, and all build list/split statistics are **exactly equal**
between the two versions in every corresponding result.

| Dimensions | nprobe | Recall@10, both versions | Vectors scanned/query, both versions |
| --- | ---: | ---: | ---: |
| 128 | 4 | 0.243750 | 6233.484375 |
| 128 | 16 | 0.5890625 | 24403.328125 |
| 128 | 64 | 1.000000 | 65536 |
| 384 | 4 | 0.146875 | 5623.875 |
| 384 | 16 | 0.4703125 | 22962.406250 |
| 384 | 64 | 1.000000 | 65536 |
| 768 | 4 | 0.131250 | 5807.765625 |
| 768 | 16 | 0.4546875 | 23499.437500 |
| 768 | 64 | 1.000000 | 65536 |

The benchmark uses an **in-memory provider**. These checks establish unchanged
logical scan work; they do not measure SSD reads, bytes, cache misses, or Mimir
write amplification. A physical-I/O claim requires repeating the experiment
with the actual storage provider. Bitwise differential tests additionally
verify unchanged fitted centroids, final membership IDs/order, and RNG state.

Search QPS is also recorded, but this change does not optimize search. Across
the nine settings its median change ranged from -10.98% to +18.83% on this
noisy host; no search-latency improvement or non-regression claim is made.

## Reproduce

Build the benchmark on the base commit and save its executable as `before.exe`.
Build this stack layer with identical compiler flags and save it as `after.exe`.
Run the two workload files on each executable and retain separate JSON outputs.
For the table above, execute six build pairs and two search pairs, reversing
order on odd rounds; pin each process to the same CPU.

```powershell
cargo build --release -p diskann-benchmark --offline
./after.exe run --input-file diskann-benchmark/perf_test_inputs/graph-ivf-two-means-build.json --output-file target/after-build.json
./after.exe run --input-file diskann-benchmark/perf_test_inputs/graph-ivf-two-means-search.json --output-file target/after-search.json
```

The benchmark source, provider, measurement boundaries, and JSON configuration
are identical on both sides. Data generation, groundtruth, membership checking,
and recall calculation remain outside the timers.

## Correctness and portability

- All 160 `diskann-vector` tests passed in test and release profiles; both
  rustdoc examples passed.
- Paired distances match two independent existing L2 evaluations bit for bit
  on Scalar, V3, and V4. V4 was required with `WIDE_TEST_MIN_ARCH=x86-64-v4`
  and executed on this host. Tests cover tails, offset slices, near/exact ties,
  signed zero, extreme finite inputs, and an independent exact integer oracle.
- Five new clustering tests cover full Lloyd results, RNG state, f64 sums,
  empty children, invalid work, known means, and all-child membership order.
  The five build/search benchmark tests also passed.
- Full IVF testing reports **11 passed / 1 failed**: the pre-existing
  `invalid_initialization_and_uninitialized_insertion_fail` fails because
  initialization accepts a NaN centroid. Replacing the clustering source with
  the base commit's original source reproduces the same failure at
  `diskann/src/ivf/tests.rs:623`. Initialization is outside this optimization.
- A baseline `-C target-cpu=x86-64` build passed the paired tests on this host.
  AArch64 Windows library/tests cross-compilation passed; Neon execution and
  old-x86 CPU emulation were not available locally.
- Local LLVM line coverage: paired kernel 90/94 lines (95.74%); clustering
  94/94 lines (100%). This is not a Codecov patch-status result.
