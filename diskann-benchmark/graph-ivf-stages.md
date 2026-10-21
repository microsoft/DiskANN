# Graph-IVF split phase comparison

Measurements captured on 2026-10-21. All three stack layers use the phase timing
implementation from the first-layer commit `f7b706bed29a5b7768afccfaae63084bba79e54e`.
This report records observed wall times, including slowdowns. The host was noisy;
the measurements validate the new breakdown and quality checks, but are not a
stable performance gate.

## Versions and workload

| Layer | Optimization | Measured commit |
| --- | --- | --- |
| 1 / #1502 | Scalar selection over existing SIMD L2 calls | `f7b706bed29a5b7768afccfaae63084bba79e54e` |
| 2 / #1503 | Paired L2 in two-means | `1db048bd2b7ca7ae87272119445dc3322dbb9ac9` |
| 3 / #1504 | Paired L2 and grouped top-1 in final reassignment | `e4aefd1026a2449d2ea22827fead924eda49ff4b` |

The benchmark code, data, and timer boundaries are identical across these commits.
Each layer retains the real initialization, insertion, split, and publication path.
The code being compared is the optimization added by each successive layer.

Windows, AMD Ryzen AI Max+ 395, Rust 1.97.1, repository x86-64-v3 release flags,
one thread, process affinity mask 16 (logical processor 4). Runs are sequential;
this task did not run builds/tests concurrently with timing. Background system
activity was not controlled or measured.

The primary workload uses 65,536 synthetic FP32 vectors at 128/384/768 dimensions,
corpus seed 42, **batch size 4,096**, 16 initial centroids, split threshold 2,048,
five Lloyd steps, build seed 7, and centroid graph degree 16 / L=64. The rationale
and pinned PoC source references are in [the benchmark guide](graph-ivf.md#batch-size-for-phase-comparisons).
The PoC's large-corpus example specifies 4,096; its 256-point smoke example uses 64.
Neither is a hard-coded library default. The internal 4,096-point reassignment tile
is a separate parameter from the insertion batch.

Six build rounds use all orderings: 123, 321, 231, 132, 312, 213. Each invocation
performs 13 builds per dimension; its first build is discarded. This leaves
72 retained samples per layer/dimension, from 702 completed builds in total.
No slow samples other than this uniform warmup rule are removed.

## Phase wall times

Each row reports independently pooled medians in milliseconds. Two-means includes
centroid/sum scratch allocation, seeding, Lloyd assignment, and centroid updates.
Reassignment includes scratch, distances/top-1 selection, counting, and materializing
member lists. Other IVF is the measured IVF interval minus those two disjoint phases;
it includes routing, provider work, and orchestration. Total also includes the final
centroid graph.

| Dimensions | Layer | Two-means (ms) | Reassignment (ms) | Other IVF (ms) | Total build (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 1 | 53.216 | 12.318 | 101.479 | 156.489 |
| 128 | 2 | 31.578 | 15.156 | 104.685 | 164.465 |
| 128 | 3 | 26.753 | 8.446 | 112.452 | 148.255 |
| 384 | 1 | 95.786 | 54.759 | 179.481 | 325.823 |
| 384 | 2 | 102.515 | 55.367 | 178.821 | 330.072 |
| 384 | 3 | 94.774 | 33.346 | 179.914 | 331.822 |
| 768 | 1 | 255.600 | 66.755 | 300.091 | 615.038 |
| 768 | 2 | 222.500 | 72.161 | 289.518 | 572.290 |
| 768 | 3 | 229.254 | 67.931 | 297.458 | 590.451 |

For **each individual build**, the two split phases plus other IVF equal IVF time.
Independently computed medians need not add up to the median total; use individual
JSON rows for time-allocation charts or sums.

| Dimensions | Two-means reduction, 1 to 2 | Reassignment reduction, 2 to 3 | Total reduction, 1 to 3 |
| ---: | ---: | ---: | ---: |
| 128 | 40.66% | 44.27% | 5.26% |
| 384 | -7.03% | 39.77% | -1.84% |
| 768 | 12.95% | 5.86% | 4.00% |

Negative reduction means slower. In particular, two-means at 384 dimensions is
7.03% slower in the pooled comparison, and total build is 1.84% slower. Also,
two-means code is identical in layers 2 and 3 but their timings differ. These
variations prevent attributing every measured change to the optimization. Do not
present the positive values above as guaranteed speedups or select a different
batch size just to obtain a larger gain.

Per-round medians show the variation. Every timing cell is layer 1 / 2 / 3 in ms:

| Dimensions | Round | Two-means | Reassignment | Total build |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 0 | 41.095 / 39.056 / 34.559 | 12.388 / 14.993 / 12.275 | 158.522 / 206.920 / 120.512 |
| 128 | 1 | 26.896 / 22.117 / 61.082 | 26.748 / 8.585 / 39.982 | 155.859 / 145.970 / 207.695 |
| 128 | 2 | 28.962 / 23.351 / 22.920 | 10.267 / 40.638 / 5.668 | 148.813 / 150.214 / 141.192 |
| 128 | 3 | 57.635 / 42.981 / 27.187 | 11.200 / 17.198 / 9.247 | 154.219 / 181.989 / 156.631 |
| 128 | 4 | 69.473 / 62.863 / 23.430 | 47.887 / 16.392 / 7.011 | 230.782 / 211.114 / 147.273 |
| 128 | 5 | 27.004 / 29.708 / 23.113 | 9.429 / 43.165 / 6.426 | 150.154 / 159.523 / 145.458 |
| 384 | 0 | 105.370 / 162.747 / 120.305 | 53.786 / 66.113 / 29.658 | 326.558 / 446.069 / 334.036 |
| 384 | 1 | 86.372 / 82.694 / 125.374 | 55.104 / 34.247 / 24.357 | 324.193 / 305.177 / 402.271 |
| 384 | 2 | 84.675 / 72.064 / 72.673 | 53.648 / 21.070 / 17.318 | 318.236 / 272.449 / 269.682 |
| 384 | 3 | 99.329 / 115.674 / 115.238 | 58.607 / 60.144 / 55.087 | 382.726 / 340.727 / 341.225 |
| 384 | 4 | 166.519 / 116.061 / 134.420 | 61.660 / 59.757 / 54.233 | 457.747 / 360.604 / 346.657 |
| 384 | 5 | 87.376 / 82.113 / 80.776 | 54.169 / 24.567 / 18.489 | 321.881 / 287.738 / 280.912 |
| 768 | 0 | 335.048 / 283.860 / 214.952 | 99.740 / 99.450 / 84.103 | 742.337 / 725.229 / 557.645 |
| 768 | 1 | 211.898 / 189.751 / 223.809 | 63.279 / 63.180 / 84.951 | 531.783 / 530.325 / 614.096 |
| 768 | 2 | 236.125 / 184.051 / 186.677 | 74.446 / 81.075 / 65.492 | 558.855 / 537.063 / 522.100 |
| 768 | 3 | 316.806 / 191.525 / 266.776 | 67.023 / 64.234 / 63.102 | 806.520 / 525.652 / 588.626 |
| 768 | 4 | 292.189 / 253.527 / 384.769 | 47.818 / 70.673 / 83.848 | 740.222 / 599.621 / 877.633 |
| 768 | 5 | 194.949 / 195.746 / 222.087 | 66.958 / 64.209 / 82.423 | 584.882 / 531.403 / 584.001 |

## Work and search quality

Every corresponding build has equal inserted counts, split-parent counts,
split-point visits, histograms, list counts, empty-list counts, and maximum list
sizes. Each insertion contributes once to the histogram, including zero splits;
its key P implies 2P reassignment candidates. A point may be visited again by a
later split, so split-point visits can exceed corpus size.

| Dimensions | Split parents | Split-point visits | Final lists | Largest list | Parents per insertion: count |
| ---: | ---: | ---: | ---: | ---: | --- |
| 128 | 35 | 82116 | 51 | 1976 | {"0":5,"1":5,"2":1,"4":2,"5":1,"7":1,"8":1} |
| 384 | 35 | 79918 | 51 | 2028 | {"0":5,"1":4,"2":2,"4":2,"5":1,"7":2} |
| 768 | 38 | 85116 | 54 | 2033 | {"0":3,"1":1,"2":4,"3":5,"4":1,"5":2} |

Each histogram sums to 16 insertions. This directly measures candidate-set coverage
rather than inferring it from a mean parent count. No list exceeds the split
threshold in this particular workload; the algorithm does not generally guarantee
that final bound.

Search uses 64 queries, k=10, nprobe=4/16/64, L=128, one warmup and two measured
passes after each of three builds. Two rounds in orders 123 and 321 produce
108 passes per layer, 324 total. All corresponding non-timing search fields are
exactly equal, including returned results (10 per query), recall, and logical work:

| Dimensions | nprobe | Recall@10, all layers | Vectors scanned/query | Lists scanned/query | Centroid comparisons/query |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 4 | 0.2359375 | 6251.828125 | 4 | 52 |
| 128 | 16 | 0.6078125 | 24942.78125 | 16 | 52 |
| 128 | 64 | 1 | 65536 | 51 | 52 |
| 384 | 4 | 0.171875 | 6236.265625 | 4 | 52 |
| 384 | 16 | 0.4875 | 25051.234375 | 16 | 52 |
| 384 | 64 | 1 | 65536 | 51 | 52 |
| 768 | 4 | 0.140625 | 6048.578125 | 4 | 55 |
| 768 | 16 | 0.4578125 | 23916.234375 | 16 | 55 |
| 768 | 64 | 1 | 65536 | 54 | 55 |

Search timings are particularly unstable: one 64-query pass lasted 248.755 seconds,
while other passes with the same setting were much shorter. The cause of that
stall was not measured. Raw times are retained, but search QPS is not used to claim
speedup or non-regression. This is an in-memory provider: logical scan counts do not
measure physical SSD reads, bytes, cache misses, or Mimir write amplification.

## Reproduce and validate

Build each measured commit with the same release flags and save a separate
benchmark executable. Run the shared [build workload](perf_test_inputs/graph-ivf-stages-build.json)
and [search workload](perf_test_inputs/graph-ivf-stages-search.json) on each executable.
Use the round orders and warmup rule above. The job runner records inputs alongside
individual build/search results; no data generation, groundtruth, membership check,
or recall evaluation is included in phase timers.

```powershell
cargo build --release -p diskann-benchmark --offline
./layer1.exe run --input-file diskann-benchmark/perf_test_inputs/graph-ivf-stages-build.json --output-file target/layer1-build.json
./layer1.exe run --input-file diskann-benchmark/perf_test_inputs/graph-ivf-stages-search.json --output-file target/layer1-search.json
```

Substitute the saved layer2/layer3 executables and unique output files. Stage reduction
is `100 * (1 - median(after) / median(before))`; use layer 1 to 2 for two-means and
layer 2 to 3 for reassignment. Retain end-to-end time and verify identical work/quality.
Existing 1,024 and 16,384 workloads remain useful sensitivity cases but must not be
mixed into the same before/after comparison.

Validation completed for the timing change:

- Six benchmark tests pass on every stack layer, including phase-sum invariants,
  histogram counts, zero-split builds, and a partial final insertion batch.
- Five two-means tests pass on layer 2; nine clustering/assignment tests pass on
  layer 3 with V4 required. Differential tests retain the independent old selector.
- Full first-layer IVF suite: six pass and the existing NaN-initialization test
  fails. It still expects initialization to reject a NaN centroid; the timing
  change does not alter initialization. No test was removed or weakened to hide it.
- Workspace formatting and Clippy with warnings denied pass. Local LLVM line
  coverage: benchmark module 330/344 (95.93%), clustering 87/91 (95.60%), index
  96/103 (93.20%). These are local file-coverage results, not Codecov patch status.
- Timings are returned through new `InsertStats` fields and are nondeterministic;
  exhaustive struct construction/matching needs updating. No provider trait changes.
