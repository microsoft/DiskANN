# FBV8 Mimir: improving split reassignment locality

The follow-up to the existing third stack layer reduces reassignment time on
the complete 958,152-vector, 896-dimensional FBV8 corpus. At batch size 16,384,
the arithmetic mean falls from **4.175 s to 2.949 s (29.36%)**. Total build
falls from **49.539 s to 47.351 s (4.42%)**. The smaller batch of 4,096 has no
established gain in this experiment. Search recall and logical scan work match.

## What changed

The baseline is `a57222223d842b9de1f4b52ab5169c0b6d45e258`, including the existing
paired-distance and SIMD top-1 optimizations. Both measured binaries also include
the same untimed layout fingerprint. The candidate adds point tiling in
`diskann/src/ivf/clustering/assignment.rs`; two-means and build routing are unchanged.

Previously each point traversed every child centroid, wrote a distance row, then
ranked that row. For larger child matrices the new path holds four borrowed point
references, streams each centroid across those points, and updates four running
top-1 selections. The paired L2 kernel shares centroid loads across two points.
Each point still compares against **every child**, in original centroid order.
This reduces repeated centroid traffic and distance-row traffic without copying
point coordinates or changing the distance accumulation tree.

The existing path remains for fewer than 16 children or less than 32 KiB of
centroid coordinates. Four-point tiles use bounded stack scratch and preserve
member order, first-on-ties, and overflow behavior. Reversing L2 operands is exact
for finite inputs; NaN distances are recomputed in original operand order to
preserve total-order selection. No public API or dependency is added.

This is an optimization of the current exhaustive **k=1 owner assignment** after
splitting. It does not implement replication, configurable top-k, neighboring-list
reassignment, or graph-based build routing.

## Workload and measurement

- Windows, Ryzen AI Max+ 395, Rust 1.97.1, repository release x86-64-v3 flags.
- One benchmark thread, sequential processes, logical CPU 4 (affinity mask 16),
  Normal priority. No power-plan or background-application changes.
- All 958,152 base vectors, original row order, 896 dimensions; the source is
  FP16 and is exactly promoted to FP32. No new normalization or corpus sampling.
- 16 initial centroids, threshold 2,048, five Lloyd iterations, build seed 7,
  centroid graph degree 16 / L=64. Batch sizes are 4,096 and 16,384.
- Eight predeclared alternating AB/BA rounds; three builds per process and batch.
  Repetition 0 is warmup; repetitions 1 and 2 are retained. This gives **16 samples
  per version/batch**, 96 complete builds including warmup.
- Arithmetic means include every post-warmup sample. Reduction is
  `100 * (1 - mean(after) / mean(before))`. No outliers are removed.
- Approximate 95% intervals use 20,000 deterministic bootstrap resamples of whole
  paired process rounds, keeping the two retained builds together. Only eight
  round pairs are available; serial dependence and uncontrolled desktop activity
  limit the inference.

The formal build runs were completed on 2026-10-22, approximately 07:40–09:00
Asia/Shanghai. Data preparation, groundtruth, layout fingerprints and membership
validation are outside timers. No compilation or other benchmark from this task
ran concurrently with these measurements.

| Batch | Phase | Before mean ± SD (s) | After mean ± SD (s) | Time reduction | Approx. 95% interval |
| ---: | --- | ---: | ---: | ---: | ---: |
| 4,096 | Reassignment | 1.850 ± 0.383 | 1.797 ± 0.444 | 2.86% | [-10.32, 18.33]% |
| 4,096 | Total build | 48.347 ± 9.528 | 46.078 ± 10.364 | 4.69% | [-4.93, 16.40]% |
| 16,384 | Reassignment | 4.175 ± 0.736 | 2.949 ± 0.590 | 29.36% | [26.68, 32.23]% |
| 16,384 | Total build | 49.539 ± 8.910 | 47.351 ± 9.351 | 4.42% | [0.48, 8.38]% |

All eight large-batch round means improve in reassignment; six improve in total
build. At batch 4,096 only three reassignment round means improve. Its split
histogram contains at most eight parents, and only one insertion reaches eight:
almost all calls keep the old small-matrix path. A stable small-batch speedup
should therefore not be inferred from the pooled mean.

| Batch | Two-means before / after (s) | Other IVF before / after (s) |
| ---: | ---: | ---: |
| 4,096 | 5.103 / 4.929 | 41.335 / 39.301 |
| 16,384 | 5.861 / 5.517 | 39.452 / 38.836 |

These unchanged phases also vary, so the entire total-build difference cannot be
attributed to the reassignment kernel. The direct large-batch phase saving is
1.226 s, about 2.47% of baseline total build; the observed total saving is 2.188 s.
The remaining difference is not claimed as an additional algorithmic optimization.

## Correctness and query quality

All 96 builds agree in inserted points, split parents, split-point visits, split
histogram, live lists, maximum list size, and layout fingerprints. Every build
also validates full membership. The fingerprint includes list IDs, exact centroid
bits and ordered members; it is a same-toolchain regression signal, not a
cryptographic proof or persistent format. Differential unit tests compare actual
centroid bits and membership arrays against the original selector.

| Batch | Lists | Splits | Split-point visits | Layout fingerprint |
| ---: | ---: | ---: | ---: | --- |
| 4,096 | 758 | 742 | 1,543,226 | `f522b801485dedf2` |
| 16,384 | 760 | 744 | 1,711,508 | `6edf7d5a58056288` |

Quality checks use 256 evenly spaced rows from the 6,380 provided queries, fixed
before the comparison. Fresh exhaustive squared-L2 top-10 groundtruth uses the
**complete** promoted base corpus. Eight selected query answers were independently
checked against exhaustive FP64 distances and have identical top-10 IDs.

Each version runs one pass per batch/nprobe with L=128. All six corresponding
passes have equal recall, returned count, vector/list scans and centroid
comparisons. This is a query-subset quality check, not all 6,380 queries and not
a repeated, warmed search-throughput experiment.

| Batch | nprobe | Recall@10, both | Vectors scanned/query, both |
| ---: | ---: | ---: | ---: |
| 4,096 | 4 | 0.266796875 | 4,949.40625 |
| 4,096 | 16 | 0.482812500 | 20,593.36328 |
| 4,096 | 64 | 0.726171875 | 85,011.41797 |
| 16,384 | 4 | 0.276171875 | 5,484.51563 |
| 16,384 | 16 | 0.464062500 | 22,060.20313 |
| 16,384 | 64 | 0.728906250 | 87,858.55078 |

This provider is in memory. Equal logical scans do not measure physical SSD I/O,
bytes, caches, or Mimir write amplification. No search QPS improvement is claimed.

## Other options investigated

An eight-centroid transposed SIMD prototype with exact fallback did not improve
the full-corpus reassignment pilot; it is not in the patch. Simply streaming
paired distances into scalar top-1 also failed to reduce local kernel CPU cycles.
Small point tiles provided a simpler improvement while preserving the existing
floating-point reduction.

A blocked GEMM prototype was faster for larger candidate sets in a local kernel
experiment, but `||c||² - 2 x·c` changes rounding and needs a defensible uncertainty
bound plus exact reranking near ties. It also adds packing and an algorithm-layer
linear-algebra dependency. It remains a possible follow-up, not a measured
production gain.

Further build gains should focus on the remaining work: exhaustive initial
route selection is exhaustive and other IVF work dominates total time; these
measurements do not separately isolate routing from provider work. Two-means
still rereads each parent for five iterations. Batch routing, reuse of parent
data in cache, and vectorized f64
centroid accumulation are separate opportunities. They must preserve the same
candidate/iteration work or explicitly measure any resulting quality tradeoff.

## Validation

Ten clustering/assignment tests pass in test and release profiles, baseline
`x86-64`, and without default features. Seven static benchmark tests pass. The
new differential covers scattered borrowed rows, list and tile tails, odd child
counts, dimensions through 897, ties, extreme finite values, infinity and NaN.
AArch64 Windows library/tests cross-compilation passes; Neon execution and
old-x86 emulation were not available locally.

The complete IVF suite reports 16 passed, one manual kernel benchmark ignored,
and the inherited NaN-centroid-initialization failure. It is reproduced using
the pre-change test binary and is outside this reassignment change. Workspace
formatting and Clippy with warnings denied pass. Local instrumented line
coverage is 106/108 (98.15%) for assignment and 93/93 for clustering; this is not
a Codecov patch-status result.

An independent Python verifier recomputes all build means/SD, work equality and
phase partitions from raw JSON, then compares all six search-quality records.
The published statistics are not selected from the fastest pilot or best round.

## Reproduction and evidence

[Complete evidence](graph-ivf-fbv8-results.json) includes all build/search outputs,
per-round means, SD, intervals, process metadata and executable/input/data hashes.
The local raw runs and runner are under `target/split-reassignment-optimization/`.
The committed preparation script was rerun over all source rows; its generated
base and selected-query hashes match the actual measured files.

```powershell
python diskann-benchmark/scripts/prepare_graph_ivf_fbv8.py --source C:/Users/xiajiang/src/big-ann-benchmarks/data/fbv8_mimir --output target/fbv8-comparison/data --query-count 256
./target/release/compute_groundtruth.exe --data-type float --dist-fn l2 --base-file target/fbv8-comparison/data/fbv8-base-f32.bin --query-file target/fbv8-comparison/data/fbv8-queries-f32.bin --gt-file target/fbv8-comparison/data/fbv8-groundtruth-k10.bin --recall-at 10
./before.exe run --input-file diskann-benchmark/perf_test_inputs/graph-ivf-fbv8-build.json --output-file target/before-build.json
./after.exe run --input-file diskann-benchmark/perf_test_inputs/graph-ivf-fbv8-build.json --output-file target/after-build.json
./after.exe run --input-file diskann-benchmark/perf_test_inputs/graph-ivf-fbv8-search.json --output-file target/after-search.json
```

Compile baseline and candidate with identical flags. Use distinct output names
for every process, alternate AB/BA for eight rounds, and discard only the first
build per process/batch. Run the search input on both versions. The earlier
Enron and synthetic reports remain separate datasets and experiments.
