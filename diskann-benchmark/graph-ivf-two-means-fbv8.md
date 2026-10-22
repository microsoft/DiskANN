# FBV8: complete two-means and Graph-IVF build comparison

The multi-point kernels reduce fitting time in this experiment, but change the
resulting index. The batched_4 quality run loses 0.856 recall percentage points
at nprobe=4 and scans 1.11%-2.36% more vectors across the three settings. It does
not meet the requested unchanged recall/scan contract. **The default remains
paired L2.** Packing alone
does not establish an end-to-end gain; SGEMM is slower in the fitting phase.
These results extend the [strategy implementation](graph-ivf-two-means-experiments.md)
and the earlier [24-round synthetic study](https://github.com/microsoft/DiskANN/blob/3e906e3a3990ae4ac1d0bed0308a6fd026fe4f10/diskann-benchmark/graph-ivf-stages-means.md).

## Corpus and protocol

FBV8 contains **958,152 base vectors and 6,380 queries, 896 dimensions**. Both
vector files have an eight-byte `(rows, dimensions)` header followed by IEEE
FP16 values; the benchmark promotes them exactly to FP32 before timing and
does not renormalize. Groundtruth contains 2,000 uint32 IDs per query followed
by distances; recall uses the first 10 IDs.

Build settings: 16 sampled initial centroids, insertion batches of 4,096, split
threshold 2,048, five Lloyd steps, seed 7, final centroid graph degree 16 and
build L=64. The benchmark uses exhaustive centroid routing during ingestion
and a real DiskANN centroid graph for search. It is a static, in-memory adapter.

Windows, Ryzen AI Max+ 395, Rust 1.97.1, release x86-64-v3, one worker, affinity
mask 16. Measurements used one executable with SHA256
`7c75ccb211c013387d19679626b3843f7b3b32bcbcd473b680b0012002545130`.
No compilation or other intentionally launched heavy work overlapped these
measurements; other desktop activity was uncontrolled.

The fixed protocol ran **seven complete rotations of seven strategies: 49 full
builds**, from 2026-10-22 08:17 to 09:00 Asia/Shanghai. Each strategy occupied
each ordinal position once. Every build is retained, including stalls. There
is no discarded per-process warmup. Pilot observations are stored separately.

Arithmetic means are primary. SD is sample standard deviation. Reductions are
`100 * (1 - candidate_mean / paired_mean)`. Approximate intervals use 20,000
seeded paired-round bootstrap samples, resampling complete seven-strategy
rounds. Seven rounds do not establish a hardware-independent result, and this
resampling does not model temporal dependence or uncontrolled system activity.

## Full builds

Times are seconds. Negative reduction means slower than paired L2.

| Strategy | Two-means mean +/- SD | Fitting reduction | Total build mean +/- SD | Total reduction | Index fingerprint matches paired |
| --- | ---: | ---: | ---: | ---: | --- |
| separate | 5.397 +/- 0.640 | 5.50% | 52.170 +/- 4.010 | -2.45% | Yes |
| paired_point | 5.711 +/- 1.409 | reference | 50.922 +/- 4.202 | reference | Yes |
| packed_paired | 5.152 +/- 0.243 | 9.79% | 49.853 +/- 1.909 | 2.10% | Yes |
| batched_4 | 3.663 +/- 0.350 | 35.86% | 47.319 +/- 2.156 | 7.08% | No |
| batched_8 | 3.595 +/- 0.712 | 37.05% | 46.816 +/- 2.337 | 8.06% | No |
| sgemm | 7.006 +/- 1.271 | -22.67% | 49.209 +/- 9.061 | 3.36% | No |
| center_difference | 4.742 +/- 0.321 | 16.96% | 49.701 +/- 3.114 | 2.40% | No |

| Strategy | Fitting reduction: approximate 95% interval | Total reduction: approximate 95% interval | Rounds with faster total build |
| --- | --- | --- | ---: |
| separate | [-11.38%, 23.92%] | [-9.53%, 5.39%] | 3/7 |
| packed_paired | [-2.22%, 25.70%] | [-1.93%, 7.80%] | 3/7 |
| batched_4 | [25.89%, 47.91%] | [3.24%, 12.66%] | 7/7 |
| batched_8 | [21.77%, 50.36%] | [3.68%, 12.87%] | 6/7 |
| sgemm | [-44.34%, -3.53%] | [-5.36%, 14.75%] | 3/7 |
| center_difference | [7.29%, 27.46%] | [-0.71%, 5.76%] | 5/7 |

The batched strategies improve measured fitting time in all seven rounds. Their
new reduction tree also changes centers and membership, so the full-build
comparison includes the resulting change in later routing and splitting work.
It cannot be described as the same index built faster.

Paired fitting is 5.82% slower in mean than separate in this sample; its total
build is 2.39% faster. Neither comparison establishes a repeatable gain.
Packed paired preserves the output but its intervals span zero for both fitting
and total build. No additional exact strategy is enabled by default.

## Work and timing limits

Every completed build inserted all 958,152 points and passed exact single-list
coverage validation. Output statistics are deterministic across the seven
repetitions of each strategy.

| Strategy family | Live lists | Splits | Split point visits | Fingerprint |
| --- | ---: | ---: | ---: | --- |
| separate / paired / packed paired | 758 | 742 | 1,543,226 | `30524780b073489f` |
| batched_4 / batched_8 | 757 | 741 | 1,541,140 | `a8f5609bbfb98ed5` |
| sgemm | 754 | 738 | 1,534,911 | `19f48f513d39ebce` |
| center_difference | 762 | 746 | 1,551,429 | `9110e8eaabe74d67` |

The fingerprint hashes ordered list IDs, center bits and ordered memberships
with `DefaultHasher`; it is a same-executable diagnostic rather than a
cryptographic equality proof. Batched_4 and batched_8 produce the same
fingerprint and work statistics here.

Paired mean phase breakdown: two-means **5.711 s**, reassignment **1.979 s**,
other IVF **43.182 s**, centroid graph **0.050 s**. Only about 11.2% of total
build time is fitting. The residual includes exhaustive routing, provider
preparation and publication, working-set fill and orchestration; it has not
been separately profiled as pure routing time.

Process CPU/wall means are 26.96/54.44 s for paired and 25.77/50.97 s for
batched_4. These include loading, validation and teardown, not just build.
Round 2 paired fitting took 8.831 s, and the last SGEMM build took only 28.880 s
while other SGEMM builds took 50.953-54.857 s. All samples remain included.
The latter process used 27.53 s CPU over 32.79 s wall, unlike the roughly
half-CPU/wall ratios in most earlier runs. The measurements do not identify
the cause; continued averaging cannot substitute for a quiet dedicated host.

## Complete-fit matrix

The standalone matrix uses random samples of the corpus, not actual captured
split parents. It covers 32/128/512/1,024/4,096 points, 1/5/10 Lloyd steps and
contiguous/scattered allocation, with 21 retained samples after two warmup
rounds per setting. Packing and allocations are timed. All observed final
centers and diagnostic assignments match paired on these 30 workloads. The
full builds above demonstrate why that observation is insufficient evidence
of general numerical equivalence.

Selected five-step means in milliseconds:

| Layout / rows | paired | packed paired | batched_4 | batched_8 | SGEMM | center difference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| contiguous / 1,024 | 0.639 | 1.365 | 0.552 | 0.584 | 2.238 | 0.578 |
| contiguous / 4,096 | 3.126 | 6.310 | 2.501 | 2.449 | 9.242 | 2.564 |
| scattered / 1,024 | 0.708 | 1.371 | 0.632 | 0.680 | 2.109 | 0.648 |
| scattered / 4,096 | 4.640 | 5.991 | 4.682 | 4.730 | 9.106 | 4.629 |

Packing can lose badly at five steps and still help another workload: for
4,096 scattered rows with ten steps, paired/packed means were 14.368/9.698 ms.
This is a workload-dependent observation, not a dispatch threshold. The matrix
and pilots used executable SHA256
`0378c06ab77f7d7d3341b64088b71f11c53cb062062194118be4c77eafb9eca2`; the later
binary removes unused layout-index bookkeeping and corrects documentation,
with unchanged numerical kernels. Keep these experiments separately identified.

## Search and decision

Both paired and batched_4 process all 6,380 queries at k=10, L=128. The paired
baseline uses one warmup and one measured pass; the batched_4 quality run uses
one measured pass without a warmup. This compares deterministic quality/work
only, not search latency. Both use the same executable and index/search settings.

| nprobe | Paired recall@10 | Batched_4 recall@10 | Recall change, percentage points | Paired vectors/query | Batched_4 vectors/query | Scan change |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 23.9154% | 23.0596% | -0.8558 | 4,989.689 | 5,045.189 | +1.1123% |
| 16 | 45.8981% | 46.0016% | +0.1034 | 20,815.760 | 21,305.974 | +2.3550% |
| 64 | 71.3856% | 72.0408% | +0.6552 | 85,293.169 | 86,746.442 | +1.7039% |

Both return 10 results and scan exactly 4/16/64 lists per query. Mean centroid
comparisons rise from 469.212 to 472.837 at each setting; ordered result-ID
fingerprints differ at all three settings. A similar list count therefore does
not establish unchanged scan work or recall. The quality run reproduces the
batched_4 index fingerprint from all seven build repetitions.

Batched_8 has the same index fingerprint and build-work statistics as batched_4
in every round. Its search quality was not independently measured; hash agreement
is not an exhaustive equality proof. SGEMM and center-difference search quality
was also not measured. No all-strategy search non-regression claim is made.

**Do not enable the current batched kernel by default under an unchanged-output
or unchanged recall/scan requirement.** It is a promising computation strategy,
but the present arithmetic changes the index and fails that acceptance condition.
For behavior preservation, the next candidate should retain the existing
per-distance reduction tree while sharing center loads, or vectorize f64 sums
across coordinates while retaining each coordinate's point order. Re-run the
complete-fit and full-build/search checks before adoption.

SGEMM does not justify adoption for this two-column workload. Center difference
also remains experimental: nonfinite fallback does not cover finite cancellation.
Packed paired preserves the observed output but has no established total-build
gain. No candidate from this study is promoted over the default paired path.

## Validation

The focused two-means suite passes all 13 tests and the feature-enabled benchmark
suite passes all 10 tests; the default benchmark suite passes seven. Four paired
kernel tests execute Scalar, V3, V4 and the public entry point, with V4 required
on this host. Baseline x86-64 tests pass on the same CPU; AArch64 Windows
library/tests cross-check succeeds. Neither Neon execution nor old-x86
instruction emulation was available.
Formatting and workspace Clippy with warnings denied pass with default features
and with `diskann-benchmark/ivf-two-means` enabled.

The full IVF suite reports 19 passing tests and the existing
`invalid_initialization_and_uninitialized_insertion_fail` failure: initialization
accepts a NaN centroid. This failure was reproduced on the baseline; the test
and initialization behavior are unchanged. Coverage collection explicitly
skips that known failure.

Local LLVM line coverage for the reviewed sources is 100% for clustering,
97.31% for experimental kernels, 95.75% for the complete-fit matrix, and 93.18%
over the selected IVF/benchmark files. LLVM reports three functions with
mismatched profile data when combining the library and benchmark objects, so
these are diagnostic local totals, not a clean Codecov patch-status result.
The Windows runner's synthetic smoke test and refusal to overwrite existing
artifacts both pass.

Raw phase samples, pilot results, matrix samples, process CPU/wall times,
fingerprints and input hashes are retained in the
[machine-readable evidence](graph-ivf-two-means-fbv8.json). The
[implementation and reproduction instructions](graph-ivf-two-means-experiments.md)
describe the feature and workloads. Logical scan counts do not measure physical
SSD I/O, Mimir persistence or write amplification.
