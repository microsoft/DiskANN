# Graph-IVF repeated experiments using arithmetic means

The follow-up adds **24 complete three-version rounds**, with 2,808 build
executions. After the same one-build warmup rule, each version/dimension has
**288 retained samples**. Arithmetic means are primary here; the original
[six-round median report](graph-ivf-stages.md) remains a separate experiment.
Measurements ran on **2026-10-21 22:11–22:23 Asia/Shanghai**; analysis was completed
on 2026-10-22. No implementation changes were made for this repetition study.

## Setup and analysis

The executable hashes were verified against the earlier experiment. Measured
commits are unchanged: layer 1 `f7b706be`, layer 2 `1db048bd`, layer 3 `e4aefd10`.
All use the same [build workload](perf_test_inputs/graph-ivf-stages-build.json):
65,536 vectors, 128/384/768 dimensions, batch size 4,096, split threshold 2,048,
five Lloyd iterations, and the same corpus/build seeds and graph settings.

Windows, Ryzen AI Max+ 395, release x86-64-v3, one worker. Each process was pinned
to logical processor 4 (mask 16), at Normal priority. The recorded metadata confirms
those settings and successful completion for all 72 processes. Runs were sequential;
this task did not compile or run tests concurrently. Other system activity was not
controlled. No power plan, process priority, or background applications were changed.

Before running, the protocol fixed 24 rounds and these six balanced version
orderings, repeated four times: 123, 321, 231, 132, 312, 213. Every invocation
builds each dimension 13 times; discard repetition 0 and retain the remaining 12.
**All remaining samples, including stalls, are included in the mean.** No
post-hoc outlier deletion, trim, or best-run selection is applied.

Time reduction is `100 * (1 - mean(after) / mean(before))`, not the arithmetic
mean of per-run speedup ratios. SD below is sample standard deviation of the
288 retained wall times, not uncertainty of the mean. Approximate 95% intervals
use 20,000 deterministic paired-round bootstrap resamples: resample complete
three-version rounds, retaining the 12-build block within each process. This avoids
treating correlated builds from one process as 288 independent experiments.
It still does not model serial dependence between adjacent rounds, establish
causation for background stalls, or provide a universal performance guarantee.

## Results from the 24 new rounds

Lower time is better; negative reduction means slower. Timing columns are milliseconds.

| Dimensions | Comparison | Before mean ± SD | After mean ± SD | Mean time reduction | Approx. 95% interval |
| ---: | --- | ---: | ---: | ---: | ---: |
| 128 | Two-means (1 → 2) | 32.258 ± 7.922 | 31.095 ± 9.075 | 3.61% | [-3.81, 10.26]% |
| 128 | Reassignment (2 → 3) | 15.393 ± 9.597 | 9.870 ± 5.555 | 35.88% | [24.99, 46.37]% |
| 128 | Total build (1 → 3) | 114.676 ± 27.235 | 103.537 ± 22.528 | 9.71% | [3.45, 15.99]% |
| 384 | Two-means (1 → 2) | 71.297 ± 26.782 | 67.815 ± 19.335 | 4.88% | [-7.82, 16.25]% |
| 384 | Reassignment (2 → 3) | 28.769 ± 7.246 | 26.114 ± 15.774 | 9.23% | [-4.43, 20.65]% |
| 384 | Total build (1 → 3) | 214.961 ± 61.534 | 289.243 ± 1352.869 | -34.56% | [-123.70, 10.18]% |
| 768 | Two-means (1 → 2) | 147.264 ± 76.774 | 153.713 ± 57.080 | -4.38% | [-22.92, 11.32]% |
| 768 | Reassignment (2 → 3) | 45.780 ± 11.792 | 43.367 ± 9.938 | 5.27% | [-3.39, 14.02]% |
| 768 | Total build (1 → 3) | 372.496 ± 164.149 | 361.727 ± 83.816 | 2.89% | [-8.97, 14.41]% |

The clearest repeated result is 128d reassignment: mean time falls from 15.393 ms
to 9.870 ms (35.88%), with 22 of 24 round means improving. Its interval remains
above zero in this resampling model. The larger-dimension reassignment results and
all three two-means comparisons have intervals spanning zero, so the new experiment
does not establish a repeatable gain for those settings. The earlier headline
40–44% reductions should not be treated as general results.

The 128d end-to-end mean improves by 9.71%; 768d is inconclusive. The 384d
end-to-end mean is **34.56% slower**, heavily affected by a long retained stall.
This is reported as measured rather than relabeled as a speedup.

## The large stall remains in the mean

Round 14, layer 3, 384d, repetition 11 took **23.0900262 s**:

- Other IVF: 22.3886159 s.
- Two-means: 0.5114288 s.
- Reassignment: 0.1886753 s.
- Centroid graph: 0.0013058 s.

The same setting's median is 192.879 ms, while its arithmetic mean is 289.243 ms.
The containing process used **9.59375 s CPU** over **34.6380768 s wall time**,
consistent with substantial time not executing on CPU. The measurements do not
identify whether scheduling, paging, suspension, or another wait caused it;
process CPU time includes the whole invocation, not just this build or phase.
The stall is retained in both the main result and combined-session sensitivity.

More repetitions reduced some uncertainty but did not make this desktop host
stable. A dedicated quiet run plus stage CPU counters would distinguish compute
cost from elapsed-time stalls more directly than continuing to average indefinitely.

## All phase means and work checks

| Dimensions | Layer | Two-means | Reassignment | Other IVF | IVF | Centroid graph | Total build |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 1 | 32.258 | 14.213 | 67.598 | 114.070 | 0.606 | 114.676 |
| 128 | 2 | 31.095 | 15.393 | 69.790 | 116.278 | 0.592 | 116.870 |
| 128 | 3 | 28.688 | 9.870 | 64.427 | 102.985 | 0.552 | 103.537 |
| 384 | 1 | 71.297 | 29.343 | 113.604 | 214.244 | 0.717 | 214.961 |
| 384 | 2 | 67.815 | 28.769 | 113.197 | 209.780 | 0.757 | 210.537 |
| 384 | 3 | 71.373 | 26.114 | 190.949 | 288.437 | 0.807 | 289.243 |
| 768 | 1 | 147.264 | 46.331 | 177.845 | 371.441 | 1.055 | 372.496 |
| 768 | 2 | 153.713 | 45.780 | 175.742 | 375.236 | 1.095 | 376.331 |
| 768 | 3 | 146.742 | 43.367 | 170.592 | 360.701 | 1.026 | 361.727 |

Unlike medians, the three phase means sum to mean IVF time apart from rounding.
Total build includes a small gap between IVF and graph timing boundaries.
Every one of the 2,808 new builds, and all 3,510 builds across both sessions, has
identical expected work for its dimension: inserted points, split parents,
split-point visits, split-parent histogram, list count, empty lists, and maximum
list size. All phase sum checks pass. Each build also runs the existing complete
membership validation outside the timer.

No new search passes were run: binaries, inputs, and algorithms are unchanged.
The earlier 324 search passes established matching recall and logical scan work
across these versions; those are earlier validations, not new measurements.
This in-memory experiment does not measure physical SSD I/O.

## Session sensitivity

The previous six rounds have 72 samples/version/dimension; the new 24 have 288.
Combined, they have 360 samples/version/dimension across 30 rounds. Keep these
sessions separate when interpreting drift; adding old noisy samples is not an
independent replication. Every column below uses **arithmetic means**, including
the recomputed old session; it therefore differs from the old median table.

| Dimensions | Comparison | Earlier 6 rounds | New 24 rounds | Combined 30 rounds |
| ---: | --- | ---: | ---: | ---: |
| 128 | Two-means (1 → 2) | 14.79% | 3.61% | 6.85% |
| 128 | Reassignment (2 → 3) | 29.31% | 35.88% | 33.88% |
| 128 | Total build (1 → 3) | 9.30% | 9.71% | 9.59% |
| 384 | Two-means (1 → 2) | 8.98% | 4.88% | 6.09% |
| 384 | Reassignment (2 → 3) | 22.00% | 9.23% | 13.31% |
| 384 | Total build (1 → 3) | 7.51% | -34.56% | -22.23% |
| 768 | Two-means (1 → 2) | 17.27% | -4.38% | 2.39% |
| 768 | Reassignment (2 → 3) | 3.05% | 5.27% | 4.59% |
| 768 | Total build (1 → 3) | 4.67% | 2.89% | 3.44% |

The new session is primary. The combined sensitivity does not supersede it or
justify selecting whichever session yields a stronger improvement. Complete
phase distributions (mean/SD/median/p95/min/max), per-round means, bootstrap
intervals, executable/input hashes, and process metadata are preserved in the
[machine-readable evidence](graph-ivf-stages-means.json).

## Reproduction

Use the three saved release executables at the pinned commits, the shared input,
24 predeclared balanced rounds, and one discarded warmup build per invocation and
dimension. Store each output separately; do not overwrite the previous six rounds.
Keep batch size, seeds, iterations, CPU affinity, and executable hashes fixed.
Compute means from every retained `builds[]` row. For the two-means comparison use
layers 1 and 2; for reassignment use layers 2 and 3; for the full stack use 1 and 3.

Local raw JSON/logs and the runner are under `target/stage-mean-comparison/`;
previous raw outputs remain under `target/stage-comparison/`. Per-round statistics
in the checked-in evidence reproduce the mean ratios and paired-round intervals.
The original exact sample distributions remain in the local raw JSON files.
