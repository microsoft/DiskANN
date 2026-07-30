# Issue #939 K-means baseline

## Experiment contract

- Revision: `59dd0481` (`origin/main` when the baseline worktree was created)
- Worktree: `C:\src\DiskANN-issue-939-baseline`
- Branch: `wuw92/issue-939-benchmark-baseline`
- Dataset: deterministic synthetic `f32` values sampled uniformly from `[-1, 1)`, seed 42
- Points: 50,000
- Centers: 256
- Lloyd iterations: 10
- Measurements: 30 per workload
- Build: release, `diskann-benchmark` with `kmeans-comparison`

Each measurement calls K-means++ once and, for `phase: "all"`, Lloyd once using the selected centers. The benchmark records initialization, Lloyd, and total wall-clock latency from that single execution. Fixture generation and quality calculation are outside the timed region.

The Disk path calls `k_meanspp_selecting_pivots` followed by `run_lloyds`, passing the explicit repository Rayon pool to both functions. The old quantization path calls `kmeans_plusplus_into` followed by `lloyds`; these baseline APIs are sequential, so changing the benchmark thread count does not change their execution.

## Dimension sweep

Thread count is 1. Latencies are medians in milliseconds.

| Dimension | Disk init | Quant init | Init change | Disk Lloyd | Quant Lloyd | Lloyd change | Disk total | Quant total | Total change |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 79.132 | 12.890 | -83.7% | 154.482 | 26.176 | -83.1% | 233.885 | 39.269 | -83.2% |
| 32 | 93.925 | 36.040 | -61.6% | 238.536 | 147.149 | -38.3% | 332.902 | 183.473 | -44.9% |
| 128 | 202.912 | 173.231 | -14.6% | 553.041 | 663.220 | +19.9% | 760.494 | 836.838 | +10.0% |
| 384 | 887.169 | 767.457 | -13.5% | 1,373.521 | 2,045.529 | +48.9% | 2,264.664 | 2,812.458 | +24.2% |
| 768 | 1,771.880 | 1,688.607 | -4.7% | 2,635.229 | 4,117.453 | +56.2% | 4,404.248 | 5,799.877 | +31.7% |
| 1,024 | 2,341.492 | 2,330.105 | -0.5% | 3,305.319 | 5,515.243 | +66.9% | 5,645.712 | 7,842.756 | +38.9% |
| 3,072 | 6,969.815 | 7,293.668 | +4.6% | 6,726.029 | 16,666.082 | +147.8% | 13,696.581 | 23,970.544 | +75.0% |

**[measured]** The baseline reproduces the high-dimensional gap from issue #939. Quantization crosses from faster to slower at dimension 128 and reaches a 75.0% total-latency regression at dimension 3,072. The gap is concentrated in Lloyd, which is 147.8% slower at dimension 3,072.

## Thread sweep

Dimension is 4. Latencies are medians in milliseconds.

| Threads | Disk init | Quant init | Disk Lloyd | Quant Lloyd | Disk total | Quant total |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 79.132 | 12.890 | 154.482 | 26.176 | 233.885 | 39.269 |
| 2 | 56.398 | 13.002 | 128.981 | 27.000 | 185.514 | 40.026 |
| 4 | 54.880 | 13.162 | 128.571 | 26.916 | 185.477 | 40.432 |
| 8 | 76.757 | 13.419 | 175.219 | 26.468 | 252.584 | 39.963 |

**[measured]** Old quantization total latency remains within 3.0% across 1, 2, 4, and 8 configured threads, confirming that this implementation does not use the benchmark's Rayon pool. Disk improves through four threads on this low-dimensional workload and regresses at eight threads.

## Quality observations

The final center hash and objective match exactly through dimension 384 and for every dimension-4 thread workload. They differ at dimensions 768, 1,024, and 3,072, so performance comparisons at those dimensions should not be interpreted as bit-identical algorithm executions.

## Raw results

- `disk-baseline.json`
- `quantization-baseline.json`

The benchmark also accepts `phase: "init"` to execute and report initialization only; `issue-939-kmeans-init.json` provides a ready-to-run configuration.
