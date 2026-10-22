# Graph-IVF two-means: YFCC, Enron and FBV8 ABBA protocol

This run uses the refactored first-layer benchmark at
`1c02ec23cc878d3e51f87b4c226792f4b5364881`, with the existing FP16 loader,
seven strategy selector and untimed fingerprints adapted to its new modules.
The benchmark now uses the shared recall implementation. Numerical kernels,
ordered f64 sums, seeds, fixed Lloyd steps and final reassignment are unchanged.

## Fixed comparison

| Dataset | Base vectors | Dimensions | Queries in every search pass | Source type |
| --- | ---: | ---: | ---: | --- |
| YFCC | 10,000 | 192 | 100 | FP32 |
| Enron | 1,087,932 | 384 | 1,000 | FP16 |
| fbv8_mimir | 958,152 | 896 | 6,380 | FP16 |

FP16 is promoted exactly to FP32 outside timing, without normalization. The
supplied query and groundtruth files are used in full. No query subsampling is
used in any ABBA pass.

For each dataset, compare A = paired_point to each B in separate, packed_paired,
batched_4, batched_8, sgemm and center_difference. Run four A-B-B-A blocks for
each pair. Each invocation builds the complete index once, then runs all queries
at nprobe 4/16/64, with one full warmup and one measured pass per setting.
This yields 288 invocations: eight A and eight B measurements for each of the
18 dataset/candidate comparisons. Each candidate has its own surrounding A
samples; do not substitute A samples from a different time block.

Fixed settings: batch size 4,096, 16 sampled starting centroids, split threshold
2,048, five Lloyd steps, seed 7, centroid graph degree 16/build L64, k10/search
L128, one thread, Windows affinity 0x10, Normal priority, release x86-64-v3.
Candidate order rotates between rounds; the four positions in every block
remain A-B-B-A. All seven strategies use one frozen executable.

Report arithmetic means, SD, complete block values and paired block reductions
for complete two-means, total build and each search setting. Four blocks are
few independent observations; individual queries or inner repetitions are not
independent timing trials. Keep all samples, including stalls. Partition,
recall, returned-count and vector/list/centroid work changes accompany timing
comparisons; no default kernel change follows from speed alone.

## Reproduce or resume

Set the three input templates' search_directories to the actual corpus paths.
Use the YFCC perf workload, graph-ivf-enron-fp16.json, and the FBV8 search input.
Build once with the ivf-two-means feature, then use the same saved executable
throughout the comparison.

```powershell
python diskann-benchmark/scripts/run_ivf_abba.py --exe target/ivf-abba-20261022/diskann-benchmark.exe --output-directory target/ivf-abba-20261022/runs --dataset yfcc=target/ivf-abba-20261022/yfcc-template.json --dataset enron=target/ivf-abba-20261022/enron-template.json --dataset fbv8_mimir=target/ivf-abba-20261022/fbv8_mimir-template.json --rounds 4
```

The runner writes the immutable protocol and one input/result/log/metadata set
per invocation. Re-running the same command resumes only after verifying every
completed result and its executable/input hashes. Failed or incomplete runs
require inspection, and are never silently overwritten or dropped. Each result
must contain all base vectors, all queries, all requested search settings, and
k returned results. The smoke test exercises all seven strategies, a clean stop,
and resumption.

## Measurement environment

The initial YFCC round and first Enron A overlapped an independently launched
cargo/benchmark job. All those samples remain retained and are identified in
the evidence. Subsequent invocations wait until no cargo, rustc or other
diskann-benchmark process is present, and sample competing processes every
30 seconds. This cannot detect every short overlap, and does not control other
applications, memory pressure or CPU clocks. Absence of recorded overlap is
not proof of an isolated machine.

The provider remains in memory. Logical scans do not measure physical SSD I/O,
Mimir persistence, cache misses or write amplification.

## Analyze

```powershell
python diskann-benchmark/scripts/analyze_ivf_abba.py target/ivf-abba-20261022/runs --output target/ivf-abba-20261022/summary.json
```

Without `--allow-partial`, analysis fails until all 288 invocations are complete.
Partial summaries count only completed ABBA blocks while retaining individual
completed runs; they must not be reported as the final experiment. The analyzer
checks exact strategies against their paired outputs and preserves approximate
strategies' actual quality/work changes. Bootstrap intervals resample whole
blocks; four blocks give only 256 distinct ordered draws, so intervals remain
descriptive and do not establish universal speedups.
