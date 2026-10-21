# Enron 384D FP32: split reassignment comparison

This full-corpus follow-up to [the synthetic measurements](graph-ivf-reassignment.md)
compares stack layer 2 at `0735fc3db269986613da9693c3ec269034ebd9b9`
with layer 3 at `dfee4c9f8e241f14ebc13234742f1727753dba64` on 2026-10-21.
The saved release binaries precede the later split-phase telemetry changes;
these results do not measure those changes or isolate time inside reassignment.

## Dataset and precision

The local Enron embedding corpus has **1,087,932 base vectors**, **1,000 supplied
queries**, and **384 dimensions**. Both versions store and compute with FP32.
The available source files are FP16:

- `normalized_dim_384_vector_fp16_1087932_vectors.bin`
- `query_vector_normalized_dim_384_fp16_top_1000.bin`

Every coordinate was promoted exactly to FP32. All rows and their ordering were
preserved, with no sampling, normalization, or new embedding generation. This
exercises FP32 kernels on the existing FP16-valued embeddings; it cannot recover
precision lost when the source was quantized. The embedding model's provenance
was not established from these files.

The complete conversion was read back and checked bit for bit against FP16-to-FP32
promotion. Squared norms are approximately 0.99983–1.00018 for base vectors.
The metric remains squared L2; it is not changed to cosine merely because the
file names say normalized.

Fresh exhaustive squared-L2 top-10 groundtruth was computed on all FP32 base
vectors for all 1,000 queries using `diskann-tools/compute_groundtruth`. The
packaged groundtruth was not reused. For eight fixed query IDs (0, 143, 285,
428, 571, 714, 857, 999), an independent exhaustive FP64 calculation produced
identical top-10 IDs; maximum distance error was below 7.62e-8.

## Build results

Windows, AMD Ryzen AI Max+ 395, Rust 1.97.1, repository x86-64-v3 release flags,
one benchmark thread pinned to logical CPU 4 (affinity mask 16). Before/after
benchmark processes ran sequentially. Settings were fixed before the comparison:
16 initial centroids, split threshold 2,048, five Lloyd iterations, build seed 7,
and centroid graph degree 16 / L=64. Only insertion batch size differs between
the two workloads. Neither corpus size, search work, nor iteration count was
reduced to obtain a speedup.

Four alternating AB/BA rounds, with five builds per version/batch in each round,
produce 20 builds per version/batch. Dropping the first build of **each** process
and batch leaves 16 samples. Pooled medians include the final centroid graph;
a negative reduction means a slowdown.

| Batch size | Before (s) | After (s) | Reduction | Lists, both | Splits, both |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 15.233 | 15.516 | -1.86% | 782 | 766 |
| 16,384 | 18.033 | 16.158 | 10.40% | 787 | 771 |

Batch 16,384 shows a 10.40% lower median in this run. Batch 1,024 is 1.86% slower.
Desktop timing is variable, so these observations do **not** establish a universal
build improvement or non-regression. The selected build seed and original input
order also limit generalization. Per-round medians, before / after in seconds:

| Batch size | 0 (AB) | 1 (BA) | 2 (AB) | 3 (BA) |
| ---: | ---: | ---: | ---: | ---: |
| 1,024 | 13.546 / 13.619 | 17.388 / 14.332 | 15.557 / 16.066 | 15.740 / 24.615 |
| 16,384 | 17.246 / 15.231 | 18.354 / 15.299 | 17.180 / 17.177 | 20.290 / 17.243 |

All corresponding build statistics match: every build inserts all 1,087,932 points,
with no empty lists and maximum list size 2,048. Comparisons are within each batch
configuration; the different layouts are not a quality-equivalent batch-size study.

## Search quality and work

Each version completed one search pass over **all 1,000 queries** at each of
nprobe=4/16/64, after one fresh build per batch configuration. Search k=10,
L=128, and warmup passes=0. This gives six corresponding passes, or 6,000 query
executions per version. The requested nprobe, returned-result count, list scans,
vector scans, centroid comparisons, and recall are exactly equal in every
corresponding result.

| Batch size | nprobe | Recall@10, both | Vectors scanned/query, both | Lists scanned/query, both |
| ---: | ---: | ---: | ---: | ---: |
| 1,024 | 4 | 0.3575 | 5972.007 | 4 |
| 1,024 | 16 | 0.5712 | 23883.467 | 16 |
| 1,024 | 64 | 0.7843 | 94864.022 | 64 |
| 16,384 | 4 | 0.3660 | 5889.010 | 4 |
| 16,384 | 16 | 0.5737 | 23696.902 | 16 |
| 16,384 | 64 | 0.7870 | 94265.834 | 64 |

Every query returns 10 results. Mean centroid comparisons are 480.78 for batch
1,024 and 478.66 for batch 16,384. These numbers measure ANN agreement with exact
vector neighbors, not human judgments about email relevance.

**Search timing is unusable for a throughput comparison.** For example, one
1,000-query nprobe=16 pass took 3,777 seconds while its median per-query latency
was about 42 ms; the optimized large-batch nprobe=64 pass also had a 2,013-second
elapsed-time outlier. The cause of the pauses was not established. Timing values
are retained in the evidence, but no QPS speedup or latency non-regression is
claimed. The completed passes support the quality/work comparison only. The
initial longer search pilot was stopped and is excluded from these results.

The benchmark uses an in-memory provider and exhaustive build routing, then
searches a real DiskANN centroid graph. Equal logical vector/list scans do not
measure physical SSD I/O, bytes, cache behavior, or Mimir write amplification.

## Reproduce and inspect

[Recorded results](graph-ivf-enron-results.json) contain all completed
build/search repetitions, input parameters, file/binary SHA-256 hashes, and the
summary. Data files are not included in the repository. The preparation script
requires NumPy and refuses to overwrite existing outputs.

From the repository root, set the paths to the two source files above:

```powershell
python diskann-benchmark/scripts/prepare_graph_ivf_enron.py --base <base-fp16-file> --queries <query-fp16-file> --output target/enron-comparison/data
cargo build --release -p diskann-tools --bin compute_groundtruth --offline
./target/release/compute_groundtruth.exe --data-type float --dist-fn l2 --base-file target/enron-comparison/data/enron-base-f32.bin --query-file target/enron-comparison/data/enron-queries-f32.bin --gt-file target/enron-comparison/data/enron-groundtruth-l2-k10.bin --recall-at 10
```

Build `diskann-benchmark` separately at each pinned commit and save the executables
as `before.exe` and `after.exe`. Run the same files on each executable, with
separate result names and CPU affinity, reversing the order on alternate rounds:

```powershell
./after.exe run --input-file diskann-benchmark/perf_test_inputs/graph-ivf-enron-build.json --output-file target/enron-after-build.json
./after.exe run --input-file diskann-benchmark/perf_test_inputs/graph-ivf-enron-search.json --output-file target/enron-after-search.json
```

The build input runs five builds for each of the two batch sizes. Repeat it in
four AB/BA pairs to reproduce the sample count. The search input is a full-query
quality check with one pass per setting; use a controlled host and warmed,
repeated passes for a valid throughput study. Data loading, conversion,
groundtruth, and recall calculation are outside benchmark timers.

The committed preparation script was rerun over the complete source corpus and
queries; both output hashes match the files used in these measurements. No
algorithm or benchmark timing implementation changed for this follow-up.
