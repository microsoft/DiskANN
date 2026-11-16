# e-means speed and quality benchmark

See [2026-11-16 results](results/2026-11-16/README.md) for figures, raw measurements,
configuration, data hashes and environment details. Raw datasets are not committed.

## Comparisons

| Case | Baseline | e-means | Training / validation |
| --- | --- | --- | --- |
| SIFT clustering | Lloyd, random initialization | Same initial centers | 100,000 x 128D / 10,000; k=1024 |
| SIFT PQ | Actual LightPQTrainingParameters, k-means++ | Random initialization | 50,000 x 128D / 10,000; 16 x 8D chunks; k=256 |
| YFCC PQ | Actual LightPQTrainingParameters, k-means++ | Random initialization | 7,838 x 192D / 2,000; 24 x 8D chunks; k=256 |

The PQ comparison changes initialization and optimizer together. The example's
adapter does not replace the production PQ trainer. Both paths use sequential
execution, including all PQ chunks, and the same SIMD implementation.

Five seeds (1, 17, 42, 99, 1234) and three timing repeats are used. Lloyd budgets
are 5, 12 and 25; e-means budgets are 1, 2, 3, 5 and 10. Run order is shuffled
per seed/repeat, and both paths are warmed up. Each point is an independent fit,
not a checkpoint. Default batch/step/revival parameters are not tuned.

## Timing and statistics

Timing starts before schema/RNG construction and ends with the completed table.
It includes initialization, input norms, transpose/batch scratch, training and
center packing. Disk I/O, dataset preparation, warmup, quality scoring and logging
are excluded. Data are in RAM. The process is not CPU-pinned.

Every repeat performs initialization and training again; no shared initialization
duration is added to a separately timed fit. Repeats must produce bit-identical
centers. Quality is measured once per codebook: five independent samples, not fifteen.

- Time: median of three repeats per seed, then median across five seeds.
- Speedup: ratio of those aggregate median baseline and e-means times.
- MSE change: median of per-seed percentage changes against the corresponding
  baseline. The speed-quality plot uses Lloyd@25 as zero.
- Recall change: median of paired per-seed percentage-point changes.
- Error bars are interquartile ranges, not confidence intervals. Win counts and
  all per-seed samples are retained; a median improvement does not imply every seed wins.

## Quality

Final centers receive fresh assignments. Clustering uses FP32 SIMD assignment
and f64 squared-error accumulation. PQ uses the production BasicTable byte
encoder and f64 reconstruction error. MSE is mean squared vector distance,
without division by dimension.

Recall@10 is exhaustive f64 PQ asymmetric-distance scoring with unquantized
queries, compared to exhaustive f64 squared-L2 ground truth on original vectors.
Ties use database row ID. This is **PQ ADC recall**, not DiskANN graph-search
recall or QPS. SIFT uses 100,000 database vectors and 1,000 queries; YFCC uses
10,000 database vectors and 100 queries. The small YFCC query set limits conclusions.

SIFT comes from ANN Benchmarks, sampled with seed 20261112. YFCC comes from the
repository LFS files: seed 20261116 selects 2,000 validation rows; 162 candidate
training rows exactly matching validation or query vectors are excluded, leaving
7,838. The retrieval database remains the complete supplied database.

## Reproduce

Recorded Python tools: Python 3.10, NumPy 1.23.3, h5py 3.16.0, Matplotlib 3.7.2.
These are only needed for data/plots. The Rust example adds no dependencies.
Run from the repository root:

```powershell
New-Item -ItemType Directory -Force target/emeans-source
curl.exe -L --fail https://ann-benchmarks.com/sift-128-euclidean.hdf5 -o target/emeans-source/sift.hdf5
python diskann-quantization/examples/emeans_benchmark/prepare_sift.py target/emeans-source/sift.hdf5 target/emeans-source
git lfs pull --include="test_data/yfcc/*.fbin"
python diskann-quantization/examples/emeans_benchmark/prepare_data.py --sift-dir target/emeans-source
cargo run --release -p diskann-quantization --example emeans_benchmark --no-default-features -- target/emeans-benchmark/config.json
python diskann-quantization/examples/emeans_benchmark/report.py target/emeans-benchmark/config.json target/emeans-benchmark/measurements.jsonl target/emeans-benchmark/report
```

prepare_data.py also accepts --yfcc-dir pointing to hydrated LFS files. Measurements
refuse to overwrite existing files; change the output path for a repeated experiment.
Two tests independently check scoring formulas, ADC and tie handling:

```powershell
cargo test -p diskann-quantization --example emeans_benchmark --no-default-features
```
