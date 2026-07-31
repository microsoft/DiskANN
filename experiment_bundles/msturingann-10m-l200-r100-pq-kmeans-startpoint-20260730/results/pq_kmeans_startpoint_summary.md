# PQ-kmeans start-point router experiment summary

Experiment bundle:

`experiment_bundles/msturingann-10m-l200-r100-pq-kmeans-startpoint-20260730`

Working directory:

`/private/tmp/diskann-pq-kmeans-bfs-cache-router`

## Summary

This rerun measures the fixed PQ-kmeans start-point router on MSTuringANN 10M with corrected IO accounting. The original PQ-kmeans implementation was invalid because it clustered and scored PQ label IDs as if label ordinals were coordinates. The fixed implementation uses PQ/codebook geometry instead:

- Build-time k-means inflates PQ codes back to reconstructed vectors, keeps centroids in original `f32` vector space, assigns compressed points to centroids with `FixedChunkPQTable::l2_distance`, and chooses representative samples by the same geometry.
- Query-time routing uses ADC-style scoring: `populate_chunk_distances(query)` plus `pq_dist_lookup_single(representative_code)`.
- The BFS cache design is still multi-source: `num_nodes_to_cache=50000` is spread from the routed candidate starts instead of only the single baseline medoid.

The useful readout changed after the IO counter fix: physical disk IOs and logical vertex loads are now reported separately. On MSTuringANN 10M, `k=1024, max_start_points=8` is the strongest row in this rerun: versus baseline it improves recall by +0.171 pp, improves QPS by +26.99%, lowers mean latency by -21.21%, lowers physical mean IOs by -2.04%, lowers traversal logical loads / hops by -19.51%, and lowers comparisons by -23.15%.

All search rows below use search L=200, K=100, beam width=64, 4 threads, `num_nodes_to_cache=50000`, 2 warmup runs, 5 measured repetitions, squared L2 distance, and no vector filters.

## MSTuringANN 10M 10%-training-sample rerun

Scope and router settings:

| Setting | Value |
| --- | ---: |
| Dataset size | N=10,000,000 |
| Router training_sample_size | 1,000,000 (10%N) |
| Router max_start_points | 8 |
| Search L | 200 |
| recall@K | 100 |
| beam width | 64 |
| threads | 4 |
| num_nodes_to_cache | 50,000 |
| warmup runs | 2 |
| measured repetitions | 5 |

Current baseline:

`results/search_baseline_10pct_iometrics_output.json`

Fixed-geometry 10%-sample router rows:

- `results/search_pq_kmeans_k256_msp8_geometry_10pct_iometrics_output.json`
- `results/search_pq_kmeans_k512_msp8_geometry_10pct_iometrics_output.json`
- `results/search_pq_kmeans_k1024_msp8_geometry_10pct_iometrics_output.json`

Build artifacts for these search rows were produced by the matching `build_pq_kmeans_router_k{256,512,1024}_geometry_10pct_iometrics_output.json` runs.

### Absolute metrics

`Mean IOs` is the corrected physical disk IO count. `Mean loads` is the total logical vertices loaded, including traversal plus rerank. Traversal and rerank metrics are shown separately to make the corrected accounting explicit.

| Variant | k | Start points | Recall@100 | QPS | Mean latency | P95 | P99.9 | Mean IOs | Mean loads | Cache hit | Trav IOs | Trav loads | Trav hit | Rerank IOs | Rerank loads | Rerank hit | Hops | Comparisons | Router time | Router codes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | — | 1 | 73.4835 | 2929.19 | 1362.45 us | 1737 us | 13013 us | 336.243 | 692.183 | 51.42% | 336.243 | 491.183 | 31.54% | 0.000 | 201.000 | 100.00% | 491.183 | 23360.0 | 0.00 us | 0 |
| PQ-geometry k=256 top8 | 256 | 8 | 73.6356 | 3391.69 | 1176.94 us | 1572 us | 3443 us | 348.125 | 640.108 | 45.61% | 348.125 | 432.108 | 19.44% | 0.000 | 208.000 | 100.00% | 432.108 | 19714.6 | 36.64 us | 256 |
| PQ-geometry k=512 top8 | 512 | 8 | 73.6069 | 3488.05 | 1143.81 us | 1546 us | 3481 us | 336.109 | 624.177 | 46.15% | 336.109 | 416.177 | 19.24% | 0.000 | 208.000 | 100.00% | 416.177 | 18919.2 | 43.90 us | 512 |
| PQ-geometry k=1024 top8 | 1,024 | 8 | 73.6540 | 3719.83 | 1073.47 us | 1442 us | 1858 us | 329.393 | 603.332 | 45.40% | 329.393 | 395.332 | 16.68% | 0.000 | 208.000 | 100.00% | 395.332 | 17952.1 | 58.21 us | 1024 |

### Delta vs baseline

Percentages in parentheses are relative to the baseline metric. Percentage-point deltas are used for recall and cache-hit rates.

| Variant | Recall delta | QPS delta | Mean latency delta | P95 delta | P99.9 delta | Mean IOs delta | Mean loads delta | Cache hit delta | Trav IOs delta | Trav loads / hops delta | Trav hit delta | Rerank loads delta | Comparisons delta | Router-time delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PQ-geometry k=256 top8 | +0.152 pp (+0.21%) | +462.50 (+15.79%) | -185.51 us (-13.62%) | -165 us (-9.50%) | -9570 us (-73.54%) | +11.883 (+3.53%) | -52.076 (-7.52%) | -5.81 pp (-11.30%) | +11.883 (+3.53%) | -59.076 (-12.03%) | -12.11 pp (-38.39%) | +7.000 (+3.48%) | -3645.4 (-15.60%) | +36.64 us |
| PQ-geometry k=512 top8 | +0.123 pp (+0.17%) | +558.86 (+19.08%) | -218.64 us (-16.05%) | -191 us (-11.00%) | -9532 us (-73.25%) | -0.133 (-0.04%) | -68.007 (-9.82%) | -5.27 pp (-10.25%) | -0.133 (-0.04%) | -75.007 (-15.27%) | -12.31 pp (-39.01%) | +7.000 (+3.48%) | -4440.8 (-19.01%) | +43.90 us |
| PQ-geometry k=1024 top8 | +0.171 pp (+0.23%) | +790.65 (+26.99%) | -288.98 us (-21.21%) | -295 us (-16.98%) | -11155 us (-85.72%) | -6.850 (-2.04%) | -88.852 (-12.84%) | -6.02 pp (-11.70%) | -6.850 (-2.04%) | -95.852 (-19.51%) | -14.86 pp (-47.13%) | +7.000 (+3.48%) | -5407.9 (-23.15%) | +58.21 us |

### MSTuringANN readout

- `k=1024/top8` is the best row in this 10%N rerun by QPS, mean latency, tail latency, corrected physical mean IOs, total logical loads, traversal loads / hops, and comparisons. Recall is also slightly above baseline.
- `k=256/top8` still improves latency and logical traversal, but its corrected physical mean IOs are higher than baseline (+3.53%). This would have been hidden by the old logical-load-only counter.
- `k=512/top8` is close to baseline on corrected physical mean IOs (-0.04%) while improving latency by -16.05% and traversal loads by -15.27%.
- Rerank physical IOs are 0.0 for all rows because the rerank vertices are cache hits in this run. Rerank logical loads are 201 for baseline and 208 for routed rows.
- Overall and traversal cache-hit percentages are lower in the routed rows even though latency and traversal work improve. The routed searches load fewer traversal vertices but touch a different vertex mix, so cache-hit percentage alone is not the optimization objective.

## Old label-ID results are invalid reference rows

Old rows that treated PQ label IDs as geometric coordinates are retained only as invalid historical reference. They should not be used as final experiment evidence.

| Dataset | Invalid variant | Why invalid |
| --- | --- | --- |
| MSTuringANN | old label-ID k=1024/top8 | clustered/scored PQ label ordinals, whose ID order has no geometric meaning |

## Blocked datasets

### BigANN 10M

Excluded from the current valid result set. The local BigANN 10M index/PQ setup is not considered comparable for this PR because the dataset's corresponding PQ configuration is suspected to be wrong. Do not use earlier BigANN rows as evidence for this iteration.

### Wikipedia/Cohere 10M

Blocked locally: the available base vector file is invalid or truncated, and no usable prebuilt index sidecars were found.

Observed file:

`/Users/xiaoweijiang/Documents/diskann/bigann10Mdatasets/wikipedia_cohere/wikipedia_base.bin.crop_nb_10000000`

The file header says `35000000 x 768`, but the file size is only 8,725,200,896 bytes. That does not match the expected f32 payload size for the header, nor a valid 10M crop.

### Enron

Blocked locally: no Enron base vectors, query file, groundtruth, or prebuilt DiskANN index triplet were found under the searched local data/output roots.

## Verification snapshot

The final verification suite is run after this report update:

```bash
cargo test -p diskann-disk pq_kmeans_router
cargo test -p diskann-disk io_tracker_counts_physical_reads_and_logical_loads_separately
cargo test -p diskann-benchmark --features disk-index cache_hit_percentage
cargo check -p diskann-benchmark --features disk-index
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
git diff --check
```
