# PQ-kmeans start-point router experiment summary

Experiment bundle:

`experiment_bundles/msturingann-10m-l200-r100-pq-kmeans-startpoint-20260730`

Working directory:

`/private/tmp/diskann-pq-kmeans-bfs-cache-router`

## Summary

The original PQ-kmeans experiment had a correctness bug: it clustered and scored PQ label IDs as if label ordinals were coordinates. That is not valid, because PQ centroid ID order has no geometric meaning.

The fixed implementation now uses PQ/codebook geometry:

- Build-time k-means inflates PQ codes back to reconstructed vectors, keeps centroids in original `f32` vector space, assigns compressed points to centroids with `FixedChunkPQTable::l2_distance`, and chooses representative samples by the same geometry.
- Query-time routing uses ADC-style scoring: `populate_chunk_distances(query)` plus `pq_dist_lookup_single(representative_code)`.
- The BFS cache design is still multi-source: `num_nodes_to_cache=50000` is spread from the routed candidate starts instead of only the single baseline medoid.

After the fix, the useful result is stronger than the old invalid label-ID result on traversal and latency. On MSTuringANN, k=1024/top8 is the best fixed-geometry row: recall improves by +0.574 pp, hops/IOs drop by 21.1%, comparisons drop by 23.2%, and mean latency drops by 13.8% versus the rerun baseline.

All valid search rows below use search L=200, K=100, beam width=64, 4 threads, `num_nodes_to_cache=50000`, 2 warmup runs, 5 measured repetitions, squared L2 distance, and no vector filters.

## MSTuringANN 10M fixed-geometry rerun

Current baseline:

`results/search_baseline_after_geometry_fix_output.json`

Fixed-geometry router rows:

- `results/search_pq_kmeans_k256_msp8_geometry_output.json`
- `results/search_pq_kmeans_k512_msp8_geometry_output.json`
- `results/search_pq_kmeans_k1024_msp8_geometry_output.json`

### Absolute metrics

| Variant | k | Start points | Recall@100 | QPS | Mean latency | P95 | P99.9 | Hops / IOs | Comparisons | Router time | Scanned codes | Artifact bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | — | 1 | 73.4835 | 3212.43 | 1243.59 us | 1569 us | 2519 us | 491.183 | 23360.0 | 0.0 us | 0 | — |
| PQ-geometry k=256 top8 | 256 | 8 | 74.3409 | 3506.10 | 1138.69 us | 1441 us | 2549 us | 404.977 | 19138.7 | 36.7 us | 256 | 17,445 |
| PQ-geometry k=512 top8 | 512 | 8 | 74.1320 | 3614.01 | 1104.49 us | 1418 us | 2720 us | 397.728 | 18591.4 | 44.6 us | 512 | 34,853 |
| PQ-geometry k=1024 top8 | 1,024 | 8 | 74.0571 | 3725.45 | 1071.73 us | 1381 us | 1761 us | 387.552 | 17936.1 | 58.6 us | 1024 | 69,669 |

### Delta vs baseline

| Variant | Recall delta | QPS delta | Mean latency delta | P95 delta | P99.9 delta | Hops / IOs delta | Comparisons delta | Router-time delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PQ-geometry k=256 top8 | +0.857 pp | +293.67 (+9.14%) | -104.89 us (-8.43%) | -128 us (-8.16%) | +30 us (+1.19%) | -86.207 (-17.55%) | -4221.4 (-18.07%) | +36.7 us |
| PQ-geometry k=512 top8 | +0.649 pp | +401.58 (+12.50%) | -139.10 us (-11.18%) | -151 us (-9.62%) | +201 us (+7.98%) | -93.455 (-19.03%) | -4768.6 (-20.41%) | +44.6 us |
| PQ-geometry k=1024 top8 | +0.574 pp | +513.02 (+15.97%) | -171.86 us (-13.82%) | -188 us (-11.98%) | -758 us (-30.09%) | -103.632 (-21.10%) | -5423.9 (-23.22%) | +58.6 us |

### MSTuringANN readout

- k=1024/top8 is the strongest fixed-geometry point in this rerun: it has the lowest latency, lowest hops/IOs, lowest comparisons, best QPS, and no recall regression.
- k=256 and k=512 are still useful if the priority is even smaller router metadata. They already capture most of the traversal reduction, with only 17 KB and 35 KB artifacts.
- Router CPU cost is now higher than the old label-ID implementation because ADC computes/query-loads codebook distances instead of doing ordinal byte subtraction. The traversal reduction more than pays for it at k=1024.

## BigANN 10M fixed-geometry rerun

Current baseline:

`experiment_bundles/cross-dataset-pq-kmeans-startpoint-20260730/results/bigann_search_baseline_after_geometry_fix_output.json`

Fixed-geometry router row:

`experiment_bundles/cross-dataset-pq-kmeans-startpoint-20260730/results/bigann_search_pq_kmeans_k1024_msp8_geometry_output.json`

### Absolute metrics

| Variant | k | Start points | Recall@100 | QPS | Mean latency | P95 | P99.9 | Hops / IOs | Comparisons | Router time | Scanned codes | Artifact bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | — | 1 | 96.6686 | 4414.30 | 902.12 us | 1043 us | 1184 us | 451.764 | 10593.2 | 0.0 us | 0 | — |
| PQ-geometry k=1024 top8 | 1,024 | 8 | 96.7224 | 4867.83 | 817.64 us | 963 us | 1149 us | 341.307 | 7262.2 | 120.1 us | 1024 | 135,205 |

### Delta vs baseline

| Variant | Recall delta | QPS delta | Mean latency delta | P95 delta | P99.9 delta | Hops / IOs delta | Comparisons delta | Router-time delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PQ-geometry k=1024 top8 | +0.054 pp | +453.53 (+10.27%) | -84.48 us (-9.37%) | -80 us (-7.67%) | -35 us (-2.96%) | -110.457 (-24.45%) | -3330.9 (-31.44%) | +120.1 us |

### BigANN readout

BigANN still benefits after the geometry fix. The result is good because the routed starts cut a large amount of graph traversal: hops/IOs fall by 24.5% and comparisons fall by 31.4%. Even with a 120 us ADC router cost, the reduced traversal produces a 9.4% mean-latency win.

The dataset likely helps because BigANN/SIFT-like uint8 vectors have strong local visual-feature structure that PQ codebooks preserve well enough for coarse routing. In that setting, finding several geometrically close representative starts avoids a lot of baseline medoid-to-neighborhood travel.

## Old label-ID results are invalid reference rows

These rows are retained only to explain earlier discrepancies. They should not be used as the final experiment result because they treated PQ label IDs as geometric coordinates.

| Dataset | Invalid variant | Recall@100 | Mean latency | Hops / IOs | Comparisons | Router time | Why invalid |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| MSTuringANN | old label-ID k=1024/top8 | 74.6721 | 1264.76 us | 447.408 | 21828.8 | 6.64 us | clustered/scored PQ label ordinals |
| BigANN | old label-ID k=1024/top8 | 96.7447 | 888.19 us | 400.051 | 8715.8 | 10.04 us | clustered/scored PQ label ordinals |

The old label-ID router looked cheaper because it only did byte-level ordinal distance. The fixed router is semantically correct but spends more time in ADC scoring. The fixed MSTuring row nevertheless has much better latency and traversal because its start points are selected in PQ codebook geometry.

## Blocked datasets

### Wikipedia/Cohere 10M

Blocked locally: the available base vector file is invalid or truncated, and no usable prebuilt index sidecars were found.

Observed file:

`/Users/xiaoweijiang/Documents/diskann/bigann10Mdatasets/wikipedia_cohere/wikipedia_base.bin.crop_nb_10000000`

The file header says `35000000 x 768`, but the file size is only 8,725,200,896 bytes. That does not match the expected f32 payload size for the header, nor a valid 10M crop.

### Enron

Blocked locally: no Enron base vectors, query file, groundtruth, or prebuilt DiskANN index triplet were found under the searched local data/output roots.

## Caveats

- The reported `cache_hit_percentage` remains 0.0% in these outputs. The BFS cache path is implemented as multi-source cache population, but the benchmark counter currently does not expose meaningful physical cache-hit attribution for this path. Use hops/IOs/comparisons/latency as the practical effect metrics for this experiment.
- The benchmark's `mean_ios` and `mean_hops` are identical in these runs, so the IO number should be read as logical traversal/load count, not verified storage-device misses.
- Router metadata memory remains tiny: 69,669 bytes for MSTuring k=1024 and 135,205 bytes for BigANN k=1024. The additional retained PQ geometry table is also small because it is just the PQ codebook/pivots, not per-point vectors.

## Verification snapshot

Commands run after the geometry fix:

```bash
cargo build -p diskann-benchmark --release --features disk-index
cargo test -p diskann-disk pq_kmeans_router
```

The full final verification set is recorded in the PR/update handoff, including format, tests, benchmark check, and clippy.
