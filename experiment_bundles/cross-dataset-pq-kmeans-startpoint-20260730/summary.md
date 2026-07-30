# Cross-dataset PQ-kmeans start-point router summary

Bundle: `/private/tmp/diskann-pq-kmeans-bfs-cache-router/experiment_bundles/cross-dataset-pq-kmeans-startpoint-20260730`

Parsed sources:

- `results/bigann_search_baseline_output.json`
- `results/bigann_build_pq_kmeans_router_k1024_output.json`
- `results/bigann_search_pq_kmeans_k1024_msp8_topk_output.json`
- `logs/wikipedia_build_disk_index.log`

Search rows are `search_l=200`, `beam_width=64`, `recall_at=100`, 4 search threads, 2 warmup runs, and 5 measured repetitions. Latency and router-time values are in microseconds.

| Dataset | Variant | Recall@100 (%) | QPS | Mean latency (us) | P95 (us) | P999 (us) | Hops | IOs | Comparisons | Router time (us) | Scanned codes | Starts | Artifact bytes | Build time | Delta vs dataset baseline |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| BigANN 10M | Baseline | 96.6686 | 3,192.90 | 1,242.96 | 1,608 | 22,514 | 451.76 | 451.76 | 10,593.16 | 0.00 | 0 | 1.00 | — | — | Baseline |
| BigANN 10M | PQ-kmeans router k=1024, max_start_points=8 | 96.7447 | 4,477.91 | 888.19 | 1,126 | 2,126 | 400.05 | 400.05 | 8,715.79 | 10.04 | 1,024 | 8.00 | 135,205 | 0.706 s (705,877 us) | Recall +0.076 pp; QPS +1,285.01 (+40.25%); mean latency -354.77 us (-28.54%) |
| Wikipedia/Cohere 10M | Baseline | — | — | — | — | — | — | — | — | — | — | — | — | — | Blocked before disk-index build completed; no dataset baseline was produced |
| Wikipedia/Cohere 10M | PQ-kmeans router k=1024, max_start_points=8 | — | — | — | — | — | — | — | — | — | — | — | — | — | Blocked by source vector-file format mismatch; router artifact was not built |

## BigANN router delta details

Compared with the BigANN dataset baseline, the PQ-kmeans router row changed:

- Recall: `+0.0760955811` percentage points (`+0.0787%` relative)
- QPS: `+1,285.010742` (`+40.2459%`)
- Mean latency: `-354.77438 us` (`-28.5426%`)
- P95 latency: `-482 us` (`-29.9751%`)
- P999 latency: `-20,388 us` (`-90.5570%`)
- Mean hops: `-51.7133` (`-11.4470%`)
- Mean IOs: `-51.7133` (`-11.4470%`)
- Mean comparisons: `-1,877.3681` (`-17.7225%`)
- Mean router time: `+10.03782 us`
- Mean router scanned codes: `+1,024`
- Mean routed start points: `+7`

The BigANN PQ-kmeans router artifact was written as:

- Artifact: `/private/tmp/diskann-pq-kmeans-bfs-cache-router/outputs/bigann_10m.pq_kmeans_router_k1024.bin`
- Artifact bytes: `135,205`
- Build time: `0.706 s` from the log, `705,877 us` from the JSON result

## Wikipedia/Cohere blocker

The Wikipedia/Cohere disk-index build failed before search or router-build metrics were produced. Exact log excerpt:

```text
Error: ANNError: DiskANN(InvalidFileFormatError)

Vector file '/Users/xiaoweijiang/Documents/diskann/bigann10Mdatasets/wikipedia_cohere/wikipedia_base.bin.crop_nb_10000000' has invalid format: size 8725200896 bytes doesn't match expected size of 107520000008 bytes based on header (35000000 vectors of dimension 768) -- (/private/tmp/diskann-pq-kmeans-bfs-cache-router/diskann-providers/src/utils/sampling.rs:149)
```
