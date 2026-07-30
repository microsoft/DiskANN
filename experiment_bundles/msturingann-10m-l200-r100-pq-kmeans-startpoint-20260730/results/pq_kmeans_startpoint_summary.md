# PQ-kmeans start-point router experiment summary

Experiment bundle:

`experiment_bundles/msturingann-10m-l200-r100-pq-kmeans-startpoint-20260730`

Working directory:

`/private/tmp/diskann-pq-kmeans-bfs-cache-router`

This report now contains the original MSTuringANN results plus cross-dataset sections for BigANN 10M and Wikipedia/Cohere 10M.

## Implementation summary

This experiment evaluates a PQ-code k-means start-point router for disk index search on the MSTuringANN 10M index.

- The router builds `k = ceil(sqrt(N))` representatives, so 10,000,000 points produce 3,163 representatives.
- Each query scans the representative PQ codes and selects multiple routed graph start points, capped by `max_start_points`.
- The search path uses multi-start routing with the existing medoid fallback behavior.
- The cache warmup path was changed to multi-source BFS so the cached nodes reflect all selected routed starts instead of a single start.
- Search runs used 2 warmup runs and 5 measured repetitions.
- Router stats are emitted per search result: mean router time, scanned representative codes, and routed start points.
- The final top-k optimization keeps only the best `max_start_points` representatives while scanning, avoiding a full representative sort. The router still scans all 3,163 representative codes, but msp8 router time dropped from the earlier roughly 39 us path to roughly 12 us.

All search rows below use search L=200, K=100, beam width=64, 4 threads, 50,000 cached nodes, squared L2 distance, and no vector filters.

## Commands

Build the router artifact:

```bash
target/release/diskann-benchmark --quiet run --input-file experiment_bundles/msturingann-10m-l200-r100-pq-kmeans-startpoint-20260730/configs/build_pq_kmeans_router.json --output-file experiment_bundles/msturingann-10m-l200-r100-pq-kmeans-startpoint-20260730/results/build_pq_kmeans_router_output.json
```

Baseline initial run:

```bash
target/release/diskann-benchmark --quiet run --input-file experiment_bundles/msturingann-10m-l200-r100-pq-kmeans-startpoint-20260730/configs/search_baseline.json --output-file experiment_bundles/msturingann-10m-l200-r100-pq-kmeans-startpoint-20260730/results/search_baseline_output.json
```

Baseline confirmation rerun:

```bash
target/release/diskann-benchmark --quiet run --input-file experiment_bundles/msturingann-10m-l200-r100-pq-kmeans-startpoint-20260730/configs/search_baseline.json --output-file experiment_bundles/msturingann-10m-l200-r100-pq-kmeans-startpoint-20260730/results/search_baseline_rerun_output.json
```

PQ-kmeans msp8 top-k initial run:

```bash
target/release/diskann-benchmark --quiet run --input-file experiment_bundles/msturingann-10m-l200-r100-pq-kmeans-startpoint-20260730/configs/search_pq_kmeans_msp8.json --output-file experiment_bundles/msturingann-10m-l200-r100-pq-kmeans-startpoint-20260730/results/search_pq_kmeans_msp8_topk_output.json
```

PQ-kmeans msp8 top-k confirmation rerun:

```bash
target/release/diskann-benchmark --quiet run --input-file experiment_bundles/msturingann-10m-l200-r100-pq-kmeans-startpoint-20260730/configs/search_pq_kmeans_msp8.json --output-file experiment_bundles/msturingann-10m-l200-r100-pq-kmeans-startpoint-20260730/results/search_pq_kmeans_msp8_topk_rerun_output.json
```

PQ-kmeans msp16 top-k run:

```bash
target/release/diskann-benchmark --quiet run --input-file experiment_bundles/msturingann-10m-l200-r100-pq-kmeans-startpoint-20260730/configs/search_pq_kmeans_msp16.json --output-file experiment_bundles/msturingann-10m-l200-r100-pq-kmeans-startpoint-20260730/results/search_pq_kmeans_msp16_topk_output.json
```

PQ-kmeans msp4 top-k was run with the same search config shape and `max_start_points=4`; its generated config file is not retained in this bundle. The retained output path is:

`experiment_bundles/msturingann-10m-l200-r100-pq-kmeans-startpoint-20260730/results/search_pq_kmeans_msp4_topk_output.json`

## Router artifact build stats

| Metric | Value |
| --- | ---: |
| Artifact | `outputs/msturingann10m_user_m59_l80_sq1_pq64.pq_kmeans_router.bin` |
| Points | 10,000,000 |
| PQ chunks | 64 |
| Representatives | 3,163 |
| Max iterations | 4 |
| Build time | 2.399693 s |
| Build time, raw output | 2,399,693 us |
| Artifact bytes | 215,121 bytes |

## Final comparison

Recommended comparison is `baseline_rerun` versus `msp8_topk_rerun`, because both were confirmation runs after the top-k optimization landed.

| Variant | Output JSON | Recall@100 | QPS | Mean latency | P95 latency | P99.9 latency | Mean hops | Mean IOs | Cache hit | Router time | Router codes | Start points |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline initial | `search_baseline_output.json` | 73.4835 | 3334.91 | 1197.00 us | 1505 us | 1873 us | 491.183 | 491.183 | 0.0% | 0.000 us | 0 | 1 |
| Baseline rerun | `search_baseline_rerun_output.json` | 73.4835 | 3113.24 | 1282.58 us | 1628 us | 3248 us | 491.183 | 491.183 | 0.0% | 0.000 us | 0 | 1 |
| PQ-kmeans msp4 top-k | `search_pq_kmeans_msp4_topk_output.json` | 74.5265 | 3066.45 | 1302.00 us | 1577 us | 2998 us | 453.689 | 453.689 | 0.0% | 11.925 us | 3163 | 4 |
| PQ-kmeans msp8 top-k initial | `search_pq_kmeans_msp8_topk_output.json` | 74.6030 | 3095.83 | 1288.53 us | 1560 us | 2535 us | 448.346 | 448.346 | 0.0% | 12.014 us | 3163 | 8 |
| PQ-kmeans msp8 top-k rerun, recommended | `search_pq_kmeans_msp8_topk_rerun_output.json` | 74.6030 | 3155.72 | 1265.80 us | 1533 us | 3024 us | 448.346 | 448.346 | 0.0% | 11.933 us | 3163 | 8 |
| PQ-kmeans msp16 top-k | `search_pq_kmeans_msp16_topk_output.json` | 74.8046 | 2980.20 | 1337.66 us | 1637 us | 3456 us | 450.488 | 450.488 | 0.0% | 14.134 us | 3163 | 16 |

## Deltas versus baseline rerun

| Variant | Recall delta | QPS delta | Mean latency delta | P95 delta | P99.9 delta | Hops / IO delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PQ-kmeans msp4 top-k | +1.0430 | -1.50% | +19.41 us | -51 us | -250 us | -37.494 (-7.63%) |
| PQ-kmeans msp8 top-k rerun, recommended | +1.1195 | +1.36% | -16.78 us (-1.31%) | -95 us | -224 us | -42.838 (-8.72%) |
| PQ-kmeans msp16 top-k | +1.3211 | -4.27% | +55.07 us | +9 us | +208 us | -40.696 (-8.29%) |

## Conclusion

For the initial max_start_points comparison, `msp8_topk_rerun` is the best confirmation row. After the k sweep below, the overall recommendation is k=1,024 with max_start_points=8 (k=1024/msp8).

- There is no recall regression: recall improves from 73.4835 to 74.6030 versus `baseline_rerun`.
- Router memory overhead is tiny: the artifact is 215,121 bytes, approximately `sqrt(N) * (4 + 64)` bytes plus bincode/container overhead.
- Hops and IOs fall from 491.183 to 448.346, a reduction of about 8.72%.
- Mean latency improves from 1282.58 us to 1265.80 us, a reduction of about 1.31% on the confirmation rerun.
- P95 improves by 95 us, from 1628 us to 1533 us.
- P99.9 improves by 224 us, from 3248 us to 3024 us.
- `msp16` gives the highest recall, but the confirmation comparison makes it less attractive because its latency and QPS are worse than `msp8`.
- `msp4` uses fewer starts, but it is less favorable than `msp8`: lower recall, lower QPS, and higher mean latency versus the recommended `msp8_topk_rerun`.

## Memory overhead detail

The router artifact is a bincode-serialized `PqKmeansRouterData` containing:

- `num_points: usize`
- `num_pq_chunks: usize`
- `representative_ids: Vec<u32>`
- `representative_codes: Vec<u8>`
- `fallback_medoid: Option<u32>`

Approximate payload formula:

```text
overhead_bytes ~= k * (sizeof(u32) + num_pq_chunks * sizeof(u8)) + serialization/container overhead
```

For this run:

```text
k = 3,163
num_pq_chunks = 64
payload ~= 3,163 * (4 + 64) = 215,084 bytes
measured artifact = 215,121 bytes
serialization/container overhead = 37 bytes
```

Measured overhead relative to the 10M-point index is about 0.0215 bytes per point.

## k sweep: PQ-kmeans router representatives

This sweep varied `num_representatives` while keeping `max_start_points=8`, warmup runs=2, repetitions=5, search L=200, beam width=64, and `num_nodes_to_cache=50000`. The default comparison row is the existing `msp8_topk_rerun` artifact with k=3,163.

### Absolute k sweep metrics

This table keeps the baseline absolute values in the same view as the k sweep, so the delta tables below can be read without losing the raw reference numbers.

| Variant | k | Max start points | Recall@100 | QPS | Mean latency | P95 | P99.9 | Mean hops / IOs | Mean comparisons | Router time | Scanned codes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline rerun | — | 1 | 73.4835 | 3113.24 | 1282.58 us | 1628 us | 3248 us | 491.183 / 491.183 | 23360.0 | 0 us | 0 |
| PQ-kmeans k=256 msp8 | 256 | 8 | 74.6622 | 3133.79 | 1274.07 us | 1559 us | 3302 us | 448.734 / 448.734 | 21872.2 | 4.754 us | 256 |
| PQ-kmeans k=512 msp8 | 512 | 8 | 74.6366 | 3140.59 | 1270.87 us | 1551 us | 2826 us | 448.262 / 448.262 | 21843.7 | 5.371 us | 512 |
| PQ-kmeans k=1,024 msp8 | 1,024 | 8 | 74.6721 | 3157.21 | 1264.76 us | 1537 us | 3059 us | 447.408 / 447.408 | 21828.8 | 6.637 us | 1024 |
| PQ-kmeans k=2,048 msp8 | 2,048 | 8 | 74.6387 | 3086.34 | 1290.39 us | 1567 us | 3483 us | 447.898 / 447.898 | 21834.8 | 9.851 us | 2048 |
| PQ-kmeans default k=3,163 msp8 top-k rerun | 3,163 | 8 | 74.6030 | 3155.72 | 1265.80 us | 1533 us | 3024 us | 448.346 / 448.346 | 21855.8 | 11.933 us | 3163 |
| PQ-kmeans k=4,096 msp8 | 4,096 | 8 | 74.6565 | 3046.44 | 1305.34 us | 1585 us | 3267 us | 447.698 / 447.698 | 21831.4 | 15.113 us | 4096 |

Build and artifact details for the k rows:

| k | Representatives | Artifact bytes, JSON | Artifact bytes, file | Build time |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 256 | 17,445 | 17,445 | 4.429069 s |
| 512 | 512 | 34,853 | 34,853 | 0.828031 s |
| 1,024 | 1,024 | 69,669 | 69,669 | 1.148607 s |
| 2,048 | 2,048 | 139,301 | 139,301 | 1.173853 s |
| 3,163 default | 3,163 | 215,121 | 215,121 | 2.399693 s |
| 4,096 | 4,096 | 278,565 | 278,565 | 1.779044 s |

### Delta vs baseline rerun

| Variant | Recall delta | QPS delta | Mean latency delta | P95 delta | P99.9 delta | Hops delta | IOs delta | Comparisons delta | Router-time delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PQ-kmeans k=256 msp8 | +1.1787 | +0.66% | -8.5 us (-0.66%) | -69 us | +54 us | -42.449 (-8.64%) | -42.449 (-8.64%) | -1487.8 | +4.8 us |
| PQ-kmeans k=512 msp8 | +1.1531 | +0.88% | -11.7 us (-0.91%) | -77 us | -422 us | -42.922 (-8.74%) | -42.922 (-8.74%) | -1516.4 | +5.4 us |
| PQ-kmeans k=1,024 msp8 | +1.1887 | +1.41% | -17.8 us (-1.39%) | -91 us | -189 us | -43.775 (-8.91%) | -43.775 (-8.91%) | -1531.2 | +6.6 us |
| PQ-kmeans k=2,048 msp8 | +1.1552 | -0.86% | +7.8 us (+0.61%) | -61 us | +235 us | -43.286 (-8.81%) | -43.286 (-8.81%) | -1525.2 | +9.9 us |
| PQ-kmeans default k=3,163 msp8 top-k rerun | +1.1195 | +1.36% | -16.8 us (-1.31%) | -95 us | -224 us | -42.837 (-8.72%) | -42.837 (-8.72%) | -1504.2 | +11.9 us |
| PQ-kmeans k=4,096 msp8 | +1.1730 | -2.15% | +22.8 us (+1.77%) | -43 us | +19 us | -43.485 (-8.85%) | -43.485 (-8.85%) | -1528.6 | +15.1 us |

Deltas versus default k=3,163 `msp8_topk_rerun`:

| k | Recall delta | QPS delta | Mean latency delta | P95 delta | P99.9 delta | Hops delta | IOs delta | Comparisons delta | Router-time delta | Scanned-codes delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | +0.0592 | -0.69% | +8.3 us (+0.65%) | +26 us | +278 us | +0.388 (+0.09%) | +0.388 (+0.09%) | +16.4 | -7.2 us | -2907 |
| 512 | +0.0336 | -0.48% | +5.1 us (+0.40%) | +18 us | -198 us | -0.084 (-0.02%) | -0.084 (-0.02%) | -12.2 | -6.6 us | -2651 |
| 1,024 | +0.0692 | +0.05% | -1.0 us (-0.08%) | +4 us | +35 us | -0.938 (-0.21%) | -0.938 (-0.21%) | -27.0 | -5.3 us | -2139 |
| 2,048 | +0.0357 | -2.20% | +24.6 us (+1.94%) | +34 us | +459 us | -0.448 (-0.10%) | -0.448 (-0.10%) | -21.0 | -2.1 us | -1115 |
| 4,096 | +0.0536 | -3.46% | +39.5 us (+3.12%) | +52 us | +243 us | -0.647 (-0.14%) | -0.647 (-0.14%) | -24.4 | +3.2 us | +933 |

Quick read:

- Best recall in this sweep: k=1,024 at 74.6721 recall@100.
- Best QPS in this sweep: k=1,024 at 3157.21 QPS.
- Lowest mean latency in this sweep: k=1,024 at 1264.76 us.
- Best P99.9 in this sweep: k=512 at 2826 us.
- Recommendation for the k sweep: use k=1,024 with max_start_points=8 (k=1024/msp8).
- All swept k values slightly beat the existing default k=3,163 row on recall; k=1,024 is the strongest row here because it also very slightly improves QPS and mean latency while cutting router time to 6.637 us.
- The smaller k=256 and k=512 rows confirm that most of the recall/hops benefit is already present with much smaller router artifacts; k=512 is especially attractive if P99.9 matters more than the small k=1,024 recall/QPS/mean-latency edge.
- k=1,024 has slightly worse tail latency than the default row (+4 us P95, +35 us P99.9), so the choice depends on whether the small mean/QPS/recall gain is worth the small tail regression.

Generated files:

- k=256 build config: `configs/build_pq_kmeans_router_k256.json`
- k=256 build JSON/log: `results/build_pq_kmeans_router_k256_output.json`, `logs/build_pq_kmeans_router_k256.stdout.log`
- k=256 search config: `configs/search_pq_kmeans_k256_msp8_topk.json`
- k=256 search JSON/log: `results/search_pq_kmeans_k256_msp8_topk_output.json`, `logs/search_pq_kmeans_k256_msp8_topk.stdout.log`
- k=512 build config: `configs/build_pq_kmeans_router_k512.json`
- k=512 build JSON/log: `results/build_pq_kmeans_router_k512_output.json`, `logs/build_pq_kmeans_router_k512.stdout.log`
- k=512 search config: `configs/search_pq_kmeans_k512_msp8_topk.json`
- k=512 search JSON/log: `results/search_pq_kmeans_k512_msp8_topk_output.json`, `logs/search_pq_kmeans_k512_msp8_topk.stdout.log`
- k=1,024 build config: `configs/build_pq_kmeans_router_k1024.json`
- k=1,024 build JSON/log: `results/build_pq_kmeans_router_k1024_output.json`, `logs/build_pq_kmeans_router_k1024.stdout.log`
- k=1,024 search config: `configs/search_pq_kmeans_k1024_msp8_topk.json`
- k=1,024 search JSON/log: `results/search_pq_kmeans_k1024_msp8_topk_output.json`, `logs/search_pq_kmeans_k1024_msp8_topk.stdout.log`
- k=2,048 build config: `configs/build_pq_kmeans_router_k2048.json`
- k=2,048 build JSON/log: `results/build_pq_kmeans_router_k2048_output.json`, `logs/build_pq_kmeans_router_k2048.stdout.log`
- k=2,048 search config: `configs/search_pq_kmeans_k2048_msp8_topk.json`
- k=2,048 search JSON/log: `results/search_pq_kmeans_k2048_msp8_topk_output.json`, `logs/search_pq_kmeans_k2048_msp8_topk.stdout.log`
- k=4,096 build config: `configs/build_pq_kmeans_router_k4096.json`
- k=4,096 build JSON/log: `results/build_pq_kmeans_router_k4096_output.json`, `logs/build_pq_kmeans_router_k4096.stdout.log`
- k=4,096 search config: `configs/search_pq_kmeans_k4096_msp8_topk.json`
- k=4,096 search JSON/log: `results/search_pq_kmeans_k4096_msp8_topk_output.json`, `logs/search_pq_kmeans_k4096_msp8_topk.stdout.log`

## Cross-dataset results

These rows extend the MSTuringANN analysis with the retained cross-dataset experiment bundle:

`experiment_bundles/cross-dataset-pq-kmeans-startpoint-20260730`

Search rows use search L=200, K=100, beam width=64, 4 threads, 2 warmup runs, and 5 measured repetitions.

### BigANN 10M

Absolute BigANN baseline and router results:

| Variant | Recall@100 | QPS | Mean latency | P95 latency | P99.9 latency | Mean hops / IOs | Mean comparisons | Router time | Scanned codes | Start points | Artifact bytes | Build time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 96.6686 | 3192.90 | 1242.96 us | 1608 us | 22514 us | 451.76 | 10593.16 | 0 us | 0 | 1 | — | — |
| PQ-kmeans k=1024 msp8 | 96.7447 | 4477.91 | 888.19 us | 1126 us | 2126 us | 400.05 | 8715.79 | 10.04 us | 1024 | 8 | 135205 bytes | 0.706 s |

Delta versus the BigANN baseline:

| Metric | Delta |
| --- | ---: |
| Recall@100 | +0.076 pp (+0.0787% relative) |
| QPS | +1285.01 (+40.25%) |
| Mean latency | -354.77 us (-28.54%) |
| P95 latency | -482 us (-29.98%) |
| P99.9 latency | -20388 us (-90.56%) |
| Mean hops / IOs | -51.71 (-11.45%) |
| Mean comparisons | -1877.37 (-17.72%) |

The BigANN PQ-kmeans router artifact was retained at:

`/private/tmp/diskann-pq-kmeans-bfs-cache-router/outputs/bigann_10m.pq_kmeans_router_k1024.bin`

### Wikipedia/Cohere 10M

Wikipedia/Cohere is blocked because the local base vector file is invalid or truncated. No usable local prebuilt index or replacement base file was found, so no baseline search, router build, or router search metrics were produced.

Exact error excerpt from the cross-dataset run:

```text
Error: ANNError: DiskANN(InvalidFileFormatError)

Vector file '/Users/xiaoweijiang/Documents/diskann/bigann10Mdatasets/wikipedia_cohere/wikipedia_base.bin.crop_nb_10000000' has invalid format: size 8725200896 bytes doesn't match expected size of 107520000008 bytes based on header (35000000 vectors of dimension 768) -- (/private/tmp/diskann-pq-kmeans-bfs-cache-router/diskann-providers/src/utils/sampling.rs:149)
```

## Caveats

- The cache hit metric remains 0.0% in these outputs. Under this path and current LFS/status situation, the present stats do not count cached hits meaningfully for this route, so use mean hops and mean IOs as the operational cache/search-traversal signal.
- `top-k` is code-path and filename provenance rather than an explicit JSON config flag; the embedded result JSON still reports router type `pq_kmeans`.
- `outputs/` scratch and other untracked run artifacts could not be removed during cleanup because the approval service did not allow the removal action.
- `test_data` appeared dirty only under the disabled LFS filter workflow; the hydrated contents match the LFS pointers, so this is a working-tree/LFS-filter artifact rather than an experiment-output change.
- The retained configs cover the router build, baseline, msp8, and msp16 runs. The msp4 top-k output JSON/log are retained, but the generated msp4 config file is not present in this bundle.
- The baseline rerun JSON and `search_baseline_rerun.stdout.log` are retained in the bundle.
- The msp8 top-k rerun log includes a non-quiet AArch64 warning; the commands above are reproduction commands using `--quiet`, so retained log verbosity may differ.

## Verification

- `cargo test -p diskann-disk pq_kmeans_router`: exit code 0; key output reported 3 passed, 0 failed, 225 filtered out.
- `cargo check -p diskann-benchmark --features disk-index`: exit code 0; key output reported `Finished dev profile` in 1.48s.
- Read-only summary check after this update should find the final `msp8_topk_rerun` metrics row in this file.

## Output files

- Build JSON: `experiment_bundles/msturingann-10m-l200-r100-pq-kmeans-startpoint-20260730/results/build_pq_kmeans_router_output.json`
- Baseline initial JSON: `experiment_bundles/msturingann-10m-l200-r100-pq-kmeans-startpoint-20260730/results/search_baseline_output.json`
- Baseline rerun JSON: `experiment_bundles/msturingann-10m-l200-r100-pq-kmeans-startpoint-20260730/results/search_baseline_rerun_output.json`
- PQ-kmeans msp4 top-k JSON: `experiment_bundles/msturingann-10m-l200-r100-pq-kmeans-startpoint-20260730/results/search_pq_kmeans_msp4_topk_output.json`
- PQ-kmeans msp8 top-k initial JSON: `experiment_bundles/msturingann-10m-l200-r100-pq-kmeans-startpoint-20260730/results/search_pq_kmeans_msp8_topk_output.json`
- PQ-kmeans msp8 top-k rerun JSON: `experiment_bundles/msturingann-10m-l200-r100-pq-kmeans-startpoint-20260730/results/search_pq_kmeans_msp8_topk_rerun_output.json`
- PQ-kmeans msp16 top-k JSON: `experiment_bundles/msturingann-10m-l200-r100-pq-kmeans-startpoint-20260730/results/search_pq_kmeans_msp16_topk_output.json`
- Build log: `experiment_bundles/msturingann-10m-l200-r100-pq-kmeans-startpoint-20260730/logs/build_pq_kmeans_router.stdout.log`
- Baseline initial log: `experiment_bundles/msturingann-10m-l200-r100-pq-kmeans-startpoint-20260730/logs/search_baseline.stdout.log`
- PQ-kmeans msp4 top-k log: `experiment_bundles/msturingann-10m-l200-r100-pq-kmeans-startpoint-20260730/logs/search_pq_kmeans_msp4_topk.stdout.log`
- PQ-kmeans msp8 top-k initial log: `experiment_bundles/msturingann-10m-l200-r100-pq-kmeans-startpoint-20260730/logs/search_pq_kmeans_msp8_topk.stdout.log`
- PQ-kmeans msp8 top-k rerun log: `experiment_bundles/msturingann-10m-l200-r100-pq-kmeans-startpoint-20260730/logs/search_pq_kmeans_msp8_topk_rerun.stdout.log`
- PQ-kmeans msp16 top-k log: `experiment_bundles/msturingann-10m-l200-r100-pq-kmeans-startpoint-20260730/logs/search_pq_kmeans_msp16_topk.stdout.log`
