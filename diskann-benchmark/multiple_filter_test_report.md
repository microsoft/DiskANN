# Multiple-Filter Search Test Report (corrected: set-membership labels)

DiskANN filtered ANN search: **multihop** (hard filter) vs **beta** (soft filter), over 9 metadata-filter predicates, plus a **live per-node filter** variant that measures the real query-time filter cost (section 8). Labels use a **set-membership** model (each token is a boolean membership label), which correctly handles **multi-valued** attributes.

_Generated: 2026-07-15 19:42_

## 0. Correction note (vs earlier version)

The source labels are positional (`GeoLocationID_a, GeoLocationID_b, EN-US, ...`) and **geo is multi-valued**: 27.9% of the 9,996,160 lines carry more than one GeoLocationID (up to 49), and token0 is always *a* GeoLocationID but not the only one.

The earlier encoding modeled only **token0** as a single-valued string field `geo`, so a predicate like `geo = GeoLocationID_9` matched only docs where 9 was the **first** token (1,357 docs), missing docs tagged with 9 elsewhere. The corrected model treats **every token as a boolean membership label**, so `contains GeoLocationID_9` matches all 363,817 docs. Effects are confined to the geo cases (S5-S8); market cases (S1-S4, S9) were already boolean and are unchanged.

| Case | old "token0-geo" selectivity | corrected "contains" selectivity |
|---|---:|---:|
| S5 GeoLocationID_190 | 55.631% | 60.1133% |
| S6 GeoLocationID_90 | 11.303% | 12.5768% |
| S7 GeoLocationID_9 | 0.014% | 3.6396% |
| S8 GeoLocationID_119 | 0.790% | 1.1682% |

Recall impact of the correction (multihop, recall@150):
- **S7 (geo 9): 0.49 -> 0.88 @L150, 0.77 -> 0.97 @L1000.** The earlier "collapse" was an encoding artifact (ultra-sparse 1,357-doc set); the true "tagged-with-9" set (363k) is dense and reachable.
- S5/S6/S8 grew slightly in selectivity with little recall change (already broad enough).

## 1. Environment & dataset

| | |
|---|---|
| Host | Windows, 16 logical cores |
| Points | 9,996,160 | 
| Dimensions | 64 (int8 source, converted to float32 for build) |
| Queries | first 1,000 embeddings |
| Distinct labels | 596 (548 GeoLocationIDs + 48 market codes) |

## 2. Index build (once, reused)

float32, distance squared_l2, max_degree 64, l_build 100, alpha 1.2, medoid start, 16 build threads; build ~400 s; saved as `idxsave_full` and loaded for every search job.

## 3. Methodology

- **Labels (set-membership):** each line token -> boolean field `{"doc_id":i,"GeoLocationID_190":true,"EN-US":true,...}`. A label matches regardless of position. Predicates use `{"<label>":{"$eq":true}}` composed with `$and`/`$or`.
- Full-vocabulary file `data_labels_set.jsonl` (all 596 labels) supports *any* query. For the 9 test cases the referenced-label subset `data_labels_min.jsonl` (11 labels) yields **identical** match-sets/recall/latency and is used here for speed.
- **Filtered groundtruth** via `compute_groundtruth` (shared evaluator with the benchmark), true top-150 among matching docs, `recall_at=150`.
- **Search:** k=150, L in {150,200,300,500,1000}, 3 reps, single thread. Beta uses `beta=0.1`. Recall = recall@150 vs the same filtered GT for both methods.

## 4. Filter cases & (corrected) selectivity

| Case | Predicate | Matches | Selectivity |
|---|---|---:|---:|
| S1 | EN-US | ~6,636,960 | 66.3951% |
| S2 | EN-CA AND FR-CA | ~759,678 | 7.5997% |
| S3 | EN-CA OR FR-CA | ~1,467,686 | 14.6825% |
| S4 | (EN-CA AND FR-CA) OR (ES-MX AND ES-AR) | ~1,003,045 | 10.0343% |
| S5 | GeoLocationID_190 | ~6,009,022 | 60.1133% |
| S6 | GeoLocationID_90 | ~1,257,197 | 12.5768% |
| S7 | GeoLocationID_9 | ~363,820 | 3.6396% |
| S8 | GeoLocationID_119 | ~116,775 | 1.1682% |
| S9 | GeoLocationID_4079 AND GeoLocationID_4092 | ~566,852 | 5.6707% |

## 5. Results (latency in ms, single thread, k=150)

### 5.1 Multihop (hard filter)

| Case | L | recall | QPS | mean | p90 | p99 | p99.9 |
|---|---:|---:|---:|---:|---:|---:|---:|
| S1 | 150 | 0.9710 | 319 | 3.15 | 6.49 | 10.19 | 13.70 |
| S1 | 200 | 0.9779 | 260 | 3.85 | 7.75 | 12.97 | 18.88 |
| S1 | 300 | 0.9847 | 196 | 5.10 | 10.51 | 17.17 | 22.23 |
| S1 | 500 | 0.9909 | 125 | 8.00 | 16.44 | 27.05 | 45.60 |
| S1 | 1000 | 0.9947 | 77 | 12.93 | 26.48 | 40.48 | 52.05 |
| S2 | 150 | 0.9458 | 230 | 4.35 | 6.66 | 8.50 | 11.10 |
| S2 | 200 | 0.9590 | 187 | 5.34 | 8.15 | 10.12 | 11.69 |
| S2 | 300 | 0.9725 | 137 | 7.27 | 11.10 | 14.03 | 16.55 |
| S2 | 500 | 0.9830 | 89 | 11.23 | 17.14 | 20.85 | 26.02 |
| S2 | 1000 | 0.9901 | 48 | 20.84 | 31.21 | 39.21 | 55.81 |
| S3 | 150 | 0.9503 | 229 | 4.37 | 7.24 | 10.06 | 11.57 |
| S3 | 200 | 0.9607 | 183 | 5.47 | 9.15 | 12.59 | 14.47 |
| S3 | 300 | 0.9731 | 131 | 7.62 | 12.73 | 16.96 | 19.41 |
| S3 | 500 | 0.9825 | 86 | 11.66 | 19.67 | 26.19 | 31.93 |
| S3 | 1000 | 0.9887 | 46 | 21.58 | 35.37 | 45.76 | 52.36 |
| S4 | 150 | 0.9534 | 222 | 4.50 | 7.24 | 9.72 | 12.63 |
| S4 | 200 | 0.9634 | 179 | 5.60 | 9.00 | 12.14 | 13.69 |
| S4 | 300 | 0.9758 | 127 | 7.88 | 12.83 | 17.28 | 22.73 |
| S4 | 500 | 0.9847 | 83 | 12.07 | 19.51 | 26.43 | 31.67 |
| S4 | 1000 | 0.9912 | 44 | 22.64 | 36.22 | 47.26 | 51.65 |
| S5 | 150 | 0.9724 | 280 | 3.57 | 6.96 | 10.05 | 13.63 |
| S5 | 200 | 0.9791 | 230 | 4.34 | 8.57 | 12.55 | 15.39 |
| S5 | 300 | 0.9862 | 170 | 5.87 | 11.40 | 17.08 | 20.15 |
| S5 | 500 | 0.9925 | 111 | 8.97 | 17.89 | 26.25 | 31.36 |
| S5 | 1000 | 0.9961 | 65 | 15.48 | 31.03 | 45.38 | 56.29 |
| S6 | 150 | 0.8707 | 170 | 5.87 | 8.30 | 9.74 | 11.13 |
| S6 | 200 | 0.8973 | 137 | 7.27 | 10.32 | 12.16 | 13.81 |
| S6 | 300 | 0.9264 | 99 | 10.13 | 14.35 | 16.64 | 21.54 |
| S6 | 500 | 0.9504 | 64 | 15.51 | 21.95 | 25.34 | 30.74 |
| S6 | 1000 | 0.9690 | 35 | 28.47 | 39.62 | 46.58 | 58.52 |
| S7 | 150 | 0.8835 | 150 | 6.64 | 8.85 | 10.25 | 12.43 |
| S7 | 200 | 0.9073 | 119 | 8.42 | 11.20 | 12.71 | 15.98 |
| S7 | 300 | 0.9320 | 83 | 12.09 | 16.02 | 19.73 | 29.56 |
| S7 | 500 | 0.9539 | 54 | 18.65 | 24.42 | 27.92 | 33.29 |
| S7 | 1000 | 0.9698 | 29 | 34.84 | 44.62 | 50.24 | 72.93 |
| S8 | 150 | 0.8337 | 146 | 6.84 | 8.63 | 10.12 | 11.60 |
| S8 | 200 | 0.8615 | 119 | 8.43 | 10.64 | 12.30 | 14.91 |
| S8 | 300 | 0.8951 | 86 | 11.63 | 14.68 | 16.29 | 20.25 |
| S8 | 500 | 0.9244 | 54 | 18.42 | 23.23 | 26.88 | 33.14 |
| S8 | 1000 | 0.9509 | 30 | 33.60 | 41.09 | 45.45 | 51.38 |
| S9 | 150 | 0.9189 | 220 | 4.55 | 6.68 | 8.05 | 9.42 |
| S9 | 200 | 0.9380 | 173 | 5.79 | 8.52 | 10.21 | 13.13 |
| S9 | 300 | 0.9594 | 125 | 8.00 | 11.89 | 14.13 | 17.20 |
| S9 | 500 | 0.9763 | 82 | 12.19 | 17.92 | 21.32 | 25.72 |
| S9 | 1000 | 0.9869 | 45 | 22.37 | 32.75 | 38.50 | 47.65 |

### 5.2 Beta (soft filter, beta=0.1)

| Case | L | recall | QPS | mean | p90 | p99 | p99.9 |
|---|---:|---:|---:|---:|---:|---:|---:|
| S1 | 150 | 0.8387 | 1481 | 0.67 | 1.09 | 1.51 | 2.22 |
| S1 | 200 | 0.8534 | 1183 | 0.84 | 1.34 | 1.88 | 2.42 |
| S1 | 300 | 0.8659 | 853 | 1.17 | 1.88 | 2.61 | 3.29 |
| S1 | 500 | 0.8766 | 548 | 1.83 | 2.93 | 4.01 | 5.10 |
| S1 | 1000 | 0.8843 | 287 | 3.48 | 5.61 | 7.53 | 10.02 |
| S2 | 150 | 0.6581 | 1157 | 0.86 | 1.24 | 1.68 | 2.12 |
| S2 | 200 | 0.6838 | 916 | 1.09 | 1.58 | 2.26 | 2.99 |
| S2 | 300 | 0.7144 | 651 | 1.54 | 2.27 | 3.16 | 9.96 |
| S2 | 500 | 0.7391 | 430 | 2.33 | 3.36 | 4.43 | 6.36 |
| S2 | 1000 | 0.7609 | 235 | 4.26 | 5.97 | 7.31 | 9.30 |
| S3 | 150 | 0.6789 | 1200 | 0.83 | 1.27 | 1.68 | 2.28 |
| S3 | 200 | 0.7037 | 990 | 1.01 | 1.53 | 1.99 | 2.46 |
| S3 | 300 | 0.7311 | 708 | 1.41 | 2.14 | 2.72 | 3.42 |
| S3 | 500 | 0.7566 | 455 | 2.20 | 3.29 | 4.37 | 5.62 |
| S3 | 1000 | 0.7751 | 239 | 4.18 | 6.10 | 7.49 | 8.77 |
| S4 | 150 | 0.6699 | 1133 | 0.88 | 1.32 | 1.80 | 2.52 |
| S4 | 200 | 0.6935 | 924 | 1.08 | 1.61 | 2.10 | 2.60 |
| S4 | 300 | 0.7235 | 679 | 1.47 | 2.20 | 2.84 | 3.36 |
| S4 | 500 | 0.7493 | 437 | 2.29 | 3.37 | 4.26 | 5.42 |
| S4 | 1000 | 0.7711 | 229 | 4.36 | 6.25 | 7.88 | 17.94 |
| S5 | 150 | 0.7996 | 1414 | 0.71 | 1.15 | 1.54 | 2.07 |
| S5 | 200 | 0.8149 | 1124 | 0.89 | 1.44 | 1.93 | 2.35 |
| S5 | 300 | 0.8311 | 817 | 1.22 | 1.99 | 2.68 | 3.20 |
| S5 | 500 | 0.8438 | 526 | 1.90 | 3.08 | 3.99 | 4.73 |
| S5 | 1000 | 0.8535 | 275 | 3.63 | 5.88 | 7.59 | 9.07 |
| S6 | 150 | 0.4630 | 852 | 1.17 | 1.55 | 1.93 | 2.24 |
| S6 | 200 | 0.4962 | 678 | 1.48 | 1.94 | 2.39 | 3.04 |
| S6 | 300 | 0.5385 | 489 | 2.04 | 2.67 | 3.38 | 4.05 |
| S6 | 500 | 0.5848 | 319 | 3.13 | 4.10 | 4.83 | 5.66 |
| S6 | 1000 | 0.6399 | 168 | 5.95 | 7.68 | 9.11 | 19.75 |
| S7 | 150 | 0.3884 | 785 | 1.27 | 1.62 | 2.08 | 2.43 |
| S7 | 200 | 0.4185 | 630 | 1.59 | 2.01 | 2.54 | 3.21 |
| S7 | 300 | 0.4653 | 459 | 2.18 | 2.80 | 3.47 | 4.32 |
| S7 | 500 | 0.5234 | 296 | 3.37 | 4.34 | 5.17 | 6.30 |
| S7 | 1000 | 0.5776 | 160 | 6.26 | 8.05 | 10.00 | 13.25 |
| S8 | 150 | 0.3774 | 730 | 1.37 | 1.63 | 2.13 | 3.39 |
| S8 | 200 | 0.4114 | 566 | 1.77 | 2.15 | 2.73 | 3.33 |
| S8 | 300 | 0.4631 | 402 | 2.49 | 3.07 | 4.18 | 10.73 |
| S8 | 500 | 0.5132 | 268 | 3.73 | 4.52 | 5.63 | 7.08 |
| S8 | 1000 | 0.5680 | 149 | 6.71 | 8.09 | 9.51 | 11.02 |
| S9 | 150 | 0.5580 | 1103 | 0.91 | 1.31 | 1.78 | 2.47 |
| S9 | 200 | 0.5913 | 904 | 1.11 | 1.58 | 2.02 | 2.49 |
| S9 | 300 | 0.6351 | 628 | 1.59 | 2.30 | 3.00 | 3.97 |
| S9 | 500 | 0.6786 | 412 | 2.43 | 3.45 | 4.35 | 5.18 |
| S9 | 1000 | 0.7213 | 223 | 4.49 | 6.27 | 7.64 | 8.89 |

## 6. Recall side-by-side (multihop vs beta)

| Case (sel.) | L=150 MH / Beta | L=1000 MH / Beta |
|---|---|---|
| S1 (66.40%) | 0.971 / 0.839 | 0.995 / 0.884 |
| S2 (7.60%) | 0.946 / 0.658 | 0.990 / 0.761 |
| S3 (14.68%) | 0.950 / 0.679 | 0.989 / 0.775 |
| S4 (10.03%) | 0.953 / 0.670 | 0.991 / 0.771 |
| S5 (60.11%) | 0.972 / 0.800 | 0.996 / 0.853 |
| S6 (12.58%) | 0.871 / 0.463 | 0.969 / 0.640 |
| S7 (3.64%) | 0.883 / 0.388 | 0.970 / 0.578 |
| S8 (1.17%) | 0.834 / 0.377 | 0.951 / 0.568 |
| S9 (5.67%) | 0.919 / 0.558 | 0.987 / 0.721 |

## 7. Observations

- **Multihop reaches high recall across all cases now** (>=0.83 @L150, >=0.95 @L1000), including the geo cases, because corrected "contains" sets are dense enough to traverse.
- **Beta is much faster but recall-limited** and collapses on selective filters; recall plateaus with L (soft re-rank keeps globally-close non-matches).
- **Selectivity drives cost/recall** more than AND/OR structure; the previous "S7 is unsolvable" conclusion was an artifact of mis-encoding a multi-valued attribute as single-valued.

## 8. Live per-node filter vs precomputed bitmap

Both multihop runs above use a **precomputed whole-corpus bitmap** for the filter: the match set of each query is computed offline (a full-dataset scan) and `is_match(id)` is an O(1) bitmap lookup. That work is **not** included in the reported search latency.

To measure the **real** filter cost, a new search type `topk-multihop-live-filter` evaluates the predicate **live, per visited node**: each vector's labels are stored as a roaring set of integer attribute-ids (built once, like an index), the query predicate is encoded once to integer terminals, and `is_match(id)` reads the node's set (lock-free) and evaluates the AND/OR expression via `contains`. No FFI, no global posting list. Only AND/OR + equality are supported (NOT/relational rejected).

Results at **L=150, k=150, single thread** (recall is identical because traversal is the same; only how `is_match` is computed differs):

| Case (sel.) | recall (live == bitmap) | mean live / bitmap | p90 live | p99 live / bitmap | p99.9 live | live QPS |
|---|---:|---|---:|---|---:|---:|
| S1 (66.4%) | 0.9710 | 20.9 / 3.2 ms | 46.1 ms | 93.9 / 10.2 ms | 139.5 ms | 48 |
| S2 (7.6%) | 0.9458 | 48.1 / 4.3 ms | 84.7 ms | 124.7 / 8.5 ms | 221.8 ms | 21 |
| S3 (14.7%) | 0.9503 | 54.1 / 4.4 ms | 99.9 ms | 150.4 / 10.1 ms | 185.3 ms | 18 |
| S4 (10.0%) | 0.9534 | 83.1 / 4.5 ms | 155.7 ms | 214.9 / 9.7 ms | 238.0 ms | 12 |
| S5 (60.1%) | 0.9724 | 24.7 / 3.6 ms | 54.4 ms | 99.5 / 10.1 ms | 132.4 ms | 41 |
| S6 (12.6%) | 0.8707 | 69.8 / 5.9 ms | 108.0 ms | 133.8 / 9.7 ms | 145.9 ms | 14 |
| S7 (3.6%) | 0.8835 | 88.9 / 6.6 ms | 123.5 ms | 146.4 / 10.3 ms | 160.9 ms | 11 |
| S8 (1.2%) | 0.8337 | 98.3 / 6.8 ms | 130.3 ms | 145.9 / 10.1 ms | 181.1 ms | 10 |
| S9 (5.7%) | 0.9189 | 55.6 / 4.5 ms | 97.4 ms | 124.8 / 8.0 ms | 150.2 ms | 18 |

- **Correctness:** live recall equals the bitmap recall to 5 decimals for every case (recall diff = 0.00000) — the live match is functionally identical.
- **The hidden cost is large:** live mean latency is ~7-18x the bitmap number (e.g. S4 4.5 -> 83 ms, S8 6.8 -> 98 ms) and p99 is ~10-22x higher. The precomputed-bitmap benchmark amortized predicate evaluation into an untimed offline pass, so its latency only reflected O(1) lookups.
- **Cost scales with expression complexity and traversal size:** the AND/OR combo (S4) and selective geo filters (S6-S8, more hops through non-matching regions -> more `is_match` calls) are the most expensive; broad single-term filters (S1, S5) are cheapest.
- This is the production-relevant number: a real system evaluates the filter at query time (per node), not from a precomputed whole-corpus answer.

## 9. Artifacts (`Q:\test6\filtered_test2\bench\full\`)

- Labels: `data_labels_set.jsonl` (596 labels, general), `data_labels_min.jsonl` (11, fast)
- Predicates: `predmin_S1..S9.jsonl` (also `predset_S1..S9.jsonl`)
- Groundtruth: `gtmin_S1..S9.bin`
- Runbooks: `runbook_setmin.json` (multihop), `runbook_beta_setmin.json` (beta), `runbook_livefilter.json` (live per-node)
- Outputs: `out_setmin.json`, `out_beta_setmin.json`, `out_livefilter.json` (+ `out_live_S8.json`, `out_live_S9.json`)
- Encoders: `gen_setmembership.py` (full), `gen_setmin.py` (minimal)
- Live-filter code: `diskann-label-filter/src/live_filter.rs` (InlineAttributeIndex / FrozenAttributeIndex + QueryLabelProvider); benchmark search-type `topk-multihop-live-filter`
- Index (reused): `idxsave_full`(+`.data`)

_Note: an earlier version of this report used a single-valued `geo` string field; it is superseded by the set-membership results above._