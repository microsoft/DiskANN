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
- Full-vocabulary file `data_labels_set.jsonl` (all 596 labels) is the authoritative source. For the 9 test cases, `data_labels_min.jsonl` is an exact 11-field projection and remains available only as a setup-speed optimization.
- **Filtered groundtruth** via `compute_groundtruth` (shared evaluator with the benchmark), true top-150 among matching docs, `recall_at=150`. The canonical `gtset_S1..S9.bin` files were regenerated from `data_labels_set.jsonl`; all nine are byte-identical to the earlier `gtmin` files.
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

## 8.1 CSR optimization of the live per-node filter

The live cost in section 8 is dominated by the roaring representation. `is_match` does a `HashMap<u32, RoaringTreemap>` probe into the 10M-entry index (random access + heap pointer-chase) plus, per equality terminal, a `RoaringTreemap::contains` (`BTreeMap` lookup → `RoaringBitmap` container binary-search). A criterion microbenchmark isolating just `is_match` (1M docs, 50k random probes) shows this is **14-20x slower** than a flat **CSR** layout that stores each node's sorted attribute-ids contiguously (`offsets: Vec<u32>` + `values: Vec<u32>`) and answers each terminal with a `binary_search` over the node's one contiguous row:

| predicate | roaring treemap | flat CSR | posting bitmap/attr |
|---|---:|---:|---:|
| 1 term | 277 ns | 13.6 ns (20.4x) | 8.8 ns |
| AND-2 | 298 ns | 20.7 ns (14.4x) | 23.6 ns |
| (a AND b) OR (c AND d) | 596 ns | 36.0 ns (16.6x) | 74.2 ns |

(microbench: `cargo bench -p diskann-label-filter --bench main -- live_filter`; posting = one doc-id bitmap per attribute, wins only for a single broad term but degrades with term count since each terminal is a separate random bitmap probe.)

A CSR-backed provider (`InlineAttributeIndexCsr` / `FrozenAttributeIndexCsr`) was added and exposed as the benchmark search type `topk-multihop-live-filter-csr`. It reuses the same `AttributeEncoder`/`EncodedFilterExpr`, so predicate semantics and errors are identical. End-to-end (same index, k=150, L=150, single thread, 1000 queries) it is a drop-in replacement — **identical traversal** (same avg cmps/hops) and **identical recall** — differing only in how `is_match` is computed:

| Case | impl | recall | mean | p90 | p99 | QPS |
|---|---|---:|---:|---:|---:|---:|
| S1 (66.4%) | live (roaring) | 0.9710 | 18.4 ms | 39.6 | 85.1 | 54 |
| S1 | **live-csr** | 0.9710 | **5.5 ms** | 11.8 | 22.4 | 183 |
| S4 (10.0%) | live (roaring) | 0.9534 | 76.7 ms | 142.1 | 197.7 | 13 |
| S4 | **live-csr** | 0.9534 | **17.4 ms** | 31.8 | 43.7 | 57 |

CSR gives a **3.4x (S1) - 4.4x (S4)** end-to-end latency/QPS improvement at equal recall, closing ~82-85% of the live-vs-precomputed-bitmap gap from section 8 while still evaluating the predicate live per node. The end-to-end factor is smaller than the ~15-20x on `is_match` alone because the ~13k vector-distance comparisons per query are unchanged and now dominate the remaining latency.

## 8.2 Posting-list + materialized bitmap, and the selectivity crossover

CSR still evaluates the predicate per visited node. The Lucene/Milvus/FAISS approach instead builds the query's whole match set **once** (roaring `AND`/`OR` over per-attribute posting lists) into a dense bitset, then answers each node with an `O(1)` bit test. `topk-multihop-live-filter-bitmap` (`InlineAttributeIndexPosting` / `MaterializedBitmapProvider`) implements this **live** — the match-set materialization is done lazily on the first `is_match`, so its cost is counted in query latency, not an offline pass. The benchmark reconstructs these lazy providers for every repetition and search-L run, preventing the materialized state from being reused across measurements.

Three-way live comparison (same index, k=150, L=150, single thread, 1000 queries; recall identical within each case):

| Case (sel.) | avg hops | live (roaring) | live-csr | live-bitmap |
|---|---:|---:|---:|---:|
| S1 (66.4%) | 1087 | 21.99 ms | **6.70 ms** | 51.44 ms |
| S4 (10.0%) | 2492 | 79.74 ms | 17.98 ms | **17.18 ms** |
| S8 (1.17%) | 4439 | 93.45 ms | 21.90 ms | **7.05 ms** |

(p99, ms: S1 100.0 / 28.5 / 74.0; S4 207.6 / 47.4 / 26.5; S8 135.5 / 32.1 / 10.2.)

There is a clean **selectivity crossover**:

- The bitmap's per-query cost is dominated by **materializing the match set** (proportional to the number of matching ids). For a broad filter (S1, ~6.6M matches) this densify dominates and bitmap is the **worst** option (51 ms); for a selective filter (S8, ~117k matches) it is cheap and the `O(1)` per-node test wins decisively (**7.05 ms — 3.1x faster than CSR, 13x faster than roaring**).
- Selective filters also **traverse more** (S8 avg hops 4439 vs S1 1087), so `is_match` is called far more often, further rewarding the bitmap's `O(1)` test / CSR's cheap row-scan and penalizing the roaring path.
- CSR has **no per-query build** and a tiny per-node cost, so it is the **robust default**: 3.3-4.4x over roaring at every selectivity, best for broad, ~tied for mid.

**Takeaway for live query latency:** no single per-node representation wins everywhere. A **selectivity-adaptive** provider is best — the posting-list index already knows `|match set|` cheaply (roaring cardinality, before any densify), so it can pick **bitmap when selective, CSR when broad**, yielding the best cell of each row (S1→CSR 6.70, S4/S8→bitmap 17.18/7.05 ms): up to **13x** over the roaring baseline for selective filters and **3.3x** for broad, all live.

_Caveat: the bitmap build is per distinct query predicate. In this benchmark all queries share one predicate, so a system that caches the match set per predicate would amortize the broad-filter build; the numbers above are the honest cost for a unique-per-query predicate._

## 8.3 Two "win everywhere" methods: adaptive (auto) and bit-sliced

Section 8.2's bitmap/CSR winner flips with selectivity. Two methods remove that weakness. `topk-multihop-live-filter-auto` (`InlineAttributeIndexAuto`) builds the cheap roaring match set once, reads its cardinality for free, then densifies to a bitset when selective or falls back to the CSR row-scan when broad. `topk-multihop-live-filter-bitslice` (`InlineAttributeIndexBitslice`) precomputes **one dense bitset per attribute at index time**, so `is_match` is one `O(1)` bit test per equality terminal with **no** per-query build.

Full live comparison (same index, k=150, L=150, single thread, 1000 queries; recall identical per case). live-roaring from section 8.2 shown for reference:

| Case (sel.) | avg hops | roaring | csr | bitmap | auto | bitslice |
|---|---:|---:|---:|---:|---:|---:|
| S1 (66.4%) | 1087 | 21.99 | 5.64 | 49.39 | 6.27 | **3.18** |
| S4 (10.0%) | 2492 | 79.74 | 18.38 | 17.38 | 17.36 | **6.61** |
| S8 (1.17%) | 4439 | 93.45 | 22.45 | 7.08 | 7.06 | **6.39** |

(mean ms; p99 ms — S1: csr 21.6 / bitmap 56.9 / auto 24.6 / bitslice 11.5; S4: 46.9 / 22.9 / 22.9 / 17.7; S8: 32.3 / 10.5 / 9.9 / 9.5.)

- **Bit-sliced wins outright in every case** (3.18 / 6.61 / 6.39 ms): 1.8-3.5x over CSR and 6.9-14.6x over roaring, fastest mean *and* p99. Because the per-attribute bitsets are built once at index time, a single-term filter costs exactly one `O(1)` bit test per node — it even edges the section-8 precomputed-bitmap "ideal", built legitimately and amortized over all queries. Its cost is **memory**: `num_attributes * ceil(N/64) * 8` bytes (~14 MB for the 11 labels here, ~745 MB for the full 596-label vocabulary, infeasible for high-cardinality labels).
- **Auto is the robust, memory-light alternative**: it dispatched S1->CSR, S4/S8->bitmap and thus **matched the better of {csr, bitmap} in every case** (never the worst) using the free match-set cardinality — no per-attribute bitset matrix required.

**Bottom line for live query latency:** if the label vocabulary is modest, **bit-sliced per-attribute bitsets win across all selectivities**; otherwise **auto** (adaptive bitmap<->CSR) reliably takes the better of the two base methods. Both keep recall identical and evaluate the filter fully live.

## 8.4 Bit-sliced layout and flat DNF query evaluation

### 8.4.1 Attribute-major bit-sliced index

`InlineAttributeIndexBitslice` builds one dense vector-id bitmap for every encoded
attribute. If `a` is an attribute id and `v` is a vector id, the membership test is:

```text
word = bitsets[a][v / 64]
matches = ((word >> (v % 64)) & 1) != 0
```

The storage is therefore an attribute-major Boolean matrix:

```text
attribute 0: [bit for node 0, bit for node 1, ...]
attribute 1: [bit for node 0, bit for node 1, ...]
...
```

Each equality terminal requires one indexed `u64` load plus a shift/mask. The immutable
bitsets are shared by every query provider through `Arc`; there is no per-query result bitmap,
lock, or allocation in `is_match`. Storage is:

```text
num_attributes * ceil(num_vectors / 64) * 8 bytes
```

For 9,996,160 vectors and all 596 labels this is 744,713,920 bytes (710.21 MiB). A
single-attribute query touches only one 1.19 MiB slice even though the complete matrix is much
larger.

### 8.4.2 Recursive encoded AST

The original Bitslice provider encodes each comparison to an integer terminal, but preserves the
recursive query tree:

```text
ASTIdExpr =
    Terminal(attribute_id)
  | And(Vec<ASTIdExpr>)
  | Or(Vec<ASTIdExpr>)
```

For every visited node, an `ASTIdExprVisitor` recursively dispatches on each node in the tree.
`AND` and `OR` short-circuit through `all()` and `any()`, but every terminal still repeats enum
dispatch, attribute-array lookup, and vector-id word/mask calculation.

### 8.4.3 Flat DNF representation

The optimized search type `topk-multihop-live-filter-bitslice-dnf` accepts expressions already in
disjunctive normal form: an OR of AND clauses. A terminal is a one-term clause and a plain AND is
a one-clause expression.

```text
(A AND B) OR (C AND D)

clause_offsets = [0, 2, 4]
attributes     = [A, B, C, D]
```

Clause `i` occupies
`attributes[clause_offsets[i]..clause_offsets[i + 1]]`. The final hot-path query plan is stored
in two contiguous boxed allocations regardless of clause count. The current prototype still
builds the existing recursive encoded AST during query setup and then flattens it; the optimization
targets repeated per-node evaluation rather than JSON/AST construction.

At search time, the provider calculates the vector word and mask once, then loops over clauses:

```text
word_index = vector_id / 64
mask = 1 << (vector_id % 64)

for each AND clause:
    if every bitsets[attribute][word_index] contains mask:
        return true
return false
```

The inner AND stops at its first missing attribute and the outer OR stops at its first matching
clause. Single-terminal predicates use a dedicated provider that performs the bit test directly.
Term order is intentionally preserved in this experiment; no cardinality-based reordering is
included.

The DNF compiler rejects trees such as `(A OR B) AND (C OR D)`. These must be normalized before
ANN search, for example:

```text
(A OR B) AND (C OR D)
  -> (A AND C) OR (A AND D) OR (B AND C) OR (B AND D)
```

Normalization belongs outside the hot search path and needs clause/terminal limits because
unrestricted distribution can grow exponentially.

Associatively nested groups such as `A OR (B OR C)` and `A AND (B AND C)` are flattened during
query setup; they do not require distributive normalization.

### 8.4.4 Isolated `is_match` result

Criterion benchmark: 1M nodes, 596-label vocabulary, 50,000 random node probes per iteration.

| Expression | Recursive Bitslice | Flat DNF | Speedup |
|---|---:|---:|---:|
| one terminal | 239.63 us | 118.31 us | **2.03x** |
| `A AND B` | 529.57 us | 385.22 us | **1.37x** |
| `(A AND B) OR (C AND D)` | 1,059.0 us | 553.62 us | **1.91x** |

### 8.4.5 Full ANN result

Full 596-label dataset, k=150, L=150, one thread, 1,000 queries. Latencies are milliseconds.
Recall, comparisons, and hops are exactly identical between recursive and DNF providers.

| Case | Terms | Recursive AVG | DNF AVG | Recursive P99 | DNF P99 | Recursive P99.9 | DNF P99.9 |
|---|---:|---:|---:|---:|---:|---:|---:|
| S1 | 1 | 2.735 | **2.555** | 9.258 | **8.850** | **14.228** | 14.342 |
| S2 | 2 | 4.906 | **4.386** | 10.837 | **9.137** | 18.209 | **10.697** |
| S3 | 2 | 4.841 | **4.354** | 12.130 | **10.613** | 13.986 | **13.443** |
| S4 | 4 | 6.789 | **5.148** | 17.297 | **13.269** | 20.875 | **15.500** |
| S5 | 1 | 3.223 | **2.985** | 9.903 | **8.618** | 14.714 | **13.835** |
| S6 | 1 | 5.563 | **4.969** | 10.030 | **8.486** | 12.275 | **10.374** |
| S7 | 1 | 6.362 | **5.660** | 9.969 | **9.057** | 12.544 | **11.061** |
| S8 | 1 | 6.600 | **5.816** | 9.590 | **8.555** | 12.954 | **10.704** |
| S9 | 2 | 4.905 | **4.612** | 8.988 | **8.776** | **11.895** | 12.150 |

Across S1-S9, flat DNF improves mean latency by **1.125x geometric mean** (6.0-24.2%)
and P99 by **1.137x geometric mean**. The four-terminal S4 expression benefits most:
24.2% lower mean and 23.3% lower P99. P99.9 improves in seven cases; the small S1/S9
regressions are within single-run tail noise.

The uniform-random single-attribute workload also improves from 3.300 to 2.856 ms mean
(13.45%), 10.242 to 9.112 ms P99 (11.03%), and 13.187 to 11.312 ms P99.9 (14.22%).

### 8.4.6 Fresh Roaring-materialized bitmap versus Bitslice-DNF

After merging `origin/main`, both providers were rerun in one counterbalanced S1-S9 runbook using
the full 596-label dataset, canonical `gtset` ground truth, k=150, L=150, one thread, 1,000
queries, and three repetitions. Values are milliseconds averaged across the three repetitions.
`topk-multihop-live-filter-bitmap` rebuilt fresh query providers for every repetition and L; its
lazy Roaring set algebra and dense result-bitmap construction were therefore included in query
latency, with no cross-query result cache.

| Case | Bitmap AVG | DNF AVG | AVG speedup | Bitmap P99 | DNF P99 | P99 speedup | Bitmap P99.9 | DNF P99.9 | P99.9 speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S1 | 50.914 | **2.130** | 23.91x | 60.307 | **7.321** | 8.24x | 76.567 | **11.090** | 6.90x |
| S2 | 11.109 | **4.121** | 2.70x | 14.936 | **8.976** | 1.66x | 18.371 | **10.849** | 1.69x |
| S3 | 16.508 | **4.221** | 3.91x | 22.508 | **10.410** | 2.16x | 30.520 | **13.176** | 2.32x |
| S4 | 17.757 | **5.205** | 3.41x | 23.850 | **13.216** | 1.80x | 34.291 | **17.202** | 1.99x |
| S5 | 48.190 | **2.723** | 17.70x | 58.693 | **8.262** | 7.10x | 73.576 | **10.871** | 6.77x |
| S6 | 15.432 | **4.468** | 3.45x | 19.774 | **7.996** | 2.47x | 29.565 | **10.705** | 2.76x |
| S7 | 8.757 | **5.206** | 1.68x | 12.453 | **8.317** | 1.50x | 20.782 | **10.945** | 1.90x |
| S8 | 6.361 | **5.135** | 1.24x | 9.002 | **7.698** | 1.17x | 11.031 | **8.704** | 1.27x |
| S9 | 9.580 | **4.311** | 2.22x | 13.963 | **7.895** | 1.77x | 17.651 | **9.942** | 1.78x |

Recall, comparisons, and hops were exactly identical for every pair. Bitslice-DNF won all 27
latency comparisons. Its geometric-mean speedups were **3.967x AVG**, **2.426x P99**, and
**2.534x P99.9**. The largest gaps were the broad single-terminal S1 and S5 cases, where iterating
millions of Roaring matches during densification dominates. The closest case was selective S8,
where cheap materialization plus frequent membership tests narrowed the DNF advantage to 1.24x
AVG.

### 8.4.7 Fixed-L InlineFilterSearch versus multihop with Bitslice-DNF

A dedicated `topk-inline-live-filter-bitslice-dnf` benchmark mode was added so InlineFilterSearch
and multihop can use the exact same attribute index, encoded DNF providers, queries, and ground
truth. The initial comparison intentionally disabled Adaptive-L and used the same fixed k=150,
L=150, one thread, 1,000 queries, full 596-label metadata, canonical `gtset` ground truth, and
three repetitions. Values are milliseconds averaged across repetitions.

| Case | Inline AVG | Multi AVG | Speedup | Inline P99 | Multi P99 | Inline P99.9 | Multi P99.9 | Inline recall | Multi recall | Recall delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S1 | **0.432** | 2.324 | 5.38x | **1.189** | 7.921 | **1.547** | 11.730 | 0.76114 | **0.97104** | -0.20990 |
| S2 | **0.466** | 4.390 | 9.42x | **1.210** | 9.492 | **1.536** | 12.295 | 0.41239 | **0.94577** | -0.53338 |
| S3 | **0.465** | 4.400 | 9.45x | **1.207** | 10.703 | **1.609** | 12.488 | 0.48748 | **0.95031** | -0.46283 |
| S4 | **0.554** | 5.606 | 10.11x | **1.564** | 14.346 | **1.830** | 19.392 | 0.44256 | **0.95338** | -0.51082 |
| S5 | **0.443** | 2.842 | 6.41x | **1.152** | 8.687 | **2.426** | 11.842 | 0.70740 | **0.97237** | -0.26497 |
| S6 | **0.399** | 4.406 | 11.05x | **1.205** | 7.716 | **2.213** | 9.134 | 0.16844 | **0.87069** | -0.70225 |
| S7 | **0.415** | 5.229 | 12.60x | **1.052** | 8.445 | **1.887** | 10.107 | 0.11883 | **0.88348** | -0.76465 |
| S8 | **0.385** | 5.192 | 13.50x | **0.965** | 7.893 | **1.451** | 9.811 | 0.05506 | **0.83374** | -0.77868 |
| S9 | **0.418** | 4.559 | 10.91x | **1.052** | 8.498 | **1.358** | 10.378 | 0.21855 | **0.91891** | -0.70035 |

Fixed-L Inline was **9.512x faster in AVG**, **7.815x in P99**, and **6.717x in P99.9**
by geometric mean, but it was not recall-competitive. Mean recall was only **0.37465**, versus
**0.92219** for multihop, a -0.54754 absolute difference. Inline performed the same
filter-independent traversal in every case (2,330.738 comparisons and 156.407 hops), while
multihop spent 11.9K-18.4K comparisons and 1.1K-4.4K hops routing through rejected nodes to recover
matching neighbors.

The latency advantage therefore represents a different recall point, not an algorithmic win at
equal quality. Adaptive-L or an L sweep is required to compare recall-versus-latency curves and
determine whether Inline can match multihop recall efficiently.

### 8.4.8 Encoded-label library: DNF, AST, and precomputed AST

The new `diskann-label-index` crate was benchmarked directly through three multihop modes:

- `topk-multihop-encoded-bitslice-dnf`: contiguous Bitslice storage plus the flat DNF plan;
- `topk-multihop-encoded-bitslice-ast`: the same Bitslice storage plus recursive AST evaluation;
- `topk-multihop-encoded-bitmap-ast`: Roaring postings plus recursive AST materialization into one
  dense query-result bitmap before timed ANN search.

The full 596-label JSONL was encoded once into a 744,726,578-byte Bitslice file and a
61,892,560-byte Roaring file. One counterbalanced 27-job runbook then covered S1-S9, k=150, L=150,
one thread, 1,000 queries, and three repetitions. Values are milliseconds averaged across
repetitions.

| Case | Bitslice DNF AVG | Bitslice AST AVG | Precomputed AST AVG | DNF P99 | AST P99 | Pre P99 | DNF P99.9 | AST P99.9 | Pre P99.9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S1 | 2.616 | 2.499 | **2.435** | 9.200 | 8.665 | **7.816** | 14.442 | **11.585** | 19.243 |
| S2 | 4.222 | 4.929 | **3.503** | 9.505 | 10.978 | **6.829** | 11.965 | 31.797 | **10.894** |
| S3 | 4.497 | 4.904 | **3.503** | 11.440 | 12.349 | **8.099** | 17.361 | 15.686 | **9.759** |
| S4 | 6.119 | 7.217 | **3.670** | 16.167 | 18.997 | **8.277** | 26.944 | 26.378 | **10.222** |
| S5 | 3.009 | 3.049 | **2.882** | 9.364 | 9.773 | **8.411** | **12.126** | 14.358 | 10.654 |
| S6 | 5.435 | 5.003 | **4.697** | 10.343 | 9.232 | **7.913** | 18.581 | 13.521 | **9.619** |
| S7 | 7.486 | 6.673 | **5.259** | 12.626 | 11.813 | **8.569** | 16.570 | 39.672 | **11.013** |
| S8 | 8.572 | 8.442 | **5.567** | 13.796 | 20.945 | **8.610** | 18.575 | 36.382 | **11.214** |
| S9 | 5.004 | 4.789 | **3.616** | 9.496 | 9.225 | **6.404** | 28.355 | 16.517 | **13.191** |

Recall, comparisons, and hops were exactly identical across all three modes for every case.
Bitslice AST versus flat DNF had a **1.013x AVG latency ratio** (1.3% slower geometric mean), a
**1.066x P99 ratio**, and a **1.180x P99.9 ratio**. This confirms that recursive AST evaluation is
mean-latency competitive for the current one-to-four-terminal workload, while its tail is noisier;
the S2/S7/S8 P99.9 values are dominated by isolated outliers.

Precomputed AST was **1.294x faster in AVG**, **1.417x in P99**, and **1.528x in P99.9** versus
Bitslice DNF by geometric mean. This is the expected warm/precomputed result: Roaring AST
evaluation and dense bitmap construction occur during benchmark setup, so timed traversal pays
only one membership lookup per visited node. Its 2.435-5.567 ms means are in the same range as the
earlier precomputed-bitmap results in section 8.

The new encoded Bitslice-DNF means were 1.218x slower by geometric mean than the immediately prior
Bitslice-DNF rerun in section 8.4.6. Because traversal statistics are identical, this is
run-to-run/system drift plus the new generic provider layer rather than a semantic difference;
the matched within-run AST/DNF/precomputed comparisons are the authoritative result.

### 8.4.9 Including per-query encoding and materialization

The encoded benchmark providers were then changed to compile lazily on their first `is_match`
inside timed ANN search. Each query row owns an independent `OnceLock`, and fresh providers are
created for every repetition and search-L. The timed path therefore includes:

- DNF: label-string lookup and flat clause-plan compilation;
- Bitslice AST: JSON AST parsing, validation, label lookup, and recursive-plan compilation;
- Bitmap AST: all AST costs plus Roaring set algebra, dense allocation/zeroing, and result
  materialization.

The persisted Bitslice/Roaring index load and benchmark predicate-file parsing remain index/setup
costs outside query latency. The same counterbalanced S1-S9 configuration was rerun.

| Case | DNF AVG | AST AVG | Bitmap AST AVG | DNF P99 | AST P99 | Bitmap P99 | DNF P99.9 | AST P99.9 | Bitmap P99.9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S1 | 2.698 | 2.651 | 46.479 | 9.836 | 9.470 | 57.984 | 13.368 | 12.521 | 76.905 |
| S2 | 4.448 | 4.913 | 9.944 | 10.057 | 11.263 | 13.580 | 12.969 | 29.329 | 16.460 |
| S3 | 4.561 | 4.781 | 14.694 | 11.687 | 11.945 | 20.159 | 17.175 | 15.934 | 24.612 |
| S4 | 5.135 | 7.659 | 16.351 | 12.650 | 20.666 | 22.637 | 18.467 | 41.756 | 43.390 |
| S5 | 3.145 | 2.796 | 43.062 | 9.698 | 8.590 | 54.658 | 14.802 | 11.736 | 76.852 |
| S6 | 5.535 | 4.978 | 14.109 | 9.812 | 9.164 | 19.467 | 13.161 | 13.473 | 26.372 |
| S7 | 6.675 | 5.837 | 7.771 | 10.772 | 9.471 | 11.038 | 33.984 | 14.685 | 16.343 |
| S8 | 6.824 | 6.044 | **6.153** | 10.699 | 9.027 | 9.774 | 15.958 | 23.023 | **13.200** |
| S9 | 4.338 | 4.677 | 8.646 | 8.158 | 8.846 | 12.546 | 9.367 | 11.445 | 15.766 |

Recall, comparisons, and hops again matched exactly. Query-inclusive Bitslice AST remained almost
tied with DNF: its geometric-mean AVG ratio was **1.014x**. The DNF/AST compilation costs are small
relative to traversal and system noise; several query-inclusive means are lower than the
setup-excluded run, so subtracting the independent runs is not a reliable microsecond estimate.

Bitmap AST became **3.130x slower than DNF** by geometric-mean AVG once its per-query materialization
was charged. The approximate setup-excluded-to-inclusive mean increase ranged from 0.585 ms for
selective S8 to 44.043 ms for broad S1, averaging 14.675 ms across S1-S9. Its broad-filter
S1/S5 means returned to 46.479/43.062 ms, while selective S8 remained competitive at 6.153 ms.

These results reproduce the previous no-cache Roaring-materialized benchmark: the new Bitmap AST
means are within the same range as section 8.4.6 and are 0.909x its geometric mean. The conclusion
is unchanged: live Bitslice DNF/AST is normally preferable for unique queries, while
materialization can compete for sufficiently selective filters or when its result is reused.

## 8.5 Exact-semantics multihop allocation reuse

After optimizing filter evaluation, the current multihop implementation still created temporary
graph-traversal storage repeatedly:

- every `expand_beam` call constructed a new adjacency-list buffer;
- each search iteration copied the selected two-hop routing IDs into a new `Vec`.

The optimized implementation adds `expand_beam_with_scratch`, allowing the in-memory full-precision
provider to reuse one adjacency list, preallocated to the graph's maximum degree, for all one-hop
and two-hop expansions in a query. Providers with specialized `expand_beam` implementations keep
their original I/O, batching, and accounting behavior. The two-hop call now consumes an iterator
over the already sorted `Neighbor` list instead of allocating a second ID vector.

This change does not alter predicate order, candidate order, distance calculations, queue
operations, or traversal limits. A fresh controlled benchmark compared matched old and optimized
binaries using the same DNF provider, full 596-label dataset, and **three repetitions per S1-S9
case**. Recall, comparisons, and hops were exactly identical.

All values below are milliseconds and are the mean of the three repetitions.

| Case | Old AVG | Optimized AVG | AVG gain | Old P99 | Optimized P99 | P99 gain | Old P99.9 | Optimized P99.9 | P99.9 gain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S1 | 2.527 | **2.350** | 7.02% | 8.423 | **7.985** | 5.20% | 11.729 | **11.180** | 4.68% |
| S2 | 4.296 | **4.280** | 0.39% | **9.070** | 9.253 | -2.02% | **11.903** | 21.286 | -78.83% |
| S3 | 4.236 | **4.101** | 3.18% | 10.304 | **10.141** | 1.58% | 13.183 | **11.758** | 10.81% |
| S4 | 5.020 | **4.940** | 1.60% | 12.571 | **12.383** | 1.50% | **17.176** | 21.160 | -23.20% |
| S5 | 3.001 | **2.765** | 7.84% | 9.489 | **8.469** | 10.75% | 12.923 | **11.643** | 9.91% |
| S6 | 4.871 | **4.714** | 3.22% | 8.388 | **8.336** | 0.62% | 10.307 | **9.315** | 9.63% |
| S7 | 5.547 | **5.413** | 2.42% | 9.126 | **8.796** | 3.61% | 13.213 | **10.430** | 21.06% |
| S8 | 5.588 | **5.467** | 2.15% | **8.179** | 8.236 | -0.70% | 13.912 | **12.383** | 10.99% |
| S9 | 4.438 | **4.254** | 4.16% | 8.510 | **8.032** | 5.62% | 11.920 | **10.219** | 14.27% |

The fresh geometric-mean speedups are **1.0372x** for AVG (3.72%), **1.0307x** for
P99 (3.07%), and **1.0095x** for P99.9 (0.95%). P99.9 is intrinsically noisy with
1,000 queries: the S2 and S4 optimized means are dominated by one repetition at
39.489 ms and 31.415 ms respectively. The repeatable signal is the AVG improvement,
with P99 also improving in seven of nine cases. This is a useful low-risk improvement,
but the larger remaining ceiling is avoiding duplicated predicate checks and unnecessary
distance work rather than removing additional small allocations.

## 8.6 Static dispatch for encoded query providers

The encoded benchmark path previously stored every per-query provider as
`Arc<dyn QueryLabelProvider<u32>>`. That erased the concrete `LazyEncodedQueryProvider` type before
search, forcing an indirect `is_match` call for every visited node and preventing the compiler
from inlining through the lazy query provider into `EncodedLabelQuery`.

`Filtered`, `FilteredAccessor`, and the benchmark `MultiHop` helper are now generic over the label
provider type, with `dyn QueryLabelProvider` retained as the default for heterogeneous callers.
The encoded Bitslice-DNF benchmark keeps `LazyEncodedQueryProvider` concrete end to end.

A fresh controlled comparison used the full 596-label encoded Bitslice file, query-inclusive lazy
DNF compilation, k=L=150, one thread, 1,000 queries, and three repetitions per S1-S9 case.
Recall, distance comparisons, and graph hops were exactly identical.

| Case | Dynamic AVG | Static AVG | AVG speedup | Dynamic P99 | Static P99 | P99 speedup |
|---|---:|---:|---:|---:|---:|---:|
| S1 | 2.545 ms | **2.436 ms** | 1.045x | 8.923 ms | **8.664 ms** | 1.030x |
| S2 | 4.290 ms | **4.035 ms** | 1.063x | 9.236 ms | **8.690 ms** | 1.063x |
| S3 | 4.290 ms | **4.083 ms** | 1.051x | 10.540 ms | **10.098 ms** | 1.044x |
| S4 | 5.098 ms | **4.795 ms** | 1.063x | 12.468 ms | **11.958 ms** | 1.043x |
| S5 | 3.045 ms | **2.924 ms** | 1.042x | 9.679 ms | **9.119 ms** | 1.061x |
| S6 | 5.609 ms | **5.277 ms** | 1.063x | 9.865 ms | **9.279 ms** | 1.063x |
| S7 | 6.590 ms | **6.247 ms** | 1.055x | 10.125 ms | **9.608 ms** | 1.054x |
| S8 | 6.905 ms | **6.425 ms** | 1.075x | 9.993 ms | **9.302 ms** | 1.074x |
| S9 | 4.366 ms | **4.089 ms** | 1.068x | 8.047 ms | **7.624 ms** | 1.055x |

The S1-S9 geometric-mean speedups are **1.0581x** for AVG (5.81%), **1.0541x** for
P99 (5.41%), and **1.0411x** for P99.9 (4.11%). This removes dispatch overhead without changing
filter semantics, traversal order, or result quality.

## 8.7 Backlog: ACORN-style filter-first traversal

ACORN-1 is deferred for later investigation. It changes traversal semantics rather than only
optimizing predicate evaluation, so it is not expected to return the exact same neighbor IDs as
the current multihop implementation. The goal would be competitive recall at lower latency and
fewer distance comparisons.

The proposed experiment would retain rejected nodes as unscored routing connectors, expand
bounded multi-hop neighborhoods, and compute vector distances only for matching nodes. Evaluation
must compare recall-versus-latency curves rather than result parity:

- keep the existing multihop search as the baseline;
- add ACORN as a separate search type over the same graph;
- sweep L and routing depth/budget across S1-S9;
- require every returned node to satisfy the hard DNF predicate;
- compare latency and distance comparisons at equal recall;
- preserve an exact/current-search fallback if fewer than k matches are found.

This work is intentionally postponed. The immediate priority is exact-semantics optimization of
the current multihop implementation, including allocation reuse and removal of duplicated work,
before changing its traversal policy.

## 8.8 Backlog: SIMD whole-query bitmap materialization

A no-cache SIMD materialization path is worth testing as the strongest whole-query-bitmap
alternative to live DNF Bitslice evaluation. It would reuse the existing attribute-major dense
bitsets, but evaluate the complete predicate over all `ceil(N / 64)` words using AVX2 or AVX-512:
AND the terminal bitsets within each DNF clause, OR the clause results, and write one dense
query-result bitmap. ANN traversal would then perform one bit lookup per visited node.

This differs from `topk-multihop-live-filter-bitmap`, which currently performs Roaring posting-list
set algebra and then iterates the matching IDs to densify the result. The SIMD experiment should
not use a cross-query result cache: construct a fresh result bitmap for each query execution and
charge all allocation, initialization, Boolean operations, and writes to query latency.

The comparison should include:

- current Roaring materialization, SIMD dense materialization, and live Bitslice-DNF;
- one-, two-, four-, and larger-terminal DNF expressions with controlled short-circuit behavior;
- total filter evaluations relative to `N / 64`, including all nodes visited during the query;
- selective and broad results, plus different terminal/clause orderings;
- separate materialization, ANN traversal, and end-to-end latency;
- exact recall, comparison, hop, and filter-result parity;
- memory bandwidth, output-bitmap allocation/zeroing, and cache-pollution measurements.

The expected opportunity is complex predicates with weak live short-circuiting and enough visited
nodes to amortize a sequential SIMD pass. Live Bitslice-DNF should remain favored for small
one-off searches and well-ordered selective predicates.

## 8.9 100M provider/language hybrid-index trial

A persisted hybrid format was tested on
`Q:\test6\filtered_test4\EmbOffer.Global.100M.tsv`, using only provider ID (TSV column 3)
and language (column 6). The input contains 100,000,000 rows, 64,524 non-empty provider IDs,
887 non-empty languages, and 67,001 rows with both fields empty.

The hybrid encoder uses the raw-`u32` posting memory break-even point:
`dense_threshold = ceil(N / 64) * 2`, which is 3,125,000 matches for 100M vectors.
Labels at or above the threshold use dense Bitslice rows; the tail uses one contiguous offset and
sorted-`u32` postings area.

| Metric | Result |
|---|---:|
| Build time | 152.1 seconds |
| Persisted file size | 566,948,950 bytes (540.7 MiB) |
| Dense labels | 8 |
| Sparse labels | 65,403 |
| Dense payload | 100,000,000 bytes |
| Sparse postings + offsets | 464,881,536 bytes |

The eight dense labels are two providers and six languages whose cardinalities exceed the
memory threshold. Query compilation orders dense terminals before sparse terminals inside
conjunctions, then orders by increasing cardinality, so equivalent input orders use the same
short-circuit plan.

Full 100M-ID membership scans, three repetitions:

| Query | Representations | Matches | Mean | Million probes/s |
|---|---|---:|---:|---:|
| `provider:112453` | dense | 3,223,639 | 1.07 s | 93.77 |
| `provider:3665615` | sparse | 2,006,362 | 4.07 s | 24.58 |
| `language:en-US` | dense | 36,202,665 | 1.44 s | 69.63 |
| `language:ja-JP` | sparse | 1,757,655 | 4.05 s | 24.67 |
| `language:en-US&provider:3475408` | dense AND sparse | 331,486 | 2.70 s | 37.08 |
| `provider:112453&language:zh-CN` | dense AND sparse | 0 | 1.18 s | 84.60 |

The mixed-query counts were independently scanned from the raw TSV and matched exactly:
2,006,362, 0, and 331,486 for the tested provider/language pairs. No 100M-vector graph index is
present in `filtered_test4`, so these results validate persisted storage and query-provider
performance rather than end-to-end ANN latency.

## 8.10 100M filtered recall and latency

The truth-only query TSV was deduplicated into 2,147 logical queries. Each predicate is exactly:

```text
language:<locale> AND provider:<provider-id>
```

The 34,229 variable-length truth IDs were mapped from external offer IDs to graph row IDs. Every
truth ID was independently checked against its hybrid predicate; all matched.

A 100M-vector UINT8 graph was built with max degree 32, L-build 100, alpha 1.2, and 16 threads.
Build time was 2,871.1 seconds. Search used k=10, one task, three repetitions, and the hybrid
provider/language index.

| L | Recall@10 | AVG | P99 | P99.9 |
|---:|---:|---:|---:|---:|
| 50 | 20.64% | 2.75 ms | 17.77 ms | 21.94 ms |
| 100 | 22.43% | 4.40 ms | 28.53 ms | 38.60 ms |
| 200 | 24.02% | 7.31 ms | 47.03 ms | 59.16 ms |
| 400 | 24.62% | 13.06 ms | 84.28 ms | 106.71 ms |
| 800 | 25.31% | 25.03 ms | 166.99 ms | 210.27 ms |
| 1600 | 25.53% | 50.34 ms | 355.92 ms | 449.42 ms |

Recall plateaus near 25.5% even as L and latency increase sharply. The filter and truth-ID mapping
are internally consistent, so the remaining gap is graph quality and/or a mismatch between the
provided production truth semantics and exact squared-L2 nearest-neighbor truth. This benchmark
should not be used to tune the hybrid representation until that truth definition is confirmed.

## 8.11 Static 10K hybrid threshold

To reduce large sparse posting probes, a second hybrid index classified every label with at least
10,000 matches as dense.

| Metric | Memory break-even | Static 10K |
|---|---:|---:|
| Dense labels | 8 | 750 |
| Sparse labels | 65,403 | 64,661 |
| Sparse AVG length | 1,775 | 146 |
| Sparse MAX length | 3,111,690 | 9,993 |
| Persisted size | 540.7 MiB | 8.77 GiB |
| Build time | 152.1 s | 192.4 s |

K=150, L=150, one task, three repetitions:

| Filter | Recall@150 | AVG baseline -> 10K | P99 baseline -> 10K | P999 baseline -> 10K |
|---|---:|---:|---:|---:|
| Language only | 78.20% | 4.17 -> **3.72 ms** | 18.62 -> **10.80 ms** | 24.32 -> **16.62 ms** |
| Language AND provider | 25.54% | 7.11 -> **2.94 ms** | 49.18 -> **14.19 ms** | 63.82 -> **18.83 ms** |

The static 10K threshold confirms that long sparse postings dominate tail latency, especially for
provider-and-language conjunctions. It is a useful latency upper bound, but its 8.77 GiB footprint
is too expensive as a universal policy. The next tuning step is field- and query-aware promotion
under a fixed dense-memory budget.

## 8.12 Indexed BruteForceKNN sidecar

The AdsSnr hybrid prototype was rebuilt with a canonical memory-mapped data file and compact
HashStore postings:

- each sparse candidate's ID, UINT8 embedding, and integer label list are stored once;
- each sparse label posting stores only sorted `uint32` canonical data indices;
- labels are translated through one deterministic string-to-integer dictionary.

At the same static 10K threshold, 528 of the 2,147 queries contain at least one sparse label and
route to exact BruteForceKNN. The remaining 1,619 dense-only queries reuse the existing 10K hybrid
DiskANN path.

| Metric | Indexed BF+ANN | Previous static-10K | Original hybrid |
|---|---:|---:|---:|
| Recall@150 | **50.04%** | 25.54% | 25.54% |
| AVG | 3.59 ms | **2.94 ms** | 7.11 ms |
| P99 | 16.10 ms | **14.19 ms** | 49.18 ms |
| P999 | 21.40 ms | **18.83 ms** | 63.82 ms |
| Persisted sidecar | **1.31 GiB** | 8.77 GiB | 540.7 MiB |
| Build time | ~202.2 s | 192.4 s | 152.1 s |

The indexed sidecar reduces storage by **85.0%** versus static-10K and improves recall by
**24.50 percentage points**, at a 22.4% AVG and roughly 13.5% tail-latency cost. Compared with the
original hybrid, it is 1.98x faster at AVG, 3.05x faster at P99, and 2.98x faster at P999.

The first dense-only repetition incurred graph page faults, so the comparison combines querywise
latencies from warm repetitions 2 and 3. Sparse-routed recall is 100%; dense-only recall is 33.75%.

## 8.13 Correctness check and selectivity-routing backlog

The 50.04% recall result was rechecked because language-only recall was previously 78.20%. The
same 1,619 ANN-routed queries were run with language-only and conjunction predicates:

| Predicate on the same dense-routed queries | Recall@150 |
|---|---:|
| Language only | **79.15%** |
| Language AND provider | **33.75%** |
| Provider AND language | **33.75%** |

The reversed conjunction produces identical recall, ruling out terminal-order or expression
preprocessing errors. This workload also contains one AND clause per query, so each request routes
wholly to ANN or BruteForceKNN; the global top-K merger is not used. The combined recall is exactly:

```text
(528 * 100% + 1,619 * 33.749%) / 2,147 = 50.042%
```

The remaining gap is the routing criterion. Individual language and provider labels can both be
dense while their intersection remains sparse. Among the 1,619 ANN-routed queries:

| Combined language-provider pass rate | Queries | Share |
|---|---:|---:|
| `< 1%` | 1,230 | 76.0% |
| `< 0.1%` | 695 | 42.9% |
| `< 0.01%` | 38 | 2.3% |

The median language density is 16.14%, the median provider density is 0.319%, but the median
intersection density is only 0.201%. Multi-hop ANN therefore still performs label-blind traversal
for many selective conjunctions.

### Backlog: adaptive clause-selectivity routing

For a two-label dense clause, first SIMD-AND the two 100M-bit rows and popcount the result. An
AVX2 microbenchmark on an AMD EPYC 7763 measured approximately:

| Two-row AND + popcount | Latency |
|---|---:|
| Cache-hot | 0.63 ms |
| After cache eviction / DRAM-resident | 1.05 ms |

Each clause reads 25 MB. The next prototype will use the resulting valid-node count:

1. If valid count is at least 1% of the index (1,000,000 nodes), use multi-hop ANN.
2. If valid count is below 1%, enumerate valid IDs and exact-scan their embeddings.
3. Merge exact results through the existing global top-K path.
4. Cache counts by normalized `(language, provider)` pair where useful.

The 1% threshold is an initial experiment, not a final constant. Exact scanning 1M UINT8-128
embeddings reads roughly 128 MB, so the end-to-end benchmark must include bitmap materialization,
ID enumeration, embedding access, distance computation, and top-K maintenance. Thresholds of 0.1%
and 0.01% should also be measured.

## 10. Artifacts

### 10.1 Original PMax benchmark (`Q:\test6\filtered_test2\bench\full\`)

- Labels: `data_labels_set.jsonl` (596 labels, general), `data_labels_min.jsonl` (11, fast)
- Predicates: `predmin_S1..S9.jsonl` (also `predset_S1..S9.jsonl`)
- Groundtruth: `gtset_S1..S9.bin` (canonical, regenerated from all 596 labels); `gtmin_S1..S9.bin` are verified byte-identical projections
- Full-label Bitslice-DNF runbook: `runbook_bitslice_dnf_set.json`
- Runbooks: `runbook_setmin.json` (multihop), `runbook_beta_setmin.json` (beta), `runbook_livefilter.json` (live per-node)
- Outputs: `out_setmin.json`, `out_beta_setmin.json`, `out_livefilter.json` (+ `out_live_S8.json`, `out_live_S9.json`)
- Encoders: `gen_setmembership.py` (full), `gen_setmin.py` (minimal)
- Live-filter code: `diskann-label-filter/src/live_filter.rs` (InlineAttributeIndex / FrozenAttributeIndex + QueryLabelProvider); benchmark search-type `topk-multihop-live-filter`
- CSR live-filter code (section 8.1): `InlineAttributeIndexCsr` / `FrozenAttributeIndexCsr` in the same module; benchmark search-type `topk-multihop-live-filter-csr`; is_match microbenchmark `diskann-label-filter/benches/benchmarks/live_filter_bench.rs`
- Posting-list + materialized-bitmap live code (section 8.2): `InlineAttributeIndexPosting` / `FrozenAttributeIndexPosting` / `MaterializedBitmapProvider`; benchmark search-type `topk-multihop-live-filter-bitmap`
- Adaptive + bit-sliced live code (section 8.3): `InlineAttributeIndexAuto` (search-type `topk-multihop-live-filter-auto`) and `InlineAttributeIndexBitslice` (search-type `topk-multihop-live-filter-bitslice`)
- Flat DNF Bitslice code (section 8.4): `EncodedDnf`, `BitsliceSingleProvider`, and `BitsliceDnfProvider`; benchmark search-type `topk-multihop-live-filter-bitslice-dnf`; microbenchmark `diskann-label-filter/benches/benchmarks/live_filter_bench.rs`
- Inline Bitslice-DNF comparison (section 8.4.7): benchmark search-type `topk-inline-live-filter-bitslice-dnf`, using the same `InlineAttributeIndexBitslice` and DNF providers with `InlineFilterSearch`
- Encoded-label library comparison (section 8.4.8): `diskann-label-index`; benchmark search-types `topk-multihop-encoded-bitslice-dnf`, `topk-multihop-encoded-bitslice-ast`, and `topk-multihop-encoded-bitmap-ast`; local encoded files `data_labels_set.bitslice.bin` and `data_labels_set.bitmap.bin`
- Index (reused): `idxsave_full`(+`.data`)

### 10.2 100M provider/language hybrid trial (`Q:\test6\filtered_test4\`)

- Raw base data: `EmbOffer.Global.100M.tsv`
- Raw query sample: `EmbQuery.1k.tsv`
- Persisted hybrid labels: `provider_language.hybrid.bin`
- Static-10K hybrid labels: `provider_language.hybrid.threshold10000.bin`
- Encoder: `diskann-tools/src/bin/build_hybrid_label_index.rs`
- Probe tool: `diskann-tools/src/bin/probe_hybrid_label_index.rs`
- UINT8 vectors: `base.provider_language.u8bin`
- Deduplicated queries: `queries.provider_language.u8bin`
- Predicates: `query_predicates.provider_language.jsonl`
- Variable-length truth: `groundtruth.provider_language.bin`
- Graph: `index.provider_language.r32_l100` and `.data`
- Recall results: `hybrid_recall_100m.results.json` and
  `hybrid_recall_100m.high_l.results.json`
- K=150/L=150 threshold comparison:
  `hybrid_threshold10000_recall150_l150.results.json`
- Indexed BruteForceKNN sidecar:
  `provider_language_bfknn_t10000.bin` and `provider_language_bfknn_t10000.bfdata`
- Indexed sidecar comparison:
  `bfknn_indexed_threshold10000_comparison.json` and
  `bfknn_indexed_threshold10000_report.md`
- Indexed sidecar correctness check:
  `bfknn_indexed_threshold10000_correctness.md` and
  `hybrid_threshold10000_dense_correctness.results.json`

_Note: an earlier version of this report used a single-valued `geo` string field; it is superseded by the set-membership results above._
