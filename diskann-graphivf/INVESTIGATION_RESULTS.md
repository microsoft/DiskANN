# Graph-IVF Investigation Results

A running summary of performance and quality experiments on the `diskann-graphivf`
index, run against the enron-email-1M corpus. Append new findings as we go.

## Dataset

- **Corpus**: `normalized_dim_384_vector_fp16_1087932_vectors.bin` — 1,087,932
  vectors, dim 384, fp16, L2-normalized.
- **Queries**: `query_vector_normalized_dim_384_fp16_top_1000.bin` — 1,000 queries.
- **Groundtruth**: `groundtruth_recall_1000_query_1000.bin`.
- **Metric**: squared L2. **Recall** is recall@10 unless noted.
- **Build threads / search threads**: 8.

## Index configurations

| Index | Centroids | k-means train sample | Lloyd iters | Graph (deg/slack/L_build/alpha) | Save path |
|---|---:|---:|---:|---|---|
| 2048 | 2,048 | 100,000 | 10 | 32 / 1.2 / 64 / 1.2 | `graphivf_index` |
| 16384 | 16,384 | 100,000 | 5 | 32 / 1.2 / 64 / 1.2 | `graphivf_index_16384` |
| 16384-full | 16,384 | 1,087,932 (all) | 5 | 32 / 1.2 / 64 / 1.2 | `graphivf_index_16384_full` |
| 40960 | 40,960 | 400,000 | 4 | 32 / 1.2 / 64 / 1.2 | `graphivf_index_40960` |
| 40960-full | 40,960 | 40960 seed + 1 full-corpus iter | 1 (refine) | 32 / 1.2 / 64 / 1.2 | `graphivf_index_40960_full` |

Benchmark JSONs live in `diskann-benchmark/example/`:
`graph-ivf-enron-build*.json` / `graph-ivf-enron-search*.json`. Analysis tooling:
`diskann-graphivf/examples/centroid_graph_ablation.rs` (ablation 5),
`diskann-graphivf/examples/refine_centroids_full_corpus.rs` (ablation 8), and
`analyze_ivf_lists.py` at the repo root (ablation 7).

---

## 1. Fine-grained latency instrumentation

Added `BuildProfile` / `SearchProfile` (`diskann-graphivf/src/profile.rs`) that
render layer-cake breakdowns of where build and search time goes. Wired through
`build_profiled` / `search_profiled` in the index and surfaced in the benchmark
output as per-stage tables.

- **Build stages**: normalize, sample, kmeans, write_centroids, build_graph,
  assign, write_lists, write_metadata.
- **Search stages**: preprocess, centroid_search, plan_io, disk_read, score, topk.

---

## 2. Build cost breakdown

| Index | Build time | k-means | build_graph | assign | write_lists | corpus_load |
|---|---:|---:|---:|---:|---:|---:|
| 2048 (100K, 10 iters) | 42.13s | 30.4s (72.2%) | 0.4s (0.9%) | 9.0s (21.3%) | 2.3s (5.4%) | 1.27s |
| 16384 (100K, 5 iters) | 194.76s | 155.2s (79.7%) | 23.4s (12.0%) | 13.7s (7.1%) | 2.3s (1.2%) | 1.33s |
| 16384-full (1.09M, 5 iters) | 1445.74s | 1407.4s (97.3%) | 22.0s (1.5%) | 13.4s (0.9%) | 2.2s (0.2%) | 1.77s |
| 40960 (400K, 4 iters) | 1337.07s | 1139.0s (85.2%) | 168.5s (12.6%) | 26.8s (2.0%) | 2.4s (0.2%) | 1.83s |
| 40960-full (seed + 1 full iter) | 710.84s | 506.7s (71.3%) | 174.3s (24.5%) | 27.3s (3.8%) | 2.4s (0.3%) | (in kmeans) |

**Takeaway**: k-means dominates build, and its cost scales with the number of
training points. Training on the full corpus instead of a 100K sample makes
k-means ~9× slower (1407s vs 155s) with no recall benefit (see ablation 5). At
40,960 centroids `build_graph` also becomes significant (168.5s, 12.6%) — graph
construction cost grows with the number of nodes (centroids). Refining existing
centroids with **one** full-corpus Lloyd iteration (40960-full) costs a single
full-corpus pass (506.7s) — cheaper than a from-scratch full-corpus build — and
improves both recall and latency (see ablation 8).

---

## 3. Search latency optimization — buffer pooling

The original searcher allocated a fresh aligned `Poly<[u8]>` per probed cluster,
making `plan_io` (I/O plan construction) a significant cost at high nlist. Buffer
pooling reuses a single grown-on-demand scratch buffer across clusters, carving
disjoint 512-byte-aligned sub-slices per `AlignedRead`, with no per-query zeroing
(the disk read overwrites every parsed byte).

- Recall is **bit-identical** before/after (correctness preserved).
- **PlanIO at nlist=1024 (16384 index): 198,808 µs → 9,698 µs (20.5× faster)**.
- At 2048: PlanIO down 1.4–2.8× across nlist.

After pooling, **disk read dominates** search (86–96% of latency) — the index is
I/O-bound, as expected for a disk-resident IVF.

---

## 4. End-to-end search results (post buffer-pooling)

All latencies are **mean µs/query per stage**. Search stages:

- **Preproc** — query widen / normalize prep.
- **Centroid** — Vamana graph search over centroids to pick the `nlist` lists.
- **PlanIO** — build the aligned I/O read plan (buffer pooling target).
- **DiskRead** — read the probed posting lists from disk (dominant cost).
- **Score** — fp16 distance computation over the read vectors.
- **TopK** — final top-`k` selection.

### 2048 centroids (centroid_search_l 256)

| NList | QPS | Recall@10 | Mean µs | P95 µs | P999 µs |
|---:|---:|---:|---:|---:|---:|
| 8 | 489.0 | 53.81 | 11,809 | 22,491 | 39,119 |
| 16 | 223.8 | 61.98 | 29,539 | 59,215 | 94,753 |
| 32 | 116.5 | 69.96 | 59,200 | 122,267 | 159,774 |
| 64 | 59.9 | 77.43 | 119,846 | 302,359 | 543,050 |
| 128 | 30.6 | 84.79 | 241,453 | 797,436 | 1,354,844 |

Latency breakdown (mean µs/query):

| NList | Preproc | Centroid | PlanIO | DiskRead | Score | TopK | Total | DiskRead % |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 2.52 | 481.44 | 498.68 | 10,484.35 | 324.84 | 14.25 | 11,807.32 | 88.8% |
| 16 | 2.53 | 455.93 | 828.96 | 27,617.69 | 604.44 | 25.25 | 29,536.48 | 93.5% |
| 32 | 2.92 | 474.01 | 1,700.28 | 55,798.87 | 1,168.79 | 46.99 | 59,193.55 | 94.3% |
| 64 | 3.33 | 495.79 | 3,213.82 | 113,749.77 | 2,264.90 | 91.62 | 119,821.23 | 94.9% |
| 128 | 3.89 | 492.86 | 5,007.36 | 231,401.45 | 4,314.93 | 177.43 | 241,401.06 | 95.9% |

Centroid search ~480 µs flat (2,048-node graph); disk read 89–96% of latency.

### 16384 centroids — 100K-sample k-means (centroid_search_l 1024)

| NList | QPS | Recall@10 | Mean µs | P95 µs | P999 µs |
|---:|---:|---:|---:|---:|---:|
| 64 | 228.7 | 72.80 | 31,440 | 69,256 | 104,924 |
| 128 | 110.8 | 79.21 | 67,538 | 183,329 | 297,181 |
| 256 | 57.0 | 85.70 | 131,397 | 298,307 | 441,822 |
| 512 | 29.7 | 90.69 | 253,617 | 500,547 | 787,707 |
| 1024 | 15.6 | 94.39 | 492,340 | 974,473 | 1,736,403 |

Latency breakdown (mean µs/query):

| NList | Preproc | Centroid | PlanIO | DiskRead | Score | TopK | Total | DiskRead % |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 2.72 | 2,694.16 | 948.73 | 27,053.61 | 709.21 | 28.29 | 31,438.49 | 86.1% |
| 128 | 3.28 | 2,808.29 | 1,788.47 | 61,547.83 | 1,333.77 | 52.48 | 67,536.26 | 91.1% |
| 256 | 3.61 | 2,838.14 | 3,453.96 | 122,482.67 | 2,512.95 | 98.93 | 131,392.91 | 93.2% |
| 512 | 4.01 | 2,955.89 | 5,904.55 | 239,585.08 | 4,913.31 | 193.42 | 253,560.08 | 94.5% |
| 1024 | 4.25 | 2,760.99 | 9,698.14 | 470,480.88 | 8,940.28 | 350.14 | 492,240.69 | 95.6% |

Centroid search ~2.8 ms flat (16,384-node graph, ~6× the 2048 graph's per-query
cost); disk read 86–96% of latency.

### 16384 centroids — full-corpus k-means (centroid_search_l 1024)

| NList | QPS | Recall@10 | Mean µs | P95 µs | P999 µs |
|---:|---:|---:|---:|---:|---:|
| 64 | 418.1 | 74.99 | 16,925 | 29,551 | 112,619 |
| 128 | 256.5 | 79.80 | 29,002 | 74,457 | 104,462 |
| 256 | 124.0 | 84.97 | 61,145 | 138,034 | 216,313 |
| 512 | 63.3 | 89.56 | 119,860 | 232,024 | 421,769 |
| 1024 | 32.1 | 93.38 | 239,795 | 445,025 | 688,763 |

Latency breakdown (mean µs/query):

| NList | Preproc | Centroid | PlanIO | DiskRead | Score | TopK | Total | DiskRead % |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 2.36 | 2,641.71 | 573.28 | 13,316.92 | 371.76 | 13.67 | 16,921.65 | 78.7% |
| 128 | 2.95 | 2,719.80 | 954.51 | 24,595.41 | 696.83 | 27.10 | 28,998.58 | 84.8% |
| 256 | 3.24 | 2,787.78 | 1,759.14 | 55,277.06 | 1,262.67 | 49.45 | 61,141.30 | 90.4% |
| 512 | 3.88 | 2,853.16 | 3,522.97 | 111,015.92 | 2,362.09 | 94.58 | 119,855.13 | 92.6% |
| 1024 | 4.11 | 2,792.49 | 4,972.26 | 227,364.60 | 4,423.88 | 182.03 | 239,745.71 | 94.8% |

> Note: this full-corpus index's DiskRead and PlanIO at matched nlist are markedly
> lower than the 100K-sample 16384 index (e.g. nlist=1024 DiskRead 227 ms vs
> 470 ms, ~2× faster QPS) despite nearly identical recall and the **same number of
> centroids**. This is not noise — it is driven by **list-size imbalance**: the
> 100K-sample k-means produced much lumpier lists, so each probe reads ~2× more
> bytes. See ablation 7 for the full distribution analysis (size-weighted mean
> 249 vs 130 points/probe ≈ the measured 2× DiskRead ratio).

### 40960 centroids — 400K-sample k-means (centroid_search_l 1024)

| NList | QPS | Recall@10 | Mean µs | P95 µs | P999 µs |
|---:|---:|---:|---:|---:|---:|
| 64 | 552.1 | 73.76 | 13,140 | 23,120 | 35,304 |
| 128 | 360.0 | 79.22 | 20,728 | 39,782 | 49,548 |
| 256 | 193.9 | 84.34 | 38,894 | 75,409 | 135,868 |
| 512 | 95.1 | 88.67 | 80,336 | 150,516 | 223,333 |
| 1024 | 49.1 | 92.42 | 157,855 | 262,270 | 399,884 |

Latency breakdown (mean µs/query):

| NList | Preproc | Centroid | PlanIO | DiskRead | Score | TopK | Total | DiskRead % |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 2.73 | 4,305.93 | 377.74 | 8,149.85 | 288.39 | 10.51 | 13,137.15 | 62.0% |
| 128 | 2.78 | 4,326.23 | 683.09 | 15,199.57 | 491.42 | 19.73 | 20,724.74 | 73.3% |
| 256 | 3.30 | 4,179.77 | 1,334.36 | 32,473.35 | 864.41 | 34.03 | 38,891.59 | 83.5% |
| 512 | 3.45 | 4,014.15 | 2,470.00 | 72,188.73 | 1,587.55 | 65.31 | 80,332.66 | 89.9% |
| 1024 | 4.02 | 3,933.08 | 3,803.00 | 147,074.89 | 2,902.38 | 122.72 | 157,845.81 | 93.2% |

Centroid search ~4.0–4.3 ms flat (40,960-node graph); the smaller lists make this
the **dominant cost at low nlist** (33% of total at nlist=64). At matched nlist,
DiskRead is far lower than either 16384 index (e.g. nlist=1024: 147 ms vs 227/470
ms) because lists average only ~26.6 points — ~2.4–3.4× the QPS of the 100K-sample
16384 index. Recall is slightly lower at high nlist (each probe covers a smaller
fraction of the corpus), so to *match* recall you need a higher nlist.

### 40960 centroids — full-corpus refine (seed + 1 Lloyd iter, centroid_search_l 1024)

Built by seeding Lloyd's with the 40960 (400K-sample) centroids and running one
full-corpus iteration (see ablation 8).

| NList | QPS | Recall@10 | Mean µs | P95 µs | P999 µs |
|---:|---:|---:|---:|---:|---:|
| 64 | 624.6 | 74.97 | 11,877 | 21,154 | 27,916 |
| 128 | 357.6 | 80.28 | 20,685 | 42,960 | 62,724 |
| 256 | 202.2 | 85.42 | 36,831 | 75,571 | 106,342 |
| 512 | 102.2 | 89.58 | 74,784 | 136,888 | 214,487 |
| 1024 | 51.8 | 93.02 | 148,799 | 249,961 | 394,286 |

Latency breakdown (mean µs/query):

| NList | Preproc | Centroid | PlanIO | DiskRead | Score | TopK | Total | DiskRead % |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 2.65 | 3,981.78 | 317.14 | 7,301.71 | 260.33 | 9.87 | 11,875.27 | 61.5% |
| 128 | 2.57 | 3,872.73 | 653.28 | 15,666.78 | 467.39 | 17.34 | 20,682.55 | 75.7% |
| 256 | 3.07 | 3,879.69 | 1,214.11 | 30,858.24 | 835.07 | 35.10 | 36,827.52 | 83.8% |
| 512 | 3.29 | 3,800.13 | 2,275.89 | 67,163.67 | 1,473.31 | 61.06 | 74,781.09 | 89.8% |
| 1024 | 3.65 | 3,731.51 | 4,006.85 | 138,186.80 | 2,743.67 | 117.20 | 148,795.30 | 92.9% |

One full-corpus Lloyd iteration over the sample-trained seeds improves **both**
recall (+0.6 to +1.2 pts at every nlist) **and** latency (DiskRead @ nlist=1024
147.1 → 138.2 ms; QPS 49.1 → 51.8) — the refined centroids fit the corpus slightly
better and rebalance the lists (size-weighted mean 80.4 → 73.9, ablation 7).

### Cross-index observations

- **DiskRead dominates** at moderate-to-high nlist (62–96%) and grows roughly
  linearly with nlist (more lists → more bytes). Search is **I/O-bound** there.
- **Centroid search is a fixed per-query overhead** set by the centroid-graph size:
  ~480 µs for 2,048 centroids, ~2.8 ms for 16,384, ~4.0 ms for 40,960. It is
  amortized at high nlist but becomes the bottleneck at low nlist with many small
  lists (33% of total at nlist=64 for the 40960 index).
- **PlanIO and Score** are minor (each ≤ ~4% of total) and scale with nlist.
- **More centroids → smaller lists → fewer bytes per probe → higher QPS at a given
  nlist**, but a higher fixed centroid-search cost and slightly lower recall per
  nlist. The list-size *distribution* (not just the count) matters: see ablation 7.

---

## 5. Ablation: centroid-graph quality

**Question**: how much does the Vamana graph over the centroids (vs an exact
nearest-centroid scan) cap end-to-end recall?

**Method** (`diskann-graphivf/examples/centroid_graph_ablation.rs`): rebuild the
same centroid graph, then per query compare graph-returned top-`nlist` centroids
against brute-force (exact squared-L2) top-`nlist` centroids.
`recall@nlist = mean |graph ∩ exact| / nlist`. Run on the 16384 (100K) index.

| NList | Graph-centroid recall | End-to-end recall@10 |
|---:|---:|---:|
| 64 | 99.91% | 72.80% |
| 128 | 99.89% | 79.21% |
| 256 | 99.86% | 85.70% |
| 512 | 99.76% | 90.69% |
| 1024 | 99.48% | 94.39% |

**Takeaway**: the centroid graph recovers 99.5–99.9% of the exact nearest
centroids — it is **not** the bottleneck. The graph steers the IVF to essentially
the same lists an exhaustive centroid scan would pick. The large gap to perfect
end-to-end recall comes from **IVF partition geometry** (a query's true neighbors
often live in lists outside the top-`nlist` probed) plus fp16 scoring, not graph
approximation.

---

## 6. Ablation: k-means training set size

**Question**: does training k-means on the full corpus (vs a 100K sample) improve
clustering quality and therefore recall?

**Method**: build a second 16384-centroid index identical in every parameter
except `sample_size` = full corpus (1,087,932), 5 Lloyd iters. Saved separately as
`graphivf_index_16384_full`.

| NList | 100K-sample recall | Full-corpus recall | Δ |
|---:|---:|---:|---:|
| 64 | 72.80 | 74.99 | +2.19 |
| 128 | 79.21 | 79.80 | +0.59 |
| 256 | 85.70 | 84.97 | −0.73 |
| 512 | 90.69 | 89.56 | −1.13 |
| 1024 | 94.39 | 93.38 | −1.01 |

**Cost**: k-means ~9× slower (1407s vs 155s); total build 1445.7s vs 194.8s.

**Takeaway**: full-corpus training gives **no meaningful recall benefit** — a small
gain at low nlist, a slight regression at high nlist. 100K is already a
representative sample for the *recall* of 16,384 centroids; the 9× extra training
cost buys no recall.

> But it is **not** a wash for *latency*: the full-corpus index is ~2× faster at
> matched nlist (and recall) because it produces far more balanced lists. So the
> right read is "100K sampling gives equal recall but lumpier, slower lists." If
> you can afford the build cost, more training data buys **throughput**, not
> recall. See ablation 7.

---

## 7. Ablation: posting-list size distribution

**Question**: the 100K-sample and full-corpus 16384 indexes have the *same* number
of centroids and near-identical recall, yet the 100K index has ~2× the search
latency at matched nlist. Why?

**Method** (`analyze_ivf_lists.py`): parse the per-cluster point counts straight
out of each index's `.graphivf_meta` file and compute the list-size distribution.
The key metric is the **size-weighted mean** = Σcᵢ²/Σcᵢ = the expected size of the
list a uniformly-random *point* (≈ a query) lands in. DiskRead time is proportional
to this, not to the plain mean, because queries land in dense regions, which are
exactly the big lists.

| Index | mean | p50 | p90 | p99 | max | empty | CV | Gini | **size-wt mean** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 531.2 | 439 | 1083 | 1930 | 4066 | 4 | 0.84 | 0.435 | 904.2 |
| 16384 (100K) | 66.4 | 24 | 183 | 511 | 2051 | 647 | **1.66** | **0.679** | **249.2** |
| 16384 (full) | 66.4 | 52 | 135 | 269 | 3229 | 48 | 0.98 | 0.432 | **129.7** |
| 40960 (400K) | 26.6 | 14 | 66 | 169 | 1466 | 1237 | 1.42 | 0.595 | 80.4 |
| 40960 (full refine) | 26.6 | 15 | 66 | 153 | 1481 | 1219 | 1.34 | 0.572 | 73.9 |

The two 16384 indexes have **identical mean** (66.4 = points/clusters) but the
100K-sample run is far lumpier: median list 24 vs 52, p99 511 vs 269, 647 empty
clusters (4.0%) vs 48 (0.3%), Gini 0.679 vs 0.432.

**The size-weighted means are 249.2 vs 129.7 — a 1.92× ratio that matches the
measured DiskRead ratio almost exactly** (nlist=1024: 469.8 ms / 227.4 ms = 2.07×).

Histograms (100K-sample is a spike at zero with a long tail; full-corpus is a tight
bell around the mean):

```
16384 (100K sample)            16384 (full corpus)
[  0, 31) 55.6% ##############  [  0, 16) 13.8% ########
[ 31, 63) 14.6% ###             [ 16, 33) 18.5% ##########
[ 63, 94)  8.3% ##              [ 33, 49) 15.9% #########
[ 94,126)  5.5% #               [ 49, 66) 14.0% ########
 ...tail to 2051, 647 empty      ...short tail, 48 empty
```

The 40960 index (~26.6 points/list mean) is the most spiked-at-zero of all — 42.6%
of lists hold <10 points and 1,237 (3.0%) are empty — but its absolute list sizes
are the smallest, so the size-weighted mean (80.4) and hence bytes-per-probe are
still the lowest of the four indexes:

```
40960 (400K sample)
[  0, 10) 42.6% ########################################
[ 10, 20) 18.7% ##################
[ 20, 30) 10.9% ##########
[ 30, 41)  7.7% #######
[ 41, 51)  5.0% #####
[ 51, 61)  3.7% ###
 ...tail to 1466, 1237 empty (mean 26.6, size-wt mean 80.4)
```

One full-corpus Lloyd iteration on the 40960 seeds (40960 full refine) trims the
tail and rebalances slightly: p99 169 → 153, CV 1.42 → 1.34, Gini 0.595 → 0.572,
size-weighted mean 80.4 → 73.9 (−8%). Empty-list count barely moves (1,237 → 1,219)
— refinement repositions centroids but doesn't repopulate dead ones.

**Mechanism**: k-means++ seeded on only 100K points (~6 samples per centroid) is
too sparse to place 16,384 centroids well — many initialize in near-empty regions
(→ empty/tiny lists) while a few land in dense regions and absorb huge populations
(→ fat tail). Training on the full 1.09M corpus gives k-means enough data to spread
centroids evenly, halving the size-weighted list size and therefore the
bytes-read-per-probe.

**Takeaway**: at a fixed centroid count, **clustering balance — not the centroid
count or the mean list size — controls search throughput**. The size-weighted mean
list size (Σcᵢ²/Σcᵢ) is the right predictor of DiskRead cost. More/denser k-means
training data improves balance (hence QPS) even when it doesn't improve recall.

---

## 8. Ablation: refining sample-trained centroids over the full corpus

**Question**: the 40960 index was trained on a 400K sample. Does one Lloyd
iteration over the **entire** corpus — seeded from those centroids — improve recall
and/or throughput, and at what cost?

**Method** (`diskann-graphivf/examples/refine_centroids_full_corpus.rs`): read the
40960 (400K) centroids back from `*.graphivf_centroids.fbin`, seed Lloyd's with
them, run **one** iteration over all 1,087,932 points (no re-seeding, no sampling),
then rebuild the graph and inverted lists. Implemented via a new
`GraphIvfIndex::build_from_seed_centroids_profiled` (uses `run_lloyds` on the
pre-seeded centers). Saved separately as `graphivf_index_40960_full`.

| NList | 40960 (400K) recall | refined recall | Δ recall | 40960 QPS | refined QPS |
|---:|---:|---:|---:|---:|---:|
| 64 | 73.76 | 74.97 | +1.21 | 552.1 | 624.6 |
| 128 | 79.22 | 80.28 | +1.06 | 360.0 | 357.6 |
| 256 | 84.34 | 85.42 | +1.08 | 193.9 | 202.2 |
| 512 | 88.67 | 89.58 | +0.91 | 95.1 | 102.2 |
| 1024 | 92.42 | 93.02 | +0.60 | 49.1 | 51.8 |

**Cost**: 710.8s total (kmeans = one full-corpus pass 506.7s + build_graph 174.3s),
vs the original from-scratch 40960 build of 1337s.

**Takeaway**: a single full-corpus refinement improves **both** axes — recall by
~0.6–1.2 pts at every nlist, and throughput via more balanced lists (size-weighted
mean 80.4 → 73.9, ablation 7). Unlike the 16384 full-corpus retrain (ablation 6,
which helped only latency), *refining* good seeds with the exact corpus means nudges
the partition geometry in the right direction, so recall rises too. The gain is
small because recall is fundamentally capped by partition geometry, but it is free
relative to a from-scratch full-corpus build (and faster than the original build).

---

## Overall conclusions so far

- Search is **I/O-bound** at moderate-to-high nlist (disk read 62–96% of latency);
  at low nlist with many small lists, **centroid-graph search** becomes the floor.
- The recall ceiling is set by **IVF partition geometry**, not by:
  - centroid-graph approximation (99.5%+ graph recall, ablation 5), nor
  - k-means training convergence (full corpus ≈ 100K sample on recall, ablation 6).
- **Search throughput is set by list-size balance** (ablation 7): the
  size-weighted mean list size Σcᵢ²/Σcᵢ predicts DiskRead cost, and denser k-means
  training improves balance (≈2× QPS) independently of recall.
- **Refining sample-trained centroids with one full-corpus Lloyd iteration**
  (ablation 8) improves both recall (+~1 pt) and throughput (more balanced lists)
  for roughly the cost of a single corpus pass — cheaper than a from-scratch
  full-corpus build.
- Levers:
  - **Recall** → more centroids, or multi/soft assignment.
  - **Throughput at fixed recall** → smaller, more *balanced* lists (more centroids;
    more/denser k-means training; refine seeds over the full corpus; or a
    balanced-assignment scheme).
