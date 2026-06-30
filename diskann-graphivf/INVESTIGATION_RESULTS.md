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

## 9. Disk-read latency: parallelism, caching, prefetch, async

**Goal**: minimize *per-query* latency (mean and p95) on the `16384-full` index,
which is what we optimize for in production (4 search threads). The question:
are we using parallelism / caching / prefetching / async optimally to fetch the
posting lists from disk? All runs use `centroid_search_l = 1024`, squared L2,
recall@10; the harness runs `num_threads` *queries* concurrently — each query is
internally single-threaded.

### 9.0 The read path (how a query touches disk)

A query's `disk_read` stage issues **one batch of `nlist` reads** (one per probed
posting list) through the platform `AlignedFileReader`:

- **Windows** (`WindowsAlignedFileReader`): IOCP with
  `FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED` — i.e. **unbuffered/direct I/O
  that bypasses the OS page cache**. It submits up to `MAX_IO_CONCURRENCY = 128`
  reads *concurrently*, then blocking-polls completions. Batches larger than 128
  are split into sequential rounds.
- **Linux** (`LinuxAlignedFileReader`): `io_uring` with `O_DIRECT`,
  `submit_and_wait(batch)` — true async submission, also 128-deep, also no page
  cache.
- The reader holds `&mut self` (per-thread, not shareable), and there is **no
  caching / LRU / page-cache layer anywhere** in the stack.

Two consequences fall out of this immediately:
1. A *single* query already submits all of its reads concurrently (up to 128 in
   flight). The OS sees a deep queue from just one query — we are **not**
   submission-starved at the single-query level.
2. Because direct I/O bypasses the page cache, **re-reading the same hot list
   costs a full device read every time**. Nothing is amortized across queries.

### 9.1 New telemetry

To reason about bytes rather than just time, `SearchProfile` now records, per
query: `io_count` (reads issued to disk), `bytes_read` (disk bytes), and
`cache_hits` / `cache_bytes` (lists/bytes served from the new RAM cache below).
The benchmark prints a **"Search I/O volume (mean per query)"** table
(`NList / Reads / DiskKiB / CacheHits / CacheKiB`) alongside the latency cake.

### 9.2 Experiment — concurrency sweep: is parallelism the lever?

Swept `num_threads ∈ {1, 2, 4, 8}` on `16384-full`. The user is *especially*
interested in parallelism, so this is the first question to settle empirically.

| nlist | metric | T=1 | T=2 | T=4 | T=8 |
|---:|---|---:|---:|---:|---:|
| 1024 | QPS | 22.0 | 31.8 | 32.0 | 31.1 |
| 1024 | mean µs | 45,514 | 62,630 | 122,250 | 241,977 |
| 1024 | p95 µs | 56,289 | 89,095 | 161,974 | 402,159 |
| 1024 | DiskRead µs | 37,711 | 54,500 | 110,343 | 227,906 |
| 256 | QPS | 72.9 | 125.9 | 124.4 | 124.9 |
| 256 | mean µs | 13,711 | 15,792 | 31,241 | 59,080 |
| 64 | QPS | 154.5 | 291.7 | 509.0 | 440.3 |

Recall is identical across thread counts (74.99 / 84.97 / 93.38 at nlist
64 / 256 / 1024), as expected.

**Reading the result**: QPS plateaus by **T=2** at high nlist (22→32→32→31) while
per-query mean and p95 latency **inflate roughly linearly** with thread count
(45.5ms → 122ms → 242ms at nlist 1024). That is the signature of a **saturated
device**: aggregate read bandwidth is pinned (~1.4–1.7 GB/s here), so adding
concurrent queries just queues them behind one another. A single T=1 query
already nearly saturates the SSD because IOCP keeps 128 reads in flight.

**Conclusion on parallelism**: more *query-level* threads cannot reduce per-query
latency — beyond ~2 threads they actively *worsen* it. By the same token,
**intra-query read parallelism (splitting one query's reads across worker threads,
each with its own reader) cannot help either**: the bottleneck is device
bandwidth, not I/O submission depth, and a single query is already submitting a
128-deep queue. We are *not* under-using parallelism, async, or prefetch on the
fetch path; the platform reader already issues the whole batch asynchronously and
concurrently. **The only way to cut disk-read latency is to read fewer bytes.**

### 9.3 Experiment — hot-list RAM cache (the working lever)

Since the device is bandwidth-bound and direct I/O re-reads hot lists every time,
the fix is to keep the most-read lists resident in RAM and score them without
touching disk. Implemented `ListCache<T>` (`src/cache.rs`): at load time, greedily
select the **largest** non-empty posting lists (by point count) up to a byte
budget, hold their ids+vectors in RAM, and at search time serve any probed list
that is resident from RAM (`cache_hits`) while only disk-reading the rest. Wired
through `load_with_cache(prefix, num_threads, cache_budget_bytes)` and a
`cache_budget_mb` knob in the benchmark. Selecting *largest-first* is deliberate:
the size-weighted list distribution (ablation 7) means a few big lists dominate
both the bytes read and the probe probability.

Sweep on `16384-full`, **T=4** (production), `cache_budget_mb ∈ {0, 256, 512, 1024}`.

> **Sizing caveat — read this before the table.** The list file stores the *full*
> fp16 vectors plus their u32 ids, so the entire on-disk payload is only
> ~**801 MiB**: vectors `1,087,932 × 384 × 2 B ≈ 796.8 MiB` + ids
> `1,087,932 × 4 B ≈ 4.2 MiB`. In other words the whole "disk" index fits in
> under a gigabyte of RAM. That reframes the budgets:
>
> | budget | fraction of the 801 MiB payload |
> |---:|---|
> | 256 MiB | ~32% (genuine partial cache) |
> | 512 MiB | ~64% (still *smaller* than the dataset) |
> | 1024 MiB | **caps at 801 MiB = the entire index in RAM** |
>
> So the **1024 MiB row is not really "a cache"** — it is *"what if the whole
> index lived in RAM"*, i.e. the DiskRead→0 upper bound, and it's only reachable
> here because this corpus is tiny (~0.8 GB). For a corpus 10–100× larger — the
> regime that justifies a disk-based index in the first place — full residency is
> impossible and only the **partial-cache rows (256/512 MiB) are
> representative**. The generalizable takeaway is therefore the *super-linear
> byte payoff of largest-first caching*, not the full-residency 13×.

| budget | lists resident | nlist | mean µs | p95 µs | DiskRead µs | QPS | mean disk KiB/q |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 MiB | 0 | 1024 | 122,842 | 156,447 | 113,866 | 32.1 | 49,691 |
| 256 MiB | 1,825 | 1024 | 97,115 | 117,548 | 87,405 | 40.2 | 39,697 |
| 512 MiB | 5,422 | 1024 | 51,939 | 74,383 | 42,492 | 74.8 | 21,585 |
| **1024 MiB** | **16,336 (all)** | 1024 | **9,217** | **11,193** | **1.1** | **427.8** | **0.0** |
| 0 MiB | 0 | 256 | 31,431 | 53,130 | 26,748 | 124.1 | 12,889 |
| 512 MiB | 5,422 | 256 | 12,656 | 18,868 | 7,958 | 304.5 | 5,346 |
| **1024 MiB** | all | 256 | **4,167** | **5,064** | **1.0** | **948.6** | **0.0** |
| **1024 MiB** | all | 64 | **2,820** | **3,767** | **0.9** | **1399.8** | 0.0 |
| 0 MiB | 0 | 64 | 7,293 | 11,159 | 4,224 | 525.3 | 3,426 |

Recall is unchanged at every budget (the cache returns the *same* vectors, just
from RAM): 74.99 / 84.97 / 93.38.

**Findings**:
- **Partial budgets pay off super-linearly in bytes** (the generalizable result,
  per the sizing caveat above) thanks to largest-first selection. A 256 MiB cache
  (~32% of the payload) holds only 78 of 1023 probed lists (7.6% hit *count*) yet
  removes **20% of disk bytes** (49.7 → 39.7 MiB/query) — big lists carry the
  bytes. A 512 MiB cache (~64% of the payload, still smaller than the dataset)
  already halves nlist-1024 latency (122.8 → 51.9 ms) and more than doubles QPS.
- **Full residency** (the 1024 MiB row) is the DiskRead→0 *upper bound*, not a
  realistic cache: it loads the whole 801 MiB index into RAM, cutting nlist-1024
  **mean latency 13.3×** (122.8 ms → 9.2 ms) and **p95 14×** (156 ms → 11.2 ms),
  with DiskRead collapsing from 114 ms to ~1 µs. It is only attainable because
  this corpus is ~0.8 GB; treat it as "the ceiling if the index were in-memory."
- Once disk is removed, the **new floor is the centroid-graph search** (~2.4–3.2
  ms/query, independent of nlist) plus scoring — exactly the low-nlist floor
  identified in earlier sections. That is the next target (e.g. quantized /
  smaller centroid set, or a cheaper centroid ANN).

### 9.4 Verdict on the four levers

- **Parallelism**: already optimal on the fetch path and *cannot* lower per-query
  latency — the SSD is bandwidth-saturated even by one query (§9.2). Use *fewer*
  concurrent queries for latency; ~2 threads already max throughput.
- **Async**: already in use (IOCP / io_uring submit the full batch
  asynchronously, 128-deep). No headroom there.
- **Prefetch**: would not help — there is no idle device time to hide latency
  behind; the device is the bottleneck, and probed lists aren't known until after
  the centroid search.
- **Caching**: the decisive lever. A RAM-resident hot-list cache turns the
  bandwidth-bound disk reads into memory scores. Full residency gives **~13× mean
  / ~14× p95** — but on this tiny ~0.8 GB corpus that is just "the whole index in
  RAM" (the 1024 MiB budget exceeds the 801 MiB payload). The *durable* result is
  the **largest-first partial cache**: bytes removed scale far faster than hit
  count (256 MiB ⇒ 7.6% of lists but 20% of bytes; 512 MiB ⇒ ~2× QPS at ~64% of
  the payload), which is what generalizes to corpora too large to fully cache.
  This directly serves the per-query mean/p95 objective.

---

## 10. 8-bit MinMax quantized posting lists

The `16384-full` index rebuilt with 8-bit MinMax-quantized vectors stored in the
inverted lists (1 byte/dim + a 20-byte per-vector header) instead of `fp16`
(2 bytes/dim), reusing the baseline `fp16` centroids unchanged (zero Lloyd
iterations). Posting-list file: **444 MB** vs **840 MB** (`fp16`).

### Build latency breakdown (total 79,741 ms)

| Stage | ms | % |
|---|---:|---:|
| normalize | 496.6 | 0.6 |
| sample | 0.0 | 0.0 |
| kmeans | 52.4 | 0.1 |
| write_centroids | 45.6 | 0.1 |
| build_graph | 38,227.5 | 47.9 |
| assign | 39,515.6 | 49.6 |
| write_lists | 1,400.2 | 1.8 |
| write_metadata | 1.7 | 0.0 |
| other | 1.4 | 0.0 |

### Search results (num_threads = 4, centroid_search_l = 1024, squared L2, recall@10, cache off)

| NList | Recall | Mean µs | P95 µs | DiskRead µs | Bytes/query | QPS |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 74.74 | 9,994 | 11,452 | 6,617 | 1,868,542 | 365.7 |
| 256 | 84.61 | 16,634 | 25,013 | 11,715 | 7,034,458 | 230.8 |
| 1024 | 92.86 | 64,746 | 90,256 | 56,028 | 27,132,310 | 61.1 |


---

## 11. Graph-IVF vs disk-index latency: why fewer bytes ≠ lower latency

Comparing the two on-disk designs at high search effort (single-threaded, T=1)
surfaced a counter-intuitive result: at recall@1000 ≈ 85%, the `diskann-disk`
Vamana+PQ index (`L_search = 1000`, `beam_width = 4`) has **~88 ms** mean latency
while the graph-IVF indexes at comparable effort sit at **31–39 ms** — *even
though graph-IVF reads far more bytes per query* (16–30 MB vs ~4 MB).

The per-stage breakdowns show I/O is ~83–88% of latency in **both** designs, so
the difference is entirely in *how each one pulls bytes off the SSD*:

| Design | mean | I/O time | bytes / query | # I/Os | I/O pattern | effective BW |
|---|---:|---:|---:|---:|---|---:|
| graph-IVF 40960, nlist 1024 | 31.2 ms | 26.1 ms | 16.9 MB | 1 batch | one wide batched read | ~650 MB/s |
| graph-IVF 16384, nlist 1147 | 38.9 ms | 32.2 ms | 30.3 MB | 1 batch | one wide batched read | ~940 MB/s |
| disk-index, L=1000, bw 4 | 88.4 ms | 77.5 ms | ~4 MB | 1013 | long dependent chain | ~53 MB/s |

**Root cause = I/O queue depth, not data volume.** Graph-IVF, after the centroid
search, knows *all* `nlist` target lists up front and submits them as a **single
batched read with up to 128 concurrent I/Os** (Windows IOCP). The SSD pipelines
them, so wall time ≈ `bytes / bandwidth` — it moves tens of MB but at 650–940
MB/s effective. The disk-index Vamana beam search is inherently **sequential**:
each hop's next candidates depend on the distances computed from the *previous*
hop's fetched nodes, so only `beam_width` I/Os are ever in flight. With
`beam_width = 4` and ~1013 node reads, the query is a chain of ~253 dependent
round-trips, each a 4 KB random read costing ~75 µs under O_DIRECT (no page
cache). That latency can't be hidden behind a deep queue, so the device is used
~12–18× less efficiently (~53 MB/s).

### beam_width sweep (L=1000, T=1, recall@1000)

Confirming the hypothesis — widening the beam (deeper I/O queue) should cut
latency *without* materially changing the bytes read or recall:

| beam_width | mean | I/O time | CPU time | p95 | p99 | mean I/Os | recall@1000 | QPS |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 88.4 ms | 77.5 ms | 10.9 ms | 99.3 ms | 107.4 ms | 1012.7 | 84.87 | 11.3 |
| 8 | 60.7 ms | 51.5 ms | 9.1 ms | 66.1 ms | 70.3 ms | 1022.8 | 84.91 | 16.5 |
| 16 | 48.9 ms | 40.1 ms | 8.8 ms | 55.0 ms | 65.7 ms | 1043.9 | 84.99 | 20.4 |
| 32 | 44.0 ms | 35.0 ms | 8.9 ms | 49.3 ms | 61.3 ms | 1089.3 | 85.15 | 22.7 |

**Result**: raising `beam_width` 4 → 32 **halves mean latency (88 → 44 ms)**,
almost entirely from I/O time (77 → 35 ms, −55%), for the **same I/O count**
(1013 → 1089, +7%) and **flat recall** (84.87 → 85.15%). This is the smoking gun:
the original 88 ms was a **queue-depth** problem, not a bytes/bandwidth one — at
beam 4 most of the dependency chain stalls on per-read latency; a wider beam lets
the SSD service more reads concurrently and hides it.

**Why it plateaus at ~44 ms (still above graph-IVF's 31 ms)**: returns diminish
past beam 16 (I/O time 77 → 51 → 40 → 35 ms) as the device's IOPS/queue saturates
and a wider beam also wastes a few extra node reads (I/Os creep up). Graph-IVF
still wins *here* because its single batched read is 128-deep (vs a 32-max beam)
and this corpus is RAM-resident-cheap, so its large cluster scans hit high
bandwidth.

**Caveat — this ordering is dataset-specific.** The enron corpus (<1 GB) makes
graph-IVF's large per-query byte volume cheap. On a corpus far larger than RAM,
graph-IVF's bytes/query scale with `nlist × list_size` and eventually dominate,
while disk-index's I/O count stays bounded by graph hops — so the crossover
flips. The takeaway is the *mechanism*: graph-IVF is **bandwidth-bound** (one wide
batch), disk-index is **random-I/O-latency-bound** (a deep dependency chain whose
effective queue depth is `beam_width`).

### Re-running the full L-sweeps at beam_width = 32

To see whether the queue-depth win holds across the whole operating curve, both
original `beam_width = 4` L-sweeps were re-run at `beam_width = 32` (T=1).

**recall@1000 (L = 1000 / 1500 / 2000):**

| L_search | mean (bw32) | mean (bw4) | p99 (bw32) | p99 (bw4) | I/Os (bw32) | I/Os (bw4) | recall@1000 (bw32) | recall@1000 (bw4) | QPS (bw32) | QPS (bw4) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 43.8 ms | 86.5 ms | 57.5 ms | 104.3 ms | 1089.3 | 1012.7 | 85.15 | 84.87 | 22.8 | 11.6 |
| 1500 | 63.6 ms | 128.7 ms | 76.2 ms | 151.7 ms | 1583.0 | 1511.5 | 90.63 | 90.46 | 15.7 | 7.8 |
| 2000 | 82.9 ms | 169.1 ms | 98.8 ms | 201.5 ms | 2078.5 | 2010.9 | 92.97 | 92.87 | 12.1 | 5.9 |

At every L, beam 32 **≈halves mean latency and roughly doubles QPS at the same
recall** (+0.1–0.2 pt) for only ~5–7% more I/Os — the §11 queue-depth effect holds
across the entire high-recall curve.

**recall@50 (L = 50 / 75 / 100 / 150 / 200):**

| L_search | mean (bw32) | mean (bw4) | I/Os (bw32) | I/Os (bw4) | recall@50 (bw32) | recall@50 (bw4) | QPS (bw32) | QPS (bw4) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 8.2 ms | 6.5 ms | 188.1 | 71.9 | 81.23 | 75.84 | 121.4 | 154.5 |
| 75 | 9.0 ms | 8.3 ms | 208.5 | 96.2 | 85.15 | 81.27 | 111.6 | 120.0 |
| 100 | 9.8 ms | 10.5 ms | 228.6 | 120.4 | 86.67 | 83.89 | 102.3 | 95.4 |
| 150 | 11.3 ms | 14.6 ms | 270.2 | 168.8 | 88.88 | 87.28 | 88.2 | 68.5 |
| 200 | 13.1 ms | 18.7 ms | 315.1 | 217.8 | 90.45 | 89.39 | 76.3 | 53.4 |

The low-recall regime is more nuanced. A wider beam fetches **more nodes per hop**,
so at the same L it issues ~2× the I/Os (e.g. 188 vs 72 at L=50). Those extra
reads (a) **raise recall** at fixed L (wider frontier → +3–5 pts), but (b) at very
small L the added I/O volume *outweighs* the queue-depth saving, so beam 32 is
actually a touch **slower** (8.2 vs 6.5 ms at L=50). From L≥100 the queue-depth
win dominates again and beam 32 is both faster *and* higher recall. Net: at
recall@50 the right `beam_width` is a budget trade-off (more in-flight reads buy
recall but cost bytes), whereas at recall@1000 the deep chain is long enough that
a wide beam is unambiguously better.

Configs: `diskann-benchmark/example/disk-index-enron-t1-beamsweep.json` (beam
sweep), `disk-index-enron-t1-recall1000-bw32.json`,
`disk-index-enron-t1-recall50-bw32.json`; results in the matching `*-out.json`
files.


---

## 12. How a query becomes disk reads: cluster → AlignedRead → SSD

End-to-end mechanics of turning probed centroids into bytes off the SSD, for
reference. Code: `diskann-graphivf/src/storage.rs`,
`diskann-graphivf/src/index.rs` (`Searcher::search_profiled`),
`diskann-disk/src/utils/aligned_file_reader/windows_aligned_file_reader.rs`,
`diskann-platform/src/win/file_handle.rs`.

**On-disk layout.** All clusters live in one file `<prefix>.graphivf_lists`, in
ascending cluster-id order, each record packed back-to-back with no inter-list
padding as `[ids: u32 × count][vectors: Elem × dim × count]`. The companion
`.graphivf_meta` stores per-cluster `counts`; the byte offset of every cluster is
a prefix sum over the record sizes, recomputed on load (`Layout.offsets`). Lists
are **variable length** (one centroid per ~10–20 points), so only the whole file
is zero-padded to a 512-byte multiple — clusters themselves are not padded to a
fixed stride. This is why mean request size ≈ `corpus_bytes / K`.

**Query path** (`search_profiled`):

1. **centroid_search** — graph search over the in-memory centroid (Vamana) graph
   yields the `nlist` nearest cluster ids. Pure RAM; no disk. `nlist` *is* nprobe,
   which is why **`io_count` ≈ nprobe**.
2. **plan_io** — for each non-empty probed cluster, `cluster_window` computes a
   **sector-aligned read window**: `aligned_start = align_down(offset, 512)`,
   `aligned_end = align_up(offset + used_bytes, 512)`, and `inner_offset =
   offset − aligned_start` (where the real data begins inside the window).
   Rounding *outward* to 512 is mandatory because the reads use direct I/O, which
   can only transfer whole 512-byte sectors. This wastes ≤511 B at each end, so
   the issued `aligned_len` is slightly larger than the raw list bytes.
3. One reusable, 512-aligned `scratch` buffer is grown to the sum of all windows
   and carved (via `split_at_mut`) into one disjoint sub-slice per cluster. Each
   becomes an `AlignedRead::<u8, A512>::new(aligned_start, slice)`, whose
   constructor validates that the **offset, buffer length, and buffer memory
   pointer** are all 512-multiples. A constructed `AlignedRead` is a type-level
   witness that the request is legal for direct I/O. At this point
   `profile.io_count = reads.len()` and `profile.bytes_read = total_len`.
4. **disk_read** — all of a query's requests are submitted in a **single batch**:
   `reader.read(&mut reads)`. `reader` is the platform `AlignedFileReader`
   (Windows IOCP, Linux io_uring, buffered elsewhere).

**Inside `WindowsAlignedFileReader`.** Requests are issued in **batches of up to
`MAX_IO_CONCURRENCY = 128`**. For each request it fills an `OVERLAPPED` (carrying
the file offset) and calls `ReadFile` via `read_file_to_slice`; with overlapped
I/O `ReadFile` returns immediately, so up to 128 reads are **in flight on the SSD
at once**. It then drains the I/O completion port with
`get_queued_completion_status` until all reads in the batch complete. A query
probing 1214 clusters is therefore `ceil(1214/128) = 10` batches of concurrent
reads — this is what lets many small random reads run efficiently (§9, §11: the
deep queue is why graph-IVF is bandwidth-bound rather than latency-bound).

**Direct I/O / no page cache.** The file handle is opened once with
`FILE_ATTRIBUTE_READONLY | FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED |
FILE_FLAG_RANDOM_ACCESS` (Linux equivalent: `O_DIRECT`). `FILE_FLAG_NO_BUFFERING`
is exactly why the 512-alignment is required, and it means every byte reported in
`bytes_read` is a real device read with no OS page-cache between the buffer and
the SSD — so the measured IO counts/sizes are not inflated or hidden by caching.

**After the read**, scoring walks each window, skips the leading partial sector
via `inner_offset`, and reinterprets the bytes as `[u32 ids][Elem vectors]`
(zero-copy `bytemuck` casts in `parse_cluster`), scoring the query against each
fetched vector.

**Measured relationships** (MERB-Corpus-12K, see workbook sheet *MERB IO & Req
Size*):

| Quantity | Source | Relationship |
|---|---|---|
| `ios/q` | `reads.len()` = non-empty probed clusters | ≈ nprobe (one request per cluster) |
| `req size (B)` | mean `aligned_len` per window | ≈ `corpus_bytes / K`, rounded up to 512 |
| `bytes/q` | `total_len` = Σ aligned windows | ≈ nprobe × req_size |

So **more centroids → smaller clusters → smaller per-request reads but more
requests** for the same recall target; the 128-deep IOCP batch keeps those many
small reads cheap. (The disk-index Vamana+PQ reader instead issues fixed 4 KB
sector reads, bounded in flight by `beam_width` — see §11.)


---

## Overall conclusions so far

- Search is **I/O-bound** at moderate-to-high nlist (disk read 62–96% of latency);
  at low nlist with many small lists, **centroid-graph search** becomes the floor.
  The disk reads are **device-bandwidth-bound**, not parallelism-bound (§9): the
  SSD saturates even on a single query, so a RAM hot-list cache (not more threads)
  is what cuts per-query latency. With disk removed, centroid-graph search becomes
  the floor at *every* nlist.
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
  - **Per-query latency (mean/p95)** → a **RAM hot-list cache** (§9). The fetch
    path is already optimally parallel/async and the SSD is bandwidth-saturated,
    so reading *fewer bytes* is the only lever. Caching the *largest* lists pays
    off super-linearly in bytes (256 MiB ⇒ 7.6% of lists but 20% fewer disk
    bytes), which is the result that generalizes. Full residency gives ~13× mean /
    ~14× p95, but on this ~0.8 GB corpus that just means "the whole index in RAM"
    (the 1 GiB budget exceeds the 801 MiB payload) — an upper bound, not a
    realistic cache for larger corpora. More query threads *worsen* latency.
- **Graph-IVF vs disk-index is a bandwidth-vs-latency story** (§11): graph-IVF
  reads more bytes but in *one 128-deep batched I/O* (bandwidth-bound), while the
  disk-index beam search reads far fewer bytes as a *long dependency chain* whose
  effective queue depth is only `beam_width` (random-I/O-latency-bound). At T=1,
  L=1000, raising `beam_width` 4 → 32 halves disk-index latency (88 → 44 ms) at
  the same I/O count and recall — confirming the gap was queue depth, not data
  volume. On this <1 GB corpus graph-IVF wins on latency; on corpora ≫ RAM the
  ordering flips, since graph-IVF's bytes/query grow with `nlist × list_size`
  while disk-index's I/O count stays bounded by graph hops.
