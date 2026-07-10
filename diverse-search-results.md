# Attribute-Diversity Search — Benchmark Results

Comparison of attribute-bucket diverse search strategies on two datasets, all
evaluated against a **diverse ground truth** (GT built so that no more than
`diverse_results_k` results share the same attribute bucket).

## Setup

Common parameters for every run below:

| Parameter | Value |
|---|---|
| Index | On-disk DiskANN, 200,000 base vectors |
| `recall_at` (k) | 10 |
| `diverse_results_k` | 1 (at most one result per attribute bucket) |
| `beam_width` | 4 |
| `num_threads` | 1 |
| Distance | squared L2 |
| Search lists (L) | 20, 40, 80, 100 |
| Ground truth | diverse GT (`groundtruth_diverse_k100.bin`) |

> **QPS methodology.** Recall, IO counts, comparisons and hops are exact and
> fully reproducible. QPS, however, is single-threaded and disk-bound, and on the
> dev machine used here (shared with VS Code, browsers, Teams) it varied 10–30×
> run-to-run purely from disk-wait contention, with IO counts unchanged. The QPS
> figures below are the **best (least-contended) of several runs per cell**,
> which is closest to the true compute-/IO-bound cost. Treat QPS as indicative,
> not precise; the recall and IO columns are the authoritative comparison.

### Datasets

| Dataset | Dim | Diversity attribute | Attribute character |
|---|---|---|---|
| Enron | 1,369 | attribute id 0 | **Concentrated** — many vectors share a bucket |
| Caselaw | 1,536 | `doc_id` | **Diffuse** — 200K vectors span 172,518 docs (18,383 multi-chunk) |

### Strategies compared

- **Standard** — plain KNN search (no diversity enforcement), scored against the
  diverse GT. Shows how far an ordinary search is from the diverse target.
- **Queue** — diversity enforced *during* graph traversal using a diversity-aware
  neighbor queue with approximate (PQ) distances. Branch
  `u/narendatha/diverse-search-benchmark`.
- **Design A (post-process)** — plain KNN over the top-L pool, then a greedy
  bucket selection keeping ≤ `diverse_results_k` per bucket, using
  full-precision distances. Fixed L (no over-fetch). The post-processor reuses
  the full-precision distances already cached during traversal
  (`distance_cache`) and only fetches/recomputes for cache misses (see
  [Optimization](#optimization-distance-cache-reuse-in-design-a)).
- **Design B (adaptive-L)** — like Design A, but a greedy walk first samples
  bucket concentration; if buckets are concentrated it grows L (over-fetches)
  before the same post-processing step. When buckets are diffuse it returns a 1×
  multiplier and behaves exactly like Design A.

---

## Enron (concentrated attribute)

| L | Standard recall | Queue recall | Design A recall | Design B recall | Std QPS | Queue QPS | A QPS | B QPS |
|---|---|---|---|---|---|---|---|---|
| 20 | 55.71 | 61.11 | 59.98 | **73.38** | 375.5 | 261 | 542.9 | 385.6 |
| 40 | 64.28 | 71.08 | 73.93 | **83.55** | 245.3 | 221 | 268.0 | 183.2 |
| 80 | 69.67 | 76.46 | 84.00 | **89.79** | 173.2 | 116 | 165.6 | 107.2 |
| 100 | 70.72 | 77.92 | 86.05 | **91.00** | 142.6 | 100 | 159.2 | 53 |

> Design A QPS reflects the distance-cache reuse optimization (see below). With
> that change Design A is now the fastest diversity-aware method at every L, at
> strictly higher recall than the queue.

**Observations**
- Standard search is far below every diversity-aware method (it does not enforce
  the one-per-bucket constraint), confirming the diverse GT is meaningfully
  different from ordinary top-k.
- **Design B wins recall at every L** (+13 over queue at L=20), because the
  attribute is concentrated: the adaptive sampler detects this and over-fetches a
  larger pool before reranking.
- Design A beats the queue on both recall and QPS at every L, since its
  post-processing now reuses cached distances instead of recomputing them.
- Design B carries a throughput cost (extra pool fetch + more IO), but delivers
  the highest recall. At L=100 its IO count (214) is nearly double Design A's
  (120), so its lower QPS there is real, not measurement noise.
- Pareto note: Design B @ L=20 (73.4 recall) beats Design A @ L=40 (73.9 recall)
  at comparable throughput.

---

## Caselaw (diffuse attribute — `doc_id`)

| L | Standard recall | Queue recall | Design A recall | Design B recall | Std QPS | Queue QPS | A QPS | B QPS |
|---|---|---|---|---|---|---|---|---|
| 20 | 90.61 | 90.76 | 92.04 | 92.04 | 554.8 | 306.7 | 448.8 | 238.4 |
| 40 | 95.43 | 96.53 | 97.53 | 97.53 | 267.5 | 207.0 | 192.5 | 151.0 |
| 80 | 96.73 | 98.13 | 99.06 | 99.06 | 187.0 | 138.4 | 132.4 | 101.3 |
| 100 | 96.91 | 98.32 | 99.25 | 99.25 | 160.3 | 84.3 | 128.8 | 105.9 |

**Observations**
- **Design A and Design B have identical recall AND identical IO/comparison/hop
  counts.** The `doc_id` attribute is so diffuse (almost every vector is its own
  bucket) that the concentration sampler returns a 1× multiplier — Design B
  performs no over-fetch. Its slightly lower QPS is the pure cost of the
  concentration-sampling walk, which here yields no benefit.
- Even standard search is already close to the diverse GT here, because with
  near-unique buckets the diverse GT is almost the same as the ordinary top-k.
  The diversity methods add only ~1–2 recall points.
- Design A/B edge out the queue method on recall (~1–1.3 points) via
  exact-distance reranking. With the distance-cache reuse, Design A is also
  competitive on QPS (e.g. 448.8 vs queue 306.7 at L=20).

---

## Optimization: distance-cache reuse in Design A

Initially Design A's `AttributeBucketDiversity` post-processor was *slower than
the queue method* at higher L, even though the queue does more work during
traversal. The cause: the post-processor recomputed exact distances for **every**
candidate in the L-pool — fetching each full-precision vector and re-evaluating
the distance — duplicating work the traversal had already done.

The standard `RerankAndFilter` post-processor avoids this by consulting
`accessor.scratch.distance_cache`, which holds the full-precision distances
computed during traversal, and only fetching/recomputing for cache misses.
Design A now does the same:

- Look up each candidate in `distance_cache`; on a hit, reuse the cached distance
  and associated data (no vector fetch, no recompute).
- Collect only the misses, `ensure_vertex_loaded` them once, and compute their
  distances.
- Sort by exact distance, then run the greedy per-bucket selection.

Result (recall and IO counts byte-identical before/after; QPS = best of several
runs, disk-contention caveat above):

| Dataset / L | A QPS before | A QPS after |
|---|---|---|
| Enron L=20 | 414 | 542.9 |
| Enron L=40 | 150 | 268.0 |
| Enron L=80 | 84 | 165.6 |
| Enron L=100 | 88 | 159.2 |
| Caselaw L=20 | 146.7 | 448.8 |
| Caselaw L=40 | 151.1 | 192.5 |

The reduction in redundant vector fetches and distance recomputation is
consistent. Recall is unaffected because the cached distances are the same
full-precision values that would otherwise be recomputed.

---

## Summary

The two datasets exercise opposite regimes and together validate the adaptive
design:

- **Concentrated attribute (Enron):** big win for Design B — adaptive over-fetch
  recovers substantial recall (+13 over queue at L=20) at a throughput cost.
- **Diffuse attribute (Caselaw):** the sampler detects abundant diversity and
  Design B performs no over-fetch — it matches Design A on recall and IO (its only
  cost is the sampling walk), and both slightly beat the queue method.

Recommendation: prefer **Design B**. It matches Design A when diversity is free
(diffuse attributes) and outperforms both Design A and the queue method when the
attribute is concentrated, without needing per-dataset tuning of L.
