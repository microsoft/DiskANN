# Attribute-Diversity Search — Benchmark Results

Comparison of attribute-bucket diverse search strategies across several dataset
and attribute regimes, all
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
| Caselaw | 1,536 | `doc_id` | **Diffuse, uncorrelated** — 200K vectors span 172,518 docs (18,383 multi-chunk) |
| Caselaw | 1,536 | `court_jurisdiction` | **Concentrated + distance-correlated** — 60 courts, largest bucket 47,358 (23.7%), median 1,668 |
| YFCC | 192 | `camera` model | **Concentrated, ~uncorrelated** — 67 distinct cameras, largest bucket 67,783 (33.9%), median bucket 110 |

Two axes matter for diversity search: how **concentrated** the buckets are (does
the sampler need to over-fetch?) and how **correlated** the attribute is with
distance (are same-bucket vectors clustered together in the search pool?). The
caselaw `court_jurisdiction` attribute is the interesting case: cases from the
same court share citations and legal boilerplate, so their embeddings cluster —
the attribute is *both* concentrated and strongly correlated with distance.

### Strategies compared

- **Standard** — plain KNN search (no diversity enforcement), scored against the
  diverse GT. Shows how far an ordinary search is from the diverse target.
- **Queue** — diversity enforced *during* graph traversal using a diversity-aware
  neighbor queue with approximate (PQ) distances. Branch
  `u/narendatha/diverse-search-benchmark`.
- **Design A (post-process)** — plain KNN over the top-L pool, then a greedy
  bucket selection keeping ≤ `diverse_results_k` per bucket, using
  full-precision distances. Fixed L (no over-fetch).
- **Design B (adaptive-L + full-visited rerank)** — like Design A, but a greedy
  walk first samples bucket concentration; if buckets are concentrated it grows L
  (over-fetches) before the same post-processing step. Crucially, it also hands
  the post-processor the **entire distance-sorted set of nodes visited during
  traversal**, not just the `L`-bounded frontier queue. This mirrors the
  candidate-pool decoupling used by filtered search: diverse-but-distant nodes
  that the walk encountered — but that the frontier evicted in favour of closer
  same-bucket nodes — remain eligible as final answers. When buckets are diffuse
  it returns a 1× multiplier (no over-fetch) but still reranks the full visited
  set, so it can beat Design A even without over-fetching.

---

## Enron (concentrated attribute)

| L | Standard recall | Queue recall | Design A recall | Design B recall | Std QPS | Queue QPS | A QPS | B QPS |
|---|---|---|---|---|---|---|---|---|
| 20 | 55.71 | 61.11 | 59.98 | **84.16** | 375.5 | 261 | 542.9 | 385.6 |
| 40 | 64.28 | 71.08 | 73.93 | **88.88** | 245.3 | 221 | 268.0 | 183.2 |
| 80 | 69.67 | 76.46 | 84.00 | **92.83** | 173.2 | 116 | 165.6 | 107.2 |
| 100 | 70.72 | 77.92 | 86.05 | **93.74** | 142.6 | 100 | 159.2 | 53 |

**QPS vs recall @ L=100** (up and to the right is better)

![Enron QPS vs recall](assets/diverse-search/enron-qps-vs-recall.png)

**Observations**
- Standard search is far below every diversity-aware method (it does not enforce
  the one-per-bucket constraint), confirming the diverse GT is meaningfully
  different from ordinary top-k.
- **Design B wins recall at every L** (+23 over queue at L=20), because the
  attribute is concentrated: the adaptive sampler detects this and over-fetches a
  larger pool before reranking, and the full-visited rerank keeps distant
  distinct-bucket nodes the frontier would have dropped.
- Design A beats the queue on both recall and QPS at every L.
- Design B carries a throughput cost (extra pool fetch + more IO), but delivers
  the highest recall. At L=100 its IO count (320) is well over double Design A's
  (112), so its lower QPS there is real, not measurement noise.
- Pareto note: Design B @ L=20 (84.2 recall) beats Design A @ L=40 (73.9 recall)
  at comparable throughput.

---

## Caselaw (diffuse, uncorrelated attribute — `doc_id`)

| L | Standard recall | Queue recall | Design A recall | Design B recall | Std QPS | Queue QPS | A QPS | B QPS |
|---|---|---|---|---|---|---|---|---|
| 20 | 90.61 | 90.76 | 92.04 | **95.97** | 554.8 | 306.7 | 448.8 | 238.4 |
| 40 | 95.43 | 96.53 | 97.53 | **98.00** | 267.5 | 207.0 | 192.5 | 151.0 |
| 80 | 96.73 | 98.13 | 99.06 | **99.11** | 187.0 | 138.4 | 132.4 | 101.3 |
| 100 | 96.91 | 98.32 | 99.25 | **99.26** | 160.3 | 84.3 | 128.8 | 105.9 |

**QPS vs recall @ L=100** (up and to the right is better)

![Caselaw QPS vs recall](assets/diverse-search/caselaw-qps-vs-recall.png)

**Observations**
- **Design B now edges ahead of Design A on recall at equal IO** (95.97 vs 92.04
  at L=20), even though the `doc_id` attribute is so diffuse that the
  concentration sampler returns a 1× multiplier and Design B performs *no
  over-fetch*. The gain comes purely from the full-visited rerank: Design A
  reranks only the `L`-bounded frontier, whereas Design B reranks every node the
  walk touched, so it recovers near-tie candidates the frontier dropped. IO,
  comparison and hop counts are identical to Design A — this is free recall.
- Even standard search is already close to the diverse GT here, because with
  near-unique buckets the diverse GT is almost the same as the ordinary top-k.
  The diversity methods add only a few recall points at higher L.
- Design B posts the best recall of any method here (99.26 at L=100), and the gap
  over Design A is largest at low L where the frontier is tightest.

---

## Caselaw (concentrated + distance-correlated attribute — `court_jurisdiction`)

Same 200K caselaw vectors and index as above, but the diversity attribute is now
the natural `court_jurisdiction` field (60 US courts) instead of `doc_id`. This
attribute is **both** concentrated (largest bucket 23.7%, median 1,668) **and
strongly correlated with distance** — cases from the same court cluster together
in embedding space. This is the hardest and most realistic regime: a query's
nearest neighbors are dominated by a few jurisdictions, so enforcing one-per-bucket
forces the search to reach far past the plain top-L pool.

| L | Standard recall | Queue recall | Design A recall | Design B recall | Std QPS | Queue QPS | A QPS | B QPS |
|---|---|---|---|---|---|---|---|---|
| 20 | 34.74 | 71.35 | 46.30 | **81.89** | 387.9 | 434.9 | 413.5 | 317.1 |
| 40 | 35.37 | 76.19 | 59.68 | **85.77** | 394.3 | 278.7 | 303.8 | 203.3 |
| 80 | 35.79 | 78.72 | 69.74 | **89.16** | 212.2 | 176.4 | 211.1 | 114.2 |
| 100 | 35.91 | 79.12 | 72.59 | **90.19** | 165.6 | 152.1 | 174.5 | 96.0 |

**QPS vs recall @ L=100** (up and to the right is better)

![Caselaw jurisdiction QPS vs recall](assets/diverse-search/caselaw-jurisdiction-qps-vs-recall.png)

**Observations**
- **This is the regime the full-visited rerank was built for, and it now inverts
  the earlier ranking in Design B's favour: Design B beats the queue at every L**
  (81.9 vs 71.4 at L=20; 90.2 vs 79.1 at L=100). Because the attribute is
  correlated with distance, the `L`-bounded frontier is *saturated* with the
  dominant jurisdictions — but the greedy walk still *visits* far-away
  distinct-jurisdiction nodes on its way there. Design A (and the old
  frontier-only Design B) threw those away; handing the full visited set to the
  post-processor keeps them, which is exactly what recovers the missing buckets.
- **Design A still falls far behind** (46.3 at L=20): reranking only the plain
  top-L pool cannot recover buckets that never enter that pool. The difference
  between Design A and Design B here (46.3 → 81.9 at L=20) is entirely the
  candidate-pool decoupling plus adaptive over-fetch.
- Design B's win costs throughput — it fetches a larger pool (IO 193 vs queue's
  114 at L=100), so its QPS is the lowest of the methods there. On the
  QPS-vs-recall plot Design B sits up-and-to-the-left of the queue: markedly
  higher recall at lower QPS, a genuine recall/throughput tradeoff rather than the
  queue-dominates picture seen before the accumulator.
- Standard search collapses to ~35% and never improves with L — with buckets both
  concentrated and distance-correlated, the ordinary top-k is almost entirely the
  wrong jurisdictions.
- Takeaway: **once the post-processor sees the full visited set, in-traversal
  diversity (queue) no longer has an edge even on distance-correlated attributes** —
  the distant distinct-bucket nodes the walk already visited are enough for
  post-processing to beat it, at the cost of a heavier pool fetch.

---

## YFCC (concentrated attribute — `camera` model)

YFCC-100M image embeddings (200K subset, 192-dim). The diversity attribute is the
`camera` model tag: 67 distinct cameras, but very skewed — the largest bucket holds
33.9% of all vectors and the median bucket only 110, so this is a strongly
concentrated regime (like Enron).

| L | Standard recall | Queue recall | Design A recall | Design B recall | Std QPS | Queue QPS | A QPS | B QPS |
|---|---|---|---|---|---|---|---|---|
| 20 | 49.35 | 79.26 | 66.92 | **96.50** | 347.0 | 592.6 | 499.4 | 442.6 |
| 40 | 49.60 | 81.23 | 83.54 | **97.03** | 499.7 | 397.3 | 441.5 | 261.8 |
| 80 | 49.72 | 81.60 | 92.38 | **97.30** | 256.5 | 240.2 | 248.3 | 140.6 |
| 100 | 49.73 | 81.78 | 93.98 | **97.49** | 227.1 | 186.1 | 192.6 | 111.9 |

**QPS vs recall @ L=100** (up and to the right is better)

![YFCC QPS vs recall](assets/diverse-search/yfcc-qps-vs-recall.png)

**Observations**
- Standard search is stuck at ~49.5 recall for every L — with a third of the
  corpus in one camera bucket, ordinary top-k returns many same-bucket neighbors
  that the diverse GT rejects. Growing L does not help it at all.
- **Design B wins recall at every L** (+17 over the queue at L=20), because the
  adaptive sampler detects the concentration and over-fetches before reranking the
  full visited set.
- **The queue method plateaus at ~81%** — its approximate (PQ) in-traversal
  diversity saturates and extra L barely helps. Design A overtakes it from L=40
  onward and reaches 94% at L=100; Design B is far ahead of the queue at every L
  (96.5 already at L=20).
- Design B's higher recall again costs throughput (its IO roughly triples by
  L=100: 286 vs Design A's 112), so its lower QPS is real work, not noise.
- Pareto note: Design B @ L=20 (96.5 recall) already beats Design A at any L
  and the queue at any L, at competitive throughput.

---

## Summary

The datasets exercise the two axes that govern diverse search — bucket
**concentration** and attribute/distance **correlation**:

- **Concentrated, ~uncorrelated attribute (Enron, YFCC):** big win for Design B —
  adaptive over-fetch plus the full-visited rerank recover substantial recall
  (+23 over queue on Enron at L=20; +17 on YFCC at L=20) at a throughput cost. On
  YFCC the queue method plateaus near 81% while Design B climbs to 97%.
- **Diffuse attribute (Caselaw `doc_id`):** the sampler detects abundant diversity
  and Design B performs no over-fetch, so its IO matches Design A exactly — yet it
  still edges ahead on recall (95.97 vs 92.04 at L=20) because it reranks the full
  visited set rather than only the frontier. Free recall at no extra IO.
- **Concentrated + distance-correlated attribute (Caselaw `court_jurisdiction`):**
  the earlier version of Design B fell just below the queue here (79.0 vs 79.1 at
  L=100). Decoupling the post-processor's candidate pool from the `L`-bounded
  frontier — keeping every distinct-bucket node the walk visited — flips this: Design
  B now beats the queue at every L (90.2 vs 79.1 at L=100), at the cost of a heavier
  pool fetch (IO 193 vs 114). In-traversal diversity no longer holds an edge even in
  its best-case regime.

Recommendation: with the candidate-pool decoupling, **Design B is the strongest
recall strategy across all four regimes**, including the distance-correlated case
that previously favoured the in-traversal queue. Its cost is throughput: when
buckets are concentrated it over-fetches (2–3× the IO of Design A), so on a
recall/QPS frontier it trades throughput for recall rather than dominating
outright. Design A remains the cheapest option whenever the top-L pool already
contains enough distinct buckets (diffuse or weakly-correlated attributes) and the
extra recall from the full-visited rerank is not needed.
