<!--
 Copyright (c) Microsoft Corporation.
 Licensed under the MIT license.
-->

# `diskann-graphivf`

A hybrid **graph + clustered-IVF** approximate nearest neighbor index. Two parts:

1. An in-memory full-precision DiskANN graph over the cluster **centroids** (routing).
2. An on-disk file holding, per cluster, the corpus vectors assigned to it laid out
   contiguously so one read fetches a whole **inverted list**.

**Search:** graph-find the `nlist` nearest centroids → one batched read of those lists
→ exhaustively score the query against the fetched vectors → top-k.

Two build paths share this on-disk format and the same flushed-index search path:

- **Online** (primary) — mutable streaming partition: route inserts, split
  overflowing clusters with local reassignment, remove deleted points, and
  dissolve underfull clusters onto nearby survivors. The live cluster count
  emerges from the stream. See [`ONLINE.md`](ONLINE.md) for the algorithm.
- **Static** — batch build: sample/select centroids, k-means, assign, write lists.
  Useful as a fixed-`k` baseline.

The library API lives in [`src/`](src/); all experiments run through the shared
benchmark harness — see the [graph-IVF section of
`diskann-benchmark/README.md`](../diskann-benchmark/README.md#graph-ivf) and the worked
configs in [`diskann-benchmark/example/graph-ivf-*.json`](../diskann-benchmark/example).
The corpus-preparation and diagnostic tools that the harness does not cover are
catalogued in [`scripts/README.md`](scripts/README.md).

---

## 1. Data preparation

All tools read/write the DiskANN binary matrix format:
`[npoints: u32][ndims: u32][row-major data]`.

A build accepts the corpus in several on-disk element types, chosen with the
`data_type` field: **`minmax8`** (`MinMaxElement<8>`: 8-bit quantized codes + embedded
per-vector min/max), **`float16`**, **`float32`**, **`uint8`**, or **`int8`**.
Clustering always runs in decoded `f32`; the chosen type is only what the inverted
lists store on disk. For the plain scalar types, feed the corpus `.bin` directly. For
`minmax8`, compress full-precision inputs **once**, for both the corpus and the
queries, before any build or search:

```text
cargo run --release --example compress_minmax -- corpus_f32.bin  corpus_minmax8.bin  f32
cargo run --release --example compress_minmax -- queries_f32.bin queries_minmax8.bin f32
```

| Input | Format | Notes |
| --- | --- | --- |
| corpus | `.bin` in the `data_type` element type | one row per corpus vector |
| queries | matching `.bin` | same width/type as the corpus; scored directly in `T` |
| groundtruth | `.bin` of `u32` | `num_queries × gt_dim`, true neighbor ids per query |

Cosine: normalize vectors **before** compression (stored rows are written verbatim).

---

## 2. Shared parameters

**Centroid graph** (`GraphParams`, same fields for both build paths): `graph_degree=32`,
`graph_slack=1.2`, `graph_l_build=64`, `graph_alpha=1.2`.

**Metric** — clustering and assignment are **always squared-L2**. The `distance` field
only sets how the *loaded* index scores at search time:

| `distance` | Search scoring / graph navigation |
| --- | --- |
| `squared_l2` | squared-L2 |
| `cosine_normalized` | squared-L2 (same ranking on unit vectors, no re-normalization pass) |
| `inner_product` | inner product (MIPS) |
| `cosine` | cosine — **static builds only**; an online build writes rows verbatim and so cannot normalize them |

The graph-navigation column describes a flushed index rebuilt by `load`. Live
`OnlineSearcher` queries always navigate the mutable centroid graph with L2,
while scoring candidates with the configured metric.

**Search** (the harness's `search_phase`):

| Field | Meaning |
| --- | --- |
| `cluster_fractions` | fractions of the index's clusters to probe, in `(0.0, 1.0]` — one search per value |
| `centroid_search_alpha` | centroid-graph search list as a multiple of `nlist`; defaults to 1.5 |
| `recall_at` | the `k` recall is measured at — one value, or a list scored from a single search |

For `C` live clusters, the harness computes `nlist = ceil(cluster_fraction · C)`.
It reports both values. A runbook recomputes `C` at every search stage, so the same
fraction continues to mean the same share while the index grows and shrinks. The
library's `SearchParams` remains concrete and continues to accept `nlist` directly.

The centroid beam is `max(128, ceil(alpha · nlist))`. Expressing it as a multiple
rather than a constant matters most on an index that changes size: a constant has
to be picked for the largest `nlist` a run will reach, and until it does the beam
is pure overhead charged to every query.

A `Load` job needs only `data_type` and the index prefix — the same `data_type` the index
was built with, since it selects the backend that decodes the lists.

---

## 3. Online index (primary)

No target cluster count is required: points stream in, clusters **split** when
they exceed `split_threshold`, and delete-driven **dissolves** can reduce the
live count. A plain `"graph-ivf-source": "Online"` job inserts the corpus once;
an `OnlineRunbook` job replays insert, delete, and live-search stages.

### Build parameters

| Field | Default | Meaning |
| --- | --- | --- |
| `split_threshold` | *(required)* | split a cluster once it holds **more** than this many points (dominant granularity knob) |
| `merge_threshold` | 0 | dissolve a cluster once deletes leave it below this size; `0` disables dissolves and `2 · merge_threshold ≤ split_threshold` is required |
| `min_clusters` | 1 | live-cluster floor enforced when admitting dissolves |
| `batch_size` | 1 | points routed and planned together per insert batch; larger batches enable parallel routing and joint split k-means |
| `warmup_centroids` | 100 | initial centroids from a light k-means over a corpus prefix |
| `warmup_points` | 10000 | leading corpus points used for the warmup |
| `warmup_iters` | 15 | Lloyd iterations for that warmup |
| `num_threads` | *(required)* | worker threads |
| `assign_l` | 64 | centroid-graph search-list size for routing inserts |
| `two_means_iters` | 12 | Lloyd iterations for split k-means (two children per admitted parent) |
| `distance` | *(required)* | live/flushed candidate-scoring metric; `cosine` is rejected (rows are stored verbatim) |
| `normalize` | `false` | L2-normalize warmup and split-child centroids (unit-sphere corpora) |
| `capacity_mult` | 3 | centroid id-budget headroom (`≈ capacity_mult · 2N / split_threshold`) |
| `reassign_neighbors` | 8 | split-neighbor candidates and maximum survivor landing sites per dissolve |
| `reassign_l` | `max(s, assign_l)` | search-list size for split-neighbor and dissolve-survivor selection |
| `max_clusters` | *(omit)* | omitted/`null` = uncapped growth; else a hard cap on live clusters |
| `data_type` | *(required)* | on-disk element type: `minmax8` \| `float16` \| `float32` \| `uint8` \| `int8` |
| `telemetry_csv` | *(omit)* | split telemetry path; an `OnlineRunbook` also writes a derived `<stem>_merges.<ext>` sibling |

For an insert-only uncapped build, equilibrium live clusters are approximately
`2 · num_points / split_threshold`. Under churn the count also depends on the
delete stream, `merge_threshold`, and `min_clusters`.

Writes `<save_path>.graphivf_{lists,meta,centroids.fbin}`. If requested, a plain
online build writes the split telemetry CSV; an `OnlineRunbook` writes that file
plus its derived merge sibling. Unlike the retired `build_online` script, the
prefix is used verbatim — encode the knobs in `save_path` yourself if you want
them in the filename.

Example — ~16384 clusters (`th=106`), `s=5` reassignment neighbors:

```json
{
  "graph-ivf-source": "Online",
  "data_type": "minmax8",
  "data": "corpus_minmax8.bin",
  "distance": "squared_l2",
  "dim": 384,
  "split_threshold": 106,
  "reassign_neighbors": 5,
  "max_clusters": 16384,
  "graph_degree": 32,
  "graph_slack": 1.2,
  "graph_l_build": 64,
  "graph_alpha": 1.2,
  "num_threads": 16,
  "seed": 0,
  "save_path": "/abs/path/out_prefix_th106_minmax8"
}
```

### Live mutation, search, and failure behavior

The library's [`OnlineClusterer`](src/online.rs) exposes `insert_batch` and
`delete_batch`. Both validate and compute projected cluster sizes before
changing live state; each also completes its applicable fallible routing,
neighborhood search, snapshots, and k-means. Structural commit goes through one
private owner of the centroid table and graph, while point movement goes through
one owner of the inverted lists and reverse assignments. If a commit fails after
an irreversible graph operation, [`OnlineClusterer::is_poisoned`](src/online.rs)
becomes true and later insert, delete, search, and flush operations return
[`GraphIvfError::Poisoned`](src/error.rs); preparation errors leave the clusterer
usable. See [the detailed failure contract](ONLINE.md#3d-state-ownership-and-failure-semantics).

Before flush, [`OnlineClusterer::searcher`](src/online.rs) opens an
[`OnlineSearcher`](src/online/search.rs) over the current full-precision
partition. `search` is an allocating convenience method. `search_into` reuses a
caller-owned vector as both the candidate buffer and sorted top-k output, so
steady-state calls avoid allocating and copying a second result; it returns
per-query [`OnlineSearchStats`](src/online/search.rs), while `points_scanned()`
on the handle is cumulative. Keep one handle per worker. Its immutable borrow
prevents concurrent mutation.

The benchmark's `OnlineRunbook` source uses this live API at each search stage
while replaying a BigANN runbook. It reports recall at every configured depth,
latency, `mean_points_scanned`, and the scanned percentage of the current live
set. Runbook insert/delete ranges are internally sub-batched by `batch_size`, while
searches happen only at explicit runbook stages. It flushes once at the end, so an
optional outer `search_phase` can still measure the resulting on-disk index.

For a plain `Online` job, `search_phase` sweeps `cluster_fractions` over the index after
it is flushed, printing one row per value with its effective `nlist`: one recall column per `recall_at`,
mean/p95/p999 latency (µs), bytes read/query, IOs/query, request bytes, QPS, and
a per-stage latency breakdown (preprocess, centroid search, plan I/O, disk read,
score, top-k). Each search runs to the largest `recall_at` and every listed `k`
is scored from that one result set, so `[50, 1000]` costs what `50` alone did.
Point a later `Load` job at the same prefix to re-sweep without rebuilding.

---

## 4. Static index

Centroids are chosen once and fixed; the target cluster count `num_clusters` is explicit.
Use it as a fixed-`k` baseline against the online build, via a
`"graph-ivf-source": "Static"` job.

### Build parameters

| Field | Meaning |
| --- | --- |
| `num_clusters` | number of centroids `k` |
| `sample_size` | corpus rows sampled for the Forgy k-means seeding (`>= num_clusters`) |
| `kmeans_iters` | Lloyd iterations (`0` = use initial centers unrefined) |
| `num_threads` | build worker threads |
| `assign_l` | centroid-graph search-list size for point→centroid assignment |
| `assign_method` | `"Exact"` (default) or `{ "Graph": { "rebuild_every": n, "rerank": n } }` |
| `empty_clusters` | policy for clusters emptied during refinement: `"PreserveOld"` (default), `"Zero"`, `"ReseedFarthest"` |
| `distance` | search-time metric (clustering/assignment are always squared-L2) |
| `seed` | RNG seed for sampling and k-means |

It writes `<save_path>.graphivf_{lists,meta,centroids.fbin}`.

Example — sampled build, 16384 clusters:

```json
{
  "graph-ivf-source": "Static",
  "data_type": "minmax8",
  "data": "corpus_minmax8.bin",
  "distance": "squared_l2",
  "dim": 384,
  "num_clusters": 16384,
  "sample_size": 200000,
  "kmeans_iters": 10,
  "assign_l": 32,
  "graph_degree": 32,
  "graph_slack": 1.2,
  "graph_l_build": 64,
  "graph_alpha": 1.2,
  "num_threads": 16,
  "seed": 0,
  "save_path": "/abs/path/out_prefix_16384_minmax8"
}
```

The **same** `search_phase` shape sweeps either build path, and a `Load` job searches
either one.

---

## 5. Output artifacts

| File | Contents |
| --- | --- |
| `<prefix>.graphivf_centroids.fbin` | centroid matrix (`f32`), reloaded to rebuild the graph |
| `<prefix>.graphivf_lists` | contiguous per-cluster inverted lists (`T` = the `data_type` element type) |
| `<prefix>.graphivf_meta` | layout: counts, offsets, dim, metric, element size, graph params |
| `<telemetry_csv>` | plain online and online runbook — per-split telemetry timeline |
| `<stem>_merges.<ext>` | online runbook only — per-dissolve telemetry timeline derived from `telemetry_csv` |

---

## See also

- [graph-IVF in `diskann-benchmark/README.md`](../diskann-benchmark/README.md#graph-ivf) —
  how to configure and run a job, and worked configs for each source.
- [`ONLINE.md`](ONLINE.md) — online insert/delete, split/dissolve, failure, and search internals.
- [`scripts/README.md`](scripts/README.md) — corpus preparation and centroid-graph diagnostics.
