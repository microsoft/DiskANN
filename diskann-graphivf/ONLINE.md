<!--
 Copyright (c) Microsoft Corporation.
 Licensed under the MIT license.
-->

# Online graph-IVF index — build & search

The **online graph-IVF index** is an inverted-file (IVF) ANN index whose
partition is built **incrementally**, one point at a time, instead of by a
single batch k-means pass. Cluster centroids are organized in an in-memory
**Vamana graph** used both to route inserts during the build and to select
lists to probe at query time; the inverted lists themselves live on disk. The
index is built in memory and saved and served over disk.

This document describes the incremental clusterer
([`OnlineClusterer`](src/online.rs)) and the shared search path
([`Searcher`](src/index.rs)). For the batch build see
[`CentroidInit`](src/index.rs); the two builds differ only in how the partition
is produced, not in how it is stored or searched.

---

## High-level model

- **Two-level index.** Level 1 is a set of `k` centroids indexed by a Vamana
  graph (in memory, always `f32`). Level 2 is `k` variable-length **inverted
  lists** on disk, list `c` holding the stored vectors of the points assigned to
  centroid `c`.
- **Search = route then scan.** A query navigates the centroid graph to its
  `nlist` nearest centroids, reads those lists from disk, and exhaustively scores
  the query against their members.
- **Online build = stream, route, split.** Points are streamed in insert order;
  each is routed to its nearest centroid via the same graph. When a cluster
  overflows a confugrable size threshold it is **split** in two and its neighborhood is
  locally **reassigned**. The centroid count is not fixed up front — it grows
  with the data.

```mermaid
flowchart LR
    Q([query]) -.route.-> C1
    subgraph MEM["Level 1 · centroid Vamana graph — in memory, f32"]
        C0((c0)) --- C1((c1))
        C1 --- C2((c2))
        C1 --- C3((c3))
        C2 --- C3
    end
    subgraph DISK["Level 2 · inverted lists — on disk, element type T"]
        L0["c0 · ids u32 + vectors T"]
        L1["c1 · ids u32 + vectors T"]
        L2["c2 · ids u32 + vectors T"]
        L3["c3 · ids u32 + vectors T"]
    end
    C0 -.owns list.-> L0
    C1 -.-> L1
    C2 -.-> L2
    C3 -.-> L3
```

### Key assumptions & scope

- **In-memory build.** The driver loads the stored corpus `T` and decodes a full
  `f32` copy for clustering (row `pid` = point `pid`). It then "streams" points
  by feeding row indices to `insert`; there is no per-insert disk I/O. This is
  not an out-of-core builder, so budget memory for both representations.
- **Clustering is always squared-L2.** Routing, 2-means splits, and reassignment
  all run in full-precision `f32` under squared-L2. The configured
  [`Metric`](src/params.rs) controls search-time routing/scoring after load. For
  cosine-style or normalized inner-product data, the caller must pre-normalize
  corpus and query vectors; `--normalize` normalizes centroids, not the input.
- **Decoupled clustering vs. stored precision.** Clustering uses `f32`, but the
  inverted lists are written from the corpus in its on-disk element type `T`,
  copied verbatim at flush. `T` is chosen at build time with `--format`
  (`minmax8` — 1 B/component, `f16` — 2 B, or `f32` — 4 B) and is recorded in the
  metadata, so search tooling can recover it without being told. The centroid
  graph is always `f32`.
- **Single writer.** The build is single-threaded at the insert level (a thread
  pool is used only inside 2-means and graph construction). There is no
  concurrent insert/search; the index is immutable once flushed.

---

## Quick start

Build, then sweep. `--split-threshold` is the only required knob; everything
else has a default (full flag reference in
[`scripts/README.md`](scripts/README.md)).

```text
# build: ~16384 clusters at th=106, s=5 reassignment neighbors, f16 lists
cargo run --release --example build_online -- \
    corpus_f16.bin out_prefix --split-threshold 106 \
    --reassign-neighbors 5 --max-clusters 16384 --format f16

# search: sweep nlist on the result (queries must be in the same element type)
cargo run --release --example sweep -- \
    out_prefix_th106_f16 "164,410,656,901,1147,1638,2458" 1 \
    queries_f16.bin groundtruth.bin
```

The build writes `<out_prefix>_th<split_threshold>_<format>.graphivf_*` plus a
`.splits.csv` telemetry file. `sweep` reads the stored element type from the
metadata, so the same command serves `minmax8`, `f16`, and `f32` indexes.

---

## On-disk format (shared with the batch build)

Written next to a path prefix by `flush` (see [`storage`](src/storage.rs)):

| File | Contents |
| --- | --- |
| `<prefix>.graphivf_centroids.fbin` | The `k × logical_dim` centroid matrix, always `f32`. |
| `<prefix>.graphivf_lists` | Per cluster, in ascending id order: `[ids: u32 × count][vectors: T × stored_width × count]`, packed back-to-back. Each record start is 4-byte aligned; the file is zero-padded to a 512-byte multiple. |
| `<prefix>.graphivf_meta` | Fixed header — `magic`, `version`, `metric`, `element_size`, `dim` (the stored row width; `u32` each), `num_points`, `num_clusters` (`u64`), graph params (`degree`, `l_build`, `slack`, `alpha`) — followed by per-cluster counts. List offsets are recomputed from the counts on load. |

Lists are variable length with no per-list disk padding. A read for cluster `c`
fetches the smallest 512-aligned window that fully contains its list and indexes
into it. For plain `f16`/`f32`, `stored_width == logical_dim`; MinMax8 rows also
contain per-row quantization metadata, so their stored width is larger.

```mermaid
flowchart TB
    subgraph FILE["&lt;prefix&gt;.graphivf_lists (ascending cluster id)"]
        direction LR
        R0["c0<br/>ids u32 · vectors T"] --> R1["c1<br/>ids u32 · vectors T"] --> R2["c2<br/>ids u32 · vectors T"] --> PAD["… zero-pad<br/>to 512 multiple"]
    end
    W["read window for c1:<br/>smallest 512-aligned span ⊇ c1,<br/>then index to exact bytes"]
    R1 -.probe.-> W
    classDef pad fill:#eee,stroke-dasharray:3 3;
    classDef win fill:#eef7ff;
    class PAD pad;
    class W win;
```

The online build additionally emits `<prefix>.splits.csv` — per-split telemetry
(see [Telemetry](#telemetry)) — which is **not** part of the loadable index.

---

## Build algorithm

Driver: [`scripts/build/build_online.rs`](scripts/build/build_online.rs).
Core: [`OnlineClusterer`](src/online.rs).

### 1. Seed the initial centroids

The clusterer starts from a small initial centroid set, chosen by a
[`SeedStrategy`](src/online.rs):

- **`Warmup { num_centroids, warmup_points, iters }`** (used by the example): run an exact
  k-means (Forgy init + `iters` Lloyd iterations) over the **first**
  `warmup_points` corpus points. `warmup_points` is clamped to
  `[num_centroids, corpus_len]`. The example exposes the centroid/point counts
  as `--warmup-centroids` and `--warmup-points`, and uses 15 iterations.
- **`Explicit(matrix)`** (library API): use a precomputed centroid matrix as-is.

The initial centroids are inserted into a **mutable Vamana centroid graph**
(`degree` R, `slack`, `l_build`, `alpha`; L2 navigation), pre-allocated to
`centroid_capacity` id slots.

### 2. Insert a point (route)

For each streamed point `pid` (`insert`):

1. **Route** it to its nearest live centroid `c` by searching the centroid graph
   with search-list size `assign_l` (`assign_nearest`). Because splits leave
   **tombstoned** (soft-deleted) slots in the graph, a narrow beam can rarely
   return no live centroid; the router then retries with a widened list
   (`max(8·assign_l, 512)`) and, as a last resort, falls back to a brute-force
   scan over the live centroids.
2. **Append** `pid` to list `c` and record `assignments[pid] = c`.
3. **Maybe split:** if `len(list[c]) > split_threshold`, and the optional
   `max_clusters` cap is not yet reached, and the id budget can fit two more
   centroids, trigger a split of `c`.

Routing an insert never changes the partition unless it triggers a split, so
**splits are the only structural events** in the build.

```mermaid
flowchart TD
    A[next point pid] --> B[route via centroid graph<br/>search-list = assign_l]
    B --> C{live centroid<br/>returned?}
    C -- no --> W[widen L to max 8·assign_l, 512<br/>else brute-force over live centroids]
    W --> D
    C -- yes --> D["append pid to list c<br/>assignments[pid] = c"]
    D --> E{"len(list c) &gt; split_threshold<br/>AND live &lt; max_clusters<br/>AND id budget fits +2?"}
    E -- no --> A
    E -- yes --> F[[split c]]
    F --> A
```

### 3. Split a cluster (split-and-reassign)

`split(c)` turns one overflowing cluster into two and re-optimizes its local
neighborhood:

1. **2-means.** Run a local Lloyd 2-means (`two_means_iters`, seeded from two
   distinct members) over `c`'s members, producing two child centroids
   (L2-normalized if `normalize_centroids`).
2. **Select neighbor clusters.** Search `c`'s own centroid vector in the
   centroid graph and take the `reassign_neighbors` (`s`) nearest **live**
   centroids, excluding `c` (done before deleting `c`). The search uses a
   search-list of `max(reassign_l, s + 1)`.
3. **Update centroid table.** Allocate two fresh ids for the children; **retire**
   `c`'s id (soft-delete — its id is never reused).
4. **Mutate the graph.** Delete `c` and insert the two children into the centroid
   graph.
5. **Reassign the neighborhood.** Form
   - *candidate centroids* = selected neighbors ∪ {child₁, child₂}
   - *candidate points* = `c`'s members ∪ all points of the neighbor clusters

   and reassign every candidate point to its nearest candidate centroid by exact
   squared-L2, rebuilding the affected lists. Every member of the retired cluster
   necessarily moves (its old id is not a candidate); a neighbor point counts as
   "reassigned" only if it lands on a different centroid than before.

Each split is a net **+1** to the live-cluster count (−1 retired, +2 children).

```mermaid
flowchart TD
    S0[cluster c overflows split_threshold] --> S1["2-means over c's members → child1, child2"]
    S1 --> S2["search c's centroid in the graph →<br/>s nearest live centroids (excluding c)"]
    S2 --> S3[alloc 2 new ids; retire c's id as tombstone]
    S3 --> S4[graph: delete c, insert child1 &amp; child2]
    S4 --> S5["candidate centroids = neighbors ∪ {child1, child2}<br/>candidate points = c.members ∪ all neighbor-list points"]
    S5 --> S6[reassign each candidate point to its<br/>nearest candidate centroid by exact L2]
    S6 --> S7[rebuild affected lists · live count +1]
```

Only `c` and the `reassign_neighbors` clusters selected by graph search are
touched; the rest of the partition is unchanged, keeping each split's cost
**local**.

### 4. Id budget & termination

- `centroid_capacity` is the **total** id budget (live + retired). Ids consumed
  over a build is `initial + 2 · splits`; size it to roughly **2×** the expected
  final live-cluster count. Splitting stops when the budget is exhausted.
- `max_clusters = Some(k)` stops splitting at `k` live clusters (fixed
  granularity). `None` lets the count grow driven solely by `split_threshold`;
  the natural equilibrium is `≈ 2 · num_points / split_threshold` live clusters.

The example maps `--max-clusters 0` to `None` and auto-sizes the id budget from
the corpus size, threshold, `--capacity-mult` (default 3), warmup size, and any
cluster cap. Users of the library API provide `centroid_capacity` directly.

### 5. Flush

`flush` serializes the in-memory mapping to the on-disk format above:

1. Densely remap the live centroid ids to a contiguous `0..k`.
2. Write the `k × dim` centroids (`f32`).
3. Write the inverted lists from the **stored** corpus `T` (verbatim) in remapped
   id order, and write the metadata (counts, `dim`, `metric`, element size,
   graph params).

The output loads through
[`GraphIvfIndex::<T>::load`](src/index.rs) exactly like a batch-built index.

---

## Search algorithm

Per-thread [`Searcher::search`](src/index.rs) (one searcher per thread). Given a
query in the stored type `T` and target `k`:

```mermaid
flowchart TD
    Q[query in type T] --> P["preprocess: build T-space scorer<br/>+ decode query to f32 for the graph"]
    P --> K["centroid KNN: nearest nlist centroids<br/>L = max(centroid_search_l, nlist)"]
    K --> IO["plan I/O: smallest 512-aligned window per<br/>non-empty probed list → one reusable buffer"]
    IO --> R[single batched direct read of all probed lists]
    R --> SC[exhaustively score query vs list vectors in T]
    SC --> TK["top-k: select_nth + sort ascending (smaller is better)"]
    TK --> O["return k nearest (id, score)"]
```

1. **Preprocess.** Build a `T`-space scorer once (query and corpus are both `T`,
   so scoring needs no per-candidate decode). Decode the query to `f32` once for
   the centroid graph (a no-op when `T == f32`). Scoring metric: squared-L2 for
   `L2`/`Cosine`, negated inner product for `InnerProduct` — all "smaller is
   better".
2. **Centroid KNN.** Search the centroid graph for the query's nearest `nlist`
   centroids, using search-list size `effective_l = max(centroid_search_l,
   nlist)`. These are the lists to probe.
3. **Plan I/O.** For each non-empty probed cluster, compute the smallest
   512-aligned byte window that fully contains its list, and carve each a
   disjoint slice of **one reusable 512-aligned buffer** (grown only when a query
   needs more space than any prior one — steady state does no allocation).
4. **Batched read.** Issue all probed-list reads as a **single batch** through
   the platform aligned reader (io_uring on Linux, IOCP on Windows, buffered
   elsewhere), with direct/unbuffered I/O bypassing the OS page cache.
5. **Score.** Parse each fetched cluster into `(ids, vectors)` and exhaustively
   score the query against every stored vector directly in `T`.
6. **Top-k.** `select_nth` + sort ascending by score; return the `k` best
   `(id, score)` pairs.

End-to-end recall is therefore capped by two things: whether the centroid graph
returns the *truly* nearest `nlist` centroids (increase `centroid_search_l` /
`nlist`), and whether the target neighbors actually live in those lists (a
function of clustering quality — the reason for the split-and-reassign build).

---

## Parameters and tuning

Online build ([`OnlineParams`](src/params.rs)):

| Parameter | Role |
| --- | --- |
| `split_threshold` | A cluster splits once it holds **more** than this many points (`≥ 2`). The dominant knob on final granularity. |
| `max_clusters` | Optional hard cap on live clusters (`None` = data-driven growth). |
| `centroid_capacity` | Total id budget (live + retired); size to `≈ 2×` expected final clusters. |
| `assign_l` | Centroid-graph search-list size for routing inserts. |
| `reassign_neighbors` | Number of nearest centroid clusters (besides the two children) pooled as reassignment candidates on a split (`≥ 1`). |
| `reassign_l` | Centroid-graph search-list size for the nearest-centroid search that selects `reassign_neighbors` (clamped up to `reassign_neighbors + 1`). |
| `two_means_iters` | Lloyd iterations per split 2-means (internally at least one). |
| `graph` | Centroid-graph build params: `degree` (R), `slack`, `l_build`, `alpha`. |
| `metric` | Search-time routing/scoring metric recorded in the index (clustering is always L2). |
| `normalize_centroids` | L2-normalize warmup and child centroids (unit-sphere corpora). |
| `num_threads`, `seed` | Worker pool for warmup/2-means/graph build; RNG seed. |

Search ([`SearchParams`](src/params.rs)):

| Parameter | Role |
| --- | --- |
| `nlist` | Number of nearest centroids (lists) to probe (`≤ num_clusters`). |
| `centroid_search_l` | Centroid-graph search-list size; effective L is `max(centroid_search_l, nlist)`. |

Practical tuning order:

1. Pick `split_threshold` for the desired list size / cluster count; use
  `max_clusters` only when a hard cap is required.
2. Increase `reassign_neighbors` (and, if needed, `reassign_l`) when build
  quality matters more than split cost.
3. Sweep `nlist` to choose the query-time recall, I/O, and latency trade-off.
4. Raise `assign_l` or `centroid_search_l` only if routing quality is limiting
  build or search recall.

The example's complete flag/default table is in
[`scripts/README.md`](scripts/README.md).

---

## Telemetry

[`BuildTelemetry`](src/online.rs) records routing/split totals and one event per
split: triggering insert, retired cluster and size, neighbor count, points that
changed cluster, resulting live-cluster count, and 2-means/reassignment/total
latencies. The example writes those events to
`<out_prefix>_th<split_threshold>_<format>.splits.csv`. Because splits are the
only structural events, the CSV is a complete timeline of cluster-count growth
and split cost.
