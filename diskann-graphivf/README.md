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

Two build paths share this on-disk format and the same search path:

- **Online** (primary) — streaming build: insert points one at a time; split
  overflowing clusters and locally reassign. The cluster count emerges from the
  data. See [`ONLINE.md`](ONLINE.md) for the algorithm.
- **Static** — batch build: sample/select centroids, k-means, assign, write lists.
  Useful as a fixed-`k` baseline.

The library API lives in [`src/`](src/); all experiments run through the CLI
harnesses in [`scripts/`](scripts/) (full catalog: [`scripts/README.md`](scripts/README.md)).

---

## 1. Data preparation

All tools read/write the DiskANN binary matrix format:
`[npoints: u32][ndims: u32][row-major data]`.

The builders (`build_static`, `build_online`) accept the corpus in three
on-disk element types via `--format`: **`minmax8`** (`MinMaxElement<8>`: 8-bit
quantized codes + embedded per-vector min/max; default), **`f16`**, or **`f32`**.
Clustering always runs in decoded `f32`; the chosen type is only what the inverted
lists store on disk. For `f16`/`f32`, feed the corpus `.bin` directly. For
`minmax8`, compress full-precision inputs **once**, for both the corpus and the
queries, before any build or search:

```text
cargo run --release --example compress_minmax -- corpus_f32.bin  corpus_minmax8.bin  f32
cargo run --release --example compress_minmax -- queries_f32.bin queries_minmax8.bin f32
```

| Input | Format | Notes |
| --- | --- | --- |
| corpus | `.bin` in the `--format` type | one `minmax8`/`f16`/`f32` row per corpus vector |
| queries | matching `.bin` | same width/type as the corpus; scored directly in `T` |
| groundtruth | `.bin` of `u32` | `num_queries × gt_dim`, true neighbor ids per query |

Cosine: normalize vectors **before** compression (stored rows are written verbatim).

---

## 2. Shared parameters

**Centroid graph** (`GraphParams`, same defaults for both build paths): `degree=32`,
`slack=1.2`, `l_build=64`, `alpha=1.2`.

**Metric** — clustering and assignment are **always squared-L2**. The metric flag only
sets how the *loaded* index scores at search time:

| `metric` | Search scoring / graph navigation |
| --- | --- |
| `l2` (default) | squared-L2 |
| `ip` | inner product (MIPS) |

**Search** (`SearchParams`, used by `sweep`):

| Param | Meaning |
| --- | --- |
| `nlist` | number of nearest clusters (lists) to probe |
| `centroid_search_l` | centroid-graph search-list size; auto-raised to `nlist` (sweep default `1024`) |

`sweep` reads the stored element type from the index metadata, so the same
command works for `minmax8`, `f16`, and `f32` indexes.

---

## 3. Online index (primary)

No target cluster count: points stream in and clusters **split** when they exceed
`split_threshold`; the final count emerges from the data. Builder:
[`build_online`](scripts/build/build_online.rs). Positional `<corpus.bin>
<out_prefix>`, then labeled `--flag value` options:

### Build parameters (`OnlineParams`)

| Flag | Default | Meaning |
| --- | --- | --- |
| `--split-threshold` | *(required)* | split a cluster once it holds **more** than this many points (dominant granularity knob) |
| `--warmup-centroids` | 100 | initial centroids from a light k-means over a corpus prefix |
| `--warmup-points` | 10000 | leading corpus points used for the warmup |
| `--threads` | 16 | worker threads |
| `--assign-l` | 64 | centroid-graph search-list size for routing inserts |
| `--two-means-iters` | 12 | Lloyd iterations per split 2-means |
| `--metric` | l2 | `l2` \| `ip` (recorded scoring metric) |
| `--normalize` | off | presence flag; L2-normalize child centroids after a split (unit-sphere corpora) |
| `--capacity-mult` | 3 | centroid id-budget headroom (`≈ capacity_mult · 2N / split_threshold`) |
| `--reassign-neighbors` | 8 | `s` nearest neighbor clusters pooled for reassignment on a split |
| `--reassign-l` | `max(s, assign_l)` | search-list size selecting those `s` neighbors |
| `--max-clusters` | 0 | `0` = uncapped growth; else hard cap on live clusters |
| `--format` | minmax8 | on-disk element type: `minmax8` \| `f16` \| `f32` |

Equilibrium live clusters ≈ `2 · num_points / split_threshold`.

Writes `<out_prefix>_th<split_threshold>_<format>.graphivf_{lists,meta,centroids.fbin}`
plus `<...>.splits.csv` (per-split telemetry: cluster size, neighbors pooled, points
reassigned, live count, and 2-means / reassign / total latency).

Example — ~16384 clusters (`th=106`), `s=5` reassignment neighbors:

```text
cargo run --release --example build_online -- \
    corpus_minmax8.bin out_prefix --split-threshold 106 \
    --reassign-neighbors 5 --max-clusters 16384
```

Sweep the result with the `sweep` tool (it auto-detects the stored element
type from the index metadata; supply queries in that same format):

```text
cargo run --release --example sweep -- \
    out_prefix_th106_minmax8 "164,410,656,901,1147,1638,2458" 1 \
    queries_minmax8.bin groundtruth.bin
```

Prints one row per `nlist`: **recall@50**, **recall@1000**, mean/p95/p99 latency (µs),
bytes read/query, IOs/query, request bytes, QPS, and a per-stage latency breakdown
(preprocess, centroid search, plan I/O, disk read, score, top-k).

---

## 4. Static index

Centroids are chosen once and fixed; the target cluster count `num_clusters` is explicit.
Use it as a fixed-`k` baseline against the online build.

### Build parameters (`BuildParams`)

| Param | Meaning |
| --- | --- |
| `num_clusters` | number of centroids `k` |
| `sample_size` | corpus rows sampled for k-means (sampled seeding only) |
| `kmeans_iters` | Lloyd iterations (`0` = use initial centers unrefined) |
| `num_threads` | build worker threads |
| `assign_l` | centroid-graph search-list size for point→centroid assignment (const `32`) |
| `rebuild_every`, `rerank` | graph-accelerated assignment: rebuild cadence / exact re-rank depth |
| `normalize` | L2-normalize centroids after each Lloyd iter (unit-sphere corpora) |
| `metric` | `l2` \| `ip` |

### Builder

All single-stage builds run through one example,
[`build_static`](scripts/build/build_static.rs); `--seed` picks the centroid
source and `--format` picks the on-disk element type. It writes
`<prefix>_<num_clusters>_<format>.graphivf_{lists,meta,centroids.fbin}`.

```text
cargo run --release --example build_static -- \
    <corpus_minmax8> <out_prefix> --seed <strategy> --clusters <k> [options]
```

| `--seed` | Centroid source | Extra flag |
| --- | --- | --- |
| `sampled` | Forgy k-means over a random sample | `--sample-size <n>` |
| `random` | exactly `<k>` random corpus rows (`--iters 0` ⇒ pure random partition) | — |
| `precomputed` | an existing centroid `fbin`, reused verbatim (`--iters` forced 0) | `--centroids <path>` |
| `forgy-f32` | Forgy init drawn from a separate f32 corpus (memory-frugal) | `--init-corpus <f32.bin>` |

Common options: `--format minmax8|f16|f32` (default `minmax8`), `--iters`,
`--threads`, `--assign auto|exact|graph` (`auto` = exact below 16384 clusters, else
graph), `--rebuild-every`, `--rerank`, `--metric l2|ip`, `--normalize`, `--rng-seed`.

Example — sampled build, 16384 clusters:

```text
cargo run --release --example build_static -- \
    corpus_minmax8.bin out_prefix --seed sampled --clusters 16384 \
    --sample-size 200000 --iters 10 --threads 16
```

Sweep the result with the **same** `sweep` tool:

```text
cargo run --release --example sweep -- \
    out_prefix_16384_minmax8 "164,410,656,901,1147,1638,2458" 1 \
    queries_minmax8.bin groundtruth.bin
```

---

## 5. Output artifacts

| File | Contents |
| --- | --- |
| `<prefix>.graphivf_centroids.fbin` | centroid matrix (`f32`), reloaded to rebuild the graph |
| `<prefix>.graphivf_lists` | contiguous per-cluster inverted lists (`T` = `--format` type) |
| `<prefix>.graphivf_meta` | layout: counts, offsets, dim, metric, element size, graph params |
| `<prefix>.splits.csv` | online only — per-split telemetry timeline |

---

## See also

- [`scripts/README.md`](scripts/README.md) — full example catalog (analysis, profiling, dataprep).
- [`ONLINE.md`](ONLINE.md) — online split-and-reassign algorithm and search internals.
- Per-tool argument lists and defaults live in each example's `//!` header and `USAGE` string.
