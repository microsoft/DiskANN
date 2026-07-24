<!--
 Copyright (c) Microsoft Corporation.
 Licensed under the MIT license.
-->

# `diskann-graphivf` scripts catalog

Command-line harnesses for building, searching, profiling, and preparing data
for graph-IVF MinMax8 indexes. They are grouped by function into subdirectories;
each file is a standalone `cargo` example (binary name == file stem, kept stable
via explicit `[[example]]` targets in [`../Cargo.toml`](../Cargo.toml)).

## Running

```text
cargo run --release --example <name> -- <args...>
```

The compiled binaries are also emitted to `target/release/examples/<name>[.exe]`
and can be invoked directly (faster for repeated sweeps). All builders/searchers
operate on the DiskANN binary matrix format (`[npoints u32][ndims u32][row-major
data]`). "MinMax8" is the canonical `MinMaxElement<8>` row type (8-bit MinMax
quantized codes + embedded min/max meta); compress full-precision inputs with
[`dataprep/compress_minmax`](dataprep/compress_minmax.rs) first.

### Shared build conventions

Unless a tool documents otherwise, the static builders share: graph degree 32,
slack 1.2, `L_build` 64, alpha 1.2, assignment `L` 32, seed 0. Clustering and
assignment are always squared-L2; `metric = ip` only changes the stored/search
metric. Builders write `<prefix>.graphivf_{lists,meta,centroids.fbin}`.

---

## `build/` — index construction

| Example | Centroid source | What it's for |
| --- | --- | --- |
| [`build_online`](build/build_online.rs) | **Online** streaming clusterer (splits on threshold) — primary path | Incremental build: no target cluster count; splits when a cluster exceeds `--split-threshold`. Labeled `--flag value` args; on-disk element type via `--format minmax8\|f16\|f32` (default `minmax8`). Emits `.splits.csv` per-split telemetry. |
| [`build_static`](build/build_static.rs) | `--seed sampled` \| `random` \| `precomputed` \| `forgy-f32` | Unified static build (fixed-`k` baseline). Seeding strategy + all build params (`--clusters`, `--iters`, `--assign`, `--normalize`, `--metric`, …) are flags. Covers sampled/random/precomputed k-means, unit-sphere (`--normalize`), and memory-frugal f32-sourced Forgy init. On-disk element type via `--format minmax8\|f16\|f32` (default `minmax8`). |

Both builders take two positional arguments — `<corpus.bin>` (rows already in the
`--format` element type) and `<out_prefix>` — followed by labeled `--flag value`
options. They write `<prefix>.graphivf_{lists,meta,centroids.fbin}`, where
`<prefix>` embeds the key knobs (see each tool below).

### `build_online` arguments (primary)

```text
cargo run --release --example build_online -- \
    <corpus.bin> <out_prefix> --split-threshold <n> [options]
```

Output prefix: `<out_prefix>_th<split_threshold>_<format>`. Also writes
`<prefix>.splits.csv` (per-split telemetry).

| Flag | Default | Description |
| --- | --- | --- |
| `--split-threshold <n>` | *(required)* | Split a cluster once it holds **more** than `n` points. Primary granularity knob; smaller ⇒ more, smaller clusters (equilibrium ≈ `2·N / n` live clusters). |
| `--warmup-centroids <n>` | 100 | Number of seed centroids from a light k-means over the corpus prefix. |
| `--warmup-points <n>` | 10000 | Leading corpus points used for that warmup k-means. |
| `--threads <n>` | 16 | Build worker threads. |
| `--assign-l <n>` | 64 | Centroid-graph search-list size when routing each insert to its cluster. |
| `--two-means-iters <n>` | 12 | Lloyd iterations run for each split's 2-means. |
| `--metric <l2\|ip>` | l2 | Metric recorded in the index (search/graph navigation). Clustering itself is always squared-L2. |
| `--normalize` | off | Presence flag: L2-normalize child centroids after each split (unit-sphere / cosine corpora). |
| `--capacity-mult <n>` | 3 | Centroid id-budget headroom, sized `≈ capacity_mult · 2N / split_threshold`. |
| `--reassign-neighbors <n>` | 8 | `s`: number of nearest neighbor clusters pooled for local reassignment on a split. |
| `--reassign-l <n>` | `max(s, assign_l)` | Centroid-graph search-list size used to select those `s` neighbors. |
| `--max-clusters <n>` | 0 | `0` = uncapped, data-driven growth; otherwise a hard cap on live clusters. |
| `--format <fmt>` | minmax8 | On-disk element type of the corpus and inverted lists: `minmax8` \| `f16` \| `f32`. |

### `build_static` arguments

```text
cargo run --release --example build_static -- \
    <corpus.bin> <out_prefix> --seed <strategy> --clusters <k> [options]
```

Output prefix: `<out_prefix>_<clusters>_<format>`.

Centroid seeding strategy (`--seed`, required):

| Strategy | Centroids come from | Requires |
| --- | --- | --- |
| `sampled` | Forgy k-means over a random corpus sample | `--sample-size` |
| `random` | exactly `--clusters` corpus rows drawn uniformly (`--iters 0` ⇒ pure random partition) | — |
| `precomputed` | an existing centroid `fbin`, reused verbatim (forces `--iters 0`) | `--centroids` |
| `forgy-f32` | Forgy init streamed from a separate f32 corpus (memory-frugal) | `--init-corpus` |

| Flag | Default | Description |
| --- | --- | --- |
| `--seed <strategy>` | *(required)* | Centroid-seeding strategy (table above). |
| `--clusters <k>` | *(required)* | Number of centroids `k`. |
| `--format <fmt>` | minmax8 | On-disk element type of the corpus and inverted lists: `minmax8` \| `f16` \| `f32`. |
| `--iters <n>` | 0 | Lloyd (k-means) refinement iterations; `0` uses the initial centers unrefined. |
| `--sample-size <n>` | corpus size | Rows sampled for k-means under `--seed sampled`. |
| `--centroids <path>` | — | Centroid `fbin` for `--seed precomputed`. |
| `--init-corpus <path>` | — | Separate f32 corpus for `--seed forgy-f32` seeding. |
| `--threads <n>` | 16 | Build worker threads. |
| `--assign <mode>` | auto | Point→centroid assignment: `auto` (exact below 16384 clusters, else graph) \| `exact` \| `graph`. |
| `--rebuild-every <n>` | 1 | Graph-assigner: centroid-graph rebuild cadence (assignment passes between rebuilds). |
| `--rerank <n>` | 8 | Graph-assigner: exact re-rank depth applied to graph candidates. |
| `--metric <l2\|ip>` | l2 | Metric recorded in the index (search/graph navigation). Clustering/assignment are always squared-L2. |
| `--normalize` | off | Presence flag: L2-normalize centroids after each Lloyd iteration (unit-sphere / cosine corpora). |
| `--rng-seed <n>` | 0 | RNG seed for sampling and random seeding. |

## `search/` — search and combined harnesses

| Example | What it does |
| --- | --- |
| [`sweep`](search/sweep.rs) | Search-only `nlist` sweep over an existing index; reports recall@50 & @1000, mean/p95/p99 latency, and bytes read/query. Works with any stored format (`minmax8`/`f16`/`f32`) — auto-detected from the index metadata; queries must be in that same format. |

## `analysis/` — profiling, ablation, reliability

| Example | What it isolates |
| --- | --- |
| [`profile_layercake`](analysis/profile_layercake.rs) | Per-query latency attribution ("layer cake") at a chosen thread count; direct I/O, no cold/warm distinction. |
| [`centroid_graph_ablation`](analysis/centroid_graph_ablation.rs) | Centroid-graph recall: graph top-`nlist` vs exact nearest centroids (caps end-to-end recall independent of list scoring). |

## `dataprep/` — corpus & query preparation

| Example | What it produces |
| --- | --- |
| [`compress_minmax`](dataprep/compress_minmax.rs) | 8-bit MinMax-quantized `.bin` from an `fp16`/`f32` `.bin` (run on both corpus and queries before any MinMax8 build/search). |

---

## Typical end-to-end flow

```text
# 1. Compress corpus + queries to MinMax8
cargo run --release --example compress_minmax -- corpus_f32.bin  corpus_minmax8.bin  f32
cargo run --release --example compress_minmax -- queries_f32.bin queries_minmax8.bin f32

# 2a. Online (streaming) build — primary path
cargo run --release --example build_online -- \
    corpus_minmax8.bin out_prefix --split-threshold <n>

# 2b. …or static build (sampled k-means, fixed-k baseline)
cargo run --release --example build_static -- \
    corpus_minmax8.bin out_prefix --seed sampled --clusters <k> \
    --sample-size <n> --iters <kmeans_iters> --threads <num_threads>

# 3. Sweep nlist for recall / latency
cargo run --release --example sweep -- \
    out_prefix_minmax8 "64,128,256,512" <num_threads> queries_minmax8.bin groundtruth.bin
```

Per-tool argument lists and defaults are documented in each file's module
(`//!`) header and its `USAGE` string.
