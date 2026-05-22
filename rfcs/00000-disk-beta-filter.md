# Search Plan API for Disk Search (with Beta-Biased Filtering)

|                  |                                |
|------------------|--------------------------------|
| **Authors**      | yaohongdeng                    |
| **Contributors** |                                |
| **Created**      | 2026-05-21                     |
| **Updated**      | 2026-05-21                     |

## Summary

Replace the disk search API's `(vector_filter: Option<Box<dyn Fn>>, is_flat_search: bool)` parameter pair with a single `plan: SearchPlan` enum. The new enum is hierarchical (`SearchPlan { FlatScan, Graph(GraphMode) }`), makes invalid combinations unrepresentable, and introduces a new capability — *beta-biased graph search* — as one of its variants. The change also closes the design's only extension point for future graph algorithms (e.g. `MultihopSearch`) without further growing the public signature of `searcher.search()`.

## Motivation

### Background

The disk search API today exposes filtering as a raw closure type alias paired with a separate boolean for flat vs. graph dispatch:

```rust
// diskann-disk/src/build/configuration/filter_parameter.rs
pub type VectorFilter<'a, Data> =
    Box<dyn Fn(&<Data as GraphDataType>::VectorIdType) -> bool + Send + Sync + 'a>;

// searcher.search(..., vector_filter: Option<VectorFilter>, is_flat_search: bool)
```

The benchmark layer exposes both as independent inputs ([diskann-benchmark/src/inputs/disk.rs:83-85](../diskann-benchmark/src/inputs/disk.rs)). Today this is orthogonal — all four `(vector_filter, is_flat_search)` combinations correspond to valid disk-search configurations:

| `vector_filter` | `is_flat_search` | Meaning |
|---|---|---|
| `None` | `false` | Graph search, no filter |
| `Some(p)` | `false` | Graph search + post-filter |
| `None` | `true` | Flat scan baseline (brute-force recall floor) |
| `Some(p)` | `true` | Flat scan + hard filter |

Meanwhile, the in-memory side has gained a `BetaFilter` strategy ([diskann-providers/src/model/graph/provider/layers/betafilter.rs](../diskann-providers/src/model/graph/provider/layers/betafilter.rs)) that biases beam traversal toward labelled vectors by multiplying their distances by a factor `β ∈ (0, 1]`. The disk path has no equivalent today: a query that wants beta-biased filtering on a disk index has no way to express it.

### Problem Statement

Three concrete problems with the current shape:

1. **A raw closure can't carry `beta`.** Adding beta-biased graph search requires threading another parameter (`beta: Option<f32>`) through `search()` → `search_internal()` → `DiskAccessor`. The closure has no field to attach metadata to.

2. **Beta breaks the existing orthogonality.** Beta is only meaningful in graph search (it biases beam expansion; flat scan has no beam). With three independent inputs `(vector_filter, is_flat_search, beta)`, 2 of the 2³ = 8 combinations are meaningless (`is_flat_search=true` with `beta=Some(_)`; `vector_filter=None` with `beta=Some(_)`). A flat `bool` + `Option<f32>` cannot reject them at compile time — validation has to run at runtime, and every call site has to know which combinations are valid.

3. **No integration point for future graph algorithms.** `MultihopSearch` already exists in [diskann/src/graph/search/multihop_search.rs](../diskann/src/graph/search/multihop_search.rs) but the disk API has no way to select it. Each new algorithm would mean another boolean flag and another runtime validation rule.

### Goals

1. Express all five valid configurations — flat-no-filter, flat-with-filter, graph-no-filter, graph-with-post-filter, graph-with-beta — as one named value each, with invalid combinations unrepresentable.
2. Add beta-biased disk graph search as a first-class capability without further growing `searcher.search()`'s parameter list.
3. Provide a single extension point for future graph algorithms that doesn't require changing the public `search()` signature.
4. Preserve the zero-allocation, zero-overhead property for the most common case (graph search, no filter).
5. Keep the disk filter API insulated from upstream changes to `diskann::graph::index::QueryLabelProvider<u32>`.

## Proposal

### Core types

Two new types replace the existing `VectorFilter` alias:

```rust
// diskann-disk/src/search/filter_parameter.rs
// (Moved from build/configuration/ — this is a search-time concept, not build-time.)

pub type Predicate = Box<dyn Fn(u32) -> bool + Send + Sync>;

/// Top-level search plan: graph traversal vs. linear scan.
pub enum SearchPlan {
    /// Brute-force linear scan. `Some(p)` applies `p` inline; `None`
    /// scans every vector (recall baseline).
    FlatScan { filter: Option<Predicate> },

    /// Graph traversal; `GraphMode` picks the algorithm and any modifier.
    Graph(GraphMode),
}

/// Graph-search variant. Invalid combinations (e.g. beta without a predicate)
/// are unrepresentable by construction.
pub enum GraphMode {
    /// Plain greedy beam.
    Unfiltered,

    /// Greedy beam + hard post-filter (applied in `RerankAndFilter`).
    /// Traversal identical to `Unfiltered`.
    PostFilter(Predicate),

    /// Beta-biased beam: matching vectors' PQ distances multiplied by
    /// `beta` ∈ (0, 1] in `pq_distances`. Predicate also post-filters.
    BetaFilter { predicate: Predicate, beta: f32 },

    // Future graph algorithms slot in here as new variants. e.g.:
    //   Multihop { predicate: Predicate },
}

impl GraphMode {
    pub fn post_filter<F>(predicate: F) -> Self
    where F: Fn(u32) -> bool + Send + Sync + 'static {
        Self::PostFilter(Box::new(predicate))
    }

    pub fn beta_filter<F>(predicate: F, beta: f32) -> Self
    where F: Fn(u32) -> bool + Send + Sync + 'static {
        assert!(beta > 0.0 && beta <= 1.0, "beta must be in (0, 1]");
        Self::BetaFilter { predicate: Box::new(predicate), beta }
    }
}

impl SearchPlan {
    pub fn flat() -> Self { Self::FlatScan { filter: None } }

    pub fn flat_filtered<F>(predicate: F) -> Self
    where F: Fn(u32) -> bool + Send + Sync + 'static {
        Self::FlatScan { filter: Some(Box::new(predicate)) }
    }

    pub fn graph() -> Self { Self::Graph(GraphMode::Unfiltered) }
    pub fn graph_with(mode: GraphMode) -> Self { Self::Graph(mode) }
}
```

The five cases:

| # | Case | Call |
|---|------|------|
| 1 | Flat scan, no filter | `SearchPlan::flat()` |
| 2 | Flat scan + filter | `SearchPlan::flat_filtered(\|id\| bm.contains(id))` |
| 3 | Graph, no filter | `SearchPlan::graph()` |
| 4 | Graph + post-filter | `SearchPlan::graph_with(GraphMode::post_filter(\|id\| bm.contains(id)))` |
| 5 | Graph + beta + post-filter (**new**) | `SearchPlan::graph_with(GraphMode::beta_filter(\|id\| bm.contains(id), 0.5))` |

### Public `search()` signature change

```rust
// Before
pub fn search(
    query: &[Data::VectorDataType],
    return_list_size: u32,
    search_list_size: u32,
    beam_width: Option<usize>,
    vector_filter: Option<VectorFilter<Data>>,
    is_flat_search: bool,
) -> ANNResult<SearchResult<Data::AssociatedDataType>>

// After
pub fn search(
    query: &[Data::VectorDataType],
    return_list_size: u32,
    search_list_size: u32,
    beam_width: Option<usize>,
    plan: SearchPlan,
) -> ANNResult<SearchResult<Data::AssociatedDataType>>
```

### Internal plumbing

| Type | Change |
|------|--------|
| `filter_parameter.rs` | Replace `VectorFilter` + `default_vector_filter()` with `Predicate`, `GraphMode`, `SearchPlan`. Move file from `build/configuration/` to `search/`. |
| `DiskSearchStrategy` | Old: `vector_filter: &'a dyn Fn`. New: `predicate: Option<&'a Predicate>` + `beta: Option<f32>`, projected from `plan` at construction. |
| `DiskAccessor` | Carries `predicate: Option<&'a Predicate>` + `beta: Option<f32>`. |
| `DiskAccessor::pq_distances` | ~4 added lines: apply `beta` when both fields are `Some`. |
| `RerankAndFilter` | Old: `filter: &'a dyn Fn`. New: `filter: Option<&'a Predicate>`. |
| `search_internal` | `(vector_filter, is_flat_search)` → `plan: &SearchPlan`; dispatches on top-level variant only. |
| `search_strategy` | The **only** site that introspects `GraphMode` (one exhaustive match producing `(predicate, beta)`). |
| `flat_search` path | Drop redundant `vector_filter` arg; read predicate from `plan`. |
| `expand_beam`, `distances_unordered` | **No change.** |

The "project at the boundary" decision matters: every match on `GraphMode` happens in `search_strategy`. Downstream code (`DiskAccessor`, `RerankAndFilter`, `flat_search`) sees only the projected `(Option<&Predicate>, Option<f32>)` pair. Adding a new `GraphMode` variant later means touching one match arm, not three or four.

### `pq_distances` — where beta applies

```rust
// Before (disk_provider.rs:581-586)
for (i, id) in ids.iter().enumerate() {
    let distance = self.scratch.pq_scratch.aligned_dist_scratch[i];
    f(distance, *id);
}

// After
for (i, id) in ids.iter().enumerate() {
    let mut distance = self.scratch.pq_scratch.aligned_dist_scratch[i];
    if let (Some(beta), Some(predicate)) = (self.beta, self.predicate) {
        if predicate(*id) {
            distance *= beta;
        }
    }
    f(distance, *id);
}
```

When either field is `None`, the `if let` short-circuits — zero overhead on the common cases.

### Caller migration

| Today `(vector_filter, is_flat_search)` | New `plan` |
|---|---|
| `(None, false)` | `SearchPlan::graph()` |
| `(None, true)` | `SearchPlan::flat()` |
| `(Some(p), false)` | `SearchPlan::graph_with(GraphMode::post_filter(p))` |
| `(Some(p), true)` | `SearchPlan::flat_filtered(p)` |
| (not expressible) | `SearchPlan::graph_with(GraphMode::beta_filter(p, β))` |

The benchmark layer constructs `SearchPlan` once at the boundary; the `--is_flat_search` flag is removed from the disk-index input schema. JSON input schemas that still carry `is_flat_search` are migrated as a separate config-only step.

### Where beta does and doesn't apply

| Location | Beta applied? | Rationale |
|----------|--------------|-----------|
| `pq_distances` | **Yes**, only on `GraphMode::BetaFilter` | Biases beam toward matching vectors during graph traversal |
| `ensure_loaded` (full-precision cache) | No | Cache must hold true distances for honest reranking |
| `RerankAndFilter::post_process` | No | Uses true distances; predicate is hard-filter only |
| `flat_search` path | No | Type system forbids it — `FlatScan` has no `GraphMode` field |

### Extensibility — adding a new graph algorithm

Two extension axes, both compiler-enforced:

| Add… | Where it lives | `match` sites to update |
|------|-----|---|
| **New top-level search class** (e.g. range search) | New `SearchPlan` variant | `search_internal` dispatch |
| **New graph algorithm or beam modifier** | New `GraphMode` variant | `search_strategy` projection match (Rust exhaustiveness check forces the update) |

Worked example — adding `Multihop` (which already exists in `diskann::graph::search::multihop_search` and takes `&dyn QueryLabelProvider<u32>`):

```rust
// In GraphMode:
Multihop { predicate: Predicate },
```

Three localized edits:

1. Add the `GraphMode::Multihop { predicate }` arm to the projection match in `search_strategy`: `(Some(predicate), None)`. `RerankAndFilter` post-filters automatically; no beta.
2. Add a `ClosureAsLabelProvider` wrapper local to the disk crate, since `MultihopSearch::new` expects `&dyn QueryLabelProvider<u32>`:

   ```rust
   struct ClosureAsLabelProvider<'a>(&'a Predicate);

   impl std::fmt::Debug for ClosureAsLabelProvider<'_> {
       fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
           f.debug_struct("ClosureAsLabelProvider").finish_non_exhaustive()
       }
   }

   impl QueryLabelProvider<u32> for ClosureAsLabelProvider<'_> {
       fn is_match(&self, id: u32) -> bool { (self.0)(id) }
   }
   ```

3. Add a `match` arm in `search_internal` (or refine the `Graph(_)` arm) that calls `MultihopSearch::new(Knn, &adapter)` instead of plain `Knn`.

The compiler refuses to build until every `match` on `GraphMode` handles `Multihop`.

## Trade-offs

### Closure (`Box<dyn Fn(u32) -> bool>`) vs. `Arc<dyn QueryLabelProvider<u32>>` trait

**Chosen: closure.** Ergonomic at call sites (`|id| bm.contains(id)` vs. `Arc::new(BitmapProvider::new(bm))`). No coupling to a trait we don't own — upstream changes to `QueryLabelProvider` don't reach the disk path. No allocation on the no-filter path. The downside is that future algorithms requiring `&dyn QueryLabelProvider<u32>` need a thin adapter (~15 lines, scoped to the disk crate) — see the `Multihop` example above.

**Alternative considered: `Arc<dyn QueryLabelProvider<u32> + Send + Sync>`.** Would make `BetaFilter<DiskSearchStrategy>` *type-valid* immediately — same `Arc` shape the in-memory `BetaFilter::new()` already takes. Rejected because the underlying composability also requires moving beta application from `DiskAccessor::pq_distances` into the `QueryComputer` path, which is a separate (large) refactor. Type-valid composability is not behavioural composability, and the in-memory `QueryLabelProvider` will soon change anyway. Disk closures stay decoupled.

### Hierarchical `SearchPlan { FlatScan, Graph(GraphMode) }` vs. flat `SearchPlan { FlatScanNoFilter, FlatScanFiltered, GraphNoFilter, GraphPostFilter, GraphBeta }`

**Chosen: hierarchical.** Separates the graph-vs-linear-scan categorical break (different code paths) from the choice of graph algorithm/modifier. New graph algorithms (e.g. `Multihop`) slot into `GraphMode`, not into the top-level enum — they all traverse the graph and share dispatch.

**Alternative considered: a flat enum.** One variant per case (5 today, more as algorithms are added). Slightly less nesting at call sites but no clear extension axis: a new graph algorithm has to pick a name pattern (`GraphMultihop`? `GraphMultihopFiltered`?). The hierarchy makes the "this is a graph algorithm" attribute explicit.

### `Box` vs. `Arc` for `Predicate`

**Chosen: `Box`.** Predicates are owned by `SearchPlan` for the duration of one search call; not shared across queries. `Box` is sufficient and avoids atomic refcount traffic.

### `Fn(u32)` vs. `Fn(&u32)`

**Chosen: `Fn(u32)`.** `u32` is 4 bytes and `Copy`; by-value is cheaper than an 8-byte reference, eliminates `*` derefs at every call site, and matches the codebase convention of `QueryLabelProvider::is_match(id: V)` ([diskann/src/graph/index.rs:82](../diskann/src/graph/index.rs)).

### Project at strategy construction vs. defer projection

**Chosen: project at strategy construction.** `search_strategy` is the single match site for `GraphMode`; the strategy carries `(predicate, beta)` as fields, and every downstream consumer reads those fields directly. Adding a `GraphMode` variant requires updating one `match`, not three.

**Alternative considered: keep `&SearchPlan` in the strategy, project later.** Downstream consumers (`DiskAccessor`, `RerankAndFilter`, `flat_search` path) each re-match. More flexible but each new `GraphMode` variant touches more sites.

### `assert!` in `GraphMode::beta_filter` vs. `Result`

**Chosen: `assert!`.** Caller-side bug, not a runtime condition. Returning `Result<GraphMode, Error>` would force every caller to handle an error that, if it ever fires, indicates a programming mistake (passing literal `0.0` or `2.0` as `beta`). The constructor panicking on the calling thread is acceptable.

## Benchmark Results

Not yet collected. The primary risk is whether beta-biased graph search delivers a measurable recall improvement over post-filter on filter-selective workloads, and at what `β` the cost crossover sits. Planned experiments:

1. **Recall vs. effort sweep** — case 4 (`PostFilter`) vs. case 5 (`BetaFilter`) across `β ∈ {0.3, 0.5, 0.7, 0.9}` on workloads with selectivity ∈ {1%, 10%, 50%}. Establishes when beta is worth its traversal cost.
2. **No-filter zero-cost regression** — counter assertion that case 3 (`SearchPlan::graph()`) invokes no closure in `pq_distances` or `RerankAndFilter`.
3. **`is_flat_search=true, vector_filter=None` baseline parity** — confirm `SearchPlan::flat()` matches today's `(None, true)` performance and recall exactly.

Results will be added to this RFC before merge.

## Future Work

- [ ] **`BetaFilter<DiskSearchStrategy>` composability.** Move beta application from `DiskAccessor::pq_distances` into the `QueryComputer` path, then add an `IdFilter → QueryLabelProvider<u32>` adapter so the disk strategy can be wrapped by the in-memory `BetaFilter` strategy. Out of scope here.
- [ ] **`Multihop` integration on the disk path.** Add as a new `GraphMode` variant per the worked example in §Extensibility.
- [ ] **`u64` `VectorIdType` support.** `Predicate` pins `u32` today, matching the disk path's existing `Data::VectorIdType = u32` constraint at [disk_provider.rs:548-557](../diskann-disk/src/search/provider/disk_provider.rs). A future `u64` ID type would require generalizing `Predicate` and the projection.
- [ ] **Migration of benchmark JSON input schemas.** Remove `is_flat_search` from the disk-index schema and from the example/perf-test JSON files. Config-only, no code dependency after the backend update.

## References

1. [docs/disk-beta-filter-with-query-label-provider.md](../docs/disk-beta-filter-with-query-label-provider.md) — full design doc this RFC summarizes.
2. [docs/disk-beta-filter.md](../docs/disk-beta-filter.md) — earlier closure-based beta filter design (superseded by this RFC).
3. [diskann-providers/src/model/graph/provider/layers/betafilter.rs](../diskann-providers/src/model/graph/provider/layers/betafilter.rs) — in-memory `BetaFilter` strategy; the disk path implements the algorithm independently.
4. [diskann/src/graph/search/multihop_search.rs](../diskann/src/graph/search/multihop_search.rs) — `MultihopSearch`; the worked extensibility example.
5. [diskann-disk/src/search/provider/disk_provider.rs](../diskann-disk/src/search/provider/disk_provider.rs) — current `pq_distances`, `RerankAndFilter`, `DiskSearchStrategy`, `DiskAccessor`.
6. [diskann-benchmark/src/backend/disk_index/search.rs](../diskann-benchmark/src/backend/disk_index/search.rs) — current benchmark call site (lines 267-281).
