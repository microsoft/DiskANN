# Beta Filter For Disk Search

|                  |                                |
|------------------|--------------------------------|
| **Authors**      | yaohongdeng                    |
| **Contributors** |                                |
| **Created**      | 2026-05-21                     |
| **Updated**      | 2026-05-21                     |

## Summary

Adds support for **beta-biased filtering on the disk search path** by introducing a `SearchPlan` enum that replaces the existing `(vector_filter, is_flat_search)` parameter pair on `searcher.search()`. The new enum is hierarchical (`SearchPlan { FlatScan, Graph(GraphMode) }`), encodes today's four search configurations plus the new beta-biased variant as one named value each, makes invalid combinations unrepresentable, and gives future filter algorithms a single extension point that doesn't require further growing the public `search()` signature.

## Motivation

### Background

The in-memory side of DiskANN has a `BetaFilter` strategy ([diskann-providers/src/model/graph/provider/layers/betafilter.rs](../diskann-providers/src/model/graph/provider/layers/betafilter.rs)) that biases beam traversal toward vectors matching a label predicate, by multiplying their distances by a factor `β ∈ (0, 1]`. It's documented and tested for the in-memory `FullPrecision` strategy.

The disk search path has no equivalent. The disk API today exposes filtering as a raw closure type alias paired with a separate boolean for flat vs. graph dispatch:

```rust
// diskann-disk/src/build/configuration/filter_parameter.rs
pub type VectorFilter<'a, Data> =
    Box<dyn Fn(&<Data as GraphDataType>::VectorIdType) -> bool + Send + Sync + 'a>;

// searcher.search(..., vector_filter: Option<VectorFilter>, is_flat_search: bool)
```

A raw closure has no field to attach `β` to, and bolting beta on as a third `Option<f32>` parameter would create meaningless combinations (e.g. `is_flat_search=true` + `beta=Some(_)`) that the type system can't reject.

Looking forward, `MultihopSearch` already exists in [diskann/src/graph/search/multihop_search.rs](../diskann/src/graph/search/multihop_search.rs) but the disk API has no way to select it. Each new graph algorithm under the current shape would mean another boolean flag and another runtime validation rule.

### Problem Statement

1. **Beta-biased graph search is unavailable on the disk path.** Queries that would benefit from biasing beam traversal toward labelled vectors must fall back to plain post-filtering on the disk side, with no way to express the beta variant.

2. **The current API shape can't carry beta cleanly.** A raw closure has no place to attach `β`. Threading `beta: Option<f32>` through `search()` → `search_internal()` → `DiskAccessor` as a fourth parameter creates orthogonality problems: beta is only meaningful in graph search, and `(vector_filter, is_flat_search, beta)` as three independent inputs admits 2 of 8 meaningless combinations the type system can't catch.

3. **No clean integration point for future filter algorithms.** `MultihopSearch` exists in `diskann` but the disk API has no extension point — adding it (or any future algorithm) would mean another flag and more cross-field validation.

### Goals

1. Add beta-biased disk graph search as a first-class capability — equivalent in effect to the in-memory `BetaFilter` strategy.
2. Replace the `(vector_filter, is_flat_search)` parameter pair with a single value where every supported configuration is one named variant or constructor.
3. Make invalid combinations (beta on flat scan, beta without a predicate) unrepresentable by construction — caught at compile time, not by runtime assertion.
4. Provide a single extension point so future filter algorithms (`MultihopSearch`, others) can be added without changing the public `search()` signature.
5. Preserve zero allocation and zero per-iteration overhead on the no-filter graph path (today's most common case).

## Proposal

### 1. Motivation

The disk search filter is a raw `Box<dyn Fn(...)>` type alias paired with a separate `is_flat_search: bool` flag:

```rust
// diskann-disk/src/build/configuration/filter_parameter.rs
pub type VectorFilter<'a, Data> =
    Box<dyn Fn(&<Data as GraphDataType>::VectorIdType) -> bool + Send + Sync + 'a>;

// searcher.search(..., vector_filter: Option<VectorFilter>, is_flat_search: bool)
```

Three problems with this shape:

1. **A raw closure can't carry `beta`.** Adding beta-biased graph search requires threading another parameter through `search()` → `search_internal()` → `DiskAccessor`.

2. **Adding beta breaks the existing orthogonality between `is_flat_search` and `vector_filter`.** Today the two flags are genuinely orthogonal: all four `(is_flat_search, vector_filter)` combinations are meaningful, and the benchmark exposes them as independent inputs ([diskann-benchmark/src/inputs/disk.rs:83-85](../diskann-benchmark/src/inputs/disk.rs)). But beta only exists in graph search — it biases beam expansion, and a flat scan has no beam. Adding beta as a third independent parameter creates meaningless combinations (e.g. `is_flat_search=true` + `beta=Some(_)`) that the type system can't reject.

3. **No integration point for future graph algorithms.** `MultihopSearch` already exists in `diskann/src/graph/search/multihop_search.rs`, but the disk API has no way to select it. Each new algorithm would mean another boolean flag.

This design replaces both flags with a single `plan: SearchPlan` enum. Each of the five supported configurations is one explicit variant or constructor; future graph algorithms slot in as new `GraphMode` variants without changing the public `search()` signature; the compiler enforces exhaustiveness at every dispatch site.

### 2. API Change

#### 2.1 The `Predicate` type alias

The disk path identifies filtered IDs through a closure, not a trait. There is no `IdFilter` trait, no `BitmapProvider` wrapper, and no `QueryLabelProvider<u32>` dependency.

```rust
// diskann-disk/src/search/filter_parameter.rs
// (Moved from build/configuration/ — this is a search-time concept, not build-time.)

pub type Predicate = Box<dyn Fn(u32) -> bool + Send + Sync>;
```

**Why a closure, not a trait?**
- **Ergonomic at call sites.** `|id| bm.contains(id)` is shorter and clearer than `Arc::new(BitmapProvider::new(bm))`. Callers using `HashSet<u32>`, sorted `Vec<u32>`, or any custom backing structure write a closure directly.
- **No upstream coupling.** No dependency on `diskann::graph::index::QueryLabelProvider<u32>` — upstream signature changes don't reach the disk path. If a future graph algorithm (e.g. `MultihopSearch`) requires `&dyn QueryLabelProvider<u32>`, it adapts at the boundary with a thin local wrapper.
- **No allocation when absent.** `Option<Predicate>` is `None` for the no-filter cases; no closure object is constructed.

**Why `Fn(u32)` and not `Fn(&u32)`?** `u32` is 4 bytes and `Copy`; passing by value is cheaper than an 8-byte reference, eliminates `*` derefs at every call site, and matches the codebase convention of `QueryLabelProvider::is_match(id: V)` ([diskann/src/graph/index.rs:82](../diskann/src/graph/index.rs)).

**Why `Box`, not `Arc`?** Predicates are owned by `SearchPlan` for the duration of one search call. They aren't shared across queries. `Box` is sufficient and avoids atomic refcount traffic on every clone/drop.

#### 2.2 `SearchPlan` and `GraphMode` enums

**Current** (`filter_parameter.rs`):
```rust
pub type VectorFilter<'a, Data> =
    Box<dyn Fn(&<Data as GraphDataType>::VectorIdType) -> bool + Send + Sync + 'a>;
```

**Proposed**: replace the type alias with two hierarchical enums. `SearchPlan` makes the top-level graph-vs-flat-scan break; `GraphMode` describes the variant on the graph path.

```rust
/// Top-level search plan: graph traversal vs. linear scan.
pub enum SearchPlan {
    /// Brute-force linear scan. `Some(p)` applies `p` inline; `None`
    /// scans every vector (recall baseline).
    FlatScan { filter: Option<Predicate> },

    /// Graph traversal; `GraphMode` picks the algorithm and any modifier.
    Graph(GraphMode),
}

/// Graph-search variant. Invalid combinations (e.g. beta without a predicate)
/// are unrepresentable.
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

**The five cases:**

| # | Case | `SearchPlan` value |
|---|------|---|
| 1 | Flat scan, no filter | `SearchPlan::flat()` |
| 2 | Flat scan + filter | `SearchPlan::flat_filtered(\|id\| bm.contains(id))` |
| 3 | Graph, no filter | `SearchPlan::graph()` |
| 4 | Graph + post-filter | `SearchPlan::graph_with(GraphMode::post_filter(\|id\| bm.contains(id)))` |
| 5 | Graph + beta + post-filter | `SearchPlan::graph_with(GraphMode::beta_filter(\|id\| bm.contains(id), 0.5))` |

**Key design decisions**:

- **Hierarchical, not flat.** `SearchPlan { FlatScan, Graph(GraphMode) }` separates the graph-vs-linear-scan categorical break (different access patterns, different code paths) from the choice of graph algorithm/modifier. A future `Multihop` slots into `GraphMode` as a sibling to `BetaFilter`, not as a top-level variant — they both traverse the graph.
- **Invalid states unrepresentable.** `BetaFilter` carries the predicate inline; the `FlatScan` path has no `GraphMode` field. Beta-without-predicate and beta-on-flat-scan are rejected at compile time, not by runtime assertion.
- **Project at the boundary.** `GraphMode` exposes constructors only — no `predicate()` or `beta()` accessors. The strategy projects `(Option<&Predicate>, Option<f32>)` from the plan via one exhaustive match at construction (§4.2). Blocks semantically meaningless calls (asking `beta()` on `Unfiltered`) and routes every new variant through a single site, compiler-enforced.

#### 2.3 `search()` public signature

`is_flat_search: bool` and `vector_filter: Option<VectorFilter>` are **both removed** and replaced with a single `plan: SearchPlan` parameter.

```rust
// Before
pub fn search(
    query: &[Data::VectorDataType],
    return_list_size: u32,
    search_list_size: u32,
    beam_width: Option<usize>,
    vector_filter: Option<VectorFilter<Data>>,  // Box<dyn Fn(...)>
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

Dispatch is a single match on `plan` (see §4.3). `SearchPlan::graph()` is the default "graph, no filter" case — a unit variant inside a unit variant, no allocation.

#### 2.4 Caller migration

The benchmark today exposes `is_flat_search` and `vector_filters_file` as independent flags ([diskann-benchmark/src/backend/disk_index/search.rs:267-281](../diskann-benchmark/src/backend/disk_index/search.rs)). The four combinations map onto `SearchPlan` as follows:

| Today `(vector_filter, is_flat_search)` | New `plan` |
|---|---|
| `(None, false)` | `SearchPlan::graph()` |
| `(None, true)` | `SearchPlan::flat()` |
| `(Some(p), false)` | `SearchPlan::graph_with(GraphMode::post_filter(p))` |
| `(Some(p), true)` | `SearchPlan::flat_filtered(p)` |
| (not expressible) | `SearchPlan::graph_with(GraphMode::beta_filter(p, β))` |

The last row is **new** capability not available today.

**Example: flat scan + hard filter migration**

```rust
// Before — raw closure + separate boolean
let filter_list: Arc<RoaringBitmap> = /* bitmap of allowed IDs */;
let result = searcher.search(
    query,
    return_list_size,
    search_list_size,
    beam_width,
    Some(Box::new(move |vid: &u32| filter_list.contains(*vid))),
    true,   // is_flat_search
)?;

// After — single plan parameter, no boolean
let filter_list: Arc<RoaringBitmap> = /* bitmap of allowed IDs */;
let result = searcher.search(
    query,
    return_list_size,
    search_list_size,
    beam_width,
    SearchPlan::flat_filtered(move |id| filter_list.contains(id)),
)?;
```

**Example: graph + post-filter migration**

```rust
let filter_list: Arc<RoaringBitmap> = /* bitmap of allowed IDs */;
let result = searcher.search(
    query,
    return_list_size,
    search_list_size,
    beam_width,
    SearchPlan::graph_with(GraphMode::post_filter(move |id| filter_list.contains(id))),
)?;
```

**Example: beta-biased graph search (new capability)**

```rust
let active_ids: Arc<RoaringBitmap> = /* bitmap of non-deleted IDs */;
let result = searcher.search(
    query,
    return_list_size,
    search_list_size,
    beam_width,
    SearchPlan::graph_with(GraphMode::beta_filter(
        move |id| active_ids.contains(id),
        0.5,  // beta: bias beam toward matching vectors
    )),
)?;
```

**Benchmark CLI**: the `--is_flat_search` flag is removed from the disk-index benchmark input schema. The benchmark layer constructs `SearchPlan` from `(is_flat_search, vector_filters_file)` once at the boundary and passes only `SearchPlan` to `searcher.search()`. JSON input schemas that still carry `is_flat_search` should be migrated as a separate task (config-only, no code dependency after the backend update).

### 3. Internal Plumbing

**Changes per type:**

| Type | Change |
|------|--------|
| `filter_parameter.rs` | Replace `VectorFilter` + `default_vector_filter()` with `Predicate`, `GraphMode`, `SearchPlan`. Move file from `build/configuration/` to `search/`. |
| `DiskSearchStrategy` | `vector_filter: &'a dyn Fn` → carries `predicate: Option<&'a Predicate>` + `beta: Option<f32>`, projected from `plan` at construction |
| `DiskAccessor` | Carries `predicate: Option<&'a Predicate>` + `beta: Option<f32>` (both extracted from `plan` at construction) |
| `DiskAccessor::pq_distances` | ~4 lines: apply `beta` when both fields are `Some` |
| `DiskAccessor::new` | Accept `predicate` + `beta` (extracted by `search_accessor()`) |
| `RerankAndFilter` | `filter: &'a dyn Fn` → `filter: Option<&'a Predicate>` |
| `search_internal` | `(vector_filter, is_flat_search)` → `plan: &SearchPlan`; dispatches on top-level variant only |
| `search_strategy` | The **only** site that introspects `GraphMode`; produces `(predicate, beta)` for the strategy fields |
| `flat_search` path | Drop redundant `vector_filter` arg; read predicate from `plan` |
| `search()` | Replace `vector_filter` + `is_flat_search` with `plan: SearchPlan` |
| `expand_beam` | **No change** |
| `distances_unordered` | **No change** |

**No `Arc`, no allocation on the no-filter paths.** `Predicate` is `Box<dyn Fn>`; `DiskAccessor::predicate` is `Option<&'a Predicate>`. For `SearchPlan::graph()` (i.e. `GraphMode::Unfiltered`) and `SearchPlan::flat()`, both `predicate` and `beta` are `None` — no `Box` is constructed.

### 4. Core Implementation Changes

#### 4.1 `search()` — single dispatch parameter

```rust
pub fn search(&self, ..., plan: SearchPlan) -> ANNResult<SearchResult<...>> {
    self.search_internal(query, ..., &plan)
}
```

#### 4.2 `search_strategy` projection — the single `GraphMode` match site

`search_strategy` is the **only** place that introspects `GraphMode`'s variants. One exhaustive match produces `(predicate, beta)`; the strategy carries them as fields. Every downstream consumer reads `strategy.predicate` and `strategy.beta` — no further variant matching anywhere.

```rust
fn search_strategy<'a>(
    &'a self,
    query: &'a [Data::VectorDataType],
    plan: &'a SearchPlan,
) -> DiskSearchStrategy<'a, ...> {
    let (predicate, beta) = match plan {
        SearchPlan::FlatScan { filter } => (filter.as_ref(), None),
        SearchPlan::Graph(GraphMode::Unfiltered) => (None, None),
        SearchPlan::Graph(GraphMode::PostFilter(p)) => (Some(p), None),
        SearchPlan::Graph(GraphMode::BetaFilter { predicate, beta }) =>
            (Some(predicate), Some(*beta)),
    };
    DiskSearchStrategy { predicate, beta, query, ... }
}
```

#### 4.3 `pq_distances()` — apply `beta` when both fields are `Some`

**Current** ([disk_provider.rs:581-586](../diskann-disk/src/search/provider/disk_provider.rs)):
```rust
for (i, id) in ids.iter().enumerate() {
    let distance = self.scratch.pq_scratch.aligned_dist_scratch[i];
    f(distance, *id);
}
```

**Proposed**:
```rust
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

When `beta` or `predicate` is `None`, the `if let` short-circuits — zero overhead on cases 3 (`Unfiltered`) and 4 (`PostFilter`).

#### 4.4 `search_internal()` — dispatch by top-level variant

`search_internal` dispatches by the top-level `SearchPlan` variant only — the `GraphMode` match already happened in `search_strategy`.

```rust
pub(crate) fn search_internal(
    &self, ..., plan: &SearchPlan,
) -> ANNResult<SearchResultStats> {
    let strategy = self.search_strategy(query, plan);
    let stats = match plan {
        SearchPlan::FlatScan { filter } => {
            let accept_all: &(dyn Fn(u32) -> bool + Send + Sync) = &|_| true;
            let accept: &(dyn Fn(u32) -> bool + Send + Sync) = match filter {
                Some(p) => &**p,
                None => accept_all,
            };
            self.runtime.block_on(self.index.flat_search(
                &strategy, &DefaultContext, strategy.query, accept,
                &Knn::new(k, l, beam_width)?, &mut result_output_buffer,
            ))?
        }
        SearchPlan::Graph(_) => {
            let knn = Knn::new(k, l, beam_width)?;
            self.runtime.block_on(self.index.search(
                knn, &strategy, &DefaultContext, strategy.query, &mut result_output_buffer,
            ))?
        }
    };
    ...
}
```

The `Graph(_)` arm doesn't need to look at the `GraphMode` — `pq_distances` already knows about `beta` via `strategy`, and `RerankAndFilter` already knows about `predicate` via `strategy`.

#### 4.5 `RerankAndFilter` — optional predicate

```rust
pub struct RerankAndFilter<'a> {
    filter: Option<&'a Predicate>,
}

// in DiskSearchStrategy::default_post_processor():
fn default_post_processor(&self) -> RerankAndFilter<'_> {
    RerankAndFilter::new(self.predicate)
}

// in RerankAndFilter::post_process():
.filter(|id| self.filter.map_or(true, |f| f(*id)))
```

`map_or(true, ...)` short-circuits with no closure invocation when `filter` is `None`.

### 5. Where Beta Is / Isn't Applied

| Location | Beta applied? | Rationale |
|----------|--------------|-----------|
| `pq_distances()` | **Yes**, only on `GraphMode::BetaFilter` | Biases beam toward matching vectors during graph traversal |
| `ensure_loaded()` (full-precision cache) | No | Cache must hold true distances for honest reranking |
| `RerankAndFilter::post_process()` | No | Uses true distances; predicate is hard-filter only |
| `flat_search` path | No | Type system forbids it — `FlatScan` has no `GraphMode` field |

### 6. Data Flow

```
search(plan: SearchPlan)
  ▼
search_internal(plan: &SearchPlan)
  │
  ├─► step 1: strategy = search_strategy(query, plan)
  │              // exhaustive match on GraphMode projects (predicate, beta) once;
  │              // strategy carries the projection — nothing downstream reads GraphMode
  │
  └─► step 2: dispatch on top-level variant
        │
        ├─[FlatScan { filter }]─► index.flat_search(..., accept-or-accept-all, ...)
        │     (no GraphMode on this path; type system forbids beta)
        │
        └─[Graph(_)]─► index.search(Knn, &strategy, ctx, query, output)
              // index.search drives the traversal and calls back into the
              // strategy on demand:
              │
              ├─► strategy.search_accessor(...) → DiskAccessor
              │     reads strategy.predicate, strategy.beta
              │     │
              │     └─► accessor.pq_distances() — invoked per beam expansion
              │           if let (Some(beta), Some(p)) = (self.beta, self.predicate) {
              │               if p(*id) { distance *= beta }
              │           }
              │
              └─► strategy.default_post_processor() → RerankAndFilter
                    // invoked once on the final candidate set
                    filter = strategy.predicate
```

### 7. Files Modified

| File | Scope of change |
|------|-----------------|
| `diskann-disk/src/search/filter_parameter.rs` (moved from `build/configuration/`) | New types: `Predicate`, `GraphMode`, `SearchPlan` replace `VectorFilter` and `default_vector_filter()` |
| `diskann-disk/src/search/provider/disk_provider.rs` | `DiskSearchStrategy`, `DiskAccessor`, `pq_distances`, `RerankAndFilter`, `search_internal`, `search_strategy`, `flat_search`, `search()`, internal test call sites |
| `diskann-benchmark/src/backend/disk_index/search.rs` | Build `SearchPlan` at the boundary; drop `is_flat_search` parameter |
| `tools/search_disk_index.rs` (or equivalent) | Same closure-to-`SearchPlan` replacement |
| Benchmark input JSON schemas | Remove `is_flat_search`; encode the plan in inputs instead (config-only, no code dependency after the backend update) |

### 8. Backward Compatibility

- **Existing no-filter callers**: replace `vector_filter: None, is_flat_search: false` with `SearchPlan::graph()` — identical behavior, no allocation.
- **Existing flat-scan-no-filter callers** (benchmark recall baseline): replace `vector_filter: None, is_flat_search: true` with `SearchPlan::flat()`.
- **Existing filtered callers (graph + post-filter)**: replace `Some(Box::new(|vid| bm.contains(vid))), is_flat_search: false` with `SearchPlan::graph_with(GraphMode::post_filter(move |id| bm.contains(id)))`.
- **Existing flat-scan-with-filter callers**: replace `Some(Box::new(|vid| bm.contains(vid))), is_flat_search: true` with `SearchPlan::flat_filtered(move |id| bm.contains(id))`.
- **`search_internal` is `pub(crate)`** — no external API breakage beyond `search()`.
- **Zero overhead on the no-filter graph path** — `SearchPlan::graph()` constructs no `Box`; `pq_distances`' `if let` does not match; `RerankAndFilter::filter` is `None` and the `.filter` call short-circuits.

### 9. Extensibility — adding a new graph algorithm

Two extension axes, both compiler-enforced:

| Add… | Where it lives | `match` sites to update |
|------|-----|---|
| **New top-level search class** (e.g. range search) | New `SearchPlan` variant | `search_internal` dispatch |
| **New graph algorithm or beam modifier** | New `GraphMode` variant | `search_strategy` projection match; `Graph(_)` arm in `search_internal` if invocation differs |

### 10. Validation

- `beta` in `GraphMode::BetaFilter { beta }` must be in $(0, 1]$ — `GraphMode::beta_filter()` is the only constructor and panics otherwise. $\beta > 1$ would penalize matches (opposite of intent); $\beta \leq 0$ is nonsensical.
- `DiskAccessor` remains `Send` — `Predicate`'s `Send + Sync` bound propagates.
- No allocation on the `SearchPlan::graph()` and `SearchPlan::flat()` paths — both produce variants carrying `None` predicates.
- The type system forbids beta on the `FlatScan` path (no `GraphMode` field) and beta without a predicate (`BetaFilter` carries it inline).

### 11. Testing

- **Existing tests pass unchanged in spirit**: old `(None, false)` → `SearchPlan::graph()`; `(None, true)` → `SearchPlan::flat()`; `(Some(p), false)` → `SearchPlan::graph_with(GraphMode::post_filter(p))`; `(Some(p), true)` → `SearchPlan::flat_filtered(p)`.
- **Case 5 regression (beta-biased graph)**: synthetic graph where the medoid's BFS frontier provably misses a target cluster unless beta biases it. Verify `BetaFilter` returns IDs that `PostFilter` (same predicate, no beta) misses. Stat-based "matching IDs appear" assertions are flaky — the fixture must make beam ordering deterministic.
- **Recall sweep**: case 4 vs. case 5 across several $\beta$ values on a filter-selective workload. Establishes the recall-vs-effort baseline.
- **Validation**: `GraphMode::beta_filter(p, β)` panics for $\beta \notin (0, 1]$.
- **No-filter zero-cost**: counter assertion that case 3 (`SearchPlan::graph()`) invokes no closure in `pq_distances` or `RerankAndFilter`.

### 12. Non-Goals

- **`BetaFilter<DiskSearchStrategy>` composability.** Beta application stays in `DiskAccessor::pq_distances` rather than moving to the `QueryComputer` path. The disk path implements the beta algorithm independently from the in-memory `BetaFilter` strategy.
- **Adopting `QueryLabelProvider<u32>` at the user-facing API.** Closures are the primary representation; future graph algorithms that internally require `&dyn QueryLabelProvider<u32>` adapt at the boundary with a thin wrapper.
- **Beta on `flat_search`.** Brute-force enumeration doesn't benefit from traversal biasing; the type system forbids it (`FlatScan` has no `GraphMode` field).
- **`VectorIdType` genericity.** `Predicate` pins `u32`. The disk path already constrains `Data::VectorIdType = u32` ([disk_provider.rs:548-557](../diskann-disk/src/search/provider/disk_provider.rs)); a future `u64` ID type would touch this API surface — accepted cost.
- **Additional `GraphMode` variants in this change.** The enum is built for extension (§9), but each new variant should land with its own consumer integration, tests, and (for traversal modifiers) recall data.

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

N/A

## Future Work

- [ ] **Align disk and in-memory beta-filter behavior.** After this change, `GraphMode::BetaFilter` on the disk path applies *both* beta-biased traversal *and* a hard post-filter (matching IDs survive, non-matching IDs are dropped in `RerankAndFilter`). The in-memory `BetaFilter` strategy ([diskann-providers/.../betafilter.rs](../diskann-providers/src/model/graph/provider/layers/betafilter.rs)) only applies the beta bias — it does not post-filter, so non-matching IDs can still appear in results. A user asking for "beta filter" gets different result sets depending on which side they're on. Future work: pick one semantics (most likely "bias + post-filter," matching the disk side) and align the in-memory strategy, or document the divergence explicitly in both APIs.

## References

N/A
