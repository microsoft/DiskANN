# Beta Filter For Disk Search

|                  |                                |
|------------------|--------------------------------|
| **Authors**      | yaohongdeng                    |
| **Contributors** |                                |
| **Created**      | 2026-05-21                     |
| **Updated**      | 2026-05-26                     |

## Summary

Adds **beta-biased filtering on the disk search path** by introducing a `SearchPlan<'a>` enum that replaces the existing `(vector_filter, is_flat_search)` parameter pair on `searcher.search()`. The new enum is hierarchical (`SearchPlan { FlatScan, Graph(GraphMode) }`), encodes today's four search configurations plus the new beta-biased variant as one named value each, makes invalid combinations unrepresentable, and gives future filter algorithms a single extension point that doesn't require further growing the public `search()` signature.

## Motivation

### Background

The in-memory side of DiskANN has a `BetaFilter` strategy ([diskann-providers/src/model/graph/provider/layers/betafilter.rs](../diskann-providers/src/model/graph/provider/layers/betafilter.rs)) that biases beam traversal toward vectors matching a label predicate, by multiplying their distances by a factor `β ∈ (0, 1]`. The disk search path has no equivalent.

The disk API today exposes filtering as a raw closure type alias paired with a separate boolean for flat vs. graph dispatch:

```rust
// diskann-disk/src/build/configuration/filter_parameter.rs
pub type VectorFilter<'a, Data> =
    Box<dyn Fn(&<Data as GraphDataType>::VectorIdType) -> bool + Send + Sync + 'a>;

// searcher.search(..., vector_filter: Option<VectorFilter>, is_flat_search: bool)
```

`MultihopSearch` already exists in [diskann/src/graph/search/multihop_search.rs](../diskann/src/graph/search/multihop_search.rs) but the disk API has no integration point for it either.

### Problem Statement

1. **Beta-biased graph search is unavailable on the disk path.**

2. **The current API shape can't carry beta cleanly.** A raw closure has no place to attach `β`. Threading `beta: Option<f32>` as a fourth parameter creates orthogonality problems: beta is only meaningful in graph search, and `(vector_filter, is_flat_search, beta)` as three independent inputs admits meaningless combinations (`is_flat_search=true` + `beta=Some(_)`) the type system can't catch.

3. **No clean integration point for future filter algorithms.** 

### Goals

1. Add beta-biased disk graph search as a first-class capability.
2. Provide a single extension point for future filter algorithms without changing the public `search()` signature.

## Proposal

### Types

```rust
// diskann-disk/src/search/filter_parameter.rs
// (Moved from build/configuration/ — search-time concept, not build-time.)

pub type Predicate<'a> = Box<dyn Fn(u32) -> bool + Send + Sync + 'a>;

/// Top-level search plan: graph traversal vs. linear scan.
pub enum SearchPlan<'a> {
    /// Brute-force linear scan. `Some(p)` applies `p` inline; `None`
    /// scans every vector (recall baseline).
    FlatScan { filter: Option<Predicate<'a>> },

    /// Graph traversal; `GraphMode` picks the algorithm and any modifier.
    Graph(GraphMode<'a>),
}

/// Graph-search variant. Invalid combinations (e.g. beta without a predicate)
/// are unrepresentable.
pub enum GraphMode<'a> {
    /// Plain greedy beam.
    Unfiltered,

    /// Greedy beam + hard post-filter (applied in `RerankAndFilter`).
    PostFilter(Predicate<'a>),

    /// Beta-biased beam: matching vectors' PQ distances multiplied by
    /// `beta` ∈ (0, 1] in `pq_distances`. Predicate also post-filters.
    BetaFilter { predicate: Predicate<'a>, beta: f32 },

    // Future graph algorithms slot in here as new variants. e.g.:
    //   Multihop { predicate: Predicate<'a> },
}

#[derive(Debug, thiserror::Error)]
pub enum BetaError {
    #[error("beta must be in (0, 1], got {0}")]
    OutOfRange(f32),
}

impl<'a> GraphMode<'a> {
    pub fn post_filter<F>(predicate: F) -> Self
    where F: Fn(u32) -> bool + Send + Sync + 'a {
        Self::PostFilter(Box::new(predicate))
    }

    /// Fallible — returns `BetaError::OutOfRange` if `beta` is outside (0, 1].
    /// Designed for callers that read `beta` from external input (JSON config,
    /// CLI args). Programmer-supplied literals can `.unwrap()` or `?`.
    pub fn beta_filter<F>(predicate: F, beta: f32) -> Result<Self, BetaError>
    where F: Fn(u32) -> bool + Send + Sync + 'a {
        if !(beta > 0.0 && beta <= 1.0) {
            return Err(BetaError::OutOfRange(beta));
        }
        Ok(Self::BetaFilter { predicate: Box::new(predicate), beta })
    }
}

impl<'a> SearchPlan<'a> {
    pub fn flat() -> Self { Self::FlatScan { filter: None } }

    pub fn flat_filtered<F>(predicate: F) -> Self
    where F: Fn(u32) -> bool + Send + Sync + 'a {
        Self::FlatScan { filter: Some(Box::new(predicate)) }
    }

    pub fn graph() -> Self { Self::Graph(GraphMode::Unfiltered) }
    pub fn graph_with(mode: GraphMode<'a>) -> Self { Self::Graph(mode) }
}
```

The lifetime parameter `'a` carries the borrow scope of any captured data in the predicate. For closures that own (move in) their captures, callers don't need to write the lifetime — Rust infers `'static`. Callers that need to borrow from a stack frame for a single `search()` call get a shorter inferred `'a`, matching today's `VectorFilter<'a, Data>` flexibility.

### Supported configurations

| # | Case | `SearchPlan` value |
|---|------|---|
| 1 | Flat scan, no filter | `SearchPlan::flat()` |
| 2 | Flat scan + filter | `SearchPlan::flat_filtered(\|id\| bm.contains(id))` |
| 3 | Graph, no filter | `SearchPlan::graph()` |
| 4 | Graph + post-filter | `SearchPlan::graph_with(GraphMode::post_filter(\|id\| bm.contains(id)))` |
| 5 | Graph + beta + post-filter (**new**) | `SearchPlan::graph_with(GraphMode::beta_filter(\|id\| bm.contains(id), 0.5)?)` |

### Key design decisions

- **Hierarchical, not flat.** `SearchPlan { FlatScan, Graph(GraphMode) }` separates the graph-vs-linear-scan break (different code paths) from the choice of graph algorithm/modifier. Future graph algorithms slot into `GraphMode`, not the top-level enum.
- **Invalid states unrepresentable.** `BetaFilter` carries the predicate inline; the `FlatScan` path has no `GraphMode` field. Beta-without-predicate and beta-on-flat-scan are compile-time errors, not runtime asserts.
- **Project at the boundary.** `GraphMode` exposes constructors only — no accessors. `search_strategy()` projects `(Option<&Predicate>, Option<f32>)` from the plan via one exhaustive match. Adding a `GraphMode` variant updates exactly one match site.

### `search()` signature

The public entry point loses two parameters and gains one:

```rust
// Before
pub fn search(..., vector_filter: Option<VectorFilter<Data>>, is_flat_search: bool) -> ...

// After
pub fn search(..., plan: SearchPlan<'_>) -> ...
```

### `search_strategy` projection — the single `GraphMode` match site

`search_strategy` is the **only** place that introspects `GraphMode`'s variants. One exhaustive match produces `(predicate, beta)`; the strategy carries them as fields. Every downstream consumer reads `strategy.predicate` and `strategy.beta` — no further variant matching anywhere.

```rust
fn search_strategy<'a>(
    &'a self,
    query: &'a [Data::VectorDataType],
    plan: &'a SearchPlan<'a>,
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

The compiler's exhaustiveness check forces every new `GraphMode` variant to add exactly one arm here. `DiskAccessor::new` and `RerankAndFilter::new` consume `strategy.predicate` and `strategy.beta` directly — no further `GraphMode` knowledge downstream.

### `DiskSearchStrategy` struct change

The strategy stops carrying a single closure and starts carrying the two projected fields:

```rust
pub struct DiskSearchStrategy<'a, Data, ProviderFactory>
where
    Data: GraphDataType<VectorIdType = u32>,
    ProviderFactory: VertexProviderFactory<Data>,
{
    io_tracker: IOTracker,
    query: &'a [Data::VectorDataType],

    // === Changed ===
    // Removed: vector_filter: &'a VectorFilter<'a, Data>
    // Added: projected from `plan` once in search_strategy()
    predicate: Option<&'a Predicate<'a>>,
    beta: Option<f32>,

    // === Unchanged ===
    vertex_provider_factory: &'a ProviderFactory,
    scratch_pool: &'a Arc<ObjectPool<DiskSearchScratch<Data, ProviderFactory::VertexProviderType>>>,
}
```

### Caller migration

| Today `(vector_filter, is_flat_search)` | New `plan` |
|---|---|
| `(None, false)` | `SearchPlan::graph()` |
| `(None, true)` | `SearchPlan::flat()` |
| `(Some(p), false)` | `SearchPlan::graph_with(GraphMode::post_filter(p))` |
| `(Some(p), true)` | `SearchPlan::flat_filtered(p)` |
| (not expressible) | `SearchPlan::graph_with(GraphMode::beta_filter(p, β)?)` |

Boundary code in the benchmark and tools constructs `SearchPlan` once and passes it down. The `--is_flat_search` benchmark flag and the corresponding `is_flat_search` field in benchmark JSON schemas (`diskann-benchmark/example/*.json`, `diskann-benchmark/perf_test_inputs/*.json`) are removed in the same change — no external consumers depend on the old schema, so no migration grace period is needed.

### Where beta is/isn't applied

| Location | Beta? | Why |
|----------|-------|-----|
| `pq_distances()` | **Yes**, only on `GraphMode::BetaFilter` | Biases beam toward matching vectors during traversal |
| Full-precision cache | No | Cache holds true distances for honest reranking |
| `RerankAndFilter::post_process()` | No | True distances; predicate is hard-filter only |
| `flat_search` path | No | Type system forbids it — `FlatScan` has no `GraphMode` field |

### Extensibility

| Add… | Where it lives | `match` sites to update |
|------|-----|---|
| New top-level search class (e.g. range search) | New `SearchPlan` variant | `search_internal` dispatch |
| New graph algorithm or beam modifier | New `GraphMode` variant | `search_strategy` projection; `Graph(_)` arm in `search_internal` if invocation differs |

The exhaustiveness check guarantees the compiler refuses to build until every `match` on `GraphMode` handles the new variant — no silent fallback.

### Validation

- `β ∈ (0, 1]` enforced by `GraphMode::beta_filter()`, the only constructor; returns `BetaError::OutOfRange` on invalid input (no panic). This is the validation point for `β` values read from JSON/CLI config — boundary code propagates the error to the user.
- Beta on `FlatScan` and beta without a predicate are compile-time errors by enum shape.
- No allocation on `SearchPlan::graph()` or `SearchPlan::flat()` — both produce variants carrying `None` predicates.

### Files modified

| File | Scope |
|------|-------|
| `diskann-disk/src/search/filter_parameter.rs` (moved from `build/configuration/`) | New types replace `VectorFilter` |
| `diskann-disk/src/search/provider/disk_provider.rs` | `DiskSearchStrategy`, `DiskAccessor`, `pq_distances`, `RerankAndFilter`, `search_internal`, `search_strategy`, `flat_search`, `search()`, test call sites |
| `diskann-benchmark/src/backend/disk_index/search.rs` | Build `SearchPlan` at the boundary |
| `tools/search_disk_index.rs` (or equivalent) | Same closure-to-`SearchPlan` replacement |
| Benchmark input JSON schemas | Remove `is_flat_search` |

### Non-Goals

- **`BetaFilter<DiskSearchStrategy>` composability.** Beta application stays in `DiskAccessor::pq_distances` rather than moving to the `QueryComputer` path. The disk path implements beta independently from the in-memory `BetaFilter` strategy.
- **Adopting `QueryLabelProvider<u32>` at the user-facing API.** Closures are the primary representation; future algorithms that internally require `&dyn QueryLabelProvider<u32>` adapt at the boundary with a thin local wrapper.
- **Beta on `flat_search`.** Brute-force enumeration doesn't benefit from traversal biasing; the type system forbids it.
- **`VectorIdType` genericity.** `Predicate` pins `u32`. The disk path already constrains `Data::VectorIdType = u32` ([disk_provider.rs:548-557](../diskann-disk/src/search/provider/disk_provider.rs)); a future `u64` ID type would touch this API surface — accepted cost.
- **Additional `GraphMode` variants in this change.** The enum is built for extension, but each new variant should land with its own consumer integration, tests, and (for traversal modifiers) recall data.

## Trade-offs

N/A

## Benchmark Results

N/A

## Future Work

- [ ] **Align disk and in-memory beta-filter behavior.** After this change, `GraphMode::BetaFilter` on the disk path applies *both* beta-biased traversal *and* a hard post-filter (non-matching IDs are dropped in `RerankAndFilter`). The in-memory `BetaFilter` strategy ([diskann-providers/.../betafilter.rs](../diskann-providers/src/model/graph/provider/layers/betafilter.rs)) only applies the beta bias — non-matching IDs can still appear in results. A user asking for "beta filter" gets different result sets depending on which side they're on. Future work: pick one semantics (most likely "bias + post-filter," matching the disk side) and align the in-memory strategy, or document the divergence explicitly in both APIs.

## References

N/A
