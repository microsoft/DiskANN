# Flat Search

|                  |                                |
|------------------|--------------------------------|
| **Authors**      | Aditya Krishnan, Alex Razumov, Dongliang Wu                |
| **Created**      | 2026-04-24                     |
| **Updated**      | 2026-05-20                     |

## 1. Motivation

### 1.1 Background

DiskANN today exposes a single abstraction family centered on the 
[`crate::provider::Accessor`] trait. Accessors are random access by design since the graph greedy search algorithm needs to decide which ids to fetch and the accessor materializes the corresponding elements (vectors, quantized vectors and neighbor lists) on demand. This is the right contract for graph search, where neighborhood expansion is inherently random-access against the [`crate::provider::DataProvider`].

A growing class of consumers diverge from our current pattern of use by accesssing their index **sequentially**. Some consumers build their index in an "append-only" fashion and require that they walk the index in a sequential, fixed order, relying on iteration position to enforce versioning / deduplication invariants.

### 1.2 Problem Statement

The problem-statement here is simple: provide first-class support for sequential, one-pass scans over a data backend without stuffing the algorithm or the backend through the `Accessor` trait surface.

### 1.3 Goals

1. Define a fused iterate-and-score primitive — `flat::DistancesUnordered<C>` — that
   mirrors the role `Accessor` plays for graph search but exposes a sequential
   scan-and-score operation instead of random access.
2. Provide flat-search algorithm implementations built on the new primitives, so consumers can use this against their own providers / backends. 
3. (Near-future) Expose support for diferent distance computers and post-processing like re-ranking _out-of-the-box_ without having to reimplement these for the flat search path.  

## 2. Proposal

The only shared surface between graph and flat search is the `DataProvider` (for id-mapping / context).

The module exposes three layers:

| Layer | Trait | Role |
|-------|-------|------|
| Backend | `DistancesUnordered<C>` | Scan-and-score primitive |
| Factory | `SearchStrategy<P, T>` | Per-query visitor + computer construction |
| Algorithm | `FlatIndex::knn_search` | Brute-force top-k |

### 2.1 `DistancesUnordered<C>` — the core scanning trait

The single required trait for flat search. It is generic over a **computer type** `C`
rather than a query type — the algorithm supplies a pre-built computer and the visitor
drives the scan.

```rust
pub trait DistancesUnordered<C>: Send + Sync
where
    C: for<'a> PreprocessedDistanceFunction<Self::ElementRef<'a>, f32>,
{
    type ElementRef<'a>;
    type Id;
    type Error: ToRanked + Debug + Send + Sync + 'static;

    fn distances_unordered<F>(
        &mut self,
        computer: &C,
        f: F,
    ) -> impl SendFuture<Result<(), Self::Error>>
    where
        F: Send + FnMut(Self::Id, f32);
}
```

Key differences from the graph-side `Accessor` path:

- No random access — the visitor drives the entire scan internally.
- `ElementRef<'a>` and `Id` live on `DistancesUnordered` itself, decoupling the
  scan-and-score primitive from `HasId` and from any provider-specific id type. A
  visitor is free to yield ids derived from but not equal to its provider's
  `InternalId`. We expect this constraint to go away once we're able to clean up the `VectorId` trait 
  and its restrictive bounds - i.e. expects id to be scalar-like. 

### 2.2 `SearchStrategy<P, T>` — per-query factory

The strategy owns both visitor construction and query-computer construction:

```rust
pub trait SearchStrategy<P, T>: Send + Sync
where
    P: DataProvider,
{
    type ElementRef<'a>;
    type Id;
    type QueryComputer: for<'a> PreprocessedDistanceFunction<Self::ElementRef<'a>, f32>
        + Send + Sync + 'static;
    type QueryComputerError: StandardError;

    type Visitor<'a>: for<'b> DistancesUnordered<
            Self::QueryComputer,
            ElementRef<'b> = Self::ElementRef<'b>,
            Id = Self::Id,
        >
    where Self: 'a, P: 'a;

    type Error: StandardError;

    fn create_visitor<'a>(
        &'a self,
        provider: &'a P,
        context: &'a P::Context,
    ) -> Result<Self::Visitor<'a>, Self::Error>;

    fn build_query_computer(
        &self,
        query: T,
    ) -> Result<Self::QueryComputer, Self::QueryComputerError>;
}
```

`build_query_computer` lives on the **strategy**, not the visitor. This keeps the
visitor free of any distance-computation trait bounds — it only needs to implement
`DistancesUnordered<C>` for the strategy's computer type.

### 2.3 `FlatIndex::knn_search`

`FlatIndex<P>` is a thin `'static` wrapper around a `DataProvider`. The `knn_search`
method is the brute-force top-k algorithm:

```rust
impl<P: DataProvider> FlatIndex<P> {
    pub fn knn_search<S, T, OB>(
        &self,
        k: NonZeroUsize,
        strategy: &S,
        context: &P::Context,
        query: T,
        output: &mut OB,
    ) -> impl SendFuture<ANNResult<SearchStats>>
    where
        S: SearchStrategy<P, T>,
        S::Id: NeighborPriorityQueueIdType,
        T: Send + Sync,
        OB: SearchOutputBuffer<S::Id> + Send + ?Sized;
}
```

Algorithm:

1. `strategy.create_visitor(&provider, context)` — acquire the scanning visitor.
2. `strategy.build_query_computer(query)` — preprocess the query into a computer.
3. `visitor.distances_unordered(&computer, |id, dist| queue.insert(...))` — full scan.
4. Drain the priority queue into `output` in best-first order.

**No post-processing parameter (yet).** Currently `knn_search` writes
`(S::Id, f32)` directly into the `SearchOutputBuffer`. Once the graph-search
trait refactor in [PR #1076](https://github.com/microsoft/DiskANN/pull/1076)
lands, `knn_search` will accept an optional `SearchPostProcess` parameter
(the same trait graph search uses), enabling id remapping, re-ranking, and
other transformations as a composable layer.

#### Call-chain diagram

```text
        Graph                                              Flat
        ─────                                              ────

  DiskANNIndex::search                              FlatIndex::knn_search
        │                                                    │
        ▼                                                    ▼
  graph::glue::SearchStrategy                      flat::SearchStrategy
   ::search_accessor                                ::create_visitor
        │                                           ::build_query_computer
        ▼                                                    │
  Accessor + BuildQueryComputer<T>                           ▼
   → QueryComputer                                 DistancesUnordered<C>
        │                                           ::distances_unordered(&computer, f)
        ▼                                                    │
  ExpandBeam::expand_beam                                    │
  (greedy beam, random access)                               │
        │                                                    │
        ▼                                                    ▼
  NeighborPriorityQueue                            NeighborPriorityQueue
        │                                                    │
        ▼                                                    ▼
  SearchPostProcess                                SearchPostProcess (planned, PR #1076)
   → SearchOutputBuffer                             → SearchOutputBuffer
```

## Trade-offs

### No built-in post-processing (temporary)

`knn_search` currently writes `(InternalId, f32)` directly. Once the graph-search
trait refactor in [PR #1076](https://github.com/microsoft/DiskANN/pull/1076) lands
and stabilizes a shared `SearchPostProcess` trait, `knn_search` will gain an optional
post-processor parameter matching the graph-search signature. Until then, callers that
need id remapping or re-ranking compose it externally.

### Reusing `DataProvider`

The design requires implementations to provide `InternalId` / `ExternalId` conversions.
This is arguably too restrictive for some flat-index consumers, but avoids introducing a
second provider trait.

### Expand `ElementRef` and `QueryComputer` to support batched distance computation?

The design for `DistancesUnordered` assumes the computer acts on single vectors. An alternative is to allow the computer to work 
over batches, enabling (potentially) better cache utilization. Backends that need this can implement `DistancesUnordered<C>`
directly with an optimized bulk loop. Some refactoring for the bounds on `DistancesUnordered` is needed here.  

### Intra-query parallelism

`DistancesUnordered` requires `&mut self`, precluding internal parallelism within a single
scan. A parallel variant would need a different trait shape (e.g. splitting the scan across
shards). This is left for future work.

## Future Work
- **Post-processing support** — once [PR #1076](https://github.com/microsoft/DiskANN/pull/1076) lands, add a `SearchPostProcess` parameter to `knn_search` so flat search can share the same id-remapping / re-ranking infrastructure as graph search.
- Support for other flat-search algorithms like filtered, range, and diverse flat algorithms as additional methods on `FlatIndex`.
- Index build — this is just one part of the picture; more work needs to be done around how this fits in with any traits / interface we need for index build.

