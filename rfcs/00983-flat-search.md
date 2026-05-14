# Flat Search

|                  |                                |
|------------------|--------------------------------|
| **Authors**      | Aditya Krishnan, Alex Razumov, Dongliang Wu                |
| **Created**      | 2026-04-24                     |
| **Updated**      | 2026-05-14                     |

## 1. Motivation

### 1.1 Background

DiskANN today exposes a single abstraction family centered on the 
[`crate::provider::Accessor`] trait. Accessors are random access by design since the graph greedy search algorithm needs to decide which ids to fetch and the accessor materializes the corresponding elements (vectors, quantized vectors and neighbor lists) on demand. This is the right contract for graph search, where neighborhood expansion is inherently random-access against the [`crate::provider::DataProvider`].

A growing class of consumers diverge from our current pattern of use by accesssing their index **sequentially**. Some consumers build their index in an "append-only" fashion and require that they walk the index in a sequential, fixed order, relying on iteration position to enforce versioning / deduplication invariants.

### 1.2 Problem Statement

The problem-statement here is simple: provide first-class support for sequential, one-pass scans over a data backend without
stuffing the algorithm or the backend through the `Accessor` trait surface.

### 1.3 Goals

1. Define a fused iterate-and-score primitive — `flat::DistancesUnordered<T>` — that
   mirrors the role `Accessor` plays for graph search but exposes a sequential
   scan-and-score operation instead of random access.
2. Provide flat-search algorithm implementations built on the new primitives, so consumers can use this against their own providers / backends. 
3. Expose support for diferent distance computers and post-processing like re-ranking _out-of-the-box_ without having to reimplement these for the flat search path.  

## 2. Proposal

The flat-search infrastructure is built on a small sequence of traits. The only trait a
backend *must* implement is `flat::DistancesUnordered<T>`. A `flat::SearchStrategy`
then instantiates per-query visitors that implement it.

An opt-in `FlatIterator` trait plus the `Iterated<I>` adapter exist for
convenience for backends that naturally expose element-at-a-time iteration.

### 2.1 Refactor `Accessor` and `BuildQueryComputer `
We start by a small refactor we introduce to the traits in `diskann::providers` and `diskann::graph::glue` that 
will enable us to cleanly separate the query preprocessing, result post-processing from the search pattern so that 
both graph and flat search can share common components as much as possible: 

1. **Extract `HasElementRef` out of `Accessor`.** The `ElementRef<'a>` GAT moves to
   its own zero-method trait so that streaming visitors (which are not `Accessor`s)
   can still expose an element type. `Accessor` is now `Accessor: HasId +
   HasElementRef + Send + Sync`. `HasElementRef` is simply: 

   ```rust
   pub trait HasElementRef { type ElementRef<'a> } 
   ```

2. **Decouple `BuildQueryComputer<T>` from `Accessor`.** Previously a
   sub-trait of `Accessor`, `BuildQueryComputer<T>` is lifted to depend only on
   `HasElementRef`. Secondly, it now contains only a constructor `build_query_computer` 
   as an associated method and nothing else. This is the
   change that lets `BuildQueryComputer<T>` and `graph::glue::SearchPostProcess` be
   used unchanged by both the flat index and the graph.

3. **Split distance scoring into a new `DistancesUnordered<T>` trait family.**
   Previously, the unordered iterate-and-score loop was a default method tucked
   inside `Accessor` (and shadowed by overrides on a few providers). It is now its
   own subtrait of `BuildQueryComputer<T>`, with two flavors that share a name and a
   default-body shape but differ in their access super-trait:

   - **`provider::DistancesUnordered<T>: Accessor + BuildQueryComputer<T>`** — drives
     the scan via the random-access `Accessor` machinery. Used by graph search.
   - **`flat::DistancesUnordered<T>: HasId + BuildQueryComputer<T>`** — a
     self-contained fused iterate-and-score trait used by flat search. Backends
     implement the entire scan-and-score loop in a single method. More on it below.

### 2.2 Core traits for flat search

The single required trait for flat search is `flat::DistancesUnordered<T>`. It fuses
iteration and scoring into one method: implementations drive an entire scan over their
underlying data, scoring each element with the supplied query computer and invoking a
callback with `(id, distance)` pairs. Implementations choose iteration order,
prefetching, and any bulk reads; algorithms see only `(Id, f32)` pairs.

```rust
pub trait DistancesUnordered<T>: HasId + BuildQueryComputer<T> + Send + Sync {
    type Error: ToRanked + Debug + Send + Sync + 'static;

    fn distances_unordered<F>(
        &mut self,
        computer: &<Self as BuildQueryComputer<T>>::QueryComputer,
        f: F,
    ) -> impl SendFuture<Result<(), Self::Error>>
    where
        F: Send + FnMut(<Self as HasId>::Id, f32);
}
```

`DistancesUnordered<T>` is scoped to a single query. We introduce a strategy that is the per-call
constructor that hands the algorithm a freshly-bound visitor. It is meant to be stateless,
cheap to construct, and lives only for the duration of one search. 

```rust
pub trait SearchStrategy<P, T>: Send + Sync
where
    P: DataProvider,
{
    /// The per-query visitor type produced by [`Self::create_visitor`]. Borrows from
    /// `self` and the provider. The visitor implements both the streaming
    /// [`DistancesUnordered<T>`] primitive and the query preprocessor
    /// [`BuildQueryComputer<T>`].
    type Visitor<'a>: DistancesUnordered<T>
    where
        Self: 'a,
        P: 'a;

    type Error: StandardError;

    /// Construct a fresh visitor over `provider` for the given request `context`.
    fn create_visitor<'a>(
        &'a self,
        provider: &'a P,
        context: &'a P::Context,
    ) -> Result<Self::Visitor<'a>, Self::Error>;
}
```
This shape mirrors the random-access `graph::glue::SearchStrategy` and lets `FlatIndex::knn_search` accept the same
`graph::glue::SearchPostProcess` that graph search uses (see below).

### 2.3 `FlatIndex` — the top-level handle

`FlatIndex` is a thin `'static` wrapper around a `DataProvider`. The same
`DataProvider` trait used by graph search is reused — flat and graph share one
provider surface and the same `Context` / id-mapping / error machinery.

```rust
pub struct FlatIndex<P: DataProvider> {
    provider: P,
}

impl<P: DataProvider> FlatIndex<P> {
    pub fn new(provider: P) -> Self;
    pub fn provider(&self) -> &P;

    pub fn knn_search<S, T, O, OB, PP>(
        &self,
        k: NonZeroUsize,
        strategy: &S,
        processor: &PP,
        context: &P::Context,
        query: T,
        output: &mut OB,
    ) -> impl SendFuture<ANNResult<SearchStats>>
    where
        S: flat::SearchStrategy<P, T>,
        T: Copy + Send + Sync,
        O: Send,
        OB: SearchOutputBuffer<O> + Send + ?Sized,
        PP: for<'a> graph::glue::SearchPostProcess<S::Visitor<'a>, T, O> + Send + Sync,
}
```

**Note:** The `PP` bound uses the same `graph::glue::SearchPostProcess` trait as graph search;
there is no flat-specific post-process trait. Reuse is enabled by the trait splits
described above (the visitor implements `BuildQueryComputer<T> + HasId`, which is all
`SearchPostProcess` requires).

The `knn_search` method is the canonical brute-force search algorithm:

1. Construct the per-query visitor via `strategy.create_visitor`.
2. Build the query computer from the visitor via `BuildQueryComputer::build_query_computer`.
3. Drive the scan via `visitor.distances_unordered(&computer, ...)`, inserting each
   `(id, distance)` pair into a `NeighborPriorityQueue<Id>` of capacity `k`.
4. Hand the survivors (in distance order) to `processor.post_process`.
5. Return search stats.

Other algorithms (filtered, range, diverse) can be added later as additional methods on
`FlatIndex`.

#### Search call chain (AI Generated)

The diagram below traces the trait dispatch sequence inside one `search` call for
each of graph and flat search. The centre lane shows the shared traits that both
columns dip into.

```text
        Graph                       Shared                          Flat
        ─────                       ──────                          ────

  DiskANNIndex::search                                          FlatIndex::knn_search
        │                                                                │
        ▼                                                                ▼
  graph::glue::SearchStrategy                                  flat::SearchStrategy
   ::search_accessor                                            ::create_visitor
        │                                                                │
        ▼                                                                ▼
  ExpandBeam<T> visitor                                       DistancesUnordered<T> visitor
  (Accessor + BuildQueryComputer<T>)                          (HasId + BuildQueryComputer<T>)
        │                                                                │
        │                  BuildQueryComputer<T>                         │
        ├─────────────────►::build_query_computer ◄──────────────────────┤
        │                  (visitor → QueryComputer)                     │
        │                                                                │
        ▼                                                                ▼
  ExpandBeam::expand_beam                                  DistancesUnordered
  (greedy beam loop:                                       ::distances_unordered
   for each frontier id,                                   (one pass over every
    get_neighbors,                                          element; computer scores
    distances_unordered)                                    each one)
        │                                                                │
        ▼                                                                ▼
  NeighborPriorityQueue                                       NeighborPriorityQueue
        │                                                                │
        │              graph::glue::SearchPostProcess                    │
        └─────────────►::post_process ◄──────────────────────────────────┘
                                  │
                                  ▼
                          SearchOutputBuffer
```

### 2.4 `FlatIterator` and `Iterated` — convenience for element-at-a-time backends

For backends that naturally expose element-at-a-time iteration, `FlatIterator` is a
lending async iterator:

```rust
pub trait FlatIterator: HasId + HasElementRef + Send + Sync {
    type Element<'a>: for<'b> Reborrow<'b, Target = <Self as HasElementRef>::ElementRef<'b>>
        + Send + Sync
        where Self: 'a;
    type Error: StandardError;

    fn next(
        &mut self,
    ) -> impl SendFuture<Result<Option<(Self::Id, Self::Element<'_>)>, Self::Error>>;
}
```

`Iterated<I>` wraps any `FlatIterator` and implements `DistancesUnordered<T>` (when
the inner type also implements `BuildQueryComputer<T>`) by looping over `next()`,
reborrowing each element, and scoring it with the supplied query computer.

## Trade-offs

### Reusing `DataProvider`

This design leans into using the `DataProvider` trait which requires implementations to implement `InternalId` and `ExternalId` conversions (via the context). Arguably, this requirement is too restrictive for some consumers of a flat-index. Reasons for sticking with `DataProvider`: 

- Every concrete provider already implements `DataProvider`, so a separate trait adds
  an abstraction that existing consumers will have to implement if they want to opt-in to the flat-index path.
- Sharing `DataProvider` means the `Context`, id-mapping (`to_internal_id` /
  `to_external_id`), and error machinery are identical across graph and flat search,
  reducing the learning surface for new contributors.

### Expand `Element` to support batched distance computation?

The current optional iterator `FlatIterator` yields one element per `next()` call, and the query computer scores
elements one at a time via `PreprocessedDistanceFunction::evaluate_similarity`. This could leave some optimization and performance on the table; especially with the upcoming effort around batched distance kernels. Of course, a consumer can choose to implement their own optimized implementation of `distances_unordered` that uses batching.

An alternative is to make `next()` yield a *batch* instead of a single vector representation like `Element<'_>`. Some work will need to be done to define the right interaction between the batch type, the element type in the batch, the interaction with `QueryComputer`'s types and way IDs and distances are collected in the queue.

### Intra-query parallelism 

The current design of `DistancesUnordered` does not allow an implementation to exploit parallelism within a query; since the trait requires a `&mut self`. Especially for a flat index, some implementations might want to parallelize within the scan for a query. Arguably we will need a more complex extension of this architecture to support this. 

## Future Work
- Support for other flat-search algorithms like - filtered, range and diverse flat algorithms as additional methods on `FlatIndex`.
- Index build -- this is just one part of the picture; more work needs to be done around how this fits in with any traits / interface we need for index build.

