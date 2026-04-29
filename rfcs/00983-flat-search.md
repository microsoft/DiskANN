# Flat Search

|                  |                                |
|------------------|--------------------------------|
| **Authors**      | Aditya Krishnan, Alex Razumov, Dongliang Wu                |
| **Created**      | 2026-04-24                     |
| **Updated**      | 2026-04-27                     |

## Motivation

### Background

DiskANN today exposes a single abstraction family centered on the 
[`crate::provider::Accessor`] trait. Accessors are random access by design since the graph greedy search algorithm needs to decide which ids to fetch and the accessor materializes the corresponding elements (vectors, quantized vectors and neighbor lists) on demand. This is the right contract for graph search, where neighborhood expansion is inherently random-access against the [`crate::provider::DataProvider`].

A growing class of consumers diverge from our current pattern of use by accesssing their index **sequentially**. Some consumers build their index in an "append-only" fashion and require that they walk the index in a sequential, fixed order, relying on iteration position to enforce versioning / deduplication invariants.

### Problem Statement

The problem-statement here is simple: provide first-class support for sequential, one-pass scans over a data backend without
stuffing the algorithm or the backend through the `Accessor` trait surface.

### Goals

1. Define a streaming access primitive — `OnElementsUnordered` — that mirrors the role
   `Accessor` plays for graph search but exposes a callback-driven scan instead of
   random access.
2. Provide flat-search algorithm implementations (with `knn_search` as default and filtered and diverse variants to opt-into) built on the new
   primitives, so consumers can use this against their own providers / backends. 
3. Expose support for features and implementations native to the repo like quantized distance computers out-of-the-box.

## Proposal

The flat-search infrastructure is built on a small sequence of traits. The only required traits for the algorithm is `OnElementsUnordered` and its subtrait `DistancesUnordered`. A strategy - `FlatSearchStrategy` - instantiates these implementations for specific providers. An opt-in iterator trait `FlatIterator` and default implementations of the core traits - `DefaultIteratedOperator` - exist for convenience for backends that naturally expose element-at-a-time iteration.

### `OnElementsUnordered` — the core scan

```rust
pub trait OnElementsUnordered: HasId + Send + Sync {
    type ElementRef<'a>;
    type Error: StandardError;

    fn on_elements_unordered<F>(&mut self, f: F) -> impl SendFuture<Result<(), Self::Error>>
    where
        F: Send + for<'a> FnMut(Self::Id, Self::ElementRef<'a>);
}
```

A single required method: drive the entire scan via a callback. Async to match
[`crate::provider::Accessor`]. Implementations choose iteration order, prefetching, and
any SIMD-friendly bulk reads if they want; algorithms see only `(Id, ElementRef)` pairs.

### `DistancesUnordered` — the distance subtrait

```rust
pub trait DistancesUnordered: OnElementsUnordered {
    fn distances_unordered<C, F>(
        &mut self, computer: &C, mut f: F,
    ) -> impl SendFuture<Result<(), Self::Error>>
    where
        C: for<'a> PreprocessedDistanceFunction<Self::ElementRef<'a>, f32> + Send + Sync,
        F: Send + FnMut(Self::Id, f32),
    {
        // default delegates to on_elements_unordered + evaluate_similarity
    }
}
```

A subtrait that fuses scanning with scoring. The default implementation loops
`on_elements_unordered` and calls `computer.evaluate_similarity` on each element.

The query computer is a generic parameter rather than an associated type, so the same
callback type can be driven by different computers. The `FlatSearchStrategy` is the
source of truth for which computer is used in any given search.

### `FlatIterator` and `DefaultIteratedOperator` — convenience for element-at-a-time backends

For backends that naturally expose element-at-a-time iteration, `FlatIterator` is a
lending async iterator:

```rust
pub trait FlatIterator: HasId + Send + Sync {
    type ElementRef<'a>;
    // lifetime gymnastics to make lifetime of `Element<'_>` to play nice with HRTB
    type Element<'a>: for<'b> Reborrow<'b, Target = Self::ElementRef<'b>> + Send + Sync
        where Self: 'a; 
    type Error: StandardError;

    fn next(
        &mut self,
    ) -> impl SendFuture<Result<Option<(Self::Id, Self::Element<'_>)>, Self::Error>>;
}
```

`DefaultIteratedOperator<I>` wraps any `FlatIterator` and implements `OnElementsUnordered`
(and `DistancesUnordered` by inheritance) by looping over `next()` and reborrowing each
element. 


### The glue: `FlatSearchStrategy`

While `OnElementsUnordered` is the primary handle the algorithm uses to walk the index,
it is scoped to each query. We introduce a constructor — `FlatSearchStrategy` — similar
to `SearchStrategy` for `Accessor`, to instantiate the per-query callback object.
A strategy is per-call configuration that is stateless, cheap to construct and scoped to one
search. It produces both a per-query callback and a query computer.

```rust
pub trait FlatSearchStrategy<P, T>: Send + Sync
where
    P: DataProvider,
    T: ?Sized,
{
    /// The per-query callback type produced by [`Self::create_callback`]. Borrows from
    /// `self` and the provider.
    type Callback<'a>: DistancesUnordered
    where
        Self: 'a,

    /// The query computer produced by [`Self::build_query_computer`].
    type QueryComputer: for<'a, 'b> PreprocessedDistanceFunction<
            <Self::Callback<'a> as OnElementsUnordered>::ElementRef<'b>,
            f32,
        > + Send
        + Sync
        + 'static;

    /// The error type 
    type Error: StandardError;

    /// Construct a fresh callback over `provider` for the given request `context`.
    fn create_callback<'a>(
        &'a self,
        provider: &'a P,
        context: &'a P::Context,
    ) -> Result<Self::Callback<'a>, Self::Error>;

    /// Pre-process a query into a [`Self::QueryComputer`] usable for distance computation
    /// against any callback produced by [`Self::create_callback`].
    fn build_query_computer(&self, query: &T) -> Result<Self::QueryComputer, Self::Error>;
}
```

The `ElementRef<'b>` that the `QueryComputer` acts on is tied to the
`OnElementsUnordered::ElementRef` of the callback produced by `create_callback`.

### `FlatIndex`

`FlatIndex` is a thin `'static` wrapper around a `DataProvider`. The same `DataProvider`
trait used by graph search is reused here - flat and graph subsystems share a single
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
        query: &T,
        output: &mut OB,
    ) -> impl SendFuture<ANNResult<SearchStats>>
    where
        S: FlatSearchStrategy<P, T>,
        T: ?Sized + Sync,
        O: Send,
        OB: SearchOutputBuffer<O> + Send + ?Sized,
        PP: for<'a> FlatPostProcess<S::Callback<'a>, T, O> + Send + Sync,
}
```

The `knn_search` method is the canonical brute-force search algorithm:

1. Construct the per-query callback via `strategy.create_callback`.
2. Build the query computer via `strategy.build_query_computer`.
3. Drive the scan via `callback.distances_unordered(&computer, ...)`, inserting each
   `(id, distance)` pair into a `NeighborPriorityQueue<Id>` of capacity `k`.
4. Hand the survivors (in distance order) to `processor.post_process`.
5. Return search stats.

Other algorithms (filtered, range, diverse) can be added later as additional methods on
`FlatIndex`.

## Trade-offs

### Reusing `DataProvider`

This design leans into using the `DataProvider` trait which requires implementations to implement `InternalId` and `ExternalId` conversions (via the context). Arguably, this requirement is too restrictive for some consumers of a flat-index. Reasons for sticking with `DataProvider`: 

- Every concrete provider already implements `DataProvider`, so a separate trait adds
  an abstraction that existing consumers will have to implement if they want to opt-in to the flat-index path.
- Sharing `DataProvider` means the `Context`, id-mapping (`to_internal_id` /
  `to_external_id`), and error machinery are identical across graph and flat search,
  reducing the learning surface for new contributors.

### Async vs sync scan API

`on_elements_unordered` and `distances_unordered` return a future, making the scan
surface async. This is the right default for disk-backed and network-backed backends
where advancing the scan involves real I/O. It also matches the `Accessor` surface,
keeping the two subsystems shaped the same way.

The cost is paid by in-memory consumers: the scan goes through the future machinery
even when results are immediately available. In a tight brute-force loop this overhead —
poll scaffolding, pinning etc — could be measurable.

We chose async because the wider audience of consumers (disk, network, mixed) benefits
more than in-memory consumers lose.

### Expand `Element` to support batched distance computation?

The current design yields one element per `next()` call, and the query computer scores
elements one at a time via `PreprocessedDistanceFunction::evaluate_similarity`. This could leave some optimization and performance on the table; especially with the upcoming effort around batched distance kernels. Of course, a consumer can choose to implement their own optimized implementation of `distances_unordered` that uses batching.

An alternative is to make `next()` yield a *batch* instead of a single vector representation like `Element<'_>`. Some work will need to be done to define the right interaction between the batch type, the element type in the batch, the interaction with `QueryComputer`'s types and way IDs and distances are collected in the queue.

## Future Work
- Support for other flat-search algorithms like - filtered, range and diverse flat algorithms as additional methods on `FlatIndex`.
- Index build -- this is just one part of the picture; more work needs to be done around how this fits in with any traits / interface we need for index build.

