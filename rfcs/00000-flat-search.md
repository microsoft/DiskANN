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

1. Define a streaming access primitive — `FlatIterator` — that mirrors the role
   `Accessor` plays for graph search but exposes a lending-iterator interface instead of
   a random-access one.
2. Provide flat-search algorithm implementations (with `knn_search` as default and filtered and diverse variants to opt-into) built on the new
   primitives, so consumers can use this against their own providers / backends. 
3. Expose support for features and implementations native to the repo like quantized distance computers out-of-the-box.

## Proposal

Let's start with the main analog to the `Accessor` trait for the `FlatIndex` - `FlatIterator`. 


### `FlatIterator`

```rust
pub trait FlatIterator: HasId + Send + Sync { // Has Id support
    // Element yielded by iterator
    type ElementRef<'a>; 

    // Mostly machinery to play nice with HRTB 
    type Element<'a>: for<'b> Reborrow<'b, Target = Self::ElementRef<'b>> + Send + Sync
    where 
        Self: 'a; 

    type Error: StandardError;

    fn next(
        &mut self,
    ) -> impl SendFuture<Result<Option<(Self::Id, Self::Element<'_>)>, Self::Error>>;

    // Default implementation for driving a closure on the items in the index. 
    fn on_elements_unordered<F>(
        &mut self,
        mut f: F,
    ) -> impl SendFuture<Result<(), Self::Error>>
    where F: Send + for<'a> FnMut(Self::Id, Self::ElementRef<'a>),
    {
        async move {
            while let Some((id, element)) = self.next().await? {
                f(id, element.reborrow());
            }
            
            Ok(())
        }
    }
}
```

The trait combines two access patterns:

- A required lending-iterator `next()`.
- A defaulted bulk method `on_elements_unordered` that consumes the entire scan via a
  callback. The default impl loops over `next`; iterators that benefit from prefetching,
  SIMD batching, or amortized per-element cost could override it.

Both methods are **async** (returning `impl SendFuture<...>`), matching
[`crate::provider::Accessor::get_element`]. Iterators backed by I/O — disk pages,
remote shards — return a real future; in-memory iterators wrap their result in
`std::future::ready`. 

The `Element` / `ElementRef` split is identical to `Accessor` and exists for the same
reason: to keep HRTB bounds on query computers from inducing `'static` requirements on
the iterator type.


### The glue: `FlatSearchStrategy`

While the `FlatIterator` is the primary object that provides access to the elements in the index for the algorithm, it is scoped to each query. We intorduce a constructor - `FlatSearchStrategy` - similar to `SearchStrategy` for `Accessor` to instantiate this object. A strategy is per-call configuration: stateless, cheap to construct, scoped to one
search. It produces both a per-query iterator and a query computer. 

```rust
pub trait FlatSearchStrategy<P, T>: Send + Sync
where
    P: DataProvider,
    T: ?Sized,
{
    /// The iterator type produced by [`Self::create_iter`]. Borrows from `self` and the
    /// provider.
    type Iter<'a>: FlatIterator
    where
        Self: 'a,

    /// The query computer produced by [`Self::build_query_computer`].
    type QueryComputer: for<'a, 'b> PreprocessedDistanceFunction<
            <Self::Iter<'a> as FlatIterator>::ElementRef<'b>,
            f32,
        > + Send
        + Sync
        + 'static;

    /// The error type for both factory methods.
    type Error: StandardError;

    /// Construct a fresh iterator over `provider` for the given request `context`.
    fn create_iter<'a>(
        &'a self,
        provider: &'a P,
        context: &'a P::Context,
    ) -> Result<Self::Iter<'a>, Self::Error>;

    /// Pre-process a query into a [`Self::QueryComputer`] usable for distance computation
    /// against any iterator produced by [`Self::create_iter`].
    fn build_query_computer(&self, query: &T) -> Result<Self::QueryComputer, Self::Error>;
}
```

The `ElementRef<'b>` that the distance function `QueryComputer` acts on is tied to the (reborrowed) element yielded by the `FlatIterator::next()`.

### `FlatIndex`

`FlatIndex` is a thin `'static` wrapper around a `DataProvider`. The same `DataProvider`
trait used by graph search is reused here — flat and graph subsystems share a single
provider surface and the same `Context` / id-mapping / error machinery.

```rust
pub struct FlatIndex<P: DataProvider> {
    provider: P,
    /* private */
}

impl<P: DataProvider> FlatIndex<P> {
    pub fn new(provider: P) -> Self;
    pub fn provider(&self) -> &P;

    pub fn knn_search<S, T, O, OB>(
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
}
```

The `knn_search` method is the canonical brute-force search algorithm:

1. Construct the iterator via `strategy.create_iter` to obtain a scoped iterator over the elements.
2. Build the query computer via `strategy.build_query_computer`.
3. Drive the scan via `iter.on_elements_unordered`, scoring each element and
   inserting `Neighbor`s into a `NeighborPriorityQueue<Id>` of capacity `k`.
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

### Async vs sync API for `FlatIterator`

`next()` and `on_elements_unordered` return a future, making the trait
async. This is the right default for disk-backed and network-backed iterators
where advancing the cursor involves real I/O. It also matches the `Accessor` surface,
keeping the two subsystems shaped the same way.

The cost is paid by in-memory consumers: every call to `next()` goes through the future
machinery even when the result is immediately available via `std::future::ready`. In a
tight brute-force loop this overhead — poll scaffolding, pinning etc — could be measurable. 

We chose async because the wider audience of consumers (disk, network, mixed) benefits
more than in-memory consumers lose. 

### Expand `Element` to support batched distance computation?

The current design yields one element per `next()` call, and the query computer scores
elements one at a time via `PreprocessedDistanceFunction::evaluate_similarity`. This could leave some optimization and performance on the table; especially with the upcoming effort around batched distance kernels.

An alternative is to make `next()` yield a *batch* instead of a single vector representation like `Element<'_>`. Some work will need to be done to define the right interaction between the batch type, the element type in the batch, the interaction with `QueryComputer`'s types and way IDs and distances are collected in the queue.

We opted for the scalar-per-element design for now because it is simpler to implement and
reason about. The hope is that batched distance computation can be layered on later as an opt-in sub-trait without breaking
existing iterators.

## Future Work
- Support for other flat-search algorithms like - filtered, range and diverse flat algorithms as additional methods on `FlatIndex`.

