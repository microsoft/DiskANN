# Simplify the Search Traits

| | |
|---|---|
| **Authors** | Mark Hildebrand |
| **Contributors** | |
| **Created** | 2026-06-02 |
| **Updated** | 2026-06-02 |

## Summary

We can simplify our search API considerably by collapsing `Accessor`, `BuildQueryComputer`, `NeighborAccessor`, `SearchExt`, and `ExpandBeam` into a single `SearchAccessor` trait.

## Motivation

The current structure of DiskANN relies on several interacting traits for search:

* `Accessor`: Random Access to elements + optional bulk-access.
* `BuildQueryComputer<T>: Accessor`: Build an independent query computer for a query type `T`.
* `NeighborAccessor`: Access to raw adjacency lists.
* `SearchExt`: Some extensions for search - mainly around start points.
* `ExpandBeam`: The **actual** thing we care about, which can be built (somewhat inefficiently) from the previous four traits.

This hierarchy arose organically as functionality was needed, and paged search (see PR [1078](https://github.com/microsoft/DiskANN/pull/1078)) is responsible for at least some of this, with the old design requiring a `'static` sub-component of these interacting traits.
However, it has become clear that `ExpandBeam` is the most valuable portion of this, with serious implementations targeting that bulk method for the most efficiency.

With the power of hindsight, we can simplify this.

### Problem Statement

Simplify search.

### Goals

1. Collapse the interacting extension traits for search.
2. Don't lose performance.

## Proposal

Simplify the requirements of search to
```rust
pub trait SearchAccessor: HasId + Send + Sync {
    async fn starting_points(&self) -> ANNResult<Vec<Self::Id>>;

    async fn start_point_distances<F>(
        &mut self,
        f: F,
    ) -> ANNResult<()>
    where
        F: FnMut(Self::Id, f32) + Send;

    async fn expand_beam<Itr, P, F>(
        &mut self,
        ids: Itr,
        pred: P,
        on_neighbors: F,
    ) -> ANNResult<()>
    where
        Itr: Iterator<Item = Self::Id> + Send,
        P: HybridPredicate<Self::Id> + Send + Sync,
        F: FnMut(Self::Id, f32) + Send;

    // Provided Methods
    fn terminate_early(&mut self) -> bool {
        false
    }

    async fn is_not_start_point(
        &self,
    ) -> ANNResult<impl Fn(Self::Id) -> bool + Send + Sync + 'static> {
        // provided
    }

    async fn num_starting_points(&self) -> ANNResult<usize> {
        // provided
    }
}
```
and `SearchStrategy` to
```rust
pub trait SearchStrategy<'a, Provider, T>: Send + Sync
where
    Provider: DataProvider,
{
    type SearchAccessor: SearchAccessor<Id = Provider::InternalId>;
    type SearchAccessorError: StandardError;
    fn search_accessor(
        &'a self,
        provider: &'a Provider,
        context: &'a Provider::Context,
        query: T,
    ) -> Result<Self::SearchAccessor, Self::SearchAccessorError>;
}
```
The following traits are removed:

* `Accessor`: No need to element-wise access
* `BuildQueryComputer`: Fused with `SearchAccessor`.
* `SearchExt`: Superseded by `SearchAccessor`.
* `ExpandBeam`: Superseded by `SearchAccessor`.
* `SearchAccessor`s need not implement neighbor delegation.

Several important things to note:

1. The implementation of `SearchAccessor` is now responsible for computing its own distances.
   This has implications for user crates that will be discussed later.

2. `SearchStrategy::search_accessor` now accepts the query, and a lifetime has been lifted into the trait level to allow `SearchAccessor` implementations to borrow the query.

## Trade-offs

### Advantages

By funneling all necessary components for search into a single trait, we gain the following:

1. All search related functionality is in one place, rather than spread out across multiple interacting traits.
   This (in theory) makes it easier to audit implementations as there's just less to view.

2. Requiring `expand_beam` as the primitive encourages efficiency.
   When building the algorithm incrementally from `Accessor::get_element` + distance computation and friends, it becomes less clear **how** the algorithm lowers to the actual backend.
   With a monolithic design, optimization opportunities become clearer.
   This can be seen with the code simplifications to `diskann-garnet`, `diskann-disk`, and `diskann-label-filter`.
   All of these benefit from more directly matching the algorithm to the backing store.

3. There's less machinery to maintain in `diskann` with all the provided implementations.

4. Decoupling implementation from the rigid trait hierarchy for search is actually a big deal.
   This means that `diskann` can significantly relax its notion of an "internal" id as it no longer needs to explicitly navigate it.
   As a consequence, this will open up future opportunities for richer IDs.

5. Clearer path for fusing filtered traversal with data access by introducing a single `FilteredSearchAccessor` rather than replicating the complicated trait hierarchy.

6. Coarsening the granularity of the extension point (i.e., focusing on `expand_beam`) opens several opportunities:

   * Implementations can now use fallible distance functions.
     This was something that was hard to incorporate into the current abstraction hierarchy without considerable noise.

   * The need to explicitly handle non-critical errors in DiskANN is almost entirely eliminated.
     In fact, we may even be able to get rid of ranked-errors entirely.
     The contract of `SearchAccessor::expand_beam` is such that implementations do not need to be exhaustive.
     This allows implementations to handle non-critical errors locally without needing to involve the higher level algorithm.

### Disadvantages

1. Existing users need to split accessors used for search from those used for pruning.
   This work is largely mechanical since the functionality needed by each flavor of accessor is now almost completely orthogonal.

2. We lose the ability to "express" algorithms in terms of `Accessor`/`BuildQueryComputer`/`NeighborAccessor` to experiment with different traversal modes.
   I'm not entirely convinced here: already range search and multi-hop filtering piggy-back off `ExpandBeam` rather than reinventing the wheel.

3. Implementing `SearchAccessor::expand_beam` is arguably more difficult than the various bulk methods of `Accessor`.
   While superficially true, I argue that the minor extra algorithmic complexity is more than offset by collapsing everything into a single trait.

4. There is a risk that `SearchAccessor` is non-exhaustive with what we need in the future.
   This is fair, but tacking on extra functionality to our existing hierarchy is not necessarily cleaner.

## Future Work

- [ ] This PR does not touch insert - but a similar simplification can be done there.
- [ ] Rework how IDs are handled to increase flexibility there.
- [ ] Add a filtered extension to allow fusing predicate evaluation with data access.

