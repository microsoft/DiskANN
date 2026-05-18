# Overhaul Paged Search

| | |
|---|---|
| **Authors** | Mark Hildebrand |
| **Contributors** | <!-- --> |
| **Created** | 2026-05-18 |
| **Updated** | 2026-05-18 |

## Summary

Replace the `SearchState<..., ExtraState: 'static>` pattern for paged search with a lifetime-bound `PagedSearch<'a, ...>` in `diskann`, and document a channel-based spawned-task pattern for downstream consumers that need to cross `tokio::spawn` or FFI boundaries.
This removes the `'static` requirement on query computers and search strategies, enabling future trait simplification.

## Motivation

### Background

Paged search allows callers to retrieve nearest-neighbor results incrementally (one "page" at a time) without restarting the graph traversal.
The search state (scratch buffers, the priority queue, the query computer) must persist across page boundaries.

Earlier synchronous version of DiskANN did this by persisting the search state manually and passing the search state explicitly to the next-page requests.
The async rewrite stuck with this pattern where callers were required  to manage a `SearchState` struct whose `ExtraState` type parameter carried a `'static` bound:
```rust
// OLD: ExtraState must be 'static
pub struct SearchState<VectorIdType, ExtraState: 'static = NoExtraState> { ... }
pub type PagedSearchState<DP, S, C> = SearchState<<DP as DataProvider>::InternalId, (S, C)>;

// Note
// * S: SearchStrategy<DP, T> for some T
// * C: S::QueryComputer
```
In downstream crates that expose paged search across FFI or task boundaries, the state was traditionally type-erased behind `Box<dyn Searcher>` (which also captured the `DiskANNIndex`) and sent as an opaque pointer.
This required both the strategy and the query computer (parameters `S` and `C`) to be `'static`.

### Problem Statement

The `'static` bound on `BuildQueryComputer::QueryComputer` propagates throughout the trait hierarchy:

```rust
// OLD
type QueryComputer: ... + Send + Sync + 'static;
```

This prevents:

1. Query computers that borrow from the index or context (common with quantization tables).
2. Fusing the query-computer into the accessor.
   Due to the lifetime needed by accessors, they can't be persisted in a `'static` struct this way.
   However, query computers may contain non-trivial pre-processed state, meaning recreating them on each new page retrieval is a performance footgun.
3. Simplifying the `SearchStrategy` / `Accessor` trait tower by removing unnecessary indirection introduced solely to satisfy `'static`.

### Goals

1. Remove the `'static` bound from `BuildQueryComputer::QueryComputer`.
2. Remove `SearchState`, `NoExtraState`, and `PagedSearchState` from the public API.
3. Provide a `PagedSearch<'a, ...>` handle that is lifetime-bound to the index and context, encapsulating all search state.
4. Document the channel-based pattern for downstream consumers that need to cross task/FFI boundaries (where `'static` is inherently required by the runtime, not by `diskann`).

## Proposal

The key idea is this: DiskANN for better or worse is already fully async, and async Rust despite its flaws already provides a clean way of doing this without requiring our traits to bend over backwards.
So let's embrace async to actually help us for a change.

### Core library (`diskann`)

Replace the old split API:

```rust
// OLD
index.start_paged_search(strategy, ctx, query, l) -> SearchState<...>
index.next_search_results(ctx, &mut state, k, &mut buf) -> usize
```

With a self-contained handle:

```rust
// NEW
impl DiskANNIndex<DP> {
    pub fn paged_search<'a, S, T>(
        &'a self,
        strategy: S,
        context: &'a DP::Context,
        query: T,
        l_value: usize,
    ) -> impl SendFuture<ANNResult<PagedSearch<'a, DP, S, T>>>
    where
        S: SearchStrategy<DP, T>,
        T: Copy + Send + 'a;
}

pub struct PagedSearch<'a, DP: DataProvider, S: SearchStrategy<DP, T>, T> {
    index: &'a DiskANNIndex<DP>,
    context: &'a DP::Context,
    scratch: SearchScratch<DP::InternalId>,
    computed_result: Vec<Neighbor<DP::InternalId>>,
    next_result_index: usize,
    search_param_l: usize,
    strategy: S,
    computer: S::QueryComputer,
    _query: PhantomData<fn(T)>,  // covariant, always Send+Sync
}

impl PagedSearch<'a, DP, S, T> {
    pub fn next_page(&mut self, k: usize) -> impl SendFuture<ANNResult<Vec<Neighbor<...>>>>;
}
```

The key change is that `PagedSearch` borrows all the necessary components.
There is no need to deconstruct the search components into `'static` pieces after each paged search and "reassemble" them on subsequent searches.

### Crossing spawn boundaries: the channel pattern

`PagedSearch<'a, ...>` borrows the index, so it cannot be sent to a `tokio::spawn`'d task directly.
When a long-lived session or an FFI boundary requires `'static` ownership, the recommended pattern is to **spawn a task that owns the search state as a local variable** and communicate with it via channels:

```rust
// Types are illustrative — adapt names to your crate.

type PageResult = ANNResult<Vec<Neighbor<ExternalId>>>;

/// Spawn a paged search session. The index is held by Arc so the task is 'static.
///
/// Returns a request channel and a result channel. The caller sends the desired
/// page size (`k`) and awaits the corresponding result on the other end.
fn spawn_paged_session(
    index: Arc<DiskANNIndex<DP>>,
    context: Arc<DP::Context>,
    query: T,
    l: usize,
) -> (mpsc::Sender<usize>, mpsc::Receiver<PageResult>) {
    let (req_tx, mut req_rx) = mpsc::channel::<usize>(1);
    let (res_tx, res_rx) = mpsc::channel::<PageResult>(1);

    tokio::spawn(async move {
        // Borrow from the Arc — these references are scoped to the task.
        let mut search = index.paged_search(strategy, &*context, query, l).await.unwrap();

        while let Some(k) = req_rx.recv().await {
            let page = search.next_page(k).await;
            if res_tx.send(page).await.is_err() {
                break; // caller dropped the result receiver
            }
        }
        // Request channel closed -> caller dropped sender -> clean shutdown.
    });

    (req_tx, res_rx)
}
```

Key properties of this pattern:

1. **`'static` is confined to the spawn boundary**: the `Arc<Index>` satisfies the runtime's requirement, while the borrow from it lives entirely inside the task's local scope.
   Importantly, even though `PagedSearch` borrows, it can be embedded inside a `'static` future.
2. **State is fully encapsulated**: callers never see `SearchScratch`, `QueryComputer`, or any internal types.
3. **Clean shutdown**: dropping the request sender closes the channel; the task exits gracefully.
4. **Per-request context**: the request channel can carry additional metadata (profiling tokens, cancellation flags, etc.) without polluting the core API.

### Migration guide

| Old pattern | New pattern |
|---|---|
| `index.start_paged_search(s, ctx, q, l)` | `index.paged_search(s, ctx, q, l).await` |
| `index.next_search_results(ctx, &mut state, k, &mut buf)` | `search.next_page(k).await` |
| `SearchState<Id, (S, C)>` | `PagedSearch<'a, DP, S, T>` |
| `PagedSearchState<DP, S, C>` | `PagedSearch<'a, DP, S, T>` |
| Check return count for exhaustion | Check `page.is_empty()` |
| Type-erased `Box<dyn Searcher>` across task/FFI boundaries | Channel + spawned task (see above) |

## Feasibility via FFI

This is a pretty big change in API, but it enables some significant future simplifications to our trait hierarchy by removing the `'static` special case introduced by paged search.
An internal user of paged search was ported to this new approach to check the feasibility.
While it was a bit of work to overcome the impedance mismatch of the quirks of that integration, the end result is cleaner, has fewer overall task spawns, and fewer FFI related race conditions.
And really, this integration was already basically doing this same thing behind the scenes.

## Alternatives

The main alternative I see is to keep the status quo with explicit state management.
While some of planned trait simplifications are still on the table, I think the opportunity to align paged search with the rest of the trait hierarchy is well worth it.

## Benchmark Results

No performance change expected (nor observed in the simulator for the aforementioned internal FFI user) since the search algorithm is identical.
Existing Rust code will have a similar pattern of future usage as before, just packaged slightly differently.

## References

1. [RFC 3498 — Lifetime Capture Rules 2024](https://rust-lang.github.io/rfcs/3498-lifetime-capture-rules-2024.html) —
   Rust edition 2024 changes that make `impl Trait + 'a` returns more ergonomic.
2. [tokio::sync::mpsc](https://docs.rs/tokio/latest/tokio/sync/mpsc/index.html) — the channel
   primitive used in the spawned-task pattern.
