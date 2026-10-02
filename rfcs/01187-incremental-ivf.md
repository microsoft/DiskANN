# Incremental IVF Accessor Architecture

| | |
|---|---|
| **Authors** | Aditya Krishnan |
| **Created** | 2026-08-19 |
| **Updated** | 2026-08-24 |
| **Status** | Draft |

## Summary

Add an accessor-based architecture for incrementally maintained inverted-file
(IVF) indexes, along with co-location sets of centroids to assist with ordering 
spatially similar posting lists on slower storage mediums. The architecture 
separates the GraphIVF algorithm from point storage, centroid navigation, and 
inverted-list access without prescribing a cross-component snapshot or commit 
protocol.

The generic API has two principal extension points:

1. A query-bound search accessor selects centroids and scans the corresponding
   lists.
2. An operation-scoped maintenance accessor presents one unified view of point
   data, centroids, reverse assignments, inverted lists, and co-location sets 
   while planning and applying a split or dissolve.

Each concrete strategy or aggregate provider is responsible for coordinating
its components. It may use exclusive borrowing, locks, epochs, immutable roots,
database transactions, manifests, or provider-specific recovery internally.
Those mechanisms are deliberately absent from the generic IVF traits.

The first implementation serializes mutation through `&mut self`. This prevents
search and mutation from overlapping through the same index value and gives an
in-memory provider a straightforward implementation path. More concurrent or
durable providers may implement stronger private guarantees without changing
the algorithm-facing interface.

## Motivation

### Background

The initial IVF interface in `diskann/src/ivf` supports a conventional fixed
partition:

1. A coarse accessor selects `nprobe` list ids.
2. A fine accessor scans those lists and emits `(point_id, distance)` pairs.
3. Insert selects one list and appends one point.

That interface is appropriate for immutable IVF and fixed-centroid IVF-Flat.
It also follows the graph module's useful strategy/accessor split: algorithms
invoke coarse-grained operations and do not know how vectors or edges are laid
out.

The experimental online GraphIVF algorithm has a larger mutation unit. One
insert batch may:

- route many points against one centroid set;
- identify several overflowing lists;
- fit two children per overflowing parent with 2-means;
- create child centroids and retire parent centroids;
- read neighboring lists; and
- reassign all points in the affected regions.
- reassign centroids to other co-location sets, and create and retire co-location sets

A delete batch removes points, retires underfull centroids, and moves each
victim list's remaining members to nearby survivors. These are set-level
partition changes rather than independent appends.

### Problem Statement

The fixed-partition interface does not provide enough behavior for GraphIVF.

First, coarse and fine accessors are independently constructed. If mutation can
overlap search, centroid selection and list scanning may observe incompatible
states.

Second, `append(list, id, vector)` cannot represent a split or dissolve.
Adding imperative `remove`, `create_list`, and `replace_centroid` calls would
make the generic algorithm choose a partial-write order for every backend.

Third, GraphIVF planning needs list sizes, reverse assignments, list members,
canonical vectors, centroid vectors, fresh list ids, and co-location sets. These operations must
be available in batch-friendly forms so disk and blob implementations do not
devolve into one request per point.

One solution is to standardize version tokens, pinned read snapshots, staged
mutations, compare-and-swap publication, and conflict retries. That provides
strong portable semantics, but it also makes a specific MVCC-like model part of
every provider implementation. It creates many associated types before the
in-memory GraphIVF algorithm has established which guarantees are necessary in
practice.

This RFC instead makes consistency an accessor contract. The algorithm uses one
accessor for an entire operation; the implementation decides how that accessor
coordinates its provider, centroid index, and inverted-list store.

### Goals

1. Keep GraphIVF policy independent from list location, physical layout,
   encoding, caching, and I/O scheduling.
2. Keep exact and graph centroid navigation interchangeable.
3. Support batched insert, local split, and regional reassignment.
4. Support batched delete, underfull-list dissolve, and reassignment.
5. Give providers coarse-grained, batch-friendly read operations.
6. Represent the final logical mutation declaratively rather than as ordered
   low-level storage calls.
7. Keep canonical maintenance vectors independent from optimized list scan
   representations such as PQ codes.
8. Reuse execution contexts, ranked errors, and the graph module's
   strategy/accessor style, with one IVF-specific provider trait for identity.
9. Leave locking, reader visibility, durability, rollback, and recovery to each
   concrete accessor implementation.

### Non-Goals

- A portable snapshot, generation, or compare-and-swap protocol.
- A generic transaction or write-ahead-log abstraction.
- Concurrent mutation through one `DynamicIvfIndex` in the first version.
- Portable concurrent search/mutation semantics.
- Distributed consensus or cross-service transactions.
- Defining a durable list file format.
- Choosing a specific k-means implementation.
- Residual PQ training or codebook lifecycle management.
- Safely composing arbitrary independently implemented centroid and list stores.

## Terminology And Invariants

### Terminology

- **Point id**: the provider's internal vector id. External ids and the
  translation between the two are owned by the provider (`Provider`).
- **List id**: the stable logical id shared by a centroid and its inverted list.
  Like the point id, it belongs to the provider (`Provider::ListId`), so
  every search and maintenance accessor over one provider uses the same list
  ids.
- **Unified view**: mutually compatible point data, centroid state, reverse
  assignments, and inverted lists presented by one operation accessor.
- **Selected lists**: list ids and their coarse distances, nearest first. A
  search accessor scans only lists it selected itself.
- **Canonical vector**: the full-precision, or `f32`-decodable, representation
  used for centroid fitting and exact regional reassignment.
- **Scan payload**: the representation stored with a list for query scoring. It
  may be full precision, quantized, or otherwise encoded.
- **Partition update**: the complete logical centroid and point-membership
  change calculated by the algorithm.
- **Co-Location Set**: a set of list ids that share some spatial similarity. May be the identity mapping.

### Required Logical Invariants

After a maintenance accessor returns `Ok(())` from `apply`:

1. Every visible point is assigned to exactly one live list.
2. `assignment[point] == list` if and only if `point` is a member of `list`.
3. Every live list has exactly one live centroid with the same logical id.
4. Every list member has an available canonical vector and scan payload.
5. Retired list ids are never reused for a different centroid.
6. Point identity mappings agree with point visibility in the partition.
7. Inserts may trigger splits but not dissolves.
8. Deletes may trigger dissolves but not splits.
9. When dissolves are enabled,
   `2 * merge_threshold <= split_threshold` provides hysteresis.
10. Each live list may belong to only one co-location set

The generic API does not specify whether an accessor uses copied state,
exclusive access, or a versioned backend to preserve these invariants.

### Accessor Consistency Contract

A correct strategy must construct accessors with these properties:

1. All methods on one maintenance accessor operate on one compatible logical
   view until `apply` consumes it.
2. A search accessor scans only lists it selected itself.
3. Successful `apply` does not expose a mixture of old and new component state
   through subsequent index operations.
4. If an operation fails after making irreversible changes, subsequent index
   operations must either observe a coherent state or fail closed, for example
   through provider poisoning.
5. Any stronger behavior, including rollback, crash atomicity, reader isolation,
   durability, and retryability, is documented by the concrete provider.

These rules are semantic requirements and are not encoded by public version
tokens. Reusable provider contract tests are therefore an important part of the
implementation.

## Proposal

### Source Layout

```text
diskann/src/ivf/
  mod.rs
  dynamic.rs    contracts proposed by this RFC
  update.rs     declarative partition updates applied by the maintenance accessor
  grouped.rs    flat grouped layouts used by planning
  index.rs      DynamicIvfIndex initialize/insert orchestration
  online/       insert planning: staging, split planning, kernels
    batch.rs    staged and routed insert batch
    split.rs    the split plan: gather each region, fit children, place points
    kernels.rs  distance, nearest-candidate, and Lloyd kernels
  test/         in-memory provider for index tests
```

The initial fixed-partition prototype (`glue.rs`, `index.rs`, and its test
provider) is removed; the contracts in `ivf::dynamic` replace it.

### Responsibilities

| Component | Owns | Does not decide |
|---|---|---|
| Dynamic IVF algorithm | split/dissolve policy, affected regions, child fitting, reassignment, co-location sets | physical layout, I/O shape, provider consistency mechanism |
| Aggregate provider/strategy | compatible point, centroid, and list implementations | split/dissolve policy |
| `CentroidIndex` | live centroid catalog and exact or approximate navigation | inverted-list storage |
| `CoLocationSet` | assignment of centroids to co-location groups | how co-location groups dictate ordering within the data provider |
| Search accessor | query-bound selection, list scoring, coherent search view | top-k policy |
| Maintenance accessor | unified planning view, point staging, id reservation, final update application | which centroids and points should move |

The generic algorithm does not separately open a centroid index and inverted
list store and then attempt to coordinate them. A concrete strategy composes
those components and returns one accessor that already knows how they fit
together. Consequently, arbitrary implementations with matching Rust types are
not automatically safe to combine.

The canonical vector belongs to the provider rather than to a particular list.
This permits a list backend to store only PQ codes while maintenance still reads
`f32` vectors. Moving a point between lists does not change its identity or
canonical representation, although the destination scan payload may need to be
encoded again.

### Selected Lists

Selection returns plain values, nearest first:

```rust
pub struct SelectedList<ListId> {
    pub id: ListId,
    pub distance: f32,
}
```

A search accessor may privately retain a lock guard, epoch, immutable root,
manifest, prefetch state, or other read context between selection and scanning.
None of that state appears in the public type, so selected lists are meaningful
only to the accessor that selected them.

### Search Accessor

One query-bound accessor owns both coarse and fine operation state:

```rust
pub trait SearchAccessor: HasId + Send + Sync {
    type ListId: VectorId;
    type Error: ToRanked + Debug + Send + Sync + 'static;

    fn select_lists(
        &mut self,
        nprobe: usize,
    ) -> impl SendFuture<Result<Vec<SelectedList<Self::ListId>>, Self::Error>>;

    fn scan_lists<F>(
        &mut self,
        lists: &[SelectedList<Self::ListId>],
        emit: F,
    ) -> impl SendFuture<Result<ScanStats, Self::Error>>
    where
        F: FnMut(Self::Id, f32) + Send;
}
```

The interface makes no callback-order guarantee. A backend may coalesce disk
ranges, combine blob requests, process lists concurrently, and merge scored
points before invoking `emit`.

The strategy follows the graph module's accessor factory pattern. The provider
declares the identity types once and translates the ids of visible points; all
reads and writes go through accessors:

```rust
pub trait Provider: Sized + Send + Sync + 'static {
    type Context: ExecutionContext;
    type InternalId: VectorId;
    type ExternalId: PartialEq + Send + Sync + 'static;
    type ListId: VectorId;
    type Error: ToRanked + Debug + Send + Sync + 'static;

    fn to_internal_id(
        &self,
        context: &Self::Context,
        gid: &Self::ExternalId,
    ) -> Result<Self::InternalId, Self::Error>;

    fn to_external_id(
        &self,
        context: &Self::Context,
        id: Self::InternalId,
    ) -> Result<Self::ExternalId, Self::Error>;
}

pub trait SearchStrategy<'a, P: Provider, T>: Send + Sync {
    type SearchAccessor: SearchAccessor<Id = P::InternalId, ListId = P::ListId>;
    type Error: StandardError;

    fn search_accessor(
        &'a self,
        provider: &'a P,
        context: &'a P::Context,
        query: T,
    ) -> Result<Self::SearchAccessor, Self::Error>;
}
```

`ivf::dynamic::Provider` replaces the crate-wide `DataProvider` for this index.
That trait is shaped around graph insertion, where `SetElement` stores a point
immediately and a `Guard` later confirms or rolls it back; IVF points instead
stay staged in an accessor until `apply`, so the guard type and the separate
list-id trait it would need are not part of the IVF contract. Search returns
internal ids.

Search orchestration is deliberately unaware of consistency mechanics:

```text
accessor = strategy.search_accessor(provider, context, query)
lists = accessor.select_lists(nprobe)
accessor.scan_lists(&lists, |id, distance| top_k.insert(id, distance))
post_process(point statuses and external ids)
```

### Centroid Index

Centroids remain memory resident behind one interface:

```rust
pub trait CentroidIndex: Send + Sync {
    type ListId: VectorId;
    type Error: ToRanked + Debug + Send + Sync + 'static;

    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
    fn centroid(&self, id: Self::ListId) -> Option<&[f32]>;

    fn select(
        &self,
        query: &[f32],
        nprobe: usize,
    ) -> impl SendFuture<Result<Vec<SelectedList<Self::ListId>>, Self::Error>>;
}
```

An exact implementation scores a packed matrix of live centroids. A graph
implementation stores the same authoritative centroid catalog plus a DiskANN
graph used for navigation. It filters graph results against the live catalog
and completes an undersized result with exact selection.

The generic interface does not prescribe how a centroid index is updated or
rebuilt. The maintenance accessor's `apply` implementation coordinates centroid
changes with list and point changes. A provider may mutate its graph in place,
construct a replacement graph, temporarily use exact routing, or poison itself
after an unrecoverable partial graph update.

### Declarative Partition Updates

The algorithm describes the desired logical result rather than issuing ordered
storage commands. The insert path's update in `ivf::update` lists primitive
changes:

```rust
pub enum CentroidDelta<L> {
    Install { id: L, centroid: Box<[f32]> },
    Retire { id: L },
}

pub enum PointDelta<Id, L> {
    // A newly staged point joins `to`.
    Append { id: Id, to: L },
    // An existing point leaves `from` and joins `to`.
    Move { id: Id, from: L, to: L },
}

pub struct InsertionUpdate<Id, L> {
    centroids: Vec<CentroidDelta<L>>,
    points: Vec<PointDelta<Id, L>>, // sorted by destination, then id
}
```

Bootstrap and insert batches are both mixes of these deltas. Bootstrap installs
centroids. An insert appends its points; when it splits, it also installs two
children per parent, retires the parent, and moves the parent's members and any
neighbor members whose list changed. Each variant has one meaning in every
update, so a provider handles four kinds of change rather than every
combination of operations. The delete path's update is not defined yet; it
will follow the same delta style.

`InsertionUpdate`'s fields are private and its constructor is private to the
IVF module, so only the index can create updates. Providers receive them
through `MaintenanceAccessor::apply` and can only read them. The delta enums are
necessarily public so providers can match on them, but they do nothing outside
an update. This keeps the provider contract one-directional: the algorithm
decides the update, and providers decide how to store it.

The planner records each decision as a delta: every staged point is appended to
its final list, and every existing point whose list changed is moved. That
makes "every staged point is placed once" and "every parent member is moved out
once" properties of how the update is built, so the constructor only sorts the
point deltas and performs no validation.

The enums make a move with neither a source nor a destination unrepresentable.
The planner guarantees the rest by construction, and `InsertionUpdate`
documents them for providers:

- installed ids are fresh, and their centroids are finite with the accessor's
  dimension;
- retired ids are live, and each list appears in at most one centroid delta;
- each point appears in at most one point delta, and every staged point is
  appended;
- a move's source holds the point and differs from its destination;
- every destination is live after the update, so no point joins a retired
  list; and
- every member of a retired list is moved out.

Children may end up empty.

#### Layout

Point deltas are sorted by destination, then point id, so each list's incoming
points, appended and moved alike, form one contiguous run. A provider can
therefore append each list's incoming points with one write, re-encode residual
payloads against one centroid per run, and process distinct runs in parallel.
Because every member of a retired list is moved out, a provider may drop a
retired list wholesale instead of removing its points one by one. No point
vectors cross the update boundary, and each installed centroid crosses once.

#### Validating Inputs Instead Of Updates

Because the planner is trusted, the index validates untrusted data once, where
it enters, and builds updates without further checks:

- caller input: initial centroids must be non-empty, finite, and of the
  provider's dimension;
- vector reads and staging: every vector row must be written with finite
  values (rows start as NaN, so unwritten rows are caught by the same check);
- staging: one distinct id must be returned per staged point; and
- reservations: reserved list ids must be distinct and not live.

A provider that violates the accessor contract therefore fails the operation
before any update is built, rather than receiving an inconsistent update.

#### Provider Checks

View-dependent facts cannot be checked by the index. It guarantees them by
building updates only from what the same accessor returned:

- installed ids were reserved by this accessor;
- retired lists and destinations are live in the accessor's planning view;
- every moved point belonged to its source list in that view; and
- every appended point was staged by this accessor.

A provider may still verify these facts, for example in debug builds.

The update remains useful even for an in-place backend. It gives the provider
the entire mutation before it chooses write order, batching, rollback, or
recovery behavior.

Co-location set updates are expected to follow the same declarative pattern.
They are deferred until the insert path and maintenance accessor stabilize, so
the co-location types are not part of the code yet.

### Maintenance Accessor

The maintenance accessor holds planning reads, id reservation, and the final
`apply`; a separate trait stages new points, so only staging depends on the
input element type `T`:

```rust
pub trait MaintenanceAccessor<P: Provider>: Send + Sized {
    type Centroids: CentroidIndex<ListId = P::ListId, Error = Self::Error>;
    type Error: ToRanked + Debug + Send + Sync + 'static;

    fn dim(&self) -> usize;
    fn centroids(&self) -> &Self::Centroids;

    fn list_size(
        &mut self,
        list: P::ListId,
    ) -> impl SendFuture<Result<usize, Self::Error>>;

    fn read_members(
        &mut self,
        list: P::ListId,
    ) -> impl SendFuture<Result<&[P::InternalId], Self::Error>>;

    fn read_vectors(
        &mut self,
        ids: &[P::InternalId],
        out: MutMatrixView<'_, f32>,
    ) -> impl SendFuture<Result<(), Self::Error>>;

    fn reserve_list_ids(
        &mut self,
        count: usize,
    ) -> impl SendFuture<Result<Vec<P::ListId>, Self::Error>>;

    fn apply(
        self,
        update: InsertionUpdate<P::InternalId, P::ListId>,
    ) -> impl SendFuture<Result<(), Self::Error>>;
}

pub trait StageElements<P: Provider, T>: MaintenanceAccessor<P> {
    fn stage(
        &mut self,
        points: &[(P::ExternalId, T)],
        out: MutMatrixView<'_, f32>,
    ) -> impl SendFuture<Result<Vec<P::InternalId>, Self::Error>>;
}
```

The accessor takes its id types from the provider, so they are declared in one
place.

Reads are shaped by how much data they move. List reads, `list_size` and
`read_members`, take one list and return its result: an insert batch makes at
most a few hundred of them, and a list's members are bounded by the split
threshold. `read_vectors` moves far more data, one vector per region member, so
it takes a batch and writes row `i` of the caller's matrix with the canonical
vector of `ids[i]`, in any order. Providers can therefore sort, coalesce, or
parallelize those reads. `dim` fixes the length of every canonical vector and
centroid, so callers can allocate before reading. A read of several lists at
once, for example a whole co-location group, can be added when co-location sets
are integrated.

`stage` takes the whole insert batch in one call. The provider allocates
internal ids, rejects duplicate external ids, converts each element to its
stored representation, and writes the canonical `f32` vector of each point into
the caller's matrix. Staged points remain invisible, including to
`to_internal_id`, until the accessor applies an update that places them, and are
discarded if the accessor is dropped first. `read_vectors` therefore only covers
visible points. A provider that accepts several input element types implements
`StageElements` once per type. Reverse-assignment lookups and deletion staging
will be added with the delete path; the latter records point removal while
keeping whatever data the accessor needs for dissolve planning.

These methods may delegate to existing provider machinery such as
`SetElement`, but the generic GraphIVF algorithm does not independently complete
a `SetElement` guard. The accessor coordinates point lifecycle with its other
components using its private mechanism.

`apply` consumes the accessor. This prevents further planning reads through the
same operation object and provides one generic handoff containing the full
logical change. Bootstrap and inserts both produce an `InsertionUpdate`, so
`apply` is one accessor method rather than a trait per update type, and the
index needs no extra bound to apply what it planned. `apply` is not a generic
transaction protocol:

- `Ok(())` guarantees a coherent resulting index.
- `Err` does not guarantee rollback or identify a conflict.
- Dropping an accessor does not have one portable cleanup guarantee.
- Reader visibility and durability are implementation-defined.

A provider may document a stronger contract. For example, a blob provider can
privately stage immutable objects and conditionally replace a manifest even
though the generic API exposes only `apply`.

The maintenance accessor is produced by a strategy that borrows the provider
exclusively:

```rust
pub trait MaintenanceStrategy<'a, P: Provider>: Send + Sync {
    type MaintenanceAccessor: MaintenanceAccessor<P>;
    type Error: StandardError;

    fn maintenance_accessor(
        &'a self,
        provider: &'a mut P,
        context: &'a P::Context,
    ) -> Result<Self::MaintenanceAccessor, Self::Error>;
}
```

Search strategies receive `&'a P`; maintenance strategies receive `&'a mut P`.
See [Index Borrowing Model](#index-borrowing-model).

Because accessors take their ids from the provider, the index adds a bound only
where it stages: `S::MaintenanceAccessor: StageElements<P, T>` for inserts, and
none for initialization.

### Inverted-List Store Boundary

This RFC does not add a low-level `InvertedListStore` trait. Storage backends
vary substantially in useful operation shape: an in-memory store borrows slices,
a disk store coalesces ranges, and a blob store batches objects. Flattening them
into point-level `get`, `append`, and `remove` methods would discard those
optimization opportunities.

Search and maintenance accessors are the algorithm-facing inverted-list
abstraction. Concrete crates may define lower-level reusable list-store traits
and compose them inside a strategy. The aggregate accessor remains responsible
for compatibility with the selected centroid index and the provider.

### Index Borrowing Model

The initial dynamic index uses exclusive mutation:

```rust
impl<P: Provider> DynamicIvfIndex<P> {
    pub fn search(&self, /* ... */) -> impl SendFuture<ANNResult<SearchStats>>;

    pub fn insert_batch(&mut self, /* ... */)
        -> impl SendFuture<ANNResult<InsertStats>>;

    pub fn delete_batch(&mut self, /* ... */)
        -> impl SendFuture<ANNResult<DeleteStats>>;
}
```

Rust then prevents a search future borrowing this index from overlapping a
mutation future borrowing it mutably. Concurrent searches remain possible.

Mutations forward that exclusivity to the provider: the index lends `&mut P` to
`MaintenanceStrategy::maintenance_accessor`, and the resulting accessor holds it
until `apply` consumes the accessor or it is dropped. An in-memory provider can
therefore implement `apply` as direct, in-place mutation of plain owned state,
with no locks or interior mutability. If the provider were lent only as `&P`,
every provider would need interior synchronization even though the index
already guarantees exclusivity.

This rule covers operations through the index API, not aliases or administrative
paths exposed independently by a provider, such as state shared through an
`Arc`. Providers that expose such paths are responsible for coordinating them.

This choice can later be relaxed to `&self` mutation for providers that support
it, but doing so requires a new concurrency contract and should be deliberate.
Such a relaxation would add a separate shared-borrow maintenance strategy
rather than weaken the exclusive one.

## Algorithms

### Bootstrap

Bootstrap uses the same maintenance interface as later mutations:

1. Resolve initial full-precision centroids from an explicit matrix or a seed
   strategy over caller-provided sample data.
2. Construct a maintenance accessor over an empty provider.
3. Reserve one stable list id per initial centroid.
4. Apply an `InsertionUpdate` installing those centroids, with no point deltas.

Initial sample points are not implicitly assigned. Callers insert them through
the normal batch path. Reopen skips bootstrap when centroids already exist.

### Search

1. Validate `k`, `nprobe`, and query dimensions.
2. Construct one query-bound search accessor.
3. Select `min(nprobe, live_centroids)` lists.
4. Pass the resulting plan back to that accessor for scanning.
5. Accumulate internal ids in the existing neighbor priority queue.
6. Apply point-status filtering and external-id translation as appropriate.
7. Return results and coarse/fine statistics.

No generic retry or version check occurs between selection and scanning.
Coherence follows from exclusive index mutation and the accessor implementation.

Note that co-location groups are never accessed during search.

### Batch Insert

1. Construct one maintenance accessor for the operation.
2. Stage the batch in one call, obtaining internal ids and the canonical
   vectors of the staged points. Duplicate or already-live external ids fail
   according to provider semantics.
3. Check that the ids are distinct and every vector is finite.
4. Route all inputs through the accessor's centroid index, and group the batch
   by route.
5. Read routed-list sizes and compute projected post-insert sizes.
6. Admit overflow parents subject to the live-cluster cap.
7. Reserve two stable child ids per parent.
8. Select each parent's nearest live lists, excluding every parent, as its
   neighbors. Read the centroids of all neighbors, then the members and
   canonical vectors of every parent and neighbor.
9. Fit each parent's two children with 2-means over its existing members and
   the staged points routed to it, seeded with two of those points.
10. Place each parent's points on the nearest of its neighbors and two
    children.
11. Leave each neighbor's points in place unless a child of a region containing
    that neighbor is closer than the neighbor's centroid, in which case move
    them to the nearest such child.
12. Build one `InsertionUpdate`. Append every staged point to its final list,
    which is its route if nothing splits. For each split, install the two
    children, retire the parent, and move every parent member and every
    neighbor member whose list changed.
13. Consume the accessor with `apply(update)`.

Steps 1 through 8 perform every accessor read; steps 9 through 12 compute on
owned, densely indexed data. Routes are computed before structural change;
regional placement deliberately revisits points whose routes become stale due
to the split.

Neighbor points are handled as in SPFresh: they only move toward the new
children, not between existing lists, which bounds each neighbor point's work to
its own centroid and the new children. A neighbor shared by several regions is
compared against the children of all of them, which yields one placement per
point without merging per-region results.

### Split Semantics

A split retires one parent and installs two children, for a net increase of one
live cluster. Parent ids are never reused. A parent's candidate centroids are its
selected live neighbors plus its two children; a neighbor's candidates are its
own centroid plus the children of every region containing it. Candidate points
are the parent members, incoming points, and selected-neighbor members.

The provider may rewrite complete lists, append delta segments, or mutate
in-memory lists. Those choices do not change the semantics of the update.

### Batch Delete And Dissolve

1. Construct one maintenance accessor and resolve every requested point's
   current assignment.
2. Stage deletion of each point id.
3. Group deletes by list and compute projected post-delete sizes.
4. Admit underfull victims without crossing `min_clusters`.
5. Select survivor candidates for every victim, excluding all victims admitted
   in this operation.
6. Read each victim's remaining members and canonical vectors.
7. Assign those members exactly among the saved survivor candidates.
8. Build one update that retires each victim, moves its remaining members to
   their survivors, and removes the deleted points. Its type is not defined
   yet; it follows the delta style of `InsertionUpdate`, with a point delta for
   removal such as `Delete { id, from }`.
9. Consume the accessor with `apply(update)`.

A dissolve fits no centroid and does not read survivor members. Removing a
centroid cannot make a point already assigned to another surviving centroid
prefer the removed centroid. Deletes never trigger a split in the same
operation. A survivor above the split threshold remains eligible for the next
insert-driven split.

## Concurrency And Consistency

The portable baseline is:

- many concurrent searches through shared index borrows;
- one mutation through an exclusive index borrow, forwarded to the maintenance
  accessor as `&mut P`; and
- no search/mutation overlap through the same index value.

Within an operation, providers may parallelize routing, list reads, decoding,
and scoring through `ExecutionContext::wrap_spawn` and workspace thread-pool
conventions.

Concrete implementations may offer more:

| Provider | Possible private unified-view mechanism |
|---|---|
| In memory | exclusive `&mut P` borrow, one lock guard, copy-on-write root, or epoch |
| Local disk | process lock, database transaction, journal, or private manifest swap |
| Blob storage | immutable objects plus conditional manifest update |

The generic traits neither require nor expose these mechanisms. A provider that
allows concurrent mutation and search must document whether readers see old
state, new state, block, or retry. That behavior is not portable across
strategies.

## Failure Semantics

Accessor and provider failures use the existing ranked error machinery.
Transient and critical low-level errors retain their normal meaning, but the
generic algorithm cannot infer the mutation's final state solely from an
`apply` error.

Each maintenance implementation must choose and document one failure model,
for example:

| Model | Behavior after an irreversible failure |
|---|---|
| Rollback | Restore the prior coherent state and remain usable |
| Complete-forward | Repair or finish the intended update before reopening |
| Poison | Reject later search and mutation until recovery or rebuild |
| Durable transaction | Rely on backend recovery to expose one coherent state |

At minimum, an implementation must fail closed rather than return results from
a knowingly incoherent state. Generic GraphIVF returns the error and does not
automatically retry because there is no portable conflict or rollback signal.

Cancellation has similarly provider-defined behavior. An in-memory accessor may
clean provisional state in `Drop`; an in-place implementation may hold exclusive
access and poison on interruption; a transactional backend may abort. The core
traits do not promise cancellation safety.

## Configuration

The initial dynamic algorithm validates at least:

```rust
pub struct DynamicIvfConfig {
    pub split_threshold: usize,
    pub merge_threshold: usize,
    pub min_clusters: usize,
    pub max_clusters: Option<usize>,
    pub reassign_neighbors: usize,
    pub two_means_iterations: usize,
}
```

Centroid-index-specific configuration belongs to the strategy. An exact
strategy needs no graph degree or beam; a graph strategy owns its graph build
and search parameters. Durable metadata belongs to concrete provider crates.

## Implementation Plan

### Phase 0: Compile-Only Contracts

- Add `ivf::dynamic` with the traits and data types in this RFC.
- Remove the fixed-partition prototype API and its tests.
- Compile, lint, format, and build rustdoc for the skeleton.

### Phase 1: Unified In-Memory Provider

- Build one provider containing canonical vectors, reverse assignments,
  in-memory lists, and an exact centroid index.
- Implement maintenance through the exclusive `&mut P` borrow with
  lock-free, in-place `apply`.
- Add an invariant checker and provider poisoning for injected partial failures.

### Phase 2: Dynamic Search

- Implement search orchestration with one query-bound accessor.
- Compare results to brute force over exactly the selected lists.
- Test that an accessor scans only lists it selected, under its own view.

### Phase 3: Exact-Centroid GraphIVF

- Port batch routing, split admission, per-parent child fitting, and
  SPFresh-style regional placement from the prototype.
- Start with exact centroid selection to isolate partition correctness.
- Check every successful update against the reference partition model.

### Phase 4: Graph Centroid Index

- Implement `CentroidIndex` with `graph::DiskANNIndex` plus an authoritative
  centroid catalog.
- Complete undersized graph results with exact selection.
- Define and test its partial-failure behavior under centroid churn.

### Phase 5: Delete And Dissolve

- Port delete grouping, victim admission, survivor selection, and reassignment.
- Add hysteresis, minimum-cluster, delete/reinsert, and overlapping-batch tests.

### Phase 6: Durable Providers

- Design local-disk consistency and recovery inside its concrete accessor.
- Add crash/reopen tests around each irreversible write.
- Design blob coordination separately rather than forcing disk and blob through
  one generic publication protocol.

## Testing Strategy

Algorithm tests use a small exact reference model containing a centroid map,
list map, and reverse assignment map. After each successful `apply`, tests
verify every logical invariant and compare provider state to the expected
update.

Required shared tests include:

- search equivalence to brute force over selected lists;
- split and dissolve invariant preservation;
- overlapping maintenance regions;
- duplicate insert and repeated delete handling;
- staged canonical-vector decoding;
- stable, never-reused list ids;
- centroid graph exact fallback; and
- provider failure injection with coherent-state or fail-closed verification.

Durability, cancellation, reader isolation, and recovery tests are
provider-specific because those guarantees are not part of the shared traits.

## Advantages

1. Far fewer associated types and generic constraints.
2. In-memory implementations can use normal exclusive mutation without
   pretending to implement MVCC.
3. The API resembles the graph module's operation-scoped accessor model.
4. Providers retain their native consistency and durability mechanisms.
5. The GraphIVF algorithm remains readable: it plans one logical update and
   hands it to one accessor.
6. The design does not prematurely standardize conflict detection, version
   formats, reclamation, or crash recovery.
7. Exact, graph, memory, disk, and blob implementations can evolve without
   exposing backend handles in generic types.

## Disadvantages

1. Coherence is documented rather than type-enforced.
2. Arbitrary centroid and list implementations cannot safely be mixed merely
   because their associated Rust types match.
3. Concurrent search/mutation behavior is not portable.
4. `apply` errors do not communicate whether an operation rolled back, partly
   applied, completed forward, or poisoned the provider.
5. Generic code cannot detect write conflicts or automatically replan.
6. Cancellation and crash semantics differ by provider.
7. Disk and blob implementations probably still need versioning or atomic
   publication internally; the complexity is hidden, not eliminated.
8. Correctness depends more heavily on provider contract tests and operational
   documentation.

## Alternatives

### Public Snapshot And Commit Protocol

The stronger alternative exposes a provider-defined generation, pinned read
snapshot, snapshot-bearing selection plan, staged mutation, commit outcome, and
compare-and-swap conflict. It makes reader coherence, mutation visibility, and
retry behavior portable.

That design is preferable if concurrent search/mutation, multiple writers, or
uniform crash-safe persistence are immediate requirements. It is not selected
for the first implementation because it imposes substantial machinery on every
provider before the in-memory algorithm exists.

### Separate Coarse And Fine Accessors

Independent accessors are smaller individually, but duplicate query setup and
have no common owner for consistency or prefetch state. One query-bound accessor
keeps coarse and fine operations together.

### Pass The Coarse Accessor Into Fine Scan

Passing `&mut CoarseAccessor` into a fine accessor shares state but couples two
objects, complicates async lifetimes, and makes selected lists difficult to
inspect. Keeping the selection as a small value and private state in one
accessor is simpler.

### Imperative List Mutation Methods

Adding `append`, `remove`, `move`, `create_list`, and `retire_list` exposes a
partial-write schedule to GraphIVF. A complete declarative update lets each
provider choose a suitable physical implementation.

### Optional Endpoints Instead Of Delta Variants

An earlier draft used a single `PartitionUpdate` with a free-form centroid
delta, a flat list of `PointMove { id, from: Option<L>, to: Option<L> }`, and
public fields. Optional endpoints made a move with neither endpoint
representable and left the meaning of each combination implicit. Public fields
let any code build invalid values, such as a retired list that was never
emptied or centroids of different dimensions. Unsorted moves made every
provider regroup them by list. Named delta variants, construction restricted to
the index, and destination-sorted point deltas remove those problems.

### Per-Operation Update Types

An intermediate design gave each operation its own update type: `Bootstrap`,
`Appends`, `Splits`, and `Reassignments`, with `SplitInsert` combining the last
three. Each type stored its points pre-grouped, and a provider implemented an
`Apply<U>` trait once per type it supported. The types made some structure
explicit, such as the two children replacing each parent, and let a provider
that never splits leave out `Apply<SplitInsert>`. However, the insert path alone
needed eight types and three `Apply` implementations, and a destination list's
incoming points were spread across several groups. Nothing consumed the extra
structure, and inserts required every `Apply` bound anyway. Primitive deltas
express every insert-path change in one type. If a graph centroid index or
co-location planning needs a split's lineage, `CentroidDelta::Install` can gain
a parent link.

### Publicly Constructible Updates

An intermediate design gave every update type a public, validating constructor
with its own error type, so any code could build updates. Because only the index
ever needs to build them, that validation duplicated guarantees the planner
already provides by construction, re-sorted data the planner had already
grouped, and roughly doubled the size of the update module. Restricting
construction to the index and validating untrusted inputs where they enter
removes that duplication. Provider crates exercise their `apply`
implementations through the index instead of hand-built updates.

### Use `SetElement` Independently

Completing a normal `SetElement` guard separately from list and centroid changes
can expose inconsistent point lifecycle. A maintenance accessor may reuse the
same internal machinery, but it coordinates point staging and final application
as one provider-specific operation.

### Require A Public Low-Level List Store

A universal low-level store tends toward point-at-a-time methods that are poor
for range I/O and object storage. Accessors preserve coarse operations. A lower
level trait can be introduced later once multiple concrete providers reveal a
useful common shape.

## Design Questions For Review

1. **Exclusive mutation**: keep `insert_batch` and `delete_batch` on `&mut self`
   for the first implementation, as recommended, or require provider-managed
   concurrent access immediately?
2. **Failure health**: should a small common health/poison trait be required, or
   should all post-error behavior remain provider-specific?
3. **Point staging**: should `StageElements`, and deletion staging once it is
   added, remain on the maintenance accessor, or should a future
   transactional provider trait supply them?
4. **Update granularity**: point deltas are sorted by destination list, so bulk
   backends can write each touched list once. Should updates also carry a
   per-list summary of added and removed counts for exact pre-sizing?
5. **Selected-list ownership**: is the documented same-accessor requirement
   sufficient, or should a future private token make misuse dynamically
   detectable?

## Benchmark Results

This is a design-only RFC. No performance claims are made. Benchmarks begin once
the exact-centroid in-memory implementation provides a correctness baseline.

## Future Work

- [ ] Provider-specific concurrent search/mutation implementations.
- [ ] Multiple non-overlapping writers.
- [ ] A common health/poison interface if implementations converge on one.
- [ ] Background list compaction.
- [ ] Residual PQ and codebook lifecycle management.
- [ ] Soft deletes followed by asynchronous dissolve.
- [ ] Filter-aware list selection and scan integration.
- [ ] Durable checkpoint and recovery tooling.
- [ ] A low-level list-store trait informed by concrete providers.
- [ ] Distributed or sharded list providers.

## References

1. [PR #1187: WIP IVF index interface](https://github.com/microsoft/DiskANN/pull/1187)
2. [RFC 01067: Refactor Search Accessor](01067-search-accessor.md)
3. [RFC 00983: Flat Search](00983-flat-search.md)