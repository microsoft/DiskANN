# Untangling the FillSet Knot


|                  |                      |
|------------------|----------------------|
| **Authors**      | Mark Hildebrand      |
| **Contributors** | Coffee/AI Overloards |
| **Created**      | 2026-03-30           |
| **Updated**      | 2026-03-30           |

## Summary

This PR and accompanying RFC untangles a large knot in the interaction of the pruning workingset, multi-insert, and generic parameters used in `Strategies`.
The goal is to allow providers to customize the shape of their workingsets while preserving our current behavior and providing a reasonably performant and usable default implementation in `graph::workingset::Map`.

## Motivation

### Background

#### What is a "working set"?

The prune algorithm is at the heart of index construction.
It involves computing pairwise distances among a small collection of candidates to select only the best.
Because of this pairwise behavior, it is useful to cache the vectors used in prune to avoid potentially expensive, repeated vector retrievals in the hot prune loop.
Enter the "working set".
In the current code, this "working set" is simply a hash map of data provider internal IDs to the `Accessor::Extended` associated type, and indeed is the primary reason why `Accessor::Extended` exists in the first place.

Since its inception for prune, the working set has also expanded to see use in inplace-delete for auxiliary distance computations.

#### What is "fill set"?

The `FillSet` trait is the current extension point for populating the working set hash map.
It simply takes an iterator of IDs and requests that the `Accessor` retrieve the data for each ID and put it in the hash map.

#### What is multi-insert and why is it complicated?

Multi-insert is a batch-mode alternative to the single insert API that processes input vectors in a batch.
Unlike single insert, this algorithm orchestrates adjacency list updates triggered by the batch to be disjoint, avoiding situations where multiple threads try to update one adjacency list concurrently.
While some providers can deal with this just fine, some cannot.
A DiskANN-side read-modify-write interface for `NeighborAccessorMut` would greatly help the situation, but multi-insert has other properties such as being "mostly deterministic" that continue to make it attractive for some applications.

One wrinkle with multi-insert is that the elements of the batch of vectors provided to multi-insert can themselves be candidates given to prune.
It's preferable to retrieve vector data from this batch in these situations whenever possible, especially when vector retrieval is relatively expensive.
This is currently facilitated with the `AsElement` trait, which partially works but has to fall back to full element retrieval in situations where the full-precision input vector cannot be used directly.

#### What's the knot?

The current trait hierarchy makes it impossible to sanely use types with lifetimes (e.g. a mythical `Slice<'_>`) type in multi-insert and instead limits multi-insert to **only** working on slice types via `VectorIdBoxSlice`.
In turn, this prevents us from eliminating monomorphizations on the full-precision data-type `T` in some backends that are not performance dependent on this specialization, bloating compile times.
Multi-vectors will further exacerbate this issue as these large types certainly should not be copied or cloned, but rather "viewed" as much as possible, which is not possible to do behind a normal Rust reference `&` when dealing with FFIs.

For reference, some trait definitions are shared below

```rust
pub trait AsElement<T>: Accessor {
    type Error: ToRanked + std::fmt::Debug + Send + Sync;
    fn as_element(
        &mut self,
        vector: T,
        id: Self::Id,
    ) -> impl Future<Output = Result<Self::Element<'_>, Self::Error>> + Send;
}
```

Several traits interact to cause this knot:

* The value type of the HashMap working set needs to be a concrete type with a lifetime not bound to the borrow of an `Accessor` that "reborrows" to `Accessor::ElementRef`.
  `Accessor::Extended` and its somewhat complicated interaction with `Accessor::Element` and `Accessor::ElementRef` is the current solution.

* For various reasons, `AsElement` needs to take an element with a lifetime (e.g., `AsElement<&[f32]>`) to avoid [HRTBs](https://doc.rust-lang.org/nomicon/hrtb.html) from [implying `'static`](https://users.rust-lang.org/t/bounding-generic-parameter-to-hrtb/82375/2).
  Really, this should be `AsExtended` - but this doesn't solve the fundamental issue.

* Even though `AsElement` takes a lifetime, our current strategy types (e.g., `SearchStrategy<DP, T>`) are configured to take `T: ?Sized`.
  While providing nice syntax like `SearchStrategy<DP, [T]>`, this creates kind of a nasty interaction with `AsElement` when trying to generalize `T` with non-slices.
  This syntax has caused other issues as well, requiring the spooky [`Lower`](https://github.com/microsoft/DiskANN/blob/65e20ecbcc1c81daba3cfa1995b9d6fa7a349448/diskann-utils/src/reborrow.rs#L137-L247) to make `InplaceDeleteStrategy` work and issues related to the work in `diskann-label-provider`.

Changing `SearchStrategy` (and transitively `InsertStrategy`) to be parameterized by the true type (e.g. `InsertStrategy<DP, &[f32]>` instead of `InsertStrategy<DP, [f32]>`) is necessary to open multi-insert to non-slices.
But doing this change results in something that `AsElement` cannot express because for annoying reasons, this decouples the lifetime of the parameter `T` to `AsElement` and the lifetime of the accessor, which runs into implied `'static` and other issues.
And `AsElement` is a fairly important (albeit limited) optimization necessary for multi-insert.

This is a simplified account of the issue.
Suffice it to say that all my attempts to undo this have led to one or another of these interactions stopping the show.

#### What else is wrong?

In addition to the above described knot, there are other issues related to the working set that I will dump here in no particular order.

* Currently, the bound on memory usage of the working set is dubious.
  In single insert, a single working set is retained for the primary prune and all backedges.
  This puts a theoretical upper-bound of the size of the working set to something manageable, but not controlled by the user.

  In multi-insert, on the other hand, many backedge prunes are processed by a single thread.
  Here the working set is aggressively cleared to avoid an unbounded increase in its size.
  It would be great if we could reuse some of the vectors fetched from previous iterations in this situation.

* Providers like inmem do not need a working set at all since vector retrieval is already cheap.
  Unfortunately, they have to pay for the machinery they do not use, and a copy + allocation is avoided by tying the lifetime of the `Extended` types of the associated accessors to the lifetime of the accessor itself.

* The design forces one allocation per vector.
  Instead, when the types involved can be densely packed into a matrix, this can be desirable for memory locality (and seems like a great potential optimization in general).

* In situations where the batch vectors cannot be used directly (think spherical quantization that doesn't support native full-precision + quantized distances), we're forced to fall back to `Accessor::get_element`, increasing the number of such calls perhaps unnecessarily.
  It would be nice if instead we could fetch the compressed representations of these vectors once and then simply reuse that for the duration of multi-insert.

* All of this is complicated by the need to support hybrid prunes for some PQ-based customers, where some vectors are full-precision and others are quantized.
  The motivation here was to have the best candidates in full-precision for higher fidelity while allowing later candidates to be quantized, balancing full-vector retrievals and quantized retrievals.

### Problem Statement

Solve the above issues in a way that is not overly complex.

### Goals

1. Abstract the Working Set entirely, enabling:

    a. Allow non-slice elements to be provided to multi-insert.
    b. Provide pass-through providers for the inmem case.
    c. Control working-set reuse, enabling a potential reduction in `get_vector` calls for multi-insert.
    d. Allow implementations to use a densely packed implementation if they want.
    e. Enable allocations to be owned by the `Accessor` or underlying `Provider` if desired.
    f. Provide zero-copy access to the batch provided in multi-insert.
    g. Enable a one-time get-vector pattern in multi-insert when the input vectors cannot be used directly.

2. Switch all strategy types to use `&[T]` instead of `[T]` to help facilitate further development.
   I'm not the only one who's been bitten by this.
   Let's do it.

3. Provide a reasonable default implementation of the working set (and friends) for users who do not wish to exercise full customization.

## Proposal

### Trait Interface

Here's the design I've managed to come up with.
The full implementations of these are in the PR and readers of this RFC are encouraged to refer to the associated docstrings.

```rust
//-------------------------------------//
// diskann/src/graph/workingset/mod.rs //
//-------------------------------------//

pub trait Fill<WorkingSet>: Accessor {
    type Error: Into<ANNError> + std::fmt::Debug + Send + Sync;

    type View<'a>: for<'b> View<Self::Id, ElementRef<'b> = Self::ElementRef<'b>> + Send + Sync
    where
        Self: 'a,
        WorkingSet: 'a;

    fn fill<'a, Itr>(
        &'a mut self,
        working_set: &'a mut WorkingSet,
        itr: Itr,
    ) -> impl SendFuture<Result<Self::View<'a>, Self::Error>>
    where
        Itr: ExactSizeIterator<Item = Self::Id> + Clone + Send + Sync,
        Self: 'a;
}

pub trait View<I> {
    type ElementRef<'a>;

    type Element<'a>: for<'b> Reborrow<'b, Target = Self::ElementRef<'b>>
    where
        Self: 'a;

    fn get(&self, id: I) -> Option<Self::Element<'_>>;
}

// Just needed for multi-insert.
pub trait AsWorkingSet<WorkingSet> {
    fn as_working_set(&self, capacity: usize) -> WorkingSet;
}

//---------------------------//
// diskann/src/graph/glue.rs //
//---------------------------//

pub trait PruneStrategy<Provider>: Send + Sync + 'static
where
    Provider: DataProvider,
{
    // New!
    type WorkingSet: Send + Sync;

    type PruneAccessor<'a>: Accessor<Id = Provider::InternalId>
        + BuildDistanceComputer<DistanceComputer = Self::DistanceComputer>
        + AsNeighborMut
        + workingset::Fill<Self::WorkingSet>; // <---- New!

    // New!
    fn create_working_set(&self, capacity: usize) -> Self::WorkingSet;

    // The rest is unchanged.
    ...
}

// All new!
pub trait MultiInsertStrategy<Provider, B>: Send + Sync
where
    Provider: DataProvider,
    B: Batch,
{
    type WorkingSet: Send + Sync + 'static;
    type Seed: workingset::AsWorkingSet<Self::WorkingSet> + Send + Sync + 'static;
    type FinishError: Into<ANNError> + std::fmt::Debug + Send + Sync;

    type InsertStrategy: for<'a> InsertStrategy<
            Provider,
            B::Element<'a>,
            PruneStrategy: PruneStrategy<Provider, WorkingSet = Self::WorkingSet>,
        >;

    fn insert_strategy(&self) -> Self::InsertStrategy;

    fn finish<Itr>(
        &self,
        provider: &Provider,
        context: &Provider::Context,
        batch: &Arc<B>,
        ids: Itr,
    ) -> impl std::future::Future<Output = Result<Self::Seed, Self::FinishError>> + Send
    where
        Itr: ExactSizeIterator<Item = Provider::InternalId> + Send;
}

pub trait Batch: Send + Sync + 'static {
    type Element<'a>: Copy;
    fn len(&self) -> usize;
    fn get(&self, i: usize) -> Self::Element<'_>;
}
```

The prune algorithms (namely `occlude_list`) are updated to work on a `View` instead of the current `HashMap` as the **concrete** form of the working set.
`View`s are constructed by the `Fill` trait, which is nearly identical to our current `FillSet` trait with two key differences:

1. The working set is an opaque generic instead of a `HashMap`.
   This opaqueness is important!
   Almost no requirements are placed on this type, providers are free to make it whatever they want.
   For pass-through providers or for providers who want to allocate memory for the working set inside the accessor, this can be a ZST.

   The same `WorkingSet` will be reused across multiple pruning rounds during backedge insertions and thus provides a convenient place to aggregate a single allocation if desired.

2. A `View` is returned and used to interact with the working set.
   Again, this enables vectors to be stored in either the opaque `WorkingSet` or in the `Accessor`.

`PruneStrategy` gains a new `WorkingSet` associated type with a `create_working_set` constructor which will be threaded to `Fill`.
The capacity supplied to this call will be the `max_occlusion_size` (or potentially a tighter bound if known ahead of time), allowing more precise pre-allocation.

And that's it for normal insert!

For multi inserts, we introduce a new `MultiInsertStrategy` parameterized by a simple `Batch`.
The difference here is that the origin of the final working set comes from the `finish` method.
If `get_element` calls need to be made for the batch, that can be done here.
A "Seed" is returned from `finish`, rather than a working set directly.
To understand this, realize that most of the actual work in multi-insert is delegated to worker tasks, decoupling the call to `finish` from when a working set is actually needed from the main thread where `finish` is called.
We could require `WorkingSet: Clone`, but this would require potentially extra work to clone the machinery inside the working set state.
Instead, creation is deferred via `AsWorkingSet`, minimizing unnecessary work at pretty much every level.

### Default Implementation

Much of the complexity of this PR comes from the default `Map` implementation in `diskann::graph::workingset::map`.
I won't go too much into the details here, but I will summarize the high-level properties of this struct and its associates:

* It is backed by a `HashMap` and can operate in three reasonable modes,

  - A "strict" mode where data is not reused across multiple "fill" calls (minimizing memory).
  - A "caching" mode where data is reused up to some amount (usually `max_occlusion_size`).
    This caching is a little smart and reuses as much of the existing entries as possible on a `fill` during eviction.
  - An "unbounded" YOLO mode where there is no limit on its capacity.

* Has a **blanket** `Fill` implementation for Accessors that is reasonably easy to sidestep entirely if more control is needed.

* Fallback `fill` helper methods.

* Fully supports the hybrid-PQ fill model.

* Allows zero-copy composition with the batch elements of multi-insert, conforming to the trait ideas outlined above.

## Trade-offs

### Keeping the syntax `SearchStrategy<DP, [T]>`

We could try to keep this syntax, but I think it needs to go for a number of reasons.
The `?Sized` bound makes it very tricky to support while still allowing types like `MatrixView<'_>` in a generic context (see the `Lower` trait).
We can see this tension in [782](https://github.com/microsoft/DiskANN/pull/782) where wrapped query types are forced to be `&T where T: ?Sized`.
While this works for slices - it doesn't interoperate well with generic `Copy` types with lifetimes.

### Making the batch IDs explicit in `Fill`

One advantage of the current working-set implementation is that the **algorithm** knows what's an intra-batch ID and what is a "regular" ID and could use that to "prefill" the working set.
This PR requires that working set implementations "rediscover" intra-batch candidates by consulting their overlay (if any), adding an extra branch to working set element retrieval.
Intra-batch candidates could be made explicit in `Fill`, but this pessimizes "single-insert-only" backends.
Should they error if intra-batch candidates are passed, turning what could be a compile-time error for incorrect usage into a runtime one?
Should yet another "fill-like" trait be introduced?
I don't think the complexity is worth it to save a few hash table lookups, especially since `View::get` is not called during the hot prune loop.
The overhead was not observable in the integration with database "DB".

### Getting Rid of Intra Batch Candidates

Much of the complexity comes from intra-batch candidates.
While this is a valid observation, the database "DB" integration relies on this for a number of reasons.
Completely moving this integration away from this flow is not feasible in the short term.

## Benchmark Results

We will look at two integration stories.
The first is the impact on the integration with database "DB" which uses perhaps the most features of multi-insert/prune in a production setting, including:

* The hybrid PQ prune.
* The caching provider.
* Quant only pruning that cannot use full-precision vectors for intra-batch pruning.
* Custom FillSet implementations.

The second is the impact on the in-mem providers, which are the simplest and can take advantage of the passthrough capabilities of the new architecture.

### Database DB

#### Setup

The integration with this database is one of the primary motivations behind this PR.
Currently, this integration monomorphizes the indexing algorithm for a set of full-precision types A, B, C, and D.
However, since most of the work is done in the quantized space, specializing on data type at the top level is not beneficial for performance and comes at the cost of significant compilation and space overhead.
Instead, this PR allows the full precision types to be represented with something similar to
```rust
/// The new type passed to `SearchStrategy` and `InsertStrategy`.
struct Slice<'a> {
    A(&'a [A]),
    B(&'a [B]),
    C(&'a [C]),
    D(&'a [D]),
}

/// The batch for multi-insert.
struct Batch {
    A(Matrix<A>),
    B(Matrix<B>),
    C(Matrix<C>),
    D(Matrix<D>),
}

impl Batch {
    fn row(&self, i: usize) -> Slice<'_> {
        match self {
            Self::A(m) => Slice::A(m.row(i)),
            /// etc.
        }
    }
}
```
The critical pieces this PR/RFC includes are

1. The relaxation of arguments to `multi_insert` from `VectorIdBoxSlice`.
2. The ability to thread the `Slice<'_>` type through the strategy hierarchy.
3. Zero-copy batch reuse in a way not compatible with the current `AsElement` trait.

The integration was performed in two steps.
First, the specialization on the data type was retained and the algorithm was switched to the new multi-insert approach.
Only the provided `Map` and friends were used as the actual `WorkingSet` implementation.
Second, the full removal of the generic parameter was performed, again using `Map` as the `WorkingSet` implementation.
A sketch of the `Map` integration is shown below:
```rust
/// An "owned" representation of `Slice`.
struct Boxed {
    A(Box<[A]>),
    B(Box<[B]>),
    C(Box<[C]>),
    D(Box<[D]>),
}

impl<'a> Reborrow<'a> for Boxed {
    type Target = Slice<'a>;
    fn reborrow(&'a self) -> Self::Target {
        match self {
            Self::A(b) => Slice::A(b),
            /// etc.
        }
    }
}

/// The projection tying a `Boxed` to a `Slice` in a way that cooperates with `Dataset`'s
/// implementation of `diskann::graph::glue::Batch`.
#[derive(Debug, Clone, Copy)]
pub struct ToSlice;

impl map::Projection for ToSlice {
    type Element<'a> = Slice<'a>;
    type ElementRef<'a> = Slice<'a>;
}

impl map::Project<ToSlice> for Boxed {
    fn project(&self) -> tagged::Slice<'_> {
        self.reborrow()
    }
}

type Id = /* omitted */;
type WorkingSet = map::Map<Id, tagged::Boxed, ToSlice>;
```

Finally, the integration with quant-only was able to take advantage of the `MultiInsertStrategy::finish` customization point to fetch the quantized representations of the full precision vectors just once.

#### Results

This change reduced the incremental recompilation time of the integration library by 73.5% (~400s to ~100s), reduced compilation RAM consumption by 67.6% (from ~7GB to ~2GB) and reduced binary size by about 42%.
On top of that, no performance or recall losses were observed in the test simulator.
Quant only builds saw a drop in `get_element` calls through a combination of using `MultiInsertStrategy::finish` to fetch a batch only once and through the working-set reuse feature.

### Inmem Providers

All benchmarks were run on a Linux work station, pinned to NUMA node 1 with 32 available threads (16 physical cores).
Build and search were run with all available threads.
Recall is measured as recall@10.

#### BIGANN (10M vectors, uint8, L2, full-precision)

Build time: 80.9s to 75.0s (-7.3%)

| search_l | Recall (main) | Recall (PR) | Δ | QPS (main) | QPS (PR) | Change |
|----------|---------------|-------------|---|------------|----------|--------|
| 10 | 0.6994 | 0.6980 | -0.0013 | 437,235 | 417,275 | -4.6% |
| 20 | 0.8113 | 0.8113 | +0.0000 | 316,146 | 300,138 | -5.1% |
| 30 | 0.8657 | 0.8661 | +0.0004 | 236,128 | 227,593 | -3.6% |
| 40 | 0.8985 | 0.8992 | +0.0007 | 193,364 | 187,196 | -3.2% |
| 50 | 0.9195 | 0.9194 | -0.0000 | 164,061 | 157,156 | -4.2% |
| 60 | 0.9336 | 0.9338 | +0.0002 | 142,110 | 137,421 | -3.3% |
| 70 | 0.9448 | 0.9448 | +0.0000 | 124,684 | 120,208 | -3.6% |
| 80 | 0.9526 | 0.9524 | -0.0003 | 111,065 | 107,899 | -2.9% |
| 90 | 0.9595 | 0.9589 | -0.0006 | 101,246 | 97,557 | -3.6% |
| 100 | 0.9646 | 0.9639 | -0.0007 | 90,864 | 87,595 | -3.6% |

#### PQ (OpenAI 1M vectors, f32, L2, product quantization with 128 chunks)

Quant-Only Build: 105.6s to 106.4s (+0.7%)

The main goal here is to verify that recall is still poor with quant-only builds at 128 chunks.

| search_l | Recall (main) | Recall (PR) | Δ | QPS (main) | QPS (PR) | Change |
|----------|---------------|-------------|---|------------|----------|--------|
| 10 | 0.1774 | 0.1798 | +0.0024 | 38,916 | 44,654 | +14.7% |
| 20 | 0.2876 | 0.2924 | +0.0047 | 35,466 | 40,261 | +13.5% |
| 30 | 0.3587 | 0.3626 | +0.0038 | 33,016 | 37,236 | +12.8% |
| 40 | 0.4081 | 0.4113 | +0.0031 | 31,022 | 34,733 | +12.0% |
| 50 | 0.4445 | 0.4486 | +0.0042 | 29,306 | 32,795 | +11.9% |
| 60 | 0.4747 | 0.4779 | +0.0032 | 27,907 | 31,018 | +11.1% |
| 70 | 0.5004 | 0.5035 | +0.0032 | 26,583 | 29,501 | +11.0% |
| 80 | 0.5217 | 0.5246 | +0.0029 | 25,446 | 28,116 | +10.5% |
| 90 | 0.5399 | 0.5428 | +0.0029 | 24,377 | 26,876 | +10.2% |
| 100 | 0.5552 | 0.5594 | +0.0042 | 23,420 | 25,774 | +10.1% |

**Hybrid Build (48 FP vectors per-prune)**: Build: 125.7s to 123.3s (-1.9%)

| search_l | Recall (main) | Recall (PR) | Δ | QPS (main) | QPS (PR) | Change |
|----------|---------------|-------------|---|------------|----------|--------|
| 10 | 0.4486 | 0.4481 | -0.0004 | 31,914 | 36,151 | +13.3% |
| 20 | 0.6265 | 0.6249 | -0.0015 | 27,807 | 31,267 | +12.4% |
| 30 | 0.7109 | 0.7102 | -0.0007 | 24,937 | 27,838 | +11.6% |
| 40 | 0.7607 | 0.7609 | +0.0002 | 22,704 | 25,249 | +11.2% |
| 50 | 0.7952 | 0.7946 | -0.0005 | 20,923 | 23,162 | +10.7% |
| 60 | 0.8202 | 0.8200 | -0.0002 | 19,372 | 21,421 | +10.6% |
| 70 | 0.8392 | 0.8387 | -0.0005 | 18,106 | 19,912 | +10.0% |
| 80 | 0.8543 | 0.8536 | -0.0008 | 16,970 | 18,656 | +9.9% |
| 90 | 0.8669 | 0.8659 | -0.0010 | 15,996 | 17,545 | +9.7% |
| 100 | 0.8770 | 0.8764 | -0.0006 | 15,128 | 16,570 | +9.5% |

#### Spherical (OpenAI 1M vectors, f32, L2, 1-bit spherical quantization)

Build time: 14.5s to 13.9s (-3.9%)

**Full-precision Query Layout**

| search_l | Recall (main) | Recall (PR) | Δ | QPS (main) | QPS (PR) | Change |
|----------|---------------|-------------|---|------------|----------|---|
| 10 | 0.5301 | 0.5301 | +0.0000 | 136,697 | 137,860 | +0.9% |
| 20 | 0.7123 | 0.7148 | +0.0025 | 96,581 | 95,611 | -1.0% |
| 30 | 0.7843 | 0.7851 | +0.0007 | 74,860 | 74,673 | -0.2% |
| 40 | 0.8229 | 0.8244 | +0.0015 | 60,805 | 61,444 | +1.1% |
| 50 | 0.8494 | 0.8505 | +0.0011 | 52,015 | 51,918 | -0.2% |
| 60 | 0.8678 | 0.8688 | +0.0010 | 45,198 | 45,487 | +0.6% |
| 70 | 0.8819 | 0.8828 | +0.0010 | 40,229 | 40,531 | +0.8% |
| 80 | 0.8935 | 0.8941 | +0.0006 | 36,168 | 36,350 | +0.5% |
| 90 | 0.9026 | 0.9033 | +0.0007 | 32,883 | 33,050 | +0.5% |
| 100 | 0.9103 | 0.9107 | +0.0004 | 30,256 | 30,322 | +0.2% |

#### Summary

- **Recall**: No major changes.

- **Build time**: No major changes.
  BIGANN may see a slight improvement.
  I included it because the cheap distance computations make other inefficiencies stand out.

- **QPS**: BIGANN shows a ~3-5% dip and spherical is neutral (within ±1%), both within expected run-to-run variance.
  PQ shows a +10-15% improvement that persists across reruns.
  However, the number of hops and comparisons is the same, so I think this is reflective of secondary effects (allocator state maybe?) and cannot be attributed to this PR.
