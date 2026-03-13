// =============================================================================
// Post-Processing Redesign: Sketch & Rationale
// =============================================================================
//
// Context
// -------
// Two competing PRs attempted to refactor how SearchStrategy interacts with
// post-processing.  Both had structural problems:
//
//   Exhibit-A kept `type PostProcessor` on SearchStrategy and layered a new
//   `PostProcess<Processor, Provider, T, O>` trait on top.  This created two
//   parallel "what's the post-processor?" answers on the same type that could
//   silently diverge.  The GAT associated type became dead weight that every
//   implementor still had to fill in.
//
//   Exhibit-B removed `PostProcessor` from SearchStrategy (good), but replaced
//   it with a `DelegatePostProcess` marker whose blanket impl covered *all*
//   processor types `P` at once:
//
//       impl<S, Provider, T, P, O> PostProcess<Provider, T, P, O> for S
//       where S: SearchStrategy<…> + DelegatePostProcess,
//             P: for<'a> SearchPostProcess<S::SearchAccessor<'a>, T, O> + …
//
//   This makes it impossible to override `PostProcess` for a specific `P`
//   without opting out of the blanket entirely (removing DelegatePostProcess),
//   which then forces manual impls for every processor type — an all-or-nothing
//   cliff.  It also provided no `KnnWith`-style mechanism for callers to supply
//   a custom processor at the search call-site.
//
// Proposed Design
// ---------------
// Flip the blanket.  Instead of "strategy S gets PostProcess for all P",
// make it "the DefaultPostProcess ZST gets support for all strategies S
// that opt in via HasDefaultProcessor".
//
// The blanket is narrow (covers exactly one P = DefaultPostProcess), so custom
// PostProcess<…, RagSearchParams, …> impls are coherence-safe.  Strategies
// that don't need a default can skip HasDefaultProcessor and still be used via
// KnnWith<PP> with an explicit processor.
//
// =============================================================================
//
// How to read this file
// ---------------------
// This is pseudocode — it won't compile.  Signatures use real Rust syntax where
// possible but elide lifetimes, bounds, and async machinery for clarity.
// Comments marked "NOTE" call out places where the real implementation will
// need careful attention to HRTB / GAT interactions.
//
// =============================================================================

// ---------------------------------------------------------------------------
// 1. SearchStrategy — clean, no post-processing knowledge
// ---------------------------------------------------------------------------
//
// This is the same as today minus `type PostProcessor` and `fn post_processor`.

pub trait SearchStrategy<Provider, T, O = <Provider as DataProvider>::InternalId>:
    Send + Sync
where
    Provider: DataProvider,
    T: ?Sized,
    O: Send,
{
    type QueryComputer: /* PreprocessedDistanceFunction bounds */ Send + Sync + 'static;
    type SearchAccessorError: StandardError;

    // NOTE: This GAT is the source of most HRTB complexity downstream.
    type SearchAccessor<'a>: ExpandBeam<T, QueryComputer = Self::QueryComputer, Id = Provider::InternalId>
        + SearchExt;

    fn search_accessor<'a>(
        &'a self,
        provider: &'a Provider,
        context: &'a Provider::Context,
    ) -> Result<Self::SearchAccessor<'a>, Self::SearchAccessorError>;
}

// ---------------------------------------------------------------------------
// 2. SearchPostProcess — unchanged from today
// ---------------------------------------------------------------------------
//
// Low-level trait, parameterized by the *accessor* (not the strategy).
// CopyIds, Rerank, Pipeline<Head, Tail>, RemoveDeletedIdsAndCopy, etc. all
// implement this directly.  No changes needed here.

pub trait SearchPostProcess<A, T, O = <A as HasId>::Id>
where
    A: BuildQueryComputer<T>,
    T: ?Sized,
{
    type Error: StandardError;

    fn post_process<I, B>(
        &self,
        accessor: &mut A,
        query: &T,
        computer: &<A as BuildQueryComputer<T>>::QueryComputer,
        candidates: I,
        output: &mut B,
    ) -> impl Future<Output = Result<usize, Self::Error>> + Send
    where
        I: Iterator<Item = Neighbor<A::Id>> + Send,
        B: SearchOutputBuffer<O> + Send + ?Sized;
}

// Pipeline, CopyIds, FilterStartPoints, SearchPostProcessStep — all unchanged.

// ---------------------------------------------------------------------------
// 3. PostProcess — strategy-level bridge, parameterized by processor P
// ---------------------------------------------------------------------------
//
// This trait connects a strategy to a specific processor type.  It is the
// surface that the search infrastructure (Knn, KnnWith, RecordedKnn, etc.)
// bounds on.

pub trait PostProcess<Provider, T, P, O = <Provider as DataProvider>::InternalId>:
    SearchStrategy<Provider, T, O>
where
    Provider: DataProvider,
    T: ?Sized,
    O: Send,
    P: Send + Sync,
{
    fn post_process_with<'a, I, B>(
        &self,
        processor: &P,
        accessor: &mut Self::SearchAccessor<'a>,
        query: &T,
        computer: &Self::QueryComputer,
        candidates: I,
        output: &mut B,
    ) -> impl Future<Output = ANNResult<usize>> + Send
    where
        I: Iterator<Item = Neighbor<Provider::InternalId>> + Send,
        B: SearchOutputBuffer<O> + Send + ?Sized;
}

// ---------------------------------------------------------------------------
// 4. HasDefaultProcessor — opt-in "I have a default post-processor"
// ---------------------------------------------------------------------------
//
// Strategies that want to work with Knn (no explicit processor) implement this.
// It replaces the old `type PostProcessor` on SearchStrategy.
//
// NOTE: The `for<'a> SearchPostProcess<Self::SearchAccessor<'a>, T, O>` HRTB
// bound is the same one that lived on SearchStrategy::PostProcessor today.
// It's not new complexity — it just moved here.

pub trait HasDefaultProcessor<Provider, T, O = <Provider as DataProvider>::InternalId>:
    SearchStrategy<Provider, T, O>
where
    Provider: DataProvider,
    T: ?Sized,
    O: Send,
{
    type Processor: for<'a> SearchPostProcess<Self::SearchAccessor<'a>, T, O>
        + Send
        + Sync;

    fn create_processor(&self) -> Self::Processor;
}

// Convenience macro (same idea as exhibit-B's has_default_processor!).
macro_rules! has_default_processor {
    ($Processor:ty) => {
        type Processor = $Processor;
        fn create_processor(&self) -> Self::Processor {
            Default::default()
        }
    };
}

// ---------------------------------------------------------------------------
// 5. DefaultPostProcess ZST + THE blanket impl
// ---------------------------------------------------------------------------
//
// KEY DESIGN POINT: The blanket covers exactly P = DefaultPostProcess.
// Custom processor types (RagSearchParams, etc.) are free to have their own
// `impl PostProcess<…, RagSearchParams, …> for MyStrategy` without any
// coherence conflict.

#[derive(Debug, Default, Clone, Copy)]
pub struct DefaultPostProcess;

impl<S, Provider, T, O> PostProcess<Provider, T, DefaultPostProcess, O> for S
where
    S: HasDefaultProcessor<Provider, T, O>,
    Provider: DataProvider,
    T: ?Sized + Sync,
    O: Send,
{
    async fn post_process_with<'a, I, B>(
        &self,
        _processor: &DefaultPostProcess,
        accessor: &mut Self::SearchAccessor<'a>,
        query: &T,
        computer: &Self::QueryComputer,
        candidates: I,
        output: &mut B,
    ) -> ANNResult<usize>
    where
        I: Iterator<Item = Neighbor<Provider::InternalId>> + Send,
        B: SearchOutputBuffer<O> + Send + ?Sized,
    {
        self.create_processor()
            .post_process(accessor, query, computer, candidates, output)
            .await
            .into_ann_result()
    }
}

// ---------------------------------------------------------------------------
// 6. Search API split: Knn vs KnnWith<PP>
// ---------------------------------------------------------------------------
//
// Knn uses the default processor.  KnnWith<PP> allows an explicit override.
// Both delegate to a shared `search_core` that is parameterized over PP.

impl Knn {
    /// Shared core — the only axis of variation is the processor.
    async fn search_core<DP, S, T, O, OB, SR, PP>(
        &self,
        index: &DiskANNIndex<DP>,
        strategy: &S,
        /* … */
        post_processor: &PP,
    ) -> ANNResult<SearchStats>
    where
        S: PostProcess<PP, DP, T, O>,
        PP: Send + Sync,
        /* … */
    {
        let mut accessor = strategy.search_accessor(/* … */)?;
        let computer = accessor.build_query_computer(query)?;
        /* … search_internal … */
        let count = strategy
            .post_process_with(post_processor, &mut accessor, query, &computer, candidates, output)
            .await?;
        Ok(stats.finish(count as u32))
    }
}

// Knn: uses DefaultPostProcess
impl<DP, S, T, O, OB> Search<DP, S, T, O, OB> for Knn
where
    S: PostProcess<DP, T, DefaultPostProcess, O>,
    // equivalently: S: HasDefaultProcessor<DP, T, O>
{
    fn search(self, /* … */) -> impl SendFuture<ANNResult<SearchStats>> {
        async move {
            self.search_core(/* … */, &DefaultPostProcess).await
        }
    }
}

// KnnWith<PP>: uses caller-supplied processor
pub struct KnnWith<PP> {
    inner: Knn,
    post_processor: PP,
}

impl<DP, S, T, O, OB, PP> Search<DP, S, T, O, OB> for KnnWith<PP>
where
    S: PostProcess<DP, T, PP, O>,
    PP: Send + Sync,
{
    fn search(self, /* … */) -> impl SendFuture<ANNResult<SearchStats>> {
        async move {
            self.inner
                .search_core(/* … */, &self.post_processor)
                .await
        }
    }
}

// ---------------------------------------------------------------------------
// 7. Example: implementing a strategy
// ---------------------------------------------------------------------------

struct MyStrategy { /* … */ }

impl SearchStrategy<MyProvider, [f32]> for MyStrategy {
    type QueryComputer = MyComputer;
    type SearchAccessorError = ANNError;
    type SearchAccessor<'a> = MyAccessor<'a>;

    fn search_accessor<'a>(/* … */) -> Result<MyAccessor<'a>, ANNError> { /* … */ }
    // No PostProcessor, no post_processor() — clean.
}

// Opt in to the default: "my default post-processor is CopyIds"
impl HasDefaultProcessor<MyProvider, [f32]> for MyStrategy {
    has_default_processor!(CopyIds);
}
// That's it — Knn now works with MyStrategy.

// Opt in to RAG reranking too (no coherence conflict!):
impl PostProcess<MyProvider, [f32], RagSearchParams, (u32, AssocData)> for MyStrategy {
    async fn post_process_with(
        &self,
        processor: &RagSearchParams,
        accessor: &mut MyAccessor<'_>,
        /* … */
    ) -> ANNResult<usize> {
        // Custom RAG logic here
    }
}
// Now `KnnWith::new(knn, rag_params)` also works with MyStrategy.

// ---------------------------------------------------------------------------
// 8. Decorator strategies (BetaFilter)
// ---------------------------------------------------------------------------
//
// BetaFilter wraps an inner strategy and delegates.  The PostProcess<…, P, …>
// impl is generic over P, which is coherence-safe because it's on a concrete
// wrapper type (not a blanket over Self).

impl<Provider, Strategy, T, I, O, P> PostProcess<Provider, T, P, O>
    for BetaFilter<Strategy, I>
where
    Strategy: PostProcess<Provider, T, P, O>,
    P: Send + Sync,
    /* … other bounds … */
{
    async fn post_process_with(
        &self,
        processor: &P,
        accessor: &mut Self::SearchAccessor<'_>,
        /* … */
    ) -> ANNResult<usize> {
        // Unwrap the layered accessor, delegate to inner strategy
        self.strategy
            .post_process_with(processor, &mut accessor.inner, /* … */)
            .await
    }
}

impl<Provider, Strategy, T, I, O> HasDefaultProcessor<Provider, T, O>
    for BetaFilter<Strategy, I>
where
    Strategy: HasDefaultProcessor<Provider, T, O>,
    /* … */
{
    type Processor = Strategy::Processor;
    fn create_processor(&self) -> Self::Processor {
        self.strategy.create_processor()
    }
}

// ---------------------------------------------------------------------------
// 9. InplaceDeleteStrategy
// ---------------------------------------------------------------------------
//
// The delete-search phase needs exactly one processor type.  The associated
// type pins it, and the SearchStrategy bound requires PostProcess for that
// specific type.
//
// NOTE: The double `for<'a>` bound is verbose but unavoidable given the GAT.

pub trait InplaceDeleteStrategy<Provider>: Send + Sync + 'static
where
    Provider: DataProvider,
{
    type DeleteElement<'a>: Send + Sync + ?Sized;
    type DeleteElementGuard: /* … AsyncLower … */ + 'static;
    type DeleteElementError: StandardError;
    type PruneStrategy: PruneStrategy<Provider>;

    /// The processor used during the delete-search phase.
    type SearchPostProcessor: Send + Sync;

    /// The search strategy, which must support PostProcess with the above processor.
    type SearchStrategy: for<'a> SearchStrategy<Provider, Self::DeleteElement<'a>>
        + for<'a> PostProcess<
            Provider,
            Self::DeleteElement<'a>,
            Self::SearchPostProcessor,
        >;

    fn prune_strategy(&self) -> Self::PruneStrategy;
    fn search_strategy(&self) -> Self::SearchStrategy;
    fn search_post_processor(&self) -> Self::SearchPostProcessor;

    fn get_delete_element<'a>(/* … */) -> impl Future<Output = Result<Self::DeleteElementGuard, Self::DeleteElementError>> + Send;
}

// ---------------------------------------------------------------------------
// 10. Known pain points for the real implementation
// ---------------------------------------------------------------------------
//
// A. HRTB on HasDefaultProcessor::Processor
//    The bound `for<'a> SearchPostProcess<Self::SearchAccessor<'a>, T, O>`
//    is the same one that lived on SearchStrategy::PostProcessor before.
//    It's not new — it just moved.  The has_default_processor! macro
//    should absorb this.
//
// B. BetaFilter's generic P delegation
//    `impl<P> PostProcess<…, P, …> for BetaFilter<S> where S: PostProcess<…, P, …>`
//    is coherence-safe (concrete wrapper, not a blanket over Self), but verify
//    that rustc is happy with the HRTB interaction when SearchAccessor<'a> is
//    a layered type (BetaAccessor wrapping the inner accessor).
//
// C. Disk provider (DiskSearchStrategy)
//    Today it has PostProcessor = RerankAndFilter.  Under the new design:
//      - impl HasDefaultProcessor → Processor = RerankAndFilter
//      - impl PostProcess<…, RagSearchParams, …> → custom RAG reranking
//    These are independent impls with no coherence conflict.
//
// D. Caching provider (CachingAccessor)
//    Uses Pipeline<Unwrap, Inner> today.  Same pattern: HasDefaultProcessor
//    with Processor = Pipeline<Unwrap, Inner>.  The Pipeline type is just
//    another SearchPostProcess impl.
//
// E. The .send() / IntoANNResult bridge
//    The blanket impl calls `create_processor().post_process(…).await`.
//    The SearchPostProcess::Error needs to be convertible to ANNError.  Today
//    this is handled via IntoANNResult / .send().  Same pattern applies.
