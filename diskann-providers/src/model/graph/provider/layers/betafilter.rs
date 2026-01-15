/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Beta filtering is a variation of filtered search where distances between a query and
//! vectors satisfying a predicate (as determined by a [`QueryLabelProvider`])
//! are multiplied by a factor `beta` that improves similarity.
//!
//! The idea is that this encourages exploring neighbors satisfying the predicate.
//!
//! Beta filtering is implemented by using the type [`BetaFilter`] as a [`SearchStrategy`]
//! for some data provider. The [`BetaFilter`] composes with an "inner" search strategy
//! to supply the predicate checking logic.

use std::{future::Future, sync::Arc};

use diskann::{
    ANNResult,
    error::StandardError,
    graph::{
        SearchOutputBuffer,
        glue::{self, ExpandBeam, SearchExt, SearchPostProcessStep, SearchStrategy},
        index::QueryLabelProvider,
    },
    neighbor::Neighbor,
    provider::{Accessor, AsNeighbor, BuildQueryComputer, DataProvider, DelegateNeighbor, HasId},
    utils::VectorId,
};
use diskann_utils::Reborrow;
use diskann_vector::PreprocessedDistanceFunction;
use futures_util::FutureExt;

/// A [`SearchStrategy`] type that composes the inner distance computer with beta filtering.
///
/// Beta filtering works by checking whether a given vector ID satisfies the predicate in
/// the contained `QueryLabelProvider` and if so, adjusts the computed distance be `beta`
/// to encourage exploration of vectors satisfying the predicate.
///
/// See Also:
/// * [`BetaAccessor`]: For the [`Accessor`] derived from this strategy.
/// * [`BetaComputer`]: For [`PreprocessedDistanceFunction`] support for beta filtering.
#[derive(Debug, Clone)]
pub struct BetaFilter<Strategy, I> {
    /// The inner strategy to use.
    strategy: Strategy,
    labels: Arc<dyn QueryLabelProvider<I>>,
    beta: f32,
}

impl<Strategy, I> BetaFilter<Strategy, I> {
    /// Construct a new `BetaFilter` strategy composed with `strategy`.
    pub fn new(strategy: Strategy, labels: Arc<dyn QueryLabelProvider<I>>, beta: f32) -> Self {
        Self {
            strategy,
            labels,
            beta,
        }
    }

    pub fn get_labels(&self) -> Arc<dyn QueryLabelProvider<I>> {
        self.labels.clone()
    }
}

/// A post processor step the delegates [`BetaAccessor`] to the inner accessor.
#[derive(Debug, Default, Clone, Copy)]
pub struct Unwrap;

/// Delegate post-processing to the inner strategy's post-processing routine.
impl<A, T, O> SearchPostProcessStep<BetaAccessor<A>, T, O> for Unwrap
where
    T: ?Sized,
    A: BuildQueryComputer<T>,
{
    type Error<NextError>
        = NextError
    where
        NextError: StandardError;

    type NextAccessor = A;

    fn post_process_step<I, B, Next>(
        &self,
        next: &Next,
        accessor: &mut BetaAccessor<A>,
        query: &T,
        computer: &BetaComputer<A::QueryComputer, A::Id>,
        candidates: I,
        output: &mut B,
    ) -> impl Future<Output = Result<usize, Next::Error>> + Send
    where
        I: Iterator<Item = Neighbor<A::Id>> + Send,
        B: SearchOutputBuffer<O> + Send + ?Sized,
        Next: glue::SearchPostProcess<Self::NextAccessor, T, O>,
    {
        next.post_process(
            &mut accessor.inner,
            query,
            computer.inner(),
            candidates,
            output,
        )
    }
}

/// Allow `BetaFilter` to be used as a [`SearchStrategy`] for `Provider` when its enclosing
/// strategy is a valid search strategy.
///
/// This works by modifying the `Element` obtained using the inner accessor into a tuple
/// that propagates the vector ID.
///
/// The [`BetaComputer`] then uses this ID to consult the filter predicate and adjust the
/// distance accordingly.
impl<Provider, Strategy, T, I> SearchStrategy<Provider, T> for BetaFilter<Strategy, I>
where
    T: ?Sized,
    I: VectorId,
    Provider: DataProvider<InternalId = I>,
    Strategy: SearchStrategy<Provider, T>,
{
    /// An accessor that returns the ID in addition to the element yielded by the inner
    /// accessor.
    type SearchAccessor<'a> = BetaAccessor<Strategy::SearchAccessor<'a>>;

    /// A [`PreprocessedDistanceFunction`] that combines applies the beta filtering factor
    /// if the vector ID portion of `Element` satisfies the filter predicate.
    type QueryComputer = BetaComputer<Strategy::QueryComputer, I>;

    type SearchAccessorError = Strategy::SearchAccessorError;

    /// Compose the [`BetaAccessor`] with the inner search strategy's [`Accessor`].
    fn search_accessor<'a>(
        &'a self,
        provider: &'a Provider,
        context: &'a Provider::Context,
    ) -> Result<Self::SearchAccessor<'a>, Self::SearchAccessorError> {
        Ok(BetaAccessor {
            inner: self.strategy.search_accessor(provider, context)?,
            labels: self.labels.clone(),
            beta: self.beta,
        })
    }

    /// Forward the post-processing error from the inner strategy.
    type PostProcessor = glue::Pipeline<Unwrap, Strategy::PostProcessor>;

    /// Delegate post-processing to the inner strategy's post-processing routine.
    fn post_processor(&self) -> Self::PostProcessor {
        glue::Pipeline::new(Unwrap, self.strategy.post_processor())
    }
}

/////////////
// Helpers //
/////////////

/// The `Element` and `ElementRef` types used by the [`BetaAccessor`].
#[derive(Debug, Clone, PartialEq)]
pub struct Pair<I, E> {
    id: I,
    element: E,
}

impl<I, E> Pair<I, E> {
    fn new(id: I, element: E) -> Self {
        Self { id, element }
    }
}

/// `Reborrow` is implemented in terms of a full `Reborrow` of `E` while leaving the id
/// untouched.
impl<'a, I, E> Reborrow<'a> for Pair<I, E>
where
    E: Reborrow<'a>,
    I: Copy,
{
    type Target = Pair<I, E::Target>;
    fn reborrow(&'a self) -> Self::Target {
        Pair {
            id: self.id,
            element: self.element.reborrow(),
        }
    }
}

/// The extended version of `Pair`.
#[derive(Debug, Clone)]
pub struct ExtendedPair<I, E> {
    id: I,
    element: E,
}

impl<I, E, T> From<Pair<I, E>> for ExtendedPair<I, T>
where
    E: Into<T>,
{
    fn from(pair: Pair<I, E>) -> Self {
        Self {
            id: pair.id,
            element: pair.element.into(),
        }
    }
}

impl<'a, I, E> Reborrow<'a> for ExtendedPair<I, E>
where
    E: Reborrow<'a>,
    I: Copy,
{
    type Target = Pair<I, E::Target>;
    fn reborrow(&'a self) -> Self::Target {
        Pair {
            id: self.id,
            element: self.element.reborrow(),
        }
    }
}

/// An [`Accessor`] that composes with an `Inner` accessor to provide beta-filtering.
pub struct BetaAccessor<Inner>
where
    Inner: HasId,
{
    inner: Inner,
    labels: Arc<dyn QueryLabelProvider<Inner::Id>>,
    beta: f32,
}

impl<Inner> SearchExt for BetaAccessor<Inner>
where
    Inner: SearchExt,
{
    fn starting_points(&self) -> impl Future<Output = ANNResult<Vec<Inner::Id>>> + Send {
        self.inner.starting_points()
    }
}

impl<'a, Inner> DelegateNeighbor<'a> for BetaAccessor<Inner>
where
    Inner: DelegateNeighbor<'a>,
{
    type Delegate = Inner::Delegate;
    fn delegate_neighbor(&'a mut self) -> Self::Delegate {
        self.inner.delegate_neighbor()
    }
}

impl<Inner> HasId for BetaAccessor<Inner>
where
    Inner: HasId,
{
    type Id = Inner::Id;
}

impl<Inner> Accessor for BetaAccessor<Inner>
where
    Inner: Accessor,
{
    type Extended = ExtendedPair<Self::Id, Inner::Extended>;
    /// Modify `Element` to retain the vector ID.
    type Element<'a>
        = Pair<Self::Id, Inner::Element<'a>>
    where
        Self: 'a;
    type ElementRef<'a> = Pair<Self::Id, Inner::ElementRef<'a>>;

    /// Use the same error type as `Inner`.
    type GetError = Inner::GetError;

    /// Invoke `get_element` on the inner accessor and return a tuple consisting of the
    /// retrieved element and `id`.
    #[inline(always)]
    fn get_element(
        &mut self,
        id: Self::Id,
    ) -> impl Future<Output = Result<Self::Element<'_>, Self::GetError>> + Send {
        // The first `map` applies to `Future`.
        // The second `map` applies to the `Result`.
        self.inner
            .get_element(id)
            .map(move |result| result.map(move |v| Pair::new(id, v)))
    }

    /// Method `on_elements_unordered` is implemented by invoking
    /// `inner.on_elements_unordered` with a decorated version of `f`.
    async fn on_elements_unordered<Itr, F>(
        &mut self,
        itr: Itr,
        mut f: F,
    ) -> Result<(), Self::GetError>
    where
        Self: Sync,
        Itr: Iterator<Item = Self::Id> + Send,
        F: Send + for<'a> FnMut(Self::ElementRef<'a>, Self::Id),
    {
        self.inner
            .on_elements_unordered(
                itr,
                #[inline]
                move |element, id| f(Pair::new(id, element), id),
            )
            .await
    }
}

impl<Inner, T> BuildQueryComputer<T> for BetaAccessor<Inner>
where
    Inner: BuildQueryComputer<T>,
    T: ?Sized,
{
    /// Use a [`BetaComputer`] to apply filtering.
    type QueryComputer = BetaComputer<Inner::QueryComputer, Self::Id>;
    /// Use the same error as `Inner`.
    type QueryComputerError = Inner::QueryComputerError;

    fn build_query_computer(
        &self,
        from: &T,
    ) -> Result<Self::QueryComputer, Self::QueryComputerError> {
        self.inner
            .build_query_computer(from)
            .map(|computer| BetaComputer::new(computer, self.labels.clone(), self.beta))
    }
}

impl<Inner, T> ExpandBeam<T> for BetaAccessor<Inner>
where
    Inner: BuildQueryComputer<T> + AsNeighbor,
    T: ?Sized,
{
}

/// A [`PreprocessedDistanceFunction`] that applied `beta` filtering to the inner computer.
pub struct BetaComputer<Inner, I: VectorId> {
    inner: Inner,
    labels: Arc<dyn QueryLabelProvider<I>>,
    beta: f32,
}

impl<Inner, I> BetaComputer<Inner, I>
where
    I: VectorId,
{
    /// Construct a new `BetaComputer` around `Inner`.
    pub fn new(inner: Inner, labels: Arc<dyn QueryLabelProvider<I>>, beta: f32) -> Self {
        Self {
            inner,
            labels,
            beta,
        }
    }

    /// Return a reference to the inner computer.
    pub fn inner(&self) -> &Inner {
        &self.inner
    }
}

impl<T, Inner, I> PreprocessedDistanceFunction<Pair<I, T>, f32> for BetaComputer<Inner, I>
where
    I: VectorId,
    Inner: PreprocessedDistanceFunction<T, f32>,
{
    /// Check whether the ID satisfied the predicate computed by the label provider.
    ///
    /// If so, multiply the distance computed by `Inner` by `beta`.
    #[inline(always)]
    fn evaluate_similarity(&self, x: Pair<I, T>) -> f32 {
        // Inner distance computation.
        let distance = self.inner.evaluate_similarity(x.element);
        // Check beta.
        if self.labels.is_match(x.id) {
            distance * self.beta
        } else {
            distance
        }
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use diskann::{
        ANNError, ANNResult, always_escalate,
        graph::AdjacencyList,
        graph::glue::CopyIds,
        provider::{DefaultContext, NeighborAccessor},
    };
    use futures_util::future;
    use thiserror::Error;

    use super::*;

    /// A very simple data provider.
    struct SimpleProvider;
    impl DataProvider for SimpleProvider {
        type Context = DefaultContext;
        type InternalId = u32;
        type ExternalId = u64;

        type Error = ANNError;

        fn to_internal_id(&self, _context: &DefaultContext, gid: &u64) -> ANNResult<u32> {
            Ok((*gid).try_into()?)
        }

        fn to_external_id(&self, _context: &DefaultContext, id: u32) -> ANNResult<u64> {
            Ok(id.into())
        }
    }

    /// An `Accessor` that doubles its input ID as its output element.
    ///
    /// This also tracks the number of calls made to `get_element` and
    /// `on_elements_unordered` to ensure that `BetaFilter` correctly forwards these methods.
    #[derive(Debug, Default, Clone, Copy)]
    struct Doubler {
        get_element: usize,
        on_elements_unordered: usize,
    }

    impl SearchExt for Doubler {
        async fn starting_points(&self) -> ANNResult<Vec<u32>> {
            Ok(vec![0])
        }
    }

    impl Doubler {
        fn reset(&mut self) {
            *self = Self::default();
        }
    }

    /// A simple error type to test error forwarding.
    #[derive(Debug, Error)]
    #[error("the value {0} is not allowed")]
    pub struct NotAllowed(u32);

    impl From<NotAllowed> for ANNError {
        #[inline(never)]
        fn from(value: NotAllowed) -> Self {
            ANNError::log_async_error(value)
        }
    }

    impl HasId for Doubler {
        type Id = u32;
    }

    impl NeighborAccessor for Doubler {
        fn get_neighbors(
            self,
            _id: Self::Id,
            neighbors: &mut AdjacencyList<Self::Id>,
        ) -> impl Future<Output = ANNResult<Self>> + Send {
            neighbors.clear();
            future::ok(self)
        }
    }

    always_escalate!(NotAllowed);

    impl Accessor for Doubler {
        type Extended = u64;
        type Element<'a>
            = u64
        where
            Self: 'a;
        type ElementRef<'a> = u64;

        type GetError = NotAllowed;

        fn get_element(
            &mut self,
            id: u32,
        ) -> impl std::future::Future<Output = Result<Self::Element<'_>, Self::GetError>> + Send
        {
            self.get_element += 1;
            let is_err = (100..200).contains(&id);

            async move {
                if is_err {
                    Err(NotAllowed(id))
                } else {
                    let id: u64 = id.into();
                    Ok(2 * id)
                }
            }
        }

        async fn on_elements_unordered<Itr, F>(
            &mut self,
            itr: Itr,
            mut f: F,
        ) -> Result<(), Self::GetError>
        where
            Self: Sync,
            Itr: Iterator<Item = Self::Id> + Send,
            F: Send + for<'a> FnMut(Self::ElementRef<'a>, Self::Id),
        {
            self.on_elements_unordered += 1;
            for i in itr {
                f(self.get_element(i).await?, i);
            }
            Ok(())
        }
    }

    struct AddingComputer(u64);
    impl PreprocessedDistanceFunction<u64, f32> for AddingComputer {
        fn evaluate_similarity(&self, x: u64) -> f32 {
            (self.0 + x) as f32
        }
    }

    impl BuildQueryComputer<u64> for Doubler {
        type QueryComputer = AddingComputer;
        type QueryComputerError = ANNError;

        fn build_query_computer(
            &self,
            from: &u64,
        ) -> Result<Self::QueryComputer, Self::QueryComputerError> {
            Ok(AddingComputer(*from))
        }
    }

    impl ExpandBeam<u64> for Doubler {}

    #[derive(Debug)]
    struct SimpleStrategy;

    impl SearchStrategy<SimpleProvider, u64> for SimpleStrategy {
        type SearchAccessor<'a> = Doubler;
        type QueryComputer = AddingComputer;
        type PostProcessor = CopyIds;
        type SearchAccessorError = ANNError;

        fn search_accessor<'a>(
            &'a self,
            _provider: &'a SimpleProvider,
            _context: &'a DefaultContext,
        ) -> Result<Self::SearchAccessor<'a>, Self::SearchAccessorError> {
            Ok(Doubler::default())
        }

        fn post_processor(&self) -> Self::PostProcessor {
            Default::default()
        }
    }

    /// A simple `QueryLabelProvider` that matches multiples of 3.
    #[derive(Debug)]
    struct ThreeFilter;

    impl QueryLabelProvider<u32> for ThreeFilter {
        fn is_match(&self, id: u32) -> bool {
            id.is_multiple_of(3)
        }
    }

    #[tokio::test]
    async fn test_beta_filter() {
        let provider = SimpleProvider;
        let context = &DefaultContext;
        let beta: f32 = 0.25;

        let strategy = BetaFilter::new(SimpleStrategy, Arc::new(ThreeFilter), beta);

        let mut accessor: BetaAccessor<_> = strategy.search_accessor(&provider, context).unwrap();
        assert_eq!(accessor.inner.get_element, 0);
        assert_eq!(accessor.inner.on_elements_unordered, 0);

        // Test non-erroring path.
        let v = accessor.get_element(1).await.unwrap();
        assert_eq!(v, Pair::new(1, 2));

        let v = accessor.get_element(2).await.unwrap();
        assert_eq!(v, Pair::new(2, 4));

        // Test erroring path.
        assert!(accessor.get_element(100).await.is_err());
        assert!(accessor.get_element(101).await.is_err());

        assert_eq!(accessor.inner.get_element, 4);
        assert_eq!(accessor.inner.on_elements_unordered, 0);
        accessor.inner.reset();

        // On elements unordered.
        {
            let mut v = Vec::new();
            accessor
                .on_elements_unordered([1, 2, 3, 4, 5].into_iter(), |element, id| {
                    v.push((element, id));
                })
                .await
                .unwrap();

            assert_eq!(accessor.inner.get_element, 5);
            assert_eq!(accessor.inner.on_elements_unordered, 1);
            assert_eq!(
                v,
                &[
                    (Pair::new(1, 2), 1),
                    (Pair::new(2, 4), 2),
                    (Pair::new(3, 6), 3),
                    (Pair::new(4, 8), 4),
                    (Pair::new(5, 10), 5)
                ]
            );
            accessor.inner.reset();
        }

        // On-elements-unordered propagates errors.
        assert!(
            accessor
                .on_elements_unordered([1, 2, 3, 100, 4].into_iter(), |_, _| {})
                .await
                .is_err()
        );

        // Computation.
        let query = 10;
        let computer = accessor.build_query_computer(&query).unwrap();

        assert_eq!(
            computer.evaluate_similarity(accessor.get_element(10).await.unwrap()),
            (10 * 2 + query) as f32
        );
        assert_eq!(
            computer.evaluate_similarity(accessor.get_element(11).await.unwrap()),
            (11 * 2 + query) as f32
        );
        assert_eq!(
            computer.evaluate_similarity(accessor.get_element(12).await.unwrap()),
            beta * ((12 * 2 + query) as f32)
        );

        // Extended + computation.
        {
            type Extended = ExtendedPair<u32, u64>;
            let v: Extended = accessor.get_element(10).await.unwrap().into();
            assert_eq!(
                computer.evaluate_similarity(v.reborrow()),
                (10 * 2 + query) as f32
            );
        }

        // Test dummy implementation of `get_neighbors` for code coverage.
        let mut neighbors = AdjacencyList::new();
        accessor.get_neighbors(0, &mut neighbors).await.unwrap();
        assert_eq!(neighbors.len(), 0);
    }
}
