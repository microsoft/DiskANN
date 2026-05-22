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
    provider::{Accessor, AsNeighbor, DataProvider, DelegateNeighbor, HasId},
    utils::VectorId,
};
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
    A: HasId,
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
        query: T,
        candidates: I,
        output: &mut B,
    ) -> impl Future<Output = Result<usize, Next::Error>> + Send
    where
        I: Iterator<Item = Neighbor<A::Id>> + Send,
        B: SearchOutputBuffer<O> + Send + ?Sized,
        Next: glue::SearchPostProcess<Self::NextAccessor, T, O>,
    {
        next.post_process(&mut accessor.inner, query, candidates, output)
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
impl<'a, Provider, Strategy, T, I> SearchStrategy<'a, Provider, T> for BetaFilter<Strategy, I>
where
    I: VectorId,
    Provider: DataProvider<InternalId = I>,
    Strategy: SearchStrategy<'a, Provider, T>,
    T: 'a,
{
    /// An accessor that returns the ID in addition to the element yielded by the inner
    /// accessor.
    type SearchAccessor = BetaAccessor<Strategy::SearchAccessor>;

    type SearchAccessorError = Strategy::SearchAccessorError;

    /// Compose the [`BetaAccessor`] with the inner search strategy's [`Accessor`].
    fn search_accessor(
        &'a self,
        provider: &'a Provider,
        context: &'a Provider::Context,
        query: T,
    ) -> Result<Self::SearchAccessor, Self::SearchAccessorError> {
        Ok(BetaAccessor {
            inner: self.strategy.search_accessor(provider, context, query)?,
            filter: Filter {
                labels: self.labels.clone(),
                beta: self.beta,
            },
        })
    }
}

/// [`DefaultPostProcessor`] delegation for [`BetaFilter`]. The processor is composed by
/// wrapping the inner strategy's processor with [`Unwrap`] via [`Pipeline`].
impl<'a, Provider, Strategy, T, I, O> glue::DefaultPostProcessor<'a, Provider, T, O>
    for BetaFilter<Strategy, I>
where
    I: VectorId,
    O: Send,
    Provider: DataProvider<InternalId = I>,
    Strategy: glue::DefaultPostProcessor<'a, Provider, T, O>,
    T: 'a,
{
    type Processor = glue::Pipeline<Unwrap, Strategy::Processor>;

    fn default_post_processor(&'a self) -> Self::Processor {
        glue::Pipeline::new(Unwrap, self.strategy.default_post_processor())
    }
}

/////////////
// Helpers //
/////////////

/// An [`Accessor`] that composes with an `Inner` accessor to provide beta-filtering.
pub struct BetaAccessor<Inner>
where
    Inner: HasId,
{
    inner: Inner,
    filter: Filter<Inner::Id>,
}

struct Filter<I> {
    labels: Arc<dyn QueryLabelProvider<I>>,
    beta: f32,
}

impl<I> Filter<I>
where
    I: VectorId,
{
    fn apply(&self, distance: f32, id: I) -> f32 {
        if self.labels.is_match(id) {
            distance * self.beta
        } else {
            distance
        }
    }
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
    /// Use the same error type as `Inner`.
    type GetError = Inner::GetError;

    /// Invoke `get_element` on the inner accessor and return a tuple consisting of the
    /// retrieved element and `id`.
    #[inline(always)]
    fn get_distance(
        &mut self,
        id: Self::Id,
    ) -> impl Future<Output = Result<f32, Self::GetError>> + Send {
        let filter = &self.filter;

        // The first `map` applies to `Future`.
        // The second `map` applies to the `Result`.
        self.inner
            .get_distance(id)
            .map(move |result| result.map(move |distance| filter.apply(distance, id)))
    }

    /// Method `on_elements_unordered` is implemented by invoking
    /// `inner.on_elements_unordered` with a decorated version of `f`.
    async fn distances_unordered<Itr, F>(
        &mut self,
        itr: Itr,
        mut f: F,
    ) -> Result<(), Self::GetError>
    where
        Self: Sync,
        Itr: Iterator<Item = Self::Id> + Send,
        F: Send + FnMut(f32, Self::Id),
    {
        let filter = &self.filter;
        self.inner
            .distances_unordered(
                itr,
                #[inline]
                move |distance, id| f(filter.apply(distance, id), id),
            )
            .await
    }
}

impl<Inner> ExpandBeam for BetaAccessor<Inner> where Inner: ExpandBeam + AsNeighbor {}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use diskann::{
        ANNError, ANNResult, always_escalate,
        graph::AdjacencyList,
        graph::glue::CopyIds,
        provider::{DefaultContext, NeighborAccessor, NoopGuard},
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
        type Guard = NoopGuard<u32>;

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
    #[derive(Debug, Clone, Copy)]
    struct Doubler {
        get_element: usize,
        on_elements_unordered: usize,
        computer: AddingComputer,
    }

    impl SearchExt for Doubler {
        async fn starting_points(&self) -> ANNResult<Vec<u32>> {
            Ok(vec![0])
        }
    }

    impl Doubler {
        fn new(query: u64) -> Self {
            Self {
                get_element: 0,
                on_elements_unordered: 0,
                computer: AddingComputer(query),
            }
        }

        fn reset(&mut self) {
            self.get_element = 0;
            self.on_elements_unordered = 0;
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
        type GetError = NotAllowed;

        fn get_distance(
            &mut self,
            id: u32,
        ) -> impl std::future::Future<Output = Result<f32, Self::GetError>> + Send {
            self.get_element += 1;
            let is_err = (100..200).contains(&id);

            async move {
                if is_err {
                    Err(NotAllowed(id))
                } else {
                    let id: u64 = id.into();
                    Ok(self.computer.eval(2 * id))
                }
            }
        }

        async fn distances_unordered<Itr, F>(
            &mut self,
            itr: Itr,
            mut f: F,
        ) -> Result<(), Self::GetError>
        where
            Self: Sync,
            Itr: Iterator<Item = Self::Id> + Send,
            F: Send + FnMut(f32, Self::Id),
        {
            self.on_elements_unordered += 1;
            for i in itr {
                f(self.get_distance(i).await?, i);
            }
            Ok(())
        }
    }

    #[derive(Debug, Clone, Copy)]
    struct AddingComputer(u64);

    impl AddingComputer {
        fn eval(&self, x: u64) -> f32 {
            (self.0 + x) as f32
        }
    }

    impl ExpandBeam for Doubler {}

    #[derive(Debug)]
    struct SimpleStrategy;

    impl<'a> SearchStrategy<'a, SimpleProvider, u64> for SimpleStrategy {
        type SearchAccessor = Doubler;
        type SearchAccessorError = ANNError;

        fn search_accessor(
            &'a self,
            _provider: &'a SimpleProvider,
            _context: &'a DefaultContext,
            query: u64,
        ) -> Result<Self::SearchAccessor, Self::SearchAccessorError> {
            Ok(Doubler::new(query))
        }
    }

    impl<'a> glue::DefaultPostProcessor<'a, SimpleProvider, u64> for SimpleStrategy {
        diskann::default_post_processor!(CopyIds);
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

        let query = 10;
        let mut accessor: BetaAccessor<_> =
            strategy.search_accessor(&provider, context, query).unwrap();
        assert_eq!(accessor.inner.get_element, 0);
        assert_eq!(accessor.inner.on_elements_unordered, 0);

        let unfiltered = |id: u32| (2 * (id as u64) + query) as f32;
        let filtered = |id: u32| beta * unfiltered(id);

        // Test non-erroring path.
        let v = accessor.get_distance(1).await.unwrap();
        assert_eq!(v, unfiltered(1));

        let v = accessor.get_distance(2).await.unwrap();
        assert_eq!(v, unfiltered(2));

        // Test erroring path.
        assert!(accessor.get_distance(100).await.is_err());
        assert!(accessor.get_distance(101).await.is_err());

        assert_eq!(accessor.inner.get_element, 4);
        assert_eq!(accessor.inner.on_elements_unordered, 0);
        accessor.inner.reset();

        // On elements unordered.
        {
            let mut v = Vec::new();
            accessor
                .distances_unordered([1, 2, 3, 4, 5].into_iter(), |element, id| {
                    v.push((element, id));
                })
                .await
                .unwrap();

            assert_eq!(accessor.inner.get_element, 5);
            assert_eq!(accessor.inner.on_elements_unordered, 1);
            assert_eq!(
                v,
                &[
                    (unfiltered(1), 1),
                    (unfiltered(2), 2),
                    (filtered(3), 3),
                    (unfiltered(4), 4),
                    (unfiltered(5), 5)
                ]
            );
            accessor.inner.reset();
        }

        // On-elements-unordered propagates errors.
        assert!(
            accessor
                .distances_unordered([1, 2, 3, 100, 4].into_iter(), |_, _| {})
                .await
                .is_err()
        );

        assert_eq!(accessor.get_distance(10).await.unwrap(), unfiltered(10),);
        assert_eq!(accessor.get_distance(11).await.unwrap(), unfiltered(11),);
        assert_eq!(accessor.get_distance(12).await.unwrap(), filtered(12));

        // Test dummy implementation of `get_neighbors` for code coverage.
        let mut neighbors = AdjacencyList::new();
        accessor.get_neighbors(0, &mut neighbors).await.unwrap();
        assert_eq!(neighbors.len(), 0);
    }
}
