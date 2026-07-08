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
        ext::labeled::QueryLabelProvider,
        glue::{self, SearchPostProcessStep, SearchStrategy},
    },
    neighbor::Neighbor,
    provider::{DataProvider, HasId},
    utils::VectorId,
};

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
    fn apply(&self, id: I, distance: f32) -> f32 {
        if self.labels.is_match(id) {
            distance * self.beta
        } else {
            distance
        }
    }
}

impl<Inner> HasId for BetaAccessor<Inner>
where
    Inner: HasId,
{
    type Id = Inner::Id;
}

impl<Inner> glue::SearchAccessor for BetaAccessor<Inner>
where
    Inner: glue::SearchAccessor,
{
    fn starting_points(&self) -> impl Future<Output = ANNResult<Vec<Inner::Id>>> + Send {
        self.inner.starting_points()
    }

    fn start_point_distances<F>(
        &mut self,
        mut f: F,
    ) -> impl std::future::Future<Output = ANNResult<()>> + Send
    where
        F: FnMut(Self::Id, f32) + Send,
    {
        let filter = &self.filter;
        self.inner.start_point_distances(move |id, distance| {
            f(id, filter.apply(id, distance));
        })
    }

    fn expand_beam<Itr, P, F>(
        &mut self,
        ids: Itr,
        pred: P,
        mut on_neighbors: F,
    ) -> impl std::future::Future<Output = ANNResult<()>> + Send
    where
        Itr: Iterator<Item = Self::Id> + Send,
        P: glue::HybridPredicate<Self::Id> + Send + Sync,
        F: FnMut(Self::Id, f32) + Send,
    {
        let filter = &self.filter;
        self.inner.expand_beam(ids, pred, move |id, distance| {
            on_neighbors(id, filter.apply(id, distance))
        })
    }
}
///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    use diskann::graph::{
        glue::{HybridPredicate, Predicate, PredicateMut, SearchAccessor},
        test::{provider as test_provider, synthetic::Grid},
    };
    use std::collections::HashSet;

    /// A simple `QueryLabelProvider` that matches multiples of 3.
    #[derive(Debug)]
    struct ThreeFilter;

    impl QueryLabelProvider<u32> for ThreeFilter {
        fn is_match(&self, id: u32) -> bool {
            id.is_multiple_of(3)
        }
    }

    struct NotIn<'a>(&'a mut HashSet<u32>);

    impl Predicate<u32> for NotIn<'_> {
        fn eval(&self, item: &u32) -> bool {
            !self.0.contains(item)
        }
    }

    impl PredicateMut<u32> for NotIn<'_> {
        fn eval_mut(&mut self, item: &u32) -> bool {
            self.0.insert(*item)
        }
    }

    impl HybridPredicate<u32> for NotIn<'_> {}

    #[tokio::test]
    async fn test_beta_filter() {
        // The grid of 4x4 will look like this:
        //
        // |             16
        // | 3  7 11 15
        // | 2  6 10 14
        // | 1  5  9 13
        // | 0  4  8 12
        // +---------------
        //
        let provider = test_provider::Provider::grid(Grid::Two, 4).unwrap();
        let context = test_provider::Context::new();

        let beta: f32 = 0.25;

        let strategy = BetaFilter::new(test_provider::Strategy::new(), Arc::new(ThreeFilter), beta);

        let start_point_ids: Vec<_> = provider.start_point_ids().collect();
        assert_eq!(
            start_point_ids.len(),
            1,
            "grid should only have a single start point"
        );
        let start_point = start_point_ids[0];
        assert_eq!(
            start_point,
            u32::MAX,
            "`Provider::grid` is documented to use `u32::MAX` as its start point",
        );

        let mut visited = HashSet::new();
        let mut buf = Vec::new();

        let mut accessor = strategy
            .search_accessor(&provider, &context, &[0.0, 0.0])
            .unwrap();

        assert_eq!(
            &*accessor.starting_points().await.unwrap(),
            &*start_point_ids,
            "the underlying start points should match",
        );

        accessor
            .expand_beam(
                [0, 5, 10, 15].into_iter(),
                NotIn(&mut visited),
                |id, distance| buf.push((id, distance)),
            )
            .await
            .unwrap();

        // The expansion order is unknown, but we know from the structure of the grid what
        // the result should be.
        //
        // Since each entry in the beam is connected
        buf.sort_by_key(|(id, _)| *id);
        assert_eq!(
            &*buf,
            [
                (1, 1.0),
                (4, 1.0),
                (6, 5.0 * beta),
                (9, 5.0 * beta),
                (11, 13.0),
                (14, 13.0),
            ]
        );

        buf.clear();
        accessor
            .start_point_distances(|id, distance| buf.push((id, distance)))
            .await
            .unwrap();

        assert_eq!(
            &*buf,
            [(start_point, 32.0 * beta)],
            "u32::MAX is a multiple of 3"
        );
    }
}
