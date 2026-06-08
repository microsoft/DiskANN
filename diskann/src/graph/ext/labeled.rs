/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use crate::{
    ANNResult,
    graph::glue::{self, Decision, ExpansionHint},
    neighbor::Neighbor,
    provider::{DataProvider, HasId},
    utils::VectorId,
};

// /// Decision returned by [`QueryLabelProvider::on_visit`] to control search traversal.
// #[derive(Debug, Clone, Copy, PartialEq)]
// pub enum QueryVisitDecision<I: VectorId> {
//     /// Accept this node into the frontier for further traversal.
//     Accept(Neighbor<I>),
//     /// Reject this node; do not add it to the frontier.
//     Reject,
//     /// Stop the search immediately without accepting this node.
//     Terminate,
// }

pub trait QueryLabelProvider<I: VectorId>: std::fmt::Debug + Send + Sync {
    /// This is a query scoped provider
    /// Check if the `i`'s label match the query label
    fn is_match(&self, i: I) -> bool;
}

fn decide<T, I>(provider: &T, i: I) -> Decision<I>
where
    I: VectorId,
    T: QueryLabelProvider<I> + ?Sized,
{
    if provider.is_match(i) {
        Decision::Accept(i)
    } else {
        Decision::Reject(i)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Filtered<'a, S, I> {
    strategy: S,
    labels: &'a dyn QueryLabelProvider<I>,
}

impl<'a, S, I> Filtered<'a, S, I> {
    pub fn new(strategy: S, labels: &'a dyn QueryLabelProvider<I>) -> Self {
        Self { strategy, labels }
    }

    pub fn labels(&self) -> &'a dyn QueryLabelProvider<I> {
        self.labels
    }
}

impl<'a, S, DP, T> glue::SearchStrategy<'a, DP, T> for Filtered<'_, S, DP::InternalId>
where
    DP: DataProvider,
    S: glue::SearchStrategy<'a, DP, T>,
{
    type SearchAccessor = FilteredAccessor<'a, S::SearchAccessor>;
    type SearchAccessorError = S::SearchAccessorError;

    fn search_accessor(
        &'a self,
        provider: &'a DP,
        context: &'a DP::Context,
        query: T,
    ) -> Result<Self::SearchAccessor, Self::SearchAccessorError> {
        let inner = self.strategy.search_accessor(provider, context, query)?;
        Ok(FilteredAccessor {
            inner,
            labels: self.labels,
        })
    }
}

impl<'a, S, DP, T, O> glue::DefaultPostProcessor<'a, DP, T, O> for Filtered<'_, S, DP::InternalId>
where
    S: glue::DefaultPostProcessor<'a, DP, T, O>,
    DP: DataProvider,
    O: Send,
{
    type Processor = glue::Pipeline<Unfiltered, S::Processor>;

    fn default_post_processor(&'a self) -> Self::Processor {
        glue::Pipeline::new(Unfiltered, self.strategy.default_post_processor())
    }
}

#[derive(Debug)]
pub struct FilteredAccessor<'a, A>
where
    A: HasId,
{
    inner: A,
    labels: &'a dyn QueryLabelProvider<A::Id>,
}

impl<'a, A> FilteredAccessor<'a, A>
where
    A: HasId,
{
    pub(crate) fn new(inner: A, labels: &'a dyn QueryLabelProvider<A::Id>) -> Self {
        Self { inner, labels }
    }
}

impl<A> HasId for FilteredAccessor<'_, A>
where
    A: HasId,
{
    type Id = A::Id;
}

impl<'a, A> glue::FilteredAccessor for FilteredAccessor<'a, A>
where
    A: glue::SearchAccessor,
{
    async fn starting_points(&self) -> ANNResult<Vec<Decision<Self::Id>>> {
        let points = self.inner.starting_points().await?;
        let annotated = points.into_iter().map(|i| decide(self.labels, i)).collect();
        Ok(annotated)
    }

    async fn start_point_distances<F>(&mut self, mut f: F) -> ANNResult<()>
    where
        F: FnMut(Decision<Self::Id>, f32) + Send,
    {
        self.inner
            .start_point_distances(|id, distance| f(decide(self.labels, id), distance))
            .await
    }

    async fn expand_beam_filtered<Itr, P, F>(
        &mut self,
        hint: ExpansionHint,
        ids: Itr,
        pred: P,
        mut on_neighbors: F,
    ) -> ANNResult<()>
    where
        Itr: Iterator<Item = Self::Id> + Send,
        P: glue::HybridPredicate<Self::Id> + Send + Sync,
        F: FnMut(Decision<Self::Id>, f32) + Send,
    {
        match hint {
            ExpansionHint::All => {
                self.inner
                    .expand_beam(ids, pred, |id, distance| {
                        on_neighbors(decide(self.labels, id), distance)
                    })
                    .await
            }
            ExpansionHint::AcceptOnly => {
                self.inner
                    .expand_beam(ids, Wrapper::new(pred, self.labels), |id, distance| {
                        on_neighbors(Decision::Accept(id), distance)
                    })
                    .await
            }
        }
    }

    fn terminate_early(&mut self) -> bool {
        self.inner.terminate_early()
    }

    fn is_not_start_point(
        &self,
    ) -> impl std::future::Future<
        Output = ANNResult<impl Fn(Self::Id) -> bool + Send + Sync + 'static>,
    > + Send {
        self.inner.is_not_start_point()
    }

    fn num_starting_points(&self) -> impl std::future::Future<Output = ANNResult<usize>> + Send {
        self.inner.num_starting_points()
    }
}

/// A [`PostProecessLayer`] for [`FilteredAccessor`] that delegates to the inner accessor.
#[derive(Debug, Clone, Copy)]
pub struct Unfiltered;

impl<A, T, O> glue::SearchPostProcessStep<FilteredAccessor<'_, A>, T, O> for Unfiltered
where
    A: HasId,
{
    type Error<NextError>
        = NextError
    where
        NextError: crate::error::StandardError;
    type NextAccessor = A;

    fn post_process_step<I, B, Next>(
        &self,
        next: &Next,
        accessor: &mut FilteredAccessor<'_, A>,
        query: T,
        candidates: I,
        output: &mut B,
    ) -> impl std::future::Future<Output = Result<usize, Self::Error<Next::Error>>> + Send
    where
        I: Iterator<Item = Neighbor<A::Id>> + Send,
        B: crate::graph::SearchOutputBuffer<O> + Send + ?Sized,
        Next: glue::SearchPostProcess<Self::NextAccessor, T, O> + Sync,
    {
        use glue::SearchPostProcess;
        next.post_process(&mut accessor.inner, query, candidates, output)
    }
}

/// A [`glue::HybridPredicate`] that uses an additional [`QueryLabelProvider`] to skip
/// computing distances to items that do not satisfy the predicate.
#[derive(Debug)]
struct Wrapper<'a, P, I>
where
    I: VectorId,
{
    inner: P,
    labels: &'a dyn QueryLabelProvider<I>,
}

impl<'a, P, I> Wrapper<'a, P, I>
where
    I: VectorId,
{
    fn new(inner: P, labels: &'a dyn QueryLabelProvider<I>) -> Self {
        Self { inner, labels }
    }
}

impl<P, I> glue::Predicate<I> for Wrapper<'_, P, I>
where
    P: glue::Predicate<I>,
    I: VectorId,
{
    fn eval(&self, item: &I) -> bool {
        self.inner.eval(item) && self.labels.is_match(*item)
    }
}

impl<P, I> glue::PredicateMut<I> for Wrapper<'_, P, I>
where
    P: glue::PredicateMut<I>,
    I: VectorId,
{
    fn eval_mut(&mut self, item: &I) -> bool {
        self.inner.eval_mut(item) && self.labels.is_match(*item)
    }
}

impl<P, I> glue::HybridPredicate<I> for Wrapper<'_, P, I>
where
    P: glue::HybridPredicate<I>,
    I: VectorId,
{
}
