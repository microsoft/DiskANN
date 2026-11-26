/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! A [filtered search](glue::FilteredAccessor) adaptor around traditional
//! [`SearchAccessor`](glue::SearchAccessor) where the acceptance/rejection criteria comes
//! from an externally provided [`QueryLabelProvider`].
//!
//! To use, embed a [`SearchStrategy`](glue::SearchStrategy) `S` in a [`Filtered`] strategy
//! with an associated [`QueryLabelProvider`]. This implements `SearchStrategy` under the
//! same conditions as `S` with the resulting [`FilteredAccessor`] implementing
//! [`glue::FilteredAccessor`].
//!
//! [`glue::DefaultPostProcessor`] defaults to the `<S as DefaultPostProcessor<_, _, _>>`.
//! To use a general post-processor for `S`, the [`Unfiltered`] [`glue::SearchPostProcessStep`]
//! can be used to strip off the wrapping [`FilteredAccessor`] and retrieve the inner
//! `SearchAccessor`.
//!
//! ## Limitations
//!
//! * The provided [`QueryLabelProvider`] must somehow agree on the id space as the inner
//!   `SearchAccessor`.
//!
//! * There exists a time-of-check, time-of-use window between label checks and distance
//!   computations.
//!
//! * The [`QueryLabelProvider`] is checked behind a trait object.
//!
//! * By default, the [`FilteredAccessor`] is not passed to search post-processing.
//!
//! * As of writing, [`QueryLabelProvider`] does not have batched accesses.

use crate::{
    ANNResult, HasId,
    graph::glue::{self, Accept, Decision},
    neighbor::Neighbor,
    utils::VectorId,
};

/// Decide whether or not a vector ID is accepted or rejected for a filtered search.
pub trait QueryLabelProvider<I>: std::fmt::Debug + Send + Sync
where
    I: VectorId,
{
    /// Return `true` if `i` should be accepted. Otherwise, return `false`.
    ///
    /// When used with [`Filtered`], callers may assume that `i` was yielded from a
    /// callback from [`glue::SearchAccessor::expand_beam`].
    fn is_match(&self, i: I) -> bool;
}

// Move this external to the `QueryLabelProvider` call so the indirect call has a slightly
// simpler signature. Does it matter? Probably not.
fn decide<T, I>(provider: &T, i: I) -> Decision<I>
where
    I: VectorId,
    T: QueryLabelProvider<I> + ?Sized,
{
    if provider.is_match(i) {
        Decision::accept(i)
    } else {
        Decision::reject(i)
    }
}

impl<O, A> glue::DefaultPostProcessor<O> for FilteredAccessor<'_, A>
where
    A: HasId + glue::DefaultPostProcessor<O>,
    O: Send,
{
    type Processor = glue::Pipeline<Unfiltered, A::Processor>;

    fn default_post_processor(&self) -> Self::Processor {
        glue::Pipeline::new(Unfiltered, self.inner.default_post_processor())
    }
}

/// A [`FilteredAccessor`](glue::FilteredAccessor) adaptor for the [`glue::SearchAccessor`]
/// `A` where filter accept/reject decisions come from an externally supplied
/// [`QueryLabelProvider`].
///
/// See: [`Filtered`], [`QueryLabelProvider`].
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
    pub fn new(inner: A, labels: &'a dyn QueryLabelProvider<A::Id>) -> Self {
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
        ids: Itr,
        pred: P,
        mut on_neighbors: F,
    ) -> ANNResult<()>
    where
        Itr: Iterator<Item = Self::Id> + Send,
        P: glue::HybridPredicate<Self::Id> + Send + Sync,
        F: FnMut(Decision<Self::Id>, f32) + Send,
    {
        self.inner
            .expand_beam(ids, pred, |id, distance| {
                on_neighbors(decide(self.labels, id), distance)
            })
            .await
    }

    async fn expand_beam_accept_only<Itr, P, F>(
        &mut self,
        ids: Itr,
        pred: P,
        mut on_neighbors: F,
    ) -> ANNResult<()>
    where
        Itr: Iterator<Item = Self::Id> + Send,
        P: glue::Predicate<Self::Id> + glue::PredicateMut<Accept<Self::Id>> + Send + Sync,
        F: FnMut(Accept<Self::Id>, f32) + Send,
    {
        self.inner
            .expand_beam(ids, EvalFiltered::new(pred, self.labels), |id, distance| {
                on_neighbors(Accept::new(id), distance)
            })
            .await
    }

    fn terminate_early(&mut self) -> bool {
        self.inner.terminate_early()
    }

    fn num_starting_points(
        &mut self,
    ) -> impl std::future::Future<Output = ANNResult<usize>> + Send {
        self.inner.num_starting_points()
    }
}

/// A [`SearchPostProcessStep`](glue::SearchPostProcessStep) for [`FilteredAccessor`] that
/// delegates to the inner accessor.
#[derive(Debug, Clone, Copy)]
pub struct Unfiltered;

impl<O, A> glue::SearchPostProcessStep<FilteredAccessor<'_, A>, O> for Unfiltered
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
        candidates: I,
        output: &mut B,
    ) -> impl std::future::Future<Output = Result<usize, Self::Error<Next::Error>>> + Send
    where
        I: Iterator<Item = Neighbor<A::Id>> + Send,
        B: crate::graph::SearchOutputBuffer<O> + Send + ?Sized,
        Next: glue::SearchPostProcess<Self::NextAccessor, O> + Sync,
    {
        next.post_process(&mut accessor.inner, candidates, output)
    }
}

/// A [`glue::HybridPredicate`] that uses an additional [`QueryLabelProvider`] to skip
/// computing distances to items that do not satisfy the predicate.
#[derive(Debug)]
struct EvalFiltered<'a, P, I>
where
    I: VectorId,
{
    inner: P,
    labels: &'a dyn QueryLabelProvider<I>,
}

impl<'a, P, I> EvalFiltered<'a, P, I>
where
    I: VectorId,
{
    fn new(inner: P, labels: &'a dyn QueryLabelProvider<I>) -> Self {
        Self { inner, labels }
    }
}

impl<P, I> glue::Predicate<I> for EvalFiltered<'_, P, I>
where
    P: glue::Predicate<I>,
    I: VectorId,
{
    fn eval(&self, item: &I) -> bool {
        self.inner.eval(item) && self.labels.is_match(*item)
    }
}

impl<P, I> glue::PredicateMut<I> for EvalFiltered<'_, P, I>
where
    P: glue::PredicateMut<Accept<I>>,
    I: VectorId,
{
    fn eval_mut(&mut self, item: &I) -> bool {
        // Order must be `label` -> `inner` because we have to know an ID is accepted before
        // passing it to `eval_mut`.
        self.labels.is_match(*item) && self.inner.eval_mut(&Accept::new(*item))
    }
}

impl<P, I> glue::HybridPredicate<I> for EvalFiltered<'_, P, I>
where
    P: glue::Predicate<I> + glue::PredicateMut<Accept<I>>,
    I: VectorId,
{
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use crate::graph::{
        glue::{FilteredAccessor as _, HybridPredicate, Predicate, PredicateMut},
        test::{self, index::Index, synthetic::Grid},
    };

    #[derive(Debug)]
    struct EvenFilter;

    impl QueryLabelProvider<u32> for EvenFilter {
        fn is_match(&self, id: u32) -> bool {
            id.is_multiple_of(2)
        }
    }

    struct NotIn<'a>(&'a mut HashSet<u32>);

    impl Predicate<u32> for NotIn<'_> {
        fn eval(&self, item: &u32) -> bool {
            !self.0.contains(item)
        }
    }

    impl Predicate<Accept<u32>> for NotIn<'_> {
        fn eval(&self, item: &Accept<u32>) -> bool {
            self.eval(item.get())
        }
    }

    impl PredicateMut<u32> for NotIn<'_> {
        fn eval_mut(&mut self, item: &u32) -> bool {
            self.0.insert(*item)
        }
    }

    impl PredicateMut<Accept<u32>> for NotIn<'_> {
        fn eval_mut(&mut self, item: &Accept<u32>) -> bool {
            self.eval_mut(item.get())
        }
    }

    impl HybridPredicate<u32> for NotIn<'_> {}
    impl HybridPredicate<Accept<u32>> for NotIn<'_> {}

    #[tokio::test]
    async fn filtered_accessor_wrapping() {
        // Grid::Two, size 4:
        //
        // | 3  7 11 15
        // | 2  6 10 14
        // | 1  5  9 13
        // | 0  4  8 12
        //
        // Start point: u32::MAX (odd → rejected by EvenFilter).
        let provider = Index::grid(Grid::Two, 4).unwrap();
        let filter = EvenFilter;

        let inner = test::index::Accessor::new(&provider, &[0.0, 0.0]).unwrap();
        let mut accessor = FilteredAccessor::new(inner, &filter);

        // -- starting_points: u32::MAX is odd, should be Reject --
        assert_eq!(accessor.num_starting_points().await.unwrap(), 1);

        // -- start_point_distances: verify annotation and distance --
        let mut start_results = Vec::new();
        accessor
            .start_point_distances(|d, dist| start_results.push((d, dist)))
            .await
            .unwrap();
        assert_eq!(start_results.len(), 1);
        assert!(start_results[0].0.is_reject());

        // -- expand_beam_filtered
        let count_before = accessor.inner.get_vector_count();
        let mut visited = HashSet::new();
        let mut results_all: Vec<(Decision<u32>, f32)> = Vec::new();

        accessor
            .expand_beam_filtered(
                [0, 5, 10, 15].into_iter(),
                NotIn(&mut visited),
                |d, dist| results_all.push((d, dist)),
            )
            .await
            .unwrap();

        let get_vector_all = accessor.inner.get_vector_count() - count_before;

        // Every neighbor should have a distance computed.
        assert_eq!(get_vector_all, results_all.len());

        // Verify decisions: even IDs are Accept, odd are Reject.
        for (decision, _) in &results_all {
            let id = decision.into_inner();
            if id.is_multiple_of(2) {
                assert!(decision.is_accept(), "even id {id} should be Accept");
            } else {
                assert!(decision.is_reject(), "odd id {id} should be Reject");
            }
        }

        // -- expand_beam_accept_only
        let count_before = accessor.inner.get_vector_count();
        let mut visited2 = HashSet::new();
        let mut results_accept: Vec<(Accept<u32>, f32)> = Vec::new();

        accessor
            .expand_beam_accept_only(
                [0, 5, 10, 15].into_iter(),
                NotIn(&mut visited2),
                |d, dist| results_accept.push((d, dist)),
            )
            .await
            .unwrap();

        let get_vector_accept = accessor.inner.get_vector_count() - count_before;

        // Fewer distance computations than the All path — rejected items were skipped.
        assert!(
            get_vector_accept < get_vector_all,
            "AcceptOnly should compute fewer distances: {get_vector_accept} < {get_vector_all}"
        );

        // And the count should match exactly the number of accepted results.
        assert_eq!(get_vector_accept, results_accept.len());
    }

    #[tokio::test]
    async fn filtered_strategy_produces_accessor() {
        let provider = Index::grid(Grid::Two, 4).unwrap();
        let filter = EvenFilter;

        let mut accessor =
            FilteredAccessor::new(provider.search_accessor(&[0.0, 0.0]).unwrap(), &filter);

        let mut starts = Vec::new();
        glue::FilteredAccessor::start_point_distances(&mut accessor, |decision, dist| {
            starts.push((decision, dist));
        })
        .await
        .unwrap();

        assert_eq!(starts.len(), 1);
        // u32::MAX is odd → Reject
        assert!(starts[0].0.is_reject());
        assert_eq!(starts[0].0.into_inner(), u32::MAX);
    }

    #[test]
    fn wrapper_predicate() {
        let filter = EvenFilter;
        let mut visited = HashSet::new();
        let mut wrapper = EvalFiltered::new(NotIn(&mut visited), &filter);

        // eval: both inner (not visited) and label (even) must pass.
        assert!(wrapper.eval(&2), "2 is even and not visited");
        assert!(!wrapper.eval(&3), "3 is odd → label rejects");

        // eval_mut: inserts into visited set on success.
        assert!(
            wrapper.eval_mut(&4),
            "4 is even and not visited → accept and insert"
        );
        assert!(!wrapper.eval_mut(&4), "4 is now visited → inner rejects");
        assert!(!wrapper.eval_mut(&5), "5 is odd → label rejects");

        // HybridPredicate agreement: eval and eval_mut agree on exclusion.
        // Item 4 is already visited → both return false.
        assert!(!wrapper.eval(&4));
        assert!(!wrapper.eval_mut(&4));
    }
}
