/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Range-based search within a distance radius.

use std::collections::VecDeque;
use std::num::NonZeroUsize;

use diskann_utils::future::SendFuture;
use thiserror::Error;

use crate::{
    ANNResult, convert_error,
    error::IntoANNResult,
    graph::{
        glue::{self, SearchAccessor, SearchStrategy},
        index::{DiskANNIndex, InternalSearchStats, SearchStats},
        search::{
            Knn, Search, filtered_range_search::FilteredRange, record::NoopSearchRecord,
            scratch::SearchScratch,
        },
        search_output_buffer::SearchOutputBuffer,
    },
    neighbor::Neighbor,
    provider::DataProvider,
};

/// Error type for [`Range`] parameter validation.
#[derive(Debug, Error)]
pub enum RangeSearchError {
    #[error("beam width cannot be zero")]
    BeamWidthZero,
    #[error("l_value cannot be zero")]
    LZero,
    #[error("initial_search_slack must be between 0 and 1.0")]
    StartingListSlackValueError,
    #[error("range_search_slack must be greater than or equal to 1.0")]
    RangeSearchSlackValueError,
    #[error("inner_radius must be less than or equal to radius")]
    InnerRadiusValueError,
    #[error("max_returned must be greater than or equal to starting_l")]
    MaxReturnedLessThanInitialL,
}

convert_error!(RangeSearchError);

/// Parameters for range-based search.
///
/// Finds all points within a specified distance radius from the query.
#[derive(Debug, Clone, Copy)]
pub struct Range {
    /// Maximum results to return (None = unlimited).
    max_returned: Option<usize>,
    /// Initial search list size.
    starting_l: NonZeroUsize,
    /// Beam width.
    beam_width: NonZeroUsize,
    /// Outer radius - points within this distance are candidates.
    radius: f32,
    /// Inner radius - points closer than this are excluded.
    inner_radius: Option<f32>,
    /// Slack factor for initial search phase (0.0 to 1.0).
    initial_slack: f32,
    /// Slack factor for range expansion (>= 1.0).
    range_slack: f32,
}

impl Range {
    /// Create range search with default slack values.
    pub fn new(starting_l: usize, radius: f32) -> Result<Self, RangeSearchError> {
        Self::builder(starting_l, radius).build()
    }

    /// Create a builder for range search parameters.
    ///
    /// The builder starts with the same defaults as [`Self::new`].
    pub fn builder(starting_l: usize, radius: f32) -> RangeBuilder {
        RangeBuilder {
            max_returned: None,
            starting_l,
            beam_width: None,
            radius,
            inner_radius: None,
            initial_slack: 1.0,
            range_slack: 1.0,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn validate_and_create(
        max_returned: Option<usize>,
        starting_l: usize,
        beam_width: Option<usize>,
        radius: f32,
        inner_radius: Option<f32>,
        initial_slack: f32,
        range_slack: f32,
    ) -> Result<Self, RangeSearchError> {
        let beam_width = match NonZeroUsize::new(beam_width.unwrap_or(1)) {
            Some(bw) => bw,
            None => return Err(RangeSearchError::BeamWidthZero),
        };
        let starting_l = match NonZeroUsize::new(starting_l) {
            Some(l) => l,
            None => return Err(RangeSearchError::LZero),
        };
        if let Some(max) = max_returned
            && max < starting_l.get()
        {
            return Err(RangeSearchError::MaxReturnedLessThanInitialL);
        }
        if !(0.0..=1.0).contains(&initial_slack) {
            return Err(RangeSearchError::StartingListSlackValueError);
        }
        if range_slack < 1.0 {
            return Err(RangeSearchError::RangeSearchSlackValueError);
        }
        if let Some(inner) = inner_radius
            && inner > radius
        {
            return Err(RangeSearchError::InnerRadiusValueError);
        }

        Ok(Self {
            max_returned,
            starting_l,
            beam_width,
            radius,
            inner_radius,
            initial_slack,
            range_slack,
        })
    }

    /// Returns the maximum number of results to return.
    #[inline]
    pub fn max_returned(&self) -> Option<usize> {
        self.max_returned
    }

    /// Returns either usize::MAX or the user-specified maximum number
    /// results to return incremented by a user-inputted value.
    /// Useful to enforce max results exactly when start points
    /// are filtered out during post-processing.
    #[inline]
    pub fn effective_max_returned(&self, inc: usize) -> usize {
        if let Some(max) = self.max_returned {
            max.saturating_add(inc)
        } else {
            usize::MAX
        }
    }

    /// Returns the initial search list size.
    #[inline]
    pub fn starting_l(&self) -> NonZeroUsize {
        self.starting_l
    }

    /// Returns the optional beam width.
    #[inline]
    pub fn beam_width(&self) -> NonZeroUsize {
        self.beam_width
    }

    /// Returns the outer radius.
    #[inline]
    pub fn radius(&self) -> f32 {
        self.radius
    }

    /// Returns the inner radius (points closer are excluded).
    #[inline]
    pub fn inner_radius(&self) -> Option<f32> {
        self.inner_radius
    }

    /// Returns the initial search slack factor.
    #[inline]
    pub fn initial_slack(&self) -> f32 {
        self.initial_slack
    }

    /// Returns the range search slack factor.
    #[inline]
    pub fn range_slack(&self) -> f32 {
        self.range_slack
    }

    /// Returns a [`Knn`] search parameter set with the same starting_l and beam_width.
    pub(super) fn to_knn(self) -> Knn {
        Knn::new_infallible(self.starting_l, self.beam_width)
    }
}

/// Builder for [`Range`] search parameters.
///
/// `max_returned`: If specified, the search will stop and return results once this number of points has been found within both the inner and outer radii.
/// Note that due to adding some extra slack for start points, occasionally slightly more than this number of points may be returned. Since the initial
/// search phase does not respect `max_returned`, this parameter may not be set lower than `starting_l`.
///
/// `starting_l`: the L_search parameter for the initial search phase. Must be greater than zero.
///
/// `beam_width`: the beam width for parallel graph exploration. If not specified, defaults to 1. Must be greater than zero if specified.
///
/// `radius`: the outer radius for the range search. Points within this distance from the query are candidates for inclusion in the results.
///
/// `inner_radius`: the inner radius for the range search. Points closer than this distance from the query are excluded from the results. Must be less than or equal to `radius` if specified.
///
/// `initial_slack`: after the initial knn search phase, a decision is made on whether to continue to the second round of search. This decision is based on whether
/// the number of points found within the outer radius is greater than `starting_l * initial_slack`, so lower values of `initial_slack` will make it more likely to continue to
/// the second round of search. Must be between 0.0 and 1.0.
///
/// `range_slack`: during the second round of search, points that are within `radius * range_slack` are expanded to search for candidates within the range, so greater
/// values of `range_slack` will mean more expansions. Must be greater than or equal to 1.0.
#[derive(Debug, Clone, Copy)]
pub struct RangeBuilder {
    max_returned: Option<usize>,
    starting_l: usize,
    beam_width: Option<usize>,
    radius: f32,
    inner_radius: Option<f32>,
    initial_slack: f32,
    range_slack: f32,
}

impl RangeBuilder {
    /// Build validated [`FilteredRange`] parameters.
    pub fn build_filtered(self) -> Result<FilteredRange, RangeSearchError> {
        let range_params = self.build()?;
        Ok(FilteredRange::from_range_params(range_params))
    }

    /// Set maximum results to return (`None` means unlimited).
    pub fn max_returned(mut self, value: Option<usize>) -> Self {
        self.max_returned = value;
        self
    }

    /// Set the beam width.
    pub fn beam_width(mut self, value: Option<usize>) -> Self {
        self.beam_width = value;
        self
    }

    /// Set the inner radius.
    pub fn inner_radius(mut self, value: Option<f32>) -> Self {
        self.inner_radius = value;
        self
    }

    /// Set the initial-search slack factor.
    pub fn initial_slack(mut self, value: f32) -> Self {
        self.initial_slack = value;
        self
    }

    /// Set the range-search slack factor.
    pub fn range_slack(mut self, value: f32) -> Self {
        self.range_slack = value;
        self
    }

    /// Build validated [`Range`] parameters.
    pub fn build(self) -> Result<Range, RangeSearchError> {
        Range::validate_and_create(
            self.max_returned,
            self.starting_l,
            self.beam_width,
            self.radius,
            self.inner_radius,
            self.initial_slack,
            self.range_slack,
        )
    }
}

impl<'a, DP, S, T> Search<'a, DP, S, T> for Range
where
    DP: DataProvider,
    S: SearchStrategy<'a, DP, T, SearchAccessor: SearchAccessor>,
    T: Copy + Send + Sync,
{
    type Output = SearchStats;

    fn search<O, PP, OB>(
        self,
        index: &'a DiskANNIndex<DP>,
        strategy: &'a S,
        processor: PP,
        context: &'a DP::Context,
        query: T,
        output: &mut OB,
    ) -> impl SendFuture<ANNResult<Self::Output>>
    where
        O: Send,
        PP: glue::SearchPostProcess<S::SearchAccessor, T, O> + Send + Sync,
        OB: SearchOutputBuffer<O> + Send + ?Sized,
    {
        async move {
            let mut accessor = strategy
                .search_accessor(&index.data_provider, context, query)
                .into_ann_result()?;
            let num_start_ids = accessor.num_starting_points().await?;

            let starting_l = self.starting_l().get();
            let mut scratch = index.search_scratch(starting_l, num_start_ids);

            let initial_stats = index
                .search_internal(
                    Some(self.beam_width().get()),
                    &mut accessor,
                    &mut scratch,
                    &mut NoopSearchRecord::new(),
                )
                .await?;

            let in_outer_range = InRange::new(self.radius(), None, starting_l, scratch.best.iter());

            // Increment the max results by the number of starting points, in case
            // they are filtered out later and leave us with fewer than the requested
            // number of results.
            let max_returned = self.effective_max_returned(num_start_ids);

            let mut in_range = InRange::new(
                self.radius(),
                self.inner_radius(),
                max_returned,
                in_outer_range.iter(),
            );

            let stats = if in_outer_range.len()
                >= ((starting_l as f32) * self.initial_slack()) as usize
                && in_outer_range.len() <= max_returned
            {
                // clear the visited set and repopulate it with just the in-range points
                scratch.visited.clear();
                scratch
                    .visited
                    .extend(in_outer_range.iter().map(|n| *n.id()));

                // Create a range frontier for seeding the second-round search
                let mut range_frontier: VecDeque<_> =
                    in_outer_range.take().into_iter().map(|n| *n.id()).collect();

                // Move to range search
                let range_stats = range_search_internal(
                    index.max_degree_with_slack(),
                    &self,
                    &mut accessor,
                    &mut scratch,
                    &mut range_frontier,
                    &mut in_range,
                )
                .await?;

                InternalSearchStats {
                    cmps: range_stats.cmps,
                    hops: range_stats.hops,
                    range_search_second_round: true,
                }
            } else {
                initial_stats
            };

            let result_count = processor
                .post_process(&mut accessor, query, in_range.iter(), output)
                .await
                .into_ann_result()?;

            Ok(SearchStats {
                cmps: stats.cmps,
                hops: stats.hops,
                result_count: result_count as u32,
                range_search_second_round: stats.range_search_second_round,
            })
        }
    }
}

pub(super) struct InRange<I> {
    neighbors: Vec<Neighbor<I>>,
    radius: f32,
    inner_radius: Option<f32>,
    max_returned: usize,
}

impl<I> InRange<I> {
    #[must_use]
    pub(super) fn push(&mut self, neighbor: Neighbor<I>) -> bool {
        let d = *neighbor.distance();
        if self.neighbors.len() < self.max_returned && self.check(d) {
            self.neighbors.push(neighbor);
            true
        } else {
            false
        }
    }

    pub(super) fn dedup(&mut self)
    where
        I: Ord,
    {
        self.neighbors
            .sort_unstable_by(crate::neighbor::ord::fast_distance_total);
        self.neighbors
            .dedup_by(|left, right| left.id() == right.id());
    }

    pub(super) fn len(&self) -> usize {
        self.neighbors.len()
    }

    pub(super) fn is_full(&self) -> bool {
        self.len() == self.max_returned
    }

    pub(super) fn take(self) -> Vec<Neighbor<I>> {
        self.neighbors
    }

    pub(super) fn iter(&self) -> impl ExactSizeIterator<Item = Neighbor<I>>
    where
        I: Copy,
    {
        self.neighbors.iter().copied()
    }

    #[must_use]
    pub(super) fn check(&self, distance: f32) -> bool {
        distance <= self.radius && self.inner_radius.map_or(true, |inner| distance > inner)
    }

    /// Create a new InRange with the given parameters,
    /// filtering the candidate neighbors and truncating
    /// as needed to respect the maximum number of results.
    pub(super) fn new<Itr>(
        radius: f32,
        inner_radius: Option<f32>,
        max_returned: usize,
        candidates: Itr,
    ) -> Self
    where
        Itr: IntoIterator<Item = Neighbor<I>>,
    {
        Self {
            neighbors: candidates
                .into_iter()
                .filter(|n| {
                    let dist = *n.distance();
                    if dist > radius {
                        return false;
                    }
                    if let Some(inner) = inner_radius
                        && dist <= inner
                    {
                        return false;
                    }
                    true
                })
                .take(max_returned)
                .collect(),
            radius,
            inner_radius,
            max_returned,
        }
    }
}

/////////////////////////////
// Internal Implementation //
/////////////////////////////

/// Internal range search implementation.
///
/// Expands the search frontier to find all points within the specified radius.
/// Called after the initial graph search has identified starting candidates.
pub(crate) async fn range_search_internal<A>(
    max_degree_with_slack: usize,
    search_params: &Range,
    accessor: &mut A,
    scratch: &mut SearchScratch<A::Id>,
    range_frontier: &mut VecDeque<A::Id>,
    in_range: &mut InRange<A::Id>,
) -> ANNResult<InternalSearchStats>
where
    A: SearchAccessor,
{
    let beam_width = search_params.beam_width().get();

    let mut neighbors = Vec::with_capacity(max_degree_with_slack);

    while !range_frontier.is_empty() && !in_range.is_full() {
        scratch.beam_nodes.clear();

        // In this loop we are going to find the beam_width number of remaining nodes within the radius
        // Each of these nodes will be a frontier node.
        while !range_frontier.is_empty() && scratch.beam_nodes.len() < beam_width {
            let next = range_frontier.pop_front();
            if let Some(next_node) = next {
                scratch.beam_nodes.push(next_node);
            }
        }

        neighbors.clear();
        accessor
            .expand_beam(
                scratch.beam_nodes.iter().copied(),
                glue::NotInMut::new(&mut scratch.visited),
                |id, distance| neighbors.push(Neighbor::new(id, distance)),
            )
            .await?;

        let navigation_radius = search_params.radius() * search_params.range_slack();
        for neighbor in neighbors.iter() {
            if in_range.is_full() {
                break;
            }

            if in_range.push(*neighbor) || *neighbor.distance() <= navigation_radius {
                range_frontier.push_back(*neighbor.id());
            }
        }

        scratch.cmps += neighbors.len() as u32;
        scratch.hops += scratch.beam_nodes.len() as u32;
    }

    Ok(InternalSearchStats {
        cmps: scratch.cmps,
        hops: scratch.hops,
        range_search_second_round: true,
    })
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn range_builder_defaults_match_new() {
        let from_new = Range::new(100, 0.5).unwrap();
        let from_builder = Range::builder(100, 0.5).build().unwrap();

        assert_eq!(from_builder.max_returned(), from_new.max_returned());
        assert_eq!(from_builder.starting_l(), from_new.starting_l());
        assert_eq!(from_builder.beam_width(), from_new.beam_width());
        assert_eq!(from_builder.radius(), from_new.radius());
        assert_eq!(from_builder.inner_radius(), from_new.inner_radius());
        assert_eq!(from_builder.initial_slack(), from_new.initial_slack());
        assert_eq!(from_builder.range_slack(), from_new.range_slack());
    }

    #[test]
    fn range_builder_custom_options_match_expected_values() {
        let built = Range::builder(100, 0.8)
            .max_returned(Some(101))
            .beam_width(Some(8))
            .inner_radius(Some(0.3))
            .initial_slack(0.9)
            .range_slack(1.2)
            .build()
            .unwrap();

        assert_eq!(built.max_returned(), Some(101));
        assert_eq!(built.starting_l().get(), 100);
        assert_eq!(built.beam_width().get(), 8);
        assert_eq!(built.radius(), 0.8);
        assert_eq!(built.inner_radius(), Some(0.3));
        assert_eq!(built.initial_slack(), 0.9);
        assert_eq!(built.range_slack(), 1.2);
    }

    #[test]
    fn range_builder_validation_error() {
        let err = Range::builder(100, 0.5)
            .beam_width(Some(0))
            .build()
            .unwrap_err();
        assert!(matches!(err, RangeSearchError::BeamWidthZero));
    }

    #[test]
    fn test_range_search_validation() {
        // Valid
        assert!(Range::new(100, 0.5).is_ok());

        // Invalid: zero l
        assert!(Range::new(0, 0.5).is_err());

        // Invalid slack values
        assert!(Range::builder(100, 0.5).initial_slack(1.5).build().is_err());
        assert!(Range::builder(100, 0.5).range_slack(0.5).build().is_err());

        // Invalid inner radius > radius
        assert!(
            Range::builder(100, 0.5)
                .inner_radius(Some(1.0))
                .build()
                .is_err()
        );

        assert!(
            Range::builder(100, 0.5)
                .max_returned(Some(1))
                .build()
                .is_err()
        );
    }
}
