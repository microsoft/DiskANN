/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Range-based search within a distance radius.

use diskann_utils::future::SendFuture;
use thiserror::Error;

use super::{Search, scratch::SearchScratch};
use crate::{
    ANNError, ANNErrorKind, ANNResult,
    error::IntoANNResult,
    graph::{
        glue::{self, ExpandBeam, SearchExt},
        index::{DiskANNIndex, InternalSearchStats, SearchStats},
        search::record::NoopSearchRecord,
        search_output_buffer::{self, SearchOutputBuffer},
    },
    neighbor::Neighbor,
    provider::{BuildQueryComputer, DataProvider},
    utils::IntoUsize,
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
}

impl From<RangeSearchError> for ANNError {
    #[track_caller]
    fn from(err: RangeSearchError) -> Self {
        Self::new(ANNErrorKind::IndexError, err)
    }
}

/// Parameters for range-based search.
///
/// Finds all points within a specified distance radius from the query.
#[derive(Debug, Clone, Copy)]
pub struct Range {
    /// Maximum results to return (None = unlimited).
    max_returned: Option<usize>,
    /// Initial search list size.
    starting_l: usize,
    /// Optional beam width.
    beam_width: Option<usize>,
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
        Self::with_options(None, starting_l, None, radius, None, 1.0, 1.0)
    }

    /// Create range search with full options.
    #[allow(clippy::too_many_arguments)]
    pub fn with_options(
        max_returned: Option<usize>,
        starting_l: usize,
        beam_width: Option<usize>,
        radius: f32,
        inner_radius: Option<f32>,
        initial_slack: f32,
        range_slack: f32,
    ) -> Result<Self, RangeSearchError> {
        if let Some(bw) = beam_width
            && bw == 0
        {
            return Err(RangeSearchError::BeamWidthZero);
        }
        if starting_l == 0 {
            return Err(RangeSearchError::LZero);
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

    /// Returns the initial search list size.
    #[inline]
    pub fn starting_l(&self) -> usize {
        self.starting_l
    }

    /// Returns the optional beam width.
    #[inline]
    pub fn beam_width(&self) -> Option<usize> {
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
}

impl<DP, S, T> Search<DP, S, T> for Range
where
    DP: DataProvider,
    S: glue::SearchStrategy<DP, T>,
    T: Sync + ?Sized,
{
    type Output = SearchStats;

    fn search<O, PP, OB>(
        self,
        index: &DiskANNIndex<DP>,
        strategy: &S,
        processor: PP,
        context: &DP::Context,
        query: &T,
        output: &mut OB,
    ) -> impl SendFuture<ANNResult<Self::Output>>
    where
        O: Send,
        PP: for<'a> glue::SearchPostProcess<S::SearchAccessor<'a>, T, O> + Send + Sync,
        OB: SearchOutputBuffer<O> + Send + ?Sized,
    {
        async move {
            let mut accessor = strategy
                .search_accessor(&index.data_provider, context)
                .into_ann_result()?;
            let computer = accessor.build_query_computer(query).into_ann_result()?;
            let start_ids = accessor.starting_points().await?;

            let mut scratch = index.search_scratch(self.starting_l(), start_ids.len());

            let initial_stats = index
                .search_internal(
                    self.beam_width(),
                    &start_ids,
                    &mut accessor,
                    &computer,
                    &mut scratch,
                    &mut NoopSearchRecord::new(),
                )
                .await?;

            let mut in_range = Vec::with_capacity(self.starting_l().into_usize());

            for neighbor in scratch.best.iter().take(self.starting_l().into_usize()) {
                if neighbor.distance <= self.radius() {
                    in_range.push(neighbor);
                }
            }

            // clear the visited set and repopulate it with just the in-range points
            scratch.visited.clear();
            for neighbor in in_range.iter() {
                scratch.visited.insert(neighbor.id);
            }
            scratch.in_range = in_range;

            let stats = if scratch.in_range.len()
                >= ((self.starting_l() as f32) * self.initial_slack()) as usize
            {
                // Move to range search
                let range_stats = range_search_internal(
                    index.max_degree_with_slack(),
                    &self,
                    &mut accessor,
                    &computer,
                    &mut scratch,
                )
                .await?;

                InternalSearchStats {
                    cmps: initial_stats.cmps,
                    hops: initial_stats.hops + range_stats.hops,
                    range_search_second_round: true,
                }
            } else {
                initial_stats
            };

            // Post-process results directly into the output buffer, filtering by radius.
            let radius = self.radius();
            let inner_radius = self.inner_radius();
            let mut filtered = DistanceFiltered::new(output, |dist| {
                if let Some(ir) = inner_radius
                    && dist <= ir
                {
                    return false;
                }
                dist <= radius
            });

            let result_count = processor
                .post_process(
                    &mut accessor,
                    query,
                    &computer,
                    scratch.in_range.iter().copied(),
                    &mut filtered,
                )
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

/// A [`SearchOutputBuffer`] wrapper that filters results by distance before
/// forwarding them to an inner buffer.
struct DistanceFiltered<'a, F, B: ?Sized> {
    predicate: F,
    inner: &'a mut B,
}

impl<'a, F, B: ?Sized> DistanceFiltered<'a, F, B> {
    fn new(inner: &'a mut B, predicate: F) -> Self {
        Self { predicate, inner }
    }
}

impl<I, F, B> SearchOutputBuffer<I> for DistanceFiltered<'_, F, B>
where
    F: FnMut(f32) -> bool,
    B: SearchOutputBuffer<I> + ?Sized,
{
    fn size_hint(&self) -> Option<usize> {
        self.inner.size_hint()
    }

    fn push(&mut self, id: I, distance: f32) -> search_output_buffer::BufferState {
        if (self.predicate)(distance) {
            self.inner.push(id, distance)
        } else {
            match self.inner.size_hint() {
                Some(0) => search_output_buffer::BufferState::Full,
                _ => search_output_buffer::BufferState::Available,
            }
        }
    }

    fn current_len(&self) -> usize {
        self.inner.current_len()
    }

    fn extend<Itr>(&mut self, itr: Itr) -> usize
    where
        Itr: IntoIterator<Item = (I, f32)>,
    {
        self.inner
            .extend(itr.into_iter().filter(|(_, dist)| (self.predicate)(*dist)))
    }
}

/////////////////////////////
// Internal Implementation //
/////////////////////////////

/// Internal range search implementation.
///
/// Expands the search frontier to find all points within the specified radius.
/// Called after the initial graph search has identified starting candidates.
pub(crate) async fn range_search_internal<I, A, T>(
    max_degree_with_slack: usize,
    search_params: &Range,
    accessor: &mut A,
    computer: &A::QueryComputer,
    scratch: &mut SearchScratch<I>,
) -> ANNResult<InternalSearchStats>
where
    I: crate::utils::VectorId,
    A: ExpandBeam<T, Id = I> + SearchExt,
    T: ?Sized,
{
    let beam_width = search_params.beam_width().unwrap_or(1);

    for neighbor in &scratch.in_range {
        scratch.range_frontier.push_back(neighbor.id);
    }

    let mut neighbors = Vec::with_capacity(max_degree_with_slack);

    let max_returned = search_params.max_returned().unwrap_or(usize::MAX);

    while !scratch.range_frontier.is_empty() {
        scratch.beam_nodes.clear();

        // In this loop we are going to find the beam_width number of remaining nodes within the radius
        // Each of these nodes will be a frontier node.
        while !scratch.range_frontier.is_empty() && scratch.beam_nodes.len() < beam_width {
            let next = scratch.range_frontier.pop_front();
            if let Some(next_node) = next {
                scratch.beam_nodes.push(next_node);
            }
        }

        neighbors.clear();
        accessor
            .expand_beam(
                scratch.beam_nodes.iter().copied(),
                computer,
                glue::NotInMut::new(&mut scratch.visited),
                |distance, id| neighbors.push(Neighbor::new(id, distance)),
            )
            .await?;

        // The predicate ensures that the contents of `neighbors` are unique.
        for neighbor in neighbors.iter() {
            if neighbor.distance <= search_params.radius() * search_params.range_slack()
                && scratch.in_range.len() < max_returned
            {
                scratch.in_range.push(*neighbor);
                scratch.range_frontier.push_back(neighbor.id);
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
    use crate::graph::search_output_buffer::BufferState;
    use crate::neighbor::Neighbor;

    #[test]
    fn test_range_search_validation() {
        // Valid
        assert!(Range::new(100, 0.5).is_ok());

        // Invalid: zero l
        assert!(Range::new(0, 0.5).is_err());

        // Invalid slack values
        assert!(Range::with_options(None, 100, None, 0.5, None, 1.5, 1.0).is_err());
        assert!(Range::with_options(None, 100, None, 0.5, None, 1.0, 0.5).is_err());

        // Invalid inner radius > radius
        assert!(Range::with_options(None, 100, None, 0.5, Some(1.0), 1.0, 1.0).is_err());
    }

    #[test]
    fn distance_filtered_push_accepts_passing_items() {
        let mut inner: Vec<Neighbor<u32>> = Vec::new();
        let mut filtered = DistanceFiltered::new(&mut inner, |d| d < 1.0);

        assert_eq!(filtered.push(1, 0.5), BufferState::Available);
        assert_eq!(filtered.current_len(), 1);
        assert_eq!(inner[0].id, 1);
        assert_eq!(inner[0].distance, 0.5);
    }

    #[test]
    fn distance_filtered_push_rejects_failing_items() {
        let mut inner: Vec<Neighbor<u32>> = Vec::new();
        let mut filtered = DistanceFiltered::new(&mut inner, |d| d < 1.0);

        assert_eq!(filtered.push(1, 1.5), BufferState::Available);
        assert_eq!(filtered.current_len(), 0);
    }

    #[test]
    fn distance_filtered_extend_filters_correctly() {
        let mut inner: Vec<Neighbor<u32>> = Vec::new();
        let mut filtered = DistanceFiltered::new(&mut inner, |d| d < 1.0);
        assert!(filtered.size_hint().is_none());

        let items = vec![(1u32, 0.3), (2, 1.5), (3, 0.7), (4, 2.0), (5, 0.9)];
        let count = filtered.extend(items);

        assert_eq!(count, 3);
        assert_eq!(inner.len(), 3);
        assert_eq!(inner[0].id, 1);
        assert_eq!(inner[1].id, 3);
        assert_eq!(inner[2].id, 5);
    }

    #[test]
    fn distance_filtered_respects_inner_capacity() {
        let mut ids = [0u32; 2];
        let mut dists = [0.0f32; 2];
        let mut inner = search_output_buffer::IdDistance::new(&mut ids, &mut dists);
        let mut filtered = DistanceFiltered::new(&mut inner, |d| d < 1.0);
        assert_eq!(filtered.size_hint(), Some(2));

        let items = vec![(1u32, 0.1), (2, 0.2), (3, 0.3)];
        let count = filtered.extend(items);

        assert_eq!(count, 2);
        assert_eq!(ids, [1, 2]);
    }

    #[test]
    fn distance_filtered_inner_radius_pattern() {
        let mut inner: Vec<Neighbor<u32>> = Vec::new();
        let radius = 1.0f32;
        let inner_radius = Some(0.3f32);
        let mut filtered = DistanceFiltered::new(&mut inner, |dist| {
            if let Some(ir) = inner_radius
                && dist <= ir
            {
                return false;
            }
            dist < radius
        });

        let items = vec![(1u32, 0.1), (2, 0.5), (3, 0.3), (4, 1.0), (5, 0.8)];
        let count = filtered.extend(items);

        // 0.1 and 0.3 are <= inner_radius, 1.0 is not < radius
        assert_eq!(count, 2);
        assert_eq!(inner[0].id, 2);
        assert_eq!(inner[1].id, 5);
    }
}
