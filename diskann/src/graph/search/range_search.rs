/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Range-based search within a distance radius.

use diskann_utils::future::{AssertSend, SendFuture};
use thiserror::Error;

use super::{Search, scratch::SearchScratch};
use crate::{
    ANNError, ANNErrorKind, ANNResult,
    error::IntoANNResult,
    graph::{
        glue::{self, ExpandBeam, SearchExt, SearchPostProcess, SearchStrategy},
        index::{DiskANNIndex, InternalSearchStats, SearchStats},
        search::record::NoopSearchRecord,
        search_output_buffer,
    },
    neighbor::Neighbor,
    provider::{BuildQueryComputer, DataProvider},
    utils::IntoUsize,
};

/// Result from a range search operation.
pub struct RangeSearchOutput<O> {
    /// Search statistics.
    pub stats: SearchStats,
    /// IDs of points within the radius.
    pub ids: Vec<O>,
    /// Distances corresponding to each ID.
    pub distances: Vec<f32>,
}

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

impl<DP, S, T, O> Search<DP, S, T, O, ()> for Range
where
    DP: DataProvider,
    T: Sync + ?Sized,
    S: SearchStrategy<DP, T, O>,
    O: Send + Default + Clone,
{
    type Output = RangeSearchOutput<O>;

    fn search(
        &mut self,
        index: &DiskANNIndex<DP>,
        strategy: &S,
        context: &DP::Context,
        query: &T,
        _output: &mut (),
    ) -> impl SendFuture<ANNResult<Self::Output>> {
        let search_params = *self;
        async move {
            let mut accessor = strategy
                .search_accessor(&index.data_provider, context)
                .into_ann_result()?;
            let computer = accessor.build_query_computer(query).into_ann_result()?;
            let start_ids = accessor.starting_points().await?;

            let mut scratch = index.search_scratch(search_params.starting_l(), start_ids.len());

            let initial_stats = index
                .search_internal(
                    search_params.beam_width(),
                    &start_ids,
                    &mut accessor,
                    &computer,
                    &mut scratch,
                    &mut NoopSearchRecord::new(),
                )
                .await?;

            let mut in_range = Vec::with_capacity(search_params.starting_l().into_usize());

            for neighbor in scratch
                .best
                .iter()
                .take(search_params.starting_l().into_usize())
            {
                if neighbor.distance <= search_params.radius() {
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
                >= ((search_params.starting_l() as f32) * search_params.initial_slack()) as usize
            {
                // Move to range search
                let range_stats = range_search_internal(
                    index.max_degree_with_slack(),
                    &search_params,
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

            // Post-process results
            let mut result_ids: Vec<O> = vec![O::default(); scratch.in_range.len()];
            let mut result_dists: Vec<f32> = vec![f32::MAX; scratch.in_range.len()];

            let mut output_buffer = search_output_buffer::IdDistance::new(
                result_ids.as_mut_slice(),
                result_dists.as_mut_slice(),
            );

            let _ = strategy
                .post_processor()
                .post_process(
                    &mut accessor,
                    query,
                    &computer,
                    scratch.in_range.iter().copied(),
                    &mut output_buffer,
                )
                .send()
                .await
                .into_ann_result()?;

            // Filter by inner/outer radius
            let inner_cutoff = if let Some(inner_radius) = search_params.inner_radius() {
                result_dists
                    .iter()
                    .position(|dist| *dist > inner_radius)
                    .unwrap_or(result_dists.len())
            } else {
                0
            };

            let outer_cutoff = result_dists
                .iter()
                .position(|dist| *dist > search_params.radius())
                .unwrap_or(result_dists.len());

            result_ids.truncate(outer_cutoff);
            result_ids.drain(0..inner_cutoff);

            result_dists.truncate(outer_cutoff);
            result_dists.drain(0..inner_cutoff);

            let result_count = result_ids.len();

            Ok(RangeSearchOutput {
                stats: SearchStats {
                    cmps: stats.cmps,
                    hops: stats.hops,
                    result_count: result_count as u32,
                    range_search_second_round: stats.range_search_second_round,
                },
                ids: result_ids,
                distances: result_dists,
            })
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
}
