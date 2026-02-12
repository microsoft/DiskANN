/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Range-based search within a distance radius.

use diskann_utils::future::{AssertSend, SendFuture};

use super::{dispatch::SearchDispatch, scratch::SearchScratch};
use crate::{
    ANNResult,
    error::IntoANNResult,
    graph::{
        RangeSearchParams,
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

/// Parameters for range-based search.
///
/// Finds all points within a specified distance radius from the query.
#[derive(Debug, Clone, Copy)]
pub struct RangeSearch {
    /// Maximum results to return (None = unlimited).
    pub max_returned: Option<usize>,
    /// Initial search list size.
    pub starting_l: usize,
    /// Optional beam width.
    pub beam_width: Option<usize>,
    /// Outer radius - points within this distance are candidates.
    pub radius: f32,
    /// Inner radius - points closer than this are excluded.
    pub inner_radius: Option<f32>,
    /// Slack factor for initial search phase (0.0 to 1.0).
    pub initial_slack: f32,
    /// Slack factor for range expansion (>= 1.0).
    pub range_slack: f32,
}

impl RangeSearch {
    /// Create range search with default slack values.
    pub fn new(
        starting_l: usize,
        radius: f32,
    ) -> Result<Self, super::super::RangeSearchParamsError> {
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
    ) -> Result<Self, super::super::RangeSearchParamsError> {
        use super::super::RangeSearchParamsError;

        if let Some(bw) = beam_width {
            if bw == 0 {
                return Err(RangeSearchParamsError::BeamWidthZero);
            }
        }
        if starting_l == 0 {
            return Err(RangeSearchParamsError::LZero);
        }
        if !(0.0..=1.0).contains(&initial_slack) {
            return Err(RangeSearchParamsError::StartingListSlackValueError);
        }
        if range_slack < 1.0 {
            return Err(RangeSearchParamsError::RangeSearchSlackValueError);
        }
        if let Some(inner) = inner_radius {
            if inner > radius {
                return Err(RangeSearchParamsError::InnerRadiusValueError);
            }
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

    fn to_legacy_params(&self) -> RangeSearchParams {
        RangeSearchParams {
            max_returned: self.max_returned,
            starting_l_value: self.starting_l,
            beam_width: self.beam_width,
            radius: self.radius,
            inner_radius: self.inner_radius,
            initial_search_slack: self.initial_slack,
            range_search_slack: self.range_slack,
        }
    }
}

impl From<RangeSearchParams> for RangeSearch {
    fn from(params: RangeSearchParams) -> Self {
        Self {
            max_returned: params.max_returned,
            starting_l: params.starting_l_value,
            beam_width: params.beam_width,
            radius: params.radius,
            inner_radius: params.inner_radius,
            initial_slack: params.initial_search_slack,
            range_slack: params.range_search_slack,
        }
    }
}

impl<DP, S, T, O> SearchDispatch<DP, S, T, O, ()> for RangeSearch
where
    DP: DataProvider,
    T: Sync + ?Sized,
    S: SearchStrategy<DP, T, O>,
    O: Send + Default + Clone,
{
    type Output = RangeSearchOutput<O>;

    fn dispatch<'a>(
        &'a mut self,
        index: &'a DiskANNIndex<DP>,
        strategy: &'a S,
        context: &'a DP::Context,
        query: &'a T,
        _output: &'a mut (),
    ) -> impl SendFuture<ANNResult<Self::Output>> {
        let search_params = self.to_legacy_params();
        async move {
            let mut accessor = strategy
                .search_accessor(&index.data_provider, context)
                .into_ann_result()?;
            let computer = accessor.build_query_computer(query).into_ann_result()?;
            let start_ids = accessor.starting_points().await?;

            let mut scratch = index.search_scratch(search_params.starting_l_value, start_ids.len());

            let initial_stats = index
                .search_internal(
                    search_params.beam_width,
                    &start_ids,
                    &mut accessor,
                    &computer,
                    &mut scratch,
                    &mut NoopSearchRecord::new(),
                )
                .await?;

            let mut in_range = Vec::with_capacity(search_params.starting_l_value.into_usize());

            for neighbor in scratch
                .best
                .iter()
                .take(search_params.starting_l_value.into_usize())
            {
                if neighbor.distance <= search_params.radius {
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
                >= ((search_params.starting_l_value as f32) * search_params.initial_search_slack)
                    as usize
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
            let inner_cutoff = if let Some(inner_radius) = search_params.inner_radius {
                result_dists
                    .iter()
                    .position(|dist| *dist > inner_radius)
                    .unwrap_or(result_dists.len())
            } else {
                0
            };

            let outer_cutoff = result_dists
                .iter()
                .position(|dist| *dist > search_params.radius)
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

//=============================================================================
// Internal Implementation
//=============================================================================

/// Internal range search implementation.
///
/// Expands the search frontier to find all points within the specified radius.
/// Called after the initial graph search has identified starting candidates.
pub(crate) async fn range_search_internal<I, A, T>(
    max_degree_with_slack: usize,
    search_params: &RangeSearchParams,
    accessor: &mut A,
    computer: &A::QueryComputer,
    scratch: &mut SearchScratch<I>,
) -> ANNResult<InternalSearchStats>
where
    I: crate::utils::VectorId,
    A: ExpandBeam<T, Id = I> + SearchExt,
    T: ?Sized,
{
    let beam_width = search_params.beam_width.unwrap_or(1);

    for neighbor in &scratch.in_range {
        scratch.range_frontier.push_back(neighbor.id);
    }

    let mut neighbors = Vec::with_capacity(max_degree_with_slack);

    let max_returned = search_params.max_returned.unwrap_or(usize::MAX);

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
            if neighbor.distance <= search_params.radius * search_params.range_search_slack
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

//=============================================================================
// Tests
//=============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_range_search_validation() {
        // Valid
        assert!(RangeSearch::new(100, 0.5).is_ok());

        // Invalid: zero l
        assert!(RangeSearch::new(0, 0.5).is_err());

        // Invalid slack values
        assert!(RangeSearch::with_options(None, 100, None, 0.5, None, 1.5, 1.0).is_err());
        assert!(RangeSearch::with_options(None, 100, None, 0.5, None, 1.0, 0.5).is_err());

        // Invalid inner radius > radius
        assert!(RangeSearch::with_options(None, 100, None, 0.5, Some(1.0), 1.0, 1.0).is_err());
    }
}
