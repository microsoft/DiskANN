/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Graph-degree enforcement with the Vamana RobustPrune kernel.
//!
//! Candidate merging can produce more than the graph degree for one point.
//! Lists within the degree limit pass through unchanged. Longer lists are sorted
//! by source distance and passed to RobustPrune, then overwritten with the
//! selected IDs. Reusable per-job scratch stores candidates and pruning state.
//!
//! RobustPrune owns occlusion and alpha-round behavior. This module supplies
//! source vectors and metric distances.

use crate::{
    ANNError, ANNResult,
    graph::{
        AdjacencyList, Config,
        internal::{SortedNeighbors, prune},
    },
    neighbor::Neighbor,
    utils::VectorRepr,
};
use diskann_utils::views::MatrixView;
use diskann_vector::{DistanceFunction, distance::Metric};
use rayon::prelude::*;

#[derive(Debug, thiserror::Error)]
pub(crate) enum FinalizationError {
    #[error("candidate count {actual} exceeds the u16 position limit {max}")]
    TooManyCandidates { actual: usize, max: usize },
}

/// RobustPrune state for one Rayon job.
#[derive(Default)]
struct PruneWorkspace {
    candidates: Vec<Neighbor<u32>>,
    mapped_candidates: Vec<Neighbor<Option<u32>>>,
    states: Vec<prune::State>,
}

impl PruneWorkspace {
    fn prune<T: VectorRepr>(
        &mut self,
        data: MatrixView<'_, T>,
        source: usize,
        mut source_candidates: AdjacencyList<u32>,
        graph: &Config,
        distance: &T::Distance,
    ) -> ANNResult<AdjacencyList<u32>> {
        let degree = graph.pruned_degree().get();
        let candidate_count = source_candidates.len();
        if candidate_count <= degree {
            return Ok(source_candidates);
        }
        if candidate_count > u16::MAX as usize {
            return Err(ANNError::new(FinalizationError::TooManyCandidates {
                actual: candidate_count,
                max: u16::MAX as usize,
            }));
        }

        let source_vector = data.row(source);
        self.candidates.clear();
        self.candidates
            .extend(source_candidates.iter().copied().map(|candidate| {
                Neighbor::new(
                    candidate,
                    distance.evaluate_similarity(source_vector, data.row(candidate as usize)),
                )
            }));
        // Sort before mapping: changing the element type can change the unstable
        // sort's order for equal distances, and therefore RobustPrune's choices.
        let sorted = SortedNeighbors::new(&mut self.candidates, candidate_count);
        let mapped = sorted.map_in(&mut self.mapped_candidates, |&candidate| {
            (candidate as usize != source).then_some(candidate)
        });
        self.states.clear();
        self.states.resize(candidate_count, prune::State::default());
        let selected = prune::robust_prune(
            mapped,
            &mut self.states,
            degree,
            graph.alpha(),
            graph.prune_kind(),
            |left, right| {
                distance.evaluate_similarity(data.row(*left as usize), data.row(*right as usize))
            },
        );

        let mut output = source_candidates.resize(selected);
        for (destination, state) in output.iter_mut().zip(&self.states) {
            *destination = *sorted[state.neighbor as usize].id();
        }
        output.finish(selected);
        Ok(source_candidates)
    }
}

/// Prune each candidate list that exceeds the graph degree.
///
/// Candidate builders supply one list per data row and valid dataset IDs.
pub(crate) fn prune_overfull<T: VectorRepr>(
    data: MatrixView<'_, T>,
    candidates: Vec<AdjacencyList<u32>>,
    graph: &Config,
    metric: Metric,
) -> ANNResult<Vec<AdjacencyList<u32>>> {
    let distance = T::distance(metric, Some(data.ncols()));

    // The build context supplies the Rayon pool for this terminal operation.
    #[allow(clippy::disallowed_methods)]
    candidates
        .into_par_iter()
        .enumerate()
        .map_init(
            PruneWorkspace::default,
            |workspace, (source, candidates)| {
                workspace.prune(data, source, candidates, graph, &distance)
            },
        )
        .collect()
}

#[cfg(test)]
mod prune_tests {
    use crate::graph::config::{self, MaxDegree};
    use rstest::rstest;

    use super::*;

    fn graph_config(degree: usize) -> Config {
        config::Builder::new_with(
            degree,
            MaxDegree::same(),
            degree,
            Metric::L2.into(),
            |builder| {
                builder.alpha(1.2);
            },
        )
        .build()
        .unwrap()
    }

    #[rstest]
    #[case::empty(&[])]
    #[case::below_degree(&[3])]
    #[case::at_degree(&[3, 1])]
    fn rows_within_degree_are_unchanged(#[case] ids: &[u32]) {
        // Given: preserve the caller's order, including a non-sorted row.
        let values = [0.0_f32, 1.0, 2.0, 3.0];
        let data = MatrixView::column_vector(&values[..]);
        let mut candidates = AdjacencyList::new();
        candidates.overwrite_trusted(ids);
        let distance = f32::distance(Metric::L2, Some(1));
        let mut workspace = PruneWorkspace::default();

        // When
        let actual = workspace
            .prune(data, 0, candidates, &graph_config(2), &distance)
            .unwrap();

        // Then
        assert_eq!(&*actual, ids);
    }

    #[test]
    fn overfull_rows_keep_nearest_unoccluded_neighbors_at_their_source_id() {
        // Given: source 1 is at 0. Its squared distances to IDs 2, 0, 3 are
        // 1, 4, 9. ID 2 occludes ID 0 because 4 / (2 - 1)^2 = 4 > alpha.
        // ID 3 survives because 9 / (-3 - 1)^2 = 9/16 <= 1.
        let values = [2.0_f32, 0.0, 1.0, -3.0];
        let data = MatrixView::column_vector(&values[..]);
        let candidates = vec![
            AdjacencyList::new(),
            AdjacencyList::from_iter_untrusted([0, 2, 3]),
            AdjacencyList::new(),
            AdjacencyList::new(),
        ];
        let expected = [vec![], vec![2, 3], vec![], vec![]];

        // When
        let actual: Vec<Vec<u32>> = prune_overfull(data, candidates, &graph_config(2), Metric::L2)
            .unwrap()
            .into_iter()
            .map(Vec::from)
            .collect();

        // Then
        assert_eq!(actual, expected);
    }

    #[test]
    fn self_candidates_do_not_shift_selected_neighbor_ids() {
        // Given: self occupies the first sorted position but cannot be selected.
        // Of the other points, 1 occludes 2, while -3 remains on the opposite ray.
        let values = [0.0_f32, 1.0, 2.0, -3.0];
        let data = MatrixView::column_vector(&values[..]);
        let candidates = AdjacencyList::from_iter_untrusted([0, 1, 2, 3]);
        let distance = f32::distance(Metric::L2, Some(1));
        let expected = [1, 3];
        let mut workspace = PruneWorkspace::default();

        // When
        let actual = workspace
            .prune(data, 0, candidates, &graph_config(2), &distance)
            .unwrap();

        // Then
        assert_eq!(&*actual, &expected);
    }

    #[test]
    fn reused_workspace_clears_previous_candidates_and_occlusion_states() {
        // Given: the same workspace prunes four candidates, then three, then four.
        let values = [0.0_f32, 1.0, 2.0, -3.0, 4.0];
        let data = MatrixView::column_vector(&values[..]);
        let graph = graph_config(2);
        let distance = f32::distance(Metric::L2, Some(1));
        let mut workspace = PruneWorkspace::default();
        workspace
            .prune(
                data,
                0,
                AdjacencyList::from_iter_untrusted([1, 2, 3, 4]),
                &graph,
                &distance,
            )
            .unwrap();
        workspace
            .prune(
                data,
                4,
                AdjacencyList::from_iter_untrusted([0, 1, 3]),
                &graph,
                &distance,
            )
            .unwrap();
        // For source 2 at coordinate 2, ID 1 at 1 is nearest. It occludes ID 0
        // (4/1 > alpha); ID 4 at 4 survives (4/9 <= 1).
        let expected = [1, 4];
        let candidates = AdjacencyList::from_iter_untrusted([0, 1, 3, 4]);

        // When
        let actual = workspace
            .prune(data, 2, candidates, &graph, &distance)
            .unwrap();

        // Then
        assert_eq!(&*actual, &expected);
    }

    #[test]
    fn maximum_u16_candidate_count_can_be_pruned() {
        // Given: all candidates lie on the positive ray, so ID 1 is nearest.
        let count = u16::MAX as usize;
        let values: Vec<f32> = (0..=count).map(|id| id as f32).collect();
        let data = MatrixView::column_vector(&values[..]);
        let candidates = AdjacencyList::from_iter_untrusted(1..=count as u32);
        let distance = f32::distance(Metric::L2, Some(1));
        let mut workspace = PruneWorkspace::default();

        // When
        let actual = workspace
            .prune(data, 0, candidates, &graph_config(1), &distance)
            .unwrap();

        // Then
        assert_eq!(&*actual, &[1]);
    }

    #[test]
    fn candidate_count_above_u16_limit_is_rejected() {
        // Given: the list is one entry longer than RobustPrune's position limit.
        let count = u16::MAX as usize + 1;
        let values = vec![0.0_f32; count + 1];
        let data = MatrixView::column_vector(&values[..]);
        let candidates = AdjacencyList::from_iter_untrusted(1..=count as u32);
        let distance = f32::distance(Metric::L2, Some(1));
        let mut workspace = PruneWorkspace::default();

        // When
        let error = workspace
            .prune(data, 0, candidates, &graph_config(1), &distance)
            .unwrap_err();

        // Then
        assert!(matches!(
            error.downcast_ref::<FinalizationError>(),
            Some(FinalizationError::TooManyCandidates { actual, max })
                if *actual == count && *max == u16::MAX as usize
        ));
    }
}
