/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Graph-degree enforcement with the Vamana RobustPrune kernel.
//!
//! Candidate merging can produce more than `R` IDs for one point. This module
//! checks every global ID before parallel work starts. A list at or below `R`
//! returns without distance calculations.
//!
//! For a longer list, the module computes each source distance. It sorts the
//! candidates and calls RobustPrune. The module then writes the selected IDs into
//! the original list allocation.
//!
//! RobustPrune defines occlusion and alpha-round behavior. This module supplies
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
    #[error("candidate list count {lists} does not match the dataset point count {points}")]
    CandidateListCountMismatch { lists: usize, points: usize },
    #[error(
        "candidate ID {candidate} for source {source_index} is outside a {points}-point dataset"
    )]
    InvalidCandidateId {
        source_index: usize,
        candidate: u32,
        points: usize,
    },
    #[error("candidate count {actual} exceeds the u16 position limit {max}")]
    TooManyCandidates { actual: usize, max: usize },
}

/// RobustPrune state for one Rayon job.
///
/// `candidate_slots` and `prune_states` stay positionally aligned with
/// `sorted_candidates`.
#[derive(Default)]
struct PruneWorkspace {
    sorted_candidates: Vec<Neighbor<u32>>,
    candidate_slots: Vec<(f32, Option<u32>)>,
    prune_states: Vec<prune::State>,
}

/// Check candidate IDs and prune each list that exceeds the graph degree.
pub(crate) fn prune_overfull<T>(
    data: MatrixView<'_, T>,
    candidates: Vec<AdjacencyList<u32>>,
    graph: &Config,
    metric: Metric,
) -> ANNResult<Vec<AdjacencyList<u32>>>
where
    T: VectorRepr + Send + Sync,
{
    validate_candidate_lists(&candidates, data.nrows()).map_err(ANNError::new)?;

    let degree = graph.pruned_degree().get();
    let distance = T::distance(metric, Some(data.ncols()));

    // `build_graph` runs this Rayon operation in the pool from the build context.
    #[allow(clippy::disallowed_methods)]
    candidates
        .into_par_iter()
        .enumerate()
        .map_init(
            PruneWorkspace::default,
            |workspace, (source, mut source_candidates)| {
                // Candidate merging already removes duplicate IDs. A list within
                // the degree limit needs no distance calculation.
                if source_candidates.len() <= degree {
                    return Ok(source_candidates);
                }

                let source_id = u32::try_from(source).map_err(ANNError::new)?;
                let source_vector = data.row(source);
                workspace.sorted_candidates.clear();
                workspace
                    .sorted_candidates
                    .try_reserve(source_candidates.len())
                    .map_err(ANNError::new)?;
                workspace
                    .sorted_candidates
                    .extend(source_candidates.iter().copied().map(|candidate| {
                        Neighbor::new(
                            candidate,
                            distance
                                .evaluate_similarity(source_vector, data.row(candidate as usize)),
                        )
                    }));

                let candidate_count = workspace.sorted_candidates.len();
                if candidate_count > u16::MAX as usize {
                    return Err(ANNError::new(FinalizationError::TooManyCandidates {
                        actual: candidate_count,
                        max: u16::MAX as usize,
                    }));
                }
                workspace.candidate_slots.clear();
                workspace
                    .candidate_slots
                    .try_reserve(candidate_count)
                    .map_err(ANNError::new)?;

                // Sort all candidates before the code marks a self-edge as absent.
                // Thus, self-edge removal cannot add a farther candidate. The
                // `SortedNeighbors` value carries this order into RobustPrune.
                let sorted =
                    SortedNeighbors::new(&mut workspace.sorted_candidates, candidate_count);
                workspace
                    .candidate_slots
                    .extend(sorted.iter().map(|neighbor| {
                        let id = *neighbor.id();
                        (*neighbor.distance(), (id != source_id).then_some(id))
                    }));
                workspace
                    .prune_states
                    .try_reserve(
                        workspace
                            .candidate_slots
                            .len()
                            .saturating_sub(workspace.prune_states.len()),
                    )
                    .map_err(ANNError::new)?;
                workspace
                    .prune_states
                    .resize(workspace.candidate_slots.len(), prune::State::default());
                // Each candidate list starts a separate RobustPrune state machine.
                // Reset retained entries because resize initializes only new entries.
                workspace.prune_states.fill(prune::State::default());

                let selected = prune::robust_prune(
                    &sorted,
                    &workspace.candidate_slots,
                    workspace.prune_states.as_mut_slice(),
                    degree,
                    graph.alpha(),
                    graph.prune_kind(),
                    |left, right| {
                        distance.evaluate_similarity(
                            data.row(*left as usize),
                            data.row(*right as usize),
                        )
                    },
                );

                let mut guard = source_candidates.resize(selected);
                for (destination, state) in guard.iter_mut().zip(workspace.prune_states.iter()) {
                    *destination = *sorted[state.neighbor as usize].id();
                }
                guard.finish(selected);
                Ok(source_candidates)
            },
        )
        .collect()
}

fn validate_candidate_lists(
    candidates: &[AdjacencyList<u32>],
    points: usize,
) -> Result<(), FinalizationError> {
    if candidates.len() != points {
        return Err(FinalizationError::CandidateListCountMismatch {
            lists: candidates.len(),
            points,
        });
    }
    for (source, source_candidates) in candidates.iter().enumerate() {
        if let Some(&candidate) = source_candidates.iter().find(|&&id| id as usize >= points) {
            return Err(FinalizationError::InvalidCandidateId {
                source_index: source,
                candidate,
                points,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::graph::{
        AdjacencyList,
        config::{self, MaxDegree},
    };
    use diskann_utils::views::MatrixView;

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

    fn candidate_list(ids: impl IntoIterator<Item = u32>) -> AdjacencyList<u32> {
        AdjacencyList::from_iter_untrusted(ids)
    }

    #[test]
    fn preserves_lists_within_the_degree_bound() {
        let data = [0.0_f32, 1.0, 2.0, 3.0];
        let data = MatrixView::try_from(&data[..], 4, 1).unwrap();
        let candidates = vec![
            candidate_list([3, 1]),
            candidate_list([]),
            candidate_list([]),
            candidate_list([]),
        ];

        let actual = prune_overfull(data, candidates, &graph_config(2), Metric::L2).unwrap();

        assert_eq!(&*actual[0], &[1, 3]);
    }

    #[test]
    fn prunes_an_overfull_list_with_the_vamana_kernel() {
        let data = [0.0_f32, 1.0, 2.0, -3.0];
        let data = MatrixView::try_from(&data[..], 4, 1).unwrap();
        let candidates = vec![
            candidate_list([3, 2, 1]),
            candidate_list([]),
            candidate_list([]),
            candidate_list([]),
        ];

        let actual = prune_overfull(data, candidates, &graph_config(2), Metric::L2).unwrap();

        assert_eq!(&*actual[0], &[1, 3]);
    }

    #[test]
    fn reused_workspace_matches_fresh_pruning() {
        let data = [0.0_f32, 1.0, 2.0, -3.0, 4.0];
        let data = MatrixView::try_from(&data[..], 5, 1).unwrap();
        let first = [3, 2, 1];
        let second = [4, 3, 2];
        let candidates = |first: &[u32], second: &[u32]| {
            vec![
                candidate_list(first.iter().copied()),
                candidate_list(second.iter().copied()),
                candidate_list([]),
                candidate_list([]),
                candidate_list([]),
            ]
        };
        let graph = graph_config(2);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let fresh_first = pool
            .install(|| prune_overfull(data, candidates(&first, &[]), &graph, Metric::L2))
            .unwrap();
        let fresh_second = pool
            .install(|| prune_overfull(data, candidates(&[], &second), &graph, Metric::L2))
            .unwrap();

        let reused = pool
            .install(|| prune_overfull(data, candidates(&first, &second), &graph, Metric::L2))
            .unwrap();

        assert_eq!(&*reused[0], &*fresh_first[0]);
        assert_eq!(&*reused[1], &*fresh_second[1]);
    }

    #[test]
    fn rejects_invalid_candidate_ids_without_panicking() {
        let data = [0.0_f32, 1.0, 2.0];
        let data = MatrixView::try_from(&data[..], 3, 1).unwrap();
        let candidates = vec![
            candidate_list([1, 3]),
            candidate_list([]),
            candidate_list([]),
        ];

        let error = prune_overfull(data, candidates, &graph_config(1), Metric::L2).unwrap_err();

        assert!(matches!(
            error.downcast_ref::<FinalizationError>(),
            Some(FinalizationError::InvalidCandidateId {
                source_index: 0,
                candidate: 3,
                points: 3,
            })
        ));
    }

    #[test]
    fn rejects_candidate_list_count_mismatch_without_panicking() {
        let data = [0.0_f32, 1.0, 2.0];
        let data = MatrixView::try_from(&data[..], 3, 1).unwrap();
        let candidates = vec![
            candidate_list([]),
            candidate_list([]),
            candidate_list([]),
            candidate_list([]),
        ];

        let error = prune_overfull(data, candidates, &graph_config(1), Metric::L2).unwrap_err();

        assert!(matches!(
            error.downcast_ref::<FinalizationError>(),
            Some(FinalizationError::CandidateListCountMismatch {
                lists: 4,
                points: 3
            })
        ));
    }
}
