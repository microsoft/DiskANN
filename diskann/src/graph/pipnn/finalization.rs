/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Graph-degree enforcement with the Vamana RobustPrune kernel.
//!
//! Candidate merging can produce more than `R` IDs for one point. This module
//! checks every global ID before parallel work starts. It returns a list with at
//! most `R` IDs without distance work.
//!
//! For a longer list, the module computes each source distance. It sorts the
//! candidates and calls RobustPrune. The module then writes the selected IDs into
//! the original list allocation.
//!
//! RobustPrune defines occlusion and alpha-round behavior. This module supplies
//! contiguous vector access and a distance function for input type `T`.

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

/// Reusable buffers for one Rayon job.
///
/// `pool` stores candidates with source distances. `prepared` stores each
/// distance and optional non-self ID. `states` stores one RobustPrune state for
/// each sorted candidate.
#[derive(Default)]
struct Workspace {
    pool: Vec<Neighbor<u32>>,
    prepared: Vec<(f32, Option<u32>)>,
    states: Vec<prune::State>,
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
            Workspace::default,
            |workspace, (source, mut source_candidates)| {
                // Candidate merging already removes duplicate IDs. A list within
                // the degree limit needs no distance calculation.
                if source_candidates.len() <= degree {
                    return Ok(source_candidates);
                }

                let source_id = u32::try_from(source).map_err(ANNError::new)?;
                let source_vector = data.row(source);
                workspace.pool.clear();
                workspace
                    .pool
                    .try_reserve(source_candidates.len())
                    .map_err(ANNError::new)?;
                workspace
                    .pool
                    .extend(source_candidates.iter().copied().map(|candidate| {
                        Neighbor::new(
                            candidate,
                            distance
                                .evaluate_similarity(source_vector, data.row(candidate as usize)),
                        )
                    }));

                let candidate_count = workspace.pool.len();
                if candidate_count > u16::MAX as usize {
                    return Err(ANNError::new(FinalizationError::TooManyCandidates {
                        actual: candidate_count,
                        max: u16::MAX as usize,
                    }));
                }
                workspace.prepared.clear();
                workspace
                    .prepared
                    .try_reserve(candidate_count)
                    .map_err(ANNError::new)?;

                // Sort all candidates before the code marks a self-edge as absent.
                // Thus, self-edge removal cannot add a farther candidate. The
                // `SortedNeighbors` value carries this order into RobustPrune.
                let sorted = SortedNeighbors::new(&mut workspace.pool, candidate_count);
                workspace.prepared.extend(sorted.iter().map(|neighbor| {
                    let id = *neighbor.id();
                    (*neighbor.distance(), (id != source_id).then_some(id))
                }));
                workspace
                    .states
                    .try_reserve(
                        workspace
                            .prepared
                            .len()
                            .saturating_sub(workspace.states.len()),
                    )
                    .map_err(ANNError::new)?;
                workspace
                    .states
                    .resize(workspace.prepared.len(), prune::State::default());

                let selected = prune::robust_prune(
                    &sorted,
                    &workspace.prepared,
                    workspace.states.as_mut_slice(),
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
                for (destination, state) in guard.iter_mut().zip(workspace.states.iter()) {
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
