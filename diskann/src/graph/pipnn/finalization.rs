/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Final graph-degree enforcement through the shared Vamana RobustPrune kernel.
//!
//! Candidate merging may produce more than `R` IDs for a point. This stage first
//! validates every global ID, then processes each point's candidate list in the
//! caller's Rayon pool. Lists already within the degree bound are returned without
//! distance work. Overfull lists are converted to source-distance candidates,
//! passed through RobustPrune, and rewritten from the selected output.
//!
//! The shared kernel owns occlusion and alpha-round semantics; this adapter owns
//! only contiguous dataset access and distance specialization for the source
//! representation.
//!
//! ```text
//! candidate lists ──> validate point count and every global ID
//!                                      │
//!                         ┌────────────┴────────────┐
//!                         v                         v
//!                    len <= R                  len > R
//!                  return list      source-distance candidates
//!                                                   │
//!                                                   v
//!                                           shared RobustPrune
//!                                                   │
//!                                                   v
//!                                           rewrite same list owner
//! ```
//!
//! | Path | Distance evaluations | Allocation behavior |
//! | --- | --- | --- |
//! | bounded list | none | move list directly to output |
//! | overfull list | source and occlusion distances | reuse Rayon-job workspace |

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

/// Per-Rayon-job preparation and kernel state retained across source points.
///
/// PiPNN owns sorting, allocation, and ID translation. The shared internal
/// kernel receives only the prepared candidates and an exactly sized state slice.
#[derive(Default)]
struct Workspace {
    pool: Vec<Neighbor<u32>>,
    prepared: Vec<(f32, Option<u32>)>,
    states: Vec<prune::State>,
}

/// Validate candidate IDs and prune only lists whose length exceeds graph degree.
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

    // build_graph installs the complete call tree in the caller-owned pool.
    #[allow(clippy::disallowed_methods)]
    candidates
        .into_par_iter()
        .enumerate()
        .map_init(
            Workspace::default,
            |workspace, (source, mut source_candidates)| {
                // Candidate accumulators already enforce uniqueness. A bounded list
                // therefore satisfies the graph policy without distance evaluation.
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

                // Sorting/capping precedes source exclusion so filtering cannot
                // backfill with farther candidates. Passing this witness into
                // RobustPrune makes source-distance order part of its input type.
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
