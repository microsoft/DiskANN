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

use std::convert::Infallible;

use crate::{
    graph::{prune, AdjacencyList, Config},
    neighbor::Neighbor,
    utils::VectorRepr,
    ANNError, ANNResult,
};
use diskann_utils::views::MatrixView;
use diskann_vector::{distance::Metric, DistanceFunction};
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
}

/// Per-Rayon-job state retained across source points.
///
/// `prune` owns candidate/state/output buffers. `cache` stores provider lookup
/// results required by the shared kernel. Reusing both avoids per-node
/// allocations, which would otherwise dominate finalization for millions of
/// short candidate lists.
#[derive(Default)]
struct Workspace {
    prune: prune::Scratch<u32>,
    cache: Vec<(f32, Option<u32>)>,
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
    validate_candidate_lists(&candidates, data.nrows()).map_err(ANNError::opaque)?;

    let degree = graph.pruned_degree().get();
    let policy = prune::Policy::new(degree, graph.alpha(), graph.prune_kind(), false);
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

                let source_id = u32::try_from(source).map_err(ANNError::opaque)?;
                let source_vector = data.row(source);
                let pool = workspace.prune.candidates_mut();
                pool.clear();
                pool.try_reserve(source_candidates.len())
                    .map_err(ANNError::opaque)?;
                pool.extend(source_candidates.iter().copied().map(|candidate| {
                    Neighbor::new(
                        candidate,
                        distance.evaluate_similarity(source_vector, data.row(candidate as usize)),
                    )
                }));
                // as_context sorts the active candidate prefix by source distance.
                // The callback below is needed only for selected-to-candidate
                // occlusion checks; dimension specialization stays in `distance`.
                let candidate_count = pool.len();
                let mut context = workspace.prune.as_context(candidate_count);
                prune::robust_prune(
                    &mut context,
                    policy,
                    &mut workspace.cache,
                    Some,
                    |left, right| {
                        Ok::<_, Infallible>(distance.evaluate_similarity(
                            data.row(*left as usize),
                            data.row(*right as usize),
                        ))
                    },
                    |id| id == source_id,
                )
                .map_err(ANNError::opaque)?;

                // RobustPrune selects distinct candidate positions, so its output IDs
                // are unique by construction. `extend_from_slice` would re-derive that
                // with an O(degree^2) membership scan per list; the trusted overwrite
                // is a copy and still verifies uniqueness under debug assertions.
                source_candidates.overwrite_trusted(workspace.prune.neighbors());
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
mod tests;
