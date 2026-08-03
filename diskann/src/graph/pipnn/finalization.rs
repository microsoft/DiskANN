/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Final graph-degree enforcement through the shared Vamana RobustPrune kernel.
//!
//! Candidate merging may produce more than `R` IDs for a point. This stage first
//! validates every global ID, then processes rows independently in the caller's
//! Rayon pool. Rows already within the degree bound are returned without any
//! distance work. Overfull rows are converted to source-distance candidates,
//! passed through RobustPrune, and rewritten from the selected output.
//!
//! The shared kernel owns occlusion and alpha-round semantics; this adapter owns
//! only contiguous dataset access and distance specialization for the source
//! representation.
//!
//! ```text
//! candidate rows ──> validate row count and every global ID
//!                                      │
//!                         ┌────────────┴────────────┐
//!                         v                         v
//!                    len <= R                  len > R
//!                  return row       source-distance candidates
//!                                                   │
//!                                                   v
//!                                           shared RobustPrune
//!                                                   │
//!                                                   v
//!                                           rewrite same row owner
//! ```
//!
//! | Path | Distance evaluations | Allocation behavior |
//! | --- | --- | --- |
//! | bounded row | none | move row directly to output |
//! | overfull row | source and occlusion distances | reuse Rayon-job workspace |

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
    #[error("candidate row count {rows} does not match the dataset point count {points}")]
    RowCountMismatch { rows: usize, points: usize },
    #[error("candidate ID {candidate} in row {row} is outside a {points}-point dataset")]
    InvalidCandidateId {
        row: usize,
        candidate: u32,
        points: usize,
    },
}

/// Per-Rayon-job state retained across rows.
///
/// `prune` owns candidate/state/output buffers. `cache` stores provider lookup
/// results required by the shared kernel. Reusing both avoids per-node
/// allocations, which would otherwise dominate finalization for millions of
/// short rows.
#[derive(Default)]
struct Workspace {
    prune: prune::Scratch<u32>,
    cache: Vec<(f32, Option<u32>)>,
}

/// Validate candidate IDs and prune only rows whose length exceeds graph degree.
pub(crate) fn prune_overfull<T>(
    data: MatrixView<'_, T>,
    candidates: Vec<AdjacencyList<u32>>,
    graph: &Config,
    metric: Metric,
) -> ANNResult<Vec<AdjacencyList<u32>>>
where
    T: VectorRepr + Send + Sync,
{
    validate_candidates(&candidates, data.nrows()).map_err(ANNError::opaque)?;

    let degree = graph.pruned_degree().get();
    let policy = prune::Policy::new(degree, graph.alpha(), graph.prune_kind(), false);
    let distance = T::distance(metric, Some(data.ncols()));

    // build_graph installs the complete call tree in the caller-owned pool.
    #[allow(clippy::disallowed_methods)]
    candidates
        .into_par_iter()
        .enumerate()
        .map_init(Workspace::default, |workspace, (source, mut row)| {
            // Candidate accumulators already enforce uniqueness. A bounded row
            // therefore satisfies the graph policy without distance evaluation.
            if row.len() <= degree {
                return Ok(row);
            }

            let source_id = u32::try_from(source).map_err(ANNError::opaque)?;
            let source_vector = data.row(source);
            let pool = workspace.prune.candidates_mut();
            pool.clear();
            pool.try_reserve(row.len()).map_err(ANNError::opaque)?;
            pool.extend(row.iter().copied().map(|candidate| {
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
                    Ok::<_, Infallible>(
                        distance.evaluate_similarity(
                            data.row(*left as usize),
                            data.row(*right as usize),
                        ),
                    )
                },
                |id| id == source_id,
            )
            .map_err(ANNError::opaque)?;

            // RobustPrune selects distinct candidate positions, so its output IDs
            // are unique by construction. `extend_from_slice` would re-derive that
            // with an O(degree^2) membership scan per row; the trusted overwrite is
            // a copy and still verifies uniqueness under debug assertions.
            row.overwrite_trusted(workspace.prune.neighbors());
            Ok(row)
        })
        .collect()
}

fn validate_candidates(
    candidates: &[AdjacencyList<u32>],
    points: usize,
) -> Result<(), FinalizationError> {
    if candidates.len() != points {
        return Err(FinalizationError::RowCountMismatch {
            rows: candidates.len(),
            points,
        });
    }
    for (row_id, row) in candidates.iter().enumerate() {
        if let Some(&candidate) = row.iter().find(|&&id| id as usize >= points) {
            return Err(FinalizationError::InvalidCandidateId {
                row: row_id,
                candidate,
                points,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
