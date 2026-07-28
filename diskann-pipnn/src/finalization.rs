/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Orders complete PiPNN candidate rows and applies shared Vamana RobustPrune.

use std::convert::Infallible;

use diskann::{
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

#[derive(Default)]
struct Workspace {
    prune: prune::Scratch<u32>,
    cache: Vec<(f32, Option<u32>)>,
}

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

            row.clear();
            row.extend_from_slice(workspace.prune.neighbors());
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
