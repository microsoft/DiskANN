/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Applies Vamana's robust-prune kernel to PiPNN candidate lists.

use std::convert::Infallible;

use diskann::{
    graph::{config::PruneKind, prune},
    neighbor::Neighbor,
    utils::VectorRepr,
    ANNError, ANNResult,
};
use diskann_vector::{distance::Metric, DistanceFunction};
use rayon::prelude::*;

use crate::rayon_util::ParIterInstalled;

#[derive(Debug, Default)]
struct Workspace {
    prune: prune::Scratch<u32>,
    cache: Vec<(f32, Option<u32>)>,
}

/// Prune only candidate lists that exceed the graph's degree bound.
///
/// PiPNN stores compact candidate IDs, while Vamana's robust-prune kernel consumes
/// candidates ordered by their distance from the source. This is the only
/// PiPNN-specific adaptation between those representations.
pub(crate) fn prune_overfull_lists<T: VectorRepr + Send + Sync>(
    data: &[T],
    dimensions: usize,
    candidates_per_node: Vec<Vec<u32>>,
    max_degree: usize,
    metric: Metric,
    alpha: f32,
) -> ANNResult<Vec<Vec<u32>>> {
    let distance = T::distance(metric, Some(dimensions));
    let policy = prune::Policy::new(max_degree, alpha, PruneKind::from_metric(metric), false);

    candidates_per_node
        .into_par_iter()
        .enumerate()
        .map_init(Workspace::default, |workspace, (source, mut candidates)| {
            if candidates.len() <= max_degree {
                return Ok(candidates);
            }

            let row = |id: u32| {
                let start = id as usize * dimensions;
                &data[start..start + dimensions]
            };
            let source_id = u32::try_from(source)?;
            let source_vector = row(source_id);

            let pool = workspace.prune.candidates_mut();
            pool.clear();
            pool.try_reserve(candidates.len())
                .map_err(ANNError::opaque)?;
            pool.extend(
                candidates.iter().copied().map(|id| {
                    Neighbor::new(id, distance.evaluate_similarity(source_vector, row(id)))
                }),
            );

            // Pass the full candidate set through. The shared kernel owns its
            // candidate-capacity check and returns an error when it cannot
            // represent the input; this adapter must not silently discard IDs.
            let candidate_count = pool.len();
            let mut context = workspace.prune.as_context(candidate_count);
            prune::robust_prune(
                &mut context,
                policy,
                &mut workspace.cache,
                Some,
                |left, right| {
                    Ok::<_, Infallible>(distance.evaluate_similarity(row(*left), row(*right)))
                },
                |id| id == source_id,
            )
            .map_err(ANNError::opaque)?;

            candidates.clear();
            candidates.extend_from_slice(workspace.prune.neighbors());
            Ok(candidates)
        })
        .collect_installed()
}

#[cfg(test)]
mod tests;
