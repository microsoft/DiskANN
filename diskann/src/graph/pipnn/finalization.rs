/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Final pruning: cut each candidate list to the graph degree with RobustPrune.
//!
//! A list within the degree passes through unchanged. A longer list is sorted by
//! distance to its source point and pruned with the shared Vamana RobustPrune,
//! which owns occlusion and the alpha rounds. The selected IDs overwrite the list
//! in place.

use crate::{
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

/// RobustPrune scratch for one Rayon job.
#[derive(Default)]
struct PruneWorkspace {
    candidates: Vec<Neighbor<u32>>,
    mapped_candidates: Vec<Neighbor<Option<u32>>>,
    states: Vec<prune::State>,
}

impl PruneWorkspace {
    /// Prune the candidates of point `source` to the graph degree.
    fn prune<T: VectorRepr>(
        &mut self,
        data: MatrixView<'_, T>,
        source: usize,
        mut source_candidates: AdjacencyList<u32>,
        graph: &Config,
        distance: &T::Distance,
    ) -> AdjacencyList<u32> {
        let degree = graph.pruned_degree().get();
        if source_candidates.len() <= degree {
            return source_candidates;
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
        // Like Vamana, prune only the `max_occlusion_size` nearest candidates. The
        // graph configuration keeps this size within the `u16` positions of
        // RobustPrune.
        //
        // Sort before mapping: changing the element type can change the unstable
        // sort's order for equal distances, and therefore RobustPrune's choices.
        let sorted = SortedNeighbors::new(&mut self.candidates, graph.max_occlusion_size().get());
        // A list never holds its source: leaf build only pairs two distinct points
        // of one leaf. So every candidate stays.
        let mapped = sorted.map_in(&mut self.mapped_candidates, |&candidate| Some(candidate));
        self.states.clear();
        self.states.resize(sorted.len(), prune::State::default());
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
        source_candidates
    }
}

/// Prune each candidate list that is longer than the graph degree.
///
/// `candidates` holds one list of dataset IDs for each data row.
pub(crate) fn prune_overfull<T: VectorRepr>(
    data: MatrixView<'_, T>,
    candidates: Vec<AdjacencyList<u32>>,
    graph: &Config,
    metric: Metric,
) -> Vec<AdjacencyList<u32>> {
    let distance = T::distance(metric, Some(data.ncols()));

    // This runs in the pool of the build context (see `build_graph`).
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
mod tests {
    use super::*;
    use crate::graph::{
        config,
        pipnn::test_support::{edges, thread_pool},
    };
    use std::collections::BTreeSet;

    fn pruning_config(degree: usize, metric: Metric, alpha: f32) -> Config {
        config::Builder::new_with(degree, config::MaxDegree::same(), 16, metric.into(), |b| {
            b.alpha(alpha);
        })
        .build()
        .unwrap()
    }

    fn candidates_in_order(ids: &[u32]) -> AdjacencyList<u32> {
        let mut candidates = AdjacencyList::new();
        candidates.extend_from_slice(ids);
        candidates
    }

    /// Prune the candidates of `source` in one-dimensional L2 points `x`.
    fn prune_1d(x: &[f32], source: usize, candidates: &[u32], graph: &Config) -> Vec<u32> {
        let data = MatrixView::try_from(x, x.len(), 1).unwrap();
        let distance = f32::distance(Metric::L2, Some(1));
        let pruned = PruneWorkspace::default().prune(
            data,
            source,
            candidates_in_order(candidates),
            graph,
            &distance,
        );
        pruned.to_vec()
    }

    #[test]
    fn candidates_within_degree_keep_their_order() {
        let x = [0.0, 1.0, 3.0, -2.0];
        let graph = pruning_config(2, Metric::L2, 1.0);

        // [3, 1] is not in distance order. A list within the degree skips
        // pruning and keeps its order.
        for candidates in [&[][..], &[2][..], &[3, 1][..]] {
            assert_eq!(prune_1d(&x, 0, candidates, &graph), candidates);
        }
    }

    #[test]
    fn overfull_candidates_follow_the_metric_pruning_rule() {
        // Source 0 keeps two of the candidates [3, 2, 1]. Point 1 is nearest
        // for every metric, so the metric decides which second point stays.
        for (metric, points, expected) in [
            // Point 1 occludes point 2: 9 / 4 > 1.
            (
                Metric::L2,
                [[0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [-2.0, 0.0]],
                [1, 3],
            ),
            // Points 0 and 1 share a direction, so point 2 has occlusion ratio 1 and stays.
            (
                Metric::Cosine,
                [[1.0, 0.0], [2.0, 0.0], [1.0, 1.0], [-1.0, 0.0]],
                [1, 2],
            ),
            // Point 3 is nearer than point 2 and has occlusion ratio 1 / 1.6 < 1.
            (
                Metric::CosineNormalized,
                [[1.0, 0.0], [0.8, 0.6], [-1.0, 0.0], [0.0, -1.0]],
                [1, 3],
            ),
            // Point 1 occludes point 2: their dot product is 6, against 2 with the source.
            (
                Metric::InnerProduct,
                [[1.0, 0.0], [3.0, 0.0], [2.0, 1.0], [-1.0, 4.0]],
                [1, 3],
            ),
        ] {
            let data = MatrixView::try_from(points.as_flattened(), 4, 2).unwrap();
            let distance = f32::distance(metric, Some(2));

            let actual = PruneWorkspace::default().prune(
                data,
                0,
                candidates_in_order(&[3, 2, 1]),
                &pruning_config(2, metric, 1.0),
                &distance,
            );

            assert_eq!(&*actual, expected, "{metric:?}");
        }
    }

    #[test]
    fn graph_alpha_controls_which_occluded_neighbors_return() {
        let x = [0.0, 1.0, 3.0, 4.0];

        // Point 1 is nearest. It occludes point 2 with ratio 9 / 4 and point 3
        // with ratio 16 / 9, so alpha 2 brings back point 3 only.
        for (alpha, expected) in [(1.0, &[1][..]), (2.0, &[1, 3][..])] {
            let graph = pruning_config(2, Metric::L2, alpha);

            assert_eq!(
                prune_1d(&x, 0, &[3, 2, 1], &graph),
                expected,
                "alpha {alpha}"
            );
        }
    }

    #[test]
    fn a_reused_workspace_does_not_carry_state_between_sources() {
        let x = [0.0, 1.0, 3.0, -2.0, 9.0];
        let data = MatrixView::try_from(&x[..], 5, 1).unwrap();
        let graph = pruning_config(2, Metric::L2, 1.0);
        let distance = f32::distance(Metric::L2, Some(1));
        let mut workspace = PruneWorkspace::default();

        // One workspace prunes a long list, a shorter list, then a long list again.
        for (source, candidates, expected) in [
            // Points 1 and 3 are nearest, and point 1 does not occlude point 3.
            (0, &[4, 3, 2, 1][..], &[1, 3][..]),
            // Point 2 at x = 3 occludes points 1 and 3.
            (4, &[3, 2, 1][..], &[2][..]),
            // Points 0 and 2 are nearest, and point 0 does not occlude point 2.
            (1, &[4, 3, 2, 0][..], &[0, 2][..]),
        ] {
            let actual = workspace.prune(
                data,
                source,
                candidates_in_order(candidates),
                &graph,
                &distance,
            );

            assert_eq!(&*actual, expected, "source {source}");
        }
    }

    #[test]
    fn pruning_reads_only_the_max_occlusion_size_nearest_candidates() {
        let x = [0.0, -3.0, 1.0, 2.0];

        // Source 0 is at x=0. Candidates 2 and 3 lie on one side and candidate 1
        // on the other, so 2 occludes 3 but not 1. A pool of two candidates
        // drops 1, the farthest.
        for (max_occlusion_size, expected) in [(2, &[2][..]), (3, &[2, 1][..])] {
            let graph = config::Builder::new_with(
                2,
                config::MaxDegree::same(),
                16,
                Metric::L2.into(),
                |b| {
                    b.alpha(1.0).max_occlusion_size(max_occlusion_size);
                },
            )
            .build()
            .unwrap();

            let actual = prune_1d(&x, 0, &[1, 2, 3], &graph);

            assert_eq!(actual, expected, "max_occlusion_size {max_occlusion_size}");
        }
    }

    #[test]
    fn parallel_pruning_keeps_each_result_with_its_source() {
        // Point i is at x = i², and each point lists every other point.
        let x = [0.0, 1.0, 4.0, 9.0];
        let data = MatrixView::try_from(&x[..], 4, 1).unwrap();
        let graph = pruning_config(1, Metric::L2, 1.0);
        // Degree 1: each point keeps its nearest point.
        let expected = BTreeSet::from([(0, 1), (1, 0), (2, 1), (3, 2)]);

        for workers in [1, 3] {
            let candidates = [[3, 2, 1], [3, 2, 0], [3, 1, 0], [2, 1, 0]]
                .map(|ids| candidates_in_order(&ids))
                .to_vec();

            let actual = thread_pool(workers)
                .install(|| prune_overfull(data, candidates, &graph, Metric::L2));

            let actual: Vec<Vec<u32>> = actual.into_iter().map(Vec::from).collect();
            assert_eq!(edges(&actual), expected, "{workers} workers");
        }
    }
}
