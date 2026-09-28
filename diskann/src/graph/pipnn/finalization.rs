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
        // RobustPrune stores candidate positions as `u16`, so keep the `u16::MAX`
        // nearest candidates. Vamana limits its candidate pool the same way.
        //
        // Sort before mapping: changing the element type can change the unstable
        // sort's order for equal distances, and therefore RobustPrune's choices.
        let sorted = SortedNeighbors::new(&mut self.candidates, u16::MAX as usize);
        let mapped = sorted.map_in(&mut self.mapped_candidates, |&candidate| {
            (candidate as usize != source).then_some(candidate)
        });
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
    use crate::graph::{config, pipnn::test_support::thread_pool};

    fn pruning_config(
        degree: usize,
        metric: Metric,
        alpha: f32,
    ) -> Result<Config, config::ConfigError> {
        config::Builder::new_with(degree, config::MaxDegree::same(), 16, metric.into(), |b| {
            b.alpha(alpha);
        })
        .build()
    }

    fn candidates_in_order(ids: &[u32]) -> AdjacencyList<u32> {
        let mut candidates = AdjacencyList::new();
        candidates.extend_from_slice(ids);
        candidates
    }

    /// Prune the candidates of source 0 in one-dimensional L2 data.
    fn prune_l2(values: &[f32], ids: &[u32], degree: usize, alpha: f32) -> Vec<u32> {
        let data = MatrixView::try_from(values, values.len(), 1).unwrap();
        let graph = pruning_config(degree, Metric::L2, alpha).unwrap();
        let distance = f32::distance(Metric::L2, Some(1));
        let pruned =
            PruneWorkspace::default().prune(data, 0, candidates_in_order(ids), &graph, &distance);
        pruned.to_vec()
    }

    #[test]
    fn candidates_within_degree_keep_their_order() {
        let values = [0.0_f32, 1.0, 3.0, -2.0];

        for ids in [&[][..], &[2][..], &[3, 1][..]] {
            assert_eq!(prune_l2(&values, ids, 2, 1.0), ids);
        }
    }

    #[test]
    fn overfull_candidates_follow_the_metric_pruning_rule() {
        // Source 0 has candidates [3, 2, 1]; ID 1 is nearest for every metric.
        // L2 occludes ID 2: 9 / 4 > 1.
        // Cosine retains ID 2: equal directions give an occlusion ratio of 1.
        // Normalized cosine retains ID 3: 1 / 1.6 < 1.
        // Inner product occludes ID 2: its dot with ID 1 is 6, versus 2 with the source.
        for (metric, values, expected) in [
            (
                Metric::L2,
                [0.0, 0.0, 1.0, 0.0, 3.0, 0.0, -2.0, 0.0],
                [1, 3],
            ),
            (
                Metric::Cosine,
                [1.0, 0.0, 2.0, 0.0, 1.0, 1.0, -1.0, 0.0],
                [1, 2],
            ),
            (
                Metric::CosineNormalized,
                [1.0, 0.0, 0.8, 0.6, -1.0, 0.0, 0.0, -1.0],
                [1, 3],
            ),
            (
                Metric::InnerProduct,
                [1.0, 0.0, 3.0, 0.0, 2.0, 1.0, -1.0, 4.0],
                [1, 3],
            ),
        ] {
            let data = MatrixView::try_from(&values[..], 4, 2).unwrap();
            let graph = pruning_config(2, metric, 1.0).unwrap();
            let distance = f32::distance(metric, Some(2));

            let actual = PruneWorkspace::default().prune(
                data,
                0,
                candidates_in_order(&[3, 2, 1]),
                &graph,
                &distance,
            );

            assert_eq!(&*actual, expected, "{metric:?}");
        }
    }

    #[test]
    fn pruning_excludes_the_source_without_shifting_selected_ids() {
        assert_eq!(
            prune_l2(&[0.0, 1.0, 3.0, -2.0], &[3, 0, 2, 1], 2, 1.0),
            [1, 3]
        );
    }

    #[test]
    fn graph_alpha_controls_which_occluded_neighbors_return() {
        // After selecting x=1, x=3 has ratio 9/4 and x=4 has ratio 16/9.
        // Alpha 2 admits x=4 while x=3 remains occluded.
        let values = [0.0_f32, 1.0, 3.0, 4.0];

        for (alpha, expected) in [(1.0, &[1][..]), (2.0, &[1, 3][..])] {
            assert_eq!(
                prune_l2(&values, &[3, 2, 1], 2, alpha),
                expected,
                "alpha {alpha}"
            );
        }
    }

    #[test]
    fn a_reused_workspace_does_not_carry_state_between_sources() {
        let values = [0.0_f32, 1.0, 3.0, -2.0, 9.0];
        let data = MatrixView::try_from(&values[..], 5, 1).unwrap();
        let graph = pruning_config(2, Metric::L2, 1.0).unwrap();
        let distance = f32::distance(Metric::L2, Some(1));
        let mut workspace = PruneWorkspace::default();

        // One workspace serves a large list, a smaller list, then a large list again.
        for (source, ids, expected) in [
            (0, &[4, 3, 2, 1][..], &[1, 3][..]),
            (4, &[3, 2, 1][..], &[2][..]),
            (1, &[4, 3, 2, 0][..], &[0, 2][..]),
        ] {
            let actual = workspace.prune(data, source, candidates_in_order(ids), &graph, &distance);

            assert_eq!(&*actual, expected, "source {source}");
        }
    }

    #[test]
    fn a_list_above_the_u16_position_limit_keeps_its_nearest_candidates() {
        let count = u16::MAX as usize + 1;
        let values: Vec<_> = (0..=count).map(|i| i as f32).collect();
        let data = MatrixView::try_from(values.as_slice(), count + 1, 1).unwrap();
        let graph = pruning_config(1, Metric::L2, 1.0).unwrap();
        let distance = f32::distance(Metric::L2, Some(1));
        let candidates = AdjacencyList::from_iter_untrusted(1..=count as u32);

        let actual = PruneWorkspace::default().prune(data, 0, candidates, &graph, &distance);

        assert_eq!(&*actual, [1]);
    }

    #[test]
    fn parallel_pruning_keeps_each_result_with_its_source() {
        let values = [0.0_f32, 1.0, 4.0, 9.0];
        let data = MatrixView::try_from(&values[..], 4, 1).unwrap();
        let graph = pruning_config(1, Metric::L2, 1.0).unwrap();

        for workers in [1, 3] {
            let candidates = [[3, 2, 1], [3, 2, 0], [3, 1, 0], [2, 1, 0]]
                .map(|ids| candidates_in_order(&ids))
                .to_vec();

            let actual = thread_pool(workers)
                .install(|| prune_overfull(data, candidates, &graph, Metric::L2));

            assert_eq!(
                actual.into_iter().map(Vec::from).collect::<Vec<_>>(),
                [vec![1], vec![0], vec![1], vec![2]],
                "{workers} workers"
            );
        }
    }
}
