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

    /// L2 graph configuration that prunes to `degree`. Alpha 1 means that a
    /// selected neighbor occludes a candidate when the neighbor is nearer to
    /// the candidate than the source is.
    fn pruning_config(degree: usize) -> Config {
        // This is the build search list size, which pruning does not read.
        let search_list_size = 16;
        let alpha = 1.0;
        config::Builder::new_with(
            degree,
            config::MaxDegree::same(),
            search_list_size,
            Metric::L2.into(),
            |b| {
                b.alpha(alpha);
            },
        )
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
        let dimensions = 1;
        let data = MatrixView::try_from(x, x.len(), dimensions).unwrap();
        let distance = f32::distance(Metric::L2, Some(dimensions));
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
        // x[i] is the one-dimensional vector of point i.
        let x = [0.0, 1.0, -2.0];
        let source = 0;
        // Point 1 is nearer to the source than point 2, so a prune would
        // return [1, 2].
        let candidates = [2, 1];
        let degree = candidates.len();

        let actual = prune_1d(&x, source, &candidates, &pruning_config(degree));

        // The list fits the degree, so it comes back unpruned, in its order.
        assert_eq!(actual, candidates);
    }

    #[test]
    fn an_overfull_list_is_pruned_by_robust_prune() {
        // x[i] is the one-dimensional vector of point i.
        let x = [0.0, 1.0, 2.0, -3.0];
        let source = 0;
        let candidates = [3, 2, 1];
        let degree = 2;

        let actual = prune_1d(&x, source, &candidates, &pruning_config(degree));

        // Sorted by distance the list is [1, 2, 3], so a cut to the degree
        // would return [1, 2]. Point 1 occludes point 2, so RobustPrune
        // returns [1, 3].
        assert_eq!(actual, [1, 3]);
    }

    #[test]
    fn a_reused_workspace_does_not_carry_state_between_sources() {
        // x[i] is the one-dimensional vector of point i.
        let x = [0.0, 1.0, 3.0, -2.0, 9.0];
        let dimensions = 1;
        let degree = 2;
        let data = MatrixView::try_from(x.as_slice(), x.len(), dimensions).unwrap();
        let graph = pruning_config(degree);
        let distance = f32::distance(Metric::L2, Some(dimensions));
        let mut workspace = PruneWorkspace::default();

        // One workspace prunes a long list, a shorter list, then a long list again.
        for (source, candidates, expected) in [
            // Points 1 and 3 are nearest, and point 1 does not occlude point 3.
            (0, vec![4, 3, 2, 1], vec![1, 3]),
            // Point 2 at x = 3 occludes points 1 and 3.
            (4, vec![3, 2, 1], vec![2]),
            // Points 0 and 2 are nearest, and point 0 does not occlude point 2.
            (1, vec![4, 3, 2, 0], vec![0, 2]),
        ] {
            let actual = workspace.prune(
                data,
                source,
                candidates_in_order(&candidates),
                &graph,
                &distance,
            );

            assert_eq!(&*actual, expected, "source {source}");
        }
    }

    #[test]
    fn parallel_pruning_keeps_each_result_with_its_source() {
        // x[i] = i² is the one-dimensional vector of point i.
        // Each point lists every other point, farthest first.
        let x = [0.0, 1.0, 4.0, 9.0];
        let dimensions = 1;
        let degree = 1;
        let data = MatrixView::try_from(x.as_slice(), x.len(), dimensions).unwrap();
        let graph = pruning_config(degree);
        // Degree 1 keeps each point's nearest point.
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
