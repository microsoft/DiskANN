/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Graph-degree enforcement with the Vamana RobustPrune kernel.
//!
//! Candidate merging can produce more than the graph degree for one point.
//! Lists within the degree limit pass through unchanged. Longer lists are sorted
//! by source distance and passed to RobustPrune, then overwritten with the
//! selected IDs. Reusable per-job scratch stores candidates and pruning state.
//!
//! RobustPrune owns occlusion and alpha-round behavior. This module supplies
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
    #[error("candidate count {actual} exceeds the u16 position limit {max}")]
    TooManyCandidates { actual: usize, max: usize },
}

/// RobustPrune state for one Rayon job.
#[derive(Default)]
struct PruneWorkspace {
    candidates: Vec<Neighbor<u32>>,
    mapped_candidates: Vec<Neighbor<Option<u32>>>,
    states: Vec<prune::State>,
}

impl PruneWorkspace {
    fn prune<T: VectorRepr>(
        &mut self,
        data: MatrixView<'_, T>,
        source: usize,
        mut source_candidates: AdjacencyList<u32>,
        graph: &Config,
        distance: &T::Distance,
    ) -> ANNResult<AdjacencyList<u32>> {
        let degree = graph.pruned_degree().get();
        let candidate_count = source_candidates.len();
        if candidate_count <= degree {
            return Ok(source_candidates);
        }
        if candidate_count > u16::MAX as usize {
            return Err(ANNError::new(FinalizationError::TooManyCandidates {
                actual: candidate_count,
                max: u16::MAX as usize,
            }));
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
        // Sort before mapping: changing the element type can change the unstable
        // sort's order for equal distances, and therefore RobustPrune's choices.
        let sorted = SortedNeighbors::new(&mut self.candidates, candidate_count);
        let mapped = sorted.map_in(&mut self.mapped_candidates, |&candidate| {
            (candidate as usize != source).then_some(candidate)
        });
        self.states.clear();
        self.states.resize(candidate_count, prune::State::default());
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
        Ok(source_candidates)
    }
}

/// Prune each candidate list that exceeds the graph degree.
///
/// Candidate builders supply one list per data row and valid dataset IDs.
pub(crate) fn prune_overfull<T: VectorRepr>(
    data: MatrixView<'_, T>,
    candidates: Vec<AdjacencyList<u32>>,
    graph: &Config,
    metric: Metric,
) -> ANNResult<Vec<AdjacencyList<u32>>> {
    let distance = T::distance(metric, Some(data.ncols()));

    // The build context supplies the Rayon pool for this terminal operation.
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
    use crate::graph::config;
    use rstest::rstest;

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

    #[rstest]
    #[case::empty(&[])]
    #[case::below_degree(&[2])]
    #[case::at_degree(&[3, 1])]
    fn candidates_within_degree_keep_their_order(#[case] ids: &[u32]) {
        let values = [0.0_f32, 1.0, 3.0, -2.0];
        let data = MatrixView::try_from(&values[..], 4, 1).unwrap();
        let graph = pruning_config(2, Metric::L2, 1.0).unwrap();
        let distance = f32::distance(Metric::L2, Some(1));

        let actual = PruneWorkspace::default()
            .prune(data, 0, candidates_in_order(ids), &graph, &distance)
            .unwrap();

        assert_eq!(&*actual, ids);
    }

    #[rstest]
    #[case::l2(Metric::L2, [0.0, 0.0, 1.0, 0.0, 3.0, 0.0, -2.0, 0.0], [1, 3])]
    #[case::cosine(Metric::Cosine, [1.0, 0.0, 2.0, 0.0, 1.0, 1.0, -1.0, 0.0], [1, 2])]
    #[case::normalized_cosine(Metric::CosineNormalized, [1.0, 0.0, 0.8, 0.6, -1.0, 0.0, 0.0, -1.0], [1, 3])]
    #[case::inner_product(Metric::InnerProduct, [1.0, 0.0, 3.0, 0.0, 2.0, 1.0, -1.0, 4.0], [1, 3])]
    fn overfull_candidates_follow_the_metric_pruning_rule(
        #[case] metric: Metric,
        #[case] values: [f32; 8],
        #[case] expected: [u32; 2],
    ) {
        // ID 1 is nearest. L2 occludes ID 2: 9 / 4 > 1.
        // Cosine retains ID 2: equal directions give an occlusion ratio of 1.
        // Normalized cosine retains ID 3: 1 / 1.6 < 1.
        // Inner product occludes ID 2: its dot with ID 1 is 6, versus 2 with the source.
        let data = MatrixView::try_from(&values[..], 4, 2).unwrap();
        let graph = pruning_config(2, metric, 1.0).unwrap();
        let distance = f32::distance(metric, Some(2));

        let actual = PruneWorkspace::default()
            .prune(data, 0, candidates_in_order(&[3, 2, 1]), &graph, &distance)
            .unwrap();

        assert_eq!(&*actual, expected);
    }

    #[test]
    fn pruning_excludes_the_source_without_shifting_selected_ids() {
        let values = [0.0_f32, 1.0, 3.0, -2.0];
        let data = MatrixView::try_from(&values[..], 4, 1).unwrap();
        let graph = pruning_config(2, Metric::L2, 1.0).unwrap();
        let distance = f32::distance(Metric::L2, Some(1));

        let actual = PruneWorkspace::default()
            .prune(
                data,
                0,
                candidates_in_order(&[3, 0, 2, 1]),
                &graph,
                &distance,
            )
            .unwrap();

        assert_eq!(&*actual, [1, 3]);
    }

    #[rstest]
    #[case::strict(1.0, &[1])]
    #[case::relaxed(2.0, &[1, 3])]
    fn graph_alpha_controls_which_occluded_neighbors_return(
        #[case] alpha: f32,
        #[case] expected: &[u32],
    ) {
        // After selecting x=1, x=3 has ratio 9/4 and x=4 has ratio 16/9.
        // Alpha 2 admits x=4 while x=3 remains occluded.
        let values = [0.0_f32, 1.0, 3.0, 4.0];
        let data = MatrixView::try_from(&values[..], 4, 1).unwrap();
        let graph = pruning_config(2, Metric::L2, alpha).unwrap();
        let distance = f32::distance(Metric::L2, Some(1));

        let actual = PruneWorkspace::default()
            .prune(data, 0, candidates_in_order(&[3, 2, 1]), &graph, &distance)
            .unwrap();

        assert_eq!(&*actual, expected);
    }

    #[test]
    fn reused_workspace_does_not_carry_neighbors_or_occlusion_between_sources() {
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
            let actual = workspace
                .prune(data, source, candidates_in_order(ids), &graph, &distance)
                .unwrap();

            assert_eq!(&*actual, expected, "source {source}");
        }
    }

    #[cfg(not(miri))]
    #[test]
    fn the_largest_supported_candidate_list_can_be_pruned() {
        let count = u16::MAX as usize;
        let values: Vec<_> = (0..=count).map(|i| i as f32).collect();
        let data = MatrixView::try_from(values.as_slice(), count + 1, 1).unwrap();
        let candidates = AdjacencyList::from_iter_untrusted(1..=count as u32);
        let graph = pruning_config(1, Metric::L2, 1.0).unwrap();
        let distance = f32::distance(Metric::L2, Some(1));

        let actual = PruneWorkspace::default()
            .prune(data, 0, candidates, &graph, &distance)
            .unwrap();

        assert_eq!(&*actual, [1]);
    }

    #[cfg(not(miri))]
    #[test]
    fn a_candidate_list_beyond_the_position_limit_returns_its_size() {
        let count = u16::MAX as usize + 1;
        let values: Vec<_> = (0..=count).map(|i| i as f32).collect();
        let data = MatrixView::try_from(values.as_slice(), count + 1, 1).unwrap();
        let candidates = AdjacencyList::from_iter_untrusted(1..=count as u32);
        let graph = pruning_config(1, Metric::L2, 1.0).unwrap();
        let distance = f32::distance(Metric::L2, Some(1));

        let error = PruneWorkspace::default()
            .prune(data, 0, candidates, &graph, &distance)
            .unwrap_err();

        let error = error.downcast_ref::<FinalizationError>().unwrap();
        assert!(
            matches!(error, FinalizationError::TooManyCandidates { actual, max }
            if *actual == count && *max == u16::MAX as usize)
        );
    }

    #[rstest]
    #[case::one_worker(1)]
    #[case::several_workers(3)]
    fn parallel_pruning_keeps_each_result_with_its_source(#[case] workers: usize) {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap();
        let values = [0.0_f32, 1.0, 4.0, 9.0];
        let data = MatrixView::try_from(&values[..], 4, 1).unwrap();
        let candidates = [[3, 2, 1], [3, 2, 0], [3, 1, 0], [2, 1, 0]]
            .map(|ids| candidates_in_order(&ids))
            .to_vec();
        let graph = pruning_config(1, Metric::L2, 1.0).unwrap();

        let actual = pool
            .install(|| prune_overfull(data, candidates, &graph, Metric::L2))
            .unwrap();

        assert_eq!(
            actual.into_iter().map(Vec::from).collect::<Vec<_>>(),
            [vec![1], vec![0], vec![1], vec![2]]
        );
    }
}
