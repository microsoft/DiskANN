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
//!
//! Distances come from the [`Dataset`] object, so this step compiles once for
//! every element type.

use crate::{
    graph::{
        AdjacencyList,
        config::PruneKind,
        internal::{SortedNeighbors, prune},
    },
    neighbor::Neighbor,
};
use rayon::prelude::*;

use super::{PruneConfig, dataset::Dataset};

/// RobustPrune scratch for one prune job.
#[derive(Default)]
struct PruneWorkspace {
    candidates: Vec<Neighbor<u32>>,
    mapped_candidates: Vec<Neighbor<Option<u32>>>,
    states: Vec<prune::State>,
}

impl PruneWorkspace {
    /// Prune the candidates of point `source` to the graph degree.
    fn prune(
        &mut self,
        source: u32,
        mut source_candidates: AdjacencyList<u32>,
        config: &PruneConfig,
        data: &dyn Dataset,
    ) -> AdjacencyList<u32> {
        let degree = config.degree.get();
        if source_candidates.len() <= degree {
            return source_candidates;
        }

        self.candidates.clear();
        self.candidates.extend(
            source_candidates
                .iter()
                .map(|&candidate| Neighbor::new(candidate, data.distance(source, candidate))),
        );
        // Like Vamana, prune only the `max_occlusion_size` nearest candidates. The
        // configuration keeps this size within the `u16` positions of RobustPrune.
        //
        // Sort before mapping: changing the element type can change the unstable
        // sort's order for equal distances, and therefore RobustPrune's choices.
        let sorted = SortedNeighbors::new(
            &mut self.candidates,
            usize::from(config.max_occlusion_size.get()),
        );
        // RobustPrune skips `None` candidates. PiPNN has none to skip: leaf build
        // never pairs a point with itself, and every vector is in memory.
        let mapped = sorted.map_in(&mut self.mapped_candidates, |&candidate| Some(candidate));
        self.states.clear();
        self.states.resize(sorted.len(), prune::State::default());
        let selected = prune::robust_prune(
            mapped,
            &mut self.states,
            degree,
            config.alpha,
            PruneKind::from_metric(data.metric()),
            |&left, &right| data.distance(left, right),
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
/// `candidates` holds one list of dataset IDs for each row of `data`.
pub(super) fn prune_overfull(
    candidates: Vec<AdjacencyList<u32>>,
    data: &dyn Dataset,
    config: &PruneConfig,
) -> Vec<AdjacencyList<u32>> {
    candidates
        .into_par_iter()
        .enumerate()
        .map_init(
            PruneWorkspace::default,
            |workspace, (source, candidates)| {
                workspace.prune(source as u32, candidates, config, data)
            },
        )
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ANNResult,
        graph::pipnn::{Config, test_support::nz},
    };
    use diskann_utils::views::rowmajor;
    use diskann_vector::distance::Metric;
    use std::collections::BTreeSet;

    /// One-dimensional L2 points `x[i]`. Pruning reads only the distances.
    struct Line<'a>(&'a [f32]);

    impl Dataset for Line<'_> {
        fn nrows(&self) -> usize {
            self.0.len()
        }

        fn dimensions(&self) -> usize {
            1
        }

        fn metric(&self) -> Metric {
            Metric::L2
        }

        fn gather(&self, _: &[u32], _: rowmajor::Mut<'_, f32>) -> ANNResult<()> {
            unreachable!("pruning does not gather rows")
        }

        fn distance(&self, source: u32, target: u32) -> f32 {
            (self.0[source as usize] - self.0[target as usize]).powi(2)
        }
    }

    fn as_set(ids: &[u32]) -> BTreeSet<u32> {
        ids.iter().copied().collect()
    }

    /// L2 configuration that prunes to `degree`. Alpha 1 means that a selected
    /// neighbor occludes a candidate when the neighbor is nearer to the
    /// candidate than the source is.
    fn pruning_config(degree: usize) -> PruneConfig {
        let alpha = 1.0;
        Config::builder(nz(degree))
            .alpha(alpha)
            .build()
            .unwrap()
            .prune
    }

    fn candidates_in_order(ids: &[u32]) -> AdjacencyList<u32> {
        let mut candidates = AdjacencyList::new();
        candidates.extend_from_slice(ids);
        candidates
    }

    /// Prune the candidates of `source` in one-dimensional L2 points `x`.
    fn prune_1d(x: &[f32], source: u32, candidates: &[u32], config: &PruneConfig) -> Vec<u32> {
        let pruned = PruneWorkspace::default().prune(
            source,
            candidates_in_order(candidates),
            config,
            &Line(x),
        );
        pruned.to_vec()
    }

    #[test]
    fn skip_prune_when_candidates_do_not_exceed_degree() {
        // x[i] is the one-dimensional vector of point i.
        let x = [0.0, 1.0, 2.0];
        let source = 0;
        // Point 1 occludes point 2, so a prune would keep only {1}.
        let candidates = [2, 1];
        let degree = candidates.len();

        let actual = prune_1d(&x, source, &candidates, &pruning_config(degree));

        // The list fits the degree, so both candidates stay.
        assert_eq!(as_set(&actual), as_set(&candidates));
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
        // would return {1, 2}. Point 1 occludes point 2, so RobustPrune
        // returns {1, 3}.
        assert_eq!(as_set(&actual), BTreeSet::from([1, 3]));
    }

    #[test]
    fn a_reused_workspace_does_not_carry_state_between_sources() {
        // x[i] is the one-dimensional vector of point i.
        let x = [0.0, 1.0, 3.0, -2.0, 9.0];
        let degree = 2;
        let config = pruning_config(degree);
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
            let actual =
                workspace.prune(source, candidates_in_order(&candidates), &config, &Line(&x));

            assert_eq!(as_set(&actual), as_set(&expected), "source {source}");
        }
    }

    #[test]
    fn prune_overfull_keeps_each_result_with_its_source() {
        // x[i] is the one-dimensional vector of point i.
        // Overfull, within-degree and empty lists must retain their source rows.
        let x = [0.0, 1.0, 2.0, -3.0];
        let degree = 2;
        let config = pruning_config(degree);
        let candidates = [vec![3, 2, 1], vec![0, 2], vec![1, 0, 3], vec![]]
            .map(|ids| candidates_in_order(&ids))
            .to_vec();
        let expected = vec![
            // Source 0: point 1 occludes point 2.
            BTreeSet::from([1, 3]),
            // Source 1: the list fits the degree and keeps its order.
            BTreeSet::from([0, 2]),
            // Source 2: point 1 occludes points 0 and 3.
            BTreeSet::from([1]),
            // Source 3: the empty list keeps its row.
            BTreeSet::new(),
        ];

        let actual = prune_overfull(candidates, &Line(&x), &config);

        let actual: Vec<BTreeSet<u32>> = actual.into_iter().map(|row| as_set(&row)).collect();
        assert_eq!(actual, expected);
    }
}
