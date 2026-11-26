/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Graph-build algorithm of a build job.

#[cfg(feature = "pipnn")]
use std::num::NonZeroUsize;

use serde::{Deserialize, Serialize};

/// Graph-build algorithm of a build job. Without the `pipnn` feature, a PiPNN
/// job fails to parse as an unknown variant.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "algorithm")]
pub(crate) enum BuildAlgorithm {
    /// Insert the points one by one.
    #[default]
    Vamana,
    /// Build the graph in one batch with PiPNN.
    #[cfg(feature = "pipnn")]
    PiPNN(PiPNNParameters),
}

/// PiPNN parameters of a build job.
///
/// The graph configuration of the job supplies the degree and alpha. A
/// parameter that is not set keeps the default of
/// [`diskann::graph::pipnn::Builder`].
#[cfg(feature = "pipnn")]
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct PiPNNParameters {
    c_max: Option<NonZeroUsize>,
    p_samp: Option<f64>,
    max_leaders: Option<NonZeroUsize>,
    fanout: Option<Vec<NonZeroUsize>>,
    max_depth: Option<NonZeroUsize>,
    leaf_k: Option<NonZeroUsize>,
    hash_prune: Option<HashPruneParameters>,
}

/// HashPrune parameters of a PiPNN build job.
#[cfg(feature = "pipnn")]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct HashPruneParameters {
    num_hash_planes: usize,
    l_max: usize,
    final_prune: bool,
}

#[cfg(feature = "pipnn")]
impl Default for HashPruneParameters {
    fn default() -> Self {
        Self {
            num_hash_planes: 12,
            l_max: 64,
            final_prune: true,
        }
    }
}

#[cfg(feature = "pipnn")]
impl PiPNNParameters {
    /// The PiPNN configuration of a job with the graph configuration `graph`.
    pub(crate) fn config(
        &self,
        graph: &diskann::graph::Config,
    ) -> diskann::ANNResult<diskann::graph::pipnn::Config> {
        let mut builder = diskann::graph::pipnn::Builder::from_graph(graph);
        if let Some(c_max) = self.c_max {
            builder = builder.c_max(c_max);
        }
        if let Some(p_samp) = self.p_samp {
            builder = builder.p_samp(p_samp);
        }
        if let Some(max_leaders) = self.max_leaders {
            builder = builder.max_leaders(max_leaders);
        }
        if let Some(fanout) = &self.fanout {
            builder = builder.fanout(fanout.clone());
        }
        if let Some(max_depth) = self.max_depth {
            builder = builder.max_depth(max_depth);
        }
        if let Some(leaf_k) = self.leaf_k {
            builder = builder.leaf_k(leaf_k);
        }
        if let Some(hash_prune) = &self.hash_prune {
            builder = builder.hash_prune(diskann::graph::pipnn::HashPruneConfig {
                num_hash_planes: hash_prune.num_hash_planes,
                l_max: hash_prune.l_max,
                final_prune: hash_prune.final_prune,
            });
        }
        Ok(builder.build()?)
    }
}

#[cfg(all(test, feature = "pipnn"))]
mod tests {
    use super::*;

    #[test]
    fn set_parameters_reach_the_pipnn_builder() {
        let graph = diskann::graph::config::Builder::new(
            16,
            diskann::graph::config::MaxDegree::default_slack(),
            32,
            diskann_vector::distance::Metric::L2.into(),
        )
        .build()
        .unwrap();
        let parameters: BuildAlgorithm = serde_json::from_value(serde_json::json!({
            "algorithm": "PiPNN",
            "leaf_k": 3,
            "hash_prune": {}
        }))
        .unwrap();
        let BuildAlgorithm::PiPNN(parameters) = parameters else {
            panic!("expected PiPNN parameters");
        };

        let actual = parameters.config(&graph).unwrap();
        let expected = diskann::graph::pipnn::Builder::from_graph(&graph)
            .leaf_k(NonZeroUsize::new(3).unwrap())
            .hash_prune(diskann::graph::pipnn::HashPruneConfig {
                num_hash_planes: 12,
                l_max: 64,
                final_prune: true,
            })
            .seed(actual.seed())
            .build()
            .unwrap();

        assert_eq!(actual, expected);
    }
}
