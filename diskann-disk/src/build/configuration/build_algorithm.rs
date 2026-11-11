/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Graph-build algorithm of a disk index.

#[cfg(feature = "pipnn")]
use std::num::NonZeroUsize;

use serde::{Deserialize, Serialize};

/// Graph-build algorithm of a disk index.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "algorithm")]
#[non_exhaustive]
pub enum BuildAlgorithm {
    /// Vamana: insert the points one by one, searching the graph for each.
    #[default]
    Vamana,

    /// PiPNN: build the graph from overlapping leaves in one pass.
    ///
    /// PiPNN reads the full-precision rows, so the build quantization must be
    /// [`QuantizationType::FP`](crate::QuantizationType::FP). PiPNN does not
    /// read the build memory limit.
    #[cfg(feature = "pipnn")]
    PiPNN(PiPNNParameters),
}

/// PiPNN parameters of a disk index build.
///
/// The index configuration supplies the prune policy, metric and seed. A
/// parameter that is not set keeps the default of
/// [`diskann::graph::pipnn::Builder`], which also describes each parameter.
#[cfg(feature = "pipnn")]
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PiPNNParameters {
    /// See [`diskann::graph::pipnn::Builder::c_max`].
    pub c_max: Option<NonZeroUsize>,
    /// See [`diskann::graph::pipnn::Builder::p_samp`].
    pub p_samp: Option<f64>,
    /// See [`diskann::graph::pipnn::Builder::max_leaders`].
    pub max_leaders: Option<NonZeroUsize>,
    /// See [`diskann::graph::pipnn::Builder::fanout`].
    pub fanout: Option<Vec<NonZeroUsize>>,
    /// See [`diskann::graph::pipnn::Builder::max_depth`].
    pub max_depth: Option<NonZeroUsize>,
    /// See [`diskann::graph::pipnn::Builder::leaf_k`].
    pub leaf_k: Option<NonZeroUsize>,
}

#[cfg(feature = "pipnn")]
impl PiPNNParameters {
    /// Start a PiPNN configuration from the prune policy of `graph` and the
    /// parameters that are set.
    pub fn builder(&self, graph: &diskann::graph::Config) -> diskann::graph::pipnn::Builder {
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
        builder
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_algorithm_tag_selects_the_variant() {
        let vamana: BuildAlgorithm = serde_json::from_str(r#"{"algorithm":"Vamana"}"#).unwrap();
        assert_eq!(vamana, BuildAlgorithm::Vamana);

        #[cfg(feature = "pipnn")]
        {
            let pipnn: BuildAlgorithm =
                serde_json::from_str(r#"{"algorithm":"PiPNN","c_max":64,"fanout":[8,3]}"#).unwrap();
            let nz = |value| NonZeroUsize::new(value).unwrap();
            let expected = PiPNNParameters {
                c_max: Some(nz(64)),
                fanout: Some(vec![nz(8), nz(3)]),
                ..PiPNNParameters::default()
            };
            assert_eq!(pipnn, BuildAlgorithm::PiPNN(expected));

            let unknown = serde_json::from_str::<BuildAlgorithm>(r#"{"algorithm":"PiPNN","l":1}"#);
            assert!(unknown.is_err());
        }
    }

    #[cfg(feature = "pipnn")]
    #[test]
    fn unset_parameters_keep_the_pipnn_defaults() {
        let graph = diskann::graph::config::Builder::new(
            16,
            diskann::graph::config::MaxDegree::default_slack(),
            32,
            diskann_vector::distance::Metric::L2.into(),
        )
        .build()
        .unwrap();
        let leaf_k = NonZeroUsize::new(3).unwrap();
        let parameters = PiPNNParameters {
            leaf_k: Some(leaf_k),
            ..PiPNNParameters::default()
        };

        let actual = parameters.builder(&graph).build().unwrap();
        let expected = diskann::graph::pipnn::Builder::from_graph(&graph)
            .leaf_k(leaf_k)
            .seed(actual.seed())
            .build()
            .unwrap();

        assert_eq!(actual, expected);
    }
}
