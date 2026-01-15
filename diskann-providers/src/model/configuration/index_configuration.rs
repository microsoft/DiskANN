/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
#![warn(missing_debug_implementations, missing_docs)]

//! Index configuration.

use std::num::NonZeroUsize;

use diskann::graph::Config;
use diskann_vector::distance::Metric;

use crate::model::graph::provider::async_::PrefetchCacheLineLevel;

/// The index configuration
#[derive(Debug, Clone, PartialEq)]
pub struct IndexConfiguration {
    /// The configuration that will be forwarded to the index.
    pub config: Config,

    /// The number of threads to allocate resources for.
    pub num_threads: usize,

    /// Distance metric
    pub dist_metric: Metric,

    /// Dimension of the raw data
    pub dim: usize,

    /// Total number of points in given data set
    pub max_points: usize,

    /// Number of points which are used as initial candidates when iterating to
    /// closest point(s). These are not visible externally and won't be returned
    /// by search. DiskANN forces at least 1 frozen point for dynamic index.
    /// The frozen points have consecutive locations.
    pub num_frozen_pts: NonZeroUsize,

    /// Prefetch lookahead.
    /// This controls how many vectors ahead we will prefetch data with providers that implement prefetching
    /// when doing operations on a sequence of vectors.
    pub prefetch_lookahead: Option<NonZeroUsize>,

    /// prefetch cache line level.
    /// This controls the granularity of prefetching.
    /// It can be set to None for default behavior (8 cache lines by default)
    pub prefetch_cache_line_level: Option<PrefetchCacheLineLevel>,

    /// Optional seed for random number generator.
    pub random_seed: Option<u64>,
}

impl IndexConfiguration {
    /// Create IndexConfiguration instance
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dist_metric: Metric,
        dim: usize,
        max_points: usize,
        num_frozen_pts: NonZeroUsize,
        num_threads: usize,
        config: Config,
    ) -> Self {
        Self {
            config,
            num_threads,
            dist_metric,
            dim,
            max_points,
            num_frozen_pts,
            prefetch_lookahead: None,
            prefetch_cache_line_level: None,
            random_seed: None,
        }
    }

    /// Set the default random seed.
    pub fn with_pseudo_rng(mut self) -> Self {
        self.random_seed = Some(crate::utils::DEFAULT_SEED_FOR_TESTS);
        self
    }

    /// Set random seed.
    pub fn with_pseudo_rng_from_seed(mut self, random_seed: u64) -> Self {
        self.random_seed = Some(random_seed);
        self
    }

    /// sets prefetch lookahead using builder pattern
    pub fn with_prefetch_lookahead(mut self, prefetch_lookahead: Option<NonZeroUsize>) -> Self {
        self.prefetch_lookahead = prefetch_lookahead;
        self
    }

    /// sets prefetch cache line level using builder pattern
    pub fn with_prefetch_cache_line_level(
        mut self,
        prefetch_cache_line_level: Option<PrefetchCacheLineLevel>,
    ) -> Self {
        self.prefetch_cache_line_level = prefetch_cache_line_level;
        self
    }

    /// Get the size of adjacency list that we build out.
    pub fn write_range(&self) -> usize {
        self.config.pruned_degree().get()
    }
}

#[cfg(test)]
mod tests {
    use diskann::utils::ONE;
    use diskann_vector::distance::Metric;

    use super::*;

    fn config() -> Config {
        diskann::graph::config::Builder::new(
            10,
            diskann::graph::config::MaxDegree::default_slack(),
            10,
            Metric::L2.into(),
        )
        .build()
        .unwrap()
    }

    fn sample_test_index() -> IndexConfiguration {
        IndexConfiguration {
            config: config(),
            num_threads: 1,
            dist_metric: Metric::L2,
            dim: 128,
            max_points: 1000,
            num_frozen_pts: ONE,
            prefetch_lookahead: None,
            prefetch_cache_line_level: None,
            random_seed: None,
        }
    }

    #[test]
    fn test_index_configuration() {
        let index_configuration = IndexConfiguration::new(Metric::L2, 128, 1000, ONE, 1, config());

        assert_eq!(index_configuration, sample_test_index());
    }

    #[test]
    fn test_with_prefetch_lookahead_builder() {
        let index_configuration = IndexConfiguration::new(Metric::L2, 128, 1000, ONE, 1, config())
            .with_prefetch_lookahead(NonZeroUsize::new(8));

        let mut expected_index_configuration = sample_test_index();
        expected_index_configuration.prefetch_lookahead = NonZeroUsize::new(8);
        assert_eq!(index_configuration, expected_index_configuration,);
    }

    #[test]
    fn test_with_prefetch_cache_line_level_builder() {
        let index_configuration = IndexConfiguration::new(Metric::L2, 128, 1000, ONE, 1, config())
            .with_prefetch_cache_line_level(Some(PrefetchCacheLineLevel::CacheLine8));

        let mut expected_index_configuration = sample_test_index();
        expected_index_configuration.prefetch_cache_line_level =
            Some(PrefetchCacheLineLevel::CacheLine8);
        assert_eq!(index_configuration, expected_index_configuration,);
    }

    #[test]
    fn should_get_write_range_successfully() {
        let index_configuration = IndexConfiguration::new(Metric::L2, 128, 1000, ONE, 1, config());

        assert_eq!(
            index_configuration.write_range(),
            index_configuration.config.pruned_degree().get()
        );
    }
}
