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

//////////////////////////////////
// diskann-record Save/Load     //
//////////////////////////////////
//
// The wire format preserves the same fields that `IndexConfiguration::new` takes
// (`config`, `num_threads`, `dist_metric`, `dim`, `max_points`, `num_frozen_pts`) plus
// `random_seed`, because the seed is part of reproducibility. The prefetch tunables
// (`prefetch_lookahead`, `prefetch_cache_line_level`) are intentionally not persisted;
// they are deployment knobs, not part of the index itself, so loaders apply their own
// defaults (`None`).

impl diskann_record::save::Save for IndexConfiguration {
    const VERSION: diskann_record::Version = diskann_record::Version::new(0, 0, 0);

    fn save(
        &self,
        context: diskann_record::save::Context<'_>,
    ) -> diskann_record::save::Result<diskann_record::save::Record<'_>> {
        Ok(diskann_record::save_fields!(
            self,
            context,
            [
                config,
                num_threads,
                dist_metric,
                dim,
                max_points,
                num_frozen_pts,
                random_seed,
            ]
        ))
    }
}

impl diskann_record::load::Load<'_> for IndexConfiguration {
    const VERSION: diskann_record::Version = diskann_record::Version::new(0, 0, 0);

    fn load(
        object: diskann_record::load::Object<'_>,
    ) -> diskann_record::load::Result<Self> {
        diskann_record::load_fields!(
            object,
            [
                config: Config,
                num_threads: usize,
                dist_metric: Metric,
                dim: usize,
                max_points: usize,
                num_frozen_pts: NonZeroUsize,
                random_seed: Option<u64>,
            ]
        );
        Ok(Self {
            config,
            num_threads,
            dist_metric,
            dim,
            max_points,
            num_frozen_pts,
            prefetch_lookahead: None,
            prefetch_cache_line_level: None,
            random_seed,
        })
    }

    fn load_legacy(
        _object: diskann_record::load::Object<'_>,
    ) -> diskann_record::load::Result<Self> {
        Err(diskann_record::load::error::Kind::UnknownVersion.into())
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

    /////////////////////////////////
    // diskann-record round-trips //
    /////////////////////////////////

    fn round_trip_helper<T>(value: &T) -> T
    where
        T: diskann_record::save::Saveable + for<'a> diskann_record::load::Loadable<'a>,
    {
        let dir = tempfile::tempdir().expect("tempdir");
        let manifest = dir.path().join("manifest.json");
        diskann_record::save::save_to_disk(value, dir.path(), &manifest)
            .expect("save_to_disk");
        diskann_record::load::load_from_disk::<T>(&manifest, dir.path())
            .expect("load_from_disk")
    }

    #[test]
    fn index_configuration_round_trips_minimal() {
        let original = IndexConfiguration::new(Metric::L2, 128, 1000, ONE, 1, config());
        assert_eq!(original, round_trip_helper(&original));
    }

    #[test]
    fn index_configuration_round_trips_preserves_random_seed() {
        let original = IndexConfiguration::new(Metric::Cosine, 64, 500, ONE, 4, config())
            .with_pseudo_rng_from_seed(0xDEAD_BEEF_CAFE_F00D);
        let restored = round_trip_helper(&original);
        assert_eq!(original, restored);
        assert_eq!(restored.random_seed, Some(0xDEAD_BEEF_CAFE_F00D));
    }

    #[test]
    fn index_configuration_round_trips_drops_prefetch_fields() {
        // Build a config with prefetch tunables set; they should NOT be persisted, so
        // the loaded copy will differ from the original on those fields only.
        let original = IndexConfiguration::new(Metric::L2, 128, 1000, ONE, 1, config())
            .with_prefetch_lookahead(NonZeroUsize::new(8))
            .with_prefetch_cache_line_level(Some(PrefetchCacheLineLevel::CacheLine8));
        let restored = round_trip_helper(&original);
        assert_eq!(restored.prefetch_lookahead, None);
        assert_eq!(restored.prefetch_cache_line_level, None);

        // Everything else still matches.
        let mut expected = original.clone();
        expected.prefetch_lookahead = None;
        expected.prefetch_cache_line_level = None;
        assert_eq!(expected, restored);
    }
}
