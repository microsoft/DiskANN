/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::{
    graph::AdjacencyList,
    provider::{self as core_provider, DefaultContext},
};
use diskann_utils::future::AsyncFriendly;
use diskann_vector::distance::Metric;

use crate::model::graph::provider::async_::{
    common::FullPrecision,
    debug_provider::{self, DebugProvider},
};

use super::{
    bf_cache::{self, Cache},
    error::CacheAccessError,
    provider::{self as cache_provider, CachingError, NeighborStatus},
    utils::{CacheKey, Graph, HitStats, KeyGen, LocalStats},
};

///////////////////
// Example Cache //
///////////////////

/// A representation for the adjacency list term.
///
/// The `repr(u8)` allows this to be converted to an integer and thus be used in a
/// `bytemuck::Pod` struct.
#[derive(Debug, Clone, Copy)]
#[repr(u8)]
enum Tag {
    AdjacencyList = 0,
    Vector = 1,
}

const TAGS: [Tag; 2] = [Tag::AdjacencyList, Tag::Vector];

impl KeyGen<u32> for Tag {
    type Key = CacheKey<u32>;
    fn generate(&self, id: u32) -> Self::Key {
        CacheKey::new(id, (*self as u8).into())
    }
}

/// An example cache for compatibility with the full-precision provider.
///
/// This is meant largely for demonstration purposes, showing how to build a cache tailored
/// for a certain provider.
#[derive(Debug)]
pub struct ExampleCache {
    cache: Cache,
    // This field records a collection of uncacheable IDs to ensure the cache skipping
    // logic in the provider works as expected.
    uncacheable: Option<Vec<u32>>,
    neighbor_stats: HitStats,
    vector_stats: HitStats,
}

impl ExampleCache {
    /// Construct a new cache with the given capacity.
    pub fn new(
        bytes: diskann_quantization::num::PowerOfTwo,
        uncacheable: Option<Vec<u32>>,
    ) -> Self {
        Self {
            cache: Cache::new(bytes).unwrap(),
            uncacheable,
            neighbor_stats: HitStats::new(),
            vector_stats: HitStats::new(),
        }
    }

    /// Invalidate all cached items associated with `id`.
    fn invalidate(&self, id: u32) {
        // Since we use a unified cache under the scenes type the list types disambiguated
        // by a tag - we need to delete all possible tags.
        for tag in TAGS {
            self.cache.delete(tag.generate(id))
        }
    }

    /// Return a cache accessor for adjacency list terms.
    fn neighbors(&self, max_degree: usize) -> Graph<'_, u32, Tag> {
        Graph::new(
            &self.cache,
            max_degree,
            Tag::AdjacencyList,
            &self.neighbor_stats,
        )
    }
}

impl cache_provider::Evict<u32> for ExampleCache {
    fn evict(&self, id: u32) {
        self.invalidate(id)
    }
}

/// The `CacheAccessor` is the `C` in `cache_provider::CachingAccessor` and interfaces the
/// accessor with the underlying cache.
#[derive(Debug)]
pub struct CacheAccessor<'a, T> {
    graph: Graph<'a, u32, Tag>,
    cacher: T,
    stats: LocalStats<'a>,
    uncacheable: Option<&'a [u32]>,
    /// Key generation for the element being accessed.
    keygen: Tag,
}

impl<T> cache_provider::ElementCache<u32, diskann_utils::lifetime::Slice<T>>
    for CacheAccessor<'_, bf_cache::VecCacher<T>>
where
    T: AsyncFriendly + bytemuck::Pod + Default + std::fmt::Debug,
{
    type Error = CacheAccessError;

    fn get_cached(&mut self, k: u32) -> Result<Option<&[T]>, CacheAccessError> {
        match self
            .graph
            .cache()
            .get(self.keygen.generate(k), &mut self.cacher)
        {
            Ok(Some(value)) => {
                self.stats.hit();
                Ok(Some(value))
            }
            Ok(None) => {
                self.stats.miss();
                Ok(None)
            }
            Err(err) => Err(CacheAccessError::read(k, err)),
        }
    }

    fn set_cached(&mut self, k: u32, element: &&[T]) -> Result<(), CacheAccessError> {
        self.graph
            .cache()
            .set(self.keygen.generate(k), &mut self.cacher, element)
            .map_err(|err| CacheAccessError::write(k, err))
    }
}

impl<T> cache_provider::NeighborCache<u32> for CacheAccessor<'_, T>
where
    T: AsyncFriendly,
{
    type Error = CacheAccessError;

    fn try_get_neighbors(
        &mut self,
        id: u32,
        neighbors: &mut AdjacencyList<u32>,
    ) -> Result<NeighborStatus, CacheAccessError> {
        if let Some(uncacheable) = self.uncacheable
            && uncacheable.contains(&id)
        {
            self.graph.stats_mut().miss();
            Ok(NeighborStatus::Uncacheable)
        } else {
            self.graph.try_get_neighbors(id, neighbors)
        }
    }

    fn set_neighbors(&mut self, id: u32, neighbors: &[u32]) -> Result<(), CacheAccessError> {
        self.graph.set_neighbors(id, neighbors)
    }

    fn invalidate_neighbors(&mut self, id: u32) {
        self.graph.invalidate_neighbors(id)
    }
}

/////////////////////
// Provider Bridge //
/////////////////////

impl<'a> cache_provider::AsCacheAccessorFor<'a, debug_provider::FullAccessor<'a>> for ExampleCache {
    type Accessor = CacheAccessor<'a, bf_cache::VecCacher<f32>>;
    type Error = diskann::error::Infallible;
    fn as_cache_accessor_for(
        &'a self,
        inner: debug_provider::FullAccessor<'a>,
    ) -> Result<
        cache_provider::CachingAccessor<debug_provider::FullAccessor<'a>, Self::Accessor>,
        Self::Error,
    > {
        let provider = inner.provider();
        let cache_accessor = CacheAccessor {
            graph: self.neighbors(provider.max_degree()),
            cacher: bf_cache::VecCacher::<f32>::new(provider.dim()),
            uncacheable: self.uncacheable.as_deref(),
            stats: LocalStats::new(&self.vector_stats),
            keygen: Tag::Vector,
        };
        Ok(cache_provider::CachingAccessor::new(inner, cache_accessor))
    }
}

impl<'a> cache_provider::CachedFillSet<CacheAccessor<'a, bf_cache::VecCacher<f32>>>
    for debug_provider::FullAccessor<'a>
{
}

impl<'a> cache_provider::CachedAsElement<&'a [f32], CacheAccessor<'a, bf_cache::VecCacher<f32>>>
    for debug_provider::FullAccessor<'a>
{
    type Error = CachingError<Self::GetError, CacheAccessError>;
    async fn cached_as_element<'b>(
        &'b mut self,
        cache: &'b mut CacheAccessor<'a, bf_cache::VecCacher<f32>>,
        _vector: &'a [f32],
        id: u32,
    ) -> Result<Self::Element<'b>, Self::Error> {
        cache_provider::get_or_insert(self, cache, id).await
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use diskann::{
        graph::{DiskANNIndex, glue::SearchStrategy},
        provider::{
            Accessor, DataProvider, Delete, NeighborAccessor, NeighborAccessorMut, SetElement,
        },
        utils::async_tools,
    };
    use diskann_quantization::num::PowerOfTwo;
    use diskann_utils::views::Matrix;
    use diskann_vector::{PureDistanceFunction, distance::SquaredL2};
    use rstest::rstest;

    use crate::{
        index::diskann_async::{self, tests as async_tests},
        model::graph::provider::async_::caching::provider::{AsCacheAccessorFor, CachingProvider},
        utils as crate_utils,
    };

    const CTX: &DefaultContext = &DefaultContext;

    fn test_provider(
        uncacheable: Option<Vec<u32>>,
    ) -> CachingProvider<DebugProvider, ExampleCache> {
        let dim = 2;

        let config = debug_provider::DebugConfig {
            start_id: u32::MAX,
            start_point: vec![0.0; dim],
            max_degree: 10,
            metric: Metric::L2,
        };

        let table = diskann_async::train_pq(
            Matrix::new(0.0, 1, dim).as_view(),
            2.min(dim), // Number of PQ chunks is bounded by the dimension.
            &mut crate::utils::create_rnd_from_seed_in_tests(0),
            1usize,
        )
        .unwrap();

        CachingProvider::new(
            DebugProvider::new(config, table).unwrap(),
            ExampleCache::new(PowerOfTwo::new(1024 * 16).unwrap(), uncacheable),
        )
    }

    #[tokio::test]
    async fn basic_operations_happy_path() {
        let provider = test_provider(None);
        let ctx = &DefaultContext;

        // Translations do not yet exist.
        assert!(provider.to_external_id(ctx, 0).is_err());
        assert!(provider.to_internal_id(ctx, &0).is_err());

        assert_eq!(provider.inner().data_writes.get(), 0);
        provider.set_element(CTX, &0, &[1.0, 2.0]).await.unwrap();
        assert_eq!(provider.inner().data_writes.get(), 1 /* increased */);

        assert_eq!(provider.to_external_id(ctx, 0).unwrap(), 0);
        assert_eq!(provider.to_internal_id(ctx, &0).unwrap(), 0);

        // Retrieval of a valid element.
        let mut accessor = provider
            .cache()
            .as_cache_accessor_for(debug_provider::FullAccessor::new(provider.inner()))
            .unwrap();

        // Hit served from the underlying provider.
        assert_eq!(provider.inner().full_reads.get(), 0);
        let element = accessor.get_element(0).await.unwrap();
        assert_eq!(element, &[1.0, 2.0]);
        assert_eq!(provider.inner().full_reads.get(), 1);
        assert_eq!(
            accessor.cache().stats.get_local_misses(),
            1, /* increased */
        );
        assert_eq!(accessor.cache().stats.get_local_hits(), 0);

        // This time, the hit is served from the underlying cache.
        let element = accessor.get_element(0).await.unwrap();
        assert_eq!(element, &[1.0, 2.0]);
        assert_eq!(provider.inner().full_reads.get(), 1);
        assert_eq!(accessor.cache().stats.get_local_misses(), 1);
        assert_eq!(
            accessor.cache().stats.get_local_hits(),
            1, /* increased */
        );

        // Adjacency List from Underlying
        assert_eq!(provider.inner().neighbor_writes.get(), 0);
        accessor.set_neighbors(0, &[1, 2, 3]).await.unwrap();
        assert_eq!(
            provider.inner().neighbor_writes.get(),
            1, /* increased */
        );

        let mut list = AdjacencyList::new();
        assert_eq!(provider.inner().neighbor_reads.get(), 0);
        accessor.get_neighbors(0, &mut list).await.unwrap();
        assert_eq!(
            provider.inner().neighbor_reads.get(),
            1, /* increased */
        );
        assert_eq!(
            accessor.cache().graph.stats().get_local_misses(),
            1, /* increased */
        );
        assert_eq!(accessor.cache().graph.stats().get_local_hits(), 0);
        assert_eq!(&*list, &[1, 2, 3]);

        // Adjacency List From Cache
        list.clear();
        accessor.get_neighbors(0, &mut list).await.unwrap();
        assert_eq!(&*list, &[1, 2, 3]);
        assert_eq!(provider.inner().neighbor_reads.get(), 1);
        assert_eq!(accessor.cache().graph.stats().get_local_misses(), 1);
        assert_eq!(
            accessor.cache().graph.stats().get_local_hits(),
            1, /* increased */
        );

        // If we invalidate the key - these elements should be retrieved from the backing
        // provider instead.
        provider.cache().invalidate(0);

        let element = accessor.get_element(0).await.unwrap();
        assert_eq!(element, &[1.0, 2.0]);
        assert_eq!(provider.inner().full_reads.get(), 2 /* increased */,);
        assert_eq!(
            accessor.cache().stats.get_local_misses(),
            2, /* increased */
        );
        assert_eq!(accessor.cache().stats.get_local_hits(), 1);

        // Once more from the cache.
        let element = accessor.get_element(0).await.unwrap();
        assert_eq!(element, &[1.0, 2.0]);
        assert_eq!(provider.inner().full_reads.get(), 2);
        assert_eq!(accessor.cache().stats.get_local_misses(), 2);
        assert_eq!(
            accessor.cache().stats.get_local_hits(),
            2, /* increased */
        );

        list.clear();
        accessor.get_neighbors(0, &mut list).await.unwrap();
        assert_eq!(&*list, &[1, 2, 3]);
        assert_eq!(
            provider.inner().neighbor_reads.get(),
            2, /* increased */
        );
        assert_eq!(
            accessor.cache().graph.stats().get_local_misses(),
            2, /* increased */
        );
        assert_eq!(accessor.cache().graph.stats().get_local_hits(), 1);

        // Setting adjacency lists invalidates the cache.
        accessor.set_neighbors(0, &[2, 3, 4]).await.unwrap();
        accessor.get_neighbors(0, &mut list).await.unwrap();
        assert_eq!(&*list, &[2, 3, 4]);
        assert_eq!(
            provider.inner().neighbor_writes.get(),
            2, /* increased */
        );
        assert_eq!(
            provider.inner().neighbor_reads.get(),
            3, /* increased */
        );
        assert_eq!(
            accessor.cache().graph.stats().get_local_misses(),
            3, /* increased */
        );
        assert_eq!(accessor.cache().graph.stats().get_local_hits(), 1);

        accessor.append_vector(0, &[1]).await.unwrap();
        accessor.get_neighbors(0, &mut list).await.unwrap();
        assert_eq!(&*list, &[2, 3, 4, 1]);

        assert_eq!(
            provider.inner().neighbor_writes.get(),
            3, /* increased */
        );
        assert_eq!(
            provider.inner().neighbor_reads.get(),
            4, /* increased */
        );
        assert_eq!(
            accessor.cache().graph.stats().get_local_misses(),
            4, /* increased */
        );
        assert_eq!(accessor.cache().graph.stats().get_local_hits(), 1);

        // Deletion.
        assert_eq!(
            provider.status_by_internal_id(CTX, 0).await.unwrap(),
            core_provider::ElementStatus::Valid
        );
        assert_eq!(
            provider.status_by_external_id(CTX, &0).await.unwrap(),
            core_provider::ElementStatus::Valid
        );
        assert!(provider.status_by_internal_id(CTX, 1).await.is_err());
        assert!(provider.status_by_external_id(CTX, &1).await.is_err());

        provider.delete(CTX, &0).await.unwrap();

        assert_eq!(
            provider.status_by_internal_id(CTX, 0).await.unwrap(),
            core_provider::ElementStatus::Deleted
        );
        assert_eq!(
            provider.status_by_external_id(CTX, &0).await.unwrap(),
            core_provider::ElementStatus::Deleted
        );
        assert!(provider.status_by_internal_id(CTX, 1).await.is_err());
        assert!(provider.status_by_external_id(CTX, &1).await.is_err());

        // Access the deleted element is still valid.
        let element = accessor.get_element(0).await.unwrap();
        assert_eq!(element, &[1.0, 2.0]);
        assert_eq!(provider.inner().full_reads.get(), 2);
        assert_eq!(accessor.cache().stats.get_local_misses(), 2);
        assert_eq!(
            accessor.cache().stats.get_local_hits(),
            3, /* increased */
        );

        accessor.get_neighbors(0, &mut list).await.unwrap();
        assert_eq!(&*list, &[2, 3, 4, 1]);
        assert_eq!(provider.inner().neighbor_writes.get(), 3);
        assert_eq!(provider.inner().neighbor_reads.get(), 4);
        assert_eq!(accessor.cache().graph.stats().get_local_misses(), 4);
        assert_eq!(
            accessor.cache().graph.stats().get_local_hits(),
            2, /* increased */
        );

        provider.release(CTX, 0).await.unwrap();
        assert!(provider.status_by_internal_id(CTX, 0).await.is_err());
        assert!(provider.status_by_external_id(CTX, &0).await.is_err());

        assert!(accessor.get_element(0).await.is_err());
        assert_eq!(provider.inner().full_reads.get(), 2);
        assert_eq!(
            accessor.cache().stats.get_local_misses(),
            3 /* increased */
        );
        assert_eq!(accessor.cache().stats.get_local_hits(), 3);

        assert!(accessor.get_neighbors(0, &mut list).await.is_err());
        assert_eq!(provider.inner().neighbor_writes.get(), 3);
        assert_eq!(provider.inner().neighbor_reads.get(), 4);
        assert_eq!(
            accessor.cache().graph.stats().get_local_misses(),
            5 /* increased */
        );
        assert_eq!(accessor.cache().graph.stats().get_local_hits(), 2);

        // Ensure that the stats get properly recorded when the accessor is dropped.
        let vector_hits = accessor.cache().stats.get_local_hits();
        let vector_misses = accessor.cache().stats.get_local_misses();
        let neighbor_hits = accessor.cache().graph.stats().get_local_hits();
        let neighbor_misses = accessor.cache().graph.stats().get_local_misses();

        std::mem::drop(accessor);

        assert_eq!(provider.cache().vector_stats.get_hits(), vector_hits);
        assert_eq!(provider.cache().vector_stats.get_misses(), vector_misses);

        assert_eq!(provider.cache().neighbor_stats.get_hits(), neighbor_hits);
        assert_eq!(
            provider.cache().neighbor_stats.get_misses(),
            neighbor_misses
        );
    }

    #[tokio::test]
    async fn test_uncacheable() {
        // Test that returning `Uncacheable` for an adjacency list is handled correctly by
        // the provider and a call to `set_neighbors` is not made.
        let uncacheable = u32::MAX;
        let provider = test_provider(Some(vec![uncacheable]));

        let mut accessor = provider
            .cache()
            .as_cache_accessor_for(debug_provider::FullAccessor::new(provider.inner()))
            .unwrap();

        provider.set_element(CTX, &0, &[1.0, 2.0]).await.unwrap();

        //---------------//
        // Cacheable IDs //
        //---------------//

        // Adjacency List from Underlying
        assert_eq!(provider.inner().neighbor_writes.get(), 0);
        accessor.set_neighbors(0, &[1, 2, 3]).await.unwrap();
        assert_eq!(
            provider.inner().neighbor_writes.get(),
            1, /* increased */
        );

        let mut list = AdjacencyList::new();
        assert_eq!(provider.inner().neighbor_reads.get(), 0);
        accessor.get_neighbors(0, &mut list).await.unwrap();
        assert_eq!(
            provider.inner().neighbor_reads.get(),
            1, /* increased */
        );
        assert_eq!(
            accessor.cache().graph.stats().get_local_misses(),
            1, /* increased */
        );
        assert_eq!(accessor.cache().graph.stats().get_local_hits(), 0);
        assert_eq!(&*list, &[1, 2, 3]);

        // Adjacency List From Cache
        list.clear();
        accessor.get_neighbors(0, &mut list).await.unwrap();
        assert_eq!(&*list, &[1, 2, 3]);
        assert_eq!(provider.inner().neighbor_reads.get(), 1);
        assert_eq!(accessor.cache().graph.stats().get_local_misses(), 1);
        assert_eq!(
            accessor.cache().graph.stats().get_local_hits(),
            1, /* increased */
        );

        //-----------------//
        // Uncacheable IDs //
        //-----------------//

        assert_eq!(provider.inner().neighbor_writes.get(), 1);
        accessor.set_neighbors(uncacheable, &[4, 5]).await.unwrap();
        assert_eq!(
            provider.inner().neighbor_writes.get(),
            2, /* increased */
        );

        // The retrieval is served by the inner provider.
        assert_eq!(provider.inner().neighbor_reads.get(), 1);
        accessor
            .get_neighbors(uncacheable, &mut list)
            .await
            .unwrap();
        assert_eq!(
            provider.inner().neighbor_reads.get(),
            2, /* increased */
        );
        assert_eq!(
            accessor.cache().graph.stats().get_local_misses(),
            2, /* increased */
        );
        assert_eq!(accessor.cache().graph.stats().get_local_hits(), 1);
        assert_eq!(&*list, &[4, 5]);

        // Again, retrieval is served by the inner provider.
        assert_eq!(provider.inner().neighbor_reads.get(), 2);
        accessor
            .get_neighbors(uncacheable, &mut list)
            .await
            .unwrap();
        assert_eq!(
            provider.inner().neighbor_reads.get(),
            3, /* increased */
        );
        assert_eq!(
            accessor.cache().graph.stats().get_local_misses(),
            3, /* increased */
        );
        assert_eq!(accessor.cache().graph.stats().get_local_hits(), 1);
        assert_eq!(&*list, &[4, 5]);
    }

    //----------------//
    // Standard Tests //
    //----------------//

    #[rstest]
    #[case(1, 100)]
    #[case(3, 7)]
    #[case(4, 5)]
    #[tokio::test]
    async fn grid_search(#[case] dim: usize, #[case] grid_size: usize) {
        let l = 10;
        let max_degree = 2 * dim;
        let num_points = (grid_size).pow(dim as u32);
        let start_id = u32::MAX;
        let start_point = vec![grid_size as f32; dim];
        let metric = Metric::L2;
        let cache_size = PowerOfTwo::new(128 * 1024).unwrap();

        let index_config = diskann::graph::config::Builder::new(
            max_degree,
            diskann::graph::config::MaxDegree::default_slack(),
            l,
            metric.into(),
        )
        .build()
        .unwrap();

        let test_config = debug_provider::DebugConfig {
            start_id,
            start_point: start_point.clone(),
            max_degree: index_config.max_degree().get(),
            metric,
        };

        let mut vectors = <f32 as async_tests::GenerateGrid>::generate_grid(dim, grid_size);
        let table = diskann_async::train_pq(
            async_tests::squish(vectors.iter(), dim).as_view(),
            2.min(dim),
            &mut crate::utils::create_rnd_from_seed_in_tests(0),
            1usize,
        )
        .unwrap();

        let provider = CachingProvider::new(
            DebugProvider::new(test_config, table).unwrap(),
            ExampleCache::new(cache_size, None),
        );
        let index = Arc::new(DiskANNIndex::new(index_config, provider, None));

        let adjacency_lists = match dim {
            1 => crate_utils::generate_1d_grid_adj_list(grid_size as u32),
            3 => crate_utils::genererate_3d_grid_adj_list(grid_size as u32),
            4 => crate_utils::generate_4d_grid_adj_list(grid_size as u32),
            _ => panic!("Unsupported number of dimensions"),
        };
        assert_eq!(adjacency_lists.len(), num_points);
        assert_eq!(vectors.len(), num_points);

        let strategy = cache_provider::Cached::new(FullPrecision);
        async_tests::populate_data(index.provider(), CTX, &vectors).await;
        {
            // Note: Without the fully qualified syntax - this fails to compile.
            let mut accessor = <cache_provider::Cached<FullPrecision> as SearchStrategy<
                cache_provider::CachingProvider<debug_provider::DebugProvider, ExampleCache>,
                [f32],
            >>::search_accessor(&strategy, index.provider(), CTX)
            .unwrap();
            async_tests::populate_graph(&mut accessor, &adjacency_lists).await;

            accessor
                .set_neighbors(start_id, &[num_points as u32 - 1])
                .await
                .unwrap();
        }

        let corpus: diskann_utils::views::Matrix<f32> = async_tests::squish(vectors.iter(), dim);
        let mut paged_tests = Vec::new();

        // Test with the zero query.
        let query = vec![0.0; dim];
        let gt = crate::test_utils::groundtruth(corpus.as_view(), &query, |a, b| {
            SquaredL2::evaluate(a, b)
        });
        paged_tests.push(async_tests::PagedSearch::new(query, gt));

        // Test with the start point to ensure it is filtered out.
        let gt = crate::test_utils::groundtruth(corpus.as_view(), &start_point, |a, b| {
            SquaredL2::evaluate(a, b)
        });
        paged_tests.push(async_tests::PagedSearch::new(start_point.clone(), gt));

        // Unfortunately - this is needed for the `check_grid_search` test.
        vectors.push(start_point.clone());
        async_tests::check_grid_search(&index, &vectors, &paged_tests, strategy, strategy).await;
    }

    fn check_stats(caching: &CachingProvider<DebugProvider, ExampleCache>) {
        let provider = caching.inner();
        let cache = caching.cache();

        println!("neighbor reads: {}", provider.neighbor_reads.get());
        println!("neighbor writes: {}", provider.neighbor_writes.get());
        println!("vector reads: {}", provider.full_reads.get());
        println!("vector writes: {}", provider.data_writes.get());

        println!("neighbor hits: {}", cache.neighbor_stats.get_hits());
        println!("neighbor misses: {}", cache.neighbor_stats.get_misses());
        println!("vector hits: {}", cache.vector_stats.get_hits());
        println!("vector misses: {}", cache.vector_stats.get_misses());

        // Neighbors
        assert_eq!(
            provider.neighbor_reads.get(),
            cache.neighbor_stats.get_misses()
        );

        // Vectors
        assert_eq!(provider.full_reads.get(), cache.vector_stats.get_misses());
    }

    #[rstest]
    #[tokio::test]
    async fn grid_search_with_build(
        #[values((1, 100), (3, 7), (4, 5))] dim_and_size: (usize, usize),
    ) {
        let dim = dim_and_size.0;
        let grid_size = dim_and_size.1;
        let l = 10;
        let start_id = u32::MAX;
        let start_point = vec![grid_size as f32; dim];
        let metric = Metric::L2;
        let cache_size = PowerOfTwo::new(128 * 1024).unwrap();

        // NOTE: Be careful changing `max_degree`. It needs to be high enough that the
        // graph is navigable, but low enough that the batch parallel handling inside
        // `multi_insert` is needed for the multi-insert graph to be navigable.
        //
        // With the current configured values, removing the other elements in the batch
        // from the visited set during `multi_insert` results in a graph failure.
        let max_degree = 2 * dim;
        let num_points = (grid_size).pow(dim as u32);

        let mut vectors = <f32 as async_tests::GenerateGrid>::generate_grid(dim, grid_size);
        let table = Arc::new(
            diskann_async::train_pq(
                async_tests::squish(vectors.iter(), dim).as_view(),
                2.min(dim),
                &mut crate::utils::create_rnd_from_seed_in_tests(0),
                1usize,
            )
            .unwrap(),
        );

        let index_config = diskann::graph::config::Builder::new_with(
            max_degree,
            diskann::graph::config::MaxDegree::default_slack(),
            l,
            metric.into(),
            |b| {
                b.max_minibatch_par(10);
            },
        )
        .build()
        .unwrap();

        let test_config = debug_provider::DebugConfig {
            start_id,
            start_point: start_point.clone(),
            max_degree: index_config.max_degree().get(),
            metric,
        };
        assert_eq!(vectors.len(), num_points);

        // This is a little subtle, but we need `vectors` to contain the start point as
        // its last element, but we **don't** want to include it in the index build.
        //
        // This basically means that we need to be careful with out index initialization.
        vectors.push(vec![grid_size as f32; dim]);

        // Initialize an index for a new round of building.
        let init_index = || {
            let provider = CachingProvider::new(
                DebugProvider::new(test_config.clone(), table.clone()).unwrap(),
                ExampleCache::new(cache_size, None),
            );
            Arc::new(DiskANNIndex::new(index_config.clone(), provider, None))
        };

        let strategy = cache_provider::Cached::new(FullPrecision);

        // Build with full-precision single insert
        {
            let index = init_index();
            for (i, v) in vectors.iter().take(num_points).enumerate() {
                index
                    .insert(strategy, CTX, &(i as u32), v.as_slice())
                    .await
                    .unwrap();
            }

            check_stats(index.provider());

            async_tests::check_grid_search(&index, &vectors, &[], strategy, strategy).await;
            check_stats(index.provider());
        }

        // Build with full-precision multi-insert
        {
            let index = init_index();
            let batch: Box<[_]> = vectors
                .iter()
                .take(num_points)
                .enumerate()
                .map(|(id, v)| async_tools::VectorIdBoxSlice::new(id as u32, v.as_slice().into()))
                .collect();

            index.multi_insert(strategy, CTX, batch).await.unwrap();

            async_tests::check_grid_search(&index, &vectors, &[], strategy, strategy).await;
            check_stats(index.provider());
        }
    }

    #[tokio::test]
    async fn test_inplace_delete_2d() {
        // create small index instance
        let metric = Metric::L2;
        let num_points = 4;
        let strategy = cache_provider::Cached::new(FullPrecision);
        let cache_size = PowerOfTwo::new(128 * 1024).unwrap();
        let start_id = num_points as u32;
        let start_point = vec![0.5, 0.5];
        let dim = start_point.len();

        let index_config = diskann::graph::config::Builder::new(
            4, // target_degree
            diskann::graph::config::MaxDegree::default_slack(),
            10, // l_build
            metric.into(),
        )
        .build()
        .unwrap();

        let test_config = debug_provider::DebugConfig {
            start_id,
            start_point: start_point.clone(),
            max_degree: index_config.max_degree().get(),
            metric,
        };

        // The contents of the table don't matter for this test because we use full
        // precision only.
        let table = diskann_async::train_pq(
            Matrix::new(0.5, 1, dim).as_view(),
            dim,
            &mut crate::utils::create_rnd_from_seed_in_tests(0),
            1usize,
        )
        .unwrap();

        let index = DiskANNIndex::new(
            index_config,
            CachingProvider::new(
                DebugProvider::new(test_config, table).unwrap(),
                ExampleCache::new(cache_size, None),
            ),
            None,
        );

        // vectors are the four corners of a square, with the start point in the middle
        // the middle point forms an edge to each corner, while corners form an edge
        // to their opposite vertex vertically as well as the middle
        let vectors = [
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ];
        let adjacency_lists = [
            AdjacencyList::from_iter_untrusted([4, 1]),
            AdjacencyList::from_iter_untrusted([4, 0]),
            AdjacencyList::from_iter_untrusted([4, 3]),
            AdjacencyList::from_iter_untrusted([4, 2]),
            AdjacencyList::from_iter_untrusted([0, 1, 2, 3]),
        ];

        // Note: Without the fully qualified syntax - this fails to compile.
        let mut accessor = <cache_provider::Cached<FullPrecision> as SearchStrategy<
            cache_provider::CachingProvider<debug_provider::DebugProvider, ExampleCache>,
            [f32],
        >>::search_accessor(&strategy, index.provider(), CTX)
        .unwrap();

        async_tests::populate_data(index.provider(), CTX, &vectors).await;
        async_tests::populate_graph(&mut accessor, &adjacency_lists).await;

        index
            .inplace_delete(
                strategy,
                CTX,
                &3, // id to delete
                3,  // num_to_replace
                diskann::graph::InplaceDeleteMethod::VisitedAndTopK {
                    k_value: 4,
                    l_value: 10,
                },
            )
            .await
            .unwrap();

        // Check that the vertex was marked as deleted.
        assert!(
            index
                .data_provider
                .status_by_internal_id(CTX, 3)
                .await
                .unwrap()
                .is_deleted()
        );

        // expected outcome:
        // vertex 4 (the start point) has its edge to 3 deleted
        // vertex 2 (the other point with edge pointing to 3) should have its edge to point 3 deleted,
        // and replaced with edges to points 0 and 1
        // vertices 0 and 1 should add an edge pointing to 2.
        // vertex 3 should be dropped
        {
            let mut list = AdjacencyList::new();
            accessor.get_neighbors(4, &mut list).await.unwrap();
            list.sort();
            assert_eq!(&*list, &[0, 1, 2]);
        }

        {
            let mut list = AdjacencyList::new();
            accessor.get_neighbors(2, &mut list).await.unwrap();
            list.sort();
            assert_eq!(&*list, &[0, 1, 4]);
        }

        {
            let mut list = AdjacencyList::new();
            accessor.get_neighbors(0, &mut list).await.unwrap();
            list.sort();
            assert_eq!(&*list, &[1, 2, 4]);
        }

        {
            let mut list = AdjacencyList::new();
            accessor.get_neighbors(1, &mut list).await.unwrap();
            list.sort();
            assert_eq!(&*list, &[0, 2, 4]);
        }

        {
            let mut list = AdjacencyList::new();
            accessor.get_neighbors(3, &mut list).await.unwrap();
            assert!(list.is_empty());
        }
    }
}
