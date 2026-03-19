/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! An example cache for demonstrating how to set up a [`CachingProvider`] with an inner
//! data provider. The inner provider used in this module is the test provider from
//! [`diskann::graph::test::provider`].

use diskann::{
    graph::{AdjacencyList, test::provider as test_provider},
    provider::{self as core_provider},
};
use diskann_utils::future::AsyncFriendly;
use diskann_vector::distance::Metric;

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

impl<'a> cache_provider::AsCacheAccessorFor<'a, test_provider::Accessor<'a>> for ExampleCache {
    type Accessor = CacheAccessor<'a, bf_cache::VecCacher<f32>>;
    type Error = diskann::error::Infallible;
    fn as_cache_accessor_for(
        &'a self,
        inner: test_provider::Accessor<'a>,
    ) -> Result<
        cache_provider::CachingAccessor<test_provider::Accessor<'a>, Self::Accessor>,
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
    for test_provider::Accessor<'a>
{
}

impl<'a> cache_provider::CachedAsElement<&'a [f32], CacheAccessor<'a, bf_cache::VecCacher<f32>>>
    for test_provider::Accessor<'a>
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
        graph::DiskANNIndex,
        provider::{
            Accessor, DataProvider, Delete, NeighborAccessor, NeighborAccessorMut, SetElement,
        },
        utils::async_tools,
    };
    use diskann_quantization::num::PowerOfTwo;
    use rstest::rstest;

    use crate::{
        index::diskann_async::tests as async_tests,
        model::graph::provider::async_::caching::provider::{AsCacheAccessorFor, CachingProvider},
    };

    fn test_provider(
        uncacheable: Option<Vec<u32>>,
    ) -> CachingProvider<test_provider::Provider, ExampleCache> {
        let dim = 2;
        let max_degree = 10;
        let start_id = u32::MAX;

        let config = test_provider::Config::new(
            Metric::L2,
            max_degree,
            test_provider::StartPoint::new(start_id, vec![0.0; dim]),
        )
        .unwrap();

        CachingProvider::new(
            test_provider::Provider::new(config),
            ExampleCache::new(PowerOfTwo::new(1024 * 16).unwrap(), uncacheable),
        )
    }

    fn ctx() -> test_provider::Context {
        test_provider::Context::new()
    }

    #[tokio::test]
    async fn basic_operations_happy_path() {
        let provider = test_provider(None);
        let ctx = ctx();

        // Translations do not yet exist.
        assert!(provider.to_external_id(&ctx, 0).is_err());
        assert!(provider.to_internal_id(&ctx, &0).is_err());

        assert_eq!(provider.inner().metrics().set_vector, 0);
        provider.set_element(&ctx, &0, &[1.0, 2.0]).await.unwrap();
        assert_eq!(
            provider.inner().metrics().set_vector,
            1 /* increased */
        );

        assert_eq!(provider.to_external_id(&ctx, 0).unwrap(), 0);
        assert_eq!(provider.to_internal_id(&ctx, &0).unwrap(), 0);

        // Retrieval of a valid element.
        let mut accessor = provider
            .cache()
            .as_cache_accessor_for(test_provider::Accessor::new(provider.inner()))
            .unwrap();

        // Hit served from the underlying provider (cache miss).
        let element = accessor.get_element(0).await.unwrap();
        assert_eq!(element, &[1.0, 2.0]);
        assert_eq!(
            accessor.cache().stats.get_local_misses(),
            1, /* increased */
        );
        assert_eq!(accessor.cache().stats.get_local_hits(), 0);

        // This time, the hit is served from the underlying cache.
        let element = accessor.get_element(0).await.unwrap();
        assert_eq!(element, &[1.0, 2.0]);
        assert_eq!(accessor.cache().stats.get_local_misses(), 1);
        assert_eq!(
            accessor.cache().stats.get_local_hits(),
            1, /* increased */
        );

        // Adjacency List from Underlying
        assert_eq!(provider.inner().metrics().set_neighbors, 0);
        accessor.set_neighbors(0, &[1, 2, 3]).await.unwrap();
        assert_eq!(
            provider.inner().metrics().set_neighbors,
            1, /* increased */
        );

        let mut list = AdjacencyList::new();
        assert_eq!(provider.inner().metrics().get_neighbors, 0);
        accessor.get_neighbors(0, &mut list).await.unwrap();
        assert_eq!(
            provider.inner().metrics().get_neighbors,
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
        assert_eq!(provider.inner().metrics().get_neighbors, 1);
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
        assert_eq!(
            accessor.cache().stats.get_local_misses(),
            2, /* increased */
        );
        assert_eq!(accessor.cache().stats.get_local_hits(), 1);

        // Once more from the cache.
        let element = accessor.get_element(0).await.unwrap();
        assert_eq!(element, &[1.0, 2.0]);
        assert_eq!(accessor.cache().stats.get_local_misses(), 2);
        assert_eq!(
            accessor.cache().stats.get_local_hits(),
            2, /* increased */
        );

        list.clear();
        accessor.get_neighbors(0, &mut list).await.unwrap();
        assert_eq!(&*list, &[1, 2, 3]);
        assert_eq!(
            provider.inner().metrics().get_neighbors,
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
            provider.inner().metrics().set_neighbors,
            2, /* increased */
        );
        assert_eq!(
            provider.inner().metrics().get_neighbors,
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
            provider.inner().metrics().set_neighbors,
            2,
            "append_vector doesn't go through set_neighbors counter"
        );
        assert_eq!(
            provider.inner().metrics().get_neighbors,
            4, /* increased */
        );
        assert_eq!(
            accessor.cache().graph.stats().get_local_misses(),
            4, /* increased */
        );
        assert_eq!(accessor.cache().graph.stats().get_local_hits(), 1);

        // Deletion.
        assert_eq!(
            provider.status_by_internal_id(&ctx, 0).await.unwrap(),
            core_provider::ElementStatus::Valid
        );
        assert_eq!(
            provider.status_by_external_id(&ctx, &0).await.unwrap(),
            core_provider::ElementStatus::Valid
        );
        assert!(provider.status_by_internal_id(&ctx, 1).await.is_err());
        assert!(provider.status_by_external_id(&ctx, &1).await.is_err());

        provider.delete(&ctx, &0).await.unwrap();

        assert_eq!(
            provider.status_by_internal_id(&ctx, 0).await.unwrap(),
            core_provider::ElementStatus::Deleted
        );
        assert_eq!(
            provider.status_by_external_id(&ctx, &0).await.unwrap(),
            core_provider::ElementStatus::Deleted
        );
        assert!(provider.status_by_internal_id(&ctx, 1).await.is_err());
        assert!(provider.status_by_external_id(&ctx, &1).await.is_err());

        // Accessing the deleted element is still valid.
        let element = accessor.get_element(0).await.unwrap();
        assert_eq!(element, &[1.0, 2.0]);
        assert_eq!(accessor.cache().stats.get_local_misses(), 2);
        assert_eq!(
            accessor.cache().stats.get_local_hits(),
            3, /* increased */
        );

        accessor.get_neighbors(0, &mut list).await.unwrap();
        assert_eq!(&*list, &[2, 3, 4, 1]);
        assert_eq!(provider.inner().metrics().get_neighbors, 4);
        assert_eq!(accessor.cache().graph.stats().get_local_misses(), 4);
        assert_eq!(
            accessor.cache().graph.stats().get_local_hits(),
            2, /* increased */
        );

        provider.release(&ctx, 0).await.unwrap();
        assert!(provider.status_by_internal_id(&ctx, 0).await.is_err());
        assert!(provider.status_by_external_id(&ctx, &0).await.is_err());

        assert!(accessor.get_element(0).await.is_err());
        assert_eq!(
            accessor.cache().stats.get_local_misses(),
            3 /* increased */
        );
        assert_eq!(accessor.cache().stats.get_local_hits(), 3);

        assert!(accessor.get_neighbors(0, &mut list).await.is_err());
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
        let ctx = ctx();

        let mut accessor = provider
            .cache()
            .as_cache_accessor_for(test_provider::Accessor::new(provider.inner()))
            .unwrap();

        provider.set_element(&ctx, &0, &[1.0, 2.0]).await.unwrap();

        //---------------//
        // Cacheable IDs //
        //---------------//

        // Adjacency List from Underlying
        assert_eq!(provider.inner().metrics().set_neighbors, 0);
        accessor.set_neighbors(0, &[1, 2, 3]).await.unwrap();
        assert_eq!(
            provider.inner().metrics().set_neighbors,
            1, /* increased */
        );

        let mut list = AdjacencyList::new();
        assert_eq!(provider.inner().metrics().get_neighbors, 0);
        accessor.get_neighbors(0, &mut list).await.unwrap();
        assert_eq!(
            provider.inner().metrics().get_neighbors,
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
        assert_eq!(provider.inner().metrics().get_neighbors, 1);
        assert_eq!(accessor.cache().graph.stats().get_local_misses(), 1);
        assert_eq!(
            accessor.cache().graph.stats().get_local_hits(),
            1, /* increased */
        );

        //-----------------//
        // Uncacheable IDs //
        //-----------------//

        assert_eq!(provider.inner().metrics().set_neighbors, 1);
        accessor.set_neighbors(uncacheable, &[4, 5]).await.unwrap();
        assert_eq!(
            provider.inner().metrics().set_neighbors,
            2, /* increased */
        );

        // The retrieval is served by the inner provider.
        assert_eq!(provider.inner().metrics().get_neighbors, 1);
        accessor
            .get_neighbors(uncacheable, &mut list)
            .await
            .unwrap();
        assert_eq!(
            provider.inner().metrics().get_neighbors,
            2, /* increased */
        );
        assert_eq!(
            accessor.cache().graph.stats().get_local_misses(),
            2, /* increased */
        );
        assert_eq!(accessor.cache().graph.stats().get_local_hits(), 1);
        assert_eq!(&*list, &[4, 5]);

        // Again, retrieval is served by the inner provider.
        assert_eq!(provider.inner().metrics().get_neighbors, 2);
        accessor
            .get_neighbors(uncacheable, &mut list)
            .await
            .unwrap();
        assert_eq!(
            provider.inner().metrics().get_neighbors,
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

    fn check_stats(caching: &CachingProvider<test_provider::Provider, ExampleCache>) {
        let metrics = caching.inner().metrics();
        let cache = caching.cache();

        println!("neighbor reads: {}", metrics.get_neighbors);
        println!("neighbor writes: {}", metrics.set_neighbors);
        println!("vector reads: {}", metrics.get_vector);
        println!("vector writes: {}", metrics.set_vector);

        println!("neighbor hits: {}", cache.neighbor_stats.get_hits());
        println!("neighbor misses: {}", cache.neighbor_stats.get_misses());
        println!("vector hits: {}", cache.vector_stats.get_hits());
        println!("vector misses: {}", cache.vector_stats.get_misses());

        // Neighbors
        assert_eq!(metrics.get_neighbors, cache.neighbor_stats.get_misses());

        // Vectors
        assert_eq!(metrics.get_vector, cache.vector_stats.get_misses());
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

        let vectors = <f32 as async_tests::GenerateGrid>::generate_grid(dim, grid_size);

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

        let config = test_provider::Config::new(
            metric,
            index_config.max_degree().get(),
            test_provider::StartPoint::new(start_id, start_point.clone()),
        )
        .unwrap();
        assert_eq!(vectors.len(), num_points);

        // Initialize an index for a new round of building.
        let init_index = || {
            let provider = CachingProvider::new(
                test_provider::Provider::new(config.clone()),
                ExampleCache::new(cache_size, None),
            );
            Arc::new(DiskANNIndex::new(index_config.clone(), provider, None))
        };

        let strategy = cache_provider::Cached::new(test_provider::Strategy::new());
        let ctx = ctx();

        // Build with full-precision single insert
        {
            let index = init_index();
            for (i, v) in vectors.iter().take(num_points).enumerate() {
                index
                    .insert(strategy, &ctx, &(i as u32), v.as_slice())
                    .await
                    .unwrap();
            }

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

            index.multi_insert(strategy, &ctx, batch).await.unwrap();

            check_stats(index.provider());
        }
    }
}
