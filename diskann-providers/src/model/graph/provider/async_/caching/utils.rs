/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    fmt::Debug,
    sync::atomic::{AtomicUsize, Ordering},
};

use diskann::graph::AdjacencyList;
use diskann_utils::future::AsyncFriendly;

use super::{
    bf_cache::{AdjacencyListCacher, Cache, CacheableId},
    error::CacheAccessError,
    provider,
};

/// A utility struct for recording cache hits and misses.
#[derive(Debug, Default)]
pub struct HitStats {
    hits: AtomicUsize,
    misses: AtomicUsize,
}

impl HitStats {
    /// Construct a new `HitStats` with the hit and miss counters set to 0
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a hit.
    pub fn hit(&self, count: usize) {
        self.hits.fetch_add(count, Ordering::Relaxed);
    }

    /// Record a miss.
    pub fn miss(&self, count: usize) {
        self.misses.fetch_add(count, Ordering::Relaxed);
    }

    /// Return the number of hits.
    pub fn get_hits(&self) -> usize {
        self.hits.load(Ordering::Relaxed)
    }

    /// Return the number of misses.
    pub fn get_misses(&self) -> usize {
        self.misses.load(Ordering::Relaxed)
    }
}

/// A helper view over `HitStats` that uses private, owned accumulators to record local hits
/// and misses.
///
/// The implementation of `Drop` will update the parent counter all at once, reducing
/// contention on the shared atomics.
#[derive(Debug)]
pub struct LocalStats<'a> {
    parent: &'a HitStats,
    hits: usize,
    misses: usize,
}

impl<'a> LocalStats<'a> {
    /// Construct a new `HitStats` with the local hit and miss counters set to 0.
    ///
    /// When dropped, `parent` will be updated with the total hits and misses recorded for
    /// this local counter.
    pub fn new(parent: &'a HitStats) -> Self {
        Self {
            parent,
            hits: 0,
            misses: 0,
        }
    }

    /// Record a hit.
    pub fn hit(&mut self) {
        self.hits += 1;
    }

    /// Record a miss.
    pub fn miss(&mut self) {
        self.misses += 1;
    }

    /// Return the **local** number of hits.
    pub fn get_local_hits(&self) -> usize {
        self.hits
    }

    /// Return the **local** number of misses.
    pub fn get_local_misses(&self) -> usize {
        self.misses
    }
}

impl Drop for LocalStats<'_> {
    fn drop(&mut self) {
        self.parent.hit(self.hits);
        self.parent.miss(self.misses);
    }
}

////////////
// KeyGen //
////////////

/// Utility trait to generate type-tagged keys.
///
/// The motivating problem is to use a single [`bf_cache::Cache`] to store different types of
/// payloads for a single key, such as quantized vectors, full-precision vectors, and
/// adjacency list terms.
///
/// To do this, we augment the normal key (typically a 32-bit or 64-bit ID) with an
/// additional tag, depending on the type of payload.
///
/// To keep from coupling the mechanisms of key term generation too closely with any
/// particular strategy, the `KeyGen` trait is used.
pub trait KeyGen<K> {
    /// The type of the returned key - constrained to be `bytemuck::Pod` for compatibility
    /// with [`bf_cache::Cache`].
    type Key: bytemuck::Pod;

    /// Generate a full key term from `key`.
    fn generate(&self, key: K) -> Self::Key;
}

/// A [`bytemuck::Pod`] compatible cache key consisting of the true element ID and a type tag.
///
/// The tag is used to disambiguate among multiple different values associated with the same id.
#[derive(Debug, Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C, packed)]
pub struct CacheKey<I> {
    id: I,
    tag: I,
}

impl<I> CacheKey<I>
where
    I: bytemuck::Pod,
{
    /// Create a new `CacheKey` with the given `id` and `tag`.
    pub fn new(id: I, tag: I) -> Self {
        Self { id, tag }
    }

    /// Return the `id` contained in the `CacheKey`.
    pub fn id(&self) -> I {
        self.id
    }

    /// Return the type tag contained in the `CacheKey`.
    pub fn tag(&self) -> I {
        self.tag
    }
}

/// A [`KeyGen`] generate for [`CacheKey`].
///
/// This struct's implementation of [`KeyGen`] will use the specified `key` as a prefix
/// with the local `tag` as a suffix. This allows range scans over a particular `key` to
/// work correctly for all possible tag types.
#[derive(Debug, Clone, Copy)]
pub struct Tag<I>(I);

impl<I> Tag<I> {
    /// Construct a new [`Tag`] whose implementation of [`KeyGen`] adds the argument
    /// `tag` as a suffix to the key.
    pub fn new(tag: I) -> Self {
        Self(tag)
    }

    /// Return the tag this `Tag` was created with.
    pub fn tag(self) -> I {
        self.0
    }
}

impl<I> KeyGen<I> for Tag<I>
where
    I: bytemuck::Pod,
{
    type Key = CacheKey<I>;
    fn generate(&self, key: I) -> CacheKey<I> {
        CacheKey {
            id: key,
            tag: self.0,
        }
    }
}

///////////
// Graph //
///////////

/// A write-through invalidating implementation for [`provider::NeighborCache`] storing and
/// retrieving adjacency lists from a [`Cache`].
///
/// This is useful as a building block for full cache accessors.
#[derive(Debug)]
pub struct Graph<'a, I, T> {
    cache: &'a Cache,
    stats: LocalStats<'a>,
    accessor: AdjacencyListCacher<I>,
    keygen: T,
}

impl<'a, I, T> Graph<'a, I, T>
where
    I: Default + Clone,
{
    /// Construct a new [`Graph`] for storing and retrieving adjacency list terms from `cache`.
    /// Adjacency list terms will be constrained to not exceed `max_degree` in length.
    ///
    /// To allow sharing of the underlying `cache`, a [`KeyGen`] implementation `keygen`
    /// will be used to generate the keys from vector ids.
    ///
    /// The stats counter `stats` will be updated with local hit and miss rates when this
    /// data structure is dropped. Update on drop reduces contention on the shared atomics
    /// inside [`HitStats`].
    pub fn new(cache: &'a Cache, max_degree: usize, keygen: T, stats: &'a HitStats) -> Self {
        Self {
            cache,
            stats: LocalStats::new(stats),
            accessor: AdjacencyListCacher::new(max_degree),
            keygen,
        }
    }
}

impl<'a, I, T> Graph<'a, I, T> {
    /// Return the underlying [`Cache`] used by this accessor.
    pub fn cache(&self) -> &Cache {
        self.cache
    }

    /// Return the local hit/miss stats.
    pub fn stats(&self) -> &LocalStats<'_> {
        &self.stats
    }

    /// Return the local hit/miss stats via mutable reference. Care should be taken to
    /// preserve accurate reporting of the hit/miss stats.
    pub fn stats_mut(&mut self) -> &mut LocalStats<'a> {
        &mut self.stats
    }
}

impl<I, T> provider::NeighborCache<I> for Graph<'_, I, T>
where
    I: CacheableId,
    T: KeyGen<I> + AsyncFriendly,
{
    type Error = CacheAccessError;

    fn try_get_neighbors(
        &mut self,
        id: I,
        neighbors: &mut AdjacencyList<I>,
    ) -> Result<provider::NeighborStatus, CacheAccessError> {
        let hit = self
            .cache
            .get_into(self.keygen.generate(id), &mut self.accessor, neighbors)
            .map_err(|err| CacheAccessError::read(id, err))?;

        if hit.into_inner() {
            self.stats.hit();
            Ok(provider::NeighborStatus::Hit)
        } else {
            self.stats.miss();
            Ok(provider::NeighborStatus::Miss)
        }
    }

    fn set_neighbors(&mut self, id: I, neighbors: &[I]) -> Result<(), CacheAccessError> {
        self.cache
            .set(self.keygen.generate(id), &mut self.accessor, neighbors)
            .map_err(|err| CacheAccessError::write(id, err))
    }

    fn invalidate_neighbors(&mut self, id: I) {
        self.cache.delete(self.keygen.generate(id))
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    use diskann_quantization::num::PowerOfTwo;

    use crate::model::graph::provider::async_::caching::provider::NeighborCache;

    // Hit Stats
    #[test]
    fn test_hit_stats() {
        let stats = HitStats::new();
        assert_eq!(stats.get_hits(), 0);
        assert_eq!(stats.get_misses(), 0);

        stats.hit(5);
        stats.miss(10);

        assert_eq!(stats.get_hits(), 5);
        assert_eq!(stats.get_misses(), 10);

        stats.hit(1);
        stats.miss(2);

        assert_eq!(stats.get_hits(), 6);
        assert_eq!(stats.get_misses(), 12);

        let hits = stats.get_hits();
        let misses = stats.get_misses();

        {
            let mut local = LocalStats::new(&stats);
            assert_eq!(local.get_local_hits(), 0);
            assert_eq!(local.get_local_misses(), 0);

            for _ in 0..5 {
                local.hit();
            }

            for _ in 0..10 {
                local.miss();
            }

            assert_eq!(local.get_local_hits(), 5);
            assert_eq!(local.get_local_misses(), 10);

            assert_eq!(local.parent.get_hits(), hits);
            assert_eq!(local.parent.get_misses(), misses);

            // Drop is called
        }

        // Parent stats should be updated with the aggregated values.
        assert_eq!(stats.get_hits(), hits + 5);
        assert_eq!(stats.get_misses(), misses + 10);
    }

    // Tag + CacheKey
    #[test]
    fn test_tag() {
        let tag0 = Tag::<usize>::new(0);
        let tag1 = Tag::<usize>::new(1);

        assert_eq!(tag0.tag(), 0);
        assert_eq!(tag1.tag(), 1);

        for k in 0..10 {
            let key = tag0.generate(k);
            assert_eq!(key.id(), k);
            assert_eq!(key.tag(), 0);

            let key = tag1.generate(k);
            assert_eq!(key.id(), k);
            assert_eq!(key.tag(), 1);
        }
    }

    ///////////
    // Graph //
    ///////////

    #[test]
    fn test_graph() {
        let tag = 42;

        let cache = Cache::new(PowerOfTwo::new(128 * 1024).unwrap());
        let max_degree = 4;
        let keygen = Tag::<u32>::new(tag);
        let stats = HitStats::new();

        let mut graph = Graph::new(&cache, max_degree, keygen, &stats);
        assert_eq!(graph.stats().get_local_hits(), 0);
        assert_eq!(graph.stats().get_local_misses(), 0);

        // Accessing an non-existent key should do nothing.
        let mut a = AdjacencyList::new();
        let id = 90u32;
        assert_eq!(
            graph.try_get_neighbors(id, &mut a).unwrap(),
            provider::NeighborStatus::Miss,
            "`try_get_neighbors` should return `Miss` when the term does not exist",
        );

        // Add new neighbors.
        graph.set_neighbors(id, &[1, 2, 3]).unwrap();

        // Now retrieval should succeed.
        assert_eq!(
            graph.try_get_neighbors(id, &mut a).unwrap(),
            provider::NeighborStatus::Hit,
            "`try_get_neighbors` should succeed when neighbors are present",
        );
        assert_eq!(&*a, &[1, 2, 3]);

        // Test the `keygen` is actually being used by attempting to access `id` directly
        {
            let mut cacher = AdjacencyListCacher::<u32>::new(max_degree);
            let mut a = AdjacencyList::<u32>::new();
            assert!(
                !cache
                    .get_into(id, &mut cacher, &mut a)
                    .unwrap()
                    .into_inner(),
                "attempt to access via raw `id` should fail because keys are tagged"
            );

            assert!(
                cache
                    .get_into(CacheKey { id, tag }, &mut cacher, &mut a)
                    .unwrap()
                    .into_inner()
            );
            assert_eq!(&*a, &[1, 2, 3]);
        }

        // Filling again should overwrite the existing contents.
        graph.set_neighbors(id, &[]).unwrap();

        assert_eq!(
            graph.try_get_neighbors(id, &mut a).unwrap(),
            provider::NeighborStatus::Hit,
            "`try_get_neighbors` should succeed when neighbors are present",
        );
        assert!(a.is_empty());

        // Invalidation works
        graph.invalidate_neighbors(id);

        assert_eq!(
            graph.try_get_neighbors(id, &mut a).unwrap(),
            provider::NeighborStatus::Miss,
            "attempted mutation invalidates the graph"
        );
    }
}
