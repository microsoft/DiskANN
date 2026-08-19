/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! An in-memory provider for the DiskANN graph index.
//!
//! This type supports the following:
//!
//! * Arbitrary external IDs for store data (provided they satisfy [`Id`].
//! * Support for concurrent insertions, deletions, and searches.
//! * Specialized implementations of [`glue::SearchAccessor::expand_beam`] enabling full
//!   inlining of distance kernels.
//!
//! Known areas for future work:
//!
//! * Insert and delete protection: The [`DiskANNIndex`](diskann::graph::DiskANNIndex) doesn't
//!   support ergonomic insert or delete guards to protect slots during insert or delete
//!   operations. This leaves open a situation where an item can be inserted and during
//!   the insertion algorithm, it is deleted, and then re-inserted.
//!
//!   This can cause some issue within the main indexing algorithms which assume the inserted
//!   ID is present but requires upstream changes to properly fix.
//!
//! * Failed insert rollback: again, this needs some upstream changes to full support.
//!
//! * Quantization + reranking: The current version of this index targets just a single
//!   data-store and is planned to be addressed in the near future.
//!
//! * Lack of save/load support: The index is currently ephemeral, but there are plans to
//!   address this gap.

use std::hash::Hash;

use diskann::{
    ANNError, ANNResult,
    graph::{
        AdjacencyList, SearchOutputBuffer,
        glue::{self, HybridPredicate},
        workingset,
    },
    neighbor::Neighbor,
    provider,
};

use crate::{
    counters::{Counters, LocalCounters},
    ids::IdMap,
    layers,
    neighbors::Neighbors,
    num::{IdLimit, MaxDegree},
};

/// Aggregate trait for the external ID type of [`Provider`].
pub trait Id: Send + Sync + Hash + Eq + Clone + 'static {}

impl<T> Id for T where T: Send + Sync + Hash + Eq + Clone + 'static {}

/// An in-memory data-provider for DiskANN's graph indexing algorithms.
///
/// The first type parameter `L` is a [`layers::Layer`] for describing the kind of data
/// stored within the provider. The second parameter `M` is the associated data for items
/// inserted into the provider.
#[derive(Debug)]
pub struct Provider<L, M = u32>
where
    M: Id,
{
    // Data representation and storage.
    layer: L,
    // ID translation.
    mapping: IdMap<M>,
    // `Counters` is only non-trivial under the `integration-test` feature flag. Otherwise,
    // all counter related operations are no-ops.
    counters: Counters,
}

impl<L, M> Provider<L, M>
where
    M: Id,
{
    /// A local set of counters that update the provider-wide counters in bulk.
    fn local_counters(&self) -> LocalCounters<'_> {
        self.counters.local()
    }

    /// Return a snapshot of the current event counters.
    #[cfg(feature = "integration-test")]
    pub fn counters(&self) -> crate::integration::counters::CounterSnapshot {
        self.counters.snapshot()
    }
}

impl<L, M> Provider<L, M>
where
    L: layers::Layer,
    M: Id,
{
    /// Construct a new [`Provider`].
    ///
    /// The list of `start_points` must be must be compatible with `layer`.
    pub fn new<C>(config: C) -> ANNResult<Self>
    where
        C: layers::LayerConfig<Layer = L>,
    {
        let layer = <_ as layers::LayerConfig>::build(config)?;
        let mapping = IdMap::new(layer.capacity());

        Ok(Self {
            layer,
            mapping,
            counters: Counters::new(),
        })
    }

    /// Return the maximum number of neighbors that can be stored in the provider's graph.
    pub fn max_degree(&self) -> MaxDegree {
        self.layer.max_degree()
    }
}

///////////////////
// Data Provider //
///////////////////

/// A zero-sized [`diskann::provider::ExecutionContext`] for [`Provider`].
#[derive(Debug, Clone, Default)]
pub struct Context;

impl diskann::provider::ExecutionContext for Context {}

impl<T, M> diskann::provider::DataProvider for Provider<T, M>
where
    T: Send + Sync + 'static,
    M: Id,
{
    type Context = Context;
    type InternalId = u32;
    type ExternalId = M;
    type Error = ANNError;
    type Guard = diskann::provider::NoopGuard<u32>;

    fn to_internal_id(
        &self,
        _context: &Self::Context,
        gid: &M,
    ) -> Result<Self::InternalId, Self::Error> {
        match self.mapping.to_internal(gid) {
            Some(id) => Ok(id),
            None => Err(ANNError::message("no mapping")),
        }
    }

    /// Translate an internal id to its corresponding external id.
    fn to_external_id(
        &self,
        _context: &Self::Context,
        id: Self::InternalId,
    ) -> Result<Self::ExternalId, Self::Error> {
        match self.mapping.to_external(id) {
            Some(gid) => Ok(gid),
            None => Err(ANNError::message("no mapping")),
        }
    }
}

// TODO: The element-status checks here are profoundly approximate because we try to avoid
// any kind of EBR registration.
//
// `diskann` has plans to move deletion checks behind an accessor trait, which will help
// with this situation.
impl<L, M> diskann::provider::Delete for Provider<L, M>
where
    L: layers::Layer,
    M: Id,
{
    async fn delete(&self, _context: &Context, gid: &M) -> ANNResult<()> {
        // This guarantees that we have a valid mapping, but defers the actual deletion until
        // we know it's also safe to retire the internal slot.
        //
        // This ensures both either succeed or are aborted.
        let entry = match self.mapping.occupied_entry(gid.clone()) {
            None => {
                return Err(ANNError::message("id already deleted"));
            }
            Some(e) => e,
        };

        // An early return here will cause `entry` to be dropped, which will *not* cause
        // the delete to commit.
        <L as layers::Layer>::retire(&self.layer, entry.internal())?;

        // Successfully retired the internal slot. We can safely release the ID mapping.
        entry.delete();

        Ok(())
    }

    async fn release(&self, _context: &Context, _id: Self::InternalId) -> ANNResult<()> {
        Ok(())
    }

    async fn status_by_internal_id(
        &self,
        _context: &Context,
        id: u32,
    ) -> ANNResult<diskann::provider::ElementStatus> {
        // Not that this check is approximate. A full check requires materialization of
        // a `reader`.
        match <L as layers::Layer>::is_readable(&self.layer, id) {
            Some(true) => Ok(diskann::provider::ElementStatus::Valid),
            Some(false) => Ok(diskann::provider::ElementStatus::Deleted),
            None => Err(ANNError::message("accessed invalid internal ID")),
        }
    }

    async fn status_by_external_id(
        &self,
        _context: &Context,
        gid: &M,
    ) -> ANNResult<diskann::provider::ElementStatus> {
        if self.mapping.contains_external(gid) {
            Ok(diskann::provider::ElementStatus::Valid)
        } else {
            Ok(diskann::provider::ElementStatus::Deleted)
        }
    }
}

fn ready<F, R>(f: F) -> std::future::Ready<R>
where
    F: FnOnce() -> R,
{
    std::future::ready(f())
}

impl<T, L, M> diskann::provider::SetElement<T> for Provider<L, M>
where
    L: layers::Set<T>,
    M: Id,
{
    type SetError = ANNError;

    fn set_element(
        &self,
        _context: &Self::Context,
        id: &M,
        element: T,
    ) -> impl std::future::Future<Output = Result<Self::Guard, Self::SetError>> + Send {
        let work = move || {
            // TODO: Proper cleanup via `Guard` or some other mechanism on the event of
            // insert failure after `set_element` returns.
            //
            // The internal `Guard` is sufficient for local rollback, but not after
            // `set_element` returns.
            let guard = <L as layers::Set<T>>::set(&self.layer, element)?;
            let internal = <_ as layers::Guard>::id(&guard);
            self.mapping.insert(id.clone(), internal)?;

            // Now that insert has succeeded - publish the slot. This method cannot fail, so
            // we do not need to worry about potentially unwinding the ID mapping.
            <_ as layers::Guard>::publish(guard);

            // This is a rather expensive update.
            //
            // However, counters are only active with the `integration-test` feature, which
            // is not expected to be enabled for general use.
            self.local_counters().set_vector(1);

            Ok(diskann::provider::NoopGuard::new(internal))
        };

        ready(work)
    }
}

////////////
// Search //
////////////

/// A [`glue::SearchAccessor`] for [`Provider`].
///
/// This type intentionally avoids generic parameters and instead compiles optimized
/// `expand_beam` kernels that get reused. The idea is to generate an efficient graph search
/// kernel once and reuse it to balance compile times and performance.
#[derive(Debug)]
pub struct SearchAccessor<'a> {
    neighbors: &'a Neighbors,
    ids: AdjacencyList<u32>,
    expand_beam: Box<dyn layers::ExpandBeam + 'a>,
    id_limit: IdLimit,
    buffer: Vec<(u32, f32)>,

    // The parent provider for the accessor.
    provider: &'a (dyn std::any::Any + Send + Sync),
    start_points: std::ops::Range<u32>,
    counters: LocalCounters<'a>,
}

impl<'a> SearchAccessor<'a> {
    pub(crate) fn new(
        neighbors: &'a Neighbors,
        expand_beam: Box<dyn layers::ExpandBeam + 'a>,
        provider: &'a (dyn std::any::Any + Send + Sync),
        start_points: std::ops::Range<u32>,
        counters: LocalCounters<'a>,
    ) -> Self {
        let id_limit = expand_beam.id_limit();
        Self {
            neighbors,
            ids: AdjacencyList::with_capacity(neighbors.max_degree().value()),
            expand_beam,
            id_limit,
            buffer: vec![Default::default(); neighbors.max_degree().value()],
            provider,
            start_points,
            counters,
        }
    }
}

impl diskann::provider::HasId for SearchAccessor<'_> {
    type Id = u32;
}

impl glue::SearchAccessor for SearchAccessor<'_> {
    fn starting_points(
        &self,
    ) -> impl std::future::Future<Output = ANNResult<Vec<Self::Id>>> + Send {
        std::future::ready(Ok(self.start_points.clone().collect()))
    }

    fn start_point_distances<F>(
        &mut self,
        mut f: F,
    ) -> impl std::future::Future<Output = ANNResult<()>> + Send
    where
        F: FnMut(Self::Id, f32) + Send,
    {
        let work = move || {
            for p in self.start_points.clone() {
                match self.expand_beam.evaluate(p)? {
                    Some(distance) => {
                        // Counters are no-ops without `integration-test`.
                        self.counters.get_vector(1);
                        self.counters.query_distance(1);
                        f(p, distance);
                    }
                    None => {
                        return Err(ANNError::message("could not retrieve start point"));
                    }
                }
            }
            Ok(())
        };

        ready(work)
    }

    fn expand_beam<Itr, P, F>(
        &mut self,
        ids: Itr,
        mut pred: P,
        mut on_neighbors: F,
    ) -> impl std::future::Future<Output = ANNResult<()>> + Send
    where
        Itr: Iterator<Item = Self::Id> + Send,
        P: HybridPredicate<Self::Id> + Send + Sync,
        F: FnMut(Self::Id, f32) + Send,
    {
        let work = move || -> ANNResult<()> {
            for i in ids {
                self.neighbors.get(i, &mut self.ids)?;
                self.counters.get_neighbors(1);

                // Filter out unvisited IDs and ensure that all the IDs we are about
                self.ids
                    .retain(|i| pred.eval_mut(i) && self.id_limit.is_in_bounds(*i));

                // This should always hold, but let's double check.
                assert!(self.buffer.len() >= self.ids.len());

                // SAFETY:
                //
                // 1. We've verified that each entry in `self.ids` is in-bounds using
                //   the `IdLimit` derived from this `ExpandBeam` object (enforced in the
                //   constructor).
                //
                // 2. `self.buffer` is long enough to hold all the IDs.
                let processed =
                    unsafe { self.expand_beam.expand_beam(&self.ids, &mut self.buffer) }?;

                self.counters.get_vector(processed as u64);
                self.counters.query_distance(processed as u64);

                self.buffer
                    .iter()
                    .take(processed)
                    .for_each(|(id, dist)| on_neighbors(*id, *dist));
            }

            Ok(())
        };

        ready(work)
    }
}

////////////
// Insert //
////////////

/// The [`glue::PruneAccessor`] implementation for [`Provider`].
///
/// This type implements zero-copy access to the data within its parent provider during prunes.
#[derive(Debug)]
pub struct PruneAccessor<'a> {
    prune: Box<dyn layers::Prune + 'a>,
    keys: hashbrown::HashMap<u32, Option<layers::PruneKey>>,
    neighbors: &'a Neighbors,
    counters: LocalCounters<'a>,
}

impl<'a> PruneAccessor<'a> {
    pub(crate) fn new(
        prune: Box<dyn layers::Prune + 'a>,
        neighbors: &'a Neighbors,
        counters: LocalCounters<'a>,
    ) -> Self {
        Self {
            prune,
            keys: hashbrown::HashMap::new(),
            neighbors,
            counters,
        }
    }
}

/// The distance computer for [`PruneAccessor`].
#[derive(Debug)]
pub struct Distance<'a> {
    prune: &'a dyn layers::Prune,
    counters: LocalCounters<'a>,
}

impl<'a> Distance<'a> {
    fn new(prune: &'a dyn layers::Prune, counters: LocalCounters<'a>) -> Self {
        Self { prune, counters }
    }
}

#[expect(
    clippy::unwrap_used,
    reason = "prune does not allow fallible distance functions yet"
)]
impl diskann_vector::DistanceFunction<layers::PruneKey, layers::PruneKey, f32> for Distance<'_> {
    #[inline]
    fn evaluate_similarity(&self, x: layers::PruneKey, y: layers::PruneKey) -> f32 {
        self.counters.distance_ref(1);
        self.prune.evaluate(x, y)
    }
}

impl diskann::provider::HasId for PruneAccessor<'_> {
    type Id = u32;
}

impl glue::PruneAccessor for PruneAccessor<'_> {
    type Neighbors<'a>
        = provider::Neighbors<'a, Self>
    where
        Self: 'a;

    type ElementRef<'a> = layers::PruneKey;

    type View<'a>
        = &'a Self
    where
        Self: 'a;

    type Distance<'a>
        = Distance<'a>
    where
        Self: 'a;

    fn neighbors(&mut self) -> Self::Neighbors<'_> {
        provider::Neighbors(self)
    }

    async fn fill<'a, Itr>(
        &'a mut self,
        itr: Itr,
    ) -> ANNResult<(Self::View<'a>, Self::Distance<'a>)>
    where
        Itr: ExactSizeIterator<Item = Self::Id> + Clone + Send + Sync,
    {
        self.keys.clear();
        self.keys.extend(itr.map(|i| (i, None)));
        let count = self.prune.prepare(self.keys.iter_mut())?;
        self.counters.get_vector(count.as_u64());

        Ok((self, Distance::new(&*self.prune, self.counters.fork())))
    }
}

impl provider::NeighborAccessor for PruneAccessor<'_> {
    fn get_neighbors(
        &mut self,
        id: Self::Id,
        neighbors: &mut AdjacencyList<Self::Id>,
    ) -> impl std::future::Future<Output = ANNResult<()>> + Send {
        let work = move || {
            self.counters.get_neighbors(1);
            Ok(self.neighbors.get(id, neighbors)?)
        };
        ready(work)
    }
}

impl provider::NeighborAccessorMut for PruneAccessor<'_> {
    fn set_neighbors(
        &mut self,
        id: Self::Id,
        neighbors: &[Self::Id],
    ) -> impl std::future::Future<Output = ANNResult<()>> + Send {
        let work = move || {
            self.counters.set_neighbors(1);
            Ok(self.neighbors.set(id, neighbors)?)
        };
        ready(work)
    }

    fn append_vector(
        &mut self,
        id: Self::Id,
        neighbors: &[Self::Id],
    ) -> impl std::future::Future<Output = ANNResult<()>> + Send {
        let work = move || -> ANNResult<()> {
            self.counters.append_vector(1);
            let lock = self.neighbors.lock(id)?;

            // Due to race conditions between calls to `get_neighbors` and `append_vector`
            // in `diskann` - it's possible that the state of the adjacency list has changed
            // and we're now trying to add too many neighbors.
            //
            // We take care of that here by simply truncating.
            //
            // TODO: Introduce proper atomicity in the core algorithm.
            if lock.len() + neighbors.len() > lock.capacity() {
                let slack = lock.capacity() - lock.len();
                lock.append(&neighbors[..slack])?;
            } else {
                lock.append(neighbors)?;
            }

            Ok(())
        };

        ready(work)
    }
}

impl workingset::View<u32> for &PruneAccessor<'_> {
    type ElementRef<'a> = layers::PruneKey;
    type Element<'a>
        = layers::PruneKey
    where
        Self: 'a;

    fn get(&self, id: u32) -> Option<layers::PruneKey> {
        *self.keys.get(&id)?
    }
}

////////////////
// Strategies //
////////////////

#[derive(Debug, Clone, Copy)]
pub struct Strategy;

impl<'a, L, M> glue::SearchStrategy<'a, Provider<L, M>, L::Query<'a>> for Strategy
where
    L: layers::Search,
    M: Id,
{
    type SearchAccessor = SearchAccessor<'a>;
    type SearchAccessorError = ANNError;

    fn search_accessor(
        &'a self,
        provider: &'a Provider<L, M>,
        _context: &'a Context,
        query: L::Query<'a>,
    ) -> ANNResult<SearchAccessor<'a>> {
        <L as layers::Search>::search_accessor(
            &provider.layer,
            query,
            provider,
            provider.local_counters(),
        )
    }
}

// This is a utility for helping inspect the generated code for `ExpandBeam`.
//
pub fn test_function<'a>(
    x: &'a Provider<layers::Full<u8>>,
    strategy: &'a Strategy,
    context: &'a Context,
    query: &'a [u8],
) -> ANNResult<SearchAccessor<'a>> {
    glue::SearchStrategy::search_accessor(strategy, x, context, query)
}

/// Perform ID translation during post-processing.
#[derive(Debug, Clone, Copy)]
pub struct Translate<L, M>(std::marker::PhantomData<(L, M)>);

impl<L, M> Default for Translate<L, M> {
    fn default() -> Self {
        Self(std::marker::PhantomData)
    }
}

impl<'a, L, M> glue::SearchPostProcess<SearchAccessor<'a>, L::Query<'a>, M> for Translate<L, M>
where
    L: layers::Search,
    M: Id,
{
    type Error = ANNError;

    fn post_process<I, B>(
        &self,
        accessor: &mut SearchAccessor<'_>,
        _query: L::Query<'a>,
        candidates: I,
        output: &mut B,
    ) -> impl std::future::Future<Output = Result<usize, Self::Error>> + Send
    where
        I: Iterator<Item = Neighbor<u32>> + Send,
        B: SearchOutputBuffer<M> + Send + ?Sized,
    {
        let work = move || {
            // By construction - the downcast should succeed. Otherwise, this is a program bug.
            let provider = match accessor.provider.downcast_ref::<Provider<L, M>>() {
                Some(provider) => provider,
                None => return Err(ANNError::message("bad any cast")),
            };

            let mut count = 0;
            for c in candidates {
                if let Some(ext) = provider.mapping.to_external(*c.id()) {
                    if output
                        .push(Neighbor::new(ext, *c.distance()))
                        .is_available()
                    {
                        count += 1;
                    } else {
                        break;
                    }
                }
            }
            Ok(count)
        };

        ready(work)
    }
}

impl<'a, L, M> glue::DefaultPostProcessor<'a, Provider<L, M>, L::Query<'a>, M> for Strategy
where
    L: layers::Search,
    M: Id,
{
    diskann::default_post_processor!(Translate<L, M>);
}

impl<L, M> glue::PruneStrategy<Provider<L, M>> for Strategy
where
    L: layers::Insert,
    M: Id,
{
    type PruneAccessor<'a> = PruneAccessor<'a>;
    type PruneAccessorError = ANNError;

    fn prune_accessor<'a>(
        &self,
        provider: &'a Provider<L, M>,
        _context: &'a Context,
        _capacity: usize,
    ) -> ANNResult<PruneAccessor<'a>> {
        <L as layers::Insert>::prune_accessor(&provider.layer, provider.local_counters())
    }
}

impl<'a, L, M> glue::InsertStrategy<'a, Provider<L, M>, L::Query<'a>> for Strategy
where
    L: layers::Insert,
    M: Id,
{
    type PruneStrategy = Self;
    fn prune_strategy(&self) -> Self::PruneStrategy {
        *self
    }
}

impl<T, M> glue::InplaceDeleteStrategy<Provider<layers::Full<T>, M>> for Strategy
where
    M: Id,
    T: layers::FullPrecision,
{
    type DeleteElement<'a> = &'a [T];
    type DeleteElementGuard = Box<[T]>;
    type DeleteElementError = ANNError;

    type PruneStrategy = Self;
    type DeleteSearchAccessor<'a> = SearchAccessor<'a>;
    type SearchPostProcessor = glue::CopyIds;
    type SearchStrategy = Self;

    fn prune_strategy(&self) -> Self {
        *self
    }

    fn search_strategy(&self) -> Self {
        *self
    }

    fn search_post_processor(&self) -> Self::SearchPostProcessor {
        glue::CopyIds
    }

    fn get_delete_element<'a>(
        &'a self,
        provider: &'a Provider<layers::Full<T>, M>,
        _context: &'a Context,
        id: u32,
    ) -> impl Future<Output = Result<Self::DeleteElementGuard, Self::DeleteElementError>> + Send
    {
        let work = move || provider.layer.get(id);
        ready(work)
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    use diskann::{
        graph::{DiskANNIndex, InplaceDeleteMethod, search::Knn, test::synthetic::Grid},
        neighbor::Neighbor,
        provider::{DataProvider, Delete},
    };
    use diskann_utils::views::Matrix;
    use diskann_vector::distance::Metric;

    use crate::num::Capacity;

    /// The true tests live in the integration tests for this repo.
    ///
    /// The smoke test here uses a 2D grid of points to verify that our provider
    /// implementations are more-or-less correct.
    ///
    /// Note that since `Provider` separates internal and external IDs, we multiply the
    /// coordinates of each element in the grid by 10 and add 1 to verify that the ID
    /// translation is behaving properly.
    ///
    /// For clarity, the expected structure of the grid is as follows:
    ///
    ///                       <unknown>
    ///   41 91 141 191 241
    ///   31 81 131 181 231
    ///   21 71 121 171 221
    ///   11 61 111 161 211
    ///    1 51 101 151 201
    ///
    #[tokio::test]
    async fn smoke() {
        let grid = Grid::Two;
        let size = 5;
        let data = grid.data(size);
        let start = grid.start_point(size);
        let degree = 6;

        let config = layers::full::Config::new(
            Capacity::new(grid.num_points(size)),
            MaxDegree::new(degree),
            Metric::L2,
            Matrix::row_vector(start.into()),
        );

        // let full = Full::<f32>::new(grid.dim().into(), Metric::L2);

        // let config = Config::new(grid.num_points(size), degree);

        let provider = Provider::<_, u64>::new(config).unwrap();
        assert_eq!(provider.max_degree(), MaxDegree::new(degree));

        let config = diskann::graph::config::Builder::new(
            2 * (grid.dim() as usize),
            diskann::graph::config::MaxDegree::new(provider.max_degree().value()),
            10,
            (Metric::L2).into(),
        )
        .build()
        .unwrap();

        let index = DiskANNIndex::new(config, provider, None);

        for (i, data) in data.row_iter().enumerate() {
            index
                .insert(&Strategy, &Context, &((10 * i + 1) as u64), data)
                .await
                .unwrap();
        }

        // Verify that each ID round trips.
        for i in 0..data.nrows() {
            let i = (10 * i + 1) as u64;
            let internal = index.provider().to_internal_id(&Context, &i).unwrap();
            assert_ne!(internal as u64, i);
            assert_eq!(
                index.provider().to_external_id(&Context, internal).unwrap(),
                i
            );

            assert!(
                !index
                    .provider()
                    .status_by_external_id(&Context, &i)
                    .await
                    .unwrap()
                    .is_deleted()
            );
            assert!(
                !index
                    .provider()
                    .status_by_internal_id(&Context, internal)
                    .await
                    .unwrap()
                    .is_deleted()
            );
        }

        // Assert that out-of-bounds translations returns errors.
        assert!(index.provider().to_internal_id(&Context, &0).is_err());
        assert!(index.provider().to_external_id(&Context, 26).is_err());

        // Searches should return something reasonable.
        let knn = Knn::new(10, None).unwrap();
        let mut neighbors = Vec::<Neighbor<u64>>::new();
        index
            .search(knn, &Strategy, &Context, &[0.0, 0.0], &mut neighbors)
            .await
            .unwrap();

        assert_eq!(neighbors[0].as_tuple(), (1, 0.0));
        assert_eq!(neighbors[1].as_tuple(), (11, 1.0)); // this can be swapped with 2
        assert_eq!(neighbors[2].as_tuple(), (51, 1.0));
        assert_eq!(neighbors[3].as_tuple(), (61, 2.0));

        // If we run inplace delete on point 61, it longer be present.
        index
            .inplace_delete(
                Strategy,
                &Context,
                &61,
                3,
                InplaceDeleteMethod::VisitedAndTopK {
                    k_value: 10,
                    l_value: 10,
                },
            )
            .await
            .unwrap();

        assert!(
            index
                .provider()
                .status_by_external_id(&Context, &61)
                .await
                .unwrap()
                .is_deleted()
        );

        // We can't delete the same thing twice.
        assert!(
            index
                .inplace_delete(
                    Strategy,
                    &Context,
                    &61,
                    3,
                    InplaceDeleteMethod::VisitedAndTopK {
                        k_value: 10,
                        l_value: 10
                    },
                )
                .await
                .is_err()
        );

        // Rerun search - the point 61 should now be gone.
        let mut neighbors = Vec::<Neighbor<u64>>::new();
        index
            .search(knn, &Strategy, &Context, &[0.0, 0.0], &mut neighbors)
            .await
            .unwrap();

        assert_eq!(neighbors[0].as_tuple(), (1, 0.0));
        assert_eq!(neighbors[1].as_tuple(), (51, 1.0)); // this can be swapped with 2
        assert_eq!(neighbors[2].as_tuple(), (11, 1.0));
        assert_eq!(neighbors[3].as_tuple(), (101, 4.0)); // we can also accept "21"

        // We can't insert an existing ID.
        assert!(
            index
                .insert(&Strategy, &Context, &1, &[10.0, 10.0])
                .await
                .is_err()
        );

        // If we insert a new ID but the query vector is too long - make sure we leave the
        // provider untouched.
        assert!(
            index
                .insert(&Strategy, &Context, &2, &[10.0, 10.0, 10.0])
                .await
                .is_err()
        );

        // Check that we can reinsert the same point with a different ID and have it be
        // returned from search.
        index
            .insert(&Strategy, &Context, &62, &[1.0, 1.0])
            .await
            .unwrap();

        // We can't insert an ID - but this time it's because we don't have any more internal
        // slots.
        assert!(
            index
                .insert(&Strategy, &Context, &62, &[0.0, 0.0])
                .await
                .is_err()
        );

        // Rerun search - the point 62 should be present.
        let mut neighbors = Vec::<Neighbor<u64>>::new();
        index
            .search(knn, &Strategy, &Context, &[0.0, 0.0], &mut neighbors)
            .await
            .unwrap();

        assert_eq!(neighbors[0].as_tuple(), (1, 0.0));
        assert_eq!(neighbors[1].as_tuple(), (11, 1.0)); // this can be swapped with 2
        assert_eq!(neighbors[2].as_tuple(), (51, 1.0));
        assert_eq!(neighbors[3].as_tuple(), (62, 2.0));
    }
}
