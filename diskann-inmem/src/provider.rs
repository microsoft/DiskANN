/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::hash::Hash;

use diskann::{
    ANNError, ANNErrorKind, ANNResult,
    graph::{
        AdjacencyList, SearchOutputBuffer,
        glue::{self, HybridPredicate},
        workingset,
    },
    neighbor::Neighbor,
    provider,
    utils::IntoUsize,
};
use diskann_utils::views::Matrix;

use crate::{
    counters::{Counters, LocalCounters},
    layers::{self, QueryDistance},
    num::Bytes,
    sharded::Sharded,
    store::{self, Store},
};

pub trait Id: Send + Sync + Hash + Eq + Clone + 'static {}
impl<T> Id for T where T: Send + Sync + Hash + Eq + Clone + 'static {}

#[derive(Debug)]
pub struct Provider<L, M = u32>
where
    M: Id,
{
    store: Store,
    layer: L,
    mapping: Sharded<M>,

    // `Counters` is only non-trivial under the `integration-test` feature flag. Otherwise,
    // all counter related operations are no-ops.
    counters: Counters,
}

impl<L, M> Provider<L, M>
where
    M: Id,
{
    pub fn new<I, T>(layer: L, config: Config, start_points: I) -> Self
    where
        I: IntoIterator<Item = T>,
        L: layers::Set<T>,
    {
        let start_points: Vec<_> = start_points.into_iter().collect();
        let bytes = layers::Layer::bytes(&layer);
        let mut data = Matrix::new(0u8, start_points.len(), bytes.value());

        for (row, point) in std::iter::zip(data.row_iter_mut(), start_points.into_iter()) {
            layers::Set::into_bytes(&layer, point, row).unwrap();
        }

        let store = Store::new(
            config.capacity(),
            bytes,
            config.max_degree(),
            data.as_view(),
        )
        .unwrap();

        let mapping = Sharded::new(config.capacity());

        Self {
            store,
            layer,
            mapping,
            counters: Counters::new(),
        }
    }

    fn local_counters(&self) -> LocalCounters<'_> {
        self.counters.local()
    }

    /// Return the maximum number of neighbors that can be stored in the provider's graph.
    pub fn max_degree(&self) -> usize {
        self.store.max_degree()
    }

    /// Return a snapshot of the current event counters.
    #[cfg(feature = "integration-test")]
    pub fn counters(&self) -> crate::integration::counters::CounterSnapshot {
        self.counters.snapshot()
    }
}

#[derive(Debug)]
pub struct Config {
    capacity: usize,
    max_degree: usize,
}

impl Config {
    pub fn new(capacity: usize, max_degree: usize) -> Self {
        Self {
            capacity,
            max_degree,
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn max_degree(&self) -> usize {
        self.max_degree
    }
}

///////////////////
// Data Provider //
///////////////////

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
            None => Err(ANNError::message(ANNErrorKind::Opaque, "no mapping")),
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
            None => Err(ANNError::message(ANNErrorKind::Opaque, "no mapping")),
        }
    }
}

// TODO: The element-status checks here are profoundly expensive as they require epoch
// registration for each check!
//
// `diskann` has plans to move deletion checks behind an accessor trait, which will help
// with this situation.
impl<L, M> diskann::provider::Delete for Provider<L, M>
where
    L: Send + Sync + 'static,
    M: Id,
{
    async fn delete(&self, _context: &Context, gid: &M) -> ANNResult<()> {
        // TODO: These need to actually happen in lock-step.
        let entry = match self.mapping.occupied_entry(gid.clone()) {
            None => {
                return Err(ANNError::message(
                    ANNErrorKind::Opaque,
                    "id already deleted",
                ));
            }
            Some(e) => e,
        };

        match self.store.retire(entry.internal().into_usize()) {
            Ok(()) => {
                // Successfully retired the internal slot. We can safely release the ID mapping.
                entry.delete();
                Ok(())
            }
            Err(err) => Err(ANNError::opaque(err)),
        }
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
        if self.store.can_read_approximate(id.into_usize()).unwrap() {
            Ok(diskann::provider::ElementStatus::Valid)
        } else {
            Ok(diskann::provider::ElementStatus::Deleted)
        }
    }

    /// Check the status via external ID.
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
            let mut slot = self.store.acquire().ok_or_else(|| {
                ANNError::message(ANNErrorKind::Opaque, "could not allocate a new slot")
            })?;

            // TODO: Proper cleanup via `Guard` or some other mechanism on the event of
            // insert failure.
            <L as layers::Set<T>>::into_bytes(&self.layer, element, slot.as_mut_slice())?;
            self.mapping.insert(id.clone(), slot.slot())?;

            // Now that insert has succeeded - publish the slot. This method cannot fail, so
            // we do not need to worry about potentially unwinding the ID mapping.
            let id = slot.publish();

            // This is a rather expensive update.
            //
            // However, counters are only active with the `integration-test` feature, which
            // is not expected to be enabled for general use.
            self.local_counters().set_vector(1);

            Ok(diskann::provider::NoopGuard::new(id))
        };

        ready(work)
    }
}

////////////
// Search //
////////////

#[derive(Debug)]
pub struct SearchAccessor<'a> {
    reader: store::Reader<'a>,
    ids: AdjacencyList<u32>,
    expand_beam: Box<dyn ExpandBeam2 + 'a>,

    // The parent provider for the accessor.
    provider: &'a (dyn std::any::Any + Send + Sync),
    start_points: std::ops::Range<u32>,
    counters: LocalCounters<'a>,
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
                match self.reader.read(p.into_usize()) {
                    Some(point) => {
                        // Counters are no-ops without `integration-test`.
                        self.counters.get_vector(1);
                        self.counters.query_distance(1);

                        f(p, self.expand_beam.evaluate(point)?);
                    }
                    None => {
                        return Err(ANNError::message(
                            ANNErrorKind::Opaque,
                            "could not retrieve start point",
                        ));
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
                self.reader.neighbors().get(i, &mut self.ids).unwrap();
                self.counters.get_neighbors(1);

                // Filter out unvisited IDs and ensure that all the IDs we are about
                self.ids
                    .retain(|i| pred.eval_mut(i) && self.reader.is_in_bounds(i.into_usize()));

                // TODO: Move to an external buffer to avoid any dynamic dispatcn in
                // `expand_beam_inner` - then we can do a bulk-update on the counters.
                let mut on_neighbors = |id, distance| {
                    self.counters.get_vector(1);
                    self.counters.query_distance(1);

                    on_neighbors(id, distance);
                };

                unsafe {
                    self.expand_beam
                        .expand_beam(&self.ids, 8, &self.reader, &mut on_neighbors)
                }?;
            }

            Ok(())
        };

        ready(work)
    }
}

trait ExpandBeam2: Send + Sync + std::fmt::Debug {
    /// Evaluate a raw distance function.
    fn evaluate(&self, x: &[u8]) -> ANNResult<f32>;

    unsafe fn expand_beam(
        &self,
        list: &[u32],
        lookahead: usize,
        reader: &store::Reader<'_>,
        f: &mut dyn FnMut(u32, f32),
    ) -> ANNResult<()>;
}

#[derive(Debug)]
#[repr(transparent)]
struct ExpandBeamImpl<T, const BYTES: usize>(T);

impl<T, const BYTES: usize> ExpandBeam2 for ExpandBeamImpl<T, BYTES>
where
    T: layers::QueryDistance,
{
    fn evaluate(&self, x: &[u8]) -> ANNResult<f32> {
        self.0.evaluate(x)
    }

    unsafe fn expand_beam(
        &self,
        list: &[u32],
        lookahead: usize,
        reader: &store::Reader<'_>,
        f: &mut dyn FnMut(u32, f32),
    ) -> ANNResult<()> {
        unsafe { expand_beam_inner::<T, BYTES>(&self.0, list, lookahead, reader, f) }
    }
}

#[derive(Debug)]
struct ExpandBeamVisitor {
    bytes: Bytes,
}

impl<'a> layers::QueryVisitor<'a> for ExpandBeamVisitor {
    type Output = Box<dyn ExpandBeam2 + 'a>;

    fn visit_sized<const BYTES: usize, T>(self, distance: T) -> Self::Output
    where
        T: QueryDistance + 'a,
    {
        // Make sure there's no lying.
        assert_eq!(Bytes::new(BYTES + 1), self.bytes);
        Box::new(ExpandBeamImpl::<_, BYTES>(distance))
    }

    fn visit<T>(self, distance: T) -> Self::Output
    where
        T: QueryDistance + 'a,
    {
        Box::new(ExpandBeamImpl::<_, 0>(distance))
    }
}

#[inline(always)]
unsafe fn prefetch(ptr: *const u8, len: usize) {
    use std::arch::x86_64::*;

    // Fetch the last cache line (the one with the tag) first.
    let stride = Bytes::CACHELINE.value();
    let ptr = ptr.cast::<i8>();
    let lines = len.div_ceil(stride);
    if lines == 0 {
        return;
    }

    unsafe { _mm_prefetch(ptr.add(stride * (lines - 1)), _MM_HINT_T0) };
    for i in 0..(lines - 1) {
        unsafe {
            _mm_prefetch(ptr.add(stride * i), _MM_HINT_T0);
        }
    }
}

/// Safety (no # yet because we need to revisit this - clippy will lint)
///
/// * The concrete type of `distance` must be `T`.
/// * All items in `list` must in-bounds with respect to `reader`.
/// * The number of bytes associated with `N` cache lines must "make sense".
#[inline]
unsafe fn expand_beam_inner<T, const BYTES: usize>(
    distance: &T,
    list: &[u32],
    lookahead: usize,
    reader: &store::Reader<'_>,
    f: &mut dyn FnMut(u32, f32),
) -> ANNResult<()>
where
    T: layers::QueryDistance,
{
    debug_assert!(
        BYTES + 1 <= reader.bytes().value(),
        "we really rely on this: {}, bytes = {}",
        BYTES + 1,
        reader.bytes()
    );

    let bytes = if BYTES == 0 {
        reader.bytes().value()
    } else {
        BYTES + 1
    };

    let len = list.len();
    let lookahead = lookahead.min(len);

    for j in 0..lookahead {
        unsafe {
            prefetch(
                reader
                    .read_raw_unchecked(list.get_unchecked(j).into_usize())
                    .as_ptr()
                    .cast(),
                bytes,
            )
        }
    }

    let mut j = lookahead;
    for &i in list.iter() {
        if j != len {
            unsafe {
                prefetch(
                    reader
                        .read_raw_unchecked(list.get_unchecked(j).into_usize())
                        .as_ptr()
                        .cast(),
                    bytes,
                )
            }
            j += 1;
        }

        if let Some(data) = unsafe { reader.read_in_bounds(i.into_usize()) } {
            f(i, distance.evaluate(data)?)
        }
    }

    Ok(())
}

////////////
// Insert //
////////////

/// The [`glue::PruneAccessor`] implementation for [`Provider`].
///
/// This type implements zero-copy access to the data within its parent provider during prunes.
#[derive(Debug)]
pub struct PruneAccessor<'a> {
    reader: store::Reader<'a>,
    distance: &'a dyn layers::Distance,
    counters: LocalCounters<'a>,
}

/// The distance computer for [`PruneAccessor`].
#[derive(Debug)]
pub struct Distance<'a> {
    distance: &'a dyn layers::Distance,
    counters: LocalCounters<'a>,
}

impl<'a> Distance<'a> {
    fn new(distance: &'a dyn layers::Distance, counters: LocalCounters<'a>) -> Self {
        Self { distance, counters }
    }
}

impl diskann_vector::DistanceFunction<&[u8], &[u8], f32> for Distance<'_> {
    #[inline]
    fn evaluate_similarity(&self, x: &[u8], y: &[u8]) -> f32 {
        self.counters.distance_ref(1);
        self.distance.evaluate(x, y).unwrap()
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

    type ElementRef<'a> = &'a [u8];

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
        _itr: Itr,
    ) -> ANNResult<(Self::View<'a>, Self::Distance<'a>)>
    where
        Itr: ExactSizeIterator<Item = Self::Id> + Clone + Send + Sync,
    {
        Ok((self, Distance::new(self.distance, self.counters.fork())))
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
            Ok(self.reader.neighbors().get(id, neighbors)?)
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
            Ok(self.reader.neighbors().set(id, neighbors)?)
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
            let lock = self.reader.neighbors().lock(id)?;

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
    type ElementRef<'a> = &'a [u8];
    type Element<'a>
        = &'a [u8]
    where
        Self: 'a;
    fn get(&self, id: u32) -> Option<&[u8]> {
        match self.reader.read(id.into_usize()) {
            Some(data) => {
                self.counters.get_vector_ref(1);
                Some(data)
            }
            None => None,
        }
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
        let reader = provider.store.reader()?;
        let expand_beam = <L as layers::Search>::query_distance(
            &provider.layer,
            query,
            ExpandBeamVisitor {
                bytes: provider.store.bytes(),
            },
        )?;

        let accessor = SearchAccessor {
            reader,
            ids: AdjacencyList::new(),
            expand_beam,
            provider,
            start_points: provider.store.frozen(),
            counters: provider.local_counters(),
        };
        Ok(accessor)
    }
}

pub fn test_function<'a>(
    x: &'a Provider<layers::Full<u8>>,
    strategy: &'a Strategy,
    context: &'a Context,
    query: &'a [u8],
) -> SearchAccessor<'a> {
    glue::SearchStrategy::search_accessor(strategy, x, context, query).unwrap()
}

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
            let provider = accessor.provider.downcast_ref::<Provider<L, M>>().unwrap();
            let mut count = 0;
            for c in candidates {
                if let Some(ext) = provider.mapping.to_external(c.id) {
                    if output.push(ext, c.distance).is_available() {
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
    L: layers::Layer + layers::AsDistance,
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
        Ok(PruneAccessor {
            reader: provider.store.reader()?,
            distance: <L as layers::AsDistance>::as_distance(&provider.layer),
            counters: provider.local_counters(),
        })
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

// TODO: This is such a hack.
impl<M> glue::InplaceDeleteStrategy<Provider<layers::Full<f32>, M>> for Strategy
where
    M: Id,
{
    type DeleteElement<'a> = &'a [f32];
    type DeleteElementGuard = Box<[f32]>;
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
        provider: &'a Provider<layers::Full<f32>, M>,
        _context: &'a Context,
        id: u32,
    ) -> impl Future<Output = Result<Self::DeleteElementGuard, Self::DeleteElementError>> + Send
    {
        let work = move || {
            let reader = provider.store.reader().unwrap();
            let mut buf: Box<[_]> = std::iter::repeat_n(0.0, provider.layer.dim()).collect();
            let data = reader.read(id.into_usize()).unwrap();
            bytemuck::must_cast_slice_mut::<f32, u8>(&mut buf).copy_from_slice(data);
            Ok(buf)
        };
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
    use diskann_vector::distance::Metric;

    use crate::layers::Full;

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

        let full = Full::<f32>::new(grid.dim().into(), Metric::L2);

        let config = Config::new(grid.num_points(size), degree);

        let provider = Provider::<_, u64>::new(full, config, std::iter::once(start.as_slice()));
        assert_eq!(provider.max_degree(), degree);

        let config = diskann::graph::config::Builder::new(
            2 * (grid.dim() as usize),
            diskann::graph::config::MaxDegree::new(provider.max_degree()),
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
        let knn = Knn::new(10, 10, None).unwrap();
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
