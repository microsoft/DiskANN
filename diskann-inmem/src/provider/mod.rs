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
//! * Lack of save/load support: The index is currently ephemeral, but there are plans to
//!   address this gap.

use std::hash::Hash;

use diskann::{
    ANNError, ANNResult,
    graph::{
        AdjacencyList, DiskANNIndex as Index, SearchOutputBuffer,
        glue::{self, HybridPredicate},
        workingset,
    },
    neighbor::Neighbor,
    provider,
};
use parking_lot::Mutex;

use crate::{
    bitvector::BitVector,
    counters::{Counters, LocalCounters},
    epoch,
    ids::IdMap,
    neighbors::Neighbors,
    num::{Capacity, IdLimit, MaxDegree, Range},
    repr,
};

#[doc(hidden)]
pub mod buildonly;

/// Aggregate trait for the external ID type of [`Provider`].
pub trait Id: Send + Sync + Hash + Eq + Clone + 'static {}

impl<T> Id for T where T: Send + Sync + Hash + Eq + Clone + 'static {}

/// An in-memory data-provider for DiskANN's graph indexing algorithms.
///
/// The first type parameter `R` is a [`repr::Representation`] for describing the kind of data
/// stored within the provider. The second parameter `M` is the associated data for items
/// inserted into the provider.
#[derive(Debug)]
pub struct Provider<R, M = u32>
where
    M: Id,
{
    // Data representation and storage.
    representation: R,
    // ID translation.
    mapping: IdMap<M>,
    // `Counters` is only non-trivial under the `integration-test` feature flag. Otherwise,
    // all counter related operations are no-ops.
    counters: Counters,
}

impl<R, M> Provider<R, M>
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

impl<R, M> Provider<R, M>
where
    R: repr::Representation,
    M: Id,
{
    /// Construct a new [`Provider`].
    pub fn new<C>(config: C) -> ANNResult<Self>
    where
        C: repr::RepresentationConfig<Representation = R>,
    {
        let representation = <_ as repr::RepresentationConfig>::build(config)?;
        let mapping = IdMap::new(representation.id_limit());

        Ok(Self {
            representation,
            mapping,
            counters: Counters::new(),
        })
    }

    /// Return the [`Capacity`] of the index.
    pub fn capacity(&self) -> Capacity {
        self.representation.capacity()
    }

    /// Return the maximum number of neighbors that can be stored in the provider's graph.
    pub fn max_degree(&self) -> MaxDegree {
        self.representation.max_degree()
    }

    /// Return the [`IdLimit`] for `self`.
    ///
    /// Keep private since it relates to the internal ID space.
    fn id_limit(&self) -> IdLimit {
        self.representation.id_limit()
    }
}

////////////
// Scoped //
////////////

/// Construct a [`Scoped`] view of `index`.
///
/// Items deleted in `index` while the [`Scoped`] is held will be prevented from reclamation.
/// This allows operations that require some amount of pre-processing (like [`Filtered`])
/// to maintain a coherent view of the `index` by prevent ABA style bugs that would be allowed.
///
/// # Warning
///
/// This function is intended to be temporary pending architectural reworking of `diskann`.
#[doc(hidden)]
pub fn scoped<R, M>(index: &Index<Provider<R, M>>) -> ANNResult<Scoped<'_, R, M>>
where
    R: repr::Representation,
    M: Id,
{
    let guard =
        <_ as repr::Representation>::read_guard(&index.provider().representation)?.into_inner();
    Ok(Scoped::new(index, guard))
}

/// A scoped view of [`Provider`], preventing id receclamation while held.
///
/// See: [`scope`].
#[doc(hidden)]
#[derive(Debug)]
pub struct Scoped<'a, R, M>
where
    R: repr::Representation,
    M: Id,
{
    index: &'a Index<Provider<R, M>>,
    guard: epoch::Guard<'a>,
}

impl<'a, R, M> Scoped<'a, R, M>
where
    R: repr::Representation,
    M: Id,
{
    fn new(index: &'a Index<Provider<R, M>>, guard: epoch::Guard<'a>) -> Self {
        Self { index, guard }
    }

    /// Evaluate the `predicate` on all current external-ids.
    ///
    /// Return a [`BitVector`] of length [`Provider::id_limit`] with entries set to 1 if ane
    /// only if the associated external-id satisfies `predicate`.
    fn evaluate<F>(&self, mut predicate: F) -> BitVector
    where
        F: FnMut(&M) -> bool,
    {
        let provider = self.index.provider();
        let mut bitvec = BitVector::new(<R as repr::Representation>::id_limit(
            &provider.representation,
        ));

        // We need to protect against the case where a predicate also tries to insert and
        // delete items, which would acquire the ID translation locks.
        //
        // If we were holding these locks when that happened, this would deadlock.
        //
        // Instead, we acquire and release locks for each item.
        //
        // FIXME: This is slow and can be fixed by moving internal-to-external ID translation
        // behind the epoch boundary, then we can read through it without locks.
        for internal in 0..self.index.provider().id_limit().value() {
            if let Some(external) = provider.mapping.to_external(internal)
                && predicate(&external)
            {
                bitvec.set_true(internal);
            }
        }

        bitvec
    }

    /// Construct a [`Filtered`] with `query` and a pre-processed `predicate`.
    ///
    /// The returned [`Filtered`] can be used for multiple filtered searches with the same
    /// `query`/`predicate` pair.
    ///
    /// The `predicate` will be called at most once for each ID currently registered with
    /// the parent [`Provider`].
    ///
    /// # Note
    ///
    /// While the [`Filtered`] is held, it prevents id reclamation of its parent [`Provider`].
    /// It should only be held as long as necessary, or in situations where it is known the
    /// parent [`Provider`] is not undergoing rapid inserts and deletes.
    ///
    /// In addition, items inserted into the [`Provider`] after the [`Filtered`] is
    /// constructed will not be returned from searches.
    ///
    /// # Warning
    ///
    /// This is likely temporary function pending architectural changes to `diskann`.
    pub fn filtered<'x, F>(
        &'x self,
        query: <R as repr::Search>::Query<'x>,
        predicate: F,
    ) -> ANNResult<Filtered<'x, R, M>>
    where
        R: repr::Search,
        F: FnMut(&M) -> bool,
    {
        let provider = self.index.provider();
        let search = <R as repr::Search>::search_accessor(
            &provider.representation,
            query,
            provider,
            Some(repr::ReadGuard::new(self.guard.share())),
            provider.local_counters(),
        )?;

        let filter = self.evaluate(predicate);
        Ok(Filtered {
            index: self.index,
            accessor: FilteredAccessor { search, filter },
        })
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
}

// TODO: The element-status checks here are profoundly approximate because we try to avoid
// any kind of EBR registration.
//
// `diskann` has plans to move deletion checks behind an accessor trait, which will help
// with this situation.
impl<R, M> diskann::provider::Delete for Provider<R, M>
where
    R: repr::Representation,
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
        <R as repr::Representation>::retire(&self.representation, entry.internal())?;

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
        match <R as repr::Representation>::is_readable(&self.representation, id) {
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

impl<T, R, M> diskann::provider::SetElement<T> for Provider<R, M>
where
    R: repr::Set<T>,
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
            let guard = <R as repr::Set<T>>::set(&self.representation, element)?;
            let internal = <_ as repr::Guard>::id(&guard);
            self.mapping.insert(id.clone(), internal)?;

            // Now that insert has succeeded - publish the slot. This method cannot fail, so
            // we do not need to worry about potentially unwinding the ID mapping.
            <_ as repr::Guard>::publish(guard);

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
    expand_beam: Box<dyn repr::ExpandBeam + 'a>,
    id_limit: IdLimit,
    buffer: Vec<Neighbor<u32>>,

    // Post-process
    post_process: Option<Box<dyn repr::PostProcess + 'a>>,

    // The parent provider for the accessor.
    provider: &'a (dyn std::any::Any + Send + Sync),
    start_points: Range,
    counters: LocalCounters<'a>,
}

impl<'a> SearchAccessor<'a> {
    pub(crate) fn new(
        neighbors: &'a Neighbors,
        expand_beam: Box<dyn repr::ExpandBeam + 'a>,
        post_process: Option<Box<dyn repr::PostProcess + 'a>>,
        provider: &'a (dyn std::any::Any + Send + Sync),
        start_points: Range,
        counters: LocalCounters<'a>,
    ) -> Self {
        let id_limit = expand_beam.id_limit();
        Self {
            neighbors,
            ids: AdjacencyList::with_capacity(neighbors.max_degree().value()),
            expand_beam,
            id_limit,
            buffer: vec![Default::default(); neighbors.max_degree().value()],
            post_process,
            provider,
            start_points,
            counters,
        }
    }

    fn distances_to<Itr, F>(&mut self, itr: Itr, mut f: F) -> ANNResult<()>
    where
        Itr: IntoIterator<Item = u32>,
        F: FnMut(u32, f32),
    {
        for p in itr {
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
    }

    fn expand_beam<Itr, P, F>(
        &mut self,
        ids: Itr,
        mut filter: P,
        mut on_neighbors: F,
    ) -> ANNResult<()>
    where
        Itr: IntoIterator<Item = u32>,
        P: FnMut(&u32) -> bool,
        F: FnMut(u32, f32),
    {
        // Post-processing code that reuses `buffer` could leave it too small.
        //
        // This restores the buffer size.
        if self.buffer.len() < self.neighbors.max_degree().value() {
            self.resize_buffer();
        }

        for i in ids {
            self.neighbors.get(i, &mut self.ids)?;
            self.counters.get_neighbors(1);

            // Filter out unvisited IDs and ensure that all the IDs we are about
            self.ids
                .retain(|i| filter(i) && self.id_limit.is_in_bounds(*i));

            // This should always hold, but let's double check.
            assert!(self.buffer.len() >= self.ids.len());

            // SAFETY:
            //
            // 1. We've verified that each entry in `self.ids` is in-bounds using
            //   the `IdLimit` derived from this `ExpandBeam` object (enforced in the
            //   constructor).
            //
            // 2. `self.buffer` is long enough to hold all the IDs.
            let processed = unsafe { self.expand_beam.expand_beam(&self.ids, &mut self.buffer) }?;

            self.counters.get_vector(processed as u64);
            self.counters.query_distance(processed as u64);

            self.buffer
                .iter()
                .take(processed)
                .for_each(|neighbor| on_neighbors(*neighbor.id(), *neighbor.distance()));
        }

        Ok(())
    }

    // Resize the internal buffer if needed.
    #[cold]
    fn resize_buffer(&mut self) {
        self.buffer
            .resize_with(self.neighbors.max_degree().value(), Default::default)
    }

    fn post_process_as<R, M>(
        &mut self,
        itr: impl IntoIterator<Item = Neighbor<u32>>,
        output: &mut (impl SearchOutputBuffer<M> + ?Sized),
    ) -> ANNResult<usize>
    where
        R: 'static,
        M: Id,
    {
        // By construction - the downcast should succeed. Otherwise, this is a program bug.
        let provider = match self.provider.downcast_ref::<Provider<R, M>>() {
            Some(provider) => provider,
            None => return Err(ANNError::message("bad any cast")),
        };

        let mut count = 0;
        let mut push = |neighbor: Neighbor<u32>| -> bool {
            if let Some(ext) = provider.mapping.to_external(*neighbor.id()) {
                if output
                    .push(Neighbor::new(ext, *neighbor.distance()))
                    .is_available()
                {
                    count += 1;
                    true
                } else {
                    false
                }
            } else {
                true
            }
        };

        // If there is a registered post-process step, run that first.
        if let Some(post_process) = &mut self.post_process {
            self.buffer.clear();
            Extend::extend(&mut self.buffer, itr);
            post_process.post_process(&mut self.buffer)?;

            for neighbor in self.buffer.iter() {
                if !push(*neighbor) {
                    break;
                }
            }
        } else {
            for neighbor in itr {
                if !push(neighbor) {
                    break;
                }
            }
        }

        Ok(count)
    }

    #[cfg(test)]
    pub(crate) fn get_expand_beam(&self) -> &dyn repr::ExpandBeam {
        &*self.expand_beam
    }

    #[cfg(all(test, feature = "quantization"))]
    pub(crate) fn get_post_process(&mut self) -> Option<&mut (dyn repr::PostProcess + 'a)> {
        self.post_process.as_deref_mut()
    }
}

impl diskann::provider::HasId for SearchAccessor<'_> {
    type Id = u32;
}

impl glue::SearchAccessor for SearchAccessor<'_> {
    fn starting_points(
        &self,
    ) -> impl std::future::Future<Output = ANNResult<Vec<Self::Id>>> + Send {
        std::future::ready(Ok(self.start_points.iter().collect()))
    }

    fn start_point_distances<F>(
        &mut self,
        f: F,
    ) -> impl std::future::Future<Output = ANNResult<()>> + Send
    where
        F: FnMut(Self::Id, f32) + Send,
    {
        ready(move || self.distances_to(self.start_points, f))
    }

    fn expand_beam<Itr, P, F>(
        &mut self,
        ids: Itr,
        mut pred: P,
        on_neighbors: F,
    ) -> impl std::future::Future<Output = ANNResult<()>> + Send
    where
        Itr: Iterator<Item = Self::Id> + Send,
        P: HybridPredicate<Self::Id> + Send + Sync,
        F: FnMut(Self::Id, f32) + Send,
    {
        ready(move || self.expand_beam(ids, |i| pred.eval_mut(i), on_neighbors))
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
    prune: Box<dyn repr::Prune + 'a>,
    neighbors: &'a Neighbors,
    counters: LocalCounters<'a>,
}

impl<'a> PruneAccessor<'a> {
    pub(crate) fn new(
        prune: Box<dyn repr::Prune + 'a>,
        neighbors: &'a Neighbors,
        counters: LocalCounters<'a>,
    ) -> Self {
        Self {
            prune,
            neighbors,
            counters,
        }
    }

    #[cfg(test)]
    pub(crate) fn get_prune(&mut self) -> &mut dyn repr::Prune {
        &mut *self.prune
    }
}

/// The distance computer for [`PruneAccessor`].
#[derive(Debug)]
pub struct Distance<'a> {
    prune: &'a dyn repr::Prune,
    counters: LocalCounters<'a>,
}

impl<'a> Distance<'a> {
    fn new(prune: &'a dyn repr::Prune, counters: LocalCounters<'a>) -> Self {
        Self { prune, counters }
    }
}

/// An opaque element-ref for [`PruneAccessor`].
#[derive(Debug, Clone, Copy)]
#[repr(transparent)]
pub struct ElementRef<'a>(&'a [u8]);

impl<'a> diskann_utils::Reborrow<'a> for ElementRef<'_> {
    type Target = ElementRef<'a>;
    fn reborrow(&'a self) -> Self {
        Self(self.0)
    }
}

impl diskann_vector::DistanceFunction<ElementRef<'_>, ElementRef<'_>, f32> for Distance<'_> {
    #[inline]
    fn evaluate_similarity(&self, x: ElementRef<'_>, y: ElementRef<'_>) -> f32 {
        self.counters.distance_ref(1);
        self.prune.evaluate(x.0, y.0)
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

    type ElementRef<'a> = ElementRef<'a>;

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
    type ElementRef<'a> = ElementRef<'a>;
    type Element<'a>
        = ElementRef<'a>
    where
        Self: 'a;

    fn get(&self, id: u32) -> Option<ElementRef<'_>> {
        self.counters.get_vector_ref(1);
        self.prune.get(id).map(ElementRef)
    }
}

////////////////
// Strategies //
////////////////

#[derive(Debug, Clone, Copy)]
pub struct Strategy;

impl<'a, R, M> glue::SearchStrategy<'a, Provider<R, M>, R::Query<'a>> for Strategy
where
    R: repr::Search,
    M: Id,
{
    type SearchAccessor = SearchAccessor<'a>;
    type SearchAccessorError = ANNError;

    fn search_accessor(
        &'a self,
        provider: &'a Provider<R, M>,
        _context: &'a Context,
        query: R::Query<'a>,
    ) -> ANNResult<SearchAccessor<'a>> {
        <R as repr::Search>::search_accessor(
            &provider.representation,
            query,
            provider,
            None,
            provider.local_counters(),
        )
    }
}

// This is a utility for helping inspect the generated code for `ExpandBeam`.
pub fn test_function<'a>(
    x: &'a Provider<repr::Full<u8>>,
    strategy: &'a Strategy,
    context: &'a Context,
    query: &'a [u8],
) -> ANNResult<SearchAccessor<'a>> {
    glue::SearchStrategy::search_accessor(strategy, x, context, query)
}

/// Perform ID translation during post-processing.
#[derive(Debug, Clone, Copy)]
pub struct Translate<R, M>(std::marker::PhantomData<(R, M)>);

impl<R, M> Default for Translate<R, M> {
    fn default() -> Self {
        Self(std::marker::PhantomData)
    }
}

impl<'a, R, M> glue::SearchPostProcess<SearchAccessor<'a>, R::Query<'a>, M> for Translate<R, M>
where
    R: repr::Search,
    M: Id,
{
    type Error = ANNError;

    fn post_process<I, B>(
        &self,
        accessor: &mut SearchAccessor<'_>,
        _query: R::Query<'a>,
        candidates: I,
        output: &mut B,
    ) -> impl std::future::Future<Output = Result<usize, Self::Error>> + Send
    where
        I: Iterator<Item = Neighbor<u32>> + Send,
        B: SearchOutputBuffer<M> + Send + ?Sized,
    {
        ready(move || accessor.post_process_as::<R, M>(candidates, output))
    }
}

impl<'a, R, M> glue::DefaultPostProcessor<'a, Provider<R, M>, R::Query<'a>, M> for Strategy
where
    R: repr::Search,
    M: Id,
{
    diskann::default_post_processor!(Translate<R, M>);
}

impl<R, M> glue::PruneStrategy<Provider<R, M>> for Strategy
where
    R: repr::Insert,
    M: Id,
{
    type PruneAccessor<'a> = PruneAccessor<'a>;
    type PruneAccessorError = ANNError;

    fn prune_accessor<'a>(
        &self,
        provider: &'a Provider<R, M>,
        _context: &'a Context,
        _capacity: usize,
    ) -> ANNResult<PruneAccessor<'a>> {
        <R as repr::Insert>::prune_accessor(&provider.representation, provider.local_counters())
    }
}

impl<'a, R, M> glue::InsertStrategy<'a, Provider<R, M>, R::Query<'a>> for Strategy
where
    R: repr::Insert,
    M: Id,
{
    type SearchAccessor = SearchAccessor<'a>;
    type SearchAccessorError = ANNError;
    type PruneStrategy = Self;

    fn insert_search_accessor(
        &'a self,
        provider: &'a Provider<R, M>,
        _context: &'a Context,
        vector: R::Query<'a>,
    ) -> Result<Self::SearchAccessor, Self::SearchAccessorError> {
        <R as repr::Insert>::insert_search_accessor(
            &provider.representation,
            vector,
            provider,
            None,
            provider.local_counters(),
        )
    }

    fn prune_strategy(&self) -> Self::PruneStrategy {
        *self
    }
}

impl<R, B, M> glue::MultiInsertStrategy<Provider<R, M>, B> for Strategy
where
    R: repr::Insert,
    B: for<'a> glue::Batch<Element<'a> = R::Query<'a>>,
    M: Id,
{
    type Seed = ();
    type FinishError = diskann::error::Infallible;
    type PruneStrategy = Self;
    type InsertStrategy = Self;
    fn insert_strategy(&self) -> Self {
        *self
    }
    fn finish<Itr>(
        &self,
        _provider: &Provider<R, M>,
        _context: &Context,
        _batch: &std::sync::Arc<B>,
        _ids: Itr,
    ) -> impl std::future::Future<Output = Result<Self::Seed, Self::FinishError>> + Send
    where
        Itr: ExactSizeIterator<Item = u32> + Send,
    {
        std::future::ready(Ok(()))
    }
    fn seeded_prune_accessor<'a>(
        &'a self,
        provider: &'a Provider<R, M>,
        _context: &'a Context,
        _seed: &'a (),
        _capacity: usize,
    ) -> ANNResult<PruneAccessor<'a>> {
        <R as repr::Insert>::prune_accessor(&provider.representation, provider.local_counters())
    }
}

impl<T, M> glue::InplaceDeleteStrategy<Provider<repr::Full<T>, M>> for Strategy
where
    M: Id,
    T: repr::FullPrecision,
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
        provider: &'a Provider<repr::Full<T>, M>,
        _context: &'a Context,
        id: u32,
    ) -> impl Future<Output = Result<Self::DeleteElementGuard, Self::DeleteElementError>> + Send
    {
        ready(move || provider.representation.get(id))
    }
}

//---------//
// Filters //
//---------//

/// A scoped view of a [`Provider`] with a pre-evaluate query and filter predicate.
///
/// See: [`Scoped::filtered`].
#[derive(Debug)]
pub struct Filtered<'a, R, M>
where
    R: repr::Representation,
    M: Id,
{
    // FIXME: `accessor` has everything we need. We don't need `index` except that `diskann`
    // requires it!
    index: &'a Index<Provider<R, M>>,
    accessor: FilteredAccessor<'a>,
}

impl<R, M> Filtered<'_, R, M>
where
    R: repr::Representation,
    M: Id,
{
    /// Run [`diskann::graph::search::InlineFilterSearch`] and store the results in `output`.
    pub fn inline_filtered<O>(
        &mut self,
        params: diskann::graph::search::InlineFilterSearch,
        output: &mut O,
    ) -> ANNResult<diskann::graph::index::SearchStats>
    where
        O: diskann::graph::SearchOutputBuffer<M> + Send,
    {
        let strategy = Smuggle::new(&mut self.accessor);
        let fut = self.index.search(params, &strategy, &Context, (), output);
        crate::poll(fut)
    }

    /// Run [`diskann::graph::search::MultihopFilterSearch`] and store the results in `output`.
    pub fn multihop_filtered<O>(
        &mut self,
        params: diskann::graph::search::MultihopFilterSearch,
        output: &mut O,
    ) -> ANNResult<diskann::graph::index::SearchStats>
    where
        O: diskann::graph::SearchOutputBuffer<M> + Send,
    {
        let strategy = Smuggle::new(&mut self.accessor);
        let fut = self.index.search(params, &strategy, &Context, (), output);
        crate::poll(fut)
    }
}

//~~~~~~~~~~~~~~~~~//
// HERE BE DRAGONS //
//~~~~~~~~~~~~~~~~~//

// Yes - that is right.
//
// Our strategy embeds the accessor it's going to use directly in the `Strategy` in a
// single use way. The `diskann` API doesn't allow us (yet) to just do the right thing and
// use the accesor directly, so we're forced to do slightly cursed workarounds like this
// until we can fix core `diskann`.
//
// The main planned architectural change to `diskann` is to make all the search/insert/delete
// methods receive accessors only without a `DiskANNIndex` or any strategies.
//
// Then we will be able to avoid this horrible hack and simplify `Scoped`/`Filtered`.
#[derive(Debug)]
struct Smuggle<'borrow, 'data>(Mutex<Option<FilteredAccessorRef<'borrow, 'data>>>);

impl<'borrow, 'data> Smuggle<'borrow, 'data> {
    fn new(accessor: &'borrow mut FilteredAccessor<'data>) -> Self {
        Self(Mutex::new(Some(FilteredAccessorRef(accessor))))
    }
}

#[expect(
    clippy::unwrap_used,
    reason = "this is a private type - we ensure the mutex is uncontended and used just once"
)]
impl<'call, 'borrow, 'data, R, M> glue::SearchStrategy<'call, Provider<R, M>, ()>
    for Smuggle<'borrow, 'data>
where
    R: repr::Representation,
    M: Id,
{
    type SearchAccessorError = diskann::error::Infallible;
    type SearchAccessor = FilteredAccessorRef<'borrow, 'data>;

    fn search_accessor(
        &'call self,
        _provider: &'call Provider<R, M>,
        _context: &'call Context,
        _query: (),
    ) -> Result<Self::SearchAccessor, diskann::error::Infallible> {
        Ok(self.0.try_lock().unwrap().take().unwrap())
    }
}

impl<'call, 'borrow, 'data, R, M> glue::DefaultPostProcessor<'call, Provider<R, M>, (), M>
    for Smuggle<'borrow, 'data>
where
    R: repr::Representation,
    M: Id,
{
    diskann::default_post_processor!(Translate<R, M>);
}

impl<R, M> glue::SearchPostProcess<FilteredAccessorRef<'_, '_>, (), M> for Translate<R, M>
where
    R: repr::Representation,
    M: Id,
{
    type Error = ANNError;

    fn post_process<I, B>(
        &self,
        accessor: &mut FilteredAccessorRef<'_, '_>,
        _query: (),
        candidates: I,
        output: &mut B,
    ) -> impl std::future::Future<Output = Result<usize, Self::Error>> + Send
    where
        I: Iterator<Item = Neighbor<u32>> + Send,
        B: SearchOutputBuffer<M> + Send + ?Sized,
    {
        ready(move || {
            accessor
                .0
                .search
                .post_process_as::<R, M>(candidates, output)
        })
    }
}

/// The actual [`glue::FilteredAccessor`] for filtered search.
///
/// See: [`Filtered`].
#[derive(Debug)]
struct FilteredAccessor<'a> {
    search: SearchAccessor<'a>,
    filter: BitVector,
}

/// A temporary wrapper for [`FilteredAccessor`] to inject it directly into the `diskann`
/// search methods.
///
/// This should become unnecessary in the future.
#[derive(Debug)]
struct FilteredAccessorRef<'borrow, 'data>(&'borrow mut FilteredAccessor<'data>);

impl diskann::provider::HasId for FilteredAccessorRef<'_, '_> {
    type Id = u32;
}

impl glue::FilteredAccessor for FilteredAccessorRef<'_, '_> {
    fn num_starting_points(&self) -> impl std::future::Future<Output = ANNResult<usize>> + Send {
        std::future::ready(Ok(self.0.search.start_points.len()))
    }

    fn start_point_distances<F>(
        &mut self,
        mut f: F,
    ) -> impl std::future::Future<Output = ANNResult<()>> + Send
    where
        F: FnMut(glue::Decision<Self::Id>, f32) + Send,
    {
        let work = move || {
            self.0
                .search
                .distances_to(self.0.search.start_points, |id, distance| {
                    f(glue::Decision::new(self.0.filter.get(id), id), distance)
                })
        };

        ready(work)
    }

    fn expand_beam_filtered<Itr, P, F>(
        &mut self,
        ids: Itr,
        mut pred: P,
        mut on_neighbors: F,
    ) -> impl std::future::Future<Output = ANNResult<()>> + Send
    where
        Itr: Iterator<Item = Self::Id> + Send,
        P: HybridPredicate<Self::Id> + Send + Sync,
        F: FnMut(glue::Decision<Self::Id>, f32) + Send,
    {
        let work = move || {
            self.0.search.expand_beam(
                ids,
                |i| pred.eval_mut(i),
                |id, distance| {
                    on_neighbors(glue::Decision::new(self.0.filter.get(id), id), distance)
                },
            )
        };

        ready(work)
    }

    fn expand_beam_accept_only<Itr, P, F>(
        &mut self,
        ids: Itr,
        mut pred: P,
        mut on_neighbors: F,
    ) -> impl std::future::Future<Output = ANNResult<()>> + Send
    where
        Itr: Iterator<Item = Self::Id> + Send,
        P: glue::Predicate<Self::Id> + glue::PredicateMut<glue::Accept<Self::Id>> + Send + Sync,
        F: FnMut(glue::Accept<Self::Id>, f32) + Send,
    {
        let work = move || {
            self.0.search.expand_beam(
                ids,
                |i| self.0.filter.get(*i) && pred.eval_mut(&glue::Accept::new(*i)),
                |id, distance| on_neighbors(glue::Accept::new(id), distance),
            )
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
        graph::{
            DiskANNIndex, InplaceDeleteMethod,
            search::{InlineFilterSearch, Knn, MultihopFilterSearch},
            test::synthetic::Grid,
        },
        neighbor::Neighbor,
        provider::{DataProvider, Delete},
    };
    use diskann_utils::{
        assert_contains,
        views::rowmajor::{self, Matrix},
    };
    use diskann_vector::distance::Metric;

    use crate::{num::Capacity, poll::poll};

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

        let config = repr::full::Config::new(
            Capacity::new(grid.num_points(size)),
            MaxDegree::new(degree),
            Metric::L2,
            rowmajor::Owned::row_vector(start.into()),
        )
        .unwrap();

        let provider = Provider::<_, u64>::new(config).unwrap();
        assert_eq!(provider.max_degree(), MaxDegree::new(degree));
        assert_eq!(provider.capacity(), Capacity::new(grid.num_points(size)));

        let config = diskann::graph::config::Builder::new(
            2 * (grid.dim() as usize),
            diskann::graph::config::MaxDegree::new(provider.max_degree().value()),
            10,
            (Metric::L2).into(),
        )
        .build()
        .unwrap();

        let index = DiskANNIndex::new(config, provider, None);

        for (i, data) in data.rows().enumerate() {
            index
                .insert(&Strategy, &Context, &((10 * i + 1) as u64), data)
                .await
                .unwrap();
        }

        // Verify that each ID round trips.
        for i in 0..data.nrows() {
            let i = (10 * i + 1) as u64;
            let internal = index.provider().to_internal_id(&Context, &i).unwrap();
            assert_ne!(
                internal as u64,
                i + 1,
                "this relies on the start points being located at the beginning of the id space\
                 this is **NOT** guaranteed",
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

        // Filtered searches work.
        //
        // What we do here is create a `Scoped` then a `Filtered`.
        //
        // We then hold onto these to ensure that algorithms like inplace delete can still
        // run but that IDs cannot be reclaimed until both to away.
        //
        // The filter excludes IDs where the 10s digit is odd.
        let scoped = scoped(&index).unwrap();
        let mut filtered = scoped
            .filtered(&[0.0, 0.0], |i| (*i / 10).is_multiple_of(2))
            .unwrap();

        {
            let mut neighbors = Vec::<Neighbor<u64>>::new();
            let ifs = InlineFilterSearch::new(knn, None);
            filtered
                .inline_filtered(ifs.clone(), &mut neighbors)
                .unwrap();

            // We should not observe entries with odd 10s digits.
            assert_eq!(neighbors[0].as_tuple(), (1, 0.0));
            assert_eq!(neighbors[1].as_tuple(), (61, 2.0));
            assert_eq!(neighbors[2].as_tuple(), (21, 4.0)); // This can be swapped with 3
            assert_eq!(neighbors[3].as_tuple(), (101, 4.0));

            // This is a bit sneaky.
            //
            // We mimic the case where a rerank style post-processing routine clears the
            // internal buffer and leaves it that way.
            //
            // The `expand_beam` should handle this case properly.
            filtered.accessor.search.buffer.clear();

            // Try again, this time with mulit-hop filter.
            neighbors.clear();
            let mfs = MultihopFilterSearch::new(knn);
            filtered.multihop_filtered(mfs, &mut neighbors).unwrap();

            // We should not observe entries with odd 10s digits.
            assert_eq!(neighbors[0].as_tuple(), (1, 0.0));
            assert_eq!(neighbors[1].as_tuple(), (61, 2.0));
            assert_eq!(neighbors[2].as_tuple(), (21, 4.0)); // This can be swapped with 3
            assert_eq!(neighbors[3].as_tuple(), (101, 4.0));
        }

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

        // This insert fails because even though there is an empty slot, it cannot be
        // reclaimed until the `scoped` is dropped.
        //
        // After this check, we free up these resources.
        assert_contains!(
            index
                .insert(&Strategy, &Context, &1, &[10.0, 10.0])
                .await
                .unwrap_err()
                .to_string(),
            "could not allocate a new slot",
        );

        drop(filtered);
        drop(scoped);

        // We can't insert an existing ID.
        assert_contains!(
            index
                .insert(&Strategy, &Context, &1, &[10.0, 10.0])
                .await
                .unwrap_err()
                .to_string(),
            "external id is already mapped",
        );

        // If we insert a new ID but the query vector is too long - make sure we leave the
        // provider untouched.
        assert_contains!(
            index
                .insert(&Strategy, &Context, &2, &[10.0, 10.0, 10.0])
                .await
                .unwrap_err()
                .to_string(),
            "expected slice of length 2",
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

        // Finally - test that we can create a crazy predicate for filtering that deletes
        // all the external IDs as they are evaluated.
        let scoped = super::scoped(&index).unwrap();
        let _ = scoped.filtered(&[0.0, 0.0], |i| {
            poll(index.provider().delete(&Context, i)).unwrap();
            true
        });
    }
}
