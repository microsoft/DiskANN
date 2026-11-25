/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::num::NonZeroUsize;

use diskann::{
    ANNError, ANNResult,
    graph::{SearchOutputBuffer, glue},
    neighbor::Neighbor,
};
use diskann_utils::views::rowmajor;

use crate::{
    bitvector::BitVector,
    counters::{Counters, LocalCounters},
    epoch,
    ids::IdMap,
    num::{Capacity, IdLimit, MaxDegree},
    poll::{poll, ready},
    repr,
};

//------------//
// Submodules //
//------------//

pub(crate) mod accessors;
mod algorithms;

#[doc(hidden)]
pub mod buildonly;

///////////
// Index //
///////////

/// Aggregate trait for the external ID type of [`Provider`].
pub trait Id: Send + Sync + std::hash::Hash + Eq + std::fmt::Debug + Clone + 'static {}

impl<T> Id for T where T: Send + Sync + std::hash::Hash + std::fmt::Debug + Eq + Clone + 'static {}

#[derive(Debug)]
pub struct Index<R, M>
where
    M: Id,
{
    // Data representation and storage
    representation: R,
    // ID translation.
    mapping: IdMap<M>,
    // The graph configuration.
    graph: diskann::graph::Config,

    // `Counters` is only non-trivial under the `integration-test` feature flag. Otherwise,
    // all counter related operations are no-ops.
    counters: Counters,
}

impl<R, M> Index<R, M>
where
    R: repr::Representation, // FIXME: Remove structural bounds
    M: Id,                   // FIXME: Remove structural bounds
{
    // FIXME: Cut out `provider` once feasible.
    pub fn new<C>(config: C, graph: diskann::graph::Config) -> ANNResult<Self>
    where
        C: repr::RepresentationConfig<Representation = R>,
    {
        let representation = <_ as repr::RepresentationConfig>::build(config)?;
        let mapping = IdMap::new(representation.id_limit());
        Ok(Self {
            representation,
            mapping,
            graph,
            counters: Counters::new(),
        })
    }

    pub fn read(&self) -> ANNResult<Read<'_, R, M>> {
        let guard = repr::RepresentationImpl::read_guard(&self.representation)?;
        Ok(Read { index: self, guard })
    }

    pub fn write(&self) -> ANNResult<Write<'_, R, M>> {
        let guard = repr::RepresentationImpl::write_guard(&self.representation)?;
        Ok(Write {
            index: &self,
            guard,
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

    /// Advance the epoch - returning the number of reclaimed IDs, if any.
    pub fn reclaim(&self) -> Option<usize> {
        repr::RepresentationImpl::reclaim(&self.representation)
    }

    fn id_limit(&self) -> IdLimit {
        self.representation.id_limit()
    }

    fn mapping(&self) -> &IdMap<M> {
        &self.mapping
    }

    fn to_internal(&self, id: &M) -> Option<u32> {
        self.mapping.to_internal(id)
    }

    fn to_external(&self, id: u32) -> Option<M> {
        self.mapping.to_external(id)
    }

    fn is_readable(&self, id: u32) -> Option<bool> {
        <_ as repr::RepresentationImpl>::is_readable(&self.representation, id)
    }

    fn temp_delete(&self, id: &M) -> ANNResult<u32> {
        // This guarantees that we have a valid mapping, but defers the actual deletion until
        // we know it's also safe to retire the internal slot.
        //
        // This ensures both either succeed or are aborted.
        let entry = match self.mapping().occupied_entry(id.clone()) {
            None => {
                return Err(ANNError::message("id already deleted"));
            }
            Some(e) => e,
        };

        // The internal ID.
        let internal = entry.internal();

        // An early return here will cause `entry` to be dropped, which will *not* cause
        // the delete to commit.
        <R as repr::RepresentationImpl>::retire(&self.representation, internal)?;

        // Successfully retired the internal slot. We can safely release the ID mapping.
        entry.delete();

        Ok(internal)
    }

    //------------------//
    // diskann-builtins //
    //------------------//

    #[doc(hidden)]
    #[cfg(feature = "diskann-builtins")]
    pub fn batch_insert_builtin<'a, T>(
        self: &'a std::sync::Arc<Self>,
        data: rowmajor::Owned<T>,
        ids: &'a [M],
        handle: &'a tokio::runtime::Handle,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = ANNResult<()>> + 'a>>
    where
        R: for<'q> repr::Insert<Vector<'q> = &'q [T]>,
        T: std::fmt::Debug + Send + Sync + 'static,
    {
        use diskann_utils::views::rowmajor::Matrix;

        Box::pin(async move {
            // Acquire a write guard and hold onto it for the duration of the batch insert.
            let write = self.write()?;

            // FIXME: allow all IDs to be rolled-back.
            // FIXME: assert lengths are the same.
            let ids: std::sync::Arc<[u32]> = std::iter::zip(data.rows(), ids.iter())
                .map(|(v, id)| write.set(v, id.clone()))
                .collect::<ANNResult<_>>()?;

            let batch_insert = algorithms::BatchInsert::new(self.clone(), data, &ids)?;
            diskann::graph::algorithms::batch::insert::standard(handle, ids, &self.graph, || {
                batch_insert.clone()
            })
            .await
        })
    }
}

impl<R, M> Index<R, M>
where
    M: Id,
{
    /// Return a snapshot of the current event counters.
    #[cfg(feature = "integration-test")]
    pub fn counters(&self) -> crate::integration::counters::CounterSnapshot {
        self.counters.snapshot()
    }

    /// A local set of counters that update the provider-wide counters in bulk.
    fn local_counters(&self) -> LocalCounters<'_> {
        self.counters.local()
    }

    /// Strongly restrict access to `representation` since it needs to work in combination
    /// with ID mapping.
    fn repr(&self) -> &R {
        &self.representation
    }
}

/// A read-only view over [`Index`].
///
/// # Note
///
/// Holding a [`Read`] will prevent deleted items from being reclaimed and thus should be
/// held for as short a duration as possible.
#[derive(Debug)]
pub struct Read<'a, R, M>
where
    M: Id,
{
    index: &'a Index<R, M>,
    guard: epoch::ReadGuard<'a>,
}

impl<R, M> Read<'_, R, M>
where
    M: Id,
{
    // FIXME: Remove uneecessary lifetime bounds
    pub fn search<'a, O>(
        &'a self,
        query: R::Query<'a>,
        params: diskann::graph::search::Knn,
        output: &mut O,
    ) -> ANNResult<diskann::graph::algorithms::SearchStats>
    where
        R: repr::Search,
        O: SearchOutputBuffer<M> + Send + ?Sized,
    {
        let mut accessor = repr::SearchImpl::search_accessor(
            &self.index.representation,
            query,
            Some(self.guard.share()),
            self.index.local_counters(),
        )?;

        let fut = <_ as diskann::graph::search::Search>::search(
            params,
            &mut accessor,
            Translate(self.index.mapping()),
            output,
        );

        poll(fut)
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
    pub fn filtered<'x, F>(
        &'x self,
        query: <R as repr::Search>::Query<'x>,
        predicate: F,
    ) -> ANNResult<Filtered<'x, M>>
    where
        R: repr::Search,
        F: FnMut(&M) -> bool,
    {
        let search = repr::SearchImpl::search_accessor(
            &self.index.representation,
            query,
            Some(self.guard.share()),
            self.index.local_counters(),
        )?;

        let filter = self.evaluate(predicate);
        Ok(Filtered {
            accessor: accessors::Filtered::new(search, filter),
            translate: Translate(self.index.mapping()),
        })
    }

    /// Evaluate the `predicate` on all current external-ids.
    ///
    /// Return a [`BitVector`] of length [`Provider::id_limit`] with entries set to 1 if ane
    /// only if the associated external-id satisfies `predicate`.
    fn evaluate<F>(&self, mut predicate: F) -> BitVector
    where
        R: repr::Representation,
        F: FnMut(&M) -> bool,
    {
        let mut bitvec = BitVector::new(<R as repr::RepresentationImpl>::id_limit(
            &self.index.representation,
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
        for internal in 0..self.index.id_limit().value() {
            if let Some(external) = self.index.mapping().to_external(internal)
                && predicate(&external)
            {
                bitvec.set_true(internal);
            }
        }

        bitvec
    }

    // FIXME: Make it go away
    fn share(&self) -> Self {
        Self {
            index: self.index,
            guard: self.guard.share(),
        }
    }
}

//----------------//
// Id translation //
//----------------//

#[derive(Debug)]
struct Translate<'a, M: Id>(&'a IdMap<M>);

impl<M: Id> Clone for Translate<'_, M> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<M: Id> Copy for Translate<'_, M> {}

impl<M> glue::SearchPostProcess<accessors::Search<'_>, M> for Translate<'_, M>
where
    M: Id,
{
    type Error = ANNError;

    fn post_process<I, B>(
        &self,
        accessor: &mut accessors::Search<'_>,
        candidates: I,
        output: &mut B,
    ) -> impl std::future::Future<Output = ANNResult<usize>> + Send
    where
        I: Iterator<Item = Neighbor<u32>> + Send,
        B: SearchOutputBuffer<M> + Send + ?Sized,
    {
        ready(|| accessor.post_process_as(*self, candidates, output))
    }
}

impl<M> glue::SearchPostProcess<accessors::Filtered<'_>, M> for Translate<'_, M>
where
    M: Id,
{
    type Error = ANNError;

    fn post_process<I, B>(
        &self,
        accessor: &mut accessors::Filtered<'_>,
        candidates: I,
        output: &mut B,
    ) -> impl std::future::Future<Output = ANNResult<usize>> + Send
    where
        I: Iterator<Item = Neighbor<u32>> + Send,
        B: SearchOutputBuffer<M> + Send + ?Sized,
    {
        ready(|| accessor.search().post_process_as(*self, candidates, output))
    }
}

/// A scoped view of a [`Provider`] with a pre-evaluate query and filter predicate.
///
/// See: [`Scoped::filtered`].
#[derive(Debug)]
pub struct Filtered<'a, M>
where
    M: Id,
{
    accessor: accessors::Filtered<'a>,
    translate: Translate<'a, M>,
}

impl<M> Filtered<'_, M>
where
    M: Id,
{
    /// Run [`diskann::graph::search::InlineFilterSearch`] and store the results in `output`.
    pub fn inline_filtered<O>(
        &mut self,
        params: diskann::graph::search::InlineFilterSearch,
        output: &mut O,
    ) -> ANNResult<diskann::graph::algorithms::SearchStats>
    where
        O: diskann::graph::SearchOutputBuffer<M> + Send,
    {
        let fut = <_ as diskann::graph::search::FilteredSearch>::filtered_search(
            params,
            &mut self.accessor,
            self.translate,
            output,
        );

        poll(fut)
    }

    /// Run [`diskann::graph::search::MultihopFilterSearch`] and store the results in `output`.
    pub fn multihop_filtered<O>(
        &mut self,
        params: diskann::graph::search::MultihopFilterSearch,
        output: &mut O,
    ) -> ANNResult<diskann::graph::algorithms::SearchStats>
    where
        O: diskann::graph::SearchOutputBuffer<M> + Send,
    {
        let fut = <_ as diskann::graph::search::FilteredSearch>::filtered_search(
            params,
            &mut self.accessor,
            self.translate,
            output,
        );

        poll(fut)
    }
}

///////////
// Write //
///////////

/// A writable view over [`Index`].
///
/// # Note
///
/// Holding a [`Write`] will prevent deleted items from being reclaimed and thus should be
/// held for as short a duration as possible.
#[derive(Debug)]
pub struct Write<'a, R, M>
where
    R: repr::Representation, // FIXME: Remove structural bounds
    M: Id,
{
    index: &'a Index<R, M>,
    guard: epoch::WriteGuard<'a>,
}

impl<'a, R, M> Write<'a, R, M>
where
    R: repr::Representation, // FIXME: Remove structural bounds
    M: Id,
{
    // FIXME: Add a proper roll-back guard.
    fn set<'q>(&self, vector: R::Vector<'q>, id: M) -> ANNResult<u32>
    where
        R: repr::Insert,
        'a: 'q,
    {
        // TODO: Proper cleanup via `Guard` or some other mechanism on the event of
        // insert failure after `set_element` returns.
        //
        // The internal `Guard` is sufficient for local rollback, but not after
        // `set_element` returns.
        let guard = repr::InsertImpl::set(&self.index.representation, vector)?;
        let internal = <_ as repr::Guard>::id(&guard);
        self.index.mapping().insert(id.clone(), internal)?;

        // Now that insert has succeeded - publish the slot. This method cannot fail, so
        // we do not need to worry about potentially unwinding the ID mapping.
        <_ as repr::Guard>::publish(guard);

        // This is a rather expensive update.
        //
        // However, counters are only active with the `integration-test` feature, which
        // is not expected to be enabled for general use.
        self.index.local_counters().set_vector(1);
        Ok(internal)
    }

    pub fn insert<'q>(&self, vector: R::Vector<'q>, id: M) -> ANNResult<()>
    where
        R: repr::Insert,
        'a: 'q,
    {
        // TODO: IT would be great to fuse setting with creation of the search accessor.
        //
        // For some quantization schemes, it could avoid compressing the vector twice.
        let internal = self.set(vector, id)?;
        let mut access = self.insert_access(vector)?;
        let fut =
            diskann::graph::algorithms::insert::insert(&mut access, internal, &self.index.graph);

        poll(fut)
    }

    // FIXME: Actually think this through.
    pub fn delete(&self, id: &M) -> ANNResult<()>
    where
        R: repr::Insert,
    {
        use diskann::graph::algorithms::inplace_delete;

        const THREE: NonZeroUsize = NonZeroUsize::new(3).unwrap();

        let id = self.index.temp_delete(id)?;

        let mut access = algorithms::InplaceDelete::new(self.prune_accessor()?);
        let mut workspace = inplace_delete::Workspace::new();
        let kind = inplace_delete::TwoHopAndOneHop {
            replacements: THREE,
        };

        let fut = workspace.two_hop_and_one_hop(&kind, &mut access, id, &self.index.graph);

        poll(fut)
    }

    /// Construct an [`algorithms::Insert`] with `vector`.
    ///
    /// This function does *not* any persistent data.
    fn insert_access<'q>(&self, vector: R::Vector<'q>) -> ANNResult<algorithms::Insert<'q>>
    where
        R: repr::Insert,
        'a: 'q,
    {
        Ok(algorithms::Insert::new(
            repr::InsertImpl::insert_search_accessor(
                &self.index.representation,
                vector,
                Some(self.guard.read_guard().share()),
                self.index.local_counters(),
            )?,
            self.prune_accessor()?,
        ))
    }

    /// Construct an [`accessors::PruneAccessor`].
    fn prune_accessor(&self) -> ANNResult<accessors::Prune<'a>>
    where
        R: repr::Insert,
    {
        repr::InsertImpl::prune_accessor(&self.index.representation, self.index.local_counters())
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
            search::{InlineFilterSearch, Knn, MultihopFilterSearch},
            test::synthetic::Grid,
        },
        neighbor::Neighbor,
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
    #[test]
    fn smoke() {
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

        let graph = diskann::graph::config::Builder::new(
            2 * (grid.dim() as usize),
            diskann::graph::config::MaxDegree::new(degree),
            10,
            (Metric::L2).into(),
        )
        .build()
        .unwrap();

        let index = Index::<_, u64>::new(config, graph).unwrap();
        assert_eq!(index.max_degree(), MaxDegree::new(degree));
        assert_eq!(index.capacity(), Capacity::new(grid.num_points(size)));

        {
            let write = index.write().unwrap();
            for (i, data) in data.rows().enumerate() {
                write.insert(data, (10 * i + 1) as u64).unwrap();
            }
        }

        // Verify that each ID round trips.
        for i in 0..data.nrows() {
            let i = (10 * i + 1) as u64;
            let internal = index.to_internal(&i).unwrap();
            assert_ne!(
                internal as u64,
                i + 1,
                "this relies on the start points being located at the beginning of the id space\
                 this is **NOT** guaranteed",
            );

            assert!(index.is_readable(internal).unwrap());
        }

        // Assert that out-of-bounds translations returns errors.
        assert!(index.to_internal(&0).is_none());

        // Searches should return something reasonable.
        let knn = Knn::new(10, None).unwrap();
        let mut neighbors = Vec::<Neighbor<u64>>::new();
        index
            .read()
            .unwrap()
            .search(&[0.0, 0.0], knn, &mut neighbors)
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
        let read = index.read().unwrap();
        let mut filtered = read
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
            filtered.accessor.search().mimic_destructive_post_process();

            // Try again, this time with mulit-hop filter.
            neighbors.clear();
            let mfs =
                MultihopFilterSearch::new(knn, NonZeroUsize::new(index.max_degree().value() / 2));
            filtered.multihop_filtered(mfs, &mut neighbors).unwrap();

            // We should not observe entries with odd 10s digits.
            assert_eq!(neighbors[0].as_tuple(), (1, 0.0));
            assert_eq!(neighbors[1].as_tuple(), (61, 2.0));
            assert_eq!(neighbors[2].as_tuple(), (21, 4.0)); // This can be swapped with 3
            assert_eq!(neighbors[3].as_tuple(), (101, 4.0));
        }

        // If we run inplace delete on point 61, it longer be present.
        let internal = index.to_internal(&61).unwrap();
        assert!(index.is_readable(internal).unwrap());
        index.write().unwrap().delete(&61).unwrap();

        assert!(index.to_internal(&61).is_none());
        assert!(!index.is_readable(internal).unwrap());

        // We can't delete the same thing twice.
        assert!(index.write().unwrap().delete(&61).is_err());

        // Rerun search - the point 61 should now be gone.
        let mut neighbors = Vec::<Neighbor<u64>>::new();
        read.search(&[0.0, 0.0], knn, &mut neighbors).unwrap();

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
                .write()
                .unwrap()
                .insert(&[10.0, 10.0], 1)
                .unwrap_err()
                .to_string(),
            "could not allocate a new slot",
        );

        assert!(index.reclaim().is_none());
        drop(filtered);
        drop(read);

        // Need to advance one epoch before the next implicit epoch advancement will succeed.
        assert_eq!(index.reclaim(), Some(0));
        let write = index.write().unwrap();

        // We can't insert an existing ID.
        assert_contains!(
            write.insert(&[10.0, 10.0], 1).unwrap_err().to_string(),
            "external id is already mapped",
        );

        // If we insert a new ID but the query vector is too long - make sure we leave the
        // provider untouched.
        assert_contains!(
            write
                .insert(&[10.0, 10.0, 10.0], 2)
                .unwrap_err()
                .to_string(),
            "expected slice of length 2",
        );

        // Check that we can reinsert the same point with a different ID and have it be
        // returned from search.
        write.insert(&[1.0, 1.0], 62).unwrap();

        // We can't insert an ID - but this time it's because we don't have any more internal
        // slots.
        assert!(write.insert(&[0.0, 0.0], 62).is_err());

        let read = index.read().unwrap();

        // Rerun search - the point 62 should be present.
        let mut neighbors = Vec::<Neighbor<u64>>::new();
        read.search(&[0.0, 0.0], knn, &mut neighbors).unwrap();

        assert_eq!(neighbors[0].as_tuple(), (1, 0.0));

        // These two can be swapped. Accept either ordering.
        if *neighbors[1].id() == 11 {
            assert_eq!(neighbors[1].as_tuple(), (11, 1.0));
            assert_eq!(neighbors[2].as_tuple(), (51, 1.0));
        } else {
            assert_eq!(neighbors[1].as_tuple(), (51, 1.0));
            assert_eq!(neighbors[2].as_tuple(), (11, 1.0));
        }

        assert_eq!(neighbors[3].as_tuple(), (62, 2.0));

        // Finally - test that we can create a crazy predicate for filtering that deletes
        // all the external IDs as they are evaluated.
        let _ = read.filtered(&[0.0, 0.0], |i| {
            write.delete(i).unwrap();
            true
        });
    }
}
