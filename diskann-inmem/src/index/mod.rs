/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::{
    ANNError, ANNResult,
    graph::{
        SearchOutputBuffer,
        glue,
    },
    neighbor::Neighbor,
};
use diskann_utils::views::rowmajor;

use crate::{
    bitvector::BitVector,
    num::{Capacity, MaxDegree, IdLimit},
    poll::{poll, ready},
    repr,
    ids::IdMap,
    counters::{Counters, LocalCounters},
};

//------------//
// Submodules //
//------------//

pub(crate) mod accessors;
mod algorithms;

///////////
// Index //
///////////

/// Aggregate trait for the external ID type of [`Provider`].
pub trait Id: Send + Sync + std::hash::Hash + Eq + Clone + 'static {}

impl<T> Id for T where T: Send + Sync + std::hash::Hash + Eq + Clone + 'static {}

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
        let guard = repr::Representation::read_guard(&self.representation)?;
        Ok(Read {
            index: self,
            guard,
        })
    }

    pub fn write(&self) -> ANNResult<Write<'_, R, M>> {
        let guard = repr::Representation::write_guard(&self.representation)?;
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

    fn id_limit(&self) -> IdLimit {
        self.representation.id_limit()
    }

    fn mapping(&self) -> &IdMap<M> {
        &self.mapping
    }

    fn temp_delete(&self, id: &M) -> ANNResult<()> {
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

        // An early return here will cause `entry` to be dropped, which will *not* cause
        // the delete to commit.
        <R as repr::Representation>::retire(&self.representation, entry.internal())?;

        // Successfully retired the internal slot. We can safely release the ID mapping.
        entry.delete();

        Ok(())
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
    ///
    /// FIXME: Make more private.
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
    guard: repr::ReadGuard<'a>,
}

impl<R, M> Read<'_, R, M>
where
    M: Id,
{
    // FIXME: Remove unnecessary lifetime bounds
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
        let mut accessor = repr::Search::search_accessor(
            &self.index.representation,
            query,
            &(),
            Some(self.guard.share()),
            self.index.local_counters(),
        )?;

        let fut = <_ as diskann::graph::search::Search<_>>::search(
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
        let search = <R as repr::Search>::search_accessor(
            &self.index.representation,
            query,
            self.index,
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
        let mut bitvec = BitVector::new(<R as repr::Representation>::id_limit(&self.index.representation));

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

// impl<M> Filtered<'_, M>
// where
//     M: Id,
// {
//     /// Run [`diskann::graph::search::InlineFilterSearch`] and store the results in `output`.
//     pub fn inline_filtered<O>(
//         &mut self,
//         params: diskann::graph::search::InlineFilterSearch,
//         output: &mut O,
//     ) -> ANNResult<diskann::graph::index::SearchStats>
//     where
//         O: diskann::graph::SearchOutputBuffer<M> + Send,
//     {
//         let
//
//         let strategy = provider::Smuggle::new(&mut self.accessor);
//         let fut = self
//             .read
//             .index
//             .search(params, &strategy, &Context, (), output);
//         crate::poll(fut)
//     }
//
//     /// Run [`diskann::graph::search::MultihopFilterSearch`] and store the results in `output`.
//     pub fn multihop_filtered<O>(
//         &mut self,
//         params: diskann::graph::search::MultihopFilterSearch,
//         output: &mut O,
//     ) -> ANNResult<diskann::graph::index::SearchStats>
//     where
//         O: diskann::graph::SearchOutputBuffer<M> + Send,
//     {
//         let strategy = provider::Smuggle::new(&mut self.accessor);
//         let fut = self
//             .read
//             .index
//             .search(params, &strategy, &Context, (), output);
//         crate::poll(fut)
//     }
// }

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
    guard: repr::WriteGuard<'a>,
}

impl<R, M> Write<'_, R, M>
where
    R: repr::Representation, // FIXME: Remove structural bounds
    M: Id,
{
    // FIXME: Add a proper roll-back guard.
    fn set(&self, vector: R::Query<'_>, id: M) -> ANNResult<u32>
    where
        R: repr::Insert,
    {
        // TODO: Proper cleanup via `Guard` or some other mechanism on the event of
        // insert failure after `set_element` returns.
        //
        // The internal `Guard` is sufficient for local rollback, but not after
        // `set_element` returns.
        let guard = <R as repr::Set<R::Query<'_>>>::set(&self.index.representation, vector)?;
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

    pub fn insert<'a>(&'a self, vector: R::Query<'a>, id: M) -> ANNResult<()>
    where
        R: repr::Insert,
    {
        // TODO: IT would be great to fuse setting with creation of the search accessor.
        //
        // For some quantization schemes, it could avoid compressing the vector twice.
        let internal = self.set(vector, id)?;

        let mut access = algorithms::Insert::new(
            repr::Insert::insert_search_accessor(
                &self.index.representation,
                vector,
                &(),
                Some(self.guard.as_read_guard()),
                self.index.local_counters(),
            )?,
            repr::Insert::prune_accessor(&self.index.representation, self.index.local_counters())?,
        );
        let fut = diskann::graph::algorithms::insert::insert(
            &mut access,
            internal,
            &self.index.graph,
        );

        poll(fut)
    }

    // // FIXME: allow this to just be a `Ref`.
    // // FIXME: Remove `tokio::runtime::Handle`.
    // pub fn multi_insert<T>(
    //     &self,
    //     data: Arc<rowmajor::Owned<T>>,
    //     ids: Arc<[M]>,
    // ) -> std::pin::Pin<Box<dyn std::future::Future<Output = ANNResult<()>> + '_>>
    // where
    //     R: for<'a> repr::Insert<Query<'a> = &'a [T]>,
    //     T: Send + Sync + 'static, // FIXME: simplify bounds
    // {
    //     Box::pin(async move {
    //         self.index
    //             .clone()
    //             .multi_insert(Strategy, &Context, data, ids)
    //             .await
    //     })
    // }
}
