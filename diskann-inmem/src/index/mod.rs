/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::sync::Arc;

use diskann_utils::views::rowmajor;
use diskann::{
    ANNResult,
    graph::{DiskANNIndex, SearchOutputBuffer},
};

use crate::{
    bitvector::BitVector,
    Context, Provider, Strategy,
    num::{Capacity, MaxDegree},
    poll,
    provider::{self, Id},
    repr,
};

///////////
// Index //
///////////

#[derive(Debug)]
pub struct Index<R, M>
where
    R: repr::Representation, // FIXME: Remove structural bounds
    M: Id,                   // FIXME: Remove structural bounds
{
    index: Arc<DiskANNIndex<Provider<R, M>>>,
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
        let provider = Provider::new(config)?;
        let index = Arc::new(DiskANNIndex::new(graph, provider, None));
        Ok(Self { index })
    }

    pub fn temp_new(provider: Provider<R, M>, graph: diskann::graph::Config) -> Self {
        Self {
            index: Arc::new(DiskANNIndex::new(graph, provider, None)),
        }
    }

    pub fn read(&self) -> ANNResult<Read<'_, R, M>> {
        let guard = repr::Representation::read_guard(self.index.provider().repr())?;
        Ok(Read {
            index: &self.index,
            guard,
        })
    }

    pub fn write(&self) -> ANNResult<Write<'_, R, M>> {
        let guard = repr::Representation::write_guard(self.index.provider().repr())?;
        Ok(Write {
            index: &self.index,
            guard,
        })
    }

    /// Return the [`Capacity`] of the index.
    pub fn capacity(&self) -> Capacity {
        self.index.provider().capacity()
    }

    /// Return the maximum number of neighbors that can be stored in the provider's graph.
    pub fn max_degree(&self) -> MaxDegree {
        self.index.provider().max_degree()
    }

    /// Return a snapshot of the current event counters.
    #[cfg(feature = "integration-test")]
    pub fn counters(&self) -> crate::integration::counters::CounterSnapshot {
        self.index.provider().counters()
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
    R: repr::Representation, // FIXME: Remove structural bounds
    M: Id,
{
    index: &'a DiskANNIndex<Provider<R, M>>,
    guard: repr::ReadGuard<'a>,
}

impl<R, M> Read<'_, R, M>
where
    R: repr::Representation, // FIXME: Remove structural bounds
    M: Id,
{
    // FIXME: Remove unnecessary lifetime bounds
    pub fn search<'a, O>(
        &'a self,
        query: R::Query<'a>,
        params: diskann::graph::search::Knn,
        output: &mut O,
    ) -> ANNResult<diskann::graph::index::SearchStats>
    where
        R: repr::Search,
        O: SearchOutputBuffer<M> + Send + ?Sized,
    {
        // FIXME: Push the `ReadGuard` into the accessor.
        let fut = self
            .index
            .search(params, &Strategy, &Context, query, output);

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
    ) -> ANNResult<Filtered<'x, R, M>>
    where
        R: repr::Search,
        F: FnMut(&M) -> bool,
    {
        let provider = self.index.provider();
        let search = <R as repr::Search>::search_accessor(
            provider.repr(),
            query,
            provider,
            Some(self.guard.share()),
            provider.local_counters(),
        )?;

        let filter = self.evaluate(predicate);
        Ok(Filtered {
            read: self.share(),
            accessor: provider::FilteredAccessor::new(search, filter),
        })
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
            provider.repr(),
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
            if let Some(external) = provider.mapping().to_external(internal)
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

/// A scoped view of a [`Provider`] with a pre-evaluate query and filter predicate.
///
/// See: [`Scoped::filtered`].
#[derive(Debug)]
pub struct Filtered<'a, R, M>
where
    R: repr::Representation, // FIXME: structural bounds
    M: Id,
{
    // FIXME: `accessor` has everything we need. We don't need `index` except that `diskann`
    // requires it!
    read: Read<'a, R, M>,
    accessor: provider::FilteredAccessor<'a>,
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
        let strategy = provider::Smuggle::new(&mut self.accessor);
        let fut = self.read.index.search(params, &strategy, &Context, (), output);
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
        let strategy = provider::Smuggle::new(&mut self.accessor);
        let fut = self.read.index.search(params, &strategy, &Context, (), output);
        crate::poll(fut)
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
    index: &'a Arc<DiskANNIndex<Provider<R, M>>>,
    guard: repr::WriteGuard<'a>,
}

impl<R, M> Write<'_, R, M>
where
    R: repr::Representation, // FIXME: Remove structural bounds
    M: Id,
{
    // FIXME: remove unnecessary lifetime bounds
    pub fn insert<'a>(&'a self, vector: R::Query<'a>, id: M) -> ANNResult<()>
    where
        R: repr::Insert,
    {
        let fut = self.index.insert(&Strategy, &Context, &id, vector);

        poll(fut)
    }

    // FIXME: allow this to just be a `Ref`.
    // FIXME: Remove `tokio::runtime::Handle`.
    pub fn multi_insert<T>(
        &self,
        data: Arc<rowmajor::Owned<T>>,
        ids: Arc<[M]>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = ANNResult<()>> + '_>>
    where
        R: for<'a> repr::Insert<Query<'a> = &'a [T]>,
        T: Send + Sync + 'static, // FIXME: simplify bounds
    {
        Box::pin(async move {
            self.index.clone().multi_insert(Strategy, &Context, data, ids).await
        })
    }
}
