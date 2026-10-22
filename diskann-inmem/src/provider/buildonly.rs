/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::{ANNError, ANNResult, graph::glue};

use crate::{
    counters::{Counters, LocalCounters},
    num::{IdLimit, MaxDegree},
    repr,
};

#[derive(Debug)]
pub struct Provider<R> {
    representation: R,
    counters: Counters,
}

impl<R> Provider<R> {
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

impl<R> Provider<R>
where
    R: repr::Representation,
{
    /// Construct a new [`Provider`].
    pub fn new<C>(config: C) -> ANNResult<Self>
    where
        C: repr::RepresentationConfig<Representation = R>,
    {
        let representation = <_ as repr::RepresentationConfig>::build(config)?;

        Ok(Self {
            representation,
            counters: Counters::new(),
        })
    }

    /// Return the maximum number of neighbors that can be stored in the provider's graph.
    pub fn max_degree(&self) -> MaxDegree {
        self.representation.max_degree()
    }

    pub fn id_limit(&self) -> IdLimit {
        self.representation.id_limit()
    }

    /// Return a [`crate::neighbors::Snapshot`] of the current adjacency lists.
    ///
    /// Since this receives by `&muT self`, this effectively disables all indexing operations
    /// while the snapshot is held.
    pub fn neighbor_snapshot(&mut self) -> crate::neighbors::Snapshot<'_>
    where
        R: repr::NeighborSnapshot,
    {
        <_ as repr::NeighborSnapshot>::neighbor_snapshot(&mut self.representation)
    }
}

impl<R> diskann::provider::DataProvider for Provider<R>
where
    R: Send + Sync + 'static,
{
    type Context = super::Context;
    type InternalId = u32;
    type ExternalId = u32;
    type Error = ANNError;
    type Guard = diskann::provider::NoopGuard<u32>;

    fn to_internal_id(&self, _context: &Self::Context, gid: &u32) -> ANNResult<u32> {
        Ok(*gid)
    }

    /// Translate an internal id to its corresponding external id.
    fn to_external_id(&self, _context: &Self::Context, id: u32) -> ANNResult<u32> {
        Ok(id)
    }
}

impl<T, R> diskann::provider::SetElement<T> for Provider<R>
where
    R: repr::SetAt<T>,
{
    type SetError = ANNError;

    fn set_element(
        &self,
        _context: &Self::Context,
        id: &u32,
        element: T,
    ) -> impl std::future::Future<Output = Result<Self::Guard, Self::SetError>> + Send {
        let work = move || {
            // TODO: Proper cleanup via `Guard` or some other mechanism on the event of
            // insert failure after `set_element` returns.
            //
            // The internal `Guard` is sufficient for local rollback, but not after
            // `set_element` returns.
            let guard = <R as repr::SetAt<T>>::set_at(&self.representation, *id, element)?;
            let internal = <_ as repr::Guard>::id(&guard);

            // Did someine implement `SetAt` incorrectly?
            if internal != *id {
                return Err(ANNError::message(
                    "SetAt expected to respect the requested internal ID",
                ));
            }

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

        super::ready(work)
    }
}

impl<'a, R> glue::SearchStrategy<'a, Provider<R>, R::Query<'a>> for super::Strategy
where
    R: repr::Search,
{
    type SearchAccessor = super::SearchAccessor<'a>;
    type SearchAccessorError = ANNError;

    fn search_accessor(
        &'a self,
        provider: &'a Provider<R>,
        _context: &'a super::Context,
        query: R::Query<'a>,
    ) -> ANNResult<super::SearchAccessor<'a>> {
        <R as repr::Search>::search_accessor(
            &provider.representation,
            query,
            provider,
            provider.local_counters(),
        )
    }
}

impl<'a, R> glue::DefaultPostProcessor<'a, Provider<R>, R::Query<'a>> for super::Strategy
where
    R: repr::Search,
{
    diskann::default_post_processor!(glue::CopyIds);
}

impl<R> glue::PruneStrategy<Provider<R>> for super::Strategy
where
    R: repr::Insert,
{
    type PruneAccessor<'a> = super::PruneAccessor<'a>;
    type PruneAccessorError = ANNError;

    fn prune_accessor<'a>(
        &self,
        provider: &'a Provider<R>,
        _context: &'a super::Context,
        _capacity: usize,
    ) -> ANNResult<super::PruneAccessor<'a>> {
        <R as repr::Insert>::prune_accessor(&provider.representation, provider.local_counters())
    }
}

impl<'a, R> glue::InsertStrategy<'a, Provider<R>, R::Query<'a>> for super::Strategy
where
    R: repr::Insert,
{
    type SearchAccessor = super::SearchAccessor<'a>;
    type SearchAccessorError = ANNError;
    type PruneStrategy = Self;

    fn insert_search_accessor(
        &'a self,
        provider: &'a Provider<R>,
        _context: &'a super::Context,
        vector: R::Query<'a>,
    ) -> Result<Self::SearchAccessor, Self::SearchAccessorError> {
        <R as repr::Insert>::insert_search_accessor(
            &provider.representation,
            vector,
            provider,
            provider.local_counters(),
        )
    }

    fn prune_strategy(&self) -> Self::PruneStrategy {
        *self
    }
}
