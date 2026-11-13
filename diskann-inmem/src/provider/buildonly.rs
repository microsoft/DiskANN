/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! An index provider for performing static builds.
//!
//! No internal/external ID translation is provided. Users set the internal ID directly and
//! each internal ID may be set only once. This struct does not provide the ability to
//! reclaim IDs.
//!
//! The intention is enable a static index build that is finished with
//! [`Provider::inspect_neighbors`] to read out the constructed graph.

use diskann::{ANNError, ANNResult, graph::glue};
use diskann_utils::lazy_format;

use crate::{
    counters::{Counters, LocalCounters},
    num::{Capacity, IdLimit, MaxDegree},
    repr,
};

/// Support for static index construction.
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

    /// Return the [`Capacity`] for the provider.
    pub fn capacity(&self) -> Capacity {
        self.representation.capacity()
    }

    /// Return the [`IdLimit`] for the provider.
    pub fn id_limit(&self) -> IdLimit {
        self.representation.id_limit()
    }

    /// Return a [`repr::NeighborSnapshot`] of the current adjacency lists.
    ///
    /// Since this receives by `&mut self`, this effectively disables all indexing operations
    /// while the snapshot is held.
    pub fn inspect_neighbors(&mut self) -> repr::NeighborSnapshot<'_>
    where
        R: repr::InspectNeighbors,
    {
        <_ as repr::InspectNeighbors>::inspect_neighbors(&mut self.representation)
    }
}

impl<R> diskann::provider::DataProvider for Provider<R>
where
    R: repr::Representation,
{
    type Context = super::Context;
    type InternalId = u32;
    type ExternalId = u32;
    type Error = ANNError;
    type Guard = diskann::provider::NoopGuard<u32>;

    fn to_internal_id(&self, _context: &Self::Context, gid: &u32) -> ANNResult<u32> {
        let gid = *gid;
        let limit = self.id_limit();
        if gid < limit.value() {
            Ok(gid)
        } else {
            Err(ANNError::message(lazy_format!(
                move,
                "id {gid} exceeds {limit}"
            )))
        }
    }
}

// NOTE: [`repr::SetAt`] is crate-private intentionally.
//
// Users can call this method, but don't have a great way to constrain it with just
// [`repr::SetAt`].
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

            // Did someone implement `SetAt` incorrectly?
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

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    use diskann::{
        graph::{DiskANNIndex, search::Knn, test::synthetic::Grid},
        neighbor::Neighbor,
        provider::DataProvider,
    };
    use diskann_utils::{
        assert_contains,
        views::rowmajor::{self, Matrix},
    };
    use diskann_vector::distance::Metric;

    use crate::{Context, Strategy, num::Range};

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
    ///                   25
    ///    4  9 14 19 24
    ///    3  8 13 18 23
    ///    2  7 12 17 22
    ///    1  6 11 16 21
    ///    0  5 10 15 20
    ///
    #[tokio::test]
    async fn smoke() {
        let grid = Grid::Two;
        let size = 5;
        let data = grid.data(size);
        let start = grid.start_point(size);
        let degree = 6;
        let start_id = 25;

        let config = repr::full::Config::new(
            Capacity::new(grid.num_points(size)),
            MaxDegree::new(degree),
            Metric::L2,
            rowmajor::Owned::row_vector(start.into()),
        )
        .unwrap()
        .locate_start_point_ids_at(Range::start_plus(start_id, 1).unwrap())
        .unwrap();

        let provider = Provider::<_>::new(config).unwrap();

        assert_eq!(provider.max_degree(), MaxDegree::new(degree));
        assert_eq!(provider.capacity(), Capacity::new(25));
        assert_eq!(provider.id_limit(), IdLimit::new(start_id + 1));

        let config = diskann::graph::config::Builder::new(
            2 * (grid.dim() as usize),
            diskann::graph::config::MaxDegree::new(provider.max_degree().value()),
            10,
            (Metric::L2).into(),
        )
        .build()
        .unwrap();

        let index = DiskANNIndex::new(config, provider, None);

        // We cannot insert at index 25 because that is where we requested the start point
        // be located.
        let err = index
            .insert(&Strategy, &Context, &start_id, data.row(0))
            .await
            .unwrap_err();

        assert_contains!(err.to_string(), "could not allocate a new slot");

        // If we insert a new ID but the vector is too long - the slot should be unclaimed.
        let err = index
            .insert(&Strategy, &Context, &0, &[10.0, 10.0, 10.0])
            .await
            .unwrap_err();
        assert_contains!(err.to_string(), "expected slice of length 2");

        for (i, data) in data.rows().enumerate() {
            let i = u32::try_from(i).unwrap();
            index.insert(&Strategy, &Context, &i, data).await.unwrap();
        }

        // Verify that each ID round trips.
        for i in 0..data.nrows() {
            let i = u32::try_from(i).unwrap();
            let internal = index.provider().to_internal_id(&Context, &i).unwrap();
            assert_eq!(i, internal);
        }

        // Assert that out-of-bounds translations returns errors.
        let err = index
            .provider()
            .to_internal_id(&Context, &(start_id + 1))
            .unwrap_err()
            .to_string();
        assert_contains!(err, "exceeds IdLimit(26)");

        // Searches should return something reasonable.
        let knn = Knn::new(10, None).unwrap();
        let mut neighbors = Vec::<Neighbor<u32>>::new();
        index
            .search(knn, &Strategy, &Context, &[0.0, 0.0], &mut neighbors)
            .await
            .unwrap();

        assert_eq!(neighbors[0].as_tuple(), (0, 0.0));
        assert_eq!(neighbors[1].as_tuple(), (1, 1.0)); // this can be swapped with 5
        assert_eq!(neighbors[2].as_tuple(), (5, 1.0));
        assert_eq!(neighbors[3].as_tuple(), (6, 2.0));

        // We can't insert an existing ID.
        assert!(
            index
                .insert(&Strategy, &Context, &1, &[10.0, 10.0])
                .await
                .is_err()
        );
    }
}
