/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::{
    ANNError, ANNResult,
    graph::{SearchOutputBuffer, glue},
    neighbor::Neighbor,
};

use crate::{
    counters::{Counters, LocalCounters},
    epoch,
    index::{accessors, algorithms},
    num::{Capacity, IdLimit, MaxDegree},
    poll::{poll, ready},
    repr,
};

#[derive(Debug)]
pub struct Index<R> {
    // Data representation and storage
    representation: R,
    // The graph configuration.
    graph: diskann::graph::Config,
    // `Counters` is only non-trivial under the `integration-test` feature flag. Otherwise,
    // all counter related operations are no-ops.
    counters: Counters,
}

impl<R> Index<R>
where
    R: repr::Representation, // FIXME: Remove structural bounds
{
    pub fn new<C>(config: C, graph: diskann::graph::Config) -> ANNResult<Self>
    where
        C: repr::RepresentationConfig<Representation = R>,
    {
        let representation = <_ as repr::RepresentationConfig>::build(config)?;
        Ok(Self {
            representation,
            graph,
            counters: Counters::new(),
        })
    }

    /// Return the [`Capacity`] of the index.
    pub fn capacity(&self) -> Capacity {
        self.representation.capacity()
    }

    /// Return the [`IdLimit`] of the index.
    pub fn id_limit(&self) -> IdLimit {
        self.representation.id_limit()
    }

    /// Return the maximum number of neighbors that can be stored in the provider's graph.
    pub fn max_degree(&self) -> MaxDegree {
        self.representation.max_degree()
    }

    /// Return a [`repr::NeighborSnapshot`] of the current adjacency lists.
    ///
    /// Since this receives by `&mut self`, this effectively disables all indexing operations
    /// while the snapshot is held.
    pub fn inspect_neighbors(&mut self) -> repr::NeighborSnapshot<'_> {
        repr::private::Representation::inspect_neighbors(&mut self.representation)
    }

    pub fn read(&self) -> ANNResult<Read<'_, R>> {
        let guard = repr::private::Representation::read_guard(&self.representation)?;
        Ok(Read { index: self, guard })
    }

    pub fn write(&self) -> ANNResult<Write<'_, R>> {
        let guard = repr::private::Representation::write_guard(&self.representation)?;
        Ok(Write {
            index: &self,
            guard,
        })
    }
}

impl<R> Index<R> {
    /// Return a snapshot of the current event counters.
    #[cfg(feature = "integration-test")]
    pub fn counters(&self) -> crate::integration::counters::CounterSnapshot {
        self.counters.snapshot()
    }

    /// A local set of counters that update the provider-wide counters in bulk.
    fn local_counters(&self) -> LocalCounters<'_> {
        self.counters.local()
    }
}

/// A read-only view over [`Index`].
///
/// # Note
///
/// Holding a [`Read`] will prevent deleted items from being reclaimed and thus should be
/// held for as short a duration as possible.
#[derive(Debug)]
pub struct Read<'a, R> {
    index: &'a Index<R>,
    guard: epoch::ReadGuard<'a>,
}

impl<R> Read<'_, R> {
    // FIXME: Remove uneecessary lifetime bounds
    pub fn search<'a, O>(
        &'a self,
        query: R::Query<'a>,
        params: diskann::graph::search::Knn,
        output: &mut O,
    ) -> ANNResult<diskann::graph::algorithms::SearchStats>
    where
        R: repr::Search,
        O: SearchOutputBuffer<u32> + Send + ?Sized,
    {
        let mut accessor = repr::private::Search::search_accessor(
            &self.index.representation,
            query,
            Some(self.guard.share()),
            self.index.local_counters(),
        )?;

        let fut = <_ as diskann::graph::search::Search<u32>>::search(
            params,
            &mut accessor,
            PostProcess,
            output,
        );

        poll(fut)
    }
}

#[derive(Debug, Clone, Copy)]
struct PostProcess;

impl glue::SearchPostProcess<accessors::Search<'_>, u32> for PostProcess {
    type Error = ANNError;

    fn post_process<I, B>(
        &self,
        accessor: &mut accessors::Search<'_>,
        candidates: I,
        output: &mut B,
    ) -> impl std::future::Future<Output = ANNResult<usize>> + Send
    where
        I: Iterator<Item = Neighbor<u32>> + Send,
        B: SearchOutputBuffer<u32> + Send + ?Sized,
    {
        let work = || {
            let mut count = 0;
            let push = |neighbor: Neighbor<u32>| -> bool {
                if output.push(neighbor).is_available() {
                    count += 1;
                    true
                } else {
                    false
                }
            };

            accessor.post_process_into(candidates, push)?;
            Ok(count)
        };

        ready(work)
    }
}

/// A writable view over [`Index`].
///
/// # Note
///
/// Holding a [`Write`] will prevent deleted items from being reclaimed and thus should be
/// held for as short a duration as possible.
#[derive(Debug)]
pub struct Write<'a, R>
where
    R: repr::Representation, // FIXME: Remove structural bounds
{
    index: &'a Index<R>,
    guard: epoch::WriteGuard<'a>,
}

impl<'a, R> Write<'a, R>
where
    R: repr::Representation, // FIXME: Remove structural bounds
{
    // FIXME: Add a proper roll-back guard.
    fn set<'q>(&self, vector: R::Vector<'q>, id: u32) -> ANNResult<u32>
    where
        R: repr::Insert,
        'a: 'q,
    {
        // TODO: Proper cleanup via `Guard` or some other mechanism on the event of
        // insert failure after `set_element` returns.
        //
        // The internal `Guard` is sufficient for local rollback, but not after
        // `set_element` returns.
        let guard = repr::private::Insert::set_at(&self.index.representation, vector, id)?;
        let internal = <_ as repr::Guard>::id(&guard);

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

    pub fn insert<'q>(&self, vector: R::Vector<'q>, id: u32) -> ANNResult<()>
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

    /// Construct an [`algorithms::Insert`] with `vector`.
    ///
    /// This function does *not* any persistent data.
    fn insert_access<'q>(&self, vector: R::Vector<'q>) -> ANNResult<algorithms::Insert<'q>>
    where
        R: repr::Insert,
        'a: 'q,
    {
        Ok(algorithms::Insert::new(
            repr::private::Insert::insert_search_accessor(
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
        repr::private::Insert::prune_accessor(
            &self.index.representation,
            self.index.local_counters(),
        )
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    use diskann::{
        graph::{search::Knn, test::synthetic::Grid},
        neighbor::Neighbor,
    };
    use diskann_utils::{
        assert_contains,
        views::rowmajor::{self, Matrix},
    };
    use diskann_vector::distance::Metric;

    use crate::num::Range;

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
    #[test]
    fn smoke() {
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

        let graph = diskann::graph::config::Builder::new(
            2 * (grid.dim() as usize),
            diskann::graph::config::MaxDegree::new(degree),
            10,
            (Metric::L2).into(),
        )
        .build()
        .unwrap();

        let index = Index::<_>::new(config, graph).unwrap();

        assert_eq!(index.max_degree(), MaxDegree::new(degree));
        assert_eq!(index.capacity(), Capacity::new(25));

        let write = index.write().unwrap();

        // We cannot insert at index 25 because that is where we requested the start point
        // be located.
        let err = write.insert(data.row(0), start_id).unwrap_err();

        assert_contains!(err.to_string(), "could not allocate a new slot");

        // If we insert a new ID but the vector is too long - the slot should be unclaimed.
        let err = write.insert(&[10.0, 10.0, 10.0], 0).unwrap_err();
        assert_contains!(err.to_string(), "expected slice of length 2");

        for (i, data) in data.rows().enumerate() {
            let i = u32::try_from(i).unwrap();
            write.insert(data, i).unwrap();
        }

        // Searches should return something reasonable.
        let knn = Knn::new(10, None).unwrap();
        let mut neighbors = Vec::<Neighbor<u32>>::new();
        index
            .read()
            .unwrap()
            .search(&[0.0, 0.0], knn, &mut neighbors)
            .unwrap();

        assert_eq!(neighbors[0].as_tuple(), (0, 0.0));
        assert_eq!(neighbors[1].as_tuple(), (1, 1.0)); // this can be swapped with 5
        assert_eq!(neighbors[2].as_tuple(), (5, 1.0));
        assert_eq!(neighbors[3].as_tuple(), (6, 2.0));

        // We can't insert an existing ID.
        assert!(write.insert(&[10.0, 10.0], 1).is_err());
    }
}
