/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::{
    ANNError, ANNResult,
    graph::{
        AdjacencyList, SearchOutputBuffer,
        adjacencylist::Empty,
        glue::{self, HybridPredicate},
        workingset,
    },
    neighbor::Neighbor,
};

use crate::{
    bitvector::BitVector,
    counters::LocalCounters,
    neighbors::Neighbors,
    num::{IdLimit, Range},
    poll::ready,
    repr,
};

////////////
// Search //
////////////

/// A [`glue::SearchAccessor`] for [`Provider`].
///
/// This type intentionally avoids generic parameters and instead compiles optimized
/// `expand_beam` kernels that get reused. The idea is to generate an efficient graph search
/// kernel once and reuse it to balance compile times and performance.
#[derive(Debug)]
pub(crate) struct Search<'a> {
    neighbors: &'a Neighbors,
    ids: AdjacencyList<u32>,
    expand_beam: Box<dyn repr::ExpandBeam + 'a>,
    id_limit: IdLimit,
    buffer: Vec<Neighbor<u32>>,
    post_process: Option<Box<dyn repr::PostProcess + 'a>>,
    start_points: Range,
    counters: LocalCounters<'a>,
}

impl<'a> Search<'a> {
    pub(crate) fn new(
        neighbors: &'a Neighbors,
        expand_beam: Box<dyn repr::ExpandBeam + 'a>,
        post_process: Option<Box<dyn repr::PostProcess + 'a>>,
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

    pub(super) fn post_process_as<M, I, B>(
        &mut self,
        translate: crate::index::Translate<'_, M>,
        candidates: I,
        output: &mut B,
    ) -> ANNResult<usize>
    where
        M: crate::index::Id,
        I: IntoIterator<Item = Neighbor<u32>>,
        B: SearchOutputBuffer<M> + ?Sized,
    {
        let mapping = translate.0;
        let mut count = 0;
        let push = |neighbor: Neighbor<u32>| -> bool {
            if let Some(ext) = mapping.to_external(*neighbor.id()) {
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

        self.post_process_into(candidates, push)?;
        Ok(count)
    }

    pub(super) fn post_process_into<I, F>(&mut self, candidates: I, mut output: F) -> ANNResult<()>
    where
        I: IntoIterator<Item = Neighbor<u32>>,
        F: FnMut(Neighbor<u32>) -> bool,
    {
        if let Some(post_process) = &mut self.post_process {
            self.buffer.clear();
            Extend::extend(&mut self.buffer, candidates);
            post_process.post_process(&mut self.buffer)?;

            for neighbor in self.buffer.iter() {
                if !output(*neighbor) {
                    break;
                }
            }
        } else {
            for neighbor in candidates {
                if !output(neighbor) {
                    break;
                }
            }
        }
        Ok(())
    }

    // Resize the internal buffer if needed.
    #[cold]
    fn resize_buffer(&mut self) {
        self.buffer
            .resize_with(self.neighbors.max_degree().value(), Default::default)
    }

    #[cfg(test)]
    pub(super) fn mimic_destructive_post_process(&mut self) {
        self.buffer.clear()
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

impl diskann::HasId for Search<'_> {
    type Id = u32;
}

impl glue::SearchAccessor for Search<'_> {
    fn starting_points(
        &mut self,
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

//-----------------//
// Filtered Search //
//-----------------//

/// The actual [`glue::FilteredAccessor`] for filtered search.
///
/// See: [`Filtered`].
#[derive(Debug)]
pub(super) struct Filtered<'a> {
    search: Search<'a>,
    filter: BitVector,
}

impl<'a> Filtered<'a> {
    pub(super) fn new(search: Search<'a>, filter: BitVector) -> Self {
        Self { search, filter }
    }

    pub(super) fn search(&mut self) -> &mut Search<'a> {
        &mut self.search
    }
}

impl diskann::HasId for Filtered<'_> {
    type Id = u32;
}

impl glue::FilteredAccessor for Filtered<'_> {
    fn num_starting_points(
        &mut self,
    ) -> impl std::future::Future<Output = ANNResult<usize>> + Send {
        std::future::ready(Ok(self.search.start_points.len()))
    }

    fn start_point_distances<F>(
        &mut self,
        mut f: F,
    ) -> impl std::future::Future<Output = ANNResult<()>> + Send
    where
        F: FnMut(glue::Decision<Self::Id>, f32) + Send,
    {
        let work = move || {
            self.search
                .distances_to(self.search.start_points, |id, distance| {
                    f(glue::Decision::new(self.filter.get(id), id), distance)
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
            self.search.expand_beam(
                ids,
                |i| pred.eval_mut(i),
                |id, distance| on_neighbors(glue::Decision::new(self.filter.get(id), id), distance),
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
            self.search.expand_beam(
                ids,
                |i| self.filter.get(*i) && pred.eval_mut(&glue::Accept::new(*i)),
                |id, distance| on_neighbors(glue::Accept::new(id), distance),
            )
        };

        ready(work)
    }
}

////////////
// Insert //
////////////

/// The [`glue::PruneAccessor`] implementation for [`Provider`].
///
/// This type implements zero-copy access to the data within its parent during prunes.
#[derive(Debug)]
pub(crate) struct Prune<'a> {
    prune: Box<dyn repr::Prune + 'a>,
    neighbors: &'a Neighbors,
    counters: LocalCounters<'a>,
}

impl<'a> Prune<'a> {
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

    // FIXME: Can we move this behind a bulk method?
    //
    // IT currently does a dynamic dispatch for each deleted check.
    pub(crate) fn partition_live(
        &mut self,
        ids: &mut AdjacencyList<u32>,
        deleted: Option<Empty<'_, u32>>,
    ) {
        let mut deleted = deleted.map(Empty::take);
        ids.retain(|i| {
            let keep = self.prune.get(*i).is_some();
            if keep && let Some(deleted) = deleted.as_mut() {
                deleted.push(*i);
            }
            keep
        });
    }

    #[cfg(test)]
    pub(crate) fn get_prune(&mut self) -> &mut dyn repr::Prune {
        &mut *self.prune
    }
}

/// The distance computer for [`Prune`].
#[derive(Debug)]
pub(crate) struct Distance<'a> {
    prune: &'a dyn repr::Prune,
    counters: LocalCounters<'a>,
}

impl<'a> Distance<'a> {
    fn new(prune: &'a dyn repr::Prune, counters: LocalCounters<'a>) -> Self {
        Self { prune, counters }
    }
}

/// An opaque element-ref for [`Prune`].
#[derive(Debug, Clone, Copy)]
#[repr(transparent)]
pub(crate) struct ElementRef<'a>(&'a [u8]);

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

impl diskann::HasId for Prune<'_> {
    type Id = u32;
}

impl glue::PruneAccessor for Prune<'_> {
    type Neighbors<'a>
        = Self
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

    fn neighbors(&mut self) -> impl std::borrow::BorrowMut<Self::Neighbors<'_>> + Send {
        self
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

impl glue::NeighborAccessor for Prune<'_> {
    fn get_neighbors(
        &mut self,
        id: Self::Id,
        neighbors: Empty<'_, u32>,
    ) -> impl std::future::Future<Output = ANNResult<()>> + Send {
        let work = move || {
            self.counters.get_neighbors(1);
            Ok(self.neighbors.get(id, neighbors.take())?)
        };
        ready(work)
    }
}

impl glue::NeighborAccessorMut for Prune<'_> {
    fn set_neighbors(
        &mut self,
        id: u32,
        neighbors: &[u32],
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

impl workingset::View<u32> for &Prune<'_> {
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
