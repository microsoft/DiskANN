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
    ids,
    layers::{self, Distance, QueryDistance},
    num::Bytes,
    store::{self, Primary},
    sync::epoch::Unavailable,
};

pub trait Id: Send + Sync + Hash + Eq + Clone + 'static {}
impl<T> Id for T where T: Send + Sync + Hash + Eq + Clone + 'static {}

#[derive(Debug)]
pub struct Provider<L, M = u32>
where
    M: Id,
{
    primary: Primary,
    layer: L,
    mapping: ids::Sharded<M>,
}

impl<L, M> Provider<L, M>
where
    M: Id,
{
    pub fn new<I, T>(layer: L, capacity: usize, start_points: I) -> Self
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

        let primary = Primary::new(capacity, bytes, 32, data.as_view());
        let mapping = ids::Sharded::new(capacity);
        Self {
            primary,
            layer,
            mapping,
        }
    }

    fn reader(&self) -> Result<store::Reader<'_>, Unavailable> {
        self.primary.reader()
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
        let id = self.mapping.to_internal(gid).unwrap();
        Ok(id)
    }

    /// Translate an internal id to its corresponding external id.
    fn to_external_id(
        &self,
        _context: &Self::Context,
        id: Self::InternalId,
    ) -> Result<Self::ExternalId, Self::Error> {
        let gid = self.mapping.to_external(id).unwrap();
        Ok(gid)
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
        let internal = self.mapping.remove(gid).unwrap();
        assert!(self.primary.delete(internal.into_usize()));
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
        if self
            .primary
            .reader()
            .unwrap()
            .can_read(id.into_usize())
            .unwrap()
        {
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

    fn statuses_unordered<Itr, F>(
        &self,
        _context: &Self::Context,
        itr: Itr,
        mut f: F,
    ) -> impl std::future::Future<Output = Result<(), Self::Error>> + Send
    where
        Itr: Iterator<Item = Self::InternalId> + Send,
        F: FnMut(ANNResult<diskann::provider::ElementStatus>, Self::InternalId) + Send,
    {
        let work = move || {
            let reader = self.primary.reader().unwrap();
            for i in itr {
                if reader.can_read(i.into_usize()).unwrap() {
                    f(Ok(diskann::provider::ElementStatus::Valid), i)
                } else {
                    f(Ok(diskann::provider::ElementStatus::Deleted), i)
                }
            }
            Ok(())
        };

        ready(work)
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
            let mut slot = self.primary.acquire().unwrap();

            // TODO: Proper cleanup via `Guard` or some other mechanism on the event of
            // insert failure.
            <L as layers::Set<T>>::into_bytes(&self.layer, element, slot.as_mut_slice())?;
            self.mapping.insert(id.clone(), slot.slot()).unwrap();

            Ok(diskann::provider::NoopGuard::new(slot.slot()))
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
    distance: Box<dyn QueryDistance + 'a>,
    ids: AdjacencyList<u32>,
    expand_beam: FExpandBeam,

    // The parent provider for the accessor.
    provider: &'a (dyn std::any::Any + Send + Sync),
    start_points: std::ops::Range<u32>,
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
                        f(p, self.distance.evaluate(point)?);
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
                self.reader
                    .neighbors()
                    .get(i.into_usize(), &mut self.ids)
                    .unwrap();

                // Filter out unvisited IDs and ensure that all the IDs we are about
                self.ids
                    .retain(|i| pred.eval_mut(i) && self.reader.is_in_bounds(i.into_usize()));

                unsafe {
                    (self.expand_beam)(
                        &self.ids,
                        8,
                        &self.reader,
                        &*self.distance,
                        &mut on_neighbors,
                    )
                }?;
            }

            Ok(())
        };

        ready(work)
    }
}

type FExpandBeam = unsafe fn(
    &[u32],
    usize,
    &store::Reader<'_>,
    &dyn layers::QueryDistance,
    &mut dyn FnMut(u32, f32),
) -> ANNResult<()>;

fn dispatch_expand_beam(bytes: Bytes) -> FExpandBeam {
    if bytes <= Bytes::CACHELINE {
        expand_beam_inner::<1>
    } else if bytes <= Bytes::CACHELINE.unchecked_mul(2) {
        expand_beam_inner::<2>
    } else if bytes <= Bytes::CACHELINE.unchecked_mul(3) {
        expand_beam_inner::<3>
    } else if bytes <= Bytes::CACHELINE.unchecked_mul(4) {
        expand_beam_inner::<4>
    } else if bytes <= Bytes::CACHELINE.unchecked_mul(5) {
        expand_beam_inner::<5>
    } else if bytes <= Bytes::CACHELINE.unchecked_mul(6) {
        expand_beam_inner::<6>
    } else if bytes <= Bytes::CACHELINE.unchecked_mul(7) {
        expand_beam_inner::<7>
    } else if bytes <= Bytes::CACHELINE.unchecked_mul(16) {
        expand_beam_inner::<8>
    } else {
        expand_beam_inner::<16>
    }
}

const CACHE_LINE_SIZE: usize = 64;

pub unsafe fn test_function(
    list: &[u32],
    lookahead: usize,
    reader: &store::Reader<'_>,
    distance: &dyn layers::QueryDistance,
    f: &mut dyn FnMut(u32, f32),
) -> ANNResult<()> {
    unsafe { expand_beam_inner::<4>(list, lookahead, reader, distance, f) }
}

/// Safety (no # yet because we need to revisit this - clippy will lint)
///
/// * All items in `list` must in-bounds with respect to `reader`.
/// * The number of bytes associated with `N` cache lines must "make sense".
unsafe fn expand_beam_inner<const N: usize>(
    list: &[u32],
    lookahead: usize,
    reader: &store::Reader<'_>,
    distance: &dyn layers::QueryDistance,
    f: &mut dyn FnMut(u32, f32),
) -> ANNResult<()> {
    debug_assert!(
        N * CACHE_LINE_SIZE
            <= reader
                .bytes()
                .checked_next_multiple_of(Bytes::CACHELINE)
                .unwrap()
                .value(),
        "we really rely on this: {}, bytes = {}",
        N,
        reader.bytes()
    );

    let len = list.len();
    let lookahead = lookahead.min(len);

    for j in 0..lookahead {
        unsafe {
            diskann_vector::prefetch_exactly::<N>(
                reader
                    .read_raw_unchecked(list.get_unchecked(j).into_usize())
                    .as_ptr()
                    .cast(),
            )
        }
    }

    let mut j = lookahead;
    for &i in list.iter() {
        if j != len {
            unsafe {
                diskann_vector::prefetch_exactly::<N>(
                    reader
                        .read_raw_unchecked(list.get_unchecked(j).into_usize())
                        .as_ptr()
                        .cast(),
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

#[derive(Debug)]
pub struct PruneAccessor<'a> {
    reader: store::Reader<'a>,
    distance: &'a dyn Distance,
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
        = &'a dyn Distance
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
        Ok((self, &*self.distance))
    }
}

impl provider::NeighborAccessor for PruneAccessor<'_> {
    fn get_neighbors(
        &mut self,
        id: Self::Id,
        neighbors: &mut AdjacencyList<Self::Id>,
    ) -> impl std::future::Future<Output = ANNResult<()>> + Send {
        let work = move || {
            Ok(self
                .reader
                .neighbors()
                .get(id.into_usize(), neighbors)
                .unwrap())
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
            Ok(self
                .reader
                .neighbors()
                .set(id.into_usize(), neighbors)
                .unwrap())
        };
        ready(work)
    }

    fn append_vector(
        &mut self,
        id: Self::Id,
        neighbors: &[Self::Id],
    ) -> impl std::future::Future<Output = ANNResult<()>> + Send {
        let work = move || -> ANNResult<()> {
            self.reader
                .neighbors()
                .lock(id.into_usize())
                .unwrap()
                .append(neighbors)
                .unwrap();
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
        self.reader.read(id.into_usize())
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
        let distance = <L as layers::Search>::query_distance(&provider.layer, query)?;
        let reader = provider.primary.reader()?;
        let expand_beam = dispatch_expand_beam(reader.bytes());
        let accessor = SearchAccessor {
            reader,
            distance,
            ids: AdjacencyList::new(),
            expand_beam,
            provider,
            start_points: provider.primary.frozen(),
        };
        Ok(accessor)
    }
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
            reader: provider.primary.reader()?,
            distance: <L as layers::AsDistance>::as_distance(&provider.layer),
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
    Self: glue::PruneStrategy<Provider<layers::Full<f32>, M>>,
    Self: for<'a> glue::InsertStrategy<
            'a,
            Provider<layers::Full<f32>, M>,
            &'a [f32],
            SearchAccessor = SearchAccessor<'a>,
        >,
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
            let reader = provider.primary.reader().unwrap();
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

    use diskann::graph::DiskANNIndex;
    use diskann_vector::distance::Metric;

    use crate::layers::Full;

    #[tokio::test]
    async fn smoke() {
        let full = Full::<f32>::new(1, Metric::L2);
        let start_points: [&[f32]; _] = [&[1.0], &[2.0]];

        let provider = Provider::new(full, 10, start_points);

        let config = diskann::graph::config::Builder::new(
            10,
            diskann::graph::config::MaxDegree::Same,
            100,
            (Metric::L2).into(),
        )
        .build()
        .unwrap();

        let index = DiskANNIndex::new(config, provider, None);

        index.insert(&Strategy, &Context, &0, &[3.0]).await.unwrap();
        index.insert(&Strategy, &Context, &1, &[4.0]).await.unwrap();
        index.insert(&Strategy, &Context, &2, &[5.0]).await.unwrap();
    }
}
