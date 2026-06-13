/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::{
    ANNError, ANNErrorKind, ANNResult,
    error::Infallible,
    graph::{
        AdjacencyList,
        glue::{self, HybridPredicate},
        workingset,
    },
    provider,
    utils::IntoUsize,
};
use diskann_utils::views::Matrix;

use crate::{
    arbiter::epoch,
    layers::{self, Distance, QueryDistance},
    num::Bytes,
    store::{self, Primary},
};

#[derive(Debug)]
pub struct Provider<T> {
    primary: Primary,
    layer: T,
}

impl<T> Provider<T> {
    pub fn new<I, V>(layer: T, capacity: usize, start_points: I) -> Self
    where
        I: IntoIterator<Item = V>,
        T: layers::Set<V>,
    {
        let start_points: Vec<_> = start_points.into_iter().collect();
        let bytes = layers::Layer::bytes(&layer);
        let mut data = Matrix::new(0u8, start_points.len(), bytes.value());

        for (row, point) in std::iter::zip(data.row_iter_mut(), start_points.into_iter()) {
            layers::Set::into_bytes(&layer, point, row).unwrap();
        }

        let primary = Primary::new(capacity, bytes, 32, data.as_view());
        Self { primary, layer }
    }

    fn reader(&self) -> Result<store::Reader<'_>, epoch::Unavailable> {
        self.primary.reader()
    }
}

///////////////////
// Data Provider //
///////////////////

#[derive(Debug, Clone, Default)]
pub struct Context;

impl diskann::provider::ExecutionContext for Context {}

impl<T> diskann::provider::DataProvider for Provider<T>
where
    T: Send + Sync + 'static,
{
    type Context = Context;
    type InternalId = u32;
    type ExternalId = u32;
    type Error = diskann::error::Infallible;
    type Guard = diskann::provider::NoopGuard<u32>;

    fn to_internal_id(
        &self,
        _context: &Self::Context,
        gid: &Self::ExternalId,
    ) -> Result<Self::InternalId, Self::Error> {
        Ok(*gid)
    }

    /// Translate an internal id to its corresponding external id.
    fn to_external_id(
        &self,
        _context: &Self::Context,
        id: Self::InternalId,
    ) -> Result<Self::ExternalId, Self::Error> {
        Ok(id)
    }
}

fn ready<F, R>(f: F) -> std::future::Ready<R>
where
    F: FnOnce() -> R,
{
    std::future::ready(f())
}

impl<T, L> diskann::provider::SetElement<T> for Provider<L>
where
    L: layers::Layer + layers::Set<T>,
{
    type SetError = ANNError;

    fn set_element(
        &self,
        _context: &Self::Context,
        id: &Self::ExternalId,
        element: T,
    ) -> impl std::future::Future<Output = Result<Self::Guard, Self::SetError>> + Send {
        let work = move || {
            let mut slot = self.primary.acquire().unwrap();
            <L as layers::Set<T>>::into_bytes(&self.layer, element, slot.as_mut_slice())?;
            Ok(diskann::provider::NoopGuard::new(slot.slot()))
        };

        ready(work)
    }
}

////////////
// Search //
////////////

const fn start_point() -> u32 {
    0
}

#[derive(Debug)]
pub struct SearchAccessor<'a> {
    reader: store::Reader<'a>,
    distance: Box<dyn QueryDistance + 'a>,
    ids: AdjacencyList<u32>,
    expand_beam: FExpandBeam,
}

impl diskann::provider::HasId for SearchAccessor<'_> {
    type Id = u32;
}

impl glue::SearchAccessor for SearchAccessor<'_> {
    fn starting_points(
        &self,
    ) -> impl std::future::Future<Output = ANNResult<Vec<Self::Id>>> + Send {
        std::future::ready(Ok(vec![start_point()]))
    }

    fn start_point_distances<F>(
        &mut self,
        mut f: F,
    ) -> impl std::future::Future<Output = ANNResult<()>> + Send
    where
        F: FnMut(Self::Id, f32) + Send,
    {
        let work = move || {
            let start = start_point();
            match self.reader.read(start.into_usize()) {
                Some(point) => {
                    f(start, self.distance.evaluate(point)?);
                    Ok(())
                }
                // TODO: "lock" start points.
                None => Err(ANNError::message(
                    ANNErrorKind::Opaque,
                    "could not retrieve start point",
                )),
            }
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
    ids: AdjacencyList<u32>,
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

impl<'a, T, L> glue::SearchStrategy<'a, Provider<L>, T> for Strategy
where
    L: layers::Search<'a, T>,
{
    type SearchAccessor = SearchAccessor<'a>;
    type SearchAccessorError = ANNError;

    fn search_accessor(
        &'a self,
        provider: &'a Provider<L>,
        _context: &'a Context,
        query: T,
    ) -> ANNResult<SearchAccessor<'a>> {
        let distance = <L as layers::Search<'a, T>>::query_distance(&provider.layer, query)?;
        let reader = provider.primary.reader()?;
        let expand_beam = dispatch_expand_beam(reader.bytes());
        let accessor = SearchAccessor {
            reader,
            distance,
            ids: AdjacencyList::new(),
            expand_beam,
        };
        Ok(accessor)
    }
}

impl<'a, T, L> glue::DefaultPostProcessor<'a, Provider<L>, T> for Strategy
where
    L: layers::Search<'a, T>,
{
    diskann::default_post_processor!(glue::CopyIds);
}

impl<L> glue::PruneStrategy<Provider<L>> for Strategy
where
    L: layers::Layer + layers::AsDistance,
{
    type PruneAccessor<'a> = PruneAccessor<'a>;
    type PruneAccessorError = ANNError;

    fn prune_accessor<'a>(
        &self,
        provider: &'a Provider<L>,
        _context: &'a Context,
        capacity: usize,
    ) -> ANNResult<PruneAccessor<'a>> {
        Ok(PruneAccessor {
            reader: provider.primary.reader()?,
            distance: <L as layers::AsDistance>::as_distance(&provider.layer),
            ids: AdjacencyList::new(),
        })
    }
}

impl<'a, L, T> glue::InsertStrategy<'a, Provider<L>, T> for Strategy
where
    L: layers::Insert<'a, T>,
{
    type PruneStrategy = Self;
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
