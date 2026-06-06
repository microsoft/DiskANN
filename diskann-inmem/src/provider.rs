/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::{
    error::Infallible,
    graph::{
        glue::{self, HybridPredicate},
        workingset, AdjacencyList,
    },
    provider,
    utils::IntoUsize,
    ANNError, ANNErrorKind, ANNResult,
};
use diskann_utils::future::{AsyncFriendly, SendFuture};

use crate::{
    layers::{self, Distance, QueryDistance},
    store::{self, Primary},
};

#[derive(Debug)]
pub struct Provider<T> {
    primary: Primary,
    layer: T,
}

#[derive(Debug, Clone)]
pub struct Context {}

impl diskann::provider::ExecutionContext for Context {}

impl<T> diskann::provider::DataProvider for Provider<T>
where
    T: layers::Layer,
{
    type Context = Context;
    type InternalId = u32;
    type ExternalId = u32;
    type Error = diskann::error::Infallible;
    type Guard = diskann::provider::NoopGuard<u32>;

    fn to_internal_id(
        &self,
        context: &Self::Context,
        gid: &Self::ExternalId,
    ) -> Result<Self::InternalId, Self::Error> {
        Ok(*gid)
    }

    /// Translate an internal id to its corresponding external id.
    fn to_external_id(
        &self,
        context: &Self::Context,
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
                for neighbor in self.ids.iter().filter(|i| pred.eval_mut(i)) {
                    if let Some(data) = self.reader.read(i.into_usize()) {
                        on_neighbors(*neighbor, self.distance.evaluate(data)?)
                    }
                }
            }

            Ok(())
        };

        ready(work)
    }
}

impl<T, L> diskann::provider::SetElement<T> for Provider<L>
where
    L: layers::Layer + layers::Set<T>
{
    type SetError = ANNError;

    fn set_element(
        &self,
        context: &Self::Context,
        id: &Self::ExternalId,
        element: T,
    ) -> impl std::future::Future<Output = Result<Self::Guard, Self::SetError>> + Send {
        let work = move || {
            let mut write = self.primary.write(id.into_usize()).unwrap();
            <L as layers::Set<T>>::into_bytes(&self.layer, element, write.as_mut_slice())?;
            Ok(diskann::provider::NoopGuard::new(*id))
        };

        ready(work)
    }
}

////////////
// Insert //
////////////

#[derive(Debug)]
pub struct PruneAccessor<'a> {
    reader: store::Reader<'a>,
    set: workingset::Map<u32, Box<[u8]>>,
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
        = workingset::map::View<'a, u32, Box<[u8]>>
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
        itr: Itr,
    ) -> ANNResult<(Self::View<'a>, Self::Distance<'a>)>
    where
        Itr: ExactSizeIterator<Item = Self::Id> + Clone + Send + Sync,
    {
        let v = self
            .set
            .fill(itr, |i| -> Result<_, Infallible> {
                Ok(self.reader.read(i.into_usize()).map(|v| v.into()))
            })
            .unwrap();

        Ok((v, &*self.distance))
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
            let current = self.reader.neighbors().lock(id.into_usize()).unwrap();

            // Copy out the current neighbors.
            let mut resize = self.ids.resize(current.len());
            resize.copy_from_slice(current.as_slice());
            resize.finish(current.len());

            // Append the new neighbors.
            self.ids.extend_from_slice(neighbors);
            current.write(&self.ids).unwrap();
            Ok(())
        };

        ready(work)
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
        context: &'a Context,
        query: T
    ) -> ANNResult<SearchAccessor<'a>> {
        let distance = <L as layers::AsQueryDistance<'a, T>>::as_query_distance(&provider.layer, query)?;
        let accessor = SearchAccessor {
            reader: provider.primary.reader(),
            distance,
            ids: AdjacencyList::new(),
        };
        Ok(accessor)
    }
}

impl<L> glue::PruneStrategy<Provider<L>> for Strategy
where
    L: layers::Layer + layers::AsDistance,
{
    type PruneAccessor<'a> = PruneAccessor<'a>;
    type PruneAccessorError = diskann::error::Infallible;

    fn prune_accessor<'a>(
        &self,
        provider: &'a Provider<L>,
        context: &'a Context,
        capacity: usize,
    ) -> Result<PruneAccessor<'a>, diskann::error::Infallible> {
        let set = workingset::map::Builder::new(workingset::map::Capacity::Default).build(capacity);
        Ok(PruneAccessor {
            reader: provider.primary.reader(),
            set,
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
