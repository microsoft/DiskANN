use std::{
    collections::{HashMap, hash_map::Entry},
    future, mem,
    ops::{Deref, DerefMut},
};

use dashmap::DashMap;
use diskann::{
    ANNError, ANNErrorKind, ANNResult,
    graph::{
        AdjacencyList, SearchOutputBuffer,
        config::defaults::MAX_OCCLUSION_SIZE,
        glue::{
            self, ExpandBeam, FillSet, InplaceDeleteStrategy, InsertStrategy, PruneStrategy,
            SearchExt, SearchPostProcess, SearchStrategy,
        },
    },
    neighbor::Neighbor,
    provider::{
        Accessor, BuildDistanceComputer, BuildQueryComputer, DataProvider, DelegateNeighbor,
        Delete, ElementStatus, HasId, NeighborAccessor, NeighborAccessorMut, NoopGuard, SetElement,
    },
    utils::{
        VectorRepr,
        object_pool::{AsPooled, ObjectPool, PooledRef, Undef},
    },
};
use diskann_providers::model::graph::provider::async_::common::{FullPrecision, Internal};
use diskann_vector::{PreprocessedDistanceFunction, contains::ContainsSimd, distance::Metric};
use thiserror::Error;

use crate::{
    fsm::{FreeSpaceMap, FsmError},
    garnet::{Callbacks, Context, GarnetError, GarnetId, Term},
};

#[derive(Clone)]
struct AdjList(AdjacencyList<u32>);

impl Deref for AdjList {
    type Target = AdjacencyList<u32>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for AdjList {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl AsPooled<Undef> for AdjList {
    fn create(args: Undef) -> Self {
        AdjList(AdjacencyList::with_capacity(args.len))
    }

    fn modify(&mut self, _args: Undef) {
        // AdjList is already automatically resizable, no need to do anything here.
    }
}

#[derive(Debug, Error)]
pub enum GarnetProviderError {
    #[error("Garnet operation failed")]
    Garnet(#[from] GarnetError),
    #[error("FSM error")]
    Fsm(#[from] FsmError),
    #[error("Start point invalid")]
    StartPoint,
}

impl From<GarnetProviderError> for ANNError {
    #[track_caller]
    fn from(value: GarnetProviderError) -> Self {
        ANNError::new(ANNErrorKind::GetVertexDataError, value)
    }
}

diskann::always_escalate!(GarnetProviderError);

pub struct GarnetProvider<T: VectorRepr> {
    dim: usize,
    metric_type: Metric,
    max_degree: usize,
    callbacks: Callbacks,
    id_buffer_pool: ObjectPool<AdjList>,
    filtered_ids_pool: ObjectPool<Vec<u32>>,
    neighbor_cache: DashMap<u32, Vec<u32>, foldhash::fast::RandomState>,
    start_point_cache: DashMap<u32, Vec<T>, foldhash::fast::RandomState>,
    fsm: FreeSpaceMap,
}

impl<T: VectorRepr> GarnetProvider<T> {
    pub fn new(
        dim: usize,
        metric_type: Metric,
        max_degree: usize,
        callbacks: Callbacks,
        context: Context,
    ) -> Result<Self, GarnetProviderError> {
        let parallelism = std::thread::available_parallelism().unwrap().get() * 2;
        let id_buffer_pool =
            ObjectPool::new(Undef::new(max_degree + 1), parallelism, Some(parallelism));
        let filtered_ids_pool = ObjectPool::new(
            Undef::new(MAX_OCCLUSION_SIZE.get() as usize * 2),
            parallelism,
            Some(parallelism),
        );

        let start_point_cache =
            DashMap::with_capacity_and_hasher(1, foldhash::fast::RandomState::default());
        let neighbor_cache =
            DashMap::with_capacity_and_hasher(1, foldhash::fast::RandomState::default());

        // Try to read the start point from Garnet
        let mut v = vec![T::default(); dim];
        if callbacks.read_single_iid(context.term(Term::Vector), 0, &mut v) {
            let mut neighbors = vec![0u32; max_degree + 1];
            if !callbacks.read_single_iid(context.term(Term::Neighbors), 0, &mut neighbors) {
                return Err(GarnetError::Read.into());
            }

            start_point_cache.insert(0, v);

            let len = neighbors[max_degree] as usize;
            neighbors.truncate(len);
            neighbor_cache.insert(0, neighbors);
        }

        let fsm = FreeSpaceMap::new(context, callbacks)?;

        Ok(Self {
            dim,
            metric_type,
            max_degree,
            callbacks,
            id_buffer_pool,
            filtered_ids_pool,
            start_point_cache,
            neighbor_cache,
            fsm,
        })
    }

    pub fn maybe_set_start_point(
        &self,
        context: &Context,
        point: &[T],
    ) -> Result<(), GarnetProviderError> {
        let mut v = vec![T::default(); self.dim];
        if self
            .callbacks
            .read_single_iid(context.term(Term::Vector), 0, &mut v)
        {
            // Garnet already has a start point, so use that instead of `point`
            let mut neighbors = vec![0u32; self.max_degree + 1];
            if !self
                .callbacks
                .read_single_iid(context.term(Term::Neighbors), 0, &mut neighbors)
            {
                return Err(GarnetError::Read.into());
            }

            self.start_point_cache.insert(0, v);
            let len = neighbors[self.max_degree] as usize;
            neighbors.truncate(len);
            self.neighbor_cache.insert(0, neighbors);
        } else {
            let neighbors = vec![0u32; self.max_degree + 1];

            // Grab the start point id, which must be zero.
            let id = self.fsm.next_id(*context)?;
            if id != 0 {
                self.fsm.mark_free(*context, id)?;
                return Err(GarnetProviderError::StartPoint);
            }

            if !self
                .callbacks
                .write_iid(context.term(Term::Vector), 0, point)
            {
                return Err(GarnetError::Write.into());
            }

            if !self
                .callbacks
                .write_iid(context.term(Term::Neighbors), 0, &neighbors)
            {
                return Err(GarnetError::Write.into());
            }

            self.start_point_cache.insert(0, point.to_vec());
            self.neighbor_cache
                .insert(0, Vec::with_capacity(self.max_degree + 1));
        }

        Ok(())
    }

    pub fn start_points_exist(&self) -> bool {
        self.start_point_cache.get(&0).is_some() && self.neighbor_cache.get(&0).is_some()
    }

    pub fn callbacks(&self) -> &Callbacks {
        &self.callbacks
    }

    pub fn set_attributes(
        &self,
        context: &Context,
        id: &GarnetId,
        data: &[u8],
    ) -> Result<(), GarnetProviderError> {
        if self
            .callbacks
            .write_eid(context.term(Term::Attributes), id, data)
        {
            Ok(())
        } else {
            Err(GarnetError::Write.into())
        }
    }
    pub fn vector_id_exists(&self, context: &Context, id: &GarnetId) -> bool {
        let iid = match self.to_internal_id(context, id) {
            Ok(iid) => iid,
            Err(_) => return false,
        };
        !self.fsm.is_free(*context, iid).unwrap_or(true)
    }

    pub fn vector_iid_exists(&self, context: &Context, id: u32) -> bool {
        !self.fsm.is_free(*context, id).unwrap_or(true)
    }

    pub fn max_internal_id(&self) -> u32 {
        self.fsm.max_id()
    }
}

impl<T: VectorRepr> DataProvider for GarnetProvider<T> {
    type Context = Context;
    type InternalId = u32;
    type ExternalId = GarnetId;
    type Error = GarnetProviderError;

    fn to_internal_id(
        &self,
        context: &Context,
        gid: &GarnetId,
    ) -> Result<Self::InternalId, Self::Error> {
        let mut id = 0u32;
        if !self.callbacks.read_single_eid(
            context.term(Term::IntMap),
            gid,
            bytemuck::bytes_of_mut(&mut id),
        ) {
            return Err(GarnetProviderError::Garnet(GarnetError::Read));
        }
        Ok(id)
    }

    fn to_external_id(&self, context: &Context, id: u32) -> Result<Self::ExternalId, Self::Error> {
        match self
            .callbacks
            .read_varsize_iid(context.term(Term::ExtMap), id)
        {
            Some(eid) => Ok(eid.into()),
            None => Err(GarnetProviderError::Garnet(GarnetError::Read)),
        }
    }
}

impl<T: VectorRepr> SetElement<[T]> for GarnetProvider<T> {
    type SetError = GarnetProviderError;
    type Guard = NoopGuard<u32>;

    async fn set_element(
        &self,
        context: &Self::Context,
        id: &Self::ExternalId,
        element: &[T],
    ) -> Result<Self::Guard, Self::SetError> {
        let internal_id = self.fsm.next_id(*context)?;

        let insert = || -> Result<(), Self::SetError> {
            self.callbacks
                .write_iid(context.term(Term::Vector), internal_id, element)
                .then_some(())
                .ok_or(GarnetError::Write)?;
            self.callbacks
                .write_iid(context.term(Term::ExtMap), internal_id, id)
                .then_some(())
                .ok_or(GarnetError::Write)?;
            self.callbacks
                .write_eid(
                    context.term(Term::IntMap),
                    id,
                    bytemuck::bytes_of(&internal_id),
                )
                .then_some(())
                .ok_or(GarnetError::Write)?;
            Ok(())
        };

        match insert() {
            Ok(()) => (),
            Err(e) => {
                self.fsm.mark_free(*context, internal_id)?;
                return Err(e);
            }
        }

        Ok(NoopGuard::new(internal_id))
    }
}

impl<T: VectorRepr> Delete for GarnetProvider<T> {
    fn delete(
        &self,
        context: &Context,
        gid: &GarnetId,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send {
        let id = match self.to_internal_id(context, gid) {
            Ok(id) => id,
            Err(e) => return future::ready(Err(e)),
        };

        // Mark the ID free in the FSM.
        if let Err(e) = self.fsm.mark_free(*context, id) {
            return future::ready(Err(e.into()));
        };

        // Delete all the data associated with the vector.
        let mut ok = true;
        ok &= self.callbacks.delete_iid(context.term(Term::ExtMap), id);
        ok &= self.callbacks.delete_eid(context.term(Term::IntMap), gid);
        ok &= self
            .callbacks
            .delete_eid(context.term(Term::Attributes), gid);
        // NOTE: Commented out until DiskANN fixes accessing neighbor data post-delete.
        //ok &= self.callbacks.delete_iid(context.term(Term::Neighbors), id);
        ok &= self.callbacks.delete_iid(context.term(Term::Vector), id);

        if !ok {
            return future::ready(Err(GarnetError::Delete.into()));
        }

        future::ready(Ok(()))
    }

    fn release(
        &self,
        _context: &Self::Context,
        _id: Self::InternalId,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send {
        // This is a no-op since we just do hard deletes.
        future::ready(Ok(()))
    }

    fn status_by_internal_id(
        &self,
        context: &Self::Context,
        id: Self::InternalId,
    ) -> impl Future<Output = Result<diskann::provider::ElementStatus, Self::Error>> + Send {
        let status = match self.fsm.is_free(*context, id) {
            Ok(true) => ElementStatus::Deleted,
            Ok(false) => ElementStatus::Valid,
            Err(e) => return future::ready(Err(e.into())),
        };

        future::ready(Ok(status))
    }

    async fn status_by_external_id(
        &self,
        context: &Self::Context,
        gid: &Self::ExternalId,
    ) -> Result<diskann::provider::ElementStatus, Self::Error> {
        let id = self.to_internal_id(context, gid)?;
        self.status_by_internal_id(context, id).await
    }
}

#[allow(dead_code)]
pub struct FullAccessor<'a, T: VectorRepr> {
    provider: &'a GarnetProvider<T>,
    context: &'a Context,
    is_search: bool,
    id_buffer: PooledRef<'a, AdjList>,
    filtered_ids: PooledRef<'a, Vec<u32>>,
}

impl<'a, T: VectorRepr> FullAccessor<'a, T> {
    pub(crate) fn new(
        provider: &'a GarnetProvider<T>,
        context: &'a Context,
        is_search: bool,
    ) -> Self {
        let id_buffer = provider
            .id_buffer_pool
            .get_ref(Undef::new(provider.max_degree + 1));
        let filtered_ids = provider
            .filtered_ids_pool
            .get_ref(Undef::new(MAX_OCCLUSION_SIZE.get() as usize * 2)); // x2 to allow for the length prefixes for garnet
        FullAccessor {
            provider,
            context,
            is_search,
            id_buffer,
            filtered_ids,
        }
    }

    fn get_neighbors_internal(&mut self, id: u32, dest: Option<&mut AdjacencyList<u32>>) -> bool {
        let dest = dest.unwrap_or(&mut self.id_buffer);

        let mut guard = dest.resize(self.provider.max_degree + 1);

        if id == 0
            && let Some(cached) = self.provider.neighbor_cache.get(&id)
        {
            guard[0..cached.len()].copy_from_slice(&cached);
            guard.finish(cached.len());
            return true;
        }

        if !self.provider.callbacks.read_single_iid(
            self.context.term(Term::Neighbors),
            id,
            &mut guard,
        ) {
            return false;
        }

        let len = guard[self.provider.max_degree];
        guard.finish(len as usize);

        true
    }
}

impl<T: VectorRepr> HasId for FullAccessor<'_, T> {
    type Id = u32;
}

impl<T: VectorRepr> SearchExt for FullAccessor<'_, T> {
    fn starting_points(&self) -> impl Future<Output = ANNResult<Vec<Self::Id>>> + Send {
        let points = if self.provider.start_points_exist() {
            vec![0]
        } else {
            vec![]
        };
        future::ready(Ok(points))
    }

    fn is_not_start_point(
        &self,
    ) -> impl Future<Output = ANNResult<impl Fn(Self::Id) -> bool + Send + Sync + 'static>> + Send
    {
        future::ready(Ok(move |id| id != 0))
    }
}

impl<T: VectorRepr> ExpandBeam<[T]> for FullAccessor<'_, T> {
    fn expand_beam<Itr, P, F>(
        &mut self,
        ids: Itr,
        computer: &Self::QueryComputer,
        mut pred: P,
        mut on_neighbors: F,
    ) -> impl Future<Output = ANNResult<()>> + Send
    where
        Itr: Iterator<Item = Self::Id> + Send,
        P: glue::HybridPredicate<Self::Id> + Send + Sync,
        F: FnMut(f32, Self::Id) + Send,
    {
        for nl_id in ids {
            self.get_neighbors_internal(nl_id, None);

            self.filtered_ids.clear();
            for id in self
                .id_buffer
                .iter()
                .copied()
                .filter(|id| pred.eval_mut(id))
            {
                if id == 0 {
                    let guard = if let Some(r) = self.provider.start_point_cache.get(&id) {
                        r
                    } else {
                        return future::ready(Err(
                            GarnetProviderError::Garnet(GarnetError::Read).into()
                        ));
                    };
                    let dist = computer.evaluate_similarity(&*guard);
                    on_neighbors(dist, id);
                } else {
                    self.filtered_ids.push(4);
                    self.filtered_ids.push(id);
                }
            }

            self.provider.callbacks.read_multi_lpiid(
                self.context.term(Term::Vector),
                &self.filtered_ids,
                |i, v| {
                    let dist = computer.evaluate_similarity(v);
                    on_neighbors(dist, self.filtered_ids[i as usize * 2 + 1]);
                },
            );
        }

        future::ready(Ok(()))
    }
}

impl<T: VectorRepr> Accessor for FullAccessor<'_, T> {
    type Extended = Vec<T>;
    type Element<'a>
        = Vec<T>
    where
        Self: 'a;
    type ElementRef<'a> = &'a [T];
    type GetError = GarnetProviderError;

    fn get_element(
        &mut self,
        id: Self::Id,
    ) -> impl Future<Output = Result<Self::Element<'_>, Self::GetError>> + Send {
        let mut v = vec![T::default(); self.provider.dim];

        if id == 0 {
            let guard = if let Some(r) = self.provider.start_point_cache.get(&id) {
                r
            } else {
                return future::ready(Err(GarnetError::Read.into()));
            };
            v.copy_from_slice(&guard);
            return future::ready(Ok(v));
        }

        if !self
            .provider
            .callbacks
            .read_single_iid(self.context.term(Term::Vector), id, &mut v)
        {
            return future::ready(Err(GarnetError::Read.into()));
        }

        future::ready(Ok(v))
    }
}

impl<T: VectorRepr> BuildDistanceComputer for FullAccessor<'_, T> {
    type DistanceComputer = T::Distance;
    type DistanceComputerError = GarnetProviderError;

    fn build_distance_computer(
        &self,
    ) -> Result<Self::DistanceComputer, Self::DistanceComputerError> {
        Ok(T::distance(
            self.provider.metric_type,
            Some(self.provider.dim),
        ))
    }
}

impl<T: VectorRepr> BuildQueryComputer<[T]> for FullAccessor<'_, T> {
    type QueryComputer = T::QueryDistance;
    type QueryComputerError = GarnetProviderError;

    fn build_query_computer(
        &self,
        from: &[T],
    ) -> Result<Self::QueryComputer, Self::QueryComputerError> {
        Ok(T::query_distance(from, self.provider.metric_type))
    }
}

impl<T: VectorRepr> FillSet for FullAccessor<'_, T> {
    async fn fill_set<Itr>(
        &mut self,
        set: &mut HashMap<Self::Id, Self::Extended>,
        itr: Itr,
    ) -> Result<(), Self::GetError>
    where
        Itr: Iterator<Item = Self::Id> + Send + Sync,
    {
        self.filtered_ids.clear();
        for id in itr {
            if id == 0 {
                if let Entry::Vacant(e) = set.entry(id) {
                    let guard = if let Some(r) = self.provider.start_point_cache.get(&id) {
                        r
                    } else {
                        return Err(GarnetError::Read.into());
                    };
                    e.insert(guard.to_vec());
                }
            } else if !set.contains_key(&id) {
                self.filtered_ids.push(4);
                self.filtered_ids.push(id);
            }
        }

        if !self.filtered_ids.is_empty() {
            self.provider.callbacks.read_multi_lpiid(
                self.context.term(Term::Vector),
                &self.filtered_ids,
                |id, v| {
                    set.insert(self.filtered_ids[id as usize * 2 + 1], v.to_vec());
                },
            );
        }

        Ok(())
    }
}

pub struct DelegateNeighborAccessor<'p, 'a, T: VectorRepr>(&'a mut FullAccessor<'p, T>);

impl<T: VectorRepr> HasId for DelegateNeighborAccessor<'_, '_, T> {
    type Id = u32;
}

impl<T: VectorRepr> NeighborAccessor for DelegateNeighborAccessor<'_, '_, T> {
    fn get_neighbors(
        self,
        id: Self::Id,
        neighbors: &mut AdjacencyList<Self::Id>,
    ) -> impl Future<Output = ANNResult<Self>> + Send {
        if !self.0.get_neighbors_internal(id, Some(neighbors)) {
            return future::ready(Err(GarnetProviderError::Garnet(GarnetError::Read).into()));
        }

        future::ready(Ok(self))
    }
}

impl<'p, 'a, T: VectorRepr> DelegateNeighbor<'a> for FullAccessor<'p, T> {
    type Delegate = DelegateNeighborAccessor<'p, 'a, T>;

    fn delegate_neighbor(&'a mut self) -> Self::Delegate {
        DelegateNeighborAccessor(self)
    }
}

impl<T: VectorRepr> NeighborAccessorMut for DelegateNeighborAccessor<'_, '_, T> {
    fn set_neighbors(
        self,
        id: Self::Id,
        neighbors: &[Self::Id],
    ) -> impl Future<Output = ANNResult<Self>> + Send {
        let mut guard = self.0.id_buffer.resize(self.0.provider.max_degree + 1);
        guard[0..neighbors.len()].copy_from_slice(neighbors);
        guard[self.0.provider.max_degree] = neighbors.len() as u32;

        if !self
            .0
            .provider
            .callbacks
            .write_iid(self.0.context.term(Term::Neighbors), id, &guard)
        {
            return future::ready(Err(GarnetProviderError::Garnet(GarnetError::Write).into()));
        }

        guard.finish(0);

        if id == 0 {
            self.0
                .provider
                .neighbor_cache
                .insert(id, neighbors.to_vec());
        }

        future::ready(Ok(self))
    }

    fn append_vector(
        self,
        id: Self::Id,
        neighbors: &[Self::Id],
    ) -> impl Future<Output = ANNResult<Self>> + Send {
        let max_degree = self.0.provider.max_degree;
        if !self.0.provider.callbacks.rmw_iid(
            self.0.context.term(Term::Neighbors),
            id,
            (self.0.provider.max_degree + 1) * mem::size_of::<u32>(),
            |data: &mut [u32]| {
                let mut len = (data[max_degree] as usize).min(max_degree);

                for &nbr in neighbors {
                    if len == max_degree {
                        return;
                    }

                    if u32::contains_simd(&data[0..len], nbr) {
                        continue;
                    }

                    data[len] = nbr;
                    len += 1;
                    data[max_degree] = len as u32;
                }

                if id == 0
                    && let Some(mut ns) = self.0.provider.neighbor_cache.get_mut(&id)
                {
                    ns.clear();
                    ns.extend(data.iter().copied().take(len));
                }
            },
        ) {
            return future::ready(Err(GarnetProviderError::Garnet(GarnetError::Write).into()));
        }

        future::ready(Ok(self))
    }
}

impl<T: VectorRepr> SearchStrategy<GarnetProvider<T>, [T]> for Internal<FullPrecision> {
    type SearchAccessor<'a> = FullAccessor<'a, T>;
    type SearchAccessorError = GarnetProviderError;
    type QueryComputer = T::QueryDistance;
    type PostProcessor = glue::CopyIds;

    fn search_accessor<'a>(
        &'a self,
        provider: &'a GarnetProvider<T>,
        context: &'a <GarnetProvider<T> as DataProvider>::Context,
    ) -> Result<Self::SearchAccessor<'a>, Self::SearchAccessorError> {
        Ok(FullAccessor::new(provider, context, true))
    }

    fn post_processor(&self) -> Self::PostProcessor {
        Default::default()
    }
}

/// A [`SearchPostProcess`] base object that copies each `Neighbor` to a `(ExternalId, f32)` pair
/// and writes as many as possible to the output buffer.
#[derive(Debug, Default, Clone, Copy)]
pub struct CopyExternalIds;

impl<'a, T: VectorRepr> SearchPostProcess<FullAccessor<'a, T>, [T], GarnetId> for CopyExternalIds {
    type Error = GarnetProviderError;

    fn post_process<I, B>(
        &self,
        accessor: &mut FullAccessor<'a, T>,
        _query: &[T],
        _computer: &<FullAccessor<'a, T> as BuildQueryComputer<[T]>>::QueryComputer,
        candidates: I,
        output: &mut B,
    ) -> impl Future<Output = Result<usize, Self::Error>> + Send
    where
        I: Iterator<Item = Neighbor<<FullAccessor<'a, T> as HasId>::Id>> + Send,
        B: SearchOutputBuffer<GarnetId> + Send + ?Sized,
    {
        let initial = output.current_len();
        for n in candidates {
            let id = match accessor.provider.to_external_id(accessor.context, n.id) {
                Ok(id) => id,
                Err(e) => return future::ready(Err(e)),
            };

            if output.push(id, n.distance).is_full() {
                break;
            }
        }
        let count = output.current_len() - initial;
        future::ready(Ok(count))
    }
}

impl<T: VectorRepr> SearchStrategy<GarnetProvider<T>, [T], GarnetId> for FullPrecision {
    type SearchAccessor<'a> = FullAccessor<'a, T>;
    type SearchAccessorError = GarnetProviderError;
    type QueryComputer = T::QueryDistance;
    type PostProcessor = glue::Pipeline<glue::FilterStartPoints, CopyExternalIds>;

    fn search_accessor<'a>(
        &'a self,
        provider: &'a GarnetProvider<T>,
        context: &'a <GarnetProvider<T> as DataProvider>::Context,
    ) -> Result<Self::SearchAccessor<'a>, Self::SearchAccessorError> {
        Ok(FullAccessor::new(provider, context, true))
    }

    fn post_processor(&self) -> Self::PostProcessor {
        Default::default()
    }
}
impl<T: VectorRepr> SearchStrategy<GarnetProvider<T>, [T], u32> for FullPrecision {
    type SearchAccessor<'a> = FullAccessor<'a, T>;
    type SearchAccessorError = GarnetProviderError;
    type QueryComputer = T::QueryDistance;
    type PostProcessor = glue::CopyIds;

    fn search_accessor<'a>(
        &'a self,
        provider: &'a GarnetProvider<T>,
        context: &'a <GarnetProvider<T> as DataProvider>::Context,
    ) -> Result<Self::SearchAccessor<'a>, Self::SearchAccessorError> {
        Ok(FullAccessor::new(provider, context, true))
    }

    fn post_processor(&self) -> Self::PostProcessor {
        Default::default()
    }
}

impl<T: VectorRepr> PruneStrategy<GarnetProvider<T>> for FullPrecision {
    type PruneAccessor<'a> = FullAccessor<'a, T>;
    type PruneAccessorError = GarnetProviderError;
    type DistanceComputer = T::Distance;

    fn prune_accessor<'a>(
        &'a self,
        provider: &'a GarnetProvider<T>,
        context: &'a <GarnetProvider<T> as DataProvider>::Context,
    ) -> Result<Self::PruneAccessor<'a>, Self::PruneAccessorError> {
        Ok(FullAccessor::new(provider, context, false))
    }
}

impl<'a, T: VectorRepr> glue::AsElement<&'a [T]> for FullAccessor<'_, T> {
    type Error = GarnetProviderError;

    fn as_element(
        &mut self,
        vector: &'a [T],
        _id: Self::Id,
    ) -> impl Future<Output = Result<Self::Element<'_>, Self::Error>> + Send {
        future::ready(Ok(vector.to_vec()))
    }
}

impl<T: VectorRepr> InsertStrategy<GarnetProvider<T>, [T]> for FullPrecision {
    type PruneStrategy = Self;

    fn insert_search_accessor<'a>(
        &'a self,
        provider: &'a GarnetProvider<T>,
        context: &'a <GarnetProvider<T> as DataProvider>::Context,
    ) -> Result<Self::SearchAccessor<'a>, Self::SearchAccessorError> {
        Ok(FullAccessor::new(provider, context, false))
    }

    fn prune_strategy(&self) -> Self::PruneStrategy {
        *self
    }
}

impl<T: VectorRepr> InplaceDeleteStrategy<GarnetProvider<T>> for FullPrecision {
    type DeleteElement<'a> = [T];
    type DeleteElementGuard = Box<[T]>;
    type DeleteElementError = GarnetProviderError;

    type PruneStrategy = Self;
    type SearchStrategy = Internal<Self>;

    fn prune_strategy(&self) -> Self::PruneStrategy {
        Self
    }

    fn search_strategy(&self) -> Self::SearchStrategy {
        Internal(Self)
    }

    fn get_delete_element<'a>(
        &'a self,
        provider: &'a GarnetProvider<T>,
        context: &'a <GarnetProvider<T> as DataProvider>::Context,
        id: <GarnetProvider<T> as DataProvider>::InternalId,
    ) -> impl Future<Output = Result<Self::DeleteElementGuard, Self::DeleteElementError>> + Send
    {
        let mut v = vec![T::default(); provider.dim];
        if !provider.callbacks.read_single_iid(*context, id, &mut v) {
            return future::ready(Err(GarnetError::Read.into()));
        }
        future::ready(Ok(v.into()))
    }
}
