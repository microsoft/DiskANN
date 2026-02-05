/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::sync::RwLock;

use bytemuck::Pod;
use diskann::{
    ANNResult, ANNError,
    graph::{self, glue},
    provider,
    utils::{IntoUsize, VectorRepr},
};
use diskann_quantization::multi_vector::{Chamfer, Mat, MatRef, Standard};
use diskann_utils::future::AsyncFriendly;
use diskann_vector::{PureDistanceFunction, distance::Metric};

use crate::model::graph::provider::async_::{
    SimpleNeighborProviderAsync, common, inmem, postprocess::{self, DeletionCheck},
};

type MultiVec<T> = Mat<Standard<T>>;
type MultiVecRef<'a, T> = MatRef<'a, Standard<T>>;

pub type Provider<T, D = common::NoDeletes> =
    inmem::DefaultProvider<Store<T>, common::NoStore, D, provider::DefaultContext>;

const METRIC: Metric = Metric::InnerProduct;

// Precursor
#[derive(Debug, Clone, Copy)]
pub struct Precursor<T> {
    dim: usize,
    _marker: std::marker::PhantomData<T>,
}

impl<T> Precursor<T> {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<T> common::CreateVectorStore for Precursor<T>
where
    T: Pod + AsyncFriendly,
{
    type Target = Store<T>;
    fn create(
        self,
        max_points: usize,
        metric: Metric,
        _prefetch_lookahead: Option<usize>,
    ) -> Self::Target {
        assert_eq!(metric, METRIC, "Only inner-product is supported");
        Store::new(
            max_points,
            self.dim,
        )
    }
}

#[derive(Debug)]
pub struct Store<T>
where
    T: Pod,
{
    // Guards for the fast memory store.
    guards: Vec<RwLock<()>>,
    pooled: common::AlignedMemoryVectorStore<T>,
    multi: Vec<RwLock<Option<MultiVec<T>>>>,
}

impl<T> Store<T>
where
    T: Pod,
{
    fn new(max_points: usize,dim: usize) -> Self {
        Self {
            guards: (0..max_points).map(|_| RwLock::new(())).collect(),
            pooled: common::AlignedMemoryVectorStore::with_capacity(max_points, dim),
            multi: (0..max_points).map(|_| RwLock::new(None)).collect(),
        }
    }

    fn dim(&self) -> usize {
        self.pooled.dim()
    }
}

impl<T> common::VectorStore for Store<T>
where
    T: Pod + AsyncFriendly,
{
    fn total(&self) -> usize {
        self.guards.len()
    }

    fn count_for_get_vector(&self) -> usize {
        0
    }
}

impl<T, D> provider::SetElement<MultiVec<T>> for Provider<T, D>
where
    T: VectorRepr + MeanPool,
    D: AsyncFriendly,
{
    type SetError = ANNError;
    type Guard = provider::NoopGuard<u32>;

    fn set_element(
        &self,
        _context: &provider::DefaultContext,
        id: &u32,
        element: &MultiVec<T>,
    ) -> impl Future<Output = Result<Self::Guard, Self::SetError>> + Send {
        let mut buf = vec![T::default(); self.base_vectors.dim()];
        T::mean_pool(&mut buf, element.as_view());

        let store = &self.base_vectors;
        let i = id.into_usize();

        let _pooled_guard = store.guards[i].write().unwrap();
        let mut multi_guard = store.multi[i].write().unwrap();

        // SAFETY: We hold the write guard for this slot.
        unsafe {
            store.pooled.get_mut_slice(i).copy_from_slice(&buf);
        }

        *multi_guard = Some(element.clone());

        // Success.
        std::future::ready(Ok(provider::NoopGuard::new(*id)))
    }
}

#[derive(Debug)]
pub struct Accessor<'a, T, D>
where
    T: Pod,
{
    provider: &'a Provider<T, D>,
    buffer: Vec<T>,
}

impl<'a, T, D> Accessor<'a, T, D>
where
    T: Pod,
{
    fn new(provider: &'a Provider<T, D>) -> Self {
        let dim = provider.base_vectors.dim();
        Self {
            provider,
            buffer: vec![<T as bytemuck::Zeroable>::zeroed(); dim],
        }
    }

    fn store(&self) -> &Store<T> {
        &self.provider.base_vectors
    }
}

//////////////
// Provider //
//////////////

impl<T, D> provider::HasId for Accessor<'_, T, D>
where
    T: Pod,
{
    type Id = u32;
}

impl<'a, T, D> provider::DelegateNeighbor<'a> for Accessor<'_, T, D>
where
    T: Pod + AsyncFriendly,
    D: AsyncFriendly,
{
    type Delegate = &'a SimpleNeighborProviderAsync<u32>;

    fn delegate_neighbor(&'a mut self) -> Self::Delegate {
        self.provider.neighbors()
    }
}

/// This implementation of [`Accessor`] uses the mean-pooled versions of the vectors as
/// the primary data type for search.
///
/// Reranking is performed using the full multi-vectors.
impl<'a, T, D> provider::Accessor for Accessor<'a, T, D>
where
    T: VectorRepr,
    D: AsyncFriendly,
{
    type Extended = Box<[T]>;

    /// We share a reference to the local buffer to minimize the duration of the lock.
    type Element<'b>
        = &'b [T]
    where
        Self: 'b;

    type ElementRef<'b> = &'b [T];

    /// Choose to panic on an out-of-bounds access rather than propagate an error.
    type GetError = common::Panics;

    #[inline(always)]
    fn get_element(
        &mut self,
        id: u32,
    ) -> impl Future<Output = Result<Self::Element<'_>, Self::GetError>> + Send {
        // We cannot go through `Accessor::store` because we need the borrow checker to
        // recognize that the borrow of the provider is disjoint from the borrow of the
        // buffer.
        let store = &self.provider.base_vectors;

        let id = id.into_usize();
        let _guard = match store.guards.get(id) {
            Some(guard) => guard.read().unwrap(),
            None => panic!("Index {} is out-of-bounds", id),
        };

        // SAFETY: We hold the associated guard for this data slot, so read access is safe.
        self.buffer
            .copy_from_slice(unsafe { store.pooled.get_slice(id) });

        std::future::ready(Ok(&*self.buffer))
    }

    fn on_elements_unordered<Itr, F>(
        &mut self,
        itr: Itr,
        mut f: F,
    ) -> impl Future<Output = Result<(), Self::GetError>> + Send
    where
        Self: Sync,
        Itr: Iterator<Item = Self::Id> + Send,
        F: Send + for<'b> FnMut(Self::ElementRef<'b>, Self::Id),
    {
        let store = self.store();

        // We kind of just yolo it and assume that if `f` panics - we have bigger problems
        // to worry about.
        for id in itr {
            let i = id.into_usize();

            let _guard = store.guards[i].read().unwrap();
            f(unsafe { store.pooled.get_slice(i) }, id)
        }

        std::future::ready(Ok(()))
    }
}

impl<T, D> provider::BuildDistanceComputer for Accessor<'_, T, D>
where
    T: VectorRepr,
    D: AsyncFriendly,
{
    type DistanceComputerError = common::Panics;
    type DistanceComputer = T::Distance;

    fn build_distance_computer(
        &self,
    ) -> Result<Self::DistanceComputer, Self::DistanceComputerError> {
        Ok(T::distance(METRIC, Some(self.store().dim())))
    }
}

impl<T, D> provider::BuildQueryComputer<MultiVec<T>> for Accessor<'_, T, D>
where
    T: VectorRepr + MeanPool,
    D: AsyncFriendly,
{
    type QueryComputerError = common::Panics;
    type QueryComputer = T::QueryDistance;

    fn build_query_computer(
        &self,
        from: &MultiVec<T>,
    ) -> Result<Self::QueryComputer, Self::QueryComputerError> {
        // TODO: `build_query_computer` should recieve by `&mut`.
        let mut buf = vec![T::default(); self.store().dim()];
        T::mean_pool(&mut buf, from.as_view());
        Ok(T::query_distance(&buf, METRIC))
    }
}

pub trait MeanPool: Copy + Sized {
    fn mean_pool(dst: &mut [Self], x: MultiVecRef<'_, Self>);
}

impl MeanPool for f32 {
    fn mean_pool(dst: &mut [f32], x: MultiVecRef<'_, f32>) {
        dst.fill(0.0);
        x.rows().for_each(|r| {
            dst.iter_mut().zip(r.iter()).for_each(|(d, s)| *d += *s);
        });
        let scale = 1.0 / (x.num_vectors() as f32);
        dst.iter_mut().for_each(|d| *d *= scale);
    }
}


///////////////
// Reranking //
///////////////

impl<'a, T, D> postprocess::AsDeletionCheck for Accessor<'a, T, D>
where
    T: VectorRepr,
    D: AsyncFriendly + DeletionCheck,
{
    type Checker = D;
    fn as_deletion_check(&self) -> &D {
        &self.provider.deleted
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct ChamferRerank;

impl<T, D> glue::SearchPostProcess<Accessor<'_, T, D>, MultiVec<T>> for ChamferRerank
where
    T: VectorRepr + MeanPool,
    D: AsyncFriendly + DeletionCheck,
    for<'a, 'b> Chamfer: PureDistanceFunction<MultiVecRef<'a, T>, MultiVecRef<'a, T>>,
{
    type Error = common::Panics;

    fn post_process<I, B>(
        &self,
        accessor: &mut Accessor<'_, T, D>,
        query: &MultiVec<T>,
        _computer: &T::QueryDistance,
        candidates: I,
        output: &mut B,
    ) -> impl Future<Output = Result<usize, Self::Error>> + Send
    where
        I: Iterator<Item = diskann::neighbor::Neighbor<u32>>,
        B: graph::SearchOutputBuffer<u32> + ?Sized,
    {
        let checker = &accessor.provider.deleted;

        let mut reranked: Vec<(u32, f32)> = candidates
            .filter_map(|n| {
                if checker.deletion_check(n.id) {
                    None
                } else {
                    let multi = accessor.store().multi[n.id.into_usize()].read().unwrap();
                    // let tic = std::time::Instant::now();
                    let v = Chamfer::evaluate(query.as_view(), multi.as_ref().unwrap().as_view());
                    // println!("elapsed = {}", tic.elapsed().as_secs());
                    Some((n.id, v))
                }
            })
            .collect();

        reranked
            .sort_unstable_by(|a, b| (a.1).partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        std::future::ready(Ok(output.extend(reranked)))
    }
}

//////////
// Glue //
//////////

impl<T, D> glue::ExpandBeam<MultiVec<T>> for Accessor<'_, T, D>
where
    T: VectorRepr + MeanPool,
    D: AsyncFriendly,
{
}

impl<T, D> glue::FillSet for Accessor<'_, T, D>
where
    T: VectorRepr,
    D: AsyncFriendly,
{
}

impl<T, D> glue::SearchExt for Accessor<'_, T, D>
where
    T: VectorRepr,
    D: AsyncFriendly,
{
    fn starting_points(&self) -> impl Future<Output = ANNResult<Vec<u32>>> {
        std::future::ready(self.provider.starting_points())
    }
}

////////////////
// Strategies //
////////////////

#[derive(Debug, Clone, Copy)]
pub struct Strategy {}

impl Strategy {
    pub fn new() -> Self {
        Self {}
    }
}

impl<T, D> glue::SearchStrategy<Provider<T, D>, MultiVec<T>> for common::Internal<Strategy>
where
    T: VectorRepr + MeanPool,
    D: AsyncFriendly + DeletionCheck,
{
    type QueryComputer = T::QueryDistance;
    type SearchAccessor<'a> = Accessor<'a, T, D>;
    type SearchAccessorError = common::Panics;
    type PostProcessor = postprocess::RemoveDeletedIdsAndCopy;

    fn search_accessor<'a>(
        &'a self,
        provider: &'a Provider<T, D>,
        _context: &'a provider::DefaultContext,
    ) -> Result<Self::SearchAccessor<'a>, Self::SearchAccessorError> {
        Ok(Accessor::new(provider))
    }

    fn post_processor(&self) -> Self::PostProcessor {
        Default::default()
    }
}

impl<T, D> glue::SearchStrategy<Provider<T, D>, MultiVec<T>> for Strategy
where
    T: VectorRepr + MeanPool,
    D: AsyncFriendly + DeletionCheck,
    for<'a, 'b> Chamfer: PureDistanceFunction<MultiVecRef<'a, T>, MultiVecRef<'a, T>>,
{
    type QueryComputer = T::QueryDistance;
    type SearchAccessor<'a> = Accessor<'a, T, D>;
    type SearchAccessorError = common::Panics;
    type PostProcessor = glue::Pipeline<glue::FilterStartPoints, ChamferRerank>;

    fn search_accessor<'a>(
        &'a self,
        provider: &'a Provider<T, D>,
        _context: &'a provider::DefaultContext,
    ) -> Result<Self::SearchAccessor<'a>, Self::SearchAccessorError> {
        Ok(Accessor::new(provider))
    }

    fn post_processor(&self) -> Self::PostProcessor {
        Default::default()
    }
}

impl<T, D> glue::PruneStrategy<Provider<T, D>> for Strategy
where
    T: VectorRepr,
    D: AsyncFriendly,
{
    type DistanceComputer = T::Distance;
    type PruneAccessor<'a> = Accessor<'a, T, D>;
    type PruneAccessorError = diskann::error::Infallible;

    fn prune_accessor<'a>(
        &'a self,
        provider: &'a Provider<T, D>,
        _context: &'a provider::DefaultContext,
    ) -> Result<Self::PruneAccessor<'a>, Self::PruneAccessorError> {
        Ok(Accessor::new(provider))
    }
}

impl<T, D> glue::InsertStrategy<Provider<T, D>, MultiVec<T>> for Strategy
where
    T: VectorRepr + MeanPool,
    D: AsyncFriendly + DeletionCheck,
    for<'a, 'b> Chamfer: PureDistanceFunction<MultiVecRef<'a, T>, MultiVecRef<'a, T>>,
{
    type PruneStrategy = Self;
    fn prune_strategy(&self) -> Self::PruneStrategy {
        *self
    }
}

impl<T, D> glue::InplaceDeleteStrategy<Provider<T, D>> for Strategy
where
    T: VectorRepr + MeanPool,
    D: AsyncFriendly + DeletionCheck,
{
    type DeleteElementError = common::Panics;
    type DeleteElement<'a> = MultiVec<T>;
    type DeleteElementGuard = diskann_utils::reborrow::Place<MultiVec<T>>;
    type PruneStrategy = Self;
    type SearchStrategy = common::Internal<Self>;
    fn search_strategy(&self) -> Self::SearchStrategy {
        common::Internal(Self::new())
    }

    fn prune_strategy(&self) -> Self::PruneStrategy {
        Self::new()
    }

    async fn get_delete_element<'a>(
        &'a self,
        provider: &'a Provider<T, D>,
        _context: &'a provider::DefaultContext,
        id: u32,
    ) -> Result<Self::DeleteElementGuard, Self::DeleteElementError> {
        let id = id.into_usize();
        Ok(diskann_utils::reborrow::Place(provider.base_vectors.multi[id].read().unwrap().as_ref().unwrap().clone()))
    }
}
