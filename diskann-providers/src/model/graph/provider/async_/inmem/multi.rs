/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::sync::{Arc, RwLock};

use bytemuck::Pod;
use diskann::{
    ANNError, ANNResult,
    graph::{self, glue},
    provider,
    utils::{IntoUsize, VectorRepr},
};
use diskann_quantization::{
    alloc::{GlobalAllocator, ScopedAllocator},
    multi_vector::{Chamfer, Mat, MatRef, Standard},
    spherical,
};
use diskann_utils::future::AsyncFriendly;
use diskann_vector::{
    DistanceFunction, PreprocessedDistanceFunction, PureDistanceFunction, distance::Metric,
};
use rayon::prelude::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

use crate::{
    common::AlignedBoxWithSlice,
    model::graph::provider::async_::{
        SimpleNeighborProviderAsync, common, inmem,
        postprocess::{self, DeletionCheck},
    },
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
        Store::new(max_points, self.dim)
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
    fn new(max_points: usize, dim: usize) -> Self {
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

impl<T, D> provider::SetElement<MultiVecRef<'_, T>> for Provider<T, D>
where
    T: VectorRepr + MeanPool,
    D: AsyncFriendly,
{
    type SetError = ANNError;

    fn set_element(
        &self,
        _context: &provider::DefaultContext,
        id: &u32,
        element: MultiVecRef<'_, T>,
    ) -> impl Future<Output = Result<Self::Guard, Self::SetError>> + Send {
        let mut buf = vec![T::default(); self.base_vectors.dim()];
        T::mean_pool(&mut buf, element);

        let store = &self.base_vectors;
        let i = id.into_usize();

        let _pooled_guard = store.guards[i].write().unwrap();
        let mut multi_guard = store.multi[i].write().unwrap();

        // SAFETY: We hold the write guard for this slot.
        unsafe {
            store.pooled.get_mut_slice(i).copy_from_slice(&buf);
        }

        *multi_guard = Some(element.to_owned());

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

impl<T, D> provider::BuildQueryComputer<MultiVecRef<'_, T>> for Accessor<'_, T, D>
where
    T: VectorRepr + MeanPool,
    D: AsyncFriendly,
{
    type QueryComputerError = common::Panics;
    type QueryComputer = T::QueryDistance;

    fn build_query_computer(
        &self,
        from: MultiVecRef<'_, T>,
    ) -> Result<Self::QueryComputer, Self::QueryComputerError> {
        // TODO: `build_query_computer` should recieve by `&mut`.
        let mut buf = vec![T::default(); self.store().dim()];
        T::mean_pool(&mut buf, from);
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

impl<T, D> glue::SearchPostProcess<Accessor<'_, T, D>, MultiVecRef<'_, T>> for ChamferRerank
where
    T: VectorRepr + MeanPool,
    D: AsyncFriendly + DeletionCheck,
    for<'a, 'b> Chamfer: PureDistanceFunction<MultiVecRef<'a, T>, MultiVecRef<'a, T>>,
{
    type Error = common::Panics;

    fn post_process<I, B>(
        &self,
        accessor: &mut Accessor<'_, T, D>,
        query: MultiVecRef<'_, T>,
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
                    let v = Chamfer::evaluate(query, multi.as_ref().unwrap().as_view());
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

impl<T, D> glue::ExpandBeam<MultiVecRef<'_, T>> for Accessor<'_, T, D>
where
    T: VectorRepr + MeanPool,
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

impl<T, D> glue::DefaultPostProcessor<Provider<T, D>, MultiVecRef<'_, T>> for Strategy
where
    T: VectorRepr + MeanPool,
    D: AsyncFriendly + DeletionCheck,
    for<'a, 'b> Chamfer: PureDistanceFunction<MultiVecRef<'a, T>, MultiVecRef<'b, T>>,
{
    diskann::default_post_processor!(glue::Pipeline<glue::FilterStartPoints, ChamferRerank>);
}

impl<T, D> glue::SearchStrategy<Provider<T, D>, MultiVecRef<'_, T>> for Strategy
where
    T: VectorRepr + MeanPool,
    D: AsyncFriendly + DeletionCheck,
{
    type QueryComputer = T::QueryDistance;
    type SearchAccessor<'a> = Accessor<'a, T, D>;
    type SearchAccessorError = common::Panics;

    fn search_accessor<'a>(
        &'a self,
        provider: &'a Provider<T, D>,
        _context: &'a provider::DefaultContext,
    ) -> Result<Self::SearchAccessor<'a>, Self::SearchAccessorError> {
        Ok(Accessor::new(provider))
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
    type WorkingSet = graph::workingset::Map<u32, Box<[T]>>;

    fn prune_accessor<'a>(
        &'a self,
        provider: &'a Provider<T, D>,
        _context: &'a provider::DefaultContext,
    ) -> Result<Self::PruneAccessor<'a>, Self::PruneAccessorError> {
        Ok(Accessor::new(provider))
    }

    fn create_working_set(&self, capacity: usize) -> Self::WorkingSet {
        use diskann::graph::workingset::map::{Builder, Capacity};

        Builder::new(Capacity::Default).build(capacity)
    }
}

impl<T, D> glue::InsertStrategy<Provider<T, D>, MultiVecRef<'_, T>> for Strategy
where
    T: VectorRepr + MeanPool,
    D: AsyncFriendly + DeletionCheck,
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
    type DeleteElement<'a> = MultiVecRef<'a, T>;
    type DeleteElementGuard = MultiVec<T>;
    type DeleteSearchAccessor<'a> = Accessor<'a, T, D>;

    type PruneStrategy = Self;
    type SearchStrategy = Self;
    type SearchPostProcessor = postprocess::RemoveDeletedIdsAndCopy;

    fn search_strategy(&self) -> Self::SearchStrategy {
        Self::new()
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
        Ok(provider.base_vectors.multi[id]
            .read()
            .unwrap()
            .as_ref()
            .unwrap()
            .clone())
    }

    fn search_post_processor(&self) -> Self::SearchPostProcessor {
        Default::default()
    }
}

/////////////////////////////////////
// Spherical Chamfer Reranking     //
/////////////////////////////////////

/// Pre-built data for spherical-quantized chamfer distance computation.
///
/// Stores all compressed sub-vectors in one aligned byte buffer, with per-document
/// byte-offset/count metadata for efficient retrieval during reranking.
pub struct SphericalChamferData {
    data: AlignedBoxWithSlice<u8>,
    quantizer: Box<dyn spherical::iface::Quantizer + Send + Sync>,
    bytes_per_vector: usize,
    /// `offsets[doc_id]` = starting byte offset for this document's sub-vectors.
    offsets: Vec<usize>,
    /// `counts[doc_id]` = number of sub-vectors for this document.
    counts: Vec<usize>,
}

impl SphericalChamferData {
    /// Construct a new [`SphericalChamferData`] by compressing all sub-vectors from
    /// `multi_vecs` using the provided spherical `quantizer`.
    ///
    /// # Panics
    ///
    /// Panics if any sub-vector fails to compress.
    pub fn new<P>(quantizer: P, multi_vecs: &[MultiVec<f32>]) -> Self
    where
        P: spherical::iface::Quantizer + AsyncFriendly,
    {
        let total: usize = multi_vecs.iter().map(|m| m.num_vectors()).sum();
        let bytes_per_vector = quantizer.bytes();
        let total_bytes = total
            .checked_mul(bytes_per_vector)
            .expect("spherical chamfer data byte length overflow");
        let mut data = AlignedBoxWithSlice::new(total_bytes, 64)
            .expect("failed to allocate spherical chamfer data buffer");

        let mut offsets = Vec::with_capacity(multi_vecs.len());
        let mut counts = Vec::with_capacity(multi_vecs.len());
        let mut offset = 0usize;
        for m in multi_vecs {
            offsets.push(offset);
            counts.push(m.num_vectors());
            offset += m
                .num_vectors()
                .checked_mul(bytes_per_vector)
                .expect("spherical chamfer document byte length overflow");
        }

        let mut doc_slices = Vec::with_capacity(multi_vecs.len());
        let mut remaining = data.as_mut_slice();
        for &count in &counts {
            let doc_bytes = count
                .checked_mul(bytes_per_vector)
                .expect("spherical chamfer document byte length overflow");
            let (doc, rest) = remaining.split_at_mut(doc_bytes);
            doc_slices.push(doc);
            remaining = rest;
        }
        assert!(remaining.is_empty());

        doc_slices
            .into_par_iter()
            .zip(multi_vecs)
            .for_each(|(doc_data, m)| {
                let mut offset = 0;
                for row in m.as_view().rows() {
                    let next_offset = offset + bytes_per_vector;
                    quantizer
                        .compress(
                            row,
                            spherical::iface::OpaqueMut::new(&mut doc_data[offset..next_offset]),
                            ScopedAllocator::global(),
                        )
                        .unwrap();
                    offset = next_offset;
                }
            });

        Self {
            data,
            quantizer: Box::new(quantizer),
            bytes_per_vector,
            offsets,
            counts,
        }
    }

    fn get_vector(&self, offset: usize) -> spherical::iface::Opaque<'_> {
        let end = offset
            .checked_add(self.bytes_per_vector)
            .expect("byte buffer offset overflow");
        spherical::iface::Opaque::new(&self.data[offset..end])
    }

    fn doc_range(&self, doc_id: u32) -> Option<(usize, usize)> {
        let id = doc_id as usize;
        Some((*self.offsets.get(id)?, *self.counts.get(id)?))
    }

    fn distance_computer(
        &self,
    ) -> Result<spherical::iface::DistanceComputer, inmem::spherical::AllocatorError> {
        Ok(self.quantizer.distance_computer(GlobalAllocator)?)
    }

    fn query_computer<T>(
        &self,
        query: &[T],
        layout: spherical::iface::QueryLayout,
        allow_rescale: bool,
    ) -> Result<spherical::iface::QueryComputer, inmem::spherical::QueryComputerError>
    where
        T: VectorRepr,
    {
        let qf32 = T::as_f32(query).map_err(|e| {
            inmem::spherical::QueryComputerError::FullPrecisionConversionErr(Box::new(e))
        })?;

        Ok(self.quantizer.fused_query_computer(
            qf32.as_ref(),
            layout,
            allow_rescale,
            GlobalAllocator,
            ScopedAllocator::global(),
        )?)
    }

    /// Compute approximate Chamfer distance using pre-built query computers.
    ///
    /// For each query computer (one per query sub-vector), finds the minimum distance
    /// to any of the document's compressed sub-vectors, then sums those minima.
    fn chamfer_distance(
        &self,
        query_computers: &[spherical::iface::QueryComputer],
        doc_id: u32,
    ) -> f32 {
        let Some((offset, count)) = self.doc_range(doc_id) else {
            // Unknown ID (e.g., frozen start point) - return worst-case distance.
            return f32::MAX;
        };

        let mut sum = 0.0f32;
        for computer in query_computers {
            let mut min_dist = f32::MAX;
            for j in 0..count {
                let doc_vec = self.get_vector(offset + j * self.bytes_per_vector);
                let dist = computer.evaluate_similarity(doc_vec).unwrap();
                min_dist = min_dist.min(dist);
            }
            sum += min_dist;
        }
        sum
    }
}

impl std::fmt::Debug for SphericalChamferData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SphericalChamferData")
            .field("num_docs", &self.offsets.len())
            .field("bytes_per_vector", &self.bytes_per_vector)
            .field(
                "total_sub_vectors",
                &self.counts.iter().copied().sum::<usize>(),
            )
            .field("data_bytes", &self.data.len())
            .finish()
    }
}

/// A [`SearchPostProcess`] that computes approximate Chamfer distance using
/// spherical-quantized sub-vectors from a side-loaded [`SphericalChamferData`].
///
/// The query used for reranking can differ from the first-stage query (e.g., a
/// full multi-vector set vs. a reduced set).
pub struct QuantizedChamferRerank {
    data: Arc<SphericalChamferData>,
    /// Pre-built query computers, one per query sub-vector.
    query_computers: Vec<spherical::iface::QueryComputer>,
}

impl QuantizedChamferRerank {
    /// Construct a new [`QuantizedChamferRerank`].
    ///
    /// Builds a [`spherical::iface::QueryComputer`] for each sub-vector in
    /// `rerank_query` using the specified `layout`.
    pub fn new(
        data: Arc<SphericalChamferData>,
        rerank_query: MultiVecRef<'_, f32>,
        layout: spherical::iface::QueryLayout,
    ) -> Self {
        let query_computers: Vec<_> = rerank_query
            .rows()
            .map(|q_vec| {
                data.query_computer(q_vec, layout, true)
                    .expect("failed to build query computer for sub-vector")
            })
            .collect();

        Self {
            data,
            query_computers,
        }
    }
}

impl<T, D> glue::SearchPostProcess<Accessor<'_, T, D>, MultiVecRef<'_, T>>
    for QuantizedChamferRerank
where
    T: VectorRepr + MeanPool,
    D: AsyncFriendly + DeletionCheck,
{
    type Error = common::Panics;

    fn post_process<I, B>(
        &self,
        accessor: &mut Accessor<'_, T, D>,
        _query: MultiVecRef<'_, T>,
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
                    let dist = self.data.chamfer_distance(&self.query_computers, n.id);
                    Some((n.id, dist))
                }
            })
            .collect();

        reranked
            .sort_unstable_by(|a, b| (a.1).partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        std::future::ready(Ok(output.extend(reranked)))
    }
}

/// A [`SearchPostProcess`] that computes full-precision Chamfer distance using
/// side-loaded multi-vectors that may differ from those used during index construction.
///
/// This enables reranking with a different (e.g., larger) multi-vector set than
/// what was used for building the graph.
pub struct SideloadedChamferRerank<T: Copy> {
    data: Arc<[MultiVec<T>]>,
    rerank_query: MultiVec<T>,
}

impl<T: Copy> SideloadedChamferRerank<T> {
    /// Construct a new [`SideloadedChamferRerank`].
    ///
    /// `data` contains the multi-vectors for all documents (indexed by document ID).
    /// `rerank_query` is the query multi-vector to use for computing Chamfer distance.
    pub fn new(data: Arc<[MultiVec<T>]>, rerank_query: MultiVec<T>) -> Self {
        Self { data, rerank_query }
    }
}

impl<T, D> glue::SearchPostProcess<Accessor<'_, T, D>, MultiVecRef<'_, T>>
    for SideloadedChamferRerank<T>
where
    T: VectorRepr + MeanPool,
    D: AsyncFriendly + DeletionCheck,
    for<'a, 'b> Chamfer: PureDistanceFunction<MultiVecRef<'a, T>, MultiVecRef<'b, T>>,
{
    type Error = common::Panics;

    fn post_process<I, B>(
        &self,
        accessor: &mut Accessor<'_, T, D>,
        _query: MultiVecRef<'_, T>,
        _computer: &T::QueryDistance,
        candidates: I,
        output: &mut B,
    ) -> impl Future<Output = Result<usize, Self::Error>> + Send
    where
        I: Iterator<Item = diskann::neighbor::Neighbor<u32>>,
        B: graph::SearchOutputBuffer<u32> + ?Sized,
    {
        let checker = &accessor.provider.deleted;
        let rerank_query = self.rerank_query.as_view();

        let mut reranked: Vec<(u32, f32)> = candidates
            .filter_map(|n| {
                if checker.deletion_check(n.id) {
                    None
                } else {
                    let doc = self.data[n.id.into_usize()].as_view();
                    let v = Chamfer::evaluate(rerank_query, doc);
                    Some((n.id, v))
                }
            })
            .collect();

        reranked
            .sort_unstable_by(|a, b| (a.1).partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        std::future::ready(Ok(output.extend(reranked)))
    }
}

///////////////////////////////////////////
// Spherical Chamfer Inner-Loop Strategy //
///////////////////////////////////////////

/// A [`SearchStrategy`] that uses spherical-quantized Chamfer distance for
/// first-stage graph traversal.
///
/// During beam search, distances are computed as approximate Chamfer distance
/// over compressed sub-vectors (using [`SphericalChamferData`]) rather than
/// mean-pooled inner product.
///
/// This strategy is **search-only** — it does not support index construction
/// (no `InsertStrategy` or `PruneStrategy`).
pub struct SphericalChamferSearch<D = common::NoDeletes> {
    data: Arc<SphericalChamferData>,
    layout: spherical::iface::QueryLayout,
    _deleted: std::marker::PhantomData<D>,
}

impl<D> Clone for SphericalChamferSearch<D> {
    fn clone(&self) -> Self {
        Self {
            data: Arc::clone(&self.data),
            layout: self.layout,
            _deleted: std::marker::PhantomData,
        }
    }
}

impl<D> SphericalChamferSearch<D> {
    /// Construct a new [`SphericalChamferSearch`] strategy.
    pub fn new(data: Arc<SphericalChamferData>, layout: spherical::iface::QueryLayout) -> Self {
        Self {
            data,
            layout,
            _deleted: std::marker::PhantomData,
        }
    }
}

/// Accessor for spherical-Chamfer inner-loop search.
///
/// Elements are document IDs (`u32`). Distance computation is delegated to
/// [`SphericalChamferData`] via the query/distance computers.
pub struct SphericalChamferAccessor<'a, T, D>
where
    T: Pod,
{
    provider: &'a Provider<T, D>,
    data: Arc<SphericalChamferData>,
    layout: spherical::iface::QueryLayout,
}

impl<T, D> provider::HasId for SphericalChamferAccessor<'_, T, D>
where
    T: Pod,
{
    type Id = u32;
}

impl<'a, T, D> provider::DelegateNeighbor<'a> for SphericalChamferAccessor<'_, T, D>
where
    T: Pod + AsyncFriendly,
    D: AsyncFriendly,
{
    type Delegate = &'a SimpleNeighborProviderAsync<u32>;

    fn delegate_neighbor(&'a mut self) -> Self::Delegate {
        self.provider.neighbors()
    }
}

impl<T, D> provider::Accessor for SphericalChamferAccessor<'_, T, D>
where
    T: VectorRepr,
    D: AsyncFriendly,
{
    /// Elements are document IDs — distance computation happens inside the computers.
    type Element<'b>
        = u32
    where
        Self: 'b;

    type ElementRef<'b> = u32;

    type GetError = common::Panics;

    #[inline(always)]
    fn get_element(
        &mut self,
        id: u32,
    ) -> impl Future<Output = Result<Self::Element<'_>, Self::GetError>> + Send {
        std::future::ready(Ok(id))
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
        for id in itr {
            f(id, id);
        }
        std::future::ready(Ok(()))
    }
}

/// Query computer for spherical-Chamfer distance.
///
/// Pre-builds one [`spherical::iface::QueryComputer`] per query sub-vector, then
/// computes Chamfer distance (sum of per-query-vector minima) against each candidate's
/// compressed sub-vectors.
pub struct ChamferQueryComputer {
    data: Arc<SphericalChamferData>,
    query_computers: Vec<spherical::iface::QueryComputer>,
}

impl PreprocessedDistanceFunction<u32, f32> for ChamferQueryComputer {
    #[inline]
    fn evaluate_similarity(&self, doc_id: u32) -> f32 {
        self.data.chamfer_distance(&self.query_computers, doc_id)
    }
}

/// Symmetric distance computer for spherical-Chamfer distance between two documents.
///
/// This is used during pruning. For each sub-vector in document A, finds the minimum
/// distance to any sub-vector in document B, then sums.
pub struct ChamferDistanceComputer {
    data: Arc<SphericalChamferData>,
    distance_computer: spherical::iface::DistanceComputer,
}

impl DistanceFunction<u32, u32, f32> for ChamferDistanceComputer {
    fn evaluate_similarity(&self, a: u32, b: u32) -> f32 {
        let Some((a_offset, a_count)) = self.data.doc_range(a) else {
            return f32::MAX;
        };
        let Some((b_offset, b_count)) = self.data.doc_range(b) else {
            return f32::MAX;
        };

        let mut sum = 0.0f32;
        for i in 0..a_count {
            let a_vec = self
                .data
                .get_vector(a_offset + i * self.data.bytes_per_vector);
            let mut min_dist = f32::MAX;
            for j in 0..b_count {
                let b_vec = self
                    .data
                    .get_vector(b_offset + j * self.data.bytes_per_vector);
                let dist = self
                    .distance_computer
                    .evaluate_similarity(a_vec, b_vec)
                    .unwrap();
                min_dist = min_dist.min(dist);
            }
            sum += min_dist;
        }
        sum
    }
}

impl<T, D> provider::BuildQueryComputer<MultiVecRef<'_, T>> for SphericalChamferAccessor<'_, T, D>
where
    T: VectorRepr + MeanPool,
    D: AsyncFriendly,
{
    type QueryComputerError = ANNError;
    type QueryComputer = ChamferQueryComputer;

    fn build_query_computer(
        &self,
        from: MultiVecRef<'_, T>,
    ) -> Result<Self::QueryComputer, Self::QueryComputerError> {
        let query_computers: Vec<_> = from
            .rows()
            .map(|q_vec| {
                let qf32 = T::as_f32(q_vec).map_err(|e| {
                    ANNError::new(
                        diskann::ANNErrorKind::SQError,
                        crate::utils::Bridge(
                            inmem::spherical::QueryComputerError::FullPrecisionConversionErr(
                                Box::new(e),
                            ),
                        ),
                    )
                })?;
                self.data
                    .query_computer(qf32.as_ref(), self.layout, true)
                    .map_err(|e| {
                        ANNError::new(diskann::ANNErrorKind::SQError, crate::utils::Bridge(e))
                    })
            })
            .collect::<Result<_, _>>()?;

        Ok(ChamferQueryComputer {
            data: Arc::clone(&self.data),
            query_computers,
        })
    }
}

impl<T, D> provider::BuildDistanceComputer for SphericalChamferAccessor<'_, T, D>
where
    T: VectorRepr,
    D: AsyncFriendly,
{
    type DistanceComputerError = inmem::spherical::AllocatorError;
    type DistanceComputer = ChamferDistanceComputer;

    fn build_distance_computer(
        &self,
    ) -> Result<Self::DistanceComputer, Self::DistanceComputerError> {
        Ok(ChamferDistanceComputer {
            data: Arc::clone(&self.data),
            distance_computer: self.data.distance_computer()?,
        })
    }
}

impl<T, D> glue::ExpandBeam<MultiVecRef<'_, T>> for SphericalChamferAccessor<'_, T, D>
where
    T: VectorRepr + MeanPool,
    D: AsyncFriendly,
{
}

impl<T, D> glue::SearchExt for SphericalChamferAccessor<'_, T, D>
where
    T: VectorRepr,
    D: AsyncFriendly,
{
    fn starting_points(&self) -> impl Future<Output = ANNResult<Vec<u32>>> {
        std::future::ready(self.provider.starting_points())
    }
}

impl<T, D> glue::SearchStrategy<Provider<T, D>, MultiVecRef<'_, T>> for SphericalChamferSearch<D>
where
    T: VectorRepr + MeanPool,
    D: AsyncFriendly + DeletionCheck,
{
    type QueryComputer = ChamferQueryComputer;
    type SearchAccessor<'a> = SphericalChamferAccessor<'a, T, D>;
    type SearchAccessorError = common::Panics;

    fn search_accessor<'a>(
        &'a self,
        provider: &'a Provider<T, D>,
        _context: &'a provider::DefaultContext,
    ) -> Result<Self::SearchAccessor<'a>, Self::SearchAccessorError> {
        Ok(SphericalChamferAccessor {
            provider,
            data: Arc::clone(&self.data),
            layout: self.layout,
        })
    }
}

// PassThrough working set: no caching, just forward to the accessor.
impl<T, D> graph::workingset::Fill<inmem::PassThrough> for SphericalChamferAccessor<'_, T, D>
where
    T: VectorRepr,
    D: AsyncFriendly,
{
    type Error = std::convert::Infallible;
    type View<'a>
        = &'a Self
    where
        Self: 'a;

    async fn fill<'a, Itr>(
        &'a mut self,
        _state: &'a mut inmem::PassThrough,
        _itr: Itr,
    ) -> Result<Self::View<'a>, Self::Error>
    where
        Itr: ExactSizeIterator<Item = Self::Id> + Clone + Send + Sync,
        Self: 'a,
    {
        Ok(self)
    }
}

// View returns u32 IDs directly — distance computation is handled by ChamferDistanceComputer.
impl<T, D> graph::workingset::View<u32> for &SphericalChamferAccessor<'_, T, D>
where
    T: VectorRepr,
    D: AsyncFriendly,
{
    type ElementRef<'a> = u32;
    type Element<'a>
        = u32
    where
        Self: 'a;

    fn get(&self, id: u32) -> Option<Self::Element<'_>> {
        Some(id)
    }
}

impl<T, D> glue::PruneStrategy<Provider<T, D>> for SphericalChamferSearch<D>
where
    T: VectorRepr + MeanPool,
    D: AsyncFriendly + DeletionCheck,
{
    type DistanceComputer = ChamferDistanceComputer;
    type PruneAccessor<'a> = SphericalChamferAccessor<'a, T, D>;
    type PruneAccessorError = common::Panics;
    type WorkingSet = inmem::PassThrough;

    fn prune_accessor<'a>(
        &'a self,
        provider: &'a Provider<T, D>,
        _context: &'a provider::DefaultContext,
    ) -> Result<Self::PruneAccessor<'a>, Self::PruneAccessorError> {
        Ok(SphericalChamferAccessor {
            provider,
            data: Arc::clone(&self.data),
            layout: self.layout,
        })
    }

    fn create_working_set(&self, _capacity: usize) -> Self::WorkingSet {
        inmem::PassThrough
    }
}

impl<T, D> glue::InsertStrategy<Provider<T, D>, MultiVecRef<'_, T>> for SphericalChamferSearch<D>
where
    T: VectorRepr + MeanPool,
    D: AsyncFriendly + DeletionCheck,
{
    type PruneStrategy = Self;
    fn prune_strategy(&self) -> Self::PruneStrategy {
        Self::new(Arc::clone(&self.data), self.layout)
    }
}

impl<'a, T, D> postprocess::AsDeletionCheck for SphericalChamferAccessor<'a, T, D>
where
    T: VectorRepr,
    D: AsyncFriendly + DeletionCheck,
{
    type Checker = D;
    fn as_deletion_check(&self) -> &D {
        &self.provider.deleted
    }
}

impl<T, D> glue::SearchPostProcess<SphericalChamferAccessor<'_, T, D>, MultiVecRef<'_, T>>
    for QuantizedChamferRerank
where
    T: VectorRepr + MeanPool,
    D: AsyncFriendly + DeletionCheck,
{
    type Error = common::Panics;

    fn post_process<I, B>(
        &self,
        accessor: &mut SphericalChamferAccessor<'_, T, D>,
        _query: MultiVecRef<'_, T>,
        _computer: &ChamferQueryComputer,
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
                    let dist = self.data.chamfer_distance(&self.query_computers, n.id);
                    Some((n.id, dist))
                }
            })
            .collect();

        reranked
            .sort_unstable_by(|a, b| (a.1).partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        std::future::ready(Ok(output.extend(reranked)))
    }
}

impl<T, D> glue::SearchPostProcess<SphericalChamferAccessor<'_, T, D>, MultiVecRef<'_, T>>
    for SideloadedChamferRerank<T>
where
    T: VectorRepr + MeanPool,
    D: AsyncFriendly + DeletionCheck,
    for<'a, 'b> Chamfer: PureDistanceFunction<MultiVecRef<'a, T>, MultiVecRef<'b, T>>,
{
    type Error = common::Panics;

    fn post_process<I, B>(
        &self,
        accessor: &mut SphericalChamferAccessor<'_, T, D>,
        _query: MultiVecRef<'_, T>,
        _computer: &ChamferQueryComputer,
        candidates: I,
        output: &mut B,
    ) -> impl Future<Output = Result<usize, Self::Error>> + Send
    where
        I: Iterator<Item = diskann::neighbor::Neighbor<u32>>,
        B: graph::SearchOutputBuffer<u32> + ?Sized,
    {
        let checker = &accessor.provider.deleted;
        let rerank_query = self.rerank_query.as_view();

        let mut reranked: Vec<(u32, f32)> = candidates
            .filter_map(|n| {
                if checker.deletion_check(n.id) {
                    None
                } else {
                    let doc = self.data[n.id.into_usize()].as_view();
                    let v = Chamfer::evaluate(rerank_query, doc);
                    Some((n.id, v))
                }
            })
            .collect();

        reranked
            .sort_unstable_by(|a, b| (a.1).partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        std::future::ready(Ok(output.extend(reranked)))
    }
}
