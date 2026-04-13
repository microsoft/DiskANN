/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use dashmap::DashMap;
use diskann::{
    ANNError, ANNErrorKind, ANNResult, default_post_processor,
    graph::{
        AdjacencyList, SearchOutputBuffer,
        config::defaults::MAX_OCCLUSION_SIZE,
        glue::{
            self, DefaultPostProcessor, ExpandBeam, InplaceDeleteStrategy, InsertStrategy,
            PruneStrategy, SearchExt, SearchPostProcess, SearchPostProcessStep, SearchStrategy,
        },
        workingset::{self, map::Entry},
    },
    neighbor::Neighbor,
    provider::{
        Accessor, BuildDistanceComputer, BuildQueryComputer, DataProvider, DelegateNeighbor,
        Delete, ElementStatus, HasId, NeighborAccessor, NeighborAccessorMut, NoopGuard, SetElement,
    },
    utils::VectorRepr,
};
use diskann_quantization::alloc::{AllocatorError, Poly};
use diskann_utils::object_pool::{AsPooled, ObjectPool, PooledRef, Undef};
use diskann_utils::{Reborrow, views::Matrix};
use diskann_vector::{
    DistanceFunction, PreprocessedDistanceFunction, contains::ContainsSimd, distance::Metric,
};
use std::{
    any::TypeId,
    future,
    marker::PhantomData,
    mem,
    ops::{Deref, DerefMut},
    sync::atomic::{AtomicBool, AtomicU64, Ordering},
    time::SystemTime,
};
use thiserror::Error;

use crate::{
    VectorQuantType,
    alloc::AlignToEight,
    fsm::{FreeSpaceMap, FsmError},
    garnet::{Callbacks, Context, GarnetError, GarnetId, Term},
    quantization::{
        self, DynDistanceComputer, DynQueryComputer, GarnetQuantizer, GarnetQuantizerError,
    },
};

thread_local! {
    /// Thread local flag to detect when we've reached the quantization threshold. This is needed
    /// to return the correct status at the end of insert() since the provider code that becomes
    /// aware of the state change cannot directly communicate it back to the caller.
    pub static QUANTIZER_READY: AtomicBool = const { AtomicBool::new(false) };
}

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

/// A type erased vector, properly aligned so that it can hold any size element
/// (up to 8 bytes long).
pub struct DynVector<T: VectorRepr> {
    inner: Poly<[u8], AlignToEight>,
    ty: PhantomData<T>,
}

impl<T: VectorRepr> DynVector<T> {
    fn new(inner: Poly<[u8], AlignToEight>) -> Self {
        Self {
            inner,
            ty: PhantomData,
        }
    }
}

impl<T: VectorRepr> Deref for DynVector<T> {
    type Target = Poly<[u8], AlignToEight>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<T: VectorRepr> DerefMut for DynVector<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<'a, T: VectorRepr> Reborrow<'a> for DynVector<T> {
    type Target = &'a [u8];

    fn reborrow(&'a self) -> Self::Target {
        &self.inner
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
    #[error("Allocation failed")]
    AllocFailed(#[from] AllocatorError),
    #[error("Invalid quantizer for vector data")]
    InvalidQuantizer,
    #[error("Expected quantizer is missing")]
    MissingQuantizer,
    #[error("Failed to gather quantizer training data")]
    MissingTrainingData,
    #[error("Quantizer error: {0}")]
    Quantizer(#[from] GarnetQuantizerError),
    #[error("Post processing error: {0}")]
    PostProcessing(Box<dyn std::error::Error + Send + Sync + 'static>),
}

impl From<GarnetProviderError> for ANNError {
    #[track_caller]
    fn from(value: GarnetProviderError) -> Self {
        ANNError::new(ANNErrorKind::GetVertexDataError, value)
    }
}

diskann::always_escalate!(GarnetProviderError);

/// The Garnet DataProvider implementation.
pub struct GarnetProvider<T: VectorRepr> {
    dim: usize,
    metric_type: Metric,
    max_degree: usize,
    callbacks: Callbacks,
    /// The quantizer the index will use, or None if NOQUANT is used.
    quantizer: Option<Box<dyn GarnetQuantizer>>,
    /// Tracks whether quantization backfill is complete.
    all_quantized: AtomicBool,
    /// Tracks whether training has already started.
    training_started: AtomicU64,
    id_buffer_pool: ObjectPool<AdjList>,
    filtered_ids_pool: ObjectPool<Vec<u32>>,
    quant_buffer_pool: ObjectPool<Vec<u8>>,
    neighbor_cache: DashMap<u32, Vec<u32>, foldhash::fast::RandomState>,
    start_point_cache: DashMap<u32, Poly<[u8], AlignToEight>, foldhash::fast::RandomState>,
    start_point_quant_cache: DashMap<u32, Poly<[u8], AlignToEight>, foldhash::fast::RandomState>,
    fsm: FreeSpaceMap,
    _phantom: PhantomData<T>,
}

impl<T: VectorRepr> GarnetProvider<T> {
    pub fn new(
        dim: usize,
        quant_type: VectorQuantType,
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
        let start_point_quant_cache =
            DashMap::with_capacity_and_hasher(1, foldhash::fast::RandomState::default());
        let neighbor_cache =
            DashMap::with_capacity_and_hasher(1, foldhash::fast::RandomState::default());

        // Try to read the start point from Garnet
        let mut v = Poly::broadcast(0u8, dim * mem::size_of::<T>(), AlignToEight)?;
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

        let (quantizer, canonical_bytes, all_quantized) = match quant_type {
            VectorQuantType::NoQuant | VectorQuantType::XPreQ8 => (None, 0, false),
            VectorQuantType::Invalid => return Err(GarnetProviderError::InvalidQuantizer),
            VectorQuantType::Q8 => {
                if TypeId::of::<T>() != TypeId::of::<f32>() {
                    return Err(GarnetProviderError::InvalidQuantizer);
                }

                let quantizer = Box::new(quantization::MinMax8Bit::new(dim, metric_type)?)
                    as Box<dyn GarnetQuantizer>;
                let canonical_bytes = quantizer.canonical_bytes();
                // NOTE: Q8 needs no training, so it always starts with backfill complete.
                (Some(quantizer), canonical_bytes, true)
            }
            VectorQuantType::Bin => {
                if TypeId::of::<T>() != TypeId::of::<f32>() {
                    return Err(GarnetProviderError::InvalidQuantizer);
                }

                let quantizer =
                    Box::new(quantization::Spherical1Bit::new(dim)) as Box<dyn GarnetQuantizer>;
                let canonical_bytes = quantizer.canonical_bytes();
                (Some(quantizer), canonical_bytes, false)
            }
        };
        let quant_buffer_pool =
            ObjectPool::new(Undef::new(canonical_bytes), parallelism, Some(parallelism));

        Ok(Self {
            dim,
            metric_type,
            max_degree,
            callbacks,
            quantizer,
            all_quantized: AtomicBool::new(all_quantized),
            training_started: AtomicU64::new(0),
            id_buffer_pool,
            filtered_ids_pool,
            quant_buffer_pool,
            start_point_cache,
            start_point_quant_cache,
            neighbor_cache,
            fsm,
            _phantom: PhantomData,
        })
    }

    pub fn maybe_set_start_point(
        &self,
        context: &Context,
        point: &[T],
    ) -> Result<(), GarnetProviderError> {
        let mut v = Poly::broadcast(0u8, self.dim * mem::size_of::<T>(), AlignToEight)?;
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

            if self.is_quantized()
                && let Some(quantizer) = self.quantizer()
            {
                // We are already able to quantize, so store the quantized start point

                let mut qpoint = vec![0u8; quantizer.canonical_bytes()];
                quantizer.compress(bytemuck::cast_slice::<T, f32>(point), &mut qpoint)?;

                if !self
                    .callbacks
                    .write_iid(context.term(Term::Quantized), 0, &qpoint)
                {
                    return Err(GarnetError::Write.into());
                }

                self.start_point_quant_cache
                    .insert(0, Poly::from_iter(qpoint.iter().copied(), AlignToEight)?);
            }

            if !self
                .callbacks
                .write_iid(context.term(Term::Neighbors), 0, &neighbors)
            {
                return Err(GarnetError::Write.into());
            }

            self.start_point_cache.insert(
                0,
                Poly::from_iter(
                    bytemuck::cast_slice::<T, u8>(point).iter().copied(),
                    AlignToEight,
                )?,
            );
            self.neighbor_cache
                .insert(0, Vec::with_capacity(self.max_degree + 1));
        }

        Ok(())
    }

    pub fn start_points_exist(&self) -> bool {
        self.start_point_cache.get(&0).is_some() && self.neighbor_cache.get(&0).is_some()
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

    /// Train the quantizer.
    ///
    /// This should only be called when at least `quantizer.required_vectors()` vectors exist in
    /// the provider. This will build quantization tables, but does not quantize any vectors.
    pub fn train_quantizer(&self, context: &Context) -> bool {
        // Collect up to `self.quantizer.required_vectors()` vectors and use them for quantizer training.

        let current_time: u64 =
            match SystemTime::now().duration_since(std::time::SystemTime::UNIX_EPOCH) {
                Ok(t) => {
                    let t = t.as_millis().try_into().unwrap_or(0);
                    if t == 0 {
                        return false;
                    }
                    t
                }
                Err(_) => return false,
            };

        // Ensure we don't kick off training twice.
        match self.training_started.compare_exchange(
            0,
            current_time,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => (),
            Err(_) => return false,
        }

        let quantizer = match &self.quantizer {
            Some(q) => q,
            None => {
                self.training_started.store(0, Ordering::Release);
                return false;
            }
        };

        debug_assert_eq!(std::any::TypeId::of::<T>(), std::any::TypeId::of::<f32>());

        let rows = quantizer.required_vectors();
        let mut data = Matrix::new(0f32, rows, self.dim);
        let mut row_idx = 0usize;

        if self
            .fsm
            .visit_used(*context, |id| {
                // Skip the start point.
                if id == 0 {
                    return true;
                }

                if row_idx >= rows {
                    return false;
                }

                // Read the vector into the data matrix.
                // Note that it's ok to read f32 instead of T here, because this can only get called when T == f32.
                let row = data.row_mut(row_idx);
                if !self
                    .callbacks
                    .read_single_iid(context.term(Term::Vector), id, row)
                {
                    return false;
                }

                row_idx += 1;

                true
            })
            .is_err()
        {
            // Training failed
            self.training_started.store(0, Ordering::Release);
            return false;
        }

        if row_idx < quantizer.required_vectors() {
            self.training_started.store(0, Ordering::Release);
            return false;
        }

        let view = if let Some(view) = data.subview(0..row_idx) {
            view
        } else {
            return false;
        };

        // Train the quantizer.
        match quantizer.train(self.metric_type, view) {
            Ok(()) => true,
            Err(_e) => {
                self.training_started.store(0, Ordering::Release);
                false
            }
        }
    }

    /// Bulk quantize previously inserted vectors.
    ///
    /// This function will be invoked on multiple threads. The total number of tasks and the ID of
    /// the current task are given as inputs.
    pub fn backfill_quant_vectors(&self, context: &Context, task_idx: usize, task_count: usize) {
        let quantizer = match &self.quantizer {
            Some(q) => q,
            None => return,
        };

        let max_id = self.fsm.max_id() as usize;

        let task_count = task_count.min(max_id + 1);
        if task_idx >= task_count {
            return;
        }

        let work_count = (max_id + 1) / task_count; // will be >= 1
        let start_id = (work_count * task_idx) as u32;
        let end_id = (work_count * (task_idx + 1)).min(max_id + 1) as u32;

        self.fsm.lock_reuse();

        let mut v = vec![0f32; self.dim];
        let mut q = vec![0u8; quantizer.canonical_bytes()];
        for id in start_id..end_id {
            if !self
                .callbacks
                .read_single_iid(context.term(Term::Vector), id, &mut v)
            {
                continue;
            }

            if quantizer.compress(&v, &mut q).is_err() {
                continue;
            };

            if !self
                .callbacks
                .write_iid(context.term(Term::Quantized), id, &q)
            {
                continue;
            }
        }

        let backfill_finished = self.fsm.unlock_reuse();

        if backfill_finished {
            // Finish by quantizing the start points
            if let Some(v) = self.start_point_cache.get(&0)
                && quantizer
                    .compress(bytemuck::cast_slice::<u8, f32>(&v), &mut q)
                    .is_ok()
            {
                let _ = self
                    .callbacks
                    .write_iid(context.term(Term::Quantized), 0, &q);

                // set the cache
                let point = if let Ok(p) = Poly::from_iter(q.iter().copied(), AlignToEight) {
                    p
                } else {
                    return;
                };
                self.start_point_quant_cache.insert(0, point);
            }

            self.all_quantized.store(true, Ordering::Release);
        }
    }

    /// Returns the quantizer associated with the index.
    fn quantizer(&self) -> Option<&dyn GarnetQuantizer> {
        if let Some(quantizer) = &self.quantizer {
            return Some(&**quantizer as &dyn GarnetQuantizer);
        }

        None
    }

    /// Returns quantization status. If this is true, the index is operating fully quantized.
    pub fn is_quantized(&self) -> bool {
        self.quantizer.is_some() && self.all_quantized.load(Ordering::Acquire)
    }

    pub(crate) fn get_full_vector(
        &self,
        context: &Context,
        iid: u32,
    ) -> Result<Vec<T>, GarnetProviderError> {
        let mut v = vec![T::default(); self.dim];

        if iid == 0 {
            let guard = if let Some(r) = self.start_point_cache.get(&iid) {
                r
            } else {
                return Err(GarnetError::Read.into());
            };
            v.copy_from_slice(bytemuck::cast_slice::<u8, T>(&guard));
            return Ok(v);
        }

        if !self
            .callbacks
            .read_single_iid(context.term(Term::Vector), iid, &mut v)
        {
            return Err(GarnetError::Read.into());
        }

        Ok(v)
    }
}

impl<T: VectorRepr> DataProvider for GarnetProvider<T> {
    type Context = Context;
    type InternalId = u32;
    type ExternalId = GarnetId;
    type Error = GarnetProviderError;
    type Guard = NoopGuard<u32>;

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

impl<T: VectorRepr> SetElement<&[T]> for GarnetProvider<T> {
    type SetError = GarnetProviderError;

    async fn set_element(
        &self,
        context: &Self::Context,
        id: &Self::ExternalId,
        element: &[T],
    ) -> Result<Self::Guard, Self::SetError> {
        let internal_id = self.fsm.next_id(*context)?;

        // Set quantization readiness
        if let Some(quantizer) = &self.quantizer
            && !quantizer.is_prepared()
            && self.fsm.total_used() > quantizer.required_vectors()
        {
            QUANTIZER_READY.with(|v| v.store(true, Ordering::Release));
        }

        let insert = || -> Result<(), Self::SetError> {
            self.callbacks
                .write_iid(context.term(Term::Vector), internal_id, element)
                .then_some(())
                .ok_or(GarnetError::Write)?;
            if let Some(quantizer) = &self.quantizer
                && quantizer.is_prepared()
            {
                let mut quant = self
                    .quant_buffer_pool
                    .get_ref(Undef::new(quantizer.canonical_bytes()));
                quantizer.compress(bytemuck::cast_slice::<T, f32>(element), &mut quant)?;
                self.callbacks
                    .write_iid(context.term(Term::Quantized), internal_id, &quant)
                    .then_some(())
                    .ok_or(GarnetError::Write)?;
            }
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
        ok &= self.callbacks.delete_iid(context.term(Term::Quantized), id);

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

/// Dynamic accessor that seamlessly transitions from full precision vector based operation to
/// quantized-only operation.
#[derive(Copy, Clone, Debug)]
pub struct DynamicQuantization;

pub struct DynamicAccessor<'a, T: VectorRepr> {
    provider: &'a GarnetProvider<T>,
    context: &'a Context,
    /// Whether this accessor should use quantized vectors
    quantized: bool,
    id_buffer: PooledRef<'a, AdjList>,
    filtered_ids: PooledRef<'a, Vec<u32>>,
}

impl<'a, T: VectorRepr> DynamicAccessor<'a, T> {
    pub(crate) fn new(
        provider: &'a GarnetProvider<T>,
        context: &'a Context,
        quantized: bool,
    ) -> Self {
        let id_buffer = provider
            .id_buffer_pool
            .get_ref(Undef::new(provider.max_degree + 1));
        let filtered_ids = provider
            .filtered_ids_pool
            .get_ref(Undef::new(MAX_OCCLUSION_SIZE.get() as usize * 2)); // x2 to allow for the length prefixes for garnet
        DynamicAccessor {
            provider,
            context,
            quantized,
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
            guard.finish(0);
            return false;
        }

        let len = guard[self.provider.max_degree];
        guard.finish(len as usize);

        true
    }
}

impl<T: VectorRepr> HasId for DynamicAccessor<'_, T> {
    type Id = u32;
}

impl<T: VectorRepr> SearchExt for DynamicAccessor<'_, T> {
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

impl<T: VectorRepr> ExpandBeam<&[T]> for DynamicAccessor<'_, T> {
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
                    let dist = if self.quantized
                        && let Some(_quantizer) = self.provider.quantizer()
                    {
                        let guard = if let Some(r) = self.provider.start_point_quant_cache.get(&id)
                        {
                            r
                        } else {
                            return future::ready(Err(GarnetProviderError::Garnet(
                                GarnetError::Read,
                            )
                            .into()));
                        };
                        computer.evaluate_similarity(&*guard)
                    } else {
                        let guard = if let Some(r) = self.provider.start_point_cache.get(&id) {
                            r
                        } else {
                            return future::ready(Err(GarnetProviderError::Garnet(
                                GarnetError::Read,
                            )
                            .into()));
                        };
                        computer.evaluate_similarity(&*guard)
                    };

                    on_neighbors(dist, id);
                } else {
                    self.filtered_ids.push(4);
                    self.filtered_ids.push(id);
                }
            }

            let ctx = if self.quantized {
                self.context.term(Term::Quantized)
            } else {
                self.context.term(Term::Vector)
            };

            if !self.filtered_ids.is_empty() {
                self.provider
                    .callbacks
                    .read_multi_lpiid(ctx, &self.filtered_ids, |i, v| {
                        let dist = computer.evaluate_similarity(v);
                        on_neighbors(dist, self.filtered_ids[i as usize * 2 + 1]);
                    });
            }
        }

        future::ready(Ok(()))
    }
}

impl<T: VectorRepr> Accessor for DynamicAccessor<'_, T> {
    type Element<'a>
        = DynVector<T>
    where
        Self: 'a;
    type ElementRef<'a> = &'a [u8];
    type GetError = GarnetProviderError;

    fn get_element(
        &mut self,
        id: Self::Id,
    ) -> impl Future<Output = Result<Self::Element<'_>, Self::GetError>> + Send {
        let v_len = if self.quantized
            && let Some(quantizer) = self.provider.quantizer()
        {
            quantizer.canonical_bytes()
        } else {
            self.provider.dim * mem::size_of::<T>()
        };

        let mut v = match Poly::broadcast(0u8, v_len, AlignToEight) {
            Ok(v) => DynVector::new(v),
            Err(e) => return future::ready(Err(GarnetProviderError::AllocFailed(e))),
        };

        if id == 0 {
            if self.quantized {
                let guard = if let Some(r) = self.provider.start_point_quant_cache.get(&id) {
                    r
                } else {
                    return future::ready(Err(GarnetError::Read.into()));
                };
                v.copy_from_slice(&guard);
                return future::ready(Ok(v));
            } else {
                let guard = if let Some(r) = self.provider.start_point_cache.get(&id) {
                    r
                } else {
                    return future::ready(Err(GarnetError::Read.into()));
                };
                v.copy_from_slice(&guard);
                return future::ready(Ok(v));
            }
        }

        let ctx = if self.quantized {
            self.context.term(Term::Quantized)
        } else {
            self.context.term(Term::Vector)
        };

        if !self.provider.callbacks.read_single_iid(ctx, id, &mut v) {
            return future::ready(Err(GarnetError::Read.into()));
        }

        future::ready(Ok(v))
    }
}

/// Wrapper for full precision distance computer.
pub struct FullPrecisionDistance<T: VectorRepr>(T::Distance);

impl<T: VectorRepr> DynDistanceComputer for FullPrecisionDistance<T> {
    fn evaluate_similarity(&self, a: &[u8], b: &[u8]) -> f32 {
        self.0.evaluate_similarity(
            bytemuck::cast_slice::<u8, T>(a),
            bytemuck::cast_slice::<u8, T>(b),
        )
    }
}

/// Wrapper for full precision query computer.
pub struct FullPrecisionQueryDistance<T: VectorRepr>(T::QueryDistance);

impl<T: VectorRepr> DynQueryComputer for FullPrecisionQueryDistance<T> {
    fn evaluate_similarity(&self, a: &[u8]) -> f32 {
        self.0.evaluate_similarity(bytemuck::cast_slice::<u8, T>(a))
    }
}

/// Type-erased distance computer.
pub struct GarnetDistanceComputer {
    inner: Box<dyn DynDistanceComputer>,
}

impl GarnetDistanceComputer {
    pub fn new<T: DynDistanceComputer + 'static>(computer: T) -> Self {
        Self {
            inner: Box::new(computer),
        }
    }
}
impl DistanceFunction<&[u8], &[u8]> for GarnetDistanceComputer {
    fn evaluate_similarity(&self, x: &[u8], y: &[u8]) -> f32 {
        self.inner.evaluate_similarity(x, y)
    }
}

/// Type-erased query computer.
pub struct GarnetQueryComputer {
    inner: Box<dyn DynQueryComputer>,
}

impl GarnetQueryComputer {
    pub fn new<T: DynQueryComputer + 'static>(computer: T) -> Self {
        Self {
            inner: Box::new(computer),
        }
    }
}

impl PreprocessedDistanceFunction<&[u8]> for GarnetQueryComputer {
    fn evaluate_similarity(&self, changing: &[u8]) -> f32 {
        self.inner.evaluate_similarity(changing)
    }
}

impl<T: VectorRepr> BuildDistanceComputer for DynamicAccessor<'_, T> {
    type DistanceComputer = GarnetDistanceComputer;
    type DistanceComputerError = GarnetProviderError;

    fn build_distance_computer(
        &self,
    ) -> Result<Self::DistanceComputer, Self::DistanceComputerError> {
        if self.quantized
            && let Some(quantizer) = self.provider.quantizer()
        {
            Ok(quantizer.distance_computer()?)
        } else {
            Ok(GarnetDistanceComputer::new(FullPrecisionDistance::<T>(
                T::distance(self.provider.metric_type, Some(self.provider.dim)),
            )))
        }
    }
}

impl<T: VectorRepr> BuildQueryComputer<&[T]> for DynamicAccessor<'_, T> {
    type QueryComputer = GarnetQueryComputer;
    type QueryComputerError = GarnetProviderError;

    fn build_query_computer(
        &self,
        from: &[T],
    ) -> Result<Self::QueryComputer, Self::QueryComputerError> {
        if self.quantized
            && let Some(quantizer) = self.provider.quantizer()
        {
            Ok(quantizer
                .query_computer(bytemuck::cast_slice::<T, f32>(from))
                .map_err(|e| GarnetQuantizerError::QueryComputer(Box::new(e)))?)
        } else {
            Ok(GarnetQueryComputer::new(FullPrecisionQueryDistance::<T>(
                T::query_distance(from, self.provider.metric_type),
            )))
        }
    }
}

/// An escape hatch for the blanket implementation of [`workingset::Fill`].
///
/// Without an `&[T]: Into<Escape<T>>`, the blanket implementation for `workingset::Map`
/// is not applicable, allowing customization of `Fill`.
pub struct Escape<T>(Box<[T]>);

impl<'a, T> Reborrow<'a> for Escape<T> {
    type Target = &'a [T];
    fn reborrow(&'a self) -> Self::Target {
        &self.0
    }
}

pub struct WorkingSet {
    map: workingset::Map<u32, Escape<u8>>,
    contains_unquantized: bool,
}

impl WorkingSet {
    pub fn new(capacity_type: workingset::map::Capacity, capacity: usize) -> Self {
        Self {
            map: workingset::map::Builder::new(capacity_type).build(capacity),
            contains_unquantized: false,
        }
    }
}

impl Deref for WorkingSet {
    type Target = workingset::Map<u32, Escape<u8>>;

    fn deref(&self) -> &Self::Target {
        &self.map
    }
}

impl DerefMut for WorkingSet {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.map
    }
}

type WorkingSetView<'a> = workingset::map::View<'a, u32, Escape<u8>>;

impl<T: VectorRepr> workingset::Fill<WorkingSet> for DynamicAccessor<'_, T> {
    type Error = GarnetProviderError;

    type View<'a>
        = WorkingSetView<'a>
    where
        Self: 'a;

    async fn fill<'a, Itr>(
        &'a mut self,
        set: &'a mut WorkingSet,
        itr: Itr,
    ) -> Result<Self::View<'a>, Self::Error>
    where
        Itr: ExactSizeIterator<Item = Self::Id> + Clone + Send + Sync,
        Self: 'a,
    {
        if !self.quantized {
            // Mark this working set as having full vectors if we're not quantizing yet
            set.contains_unquantized = true;
        } else if set.contains_unquantized {
            // Working set is polluted by full vectors, it must be cleared
            set.clear();
            set.contains_unquantized = false;
        }

        // Evict items from the working set to make room if needed.
        set.prepare(itr.clone());

        self.filtered_ids.clear();
        for id in itr {
            if id == 0 {
                if self.quantized
                    && let Entry::Vacant(e) = set.entry(id)
                {
                    if let Some(guard) = self.provider.start_point_quant_cache.get(&id) {
                        e.insert(Escape((&**guard).into()));
                    }
                } else if let Entry::Vacant(e) = set.entry(id) {
                    if let Some(guard) = self.provider.start_point_cache.get(&id) {
                        e.insert(Escape((&**guard).into()));
                    }
                } else {
                    continue;
                };
            } else if !set.contains_key(&id) {
                self.filtered_ids.push(4);
                self.filtered_ids.push(id);
            }
        }

        let ctx = if self.quantized {
            self.context.term(Term::Quantized)
        } else {
            self.context.term(Term::Vector)
        };

        if !self.filtered_ids.is_empty() {
            self.provider
                .callbacks
                .read_multi_lpiid(ctx, &self.filtered_ids, |id, v| {
                    set.insert(self.filtered_ids[id as usize * 2 + 1], Escape(v.into()));
                });
        }

        Ok(set.view())
    }
}

pub struct DelegateNeighborAccessor<'p, 'a, T: VectorRepr>(&'a mut DynamicAccessor<'p, T>);

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

impl<'p, 'a, T: VectorRepr> DelegateNeighbor<'a> for DynamicAccessor<'p, T> {
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

/// A [`SearchPostProcess`] base object that copies each `Neighbor` to a `(ExternalId, f32)` pair
/// and writes as many as possible to the output buffer.
#[derive(Debug, Default, Clone, Copy)]
pub struct CopyExternalIds;

impl<'a, 'b, T: VectorRepr> SearchPostProcess<DynamicAccessor<'a, T>, &'b [T], GarnetId>
    for CopyExternalIds
{
    type Error = GarnetProviderError;

    fn post_process<I, B>(
        &self,
        accessor: &mut DynamicAccessor<'a, T>,
        _query: &[T],
        _computer: &<DynamicAccessor<'a, T> as BuildQueryComputer<&'b [T]>>::QueryComputer,
        candidates: I,
        output: &mut B,
    ) -> impl Future<Output = Result<usize, Self::Error>> + Send
    where
        I: Iterator<Item = Neighbor<<DynamicAccessor<'a, T> as HasId>::Id>> + Send,
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

/// A [`SearchPostProcess`] base object that reranks quantized vectors by full precision distance.
#[derive(Debug, Default, Clone, Copy)]
pub struct Rerank;

impl<'a, 'b, T: VectorRepr> SearchPostProcessStep<DynamicAccessor<'a, T>, &'b [T], GarnetId>
    for Rerank
{
    type Error<NextError>
        = GarnetProviderError
    where
        NextError: diskann::error::StandardError;

    type NextAccessor = DynamicAccessor<'a, T>;

    async fn post_process_step<I, B, Next>(
        &self,
        next: &Next,
        accessor: &mut DynamicAccessor<'a, T>,
        query: &'b [T],
        computer: &<DynamicAccessor<'a, T> as BuildQueryComputer<&'b [T]>>::QueryComputer,
        candidates: I,
        output: &mut B,
    ) -> Result<usize, Self::Error<Next::Error>>
    where
        I: Iterator<Item = Neighbor<<DynamicAccessor<'a, T> as HasId>::Id>> + Send,
        B: SearchOutputBuffer<GarnetId> + Send + ?Sized,
        Next: SearchPostProcess<Self::NextAccessor, &'b [T], GarnetId> + Sync,
    {
        if !accessor.quantized {
            // Skip reranking if the accessor if working with full precision
            return next
                .post_process(accessor, query, computer, candidates, output)
                .await
                .map_err(|e| GarnetProviderError::PostProcessing(Box::new(e)));
        }

        let provider = &accessor.provider;
        let f = T::distance(provider.metric_type, Some(provider.dim));
        let mut v = Poly::broadcast(0u8, provider.dim * mem::size_of::<T>(), AlignToEight)?;

        // Filter before computing the full precision distances.
        let mut reranked: Vec<(u32, f32)> = candidates
            .filter_map(|n| {
                if !provider.vector_iid_exists(accessor.context, n.id) {
                    None
                } else if provider.callbacks.read_single_iid(
                    accessor.context.term(Term::Vector),
                    n.id,
                    &mut v,
                ) {
                    Some((
                        n.id,
                        f.evaluate_similarity(query, bytemuck::cast_slice::<u8, T>(&v)),
                    ))
                } else {
                    None
                }
            })
            .collect();

        // Sort the full precision distances.
        reranked
            .sort_unstable_by(|a, b| (a.1).partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        next.post_process(
            accessor,
            query,
            computer,
            reranked.into_iter().map(|(id, d)| Neighbor::new(id, d)),
            output,
        )
        .await
        .map_err(|e| GarnetProviderError::PostProcessing(Box::new(e)))
    }
}

impl<T: VectorRepr> SearchStrategy<GarnetProvider<T>, &[T]> for DynamicQuantization {
    type SearchAccessor<'a> = DynamicAccessor<'a, T>;
    type SearchAccessorError = GarnetProviderError;
    type QueryComputer = GarnetQueryComputer;

    fn search_accessor<'a>(
        &'a self,
        provider: &'a GarnetProvider<T>,
        context: &'a <GarnetProvider<T> as DataProvider>::Context,
    ) -> Result<Self::SearchAccessor<'a>, Self::SearchAccessorError> {
        let quantized = provider.is_quantized();
        Ok(DynamicAccessor::new(provider, context, quantized))
    }
}

impl<T: VectorRepr> DefaultPostProcessor<GarnetProvider<T>, &[T], GarnetId>
    for DynamicQuantization
{
    default_post_processor!(
        glue::Pipeline<glue::FilterStartPoints, glue::Pipeline<Rerank, CopyExternalIds>>
    );
}

impl<T: VectorRepr> PruneStrategy<GarnetProvider<T>> for DynamicQuantization {
    type PruneAccessor<'a> = DynamicAccessor<'a, T>;
    type PruneAccessorError = GarnetProviderError;
    type DistanceComputer<'a> = GarnetDistanceComputer;
    type WorkingSet = WorkingSet;

    fn prune_accessor<'a>(
        &'a self,
        provider: &'a GarnetProvider<T>,
        context: &'a <GarnetProvider<T> as DataProvider>::Context,
    ) -> Result<Self::PruneAccessor<'a>, Self::PruneAccessorError> {
        let quantized = provider.is_quantized();
        Ok(DynamicAccessor::new(provider, context, quantized))
    }

    fn create_working_set(&self, capacity: usize) -> Self::WorkingSet {
        // Using `Capacity::Default` means that the constructed working set will act as a
        // cache and persist up to `capacity` items across uses of the working set.
        //
        // This reuse is limited to a single collection of backedges for an insert or multi-insert.
        WorkingSet::new(workingset::map::Capacity::Default, capacity)
    }
}

impl<T: VectorRepr> InsertStrategy<GarnetProvider<T>, &[T]> for DynamicQuantization {
    type PruneStrategy = Self;

    fn insert_search_accessor<'a>(
        &'a self,
        provider: &'a GarnetProvider<T>,
        context: &'a <GarnetProvider<T> as DataProvider>::Context,
    ) -> Result<Self::SearchAccessor<'a>, Self::SearchAccessorError> {
        let quantized = provider.is_quantized();
        Ok(DynamicAccessor::new(provider, context, quantized))
    }

    fn prune_strategy(&self) -> Self::PruneStrategy {
        *self
    }
}

impl<T: VectorRepr> InplaceDeleteStrategy<GarnetProvider<T>> for DynamicQuantization {
    type DeleteElement<'a> = &'a [T];
    type DeleteElementGuard = Box<[T]>;
    type DeleteElementError = GarnetProviderError;

    type PruneStrategy = Self;
    type DeleteSearchAccessor<'a> = DynamicAccessor<'a, T>;
    type SearchPostProcessor = glue::CopyIds;
    type SearchStrategy = Self;

    fn prune_strategy(&self) -> Self::PruneStrategy {
        Self
    }

    fn search_strategy(&self) -> Self::SearchStrategy {
        Self
    }

    fn search_post_processor(&self) -> Self::SearchPostProcessor {
        glue::CopyIds
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
