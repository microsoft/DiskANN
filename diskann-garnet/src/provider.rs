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
            self, Accept, Decision, DefaultPostProcessor, FilteredAccessor, InplaceDeleteStrategy,
            InsertStrategy, PruneStrategy, SearchAccessor, SearchPostProcess,
            SearchPostProcessStep, SearchStrategy,
        },
        workingset::{self, map::Entry},
    },
    neighbor::Neighbor,
    provider::{
        DataProvider, Delete, ElementStatus, HasId, NeighborAccessor, NeighborAccessorMut,
        NoopGuard, SetElement,
    },
    utils::VectorRepr,
};
use diskann_quantization::alloc::{AllocatorError, Poly};
use diskann_utils::Matrix;
use diskann_utils::{
    matrix::MatrixView,
    object_pool::{AsPooled, ObjectPool, PooledRef, Undef},
};
use diskann_vector::{
    DistanceFunction, PreprocessedDistanceFunction, contains::ContainsSimd, distance::Metric,
};
use std::{
    any::TypeId,
    future,
    marker::PhantomData,
    mem,
    ops::{Deref, DerefMut},
    sync::{
        Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
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

/// Quantization state and table are stored under this key in Garnet under the metadata
/// term.
///
/// The first byte is a boolean reflecting whether backfill is complete. The remaining
/// bytes are the serialized quant table.
const QUANT_STATE_KEY: u32 = u32::from_be_bytes(*b"_qnt");

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
pub(crate) enum GarnetProviderError {
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
pub(crate) struct GarnetProvider<T: VectorRepr> {
    /// Dimension of the full precision vectors
    dim: usize,
    /// Metric to use for comparing distances
    metric_type: Metric,
    /// Maximum degree of the graph.
    ///
    /// Note: Unlike DiskANN, this is the true maximum. Neighbors can never
    /// exceed this degree.
    max_degree: usize,
    /// Garnet storage engine callbacks
    callbacks: Callbacks,
    /// The quantizer the index will use, or None if NOQUANT is used.
    quantizer: Option<Box<dyn GarnetQuantizer>>,
    /// Tracks whether the index is ready to operate fully quantized.
    all_quantized: AtomicBool,
    /// Per job tracker for quantization backfill completion
    backfills_completed: AtomicU64,
    /// Lock to ensure training only happens once.
    training_lock: Mutex<()>,
    /// Pool of pre-allocated buffers to use for neighbor lists
    id_buffer_pool: ObjectPool<AdjList>,
    /// Pool of pre-allocated buffers to use for IDs
    filtered_ids_pool: ObjectPool<Vec<u32>>,
    /// Pool of pre-allocated buffers to use for filter decisions during
    /// filtered search beam expansion
    filtered_decisions_pool: ObjectPool<Vec<bool>>,
    /// Pool of pre-allocated buffers to use for quantizing vectors
    quant_buffer_pool: ObjectPool<Vec<u8>>,
    /// Small cache for the start points' neighbors
    neighbor_cache: DashMap<u32, Vec<u32>, foldhash::fast::RandomState>,
    /// Small cache for the start points' full precision vector data
    start_point_cache: DashMap<u32, Poly<[u8], AlignToEight>, foldhash::fast::RandomState>,
    /// Small cache for the start points' quantized vector data
    start_point_quant_cache: DashMap<u32, Poly<[u8], AlignToEight>, foldhash::fast::RandomState>,
    /// Free space map to track internal IDs
    fsm: FreeSpaceMap,
    _phantom: PhantomData<T>,
}

impl<T: VectorRepr> GarnetProvider<T> {
    pub(crate) fn new(
        dim: usize,
        quant_type: VectorQuantType,
        metric_type: Metric,
        max_degree: usize,
        callbacks: Callbacks,
        context: &Context,
    ) -> Result<Self, GarnetProviderError> {
        let parallelism = std::thread::available_parallelism().unwrap().get() * 2;
        let id_buffer_pool =
            ObjectPool::new(Undef::new(max_degree + 1), parallelism, Some(parallelism));
        let filtered_ids_pool = ObjectPool::new(
            Undef::new(MAX_OCCLUSION_SIZE.get() as usize * 2),
            parallelism,
            Some(parallelism),
        );
        let filtered_decisions_pool = ObjectPool::new(
            Undef::new(MAX_OCCLUSION_SIZE.get() as usize),
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
        if callbacks.read_single_iid(&context.term(Term::Vector), 0, &mut v) {
            let mut neighbors = vec![0u32; max_degree + 1];
            if !callbacks.read_single_iid(&context.term(Term::Neighbors), 0, &mut neighbors) {
                return Err(GarnetError::Read.into());
            }

            start_point_cache.insert(0, v);

            let len = neighbors[max_degree] as usize;
            neighbors.truncate(len);
            neighbor_cache.insert(0, neighbors);
        }

        let (quantizer, canonical_bytes, all_quantized) = match quant_type {
            VectorQuantType::NoQuant
            | VectorQuantType::XNoQuantU8
            | VectorQuantType::XNoQuantI8 => (None, 0, false),
            VectorQuantType::Invalid => return Err(GarnetProviderError::InvalidQuantizer),
            VectorQuantType::Q8 => {
                if TypeId::of::<T>() != TypeId::of::<f32>() {
                    return Err(GarnetProviderError::InvalidQuantizer);
                }

                let quantizer = Box::new(quantization::MinMax8Bit::new(dim, metric_type)?)
                    as Box<dyn GarnetQuantizer>;
                let canonical_bytes = quantizer.bytes();

                // NOTE: Q8 needs no training, so it always starts with backfill
                // complete. However, we still need to load the start point if
                // it exists.

                let mut qsv = Poly::broadcast(0u8, canonical_bytes, AlignToEight)?;
                if callbacks.read_single_iid(&context.term(Term::Quantized), 0, &mut qsv) {
                    start_point_quant_cache.insert(0, qsv);
                }

                (Some(quantizer), canonical_bytes, true)
            }
            VectorQuantType::Bin | VectorQuantType::XBinU8 | VectorQuantType::XBinI8 => {
                let quantizer =
                    Box::new(quantization::Spherical1Bit::new(dim)) as Box<dyn GarnetQuantizer>;
                let canonical_bytes = quantizer.bytes();
                let mut all_quantized = false;

                if let Some(total_quant_state) =
                    callbacks.read_varsize_iid::<u8>(&context.term(Term::Metadata), QUANT_STATE_KEY)
                {
                    if total_quant_state.len() <= 1 {
                        return Err(GarnetProviderError::InvalidQuantizer);
                    }

                    all_quantized = total_quant_state[0] != 0;

                    quantizer.deserialize(&total_quant_state[1..])?;

                    // Cache the saved start point, which should already exist if quantization is complete
                    let mut qsv = Poly::broadcast(0u8, canonical_bytes, AlignToEight)?;
                    if callbacks.read_single_iid(&context.term(Term::Quantized), 0, &mut qsv) {
                        start_point_quant_cache.insert(0, qsv);
                    } else if all_quantized {
                        return Err(GarnetProviderError::StartPoint);
                    }
                }

                (Some(quantizer), canonical_bytes, all_quantized)
            }
        };
        let quant_buffer_pool =
            ObjectPool::new(Undef::new(canonical_bytes), parallelism, Some(parallelism));

        let fsm: FreeSpaceMap = FreeSpaceMap::new(
            context,
            callbacks,
            quantizer.is_some() && all_quantized,
            quantizer.is_none() || all_quantized,
        )?;

        Ok(Self {
            dim,
            metric_type,
            max_degree,
            callbacks,
            quantizer,
            all_quantized: AtomicBool::new(all_quantized),
            backfills_completed: AtomicU64::new(0),
            training_lock: Mutex::new(()),
            id_buffer_pool,
            filtered_ids_pool,
            filtered_decisions_pool,
            quant_buffer_pool,
            start_point_cache,
            start_point_quant_cache,
            neighbor_cache,
            fsm,
            _phantom: PhantomData,
        })
    }

    /// Called during `VADD` to ensure a start point exists.
    /// If there isn't a start point yet, the given point will be set as the start point; if there
    /// is a start point already, we ensure the caches are populated.
    pub(crate) fn maybe_set_start_point(
        &self,
        context: &Context,
        point: &[T],
    ) -> Result<(), GarnetProviderError> {
        let mut v = Poly::broadcast(0u8, self.dim * mem::size_of::<T>(), AlignToEight)?;
        if self
            .callbacks
            .read_single_iid(&context.term(Term::Vector), 0, &mut v)
        {
            // Garnet already has a start point, so use that instead of `point`
            let mut neighbors = vec![0u32; self.max_degree + 1];
            if !self
                .callbacks
                .read_single_iid(&context.term(Term::Neighbors), 0, &mut neighbors)
            {
                return Err(GarnetError::Read.into());
            }

            if self.is_quantized()
                && let Some(quantizer) = self.quantizer()
            {
                let mut qpoint = vec![0u8; quantizer.bytes()];
                if !self
                    .callbacks
                    .read_single_iid(&context.term(Term::Quantized), 0, &mut qpoint)
                {
                    return Err(GarnetError::Read.into());
                }

                self.start_point_quant_cache
                    .insert(0, Poly::from_iter(qpoint.iter().copied(), AlignToEight)?);
            }

            self.start_point_cache.insert(0, v);
            let len = neighbors[self.max_degree] as usize;
            neighbors.truncate(len);
            self.neighbor_cache.insert(0, neighbors);
        } else {
            let neighbors = vec![0u32; self.max_degree + 1];

            // Grab the start point id, which must be zero.
            let id = self.fsm.next_id(context)?;
            if id.id() != 0 {
                self.fsm.mark_free(context, id.id())?;
                return Err(GarnetProviderError::StartPoint);
            }

            if !self
                .callbacks
                .write_iid(&context.term(Term::Vector), 0, point)
            {
                return Err(GarnetError::Write.into());
            }

            if self.is_quantized()
                && let Some(quantizer) = self.quantizer()
            {
                // We are already able to quantize, so store the quantized start point

                let mut qpoint = vec![0u8; quantizer.bytes()];
                let point_f32 =
                    T::as_f32(point).map_err(|e| GarnetQuantizerError::Compression(Box::new(e)))?;
                quantizer.compress(&point_f32, &mut qpoint)?;

                if !self
                    .callbacks
                    .write_iid(&context.term(Term::Quantized), 0, &qpoint)
                {
                    return Err(GarnetError::Write.into());
                }

                self.start_point_quant_cache
                    .insert(0, Poly::from_iter(qpoint.iter().copied(), AlignToEight)?);
            }

            if !self
                .callbacks
                .write_iid(&context.term(Term::Neighbors), 0, &neighbors)
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

    pub(crate) fn start_points_exist(&self) -> bool {
        self.start_point_cache.get(&0).is_some() && self.neighbor_cache.get(&0).is_some()
    }

    pub(crate) fn set_attributes(
        &self,
        context: &Context,
        id: &GarnetId,
        data: &[u8],
    ) -> Result<(), GarnetProviderError> {
        if self
            .callbacks
            .write_eid(&context.term(Term::Attributes), id, data)
        {
            Ok(())
        } else {
            Err(GarnetError::Write.into())
        }
    }
    pub(crate) fn vector_id_exists(&self, context: &Context, id: &GarnetId) -> bool {
        let iid = match self.to_internal_id(context, id) {
            Ok(iid) => iid,
            Err(_) => return false,
        };
        !self.fsm.is_free(context, iid).unwrap_or(true)
    }

    pub(crate) fn vector_iid_exists(&self, context: &Context, id: u32) -> bool {
        !self.fsm.is_free(context, id).unwrap_or(true)
    }

    pub(crate) fn max_internal_id(&self) -> u32 {
        self.fsm.max_id()
    }

    /// Train the quantizer.
    ///
    /// This should only be called when at least `quantizer.required_vectors()` vectors exist in
    /// the provider. This will build quantization tables, but does not quantize any vectors.
    ///
    /// Note that this may be invoked multiple times due to concurrent operations, and so must
    /// ensure that training happens only once.
    pub(crate) fn train_quantizer(&self, context: &Context) -> bool {
        // Ensure we don't kick off training twice.
        let _training_guard = match self.training_lock.try_lock() {
            Ok(g) => g,
            Err(_) => return false,
        };

        let quantizer = match &self.quantizer {
            Some(q) => {
                if !q.is_trained() {
                    q
                } else {
                    // Quantizer already trained, bail.
                    return false;
                }
            }
            None => return false,
        };

        let rows = quantizer.required_vectors();
        let mut data = Matrix::<T>::new(T::default(), rows, self.dim);
        let mut row_idx = 0usize;

        if self
            .fsm
            .visit_used(context, |id| {
                // Skip the start point.
                if id == 0 {
                    return true;
                }

                if row_idx >= rows {
                    return false;
                }

                // Read the vector into the data matrix.
                let row = data.row_mut(row_idx);
                if !self
                    .callbacks
                    .read_single_iid(&context.term(Term::Vector), id, row)
                {
                    return false;
                }

                row_idx += 1;

                true
            })
            .is_err()
        {
            // Training failed.
            return false;
        }

        if row_idx < quantizer.required_vectors() {
            // The required amount of training data was not present.
            return false;
        }

        let view = if let Some(view) = data.subview(0..row_idx) {
            view
        } else {
            return false;
        };

        // Train the quantizer.
        let converted = match T::as_f32(view.as_slice()) {
            Ok(v) => v,
            Err(_) => return false,
        };
        let view = match MatrixView::try_from(&converted, view.nrows(), view.ncols()) {
            Ok(v) => v,
            Err(_) => return false,
        };
        match quantizer.train(self.metric_type, view) {
            Ok(()) => {
                let quant_state = if let Ok(s) = quantizer.serialize() {
                    s
                } else {
                    return false;
                };

                let mut total_quant_state = vec![0u8; quant_state.len() + 1];
                total_quant_state[1..].copy_from_slice(&quant_state);

                if !self.callbacks.write_iid(
                    &context.term(Term::Metadata),
                    QUANT_STATE_KEY,
                    &total_quant_state,
                ) {
                    return false;
                }

                self.fsm.enable_quantization();
                true
            }
            Err(_e) => false,
        }
    }

    /// Bulk quantize previously inserted vectors.
    ///
    /// This function will be invoked on multiple threads. The total number of tasks and the ID of
    /// the current task are given as inputs.
    pub(crate) fn backfill_quant_vectors(
        &self,
        context: &Context,
        task_idx: usize,
        task_count: usize,
    ) {
        let quantizer = match &self.quantizer {
            Some(q) => q,
            None => return,
        };

        let max_id = self.fsm.max_id_for_backfill() as usize;
        if max_id >= u32::MAX as usize {
            // The max_id was somehow not sampled, so bail.
            return;
        }

        // If we have more tasks than vectors to backfill, we exit the extra tasks early.
        let task_count = task_count.min(max_id + 1);
        if task_idx >= task_count {
            return;
        }

        // Evenly divide the ID range from 0..max_id and determine this thread's backfill
        // range.
        let work_count = (max_id + 1).div_ceil(task_count); // will be >= 1
        let start_id = (work_count * task_idx) as u32;
        let end_id = (work_count * (task_idx + 1)).min(max_id + 1) as u32;

        let mut v = vec![T::default(); self.dim];
        let mut f = vec![0f32; self.dim];
        let mut q = vec![0u8; quantizer.bytes()];
        for id in start_id..end_id {
            if !self
                .callbacks
                .read_single_iid(&context.term(Term::Vector), id, &mut v)
            {
                continue;
            }

            if T::as_f32_into(&v, &mut f).is_err() {
                continue;
            }
            if quantizer.compress(&f, &mut q).is_err() {
                continue;
            };

            if !self
                .callbacks
                .write_iid(&context.term(Term::Quantized), id, &q)
            {
                continue;
            }
        }

        // `backfills_completed` tracks how many of the worker threads have finished their backfill.
        // When the all are done, backfill is completed, aside from the start points.
        let backfill_finished =
            self.backfills_completed.fetch_add(1, Ordering::AcqRel) + 1 == task_count as u64;

        if backfill_finished {
            // The final thread to finish backfilling will add the quantized started points.
            if let Some(v) = self.start_point_cache.get(&0)
                && let Ok(v_f32) = T::as_f32(bytemuck::cast_slice::<u8, T>(&v))
                && quantizer.compress(&v_f32, &mut q).is_ok()
            {
                let _ = self
                    .callbacks
                    .write_iid(&context.term(Term::Quantized), 0, &q);

                // set the cache
                let point = if let Ok(p) = Poly::from_iter(q.iter().copied(), AlignToEight) {
                    p
                } else {
                    // NOTE: This return is unrecoverable in the current design, as there is no way
                    // to signal that backfill has failed. The index will operate on full precision
                    // vectors only from now on.
                    return;
                };
                self.start_point_quant_cache.insert(0, point);
            }

            // Now that all vectors have quant vectors associated, unlock ID reuse.
            self.fsm.enable_reuse();

            if !self.callbacks.rmw_iid::<_, u8>(
                &context.term(Term::Metadata),
                QUANT_STATE_KEY,
                1,
                |data| {
                    data[0] = 1;
                },
            ) {
                // NOTE: This return is unrecoverable in the current design, as there is no way to
                // signal that backfill failed.
                return;
            }

            // Signal to the index that it is now safe to operate in quantized mode.
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
    pub(crate) fn is_quantized(&self) -> bool {
        self.quantizer.is_some() && self.all_quantized.load(Ordering::Acquire)
    }

    pub(crate) fn quantization_needed(&self) -> bool {
        if let Some(quantizer) = &self.quantizer {
            !self.is_quantized()
                && quantizer.is_trained()
                && self.max_internal_id() as usize > quantizer.required_vectors()
        } else {
            false
        }
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
            .read_single_iid(&context.term(Term::Vector), iid, &mut v)
        {
            return Err(GarnetError::Read.into());
        }

        Ok(v)
    }

    fn get_neighbors(
        &self,
        context: &Context,
        iid: u32,
        neighbors: &mut AdjacencyList<u32>,
    ) -> bool {
        let mut guard = neighbors.resize(self.max_degree + 1);

        if iid == 0
            && let Some(cached) = self.neighbor_cache.get(&iid)
        {
            guard[0..cached.len()].copy_from_slice(&cached);
            guard.finish(cached.len());
            return true;
        }

        if !self
            .callbacks
            .read_single_iid(&context.term(Term::Neighbors), iid, &mut guard)
        {
            guard.finish(0);
            return false;
        }

        let len = guard[self.max_degree];
        guard.finish(len as usize);

        true
    }

    fn set_neighbors(
        &self,
        context: &Context,
        iid: u32,
        neighbors: &[u32],
        scratch: &mut AdjacencyList<u32>,
    ) -> Result<(), GarnetProviderError> {
        let mut guard = scratch.resize(self.max_degree + 1);
        guard[0..neighbors.len()].copy_from_slice(neighbors);
        guard[self.max_degree] = neighbors.len() as u32;

        // NOTE: We use `rmw_iid` here instead of `write_iid` to guarantee cache coherence.
        if !self.callbacks.rmw_iid(
            &context.term(Term::Neighbors),
            iid,
            (self.max_degree + 1) * mem::size_of::<u32>(),
            |data: &mut [u32]| {
                data.copy_from_slice(&guard);
                if iid == 0 {
                    self.neighbor_cache.insert(iid, neighbors.to_vec());
                }
            },
        ) {
            return Err(GarnetError::Write.into());
        }

        guard.finish(0);

        Ok(())
    }

    fn append_vector(
        &self,
        context: &Context,
        iid: u32,
        neighbors: &[u32],
    ) -> Result<(), GarnetProviderError> {
        let max_degree = self.max_degree;
        if !self.callbacks.rmw_iid(
            &context.term(Term::Neighbors),
            iid,
            (max_degree + 1) * mem::size_of::<u32>(),
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

                if iid == 0
                    && let Some(mut ns) = self.neighbor_cache.get_mut(&iid)
                {
                    ns.clear();
                    ns.extend(data.iter().copied().take(len));
                }
            },
        ) {
            return Err(GarnetError::Write.into());
        }

        Ok(())
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
            &context.term(Term::IntMap),
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
            .read_varsize_iid(&context.term(Term::ExtMap), id)
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
        let internal_id = self.fsm.next_id(context)?;

        // Set quantization readiness
        if let Some(quantizer) = &self.quantizer
            && !internal_id.should_quantize()
            && !quantizer.is_trained()
            && self.fsm.total_used() > quantizer.required_vectors()
        {
            context.set_quantizer_ready();
        }

        let insert = || -> Result<(), Self::SetError> {
            self.callbacks
                .write_iid(&context.term(Term::Vector), internal_id.id(), element)
                .then_some(())
                .ok_or(GarnetError::Write)?;
            if let Some(quantizer) = &self.quantizer
                && internal_id.should_quantize()
            {
                let mut quant = self
                    .quant_buffer_pool
                    .get_ref(Undef::new(quantizer.bytes()));
                let element_f32 = T::as_f32(element).map_err(|e| {
                    GarnetProviderError::Quantizer(GarnetQuantizerError::Compression(Box::new(e)))
                })?;
                quantizer.compress(&element_f32, &mut quant)?;
                self.callbacks
                    .write_iid(&context.term(Term::Quantized), internal_id.id(), &quant)
                    .then_some(())
                    .ok_or(GarnetError::Write)?;
            }
            self.callbacks
                .write_iid(&context.term(Term::ExtMap), internal_id.id(), id)
                .then_some(())
                .ok_or(GarnetError::Write)?;
            self.callbacks
                .write_eid(
                    &context.term(Term::IntMap),
                    id,
                    bytemuck::bytes_of(&internal_id.id()),
                )
                .then_some(())
                .ok_or(GarnetError::Write)?;
            Ok(())
        };

        match insert() {
            Ok(()) => (),
            Err(e) => {
                self.fsm.mark_free(context, internal_id.id())?;
                return Err(e);
            }
        }

        Ok(NoopGuard::new(internal_id.id()))
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

        // Delete mappings, so vector will no longer be returned.
        let mut ok = true;
        ok &= self.callbacks.delete_iid(&context.term(Term::ExtMap), id);
        ok &= self.callbacks.delete_eid(&context.term(Term::IntMap), gid);

        // It is not an error to fail deleting attributes; they may not exist.
        let _: bool = self
            .callbacks
            .delete_eid(&context.term(Term::Attributes), gid);

        // TODO: inplace_delete needs access to neighbors. Delete these once that bug is fixed.
        // See https://github.com/microsoft/DiskANN/issues/1153.
        // ok &= self
        //     .callbacks
        //     .delete_iid(&context.term(Term::Neighbors), id);

        ok &= self.callbacks.delete_iid(&context.term(Term::Vector), id);

        // It is not an error to fail deleting quantized terms; they may not exist yet.
        let _: bool = self
            .callbacks
            .delete_iid(&context.term(Term::Quantized), id);

        // Mark the ID free in the FSM.
        if let Err(e) = self.fsm.mark_free(context, id) {
            return future::ready(Err(e.into()));
        };

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
        // This is a no-op since DiskANN never calls this anyway.
        future::ready(Ok(()))
    }

    fn status_by_internal_id(
        &self,
        context: &Self::Context,
        id: Self::InternalId,
    ) -> impl Future<Output = Result<diskann::provider::ElementStatus, Self::Error>> + Send {
        let status = match self.fsm.is_free(context, id) {
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
pub(crate) struct DynamicQuantization;

pub(crate) struct DynamicAccessor<'a, T: VectorRepr> {
    provider: &'a GarnetProvider<T>,
    context: &'a Context,
    /// Whether this accessor should use quantized vectors
    quantized: bool,
    computer: GarnetQueryComputer,
    id_buffer: PooledRef<'a, AdjList>,
    filtered_ids: PooledRef<'a, Vec<u32>>,
    filtered_decisions: PooledRef<'a, Vec<bool>>,
}

impl<'a, T: VectorRepr> DynamicAccessor<'a, T> {
    const START_ID: u32 = 0;

    pub(crate) fn new(
        provider: &'a GarnetProvider<T>,
        context: &'a Context,
        query: &'a [T],
        quantized: bool,
    ) -> Result<Self, GarnetProviderError> {
        let id_buffer = provider
            .id_buffer_pool
            .get_ref(Undef::new(provider.max_degree + 1));
        let filtered_ids = provider
            .filtered_ids_pool
            .get_ref(Undef::new(MAX_OCCLUSION_SIZE.get() as usize * 2)); // x2 to allow for the length prefixes for garnet
        let filtered_decisions = provider
            .filtered_decisions_pool
            .get_ref(Undef::new(MAX_OCCLUSION_SIZE.get() as usize));

        let computer = if quantized && let Some(quantizer) = provider.quantizer() {
            let from_f32 = T::as_f32(query).map_err(|e| {
                GarnetProviderError::Quantizer(GarnetQuantizerError::Compression(Box::new(e)))
            })?;

            quantizer
                .query_computer(&from_f32)
                .map_err(|e| GarnetQuantizerError::QueryComputer(Box::new(e)))?
        } else {
            GarnetQueryComputer::new(FullPrecisionQueryDistance::<T>(T::query_distance(
                query,
                provider.metric_type,
            )))
        };

        Ok(DynamicAccessor {
            provider,
            context,
            quantized,
            computer,
            id_buffer,
            filtered_ids,
            filtered_decisions,
        })
    }

    /// Return the distance to the start point (the point with `ID == 0`).
    fn start_point_distance(&mut self) -> Result<f32, GarnetProviderError> {
        if self.quantized
            && let Some(_quantizer) = self.provider.quantizer()
        {
            match self.provider.start_point_quant_cache.get(&Self::START_ID) {
                Some(guard) => Ok(self.computer.evaluate_similarity(&*guard)),
                None => Err(GarnetProviderError::Garnet(GarnetError::Read)),
            }
        } else {
            match self.provider.start_point_cache.get(&Self::START_ID) {
                Some(guard) => Ok(self.computer.evaluate_similarity(&*guard)),
                None => Err(GarnetProviderError::Garnet(GarnetError::Read)),
            }
        }
    }
}

impl<T: VectorRepr> HasId for DynamicAccessor<'_, T> {
    type Id = u32;
}

impl<T: VectorRepr> SearchAccessor for DynamicAccessor<'_, T> {
    fn starting_points(&self) -> impl Future<Output = ANNResult<Vec<Self::Id>>> + Send {
        let points = if self.provider.start_points_exist() {
            vec![Self::START_ID]
        } else {
            vec![]
        };
        future::ready(Ok(points))
    }

    fn is_not_start_point(
        &self,
    ) -> impl Future<Output = ANNResult<impl Fn(Self::Id) -> bool + Send + Sync + 'static>> + Send
    {
        future::ready(Ok(move |id| id != Self::START_ID))
    }

    fn start_point_distances<F>(&mut self, mut f: F) -> impl Future<Output = ANNResult<()>> + Send
    where
        F: FnMut(Self::Id, f32) + Send,
    {
        // If there are no start points, just return without doing anything.
        // Searches on an empty index just return no results.
        if !self.provider.start_points_exist() {
            return future::ready(Ok(()));
        }

        let result = match self.start_point_distance() {
            Ok(dist) => {
                f(Self::START_ID, dist);
                Ok(())
            }
            Err(err) => Err(ANNError::from(err)),
        };

        std::future::ready(result)
    }

    fn expand_beam<Itr, P, F>(
        &mut self,
        ids: Itr,
        mut pred: P,
        mut on_neighbors: F,
    ) -> impl Future<Output = ANNResult<()>> + Send
    where
        Itr: Iterator<Item = Self::Id> + Send,
        P: glue::HybridPredicate<Self::Id> + Send + Sync,
        F: FnMut(Self::Id, f32) + Send,
    {
        // Pilfer the `id_buffer` for the duration of this call to ensure a disjoint
        // borrow. We put it back at the end to save the allocation.
        let mut id_buffer = mem::take(&mut **self.id_buffer);

        for nl_id in ids {
            self.provider
                .get_neighbors(self.context, nl_id, &mut id_buffer);
            self.filtered_ids.clear();
            for id in id_buffer.iter().copied().filter(|id| pred.eval_mut(id)) {
                if id == Self::START_ID {
                    let dist = match self.start_point_distance() {
                        Ok(dist) => dist,
                        Err(err) => return future::ready(Err(ANNError::from(err))),
                    };

                    on_neighbors(id, dist);
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
                    .read_multi_lpiid(&ctx, &self.filtered_ids, |i, v| {
                        let dist = self.computer.evaluate_similarity(v);
                        on_neighbors(self.filtered_ids[i as usize * 2 + 1], dist);
                    });
            }
        }

        **self.id_buffer = id_buffer;
        future::ready(Ok(()))
    }
}

/// Wrapper for full precision distance computer.
pub(crate) struct FullPrecisionDistance<T: VectorRepr>(T::Distance);

impl<T: VectorRepr> DynDistanceComputer for FullPrecisionDistance<T> {
    fn evaluate_similarity(&self, a: &[u8], b: &[u8]) -> f32 {
        self.0.evaluate_similarity(
            bytemuck::cast_slice::<u8, T>(a),
            bytemuck::cast_slice::<u8, T>(b),
        )
    }
}

/// Wrapper for full precision query computer.
pub(crate) struct FullPrecisionQueryDistance<T: VectorRepr>(T::QueryDistance);

impl<T: VectorRepr> DynQueryComputer for FullPrecisionQueryDistance<T> {
    fn evaluate_similarity(&self, a: &[u8]) -> f32 {
        self.0.evaluate_similarity(bytemuck::cast_slice::<u8, T>(a))
    }
}

/// Type-erased distance computer.
pub(crate) struct GarnetDistanceComputer {
    inner: Box<dyn DynDistanceComputer>,
}

impl GarnetDistanceComputer {
    pub(crate) fn new<T: DynDistanceComputer + 'static>(computer: T) -> Self {
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
pub(crate) struct GarnetQueryComputer {
    inner: Box<dyn DynQueryComputer>,
}

impl GarnetQueryComputer {
    pub(crate) fn new<T: DynQueryComputer + 'static>(computer: T) -> Self {
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

/// A [`SearchPostProcess`] base object that copies each `Neighbor` to a `(ExternalId, f32)` pair
/// and writes as many as possible to the output buffer.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct CopyExternalIds;

impl<'a, T: VectorRepr> SearchPostProcess<DynamicAccessor<'a, T>, &[T], GarnetId>
    for CopyExternalIds
{
    type Error = GarnetProviderError;

    fn post_process<I, B>(
        &self,
        accessor: &mut DynamicAccessor<'a, T>,
        _query: &[T],
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
                Err(_) => continue, // Can't read the mapping; skip.
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
pub(crate) struct Rerank;

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
                .post_process(accessor, query, candidates, output)
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
                    &accessor.context.term(Term::Vector),
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
            reranked.into_iter().map(|(id, d)| Neighbor::new(id, d)),
            output,
        )
        .await
        .map_err(|e| GarnetProviderError::PostProcessing(Box::new(e)))
    }
}

impl<T: VectorRepr> FilteredAccessor for DynamicAccessor<'_, T> {
    fn start_point_distances<F>(&mut self, mut f: F) -> impl Future<Output = ANNResult<()>> + Send
    where
        F: FnMut(glue::Decision<Self::Id>, f32) + Send,
    {
        if !self.provider.start_points_exist() {
            return future::ready(Ok(()));
        }

        let result = match self.start_point_distance() {
            Ok(dist) => {
                f(glue::Decision::reject(Self::START_ID), dist);
                Ok(())
            }
            Err(err) => Err(ANNError::from(err)),
        };

        future::ready(result)
    }

    fn expand_beam_filtered<Itr, P, F>(
        &mut self,
        ids: Itr,
        mut pred: P,
        mut on_neighbors: F,
    ) -> impl Future<Output = ANNResult<()>> + Send
    where
        Itr: Iterator<Item = Self::Id> + Send,
        P: glue::HybridPredicate<Self::Id> + Send + Sync,
        F: FnMut(glue::Decision<Self::Id>, f32) + Send,
    {
        // Pilfer the `id_buffer` for the duration of this call to ensure a disjoint
        // borrow. We put it back at the end to save the allocation.
        let mut id_buffer = mem::take(&mut **self.id_buffer);

        for nl_id in ids {
            self.provider
                .get_neighbors(self.context, nl_id, &mut id_buffer);

            self.filtered_ids.clear();
            self.filtered_decisions.clear();

            for id in id_buffer.iter().copied().filter(|id| pred.eval_mut(id)) {
                if id == Self::START_ID {
                    let dist = match self.start_point_distance() {
                        Ok(dist) => dist,
                        Err(err) => return future::ready(Err(ANNError::from(err))),
                    };
                    on_neighbors(Decision::reject(id), dist);
                } else {
                    let matches = self.provider.callbacks.matches_filter(self.context, id);

                    self.filtered_ids.push(4);
                    self.filtered_ids.push(id);

                    self.filtered_decisions.push(matches);
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
                    .read_multi_lpiid(&ctx, &self.filtered_ids, |i, v| {
                        let dist = self.computer.evaluate_similarity(v);
                        let decision = if self.filtered_decisions[i as usize] {
                            Decision::accept(self.filtered_ids[i as usize * 2 + 1])
                        } else {
                            Decision::reject(self.filtered_ids[i as usize * 2 + 1])
                        };
                        on_neighbors(decision, dist);
                    });
            }
        }

        **self.id_buffer = id_buffer;
        future::ready(Ok(()))
    }

    fn expand_beam_accept_only<Itr, P, F>(
        &mut self,
        ids: Itr,
        mut pred: P,
        mut on_neighbors: F,
    ) -> impl future::Future<Output = ANNResult<()>> + Send
    where
        Itr: Iterator<Item = Self::Id> + Send,
        P: glue::Predicate<Self::Id> + glue::PredicateMut<Accept<Self::Id>> + Send + Sync,
        F: FnMut(glue::Accept<Self::Id>, f32) + Send,
    {
        // Pilfer the `id_buffer` for the duration of this call to ensure a disjoint
        // borrow. We put it back at the end to save the allocation.
        let mut id_buffer = mem::take(&mut **self.id_buffer);

        for nl_id in ids {
            self.provider
                .get_neighbors(self.context, nl_id, &mut id_buffer);
            self.filtered_ids.clear();

            for id in id_buffer.iter().copied() {
                if id != Self::START_ID && pred.eval(&id) {
                    let matches = self.provider.callbacks.matches_filter(self.context, id);

                    if matches && pred.eval_mut(&Accept::new(id)) {
                        self.filtered_ids.push(4);
                        self.filtered_ids.push(id);
                    }
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
                    .read_multi_lpiid(&ctx, &self.filtered_ids, |i, v| {
                        let dist = self.computer.evaluate_similarity(v);
                        on_neighbors(Accept::new(self.filtered_ids[i as usize * 2 + 1]), dist);
                    });
            }
        }

        **self.id_buffer = id_buffer;
        future::ready(Ok(()))
    }

    fn num_starting_points(&self) -> impl future::Future<Output = ANNResult<usize>> + Send {
        if self.provider.start_points_exist() {
            future::ready(Ok(1))
        } else {
            future::ready(Ok(0))
        }
    }
}

////////////
// Insert //
////////////

pub(crate) struct PruneAccessor<'a, T>
where
    T: VectorRepr,
{
    provider: &'a GarnetProvider<T>,
    context: &'a Context,
    quantized: bool,
    id_buffer: PooledRef<'a, AdjList>,
    filtered_ids: PooledRef<'a, Vec<u32>>,
    distance: GarnetDistanceComputer,
    set: workingset::Map<u32, Box<[u8]>>,
}

impl<'a, T> PruneAccessor<'a, T>
where
    T: VectorRepr,
{
    pub(crate) fn new(
        provider: &'a GarnetProvider<T>,
        context: &'a Context,
        quantized: bool,
        capacity: usize,
    ) -> Result<Self, GarnetProviderError> {
        let distance = if quantized && let Some(quantizer) = provider.quantizer() {
            quantizer.distance_computer()?
        } else {
            GarnetDistanceComputer::new(FullPrecisionDistance::<T>(T::distance(
                provider.metric_type,
                Some(provider.dim),
            )))
        };

        let id_buffer = provider
            .id_buffer_pool
            .get_ref(Undef::new(provider.max_degree + 1));

        // x2 to allow for the length prefixes for garnet
        let filtered_ids = provider
            .filtered_ids_pool
            .get_ref(Undef::new(MAX_OCCLUSION_SIZE.get() as usize * 2));

        // Using `Capacity::Default` means that the constructed working set will act as a
        // cache and persist up to `capacity` items across uses of the working set.
        //
        // This reuse is limited to a single collection of backedges for an insert or multi-insert.
        let set = workingset::map::Builder::new(workingset::map::Capacity::Default).build(capacity);

        let this = Self {
            provider,
            context,
            quantized,
            id_buffer,
            filtered_ids,
            distance,
            set,
        };

        Ok(this)
    }
}

impl<T> HasId for PruneAccessor<'_, T>
where
    T: VectorRepr,
{
    type Id = u32;
}

impl<T> glue::PruneAccessor for PruneAccessor<'_, T>
where
    T: VectorRepr,
{
    type ElementRef<'a> = &'a [u8];
    type View<'a>
        = workingset::map::View<'a, u32, Box<[u8]>>
    where
        Self: 'a;
    type Distance<'a>
        = &'a GarnetDistanceComputer
    where
        Self: 'a;
    type Neighbors<'a>
        = DelegateNeighborAccessor<'a, T>
    where
        Self: 'a;

    async fn fill<Itr>(&mut self, itr: Itr) -> ANNResult<(Self::View<'_>, Self::Distance<'_>)>
    where
        Itr: ExactSizeIterator<Item = Self::Id> + Clone + Send + Sync,
    {
        // Evict items from the working set to make room if needed.
        self.set.prepare(itr.clone());

        self.filtered_ids.clear();
        for id in itr {
            if id == 0 {
                if self.quantized
                    && let Entry::Vacant(e) = self.set.entry(id)
                {
                    if let Some(guard) = self.provider.start_point_quant_cache.get(&id) {
                        e.insert((&**guard).into());
                    } else {
                        return Err(GarnetProviderError::StartPoint.into());
                    }
                } else if let Entry::Vacant(e) = self.set.entry(id) {
                    if let Some(guard) = self.provider.start_point_cache.get(&id) {
                        e.insert((&**guard).into());
                    } else {
                        return Err(GarnetProviderError::StartPoint.into());
                    }
                } else {
                    continue;
                };
            } else if !self.set.contains_key(&id) {
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
                .read_multi_lpiid(&ctx, &self.filtered_ids, |id, v| {
                    self.set
                        .insert(self.filtered_ids[id as usize * 2 + 1], v.into());
                });
        }

        Ok((self.set.view(), &self.distance))
    }

    fn neighbors(&mut self) -> Self::Neighbors<'_> {
        DelegateNeighborAccessor {
            provider: self.provider,
            context: self.context,
            scratch: &mut self.id_buffer,
        }
    }
}

pub(crate) struct DelegateNeighborAccessor<'a, T>
where
    T: VectorRepr,
{
    provider: &'a GarnetProvider<T>,
    context: &'a Context,
    scratch: &'a mut AdjacencyList<u32>,
}

impl<T: VectorRepr> HasId for DelegateNeighborAccessor<'_, T> {
    type Id = u32;
}

impl<T: VectorRepr> NeighborAccessor for DelegateNeighborAccessor<'_, T> {
    fn get_neighbors(
        &mut self,
        id: Self::Id,
        neighbors: &mut AdjacencyList<Self::Id>,
    ) -> impl Future<Output = ANNResult<()>> + Send {
        let result = if self.provider.get_neighbors(self.context, id, neighbors) {
            Ok(())
        } else {
            Err(ANNError::from(GarnetProviderError::Garnet(
                GarnetError::Read,
            )))
        };

        future::ready(result)
    }
}

impl<T: VectorRepr> NeighborAccessorMut for DelegateNeighborAccessor<'_, T> {
    fn set_neighbors(
        &mut self,
        id: Self::Id,
        neighbors: &[Self::Id],
    ) -> impl Future<Output = ANNResult<()>> + Send {
        let result = self
            .provider
            .set_neighbors(self.context, id, neighbors, self.scratch)
            .map_err(ANNError::from);

        std::future::ready(result)
    }

    fn append_vector(
        &mut self,
        id: Self::Id,
        neighbors: &[Self::Id],
    ) -> impl Future<Output = ANNResult<()>> + Send {
        let result = self
            .provider
            .append_vector(self.context, id, neighbors)
            .map_err(ANNError::from);
        std::future::ready(result)
    }
}

////////////////
// Strategies //
////////////////

impl<'a, T: VectorRepr> SearchStrategy<'a, GarnetProvider<T>, &'a [T]> for DynamicQuantization {
    type SearchAccessor = DynamicAccessor<'a, T>;
    type SearchAccessorError = GarnetProviderError;

    fn search_accessor(
        &'a self,
        provider: &'a GarnetProvider<T>,
        context: &'a <GarnetProvider<T> as DataProvider>::Context,
        query: &'a [T],
    ) -> Result<Self::SearchAccessor, Self::SearchAccessorError> {
        let quantized = provider.is_quantized();
        DynamicAccessor::new(provider, context, query, quantized)
    }
}

impl<'a, T: VectorRepr> DefaultPostProcessor<'a, GarnetProvider<T>, &'a [T], GarnetId>
    for DynamicQuantization
{
    default_post_processor!(
        glue::Pipeline<glue::FilterStartPoints, glue::Pipeline<Rerank, CopyExternalIds>>
    );
}

impl<T: VectorRepr> PruneStrategy<GarnetProvider<T>> for DynamicQuantization {
    type PruneAccessor<'a> = PruneAccessor<'a, T>;
    type PruneAccessorError = GarnetProviderError;

    fn prune_accessor<'a>(
        &'a self,
        provider: &'a GarnetProvider<T>,
        context: &'a <GarnetProvider<T> as DataProvider>::Context,
        capacity: usize,
    ) -> Result<Self::PruneAccessor<'a>, Self::PruneAccessorError> {
        let quantized = provider.is_quantized();
        PruneAccessor::new(provider, context, quantized, capacity)
    }
}

impl<'a, T: VectorRepr> InsertStrategy<'a, GarnetProvider<T>, &'a [T]> for DynamicQuantization {
    type PruneStrategy = Self;

    fn insert_search_accessor(
        &'a self,
        provider: &'a GarnetProvider<T>,
        context: &'a <GarnetProvider<T> as DataProvider>::Context,
        vector: &'a [T],
    ) -> Result<Self::SearchAccessor, Self::SearchAccessorError> {
        let quantized = provider.is_quantized();
        DynamicAccessor::new(provider, context, vector, quantized)
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
        if !provider.callbacks.read_single_iid(context, id, &mut v) {
            return future::ready(Err(GarnetError::Read.into()));
        }
        future::ready(Ok(v.into()))
    }
}

#[cfg(test)]
mod tests {
    use std::mem;

    use diskann::{
        graph::{
            config::{self, defaults::GRAPH_SLACK_FACTOR},
            search,
        },
        provider::{Delete, SetElement},
    };
    use diskann_providers::index::wrapped_async::DiskANNIndex;
    use diskann_vector::distance::Metric;
    use rand::Rng;

    use crate::{
        SearchResults, VectorQuantType,
        dyn_index::DynIndex,
        garnet::{Context, GarnetId, Term},
        provider::{GarnetProvider, QUANT_STATE_KEY},
        quantization::{GarnetQuantizer, Spherical1Bit},
        test_utils::Store,
    };

    #[tokio::test]
    async fn simple_insert_delete() {
        let store = Store::new();
        let ctx = Context::new(0);
        let provider = GarnetProvider::<f32>::new(
            2,
            VectorQuantType::NoQuant,
            Metric::L2,
            10,
            store.callbacks(),
            &ctx,
        )
        .unwrap();

        let id = GarnetId::from(bytemuck::bytes_of(&0));

        let res = provider.set_element(&ctx, &id, &[0f32, 0f32]).await;
        assert!(res.is_ok());

        let res = provider.delete(&ctx, &id).await;
        assert!(res.is_ok());
    }

    fn create_2d_f32_index(
        quant_type: VectorQuantType,
        metric: Metric,
        store: &Store,
        ctx: &Context,
    ) -> DiskANNIndex<GarnetProvider<f32>> {
        let provider =
            GarnetProvider::<f32>::new(2, quant_type, metric, 10, store.callbacks(), ctx).unwrap();

        let config = config::Builder::new(
            (10.0 / GRAPH_SLACK_FACTOR) as usize,
            config::MaxDegree::Value(10),
            10,
            metric.into(),
        )
        .build()
        .unwrap();

        DiskANNIndex::new_with_current_thread_runtime(config, provider)
    }

    /// Test that restarts during phase one quant bootstrap work.
    /// Phase one is all index activity before the index has the required
    /// number of vectors to begin quantization.
    #[test]
    fn restart_during_quant_bootstrap_phase_one() {
        let store = Store::new();
        let ctx = Context::new(0);
        let index = create_2d_f32_index(VectorQuantType::Bin, Metric::L2, &store, &ctx);
        let provider = index.inner.provider();
        let required_vecs = Spherical1Bit::new(2).required_vectors();

        let mut rng = rand::rng();

        let mut last_inserted_id = 0;
        let mut first_insert = true;
        for id in 0..required_vecs as u32 / 2 {
            let v = [rng.random(), rng.random()];

            if first_insert {
                provider.maybe_set_start_point(&ctx, &v).unwrap();
                first_insert = false;
            }

            DynIndex::insert(
                &index,
                &ctx,
                &GarnetId::from(bytemuck::bytes_of::<u32>(&id)),
                bytemuck::cast_slice::<f32, u8>(&v),
            )
            .unwrap();
            last_inserted_id = id;
        }

        assert!(!provider.is_quantized());
        let max_id = provider.max_internal_id();
        assert_eq!(max_id, last_inserted_id + 1);

        // There should be no saved quant state.
        assert!(
            !provider
                .callbacks
                .exists_iid(&ctx.term(Term::Metadata), QUANT_STATE_KEY),
            "quant state should not be stored yet"
        );

        // Quantization is not needed yet
        assert!(!provider.quantization_needed());

        let params = search::Knn::new(10, 10, None).unwrap();
        let mut output_ids = vec![0u8; mem::size_of::<u32>() * 2 * 10];
        let mut output_dists = vec![0f32; 10];
        let mut output = SearchResults::new(
            output_ids.as_mut_ptr(),
            output_ids.len(),
            output_dists.as_mut_ptr(),
            output_dists.len(),
        );
        let query = [0.0f32, 0.0f32];
        let results = DynIndex::search_vector(
            &index,
            &ctx,
            bytemuck::cast_slice::<f32, u8>(&query),
            params,
            &mut output,
        )
        .unwrap();

        assert_eq!(results.result_count, 10);
    }

    /// Test that restarts during phase two quant bootstrap work.
    /// Phase two starts when there are enough vectors to begin quantizing, and
    /// lasts until quant vector backfill is complete.
    #[test]
    fn restart_during_quant_bootstrap_phase_two() {
        let store = Store::new();
        let ctx = Context::new(0);
        let index = create_2d_f32_index(VectorQuantType::Bin, Metric::L2, &store, &ctx);
        let provider = index.inner.provider();
        let required_vecs = Spherical1Bit::new(2).required_vectors();

        let mut rng = rand::rng();

        let mut last_inserted_id = 0;
        let mut first_insert = true;
        for id in 0..required_vecs as u32 + 100 {
            let v = [rng.random(), rng.random()];

            if first_insert {
                provider.maybe_set_start_point(&ctx, &v).unwrap();
                first_insert = false;
            }

            DynIndex::insert(
                &index,
                &ctx,
                &GarnetId::from(bytemuck::bytes_of::<u32>(&id)),
                bytemuck::cast_slice::<f32, u8>(&v),
            )
            .unwrap();
            last_inserted_id = id;
        }

        // Train the quantizer
        assert!(provider.train_quantizer(&ctx));

        // is_quantized won't be true until backfill is complete
        assert!(!provider.is_quantized());
        let max_id = provider.max_internal_id();
        assert_eq!(max_id, last_inserted_id + 1);

        // There should be saved quant state.
        assert!(
            provider
                .callbacks
                .exists_iid(&ctx.term(Term::Metadata), QUANT_STATE_KEY),
            "quant state missing"
        );

        let tqs = provider
            .callbacks
            .read_varsize_iid::<u8>(&ctx.term(Term::Metadata), QUANT_STATE_KEY)
            .unwrap();
        assert!(tqs.len() > 1, "quant state too small");
        assert_eq!(tqs[0], 0, "quant state should be pre-backfill");

        // Drop and re-create the index, keeping the same backing store
        let index = create_2d_f32_index(VectorQuantType::Bin, Metric::L2, &store, &ctx);
        let provider = index.inner.provider();

        assert!(!provider.is_quantized());
        let max_id = provider.max_internal_id();
        assert_eq!(max_id, last_inserted_id + 1);

        // Quant should be needed now, since backfill has never run
        assert!(provider.quantization_needed());

        // Quant state should be deserialized and able to compress
        let tv = [1.0f32, -1.0];
        let mut tqv = vec![
            0u8;
            provider
                .quantizer
                .as_ref()
                .expect("quantizer_missing")
                .bytes()
        ];
        assert!(
            provider
                .quantizer
                .as_ref()
                .expect("quantizer missing")
                .compress(&tv, &mut tqv)
                .is_ok(),
            "quant compression failed"
        );
    }

    /// Test that restarts during phase three quant bootstrap work.
    /// Phase three starts once backfill is complete and lasts for the remaining
    /// life of the index.
    #[test]
    fn restart_during_quant_bootstrap_phase_three() {
        let store = Store::new();
        let ctx = Context::new(0);
        let index = create_2d_f32_index(VectorQuantType::Bin, Metric::L2, &store, &ctx);
        let provider = index.inner.provider();
        let required_vecs = Spherical1Bit::new(2).required_vectors();

        let mut rng = rand::rng();

        let mut last_inserted_id = 0;
        let mut first_insert = true;
        for id in 0..required_vecs as u32 + 100 {
            let v = [rng.random(), rng.random()];

            if first_insert {
                provider.maybe_set_start_point(&ctx, &v).unwrap();
                first_insert = false;
            }

            DynIndex::insert(
                &index,
                &ctx,
                &GarnetId::from(bytemuck::bytes_of::<u32>(&id)),
                bytemuck::cast_slice::<f32, u8>(&v),
            )
            .unwrap();
            last_inserted_id = id;
        }

        // Train the quantizer
        assert!(provider.train_quantizer(&ctx));

        // Run backfill
        for job_id in 0..4 {
            provider.backfill_quant_vectors(&ctx, job_id, 4);
        }

        // Drop and re-create the index, keeping the same backing store
        let index = create_2d_f32_index(VectorQuantType::Bin, Metric::L2, &store, &ctx);
        let provider = index.inner.provider();

        // Index should think it is fully quantized
        assert!(provider.is_quantized());

        // Quantization should not be needed anymore
        assert!(!provider.quantization_needed());

        // There should be saved quant state.
        assert!(
            provider
                .callbacks
                .exists_iid(&ctx.term(Term::Metadata), QUANT_STATE_KEY),
            "quant state missing"
        );

        // all quantized state should match index
        let tqs = provider
            .callbacks
            .read_varsize_iid::<u8>(&ctx.term(Term::Metadata), QUANT_STATE_KEY)
            .unwrap();
        assert!(tqs.len() > 1, "quant state too small");
        assert_eq!(tqs[0], 1, "quant state should be post-backfill");

        // Every quant vector should be present in the store
        for id in 0..last_inserted_id {
            assert!(
                provider
                    .callbacks
                    .exists_iid(&ctx.term(Term::Quantized), id)
            );
        }

        // Searches should still work and use quantized vectors
        let params = search::Knn::new(10, 10, None).unwrap();
        let mut output_ids = vec![0u8; mem::size_of::<u32>() * 2 * 10];
        let mut output_dists = vec![0f32; 10];
        let mut output = SearchResults::new(
            output_ids.as_mut_ptr(),
            output_ids.len(),
            output_dists.as_mut_ptr(),
            output_dists.len(),
        );
        let query = [0.0f32, 0.0f32];

        store.clear_read_counts();

        let results = DynIndex::search_vector(
            &index,
            &ctx,
            bytemuck::cast_slice::<f32, u8>(&query),
            params,
            &mut output,
        )
        .unwrap();

        assert_eq!(results.result_count, 10);

        // Should be some full reads for reranking, but most reads should be
        // quantized
        assert!(store.full_reads() < store.quant_reads());
    }
}
