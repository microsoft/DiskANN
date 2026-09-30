/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use dashmap::DashMap;
use diskann::{
    ANNError, ANNResult, default_post_processor,
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
        DataProvider, Delete, ElementStatus, Guard, HasId, NeighborAccessor, NeighborAccessorMut,
        SetElement,
    },
    utils::VectorRepr,
};
use diskann_quantization::alloc::{AllocatorError, Poly};
use diskann_utils::views::Matrix;
use diskann_utils::{
    object_pool::{AsPooled, ObjectPool, PooledRef, Undef},
    views::MatrixView,
};
use diskann_vector::{
    DistanceFunction, PreprocessedDistanceFunction, contains::ContainsSimd, distance::Metric,
};
use std::{
    any::TypeId,
    collections::HashSet,
    future,
    hash::BuildHasher,
    marker::PhantomData,
    mem,
    ops::{Deref, DerefMut, Range},
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
};
use thiserror::Error;
use tokio::sync::watch;

use crate::{
    SearchResults, VectorQuantType,
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

/// Starting capacity of the pre-allocated rerank buffers.
const RERANK_BUFFER_LENGTH: usize = 1024;

/// Size hint passed to Garnet when batch reading attributes. Attributes are variable
/// length, so this is only an estimate used to size Garnet's read buffer.
const ATTRIBUTE_LENGTH_HINT: usize = 1024;

/// Maximum number of reservation retries after waiting for another owner.
const RESERVATION_RETRY_LIMIT: usize = 5;

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
    #[error("External ID reservation retry limit reached")]
    ReservationRetryLimit,
}

diskann::convert_error!(GarnetProviderError);
diskann::always_escalate!(GarnetProviderError);

struct BackfillGuard {
    ranges: Arc<Mutex<HashSet<Range<u32>>>>,
    notify: Arc<Condvar>,
    range: Range<u32>,
}

impl Drop for BackfillGuard {
    fn drop(&mut self) {
        let _ = self.ranges.lock().unwrap().remove(&self.range);
        self.notify.notify_all();
    }
}

pub(crate) struct ExternalIdGuard<'a> {
    pending: &'a DashMap<u64, watch::Sender<()>, foldhash::fast::RandomState>,
    id_hash: u64,
}

impl Drop for ExternalIdGuard<'_> {
    fn drop(&mut self) {
        self.pending.remove(&self.id_hash);
    }
}

pub(crate) struct InsertGuard {
    callbacks: Callbacks,
    context: Context,
    external_id: GarnetId,
    internal_id: u32,
    fsm: Arc<FreeSpaceMap>,
    original: Option<[(Context, Option<Vec<u8>>); 4]>,
    completed: bool,
    _backfill: Option<BackfillGuard>,
}

impl std::fmt::Debug for InsertGuard {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("InsertGuard")
            .field("internal_id", &self.internal_id)
            .field("completed", &self.completed)
            .finish_non_exhaustive()
    }
}

impl Guard for InsertGuard {
    type Id = u32;

    async fn complete(mut self) {
        self.completed = true;
    }

    fn id(&self) -> u32 {
        self.internal_id
    }
}

impl Drop for InsertGuard {
    fn drop(&mut self) {
        if self.completed {
            return;
        }

        let mut restored = true;
        if let Some(original) = self.original.take() {
            for (context, value) in original {
                restored &= match value {
                    Some(value) => self.callbacks.write_iid(&context, self.internal_id, &value),
                    None => {
                        self.callbacks.delete_iid(&context, self.internal_id)
                            || !self.callbacks.exists_iid(&context, self.internal_id, 0)
                    }
                };
            }
        } else {
            for term in [
                Term::Vector,
                Term::Quantized,
                Term::Attributes,
                Term::Neighbors,
                Term::ExtMap,
            ] {
                let context = self.context.term(term);
                restored &= self.callbacks.delete_iid(&context, self.internal_id)
                    || !self.callbacks.exists_iid(&context, self.internal_id, 0);
            }
            let context = self.context.term(Term::IntMap);
            restored &= self.callbacks.delete_eid(&context, &self.external_id)
                || !self
                    .callbacks
                    .exists_eid(&context, &self.external_id, mem::size_of::<u32>());
            if restored {
                restored = self.fsm.mark_free(&self.context, self.internal_id).is_ok();
            }
        }
        if !restored {
            self.callbacks.log(
                &self.context,
                &format!(
                    "Error: insert rollback failed for ID {:?}; stored terms may be inconsistent.",
                    self.external_id,
                ),
            );
        }
    }
}

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
    pending_external_ids: DashMap<u64, watch::Sender<()>, foldhash::fast::RandomState>,
    /// The quantizer the index will use, or None if NOQUANT is used.
    quantizer: Option<Box<dyn GarnetQuantizer>>,
    /// Tracks whether the index is ready to operate fully quantized.
    all_quantized: AtomicBool,
    /// Per job tracker for quantization backfill completion
    backfills_completed: AtomicU64,
    /// Lock for active backfill and update ranges
    backfill_lock: Arc<Mutex<HashSet<Range<u32>>>>,
    /// Signals released range reservations
    backfill_notify: Arc<Condvar>,
    /// Lock to ensure training only happens once.
    training_lock: Mutex<()>,
    /// Pool of pre-allocated buffers to use for neighbor lists
    id_buffer_pool: ObjectPool<AdjList>,
    /// Pool of pre-allocated buffers to use for IDs
    filtered_ids_pool: ObjectPool<Vec<u32>>,
    /// Pool of pre-allocated buffers to use for filter decisions during
    /// filtered search beam expansion
    filtered_decisions_pool: ObjectPool<Vec<bool>>,
    /// Pool of pre-allocated buffers to use for reranking
    rerank_pool: ObjectPool<Vec<Neighbor<u32>>>,
    /// Pool of pre-allocated buffers to use for quantizing vectors
    quant_buffer_pool: ObjectPool<Vec<u8>>,
    /// Small cache for the start points' neighbors
    neighbor_cache: DashMap<u32, Vec<u32>, foldhash::fast::RandomState>,
    /// Small cache for the start points' full precision vector data
    start_point_cache: DashMap<u32, Poly<[u8], AlignToEight>, foldhash::fast::RandomState>,
    /// Small cache for the start points' quantized vector data
    start_point_quant_cache: DashMap<u32, Poly<[u8], AlignToEight>, foldhash::fast::RandomState>,
    /// Free space map to track internal IDs
    fsm: Arc<FreeSpaceMap>,
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
        let rerank_pool = ObjectPool::new(
            Undef::new(RERANK_BUFFER_LENGTH),
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

                let quantizer = if let Some(quant_state) =
                    callbacks.read_varsize_iid::<u8>(&context.term(Term::Metadata), QUANT_STATE_KEY)
                {
                    quantization::MinMax8Bit::new_from_bytes(metric_type, &quant_state)?
                } else {
                    if start_point_cache.contains_key(&0) {
                        // If we have a start point, we should have had a quantizer.
                        return Err(GarnetProviderError::InvalidQuantizer);
                    }
                    let quantizer = quantization::MinMax8Bit::new(dim, metric_type)?;

                    if !callbacks.write_iid(
                        &context.term(Term::Metadata),
                        QUANT_STATE_KEY,
                        &quantizer.serialize()?,
                    ) {
                        return Err(GarnetError::Write.into());
                    }

                    quantizer
                };

                let quantizer = Box::new(quantizer) as Box<dyn GarnetQuantizer>;
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
            quantizer
                .as_ref()
                .is_some_and(|quantizer| quantizer.is_trained()),
            quantizer.is_none() || all_quantized,
        )?;

        Ok(Self {
            dim,
            metric_type,
            max_degree,
            callbacks,
            pending_external_ids: DashMap::with_hasher(foldhash::fast::RandomState::default()),
            quantizer,
            all_quantized: AtomicBool::new(all_quantized),
            backfills_completed: AtomicU64::new(0),
            backfill_lock: Arc::new(Mutex::new(HashSet::new())),
            backfill_notify: Arc::new(Condvar::new()),
            training_lock: Mutex::new(()),
            id_buffer_pool,
            filtered_ids_pool,
            filtered_decisions_pool,
            rerank_pool,
            quant_buffer_pool,
            start_point_cache,
            start_point_quant_cache,
            neighbor_cache,
            fsm: Arc::new(fsm),
            _phantom: PhantomData,
        })
    }

    pub(crate) async fn reserve_external_id(
        &self,
        id: &GarnetId,
    ) -> Result<ExternalIdGuard<'_>, GarnetProviderError> {
        let id_hash = self.pending_external_ids.hasher().hash_one(&id[..]);
        for retry in 0..=RESERVATION_RETRY_LIMIT {
            let mut receiver = match self.pending_external_ids.entry(id_hash) {
                dashmap::mapref::entry::Entry::Occupied(_) if retry == RESERVATION_RETRY_LIMIT => {
                    break;
                }
                dashmap::mapref::entry::Entry::Occupied(entry) => entry.get().subscribe(),
                dashmap::mapref::entry::Entry::Vacant(entry) => {
                    let (sender, _) = watch::channel(());
                    entry.insert(sender);
                    return Ok(ExternalIdGuard {
                        pending: &self.pending_external_ids,
                        id_hash,
                    });
                }
            };
            let _ = receiver.changed().await;
        }
        Err(GarnetProviderError::ReservationRetryLimit)
    }

    fn reserve_backfill_range(&self, range: Range<u32>) -> Option<BackfillGuard> {
        if range.is_empty() {
            return None;
        }

        let mut ranges = self.backfill_lock.lock().unwrap();
        while ranges
            .iter()
            .any(|active| active.start < range.end && range.start < active.end)
        {
            ranges = self.backfill_notify.wait(ranges).unwrap();
        }
        let _ = ranges.insert(range.clone());
        Some(BackfillGuard {
            ranges: self.backfill_lock.clone(),
            notify: self.backfill_notify.clone(),
            range,
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
        let mut iid = u32::MAX;
        if !self.callbacks.read_single_eid(
            &context.term(Term::IntMap),
            id,
            bytemuck::bytes_of_mut(&mut iid),
        ) {
            return Err(GarnetError::Read.into());
        }

        if self
            .callbacks
            .write_iid(&context.term(Term::Attributes), iid, data)
        {
            Ok(())
        } else {
            Err(GarnetError::Write.into())
        }
    }

    pub(crate) fn delete_attributes(
        &self,
        context: &Context,
        id: &GarnetId,
    ) -> Result<(), GarnetProviderError> {
        let mut iid = u32::MAX;
        if !self.callbacks.read_single_eid(
            &context.term(Term::IntMap),
            id,
            bytemuck::bytes_of_mut(&mut iid),
        ) {
            return Err(GarnetError::Read.into());
        }

        if self
            .callbacks
            .delete_iid(&context.term(Term::Attributes), iid)
        {
            Ok(())
        } else {
            Err(GarnetError::Delete.into())
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

    pub(crate) fn max_degree(&self) -> usize {
        self.max_degree
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
            Some(q) if !q.is_trained() => q,
            None => return false,
            Some(_) => return false,
        };

        let rows = quantizer.required_vectors();
        let mut data = Matrix::from_element(rows, self.dim, T::default());
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
        let view = match MatrixView::try_from(&*converted, view.nrows(), view.ncols()) {
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
    ) -> bool {
        let quantizer = match &self.quantizer {
            Some(q) => q,
            None => {
                self.callbacks.log(
                    &context.term(Term::Quantized),
                    "Error: backfill_quant_vectors: Quantizer not found. Index will operate full precision only mode.",
                );
                return false;
            }
        };

        let max_id = self.fsm.max_id_for_backfill() as usize;
        if max_id >= u32::MAX as usize {
            // The max_id was somehow not sampled, so bail.
            self.callbacks.log(
                &context.term(Term::Quantized),
                "Error: backfill_quant_vectors: Couldn't calculate max id to backfill. Index will operate full precision only mode.",
            );
            return false;
        }

        // If we have more tasks than vectors to backfill, we exit the extra tasks early.
        let task_count = task_count.min(max_id + 1);
        if task_idx >= task_count {
            self.callbacks.log(
                &context.term(Term::Quantized),
                "Error: backfill_quant_vectors: Bad task index. Index will operate full precision only mode.",
            );
            return false;
        }

        // Evenly divide the ID range from 0..max_id and determine this thread's backfill
        // range.
        let work_count = (max_id + 1).div_ceil(task_count); // will be >= 1
        let start_id = (work_count * task_idx) as u32;
        let end_id = (work_count * (task_idx + 1)).min(max_id + 1) as u32;

        let _backfill_guard = self.reserve_backfill_range(start_id..end_id);
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
                    self.callbacks.log(
                        &context.term(Term::Quantized),
                        "Error quantizing start point; failed to finish backfill. Index will operate full precision only mode.",
                    );
                    return false;
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
                self.callbacks.log(
                    &context.term(Term::Quantized),
                    "Error saving quantizer state; failed to finish backfill. Index will operate full precision only mode.",
                );
                return false;
            }

            // Signal to the index that it is now safe to operate in quantized mode.
            self.all_quantized.store(true, Ordering::Release);
        }

        true
    }

    pub(crate) fn random_members(
        &self,
        context: &Context,
        count: u32,
        output: &mut SearchResults<'_>,
    ) -> bool {
        let mut rng = rand::rng();

        let id_space = self.max_internal_id() as usize + 1;
        let total_vectors = self.fsm.total_used();
        let mut remaining = (count as usize).min(total_vectors);
        let mut chosen = HashSet::new();

        // Deletions leave holes in the ID space, so scale the first batch by the density of
        // live IDs, then grow it until the request is satisfied or the whole space is covered.
        let mut batch = remaining
            .saturating_mul(id_space)
            .div_ceil(total_vectors.max(1))
            .clamp(1, id_space);

        while remaining > 0 {
            for samp in rand::seq::index::sample(&mut rng, id_space, batch) {
                let samp = samp as u32;
                if !chosen.insert(samp) {
                    // Already considered on an earlier round.
                    continue;
                }
                let Ok(eid) = self.to_external_id(context, samp) else {
                    // Deleted or otherwise unreadable.
                    continue;
                };

                let state = output.push_id(eid);
                remaining -= 1;
                if remaining == 0 || state == diskann::graph::BufferState::Full {
                    return true;
                }
            }

            if batch == id_space {
                // The whole ID space was scanned; fewer live IDs exist than were requested.
                break;
            }
            batch = batch.saturating_mul(2).min(id_space);
        }

        true
    }

    pub(crate) fn neighbors(
        &self,
        context: &Context,
        id: &GarnetId,
    ) -> ANNResult<Vec<Neighbor<GarnetId>>> {
        let iid = self.to_internal_id(context, id)?;
        let v = self.get_full_vector(context, iid)?;
        let mut neighbors = AdjacencyList::with_capacity(self.max_degree + 1);

        if !self.get_neighbors(context, iid, &mut neighbors) {
            return Err(GarnetProviderError::Garnet(GarnetError::Read).into());
        }

        let d = <T as VectorRepr>::distance(self.metric_type, Some(self.dim));
        let mut result = Vec::with_capacity(self.max_degree);
        for &nbr_id in neighbors.iter() {
            if nbr_id == 0 {
                // Skip the start point
                continue;
            }
            let nbr_v = self.get_full_vector(context, nbr_id)?;
            let nbr_eid = self.to_external_id(context, nbr_id)?;
            let nbr_d = d.evaluate_similarity(&v, &nbr_v);
            result.push(Neighbor::new(nbr_eid, nbr_d));
        }

        Ok(result)
    }

    /// Log a message to Garnet.
    pub(crate) fn log(&self, context: &Context, msg: &str) {
        self.callbacks.log(context, msg);
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

    /// The size of a stored full vector.
    fn full_vector_size(&self) -> usize {
        self.dim * mem::size_of::<T>()
    }

    /// The size of a stored quant vector.
    fn quant_vector_size(&self) -> usize {
        if let Some(quantizer) = &self.quantizer {
            quantizer.bytes()
        } else {
            0
        }
    }

    /// Provides an estimate of quantizer state size.
    /// This is allowed to be wrong, but should ideally be an overestimate.
    #[cfg(test)]
    fn quant_state_size(&self) -> usize {
        self.dim * 6 + 128
    }
}

impl<T: VectorRepr> DataProvider for GarnetProvider<T> {
    type Context = Context;
    type InternalId = u32;
    type ExternalId = GarnetId;
    type Error = GarnetProviderError;
    type Guard = InsertGuard;

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

impl<T: VectorRepr> SetElement<(&[T], &[u8])> for GarnetProvider<T> {
    type SetError = GarnetProviderError;

    async fn set_element(
        &self,
        context: &Self::Context,
        id: &Self::ExternalId,
        element: (&[T], &[u8]),
    ) -> Result<Self::Guard, Self::SetError> {
        let (internal_id, is_update) = match self.to_internal_id(context, id) {
            Ok(existing_id) => {
                context.set_insert_is_update();
                (self.fsm.existing_id(existing_id), true)
            }
            Err(_) => (self.fsm.next_id(context)?, false),
        };

        let backfill_guard = if self.quantizer.is_some()
            && !self.all_quantized.load(Ordering::Acquire)
            && (!internal_id.should_quantize()
                || internal_id.id() <= internal_id.max_id_for_backfill())
        {
            let end_id = internal_id
                .id()
                .checked_add(1)
                .ok_or(FsmError::IdOutOfRange(internal_id.id()))?;
            self.reserve_backfill_range(internal_id.id()..end_id)
        } else {
            None
        };

        let guard = InsertGuard {
            callbacks: self.callbacks,
            context: context.clone(),
            external_id: id.clone(),
            internal_id: internal_id.id(),
            fsm: self.fsm.clone(),
            original: is_update.then(|| {
                [
                    Term::Vector,
                    Term::Quantized,
                    Term::Attributes,
                    Term::Neighbors,
                ]
                .map(|term| {
                    let context = context.term(term);
                    let value = self
                        .callbacks
                        .read_varsize_iid::<u8>(&context, internal_id.id());
                    (context, value)
                })
            }),
            completed: false,
            _backfill: backfill_guard,
        };

        // Set quantization readiness
        if let Some(quantizer) = &self.quantizer
            && !internal_id.should_quantize()
            && !quantizer.is_trained()
            && self.fsm.total_used() > quantizer.required_vectors()
        {
            context.set_quantizer_ready();
        }

        let mut error_term = Term::Vector;
        let mut insert = || -> Result<(), Self::SetError> {
            self.callbacks
                .write_iid(&context.term(Term::Vector), internal_id.id(), element.0)
                .then_some(())
                .ok_or(GarnetError::Write)?;
            if let Some(quantizer) = &self.quantizer
                && internal_id.should_quantize()
            {
                error_term = Term::Quantized;
                let mut quant = self
                    .quant_buffer_pool
                    .get_ref(Undef::new(quantizer.bytes()));
                let element_f32 = T::as_f32(element.0).map_err(|e| {
                    GarnetProviderError::Quantizer(GarnetQuantizerError::Compression(Box::new(e)))
                })?;
                quantizer.compress(&element_f32, &mut quant)?;
                self.callbacks
                    .write_iid(&context.term(Term::Quantized), internal_id.id(), &quant)
                    .then_some(())
                    .ok_or(GarnetError::Write)?;
            }
            if !element.1.is_empty() {
                error_term = Term::Attributes;
                self.callbacks
                    .write_iid(&context.term(Term::Attributes), internal_id.id(), element.1)
                    .then_some(())
                    .ok_or(GarnetError::Write)?;
            }
            if !is_update {
                error_term = Term::ExtMap;
                self.callbacks
                    .write_iid(&context.term(Term::ExtMap), internal_id.id(), id)
                    .then_some(())
                    .ok_or(GarnetError::Write)?;
                error_term = Term::IntMap;
                self.callbacks
                    .write_eid(
                        &context.term(Term::IntMap),
                        id,
                        bytemuck::bytes_of(&internal_id.id()),
                    )
                    .then_some(())
                    .ok_or(GarnetError::Write)?;
            }
            Ok(())
        };

        match insert() {
            Ok(()) => (),
            Err(e) if is_update => {
                self.callbacks.log(
                    &context.term(error_term),
                    &format!("Error: update failed for ID {id:?}, term {error_term:?}: {e}."),
                );
                return Err(e);
            }
            Err(e) => return Err(e),
        }

        Ok(guard)
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
            .delete_iid(&context.term(Term::Attributes), id);

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

    /// Batch read the attributes for `filtered_ids` and record the filter result for each
    /// into `filtered_decisions`.
    ///
    /// Garnet skips ids with no stored attributes, so those keep `default_decision`.
    fn compute_filter_decisions(&mut self, default_decision: bool) {
        let Self {
            provider,
            context,
            filtered_ids,
            filtered_decisions,
            ..
        } = self;

        filtered_decisions.clear();
        filtered_decisions.resize(filtered_ids.len() / 2, default_decision);

        provider.callbacks.read_multi_lpiid::<_, u8>(
            &context.term(Term::Attributes),
            filtered_ids,
            ATTRIBUTE_LENGTH_HINT,
            |i, attrs| {
                filtered_decisions[i as usize] = provider.callbacks.matches_filter(context, attrs);
            },
        );
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

            let (ctx, length_hint) = if self.quantized {
                (
                    self.context.term(Term::Quantized),
                    self.provider.quant_vector_size(),
                )
            } else {
                (
                    self.context.term(Term::Vector),
                    self.provider.full_vector_size(),
                )
            };

            if !self.filtered_ids.is_empty() {
                self.provider.callbacks.read_multi_lpiid(
                    &ctx,
                    &self.filtered_ids,
                    length_hint,
                    |i, v| {
                        let dist = self.computer.evaluate_similarity(v);
                        on_neighbors(self.filtered_ids[i as usize * 2 + 1], dist);
                    },
                );
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
            let id = match accessor.provider.to_external_id(accessor.context, *n.id()) {
                Ok(id) => id,
                Err(_) => continue, // Can't read the mapping; skip.
            };

            if output.push(Neighbor::new(id, *n.distance())).is_full() {
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

        let provider = accessor.provider;
        let f = T::distance(provider.metric_type, Some(provider.dim));

        let mut reranked = provider
            .rerank_pool
            .get_ref(Undef::new(RERANK_BUFFER_LENGTH));
        reranked.clear();

        // Use the accessor.filtered_ids pre-allocated buffer to do a multi read from Garnet, placing the results in
        // the rerank buffer.
        accessor.filtered_ids.clear();
        for nbor in candidates {
            accessor.filtered_ids.push(4);
            accessor.filtered_ids.push(*nbor.id());
        }

        if !accessor.filtered_ids.is_empty() {
            provider.callbacks.read_multi_lpiid(
                &accessor.context.term(Term::Vector),
                &accessor.filtered_ids,
                provider.full_vector_size(),
                |i, v| {
                    let dist = f.evaluate_similarity(query, bytemuck::cast_slice::<u8, T>(v));
                    reranked.push(Neighbor::new(
                        accessor.filtered_ids[i as usize * 2 + 1],
                        dist,
                    ));
                },
            );
        }

        // Sort the full precision distances.
        reranked.sort_unstable_by(diskann::neighbor::ord::fast_distance);

        next.post_process(accessor, query, reranked.iter().copied(), output)
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

        let default_decision = self.provider.callbacks.matches_filter(self.context, &[]);

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
                    on_neighbors(Decision::reject(id), dist);
                } else {
                    self.filtered_ids.push(4);
                    self.filtered_ids.push(id);
                }
            }

            if self.filtered_ids.is_empty() {
                continue;
            }

            let (ctx, length_hint) = if self.quantized {
                (
                    self.context.term(Term::Quantized),
                    self.provider.quant_vector_size(),
                )
            } else {
                (
                    self.context.term(Term::Vector),
                    self.provider.full_vector_size(),
                )
            };

            self.compute_filter_decisions(default_decision);

            // Read vectors and calculate distances
            self.provider.callbacks.read_multi_lpiid(
                &ctx,
                &self.filtered_ids,
                length_hint,
                |i, v| {
                    let dist = self.computer.evaluate_similarity(v);
                    let decision = if self.filtered_decisions[i as usize] {
                        Decision::accept(self.filtered_ids[i as usize * 2 + 1])
                    } else {
                        Decision::reject(self.filtered_ids[i as usize * 2 + 1])
                    };
                    on_neighbors(decision, dist);
                },
            );
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

        let default_decision = self.provider.callbacks.matches_filter(self.context, &[]);

        for nl_id in ids {
            self.provider
                .get_neighbors(self.context, nl_id, &mut id_buffer);
            self.filtered_ids.clear();

            for id in id_buffer.iter().copied() {
                if id != Self::START_ID && pred.eval(&id) {
                    self.filtered_ids.push(4);
                    self.filtered_ids.push(id);
                }
            }

            if self.filtered_ids.is_empty() {
                continue;
            }

            self.compute_filter_decisions(default_decision);

            // Remove non-matching ids
            let mut index = 0;
            for (i, &matches) in self.filtered_decisions.iter().enumerate() {
                if !matches {
                    continue;
                }

                let id = self.filtered_ids[i * 2 + 1];

                if pred.eval_mut(&Accept::new(id)) {
                    self.filtered_ids[index * 2] = 4;
                    self.filtered_ids[index * 2 + 1] = id;
                    index += 1;
                }
            }
            self.filtered_ids.truncate(index * 2);

            if self.filtered_ids.is_empty() {
                continue;
            }

            let (ctx, length_hint) = if self.quantized {
                (
                    self.context.term(Term::Quantized),
                    self.provider.quant_vector_size(),
                )
            } else {
                (
                    self.context.term(Term::Vector),
                    self.provider.full_vector_size(),
                )
            };

            self.provider.callbacks.read_multi_lpiid(
                &ctx,
                &self.filtered_ids,
                length_hint,
                |i, v| {
                    let dist = self.computer.evaluate_similarity(v);
                    on_neighbors(Accept::new(self.filtered_ids[i as usize * 2 + 1]), dist);
                },
            );
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

        let (ctx, length_hint) = if self.quantized {
            (
                self.context.term(Term::Quantized),
                self.provider.quant_vector_size(),
            )
        } else {
            (
                self.context.term(Term::Vector),
                self.provider.full_vector_size(),
            )
        };

        if !self.filtered_ids.is_empty() {
            self.provider.callbacks.read_multi_lpiid(
                &ctx,
                &self.filtered_ids,
                length_hint,
                |id, v| {
                    self.set
                        .insert(self.filtered_ids[id as usize * 2 + 1], v.into());
                },
            );
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

impl<'a, T: VectorRepr> InsertStrategy<'a, GarnetProvider<T>, (&'a [T], &'a [u8])>
    for DynamicQuantization
{
    type SearchAccessor = DynamicAccessor<'a, T>;
    type SearchAccessorError = GarnetProviderError;

    type PruneStrategy = Self;

    fn insert_search_accessor(
        &'a self,
        provider: &'a GarnetProvider<T>,
        context: &'a <GarnetProvider<T> as DataProvider>::Context,
        vector_and_attrs: (&'a [T], &'a [u8]),
    ) -> Result<Self::SearchAccessor, Self::SearchAccessorError> {
        let quantized = provider.is_quantized();
        DynamicAccessor::new(provider, context, vector_and_attrs.0, quantized)
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
    use std::{
        collections::HashMap,
        ffi::c_void,
        hash::BuildHasher,
        mem,
        ops::Range,
        sync::{Arc, Mutex, atomic::Ordering, mpsc},
        thread,
        time::{Duration, Instant},
    };

    use dashmap::DashMap;
    use diskann::{
        graph::{
            config::{self, defaults::GRAPH_SLACK_FACTOR},
            search,
        },
        provider::{DataProvider, Delete, Guard, SetElement},
    };
    use diskann_providers::index::wrapped_async::DiskANNIndex;
    use diskann_utils::views::Matrix;
    use diskann_vector::distance::Metric;
    use rand::Rng;

    use crate::{
        SearchResults, VectorQuantType,
        dyn_index::DynIndex,
        garnet::{
            Callbacks, Context, GarnetId, ReadDataCallback, RmwDataCallback, TERM_BITMASK, Term,
            WriteCallback,
        },
        provider::{GarnetProvider, GarnetProviderError, QUANT_STATE_KEY, RESERVATION_RETRY_LIMIT},
        quantization::{GarnetQuantizer, Spherical1Bit},
        test_utils::{LOGS, Store},
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

        let res = provider.set_element(&ctx, &id, (&[0f32, 0f32], &[])).await;
        res.unwrap().complete().await;

        let res = provider.delete(&ctx, &id).await;
        assert!(res.is_ok());

        let guard = provider
            .set_element(&ctx, &id, (&[0f32, 0f32], &[]))
            .await
            .unwrap();
        store.clear_read_counts();
        drop(guard);
        assert_eq!(store.int_map_reads(), 0);
        assert!(store.get(ctx.term(Term::IntMap).get(), &id).is_none());
        assert_eq!(provider.fsm.total_used(), 0);
        LOGS.with(|logs| assert!(logs.lock().unwrap().is_empty()));
    }

    fn assert_waits_for_range(
        provider: &GarnetProvider<f32>,
        range: Range<u32>,
        should_wait: bool,
        operation: impl FnOnce() + Send,
    ) {
        let (started_tx, started_rx) = mpsc::channel();
        let (finished_tx, finished_rx) = mpsc::channel();
        thread::scope(|scope| {
            let reservation = provider.reserve_backfill_range(range);
            scope.spawn(move || {
                started_tx.send(()).unwrap();
                operation();
                finished_tx.send(()).unwrap();
            });

            started_rx.recv_timeout(Duration::from_secs(5)).unwrap();
            let early_result = finished_rx.recv_timeout(if should_wait {
                Duration::from_millis(50)
            } else {
                Duration::from_secs(5)
            });
            drop(reservation);
            if should_wait {
                assert_eq!(early_result, Err(mpsc::RecvTimeoutError::Timeout));
                finished_rx.recv_timeout(Duration::from_secs(5)).unwrap();
            } else {
                early_result.unwrap();
            }
        });
        assert!(provider.backfill_lock.lock().unwrap().is_empty());
    }

    fn train_for_backfill(provider: &GarnetProvider<f32>, ctx: &Context) {
        let quantizer = provider.quantizer.as_ref().unwrap();
        let mut data = Matrix::new(0.0f32, quantizer.required_vectors(), 2);
        for row in 0..data.nrows() {
            data.row_mut(row)
                .copy_from_slice(&[(row + 1) as f32, (row % 7 + 1) as f32]);
        }
        quantizer.train(Metric::L2, data.as_view()).unwrap();
        let mut state = vec![0u8];
        state.extend_from_slice(&quantizer.serialize().unwrap());
        assert!(
            provider
                .callbacks
                .write_iid(&ctx.term(Term::Metadata), QUANT_STATE_KEY, &state)
        );
        provider.fsm.enable_quantization();
    }

    #[test]
    fn backfill_ranges_exclude_overlaps() {
        let store = Arc::new(DashMap::new());
        let state = ParallelContext::new(store);
        let ctx = state.context();
        let index = create_2d_f32_index_with_callbacks(
            VectorQuantType::NoQuant,
            Metric::L2,
            ParallelContext::callbacks(),
            &ctx,
        );
        let provider = index.inner.provider();

        for (active, requested, should_wait) in [
            (0..10, 5..6, true),
            (5..6, 0..10, true),
            (0..10, 0..10, true),
            (0..10, 9..20, true),
            (0..10, 10..20, false),
            (0..10, 5..5, false),
        ] {
            assert_waits_for_range(provider, active, should_wait, || {
                let _reservation = provider.reserve_backfill_range(requested);
            });
        }
    }

    #[test]
    fn updates_reserve_backfill_ranges_only_when_needed() {
        for (quant_type, train, finish, above_boundary, should_wait) in [
            (VectorQuantType::NoQuant, false, false, false, false),
            (VectorQuantType::Q8, false, false, false, false),
            (VectorQuantType::Bin, false, false, false, true),
            (VectorQuantType::Bin, true, false, false, true),
            (VectorQuantType::Bin, true, false, true, false),
            (VectorQuantType::Bin, true, true, false, false),
        ] {
            let store = Arc::new(DashMap::new());
            let state = ParallelContext::new(store.clone());
            let ctx = state.context();
            let index = create_2d_f32_index_with_callbacks(
                quant_type,
                Metric::L2,
                ParallelContext::callbacks(),
                &ctx,
            );
            let provider = index.inner.provider();
            let original = [0.0f32, 1.0];
            let mut id = GarnetId::from(bytemuck::bytes_of(&42u32));
            let runtime = tokio::runtime::Builder::new_current_thread()
                .build()
                .unwrap();
            provider.maybe_set_start_point(&ctx, &original).unwrap();
            let guard = runtime
                .block_on(provider.set_element(&ctx, &id, (&original, &[])))
                .unwrap();
            runtime.block_on(guard.complete());
            if train {
                train_for_backfill(provider, &ctx);
            }
            if above_boundary {
                id = GarnetId::from(bytemuck::bytes_of(&43u32));
                let guard = runtime
                    .block_on(provider.set_element(&ctx, &id, (&original, &[])))
                    .unwrap();
                runtime.block_on(guard.complete());
            }
            if finish {
                assert!(provider.backfill_quant_vectors(&ctx, 0, 1));
                assert!(provider.all_quantized.load(Ordering::Acquire));
            }
            let internal_id = parallel_get(&store, ctx.term(Term::IntMap).get(), &id).unwrap();
            let internal_id = bytemuck::pod_read_unaligned::<u32>(&internal_id);
            let update_ctx = state.context();
            assert_waits_for_range(provider, internal_id..internal_id + 1, should_wait, || {
                let updated = [1.0f32, 0.0];
                let guard = runtime
                    .block_on(provider.set_element(&update_ctx, &id, (&updated, &[])))
                    .unwrap();
                runtime.block_on(guard.complete());
                assert!(update_ctx.insert_is_update());
                if let Some(quantizer) = &provider.quantizer
                    && quantizer.is_trained()
                {
                    let mut expected = vec![0u8; quantizer.bytes()];
                    quantizer.compress(&updated, &mut expected).unwrap();
                    assert_eq!(
                        parallel_get(
                            &store,
                            ctx.term(Term::Quantized).get(),
                            bytemuck::bytes_of(&internal_id),
                        ),
                        Some(expected)
                    );
                }
            });
        }
    }

    #[test]
    fn backfill_waits_for_overlapping_updates() {
        for (range, should_wait) in [(1..2, true), (2..3, false)] {
            let store = Arc::new(DashMap::new());
            let state = ParallelContext::new(store);
            let ctx = state.context();
            let index = create_2d_f32_index_with_callbacks(
                VectorQuantType::Bin,
                Metric::L2,
                ParallelContext::callbacks(),
                &ctx,
            );
            let provider = index.inner.provider();
            let original = [0.0f32, 1.0];
            let id = GarnetId::from(bytemuck::bytes_of(&42u32));
            provider.maybe_set_start_point(&ctx, &original).unwrap();
            DynIndex::insert(&index, &ctx, &id, bytemuck::cast_slice(&original), &[]).unwrap();
            train_for_backfill(provider, &ctx);
            assert_waits_for_range(provider, range, should_wait, || {
                assert!(provider.backfill_quant_vectors(&ctx, 0, 1));
            });
            assert!(provider.all_quantized.load(Ordering::Acquire));
        }
    }

    /// Per-insert fault injection and synchronization for tests.
    #[derive(Default)]
    struct InsertControl {
        /// One-shot failure: target term and matching operations to skip.
        failure: Option<(u64, usize)>,
        /// One-shot pause: target term, arrival sender, and resume receiver.
        pause: Option<(u64, mpsc::Sender<()>, mpsc::Receiver<()>)>,
        /// Store snapshot before the first vector write.
        before_vector: Option<HashMap<Vec<u8>, Vec<u8>>>,
    }

    /// Parallel test callback state; the encoded pointer's low three bits hold the term tag.
    struct ParallelContext {
        /// Mock storage shared across threads.
        store: Arc<DashMap<Vec<u8>, Vec<u8>>>,
        /// Per-insert fault, pause, and snapshot state.
        control: Mutex<InsertControl>,
    }

    impl ParallelContext {
        fn new(store: Arc<DashMap<Vec<u8>, Vec<u8>>>) -> Box<Self> {
            Box::new(Self {
                store,
                control: Mutex::new(InsertControl::default()),
            })
        }

        fn context(&self) -> Context {
            Context::new(std::ptr::from_ref(self).expose_provenance() as u64)
        }

        unsafe fn from_context<'a>(context: u64) -> &'a Self {
            let pointer =
                std::ptr::with_exposed_provenance::<Self>((context & !TERM_BITMASK) as usize);
            // SAFETY: Callers keep the boxed context alive until all callback operations finish.
            unsafe { &*pointer }
        }

        fn callbacks() -> Callbacks {
            Callbacks::new(
                parallel_read,
                controlled_insert_write,
                parallel_delete,
                controlled_insert_rmw,
                parallel_filter,
                parallel_log,
            )
        }

        fn fail_insert_operation(&self, context: u64) -> bool {
            let mut control = self.control.lock().unwrap();
            let term = context & TERM_BITMASK;
            if control
                .pause
                .as_ref()
                .is_some_and(|(pause_term, _, _)| *pause_term == term)
            {
                let (_, entered, resume) = control.pause.take().unwrap();
                let _ = entered.send(());
                if resume.recv_timeout(Duration::from_secs(10)).is_err() {
                    return true;
                }
            }
            if term == Term::Vector as u64 && control.before_vector.is_none() {
                control.before_vector = Some(parallel_snapshot(&self.store));
            }
            if let Some((failure_term, skip)) = control.failure.as_mut()
                && *failure_term == term
            {
                if *skip == 0 {
                    control.failure = None;
                    return true;
                }
                *skip -= 1;
            }
            false
        }
    }

    fn parallel_key(context: u64, key: &[u8]) -> Vec<u8> {
        let mut encoded = bytemuck::bytes_of(&(context & TERM_BITMASK)).to_vec();
        encoded.extend_from_slice(key);
        encoded
    }

    fn parallel_get(
        store: &DashMap<Vec<u8>, Vec<u8>>,
        context: u64,
        key: &[u8],
    ) -> Option<Vec<u8>> {
        store
            .get(&parallel_key(context, key))
            .map(|value| value.clone())
    }

    fn parallel_snapshot(store: &DashMap<Vec<u8>, Vec<u8>>) -> HashMap<Vec<u8>, Vec<u8>> {
        store
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect()
    }

    unsafe extern "C" fn parallel_read(
        context: u64,
        count: u32,
        _length_hint: u32,
        keys: *const u8,
        keys_len: usize,
        callback: ReadDataCallback,
        callback_context: *mut c_void,
    ) {
        let state = unsafe { ParallelContext::from_context(context) };
        let mut keys = unsafe { std::slice::from_raw_parts(keys, keys_len) };
        for index in 0..count {
            let length = bytemuck::pod_read_unaligned::<u32>(&keys[..4]) as usize;
            let key = &keys[4..4 + length];
            if let Some(value) = parallel_get(&state.store, context, key) {
                unsafe { callback(index, callback_context, value.as_ptr(), value.len()) };
            }
            keys = &keys[4 + length..];
        }
    }

    unsafe extern "C" fn parallel_delete(context: u64, key: *const u8, key_len: usize) -> bool {
        let state = unsafe { ParallelContext::from_context(context) };
        let key = unsafe { std::slice::from_raw_parts(key, key_len) };
        state.store.remove(&parallel_key(context, key)).is_some()
    }

    unsafe extern "C" fn parallel_filter(_context: u64, _data: *const u8, _length: usize) -> bool {
        true
    }

    unsafe extern "C" fn parallel_log(_context: u64, _message: *const u8, _length: usize) {}

    unsafe extern "C" fn controlled_insert_write(
        context: u64,
        key: *const u8,
        key_len: usize,
        value: *const u8,
        value_len: usize,
    ) -> bool {
        let state = unsafe { ParallelContext::from_context(context) };
        if state.fail_insert_operation(context) {
            return false;
        }
        let key = unsafe { std::slice::from_raw_parts(key, key_len) };
        let value = unsafe { std::slice::from_raw_parts(value, value_len) };
        state
            .store
            .insert(parallel_key(context, key), value.to_vec());
        true
    }

    unsafe extern "C" fn controlled_insert_rmw(
        context: u64,
        key: *const u8,
        key_len: usize,
        value_len: usize,
        callback: RmwDataCallback,
        callback_context: *mut c_void,
    ) -> bool {
        let state = unsafe { ParallelContext::from_context(context) };
        if state.fail_insert_operation(context) {
            return false;
        }
        let key = unsafe { std::slice::from_raw_parts(key, key_len) };
        let mut value = state
            .store
            .entry(parallel_key(context, key))
            .or_insert_with(|| vec![0; value_len]);
        unsafe { callback(callback_context, value.as_mut_ptr(), value.len()) };
        true
    }

    fn wait_for_pending_receiver(provider: &GarnetProvider<f32>, id: &GarnetId) -> bool {
        let id_hash = provider.pending_external_ids.hasher().hash_one(&id[..]);
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            if provider
                .pending_external_ids
                .get(&id_hash)
                .is_some_and(|sender| sender.receiver_count() > 0)
            {
                return true;
            }
            if Instant::now() >= deadline {
                return false;
            }
            thread::yield_now();
        }
    }

    fn concurrent_insert_outcomes(first_fails: bool, second_fails: bool) {
        for quant_type in [
            VectorQuantType::NoQuant,
            VectorQuantType::Bin,
            VectorQuantType::Q8,
        ] {
            // failure_term tells where to fail, and skip tells how many to succeed before we fail
            for (failure_term, skip) in [
                (Term::Vector, 0),
                (Term::Quantized, 0),
                (Term::Attributes, 0),
                (Term::ExtMap, 0),
                (Term::IntMap, 0),
                (Term::Neighbors, 0),
                (Term::Neighbors, 1),
            ] {
                if matches!(failure_term, Term::Quantized) && quant_type != VectorQuantType::Q8 {
                    continue;
                }
                // Set up the index and capture its state before either insert.
                let store = Arc::new(DashMap::new());
                let state = ParallelContext::new(store.clone());
                let ctx = state.context();
                let callbacks = ParallelContext::callbacks();
                let index =
                    create_2d_f32_index_with_callbacks(quant_type, Metric::L2, callbacks, &ctx);
                let provider = index.inner.provider();
                let id = GarnetId::from(bytemuck::bytes_of(&42u32));
                let first_vector = [0.0f32, 1.0];
                let second_vector = [1.0f32, 0.0];
                provider.maybe_set_start_point(&ctx, &first_vector).unwrap();
                let initial_used = provider.fsm.total_used();
                let mut initial_snapshot = parallel_snapshot(&store);
                initial_snapshot.retain(|key, _| {
                    bytemuck::pod_read_unaligned::<u64>(&key[..8]) != Term::Metadata as u64
                });
                let (entered_tx, entered_rx) = mpsc::channel();
                let (resume_tx, resume_rx) = mpsc::channel();
                let failure = (failure_term as u64, skip);

                // Pause the first insert while it owns the external ID, verify the second
                // waits for that ID, then resume the first and collect both results.
                let (first, second) = thread::scope(|scope| {
                    let first = scope.spawn(|| {
                        let state = ParallelContext::new(store.clone());
                        *state.control.lock().unwrap() = InsertControl {
                            failure: first_fails.then_some(failure),
                            pause: Some((failure.0, entered_tx, resume_rx)),
                            before_vector: None,
                        };
                        let context = state.context();
                        let result = DynIndex::insert(
                            &index,
                            &context,
                            &id,
                            bytemuck::cast_slice(&first_vector),
                            b"first",
                        );
                        (result.is_ok(), context.insert_is_update())
                    });
                    let entered = entered_rx.recv_timeout(Duration::from_secs(5));
                    let second = scope.spawn(|| {
                        let state = ParallelContext::new(store.clone());
                        let second_failure = if !first_fails
                            && (failure.0 == Term::ExtMap as u64
                                || failure.0 == Term::IntMap as u64)
                        {
                            (Term::Attributes as u64, 0)
                        } else {
                            (failure.0, if first_fails { failure.1 } else { 0 })
                        };
                        *state.control.lock().unwrap() = InsertControl {
                            failure: second_fails.then_some(second_failure),
                            pause: None,
                            before_vector: None,
                        };
                        let context = state.context();
                        let result = DynIndex::insert(
                            &index,
                            &context,
                            &id,
                            bytemuck::cast_slice(&second_vector),
                            b"second",
                        );
                        let snapshot = state.control.lock().unwrap().before_vector.take().unwrap();
                        ((result.is_ok(), context.insert_is_update()), snapshot)
                    });
                    let waiting = wait_for_pending_receiver(provider, &id);
                    let _ = resume_tx.send(());
                    let results = (first.join().unwrap(), second.join().unwrap());
                    assert!(entered.is_ok(), "first insert never acquired ownership");
                    assert!(waiting, "second insert did not wait for ownership");
                    results
                });

                // Verify success/update classification and rollback to the expected state.
                assert_eq!(
                    first,
                    (!first_fails, false),
                    "first: {quant_type:?}, fault {failure:?}"
                );
                assert_eq!(
                    second.0,
                    (!second_fails, !first_fails),
                    "second: {quant_type:?}, fault {failure:?}"
                );
                if second_fails {
                    let mut current_snapshot = parallel_snapshot(&store);
                    if first_fails {
                        current_snapshot.retain(|key, _| {
                            bytemuck::pod_read_unaligned::<u64>(&key[..8]) != Term::Metadata as u64
                        });
                        assert_eq!(current_snapshot, initial_snapshot);
                    } else {
                        assert_eq!(current_snapshot, second.1);
                    }
                }
                // Check ownership cleanup, live member records, and freed IDs.
                assert!(provider.pending_external_ids.is_empty());
                let has_member = !first_fails || !second_fails;
                assert_eq!(
                    provider.fsm.total_used(),
                    initial_used + usize::from(has_member)
                );
                let mapping = parallel_get(&store, ctx.term(Term::IntMap).get(), &id);
                assert_eq!(mapping.is_some(), has_member);
                let current_id = mapping.as_deref().map(bytemuck::pod_read_unaligned::<u32>);
                for internal_id in 1..=provider.fsm.max_id() {
                    let key = bytemuck::bytes_of(&internal_id);
                    if Some(internal_id) == current_id {
                        let (vector, attrs) = if second_fails {
                            (&first_vector, &b"first"[..])
                        } else {
                            (&second_vector, &b"second"[..])
                        };
                        assert_eq!(provider.get_full_vector(&ctx, internal_id).unwrap(), vector);
                        assert_eq!(
                            parallel_get(&store, ctx.term(Term::Attributes).get(), key),
                            Some(attrs.to_vec())
                        );
                        assert_eq!(
                            parallel_get(&store, ctx.term(Term::ExtMap).get(), key),
                            Some(id.to_vec())
                        );
                        if let Some(quantizer) = &provider.quantizer
                            && quantizer.is_trained()
                        {
                            let mut expected = vec![0; quantizer.bytes()];
                            quantizer.compress(vector, &mut expected).unwrap();
                            assert_eq!(
                                parallel_get(&store, ctx.term(Term::Quantized).get(), key),
                                Some(expected)
                            );
                        }
                    } else {
                        assert!(provider.fsm.is_free(&ctx, internal_id).unwrap());
                        for term in [
                            Term::Vector,
                            Term::Quantized,
                            Term::Attributes,
                            Term::Neighbors,
                            Term::ExtMap,
                        ] {
                            assert!(parallel_get(&store, ctx.term(term).get(), key).is_none());
                        }
                    }
                }

                // Insert again, which will update or insert depending on the first two.
                let retry_context = state.context();
                DynIndex::insert(
                    &index,
                    &retry_context,
                    &id,
                    bytemuck::cast_slice(&first_vector),
                    b"retry",
                )
                .unwrap();
                assert_eq!(retry_context.insert_is_update(), has_member);
                assert_eq!(provider.fsm.total_used(), initial_used + 1);
                assert!(provider.pending_external_ids.is_empty());
            }
        }
    }

    #[test]
    fn concurrent_inserts_both_succeed() {
        concurrent_insert_outcomes(false, false);
    }

    #[test]
    fn concurrent_inserts_first_fails() {
        concurrent_insert_outcomes(true, false);
    }

    #[test]
    fn concurrent_inserts_second_fails() {
        concurrent_insert_outcomes(false, true);
    }

    #[test]
    fn concurrent_inserts_both_fail() {
        concurrent_insert_outcomes(true, true);
    }

    #[tokio::test]
    async fn pending_external_ids_serialize_and_wake_waiters() {
        use std::{future::Future, pin::pin, task::Poll};

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
        let id = GarnetId::from(&b"same"[..]);
        let other_id = GarnetId::from(&b"other"[..]);
        let id_hash = provider.pending_external_ids.hasher().hash_one(&id[..]);
        let receiver_count = || {
            provider
                .pending_external_ids
                .get(&id_hash)
                .unwrap()
                .receiver_count()
        };
        // Reserve one ID and register two waiters; a different ID remains available.
        let owner = provider.reserve_external_id(&id).await.unwrap();
        assert_eq!(receiver_count(), 0);
        let mut second = pin!(provider.reserve_external_id(&id));
        let mut third = pin!(provider.reserve_external_id(&id));
        let mut task = std::task::Context::from_waker(std::task::Waker::noop());
        assert!(second.as_mut().poll(&mut task).is_pending());
        assert!(third.as_mut().poll(&mut task).is_pending());
        assert_eq!(receiver_count(), 2);
        drop(provider.reserve_external_id(&other_id).await.unwrap());

        // Release owners in turn; only one waiter can hold the ID at a time.
        drop(owner);
        let Poll::Ready(Ok(second_owner)) = second.as_mut().poll(&mut task) else {
            panic!("second waiter did not acquire the released ID");
        };
        assert!(third.as_mut().poll(&mut task).is_pending());
        assert_eq!(receiver_count(), 1);
        drop(second_owner);
        drop(third.await.unwrap());
        assert!(provider.pending_external_ids.is_empty());

        // Cancelling a waiter removes its subscription without releasing the owner's ID.
        let owner = provider.reserve_external_id(&id).await.unwrap();
        let mut cancelled = Box::pin(provider.reserve_external_id(&id));
        assert!(cancelled.as_mut().poll(&mut task).is_pending());
        assert_eq!(receiver_count(), 1);
        drop(cancelled);
        assert_eq!(receiver_count(), 0);
        drop(owner);
        drop(provider.reserve_external_id(&id).await.unwrap());
        assert!(provider.pending_external_ids.is_empty());
    }

    #[tokio::test]
    async fn external_id_reservation_retry_limit() {
        use std::{future::Future, pin::pin, task::Poll};

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
        let id = GarnetId::from(&b"contended"[..]);
        let id_hash = provider.pending_external_ids.hasher().hash_one(&id[..]);
        let mut task = std::task::Context::from_waker(std::task::Waker::noop());

        // Exercise both acquisition and continued contention on the final allowed retry.
        for acquire_on_last_retry in [false, true] {
            let mut owner = provider.reserve_external_id(&id).await.unwrap();
            let mut waiter = pin!(provider.reserve_external_id(&id));
            assert!(waiter.as_mut().poll(&mut task).is_pending());

            // Reacquire the ID before polling the waiter to force repeated contention.
            for _ in 1..RESERVATION_RETRY_LIMIT {
                drop(owner);
                owner = provider.reserve_external_id(&id).await.unwrap();
                assert!(waiter.as_mut().poll(&mut task).is_pending());
            }

            drop(owner);
            if acquire_on_last_retry {
                let Poll::Ready(Ok(guard)) = waiter.as_mut().poll(&mut task) else {
                    panic!("last reservation retry did not acquire the released ID");
                };
                drop(guard);
            } else {
                // Exhaustion removes the waiter but leaves the current owner's reservation.
                let owner = provider.reserve_external_id(&id).await.unwrap();
                assert!(matches!(
                    waiter.as_mut().poll(&mut task),
                    Poll::Ready(Err(GarnetProviderError::ReservationRetryLimit))
                ));
                assert_eq!(
                    provider
                        .pending_external_ids
                        .get(&id_hash)
                        .unwrap()
                        .receiver_count(),
                    0
                );
                drop(owner);
            }
            assert!(provider.pending_external_ids.is_empty());
        }
    }

    #[test]
    fn member_mutations_wait_for_insert_owner() {
        enum Mutation {
            SetAttributes,
            DeleteAttributes,
            Remove,
        }

        for mutation in [
            Mutation::SetAttributes,
            Mutation::DeleteAttributes,
            Mutation::Remove,
        ] {
            // Create an existing member with attributes for each mutation.
            let store = Arc::new(DashMap::new());
            let state = ParallelContext::new(store.clone());
            let ctx = state.context();
            let index = create_2d_f32_index_with_callbacks(
                VectorQuantType::NoQuant,
                Metric::L2,
                ParallelContext::callbacks(),
                &ctx,
            );
            let provider = index.inner.provider();
            let id = GarnetId::from(bytemuck::bytes_of(&42u32));
            let vector = [0.0f32, 1.0];
            provider.maybe_set_start_point(&ctx, &vector).unwrap();
            DynIndex::insert(&index, &ctx, &id, bytemuck::cast_slice(&vector), b"before").unwrap();
            let internal_id = provider.to_internal_id(&ctx, &id).unwrap();

            // Hold the ID reservation and verify the mutation waits until it is released.
            let owner = index.run(|_| provider.reserve_external_id(&id)).unwrap();
            thread::scope(|scope| {
                let task = scope.spawn(|| match mutation {
                    Mutation::SetAttributes => {
                        DynIndex::set_attributes(&index, &ctx, &id, b"after")
                    }
                    Mutation::DeleteAttributes => DynIndex::delete_attributes(&index, &ctx, &id),
                    Mutation::Remove => DynIndex::remove(&index, &ctx, &id),
                });
                let waiting = wait_for_pending_receiver(provider, &id);
                drop(owner);
                task.join().unwrap().unwrap();
                assert!(waiting, "member mutation did not wait for insert ownership");
            });

            // Check the mutation's effect on attributes and membership, then reservation cleanup.
            let attrs = parallel_get(
                &store,
                ctx.term(Term::Attributes).get(),
                bytemuck::bytes_of(&internal_id),
            );
            assert_eq!(
                attrs,
                match mutation {
                    Mutation::SetAttributes => Some(b"after".to_vec()),
                    _ => None,
                }
            );
            assert_eq!(
                provider.to_internal_id(&ctx, &id).is_ok(),
                !matches!(mutation, Mutation::Remove)
            );
            assert!(provider.pending_external_ids.is_empty());
        }
    }

    #[tokio::test]
    async fn update_reuses_id_and_preserves_entry_on_write_failure() {
        unsafe extern "C" fn fail_write(
            _context: u64,
            _key: *const u8,
            _key_len: usize,
            _value: *const u8,
            _value_len: usize,
        ) -> bool {
            false
        }

        unsafe extern "C" fn fail_after_vector_write(
            context: u64,
            key: *const u8,
            key_len: usize,
            value: *const u8,
            value_len: usize,
        ) -> bool {
            context & TERM_BITMASK == Term::Vector as u64
                && unsafe {
                    (Store::attach().callbacks().write_callback())(
                        context, key, key_len, value, value_len,
                    )
                }
        }

        for quant_type in [
            VectorQuantType::NoQuant,
            VectorQuantType::Bin,
            VectorQuantType::Q8,
        ] {
            let store = Store::new();
            let ctx = Context::new(0);
            let mut provider =
                GarnetProvider::<f32>::new(2, quant_type, Metric::L2, 10, store.callbacks(), &ctx)
                    .unwrap();
            let id = GarnetId::from(bytemuck::bytes_of(&42u32));
            let original = [0.0f32, 1.0];
            provider.maybe_set_start_point(&ctx, &original).unwrap();
            provider
                .set_element(&ctx, &id, (&original, b"old"))
                .await
                .unwrap()
                .complete()
                .await;
            let internal_id = store.get(ctx.term(Term::IntMap).get(), &id).unwrap();
            let max_id = provider.fsm.max_id();
            let total_used = provider.fsm.total_used();

            let updated = [1.0f32, 0.0];
            provider
                .set_element(&ctx, &id, (&updated, b"new"))
                .await
                .unwrap()
                .complete()
                .await;
            assert!(ctx.insert_is_update());
            assert_eq!(provider.fsm.max_id(), max_id);
            assert_eq!(provider.fsm.total_used(), total_used);
            assert_eq!(
                store.get(ctx.term(Term::IntMap).get(), &id),
                Some(internal_id.clone())
            );
            assert_eq!(
                store.get(ctx.term(Term::ExtMap).get(), &internal_id),
                Some(id.to_vec())
            );
            assert_eq!(
                store.get(ctx.term(Term::Vector).get(), &internal_id),
                Some(bytemuck::cast_slice::<f32, u8>(&updated).to_vec())
            );
            assert_eq!(
                store.get(ctx.term(Term::Attributes).get(), &internal_id),
                Some(b"new".to_vec())
            );
            if let Some(quantizer) = &provider.quantizer
                && quantizer.is_trained()
            {
                let mut expected = vec![0u8; quantizer.bytes()];
                quantizer.compress(&updated, &mut expected).unwrap();
                assert_eq!(
                    store.get(ctx.term(Term::Quantized).get(), &internal_id),
                    Some(expected)
                );
            }

            let quantized_before = store.get(ctx.term(Term::Quantized).get(), &internal_id);
            let guard = provider
                .set_element(&ctx, &id, (&original, b"discarded"))
                .await
                .unwrap();
            store.clear_read_counts();
            drop(guard);
            assert_eq!(store.full_reads(), 0);
            if quantized_before.is_some() {
                assert_eq!(store.quant_reads(), 0);
            }
            assert_eq!(
                store.get(ctx.term(Term::Vector).get(), &internal_id),
                Some(bytemuck::cast_slice::<f32, u8>(&updated).to_vec())
            );
            assert_eq!(
                store.get(ctx.term(Term::Attributes).get(), &internal_id),
                Some(b"new".to_vec())
            );
            assert_eq!(
                store.get(ctx.term(Term::Quantized).get(), &internal_id),
                quantized_before
            );

            for (write_callback, expected_vector) in [
                (fail_write as WriteCallback, &updated),
                (fail_after_vector_write as WriteCallback, &updated),
            ] {
                LOGS.with(|logs| logs.lock().unwrap().clear());
                let callbacks = store.callbacks();
                provider.callbacks = Callbacks::new(
                    callbacks.read_callback(),
                    write_callback,
                    callbacks.delete_callback(),
                    callbacks.rmw_callback(),
                    callbacks.filter_callback(),
                    callbacks.log_callback(),
                );
                let error = provider
                    .set_element(&ctx, &id, (&original, b"failed"))
                    .await
                    .unwrap_err();
                assert!(provider.backfill_lock.lock().unwrap().is_empty());
                assert_eq!(provider.fsm.max_id(), max_id);
                assert_eq!(provider.fsm.total_used(), total_used);
                assert_eq!(
                    store.get(ctx.term(Term::IntMap).get(), &id),
                    Some(internal_id.clone())
                );
                assert_eq!(
                    store.get(ctx.term(Term::Vector).get(), &internal_id),
                    Some(bytemuck::cast_slice::<f32, u8>(expected_vector).to_vec())
                );
                assert_eq!(
                    store.get(ctx.term(Term::Quantized).get(), &internal_id),
                    quantized_before
                );
                LOGS.with(|logs| {
                    let logs = logs.lock().unwrap();
                    assert_eq!(logs.len(), 2);
                    let (context, message) = &logs[0];
                    assert_eq!(*context, ctx.term(Term::Vector).get());
                    assert!(message.contains("update failed",));
                    assert!(message.contains(&error.to_string()));
                    assert!(logs[1].1.contains("insert rollback failed"));
                });
            }
        }
    }

    fn create_2d_f32_index(
        quant_type: VectorQuantType,
        metric: Metric,
        store: &Store,
        ctx: &Context,
    ) -> DiskANNIndex<GarnetProvider<f32>> {
        create_2d_f32_index_with_callbacks(quant_type, metric, store.callbacks(), ctx)
    }

    fn create_2d_f32_index_with_callbacks(
        quant_type: VectorQuantType,
        metric: Metric,
        callbacks: Callbacks,
        ctx: &Context,
    ) -> DiskANNIndex<GarnetProvider<f32>> {
        let provider =
            GarnetProvider::<f32>::new(2, quant_type, metric, 10, callbacks, ctx).unwrap();

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
                &[],
            )
            .unwrap();
            last_inserted_id = id;
        }

        assert!(!provider.is_quantized());
        let max_id = provider.max_internal_id();
        assert_eq!(max_id, last_inserted_id + 1);

        // There should be no saved quant state.
        assert!(
            !provider.callbacks.exists_iid(
                &ctx.term(Term::Metadata),
                QUANT_STATE_KEY,
                provider.quant_state_size()
            ),
            "quant state should not be stored yet"
        );

        // Quantization is not needed yet
        assert!(!provider.quantization_needed());

        let params = search::Knn::new(10, None).unwrap();
        let mut output_ids = vec![0u8; mem::size_of::<u32>() * 2 * 10];
        let mut output_dists = vec![0f32; 10];
        let mut output = SearchResults::new(
            10,
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
                &[],
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
            provider.callbacks.exists_iid(
                &ctx.term(Term::Metadata),
                QUANT_STATE_KEY,
                provider.quant_state_size()
            ),
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

        let inserted_id = last_inserted_id + 1;
        DynIndex::insert(
            &index,
            &ctx,
            &GarnetId::from(bytemuck::bytes_of(&inserted_id)),
            bytemuck::cast_slice(&tv),
            &[],
        )
        .unwrap();
        assert!(provider.callbacks.exists_iid(
            &ctx.term(Term::Quantized),
            max_id + 1,
            provider.quant_vector_size()
        ));
        assert_eq!(provider.fsm.max_id_for_backfill(), max_id);

        for job_id in 0..4 {
            assert!(provider.backfill_quant_vectors(&ctx, job_id, 4));
        }
        assert!(provider.is_quantized());

        for id in 0..=max_id + 1 {
            assert!(provider.callbacks.exists_iid(
                &ctx.term(Term::Quantized),
                id,
                provider.quant_vector_size()
            ));
        }
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
                &[],
            )
            .unwrap();
            last_inserted_id = id;
        }

        // Train the quantizer
        assert!(provider.train_quantizer(&ctx));

        // Run backfill
        for job_id in 0..4 {
            assert!(provider.backfill_quant_vectors(&ctx, job_id, 4));
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
            provider.callbacks.exists_iid(
                &ctx.term(Term::Metadata),
                QUANT_STATE_KEY,
                provider.quant_state_size()
            ),
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
            assert!(provider.callbacks.exists_iid(
                &ctx.term(Term::Quantized),
                id,
                provider.quant_vector_size()
            ));
        }

        // Searches should still work and use quantized vectors
        let params = search::Knn::new(10, None).unwrap();
        let mut output_ids = vec![0u8; mem::size_of::<u32>() * 2 * 10];
        let mut output_dists = vec![0f32; 10];
        let mut output = SearchResults::new(
            10,
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

    /// Test that restarts during phase three quant bootstrap work.
    /// Phase three starts once backfill is complete and lasts for the remaining
    /// life of the index.
    #[test]
    fn restart_q8() {
        let store = Store::new();
        let ctx = Context::new(0);
        let index = create_2d_f32_index(VectorQuantType::Q8, Metric::L2, &store, &ctx);
        let provider = index.inner.provider();

        let mut rng = rand::rng();

        let mut first_insert = true;
        for id in 0..10 {
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
                &[],
            )
            .unwrap();
        }

        // Drop and re-create the index, keeping the same backing store
        let index = create_2d_f32_index(VectorQuantType::Q8, Metric::L2, &store, &ctx);
        let provider = index.inner.provider();

        // There should be saved quant state.
        assert!(
            provider.callbacks.exists_iid(
                &ctx.term(Term::Metadata),
                QUANT_STATE_KEY,
                provider.quant_state_size()
            ),
            "quant state missing"
        );

        // Vectors should serialize identically

        let quantizer = provider.quantizer.as_ref().expect("missing quantizer");
        let mut fv = vec![0f32; 2];
        let mut orig_qv = vec![0u8; quantizer.bytes()];
        let mut qv = vec![0u8; quantizer.bytes()];

        let gid = GarnetId::from(bytemuck::bytes_of::<u32>(&0));
        let mut iid = 0u32;
        assert!(provider.callbacks.read_single_eid(
            &ctx.term(Term::IntMap),
            &gid,
            bytemuck::bytes_of_mut(&mut iid),
        ));

        assert!(provider.callbacks.read_single_iid(
            &ctx.term(Term::Vector),
            iid,
            bytemuck::cast_slice_mut::<f32, u8>(&mut fv),
        ));

        quantizer.compress(&fv, &mut qv).unwrap();

        assert!(
            provider
                .callbacks
                .read_single_iid(&ctx.term(Term::Quantized), iid, &mut orig_qv)
        );

        assert_eq!(orig_qv, qv, "quant vectors mismatched");
    }
}
