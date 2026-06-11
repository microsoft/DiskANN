/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    ffi::c_void,
    mem,
    ops::Deref,
    ptr, slice,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use diskann::{
    graph::{
        SearchOutputBuffer,
        config::{self, defaults::GRAPH_SLACK_FACTOR},
        search::{self, AdaptiveL},
    },
    utils::VectorRepr,
};
use diskann_providers::index::wrapped_async::DiskANNIndex;
use diskann_quantization::alloc::Poly;
use diskann_vector::distance::Metric;

use crate::{
    alloc::AlignToEight,
    garnet::FilterCallback,
    provider::{GarnetProvider, GarnetProviderError},
};
use crate::{
    dyn_index::DynIndex,
    garnet::{
        Callbacks, Context, DeleteCallback, GarnetId, ReadCallback, ReadModifyWriteCallback,
        WriteCallback,
    },
};

mod alloc;
mod dyn_index;
#[cfg(test)]
mod ffi_recall_tests;
#[cfg(test)]
mod ffi_tests;
mod fsm;
mod garnet;
mod provider;
mod quantization;
#[cfg(test)]
mod test_utils;

enum IndexState {
    NoStartPoints,
    SettingStartPoints,
    Ready,
}
impl From<usize> for IndexState {
    fn from(value: usize) -> Self {
        assert!(value < 3);
        match value {
            0 => IndexState::NoStartPoints,
            1 => IndexState::SettingStartPoints,
            2 => IndexState::Ready,
            _ => unreachable!(),
        }
    }
}

pub(crate) struct Index {
    inner: Box<dyn DynIndex>,
    quant_type: VectorQuantType,
    state: AtomicUsize,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(C)]
pub enum VectorValueType {
    Invalid = 0,
    FP32,
    XB8,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(C)]
pub enum VectorQuantType {
    Invalid = 0,
    NoQuant,
    Bin,
    Q8,
    XNoQuantU8,
    XNoQuantI8,
    XBinI8,
    XBinU8,
}

struct SearchResults<'a> {
    ids: &'a mut [u8],
    dists: &'a mut [f32],
    index: usize,
    id_index: usize,
}

impl SearchResults<'_> {
    fn new(ids: *mut u8, ids_len: usize, dists: *mut f32, dists_len: usize) -> Self {
        let ids = unsafe { slice::from_raw_parts_mut(ids, ids_len) };
        let dists = unsafe { slice::from_raw_parts_mut(dists, dists_len) };
        let index = 0;
        let id_index = 0;
        Self {
            ids,
            dists,
            index,
            id_index,
        }
    }
}

impl SearchOutputBuffer<GarnetId> for SearchResults<'_> {
    fn size_hint(&self) -> Option<usize> {
        Some(self.dists.len() - self.index)
    }

    fn push(&mut self, id: GarnetId, distance: f32) -> diskann::graph::BufferState {
        if self.index >= self.dists.len()
            || self.id_index + mem::size_of::<u32>() + id.len() > self.ids.len()
        {
            return diskann::graph::BufferState::Full;
        }

        let id_len = id.len() as u32;
        self.ids[self.id_index..self.id_index + mem::size_of::<u32>()]
            .copy_from_slice(bytemuck::bytes_of(&id_len));
        self.id_index += mem::size_of::<u32>();

        self.ids[self.id_index..self.id_index + id.len()].copy_from_slice(&id);
        self.dists[self.index] = distance;
        self.index += 1;
        self.id_index += id.len();

        if self.index >= self.dists.len() || self.id_index >= self.ids.len() {
            diskann::graph::BufferState::Full
        } else {
            diskann::graph::BufferState::Available
        }
    }

    fn current_len(&self) -> usize {
        self.index
    }

    fn extend<Itr>(&mut self, itr: Itr) -> usize
    where
        Itr: IntoIterator<Item = (GarnetId, f32)>,
    {
        let initial = self.current_len();

        for (id, dist) in itr {
            if self.push(id, dist).is_full() {
                break;
            }
        }

        self.current_len() - initial
    }
}

fn create_index_impl<T: VectorRepr>(
    quant_type: VectorQuantType,
    config: config::Config,
    dim: usize,
    metric_type: Metric,
    max_degree: usize,
    callbacks: Callbacks,
    context: Context,
) -> Result<Arc<Index>, GarnetProviderError> {
    let provider = GarnetProvider::<T>::new(
        dim,
        quant_type,
        metric_type,
        max_degree,
        callbacks,
        &context,
    )?;
    let state = if provider.start_points_exist() {
        AtomicUsize::new(IndexState::Ready as usize)
    } else {
        AtomicUsize::new(IndexState::NoStartPoints as usize)
    };
    Ok(Arc::new(Index {
        inner: Box::new(DiskANNIndex::new_with_current_thread_runtime(
            config, provider,
        )),
        quant_type,
        state,
    }))
}

/// # Safety
///
/// FFI
#[unsafe(no_mangle)]
pub unsafe extern "C" fn create_index(
    ctx: u64,
    dim: u32,
    _reduce_dim: u32,
    quant_type: VectorQuantType,
    metric_type: i32,
    l_build: u32,
    max_degree: u32,
    read_callback: ReadCallback,
    write_callback: WriteCallback,
    delete_callback: DeleteCallback,
    rmw_callback: ReadModifyWriteCallback,
    filter_callback: FilterCallback,
) -> *const c_void {
    let metric_type = match Metric::try_from(metric_type) {
        Ok(m) => m,
        Err(_) => return ptr::null(),
    };

    let target_degree = (max_degree as f32 / GRAPH_SLACK_FACTOR) as usize;

    let config = if let Ok(config) = config::Builder::new(
        target_degree,
        config::MaxDegree::Value(max_degree as usize),
        l_build as usize,
        metric_type.into(),
    )
    .build()
    {
        config
    } else {
        return ptr::null();
    };

    let context = Context::new(ctx);
    let callbacks = Callbacks::new(
        read_callback,
        write_callback,
        delete_callback,
        rmw_callback,
        filter_callback,
    );

    match quant_type {
        VectorQuantType::Invalid => ptr::null(),
        VectorQuantType::XNoQuantU8 | VectorQuantType::XBinU8 => {
            if let Ok(index) = create_index_impl::<u8>(
                quant_type,
                config,
                dim as usize,
                metric_type,
                max_degree as usize,
                callbacks,
                context,
            ) {
                Arc::into_raw(index).cast::<c_void>()
            } else {
                ptr::null()
            }
        }
        VectorQuantType::XNoQuantI8 | VectorQuantType::XBinI8 => {
            if let Ok(index) = create_index_impl::<i8>(
                quant_type,
                config,
                dim as usize,
                metric_type,
                max_degree as usize,
                callbacks,
                context,
            ) {
                Arc::into_raw(index).cast::<c_void>()
            } else {
                ptr::null()
            }
        }
        VectorQuantType::NoQuant | VectorQuantType::Bin | VectorQuantType::Q8 => {
            if let Ok(index) = create_index_impl::<f32>(
                quant_type,
                config,
                dim as usize,
                metric_type,
                max_degree as usize,
                callbacks,
                context,
            ) {
                Arc::into_raw(index).cast::<c_void>()
            } else {
                ptr::null()
            }
        }
    }
}

/// # Safety
///
/// FFI
#[unsafe(no_mangle)]
pub unsafe extern "C" fn drop_index(_ctx: u64, index_ptr: *const c_void) {
    // SAFETY: Caller must pass in a valid pointer returned from `create_index`.
    let _ = unsafe { Arc::from_raw(index_ptr.cast::<Index>()) };
}

enum PolyCow<'a> {
    Owned(Poly<[u8], AlignToEight>),
    Borrowed(&'a [u8]),
}

impl<'a> Deref for PolyCow<'a> {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        match self {
            PolyCow::Owned(p) => p.deref(),
            PolyCow::Borrowed(p) => p,
        }
    }
}

impl<'a, T: VectorRepr> From<&'a [T]> for PolyCow<'a> {
    fn from(value: &'a [T]) -> Self {
        PolyCow::Borrowed(bytemuck::cast_slice(value))
    }
}

impl<'a> From<Poly<[u8], AlignToEight>> for PolyCow<'a> {
    fn from(value: Poly<[u8], AlignToEight>) -> Self {
        PolyCow::Owned(value)
    }
}

fn interpret_vector<'a>(
    quant_type: VectorQuantType,
    vector_data: &'a *const u8,
    vector_len: usize,
) -> Option<PolyCow<'a>> {
    let vector_len_bytes = match quant_type {
        VectorQuantType::Invalid => return None,
        VectorQuantType::NoQuant | VectorQuantType::Bin | VectorQuantType::Q8 => vector_len * 4,
        VectorQuantType::XNoQuantU8
        | VectorQuantType::XNoQuantI8
        | VectorQuantType::XBinU8
        | VectorQuantType::XBinI8 => vector_len,
    };

    let v = unsafe { slice::from_raw_parts(*vector_data, vector_len_bytes) };

    let v = match quant_type {
        VectorQuantType::Invalid => return None,
        VectorQuantType::NoQuant | VectorQuantType::Bin | VectorQuantType::Q8 => {
            if v.as_ptr().align_offset(mem::align_of::<f32>()) == 0 {
                // pointer is correctly aligned to interpret as f32
                PolyCow::from(v)
            } else {
                // need to copy f32 data as it is unaligned
                let mut fp = if let Ok(fp) = Poly::broadcast(0u8, vector_len_bytes, AlignToEight) {
                    fp
                } else {
                    return None;
                };
                fp.copy_from_slice(v);
                PolyCow::from(fp)
            }
        }
        VectorQuantType::XNoQuantU8
        | VectorQuantType::XNoQuantI8
        | VectorQuantType::XBinU8
        | VectorQuantType::XBinI8 => PolyCow::from(v),
    };

    Some(v)
}

enum InsertResult {
    Fail,
    Success,
    SuccessStartTraining,
}

impl From<InsertResult> for u8 {
    fn from(value: InsertResult) -> Self {
        match value {
            InsertResult::Fail => 0,
            InsertResult::Success => 1,
            InsertResult::SuccessStartTraining => 2,
        }
    }
}

/// Insert a vector into the index.
///
/// Returns a status corresponding to the `InsertResult` enum. Aside from failure and success,
/// there is a third value that signals that the completed insert has reached the threshold to
/// begin quantization.
///
/// # Safety
///
/// FFI
#[unsafe(no_mangle)]
pub unsafe extern "C" fn insert(
    ctx: u64,
    index_ptr: *const c_void,
    id_data: *const u8,
    id_len: usize,
    vector_data: *const u8,
    vector_len: usize,
    attribute_data: *const u8,
    attribute_len: usize,
) -> u8 {
    let index = unsafe { &*index_ptr.cast::<Index>() };
    let ctx = Context::new(ctx);

    let id_bytes = unsafe { slice::from_raw_parts(id_data, id_len) };
    let id = GarnetId::from(id_bytes);

    let v = if let Some(v) = interpret_vector(index.quant_type, &vector_data, vector_len) {
        v
    } else {
        return InsertResult::Fail.into();
    };

    if let Some(_err) =
        ensure_index_ready_or_init(index, || index.inner.maybe_set_start_point(&ctx, &v).err())
    {
        return InsertResult::Fail.into();
    };

    // Write attributes to garnet
    let attr_data = if attribute_len > 0 && !attribute_data.is_null() {
        unsafe { slice::from_raw_parts(attribute_data, attribute_len) }
    } else {
        &[]
    };
    if index.inner.set_attributes(&ctx, &id, attr_data).is_err() {
        return InsertResult::Fail.into();
    }

    let old_ready = ctx.quantizer_ready();

    // Insert the vector
    if index.inner.insert(&ctx, &id, &v).is_ok() {
        let ready = ctx.quantizer_ready();
        if !old_ready && ready {
            InsertResult::SuccessStartTraining.into()
        } else {
            InsertResult::Success.into()
        }
    } else {
        InsertResult::Fail.into()
    }
}

fn ensure_index_ready_or_init<F, E>(index: &Index, init: F) -> Option<E>
where
    F: FnOnce() -> Option<E>,
{
    // Deal with start point initialization.
    loop {
        match index.state.load(Ordering::Acquire).into() {
            IndexState::Ready => break,                 // Index already ready to go.
            IndexState::SettingStartPoints => continue, // Another thread is setting the start point, wait.
            IndexState::NoStartPoints => {
                // No start points are set yet, so we'll do it.
                match index.state.compare_exchange(
                    IndexState::NoStartPoints as usize,
                    IndexState::SettingStartPoints as usize,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    Ok(_) => {
                        // Run the initializer to set start point.
                        if let Some(e) = init() {
                            // If init() fails, go back to the NoStartPoints state.
                            index
                                .state
                                .store(IndexState::NoStartPoints as usize, Ordering::Release);
                            return Some(e);
                        }
                        index
                            .state
                            .store(IndexState::Ready as usize, Ordering::Release);
                        break;
                    }
                    Err(_) => continue, // Someone else beat us, so wait and try again.
                }
            }
        }
    }
    None
}

/// Trigger building quantization tables. Garnet will call this once per `insert()` call that
/// returns `InsertResult::SuccessStartTraining`. Due to concurrency, it is possible this gets
/// invoked multiple times, and must ensure that quantization tables are only built once.
///
/// Once this function returns `true`, Garnet will invoke several `backfill_quant_vectors()`
/// calls from a thread pool. If it returns false, it may be re-invoked to try again.
///
///  # Safety
///
/// FFI
#[unsafe(no_mangle)]
pub unsafe extern "C" fn build_quant_table(context: u64, index_ptr: *const c_void) -> bool {
    let index = unsafe { &*index_ptr.cast::<Index>() };
    let ctx = Context::new(context);

    index.inner.train_quantizer(&ctx)
}

/// Once quantization tables are successfully built, Garnet invokes this an arbitrary number of
/// times from a thread pool. Each invocation is told its index and the total number of
/// invocations so that each invocation can correctly pick and size its work.
///
/// # Safety
///
/// FFI
#[unsafe(no_mangle)]
pub unsafe extern "C" fn backfill_quant_vectors(
    context: u64,
    index_ptr: *const c_void,
    task_index: usize,
    task_count: usize,
) {
    let index = unsafe { &*index_ptr.cast::<Index>() };
    let ctx = Context::new(context);
    index
        .inner
        .backfill_quant_vectors(&ctx, task_index, task_count);
}

/// # Safety
///
/// FFI
#[unsafe(no_mangle)]
pub unsafe extern "C" fn set_attribute(
    context: u64,
    index_ptr: *const c_void,
    id_data: *const u8,
    id_len: usize,
    attribute_data: *const u8,
    attribute_len: usize,
) -> bool {
    let index = unsafe { &*index_ptr.cast::<Index>() };
    let ctx = Context::new(context);
    let id_bytes = unsafe { slice::from_raw_parts(id_data, id_len) };
    let id = GarnetId::from(id_bytes);

    // Check if the vector exists
    if !index.inner.external_id_exists(&ctx, &id) {
        return false;
    }

    if attribute_len > 0 && !attribute_data.is_null() {
        let attr_data: &[u8] = unsafe { slice::from_raw_parts(attribute_data, attribute_len) };
        if index.inner.set_attributes(&ctx, &id, attr_data).is_err() {
            return false;
        }
    }

    true
}

/// # Safety
///
/// FFI
#[unsafe(no_mangle)]
pub unsafe extern "C" fn search_vector(
    ctx: u64,
    index_ptr: *const c_void,
    vector_data: *const u8,
    vector_len: usize,
    _delta: f32,
    search_exploration_factor: u32,
    _bitmap_data: *const u8,
    bitmap_len: usize,
    _max_filtering_effort: usize,
    output_ids: *mut u8,
    output_ids_len: usize,
    output_distances: *mut f32,
    output_distances_len: usize,
    _continuation: *mut c_void,
) -> i32 {
    let index = unsafe { &*index_ptr.cast::<Index>() };

    let v = if let Some(v) = interpret_vector(index.quant_type, &vector_data, vector_len) {
        v
    } else {
        return -1;
    };

    let ctx = Context::new(ctx);

    let mut output = SearchResults::new(
        output_ids,
        output_ids_len,
        output_distances,
        output_distances_len,
    );

    let knn_params = match search::Knn::new(
        output_distances_len,
        search_exploration_factor as usize,
        None,
    ) {
        Ok(params) => params,
        Err(_) => return -1,
    };

    let res = if bitmap_len == 0 {
        // normal KNN search

        index.inner.search_vector(&ctx, &v, knn_params, &mut output)
    } else {
        // inline filtered search

        let adaptive_l = match AdaptiveL::new(1000, 16.0) {
            Ok(al) => al,
            Err(_) => return -1,
        };
        let params = search::InlineFilterSearch::new(knn_params, Some(adaptive_l));

        index
            .inner
            .filtered_search_vector(&ctx, &v, params, &mut output)
    };
    if let Ok(stats) = res {
        if stats.result_count > i32::MAX as u32 {
            -1
        } else {
            stats.result_count as i32
        }
    } else {
        -1
    }
}

/// # Safety
///
/// FFI
#[unsafe(no_mangle)]
pub unsafe extern "C" fn search_element(
    ctx: u64,
    index_ptr: *const c_void,
    id_data: *const u8,
    id_len: usize,
    _delta: f32,
    search_exploration_factor: u32,
    _bitmap_data: *const u8,
    bitmap_len: usize,
    _max_filtering_effort: usize,
    output_ids: *mut u8,
    output_ids_len: usize,
    output_distances: *mut f32,
    output_distances_len: usize,
    _continuation: *mut c_void,
) -> i32 {
    let index = unsafe { &*index_ptr.cast::<Index>() };
    let id_bytes = unsafe { slice::from_raw_parts(id_data, id_len) };
    let id = GarnetId::from(id_bytes);
    let ctx = Context::new(ctx);

    let mut output = SearchResults::new(
        output_ids,
        output_ids_len,
        output_distances,
        output_distances_len,
    );

    let knn_params = match search::Knn::new(
        output_distances_len,
        search_exploration_factor as usize,
        None,
    ) {
        Ok(knn) => knn,
        Err(_) => return -1,
    };

    let res = if bitmap_len == 0 {
        // normal KNN search

        index
            .inner
            .search_element(&ctx, &id, knn_params, &mut output)
    } else {
        // inline filtered search

        let adaptive_l = match AdaptiveL::new(1000, 16.0) {
            Ok(al) => al,
            Err(_) => return -1,
        };
        let params = search::InlineFilterSearch::new(knn_params, Some(adaptive_l));

        index
            .inner
            .filtered_search_element(&ctx, &id, params, &mut output)
    };

    if let Ok(stats) = res {
        if stats.result_count > i32::MAX as u32 {
            -1
        } else {
            stats.result_count as i32
        }
    } else {
        -1
    }
}

/// # Safety
///
/// FFI
#[unsafe(no_mangle)]
pub unsafe extern "C" fn continue_search(
    _context: u64,
    _index_ptr: *const c_void,
    _continuation: *mut c_void,
    _output_ids: *mut u8,
    _output_ids_len: usize,
    _output_distances: *mut f32,
    _output_distances_len: usize,
    _new_continuation: *mut c_void,
) -> i32 {
    unimplemented!()
}

/// # Safety
///
/// FFI
#[unsafe(no_mangle)]
pub unsafe extern "C" fn remove(
    ctx: u64,
    index_ptr: *const c_void,
    id_data: *const u8,
    id_len: usize,
) -> bool {
    let index = unsafe { &*index_ptr.cast::<Index>() };
    let ctx = Context::new(ctx);
    let id_bytes = unsafe { slice::from_raw_parts(id_data, id_len) };
    let id = GarnetId::from(id_bytes);

    if !index.inner.external_id_exists(&ctx, &id) {
        return false;
    }

    index.inner.remove(&ctx, &id).is_ok()
}

/// # Safety
///
/// FFI
#[unsafe(no_mangle)]
pub unsafe extern "C" fn card(_ctx: u64, index_ptr: *const c_void) -> u64 {
    let index = unsafe { &*index_ptr.cast::<Index>() };

    index.inner.approximate_count()
}

/// # Safety
///
/// FFI
#[unsafe(no_mangle)]
pub unsafe extern "C" fn check_internal_id_valid(
    ctx: u64,
    index_ptr: *const c_void,
    internal_id_data: *const u8,
    internal_id_len: usize,
) -> bool {
    let index = unsafe { &*index_ptr.cast::<Index>() };
    let ctx = Context::new(ctx);
    let internal_id_bytes = unsafe { slice::from_raw_parts(internal_id_data, internal_id_len) };
    if internal_id_bytes.len() != mem::size_of::<u32>() {
        return false;
    }

    let mut id: u32 = 0;
    bytemuck::bytes_of_mut(&mut id).copy_from_slice(internal_id_bytes);

    index.inner.internal_id_exists(&ctx, id)
}

/// # Safety
///
/// FFI
#[unsafe(no_mangle)]
pub unsafe extern "C" fn check_external_id_valid(
    ctx: u64,
    index_ptr: *const c_void,
    id_data: *const u8,
    id_len: usize,
) -> bool {
    let index = unsafe { &*index_ptr.cast::<Index>() };
    let ctx = Context::new(ctx);
    let id_bytes = unsafe { slice::from_raw_parts(id_data, id_len) };
    let id = GarnetId::from(id_bytes);

    index.inner.external_id_exists(&ctx, &id)
}
