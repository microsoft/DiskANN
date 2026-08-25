/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! DiskANN integration and FFI bindings for Garnet vector sets.

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
    neighbor::Neighbor,
    utils::VectorRepr,
};
use diskann_providers::index::wrapped_async::DiskANNIndex;
use diskann_quantization::alloc::Poly;
use diskann_vector::distance::Metric;

use crate::{
    alloc::AlignToEight,
    garnet::{FilterCallback, LogCallback},
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

const ADAPTIVE_L_SAMPLES: usize = 1000;

/// State of index readiness
#[derive(Debug, PartialEq)]
enum IndexState {
    /// No starting points are present in the graph
    NoStartPoints,
    /// Some thread is currently in the process of setting start points
    SettingStartPoints,
    /// Start points set; index ready for normal operation
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

/// Index wrapper type.
/// An `&Arc<Index>` is what will be given out over the FFI.
pub(crate) struct Index {
    /// The type-erased index
    inner: Box<dyn DynIndex>,
    /// The quantizer type of the index
    quant_type: VectorQuantType,
    /// A marker for index readiness; uses `IndexState` as the value
    state: AtomicUsize,
}

/// Element type of vectors in the index
/// NOTE: This must match the definition on the C# side.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(C)]
pub enum VectorValueType {
    Invalid = 0,
    FP32,
    XB8,
}

/// Quantizer type of the index
/// NOTE: This must match the definition on the C# side.
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

/// Helper struct to manage the FFI buffers for handling search results
///
/// If the supplied buffers from Garnet aren't large enough, `overflow_ids` and
/// `overflow_dists` will allocate enough space to store the remainining entries
/// so they can be fetched by Garnet later.
///
/// Having overflow allocations means we don't allocate extra until actually
/// needed. Short result sets can happen without allocation.
///
/// NOTE: The ids will be 4-byte length prefixed, and external IDs are arbitrary
/// length byte strings.
struct SearchResults<'a> {
    k: usize,
    ids: &'a mut [u8],
    dists: &'a mut [f32],
    index: usize,
    id_index: usize,
    overflow_ids: Vec<u8>,
    overflow_dists: Vec<f32>,
}

impl SearchResults<'_> {
    /// Construct from the raw pointers
    fn new(k: usize, ids: *mut u8, ids_len: usize, dists: *mut f32, dists_len: usize) -> Self {
        let ids = unsafe { slice::from_raw_parts_mut(ids, ids_len) };
        let dists = unsafe { slice::from_raw_parts_mut(dists, dists_len) };
        let index = 0;
        let id_index = 0;
        let overflow_ids = Vec::new();
        let overflow_dists = Vec::new();
        Self {
            k,
            ids,
            dists,
            index,
            id_index,
            overflow_ids,
            overflow_dists,
        }
    }

    /// Push an ID only into the results.
    /// This is primarily used by `random_members` which does not use distances.
    fn push_id(&mut self, id: GarnetId) -> diskann::graph::BufferState {
        self.push(Neighbor::new(id, 0.0))
    }

    fn overflowing(&self) -> bool {
        !self.overflow_ids.is_empty()
    }

    fn into_overflows(self) -> (Vec<u8>, Vec<f32>) {
        (self.overflow_ids, self.overflow_dists)
    }

    fn is_full(&self) -> bool {
        self.index + self.overflow_dists.len() >= self.k
    }
}

impl SearchOutputBuffer<GarnetId> for SearchResults<'_> {
    fn size_hint(&self) -> Option<usize> {
        Some(self.k - self.index - self.overflow_dists.len())
    }

    fn push(&mut self, neighbor: Neighbor<GarnetId>) -> diskann::graph::BufferState {
        let (id, distance) = neighbor.as_tuple();

        if self.is_full() {
            return diskann::graph::BufferState::Full;
        } else if self.overflowing()
            || self.index >= self.dists.len()
            || self.id_index + mem::size_of::<u32>() + id.len() > self.ids.len()
        {
            self.overflow_ids
                .extend_from_slice(id.as_prefixed_key_bytes());
            self.overflow_dists.push(distance);

            if self.is_full() {
                return diskann::graph::BufferState::Full;
            } else {
                return diskann::graph::BufferState::Available;
            }
        }

        let id_len = id.len() as u32;
        self.ids[self.id_index..self.id_index + mem::size_of::<u32>()]
            .copy_from_slice(bytemuck::bytes_of(&id_len));
        self.id_index += mem::size_of::<u32>();

        self.ids[self.id_index..self.id_index + id.len()].copy_from_slice(&id);
        self.dists[self.index] = distance;
        self.index += 1;
        self.id_index += id.len();

        if self.is_full() {
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
        Itr: IntoIterator<Item = Neighbor<GarnetId>>,
    {
        let initial = self.current_len();

        for neighbor in itr {
            if self.push(neighbor).is_full() {
                break;
            }
        }

        self.current_len() - initial
    }
}

/// Helper generic function to create the correct type-erased `Arc<Index>`.
/// This also returns a bool indicating whether quantization is needed.
fn create_index_impl<T: VectorRepr>(
    quant_type: VectorQuantType,
    config: config::Config,
    dim: usize,
    metric_type: Metric,
    max_degree: usize,
    callbacks: Callbacks,
    context: Context,
) -> Result<(Arc<Index>, bool), GarnetProviderError> {
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

    let quant_needed = match quant_type {
        VectorQuantType::Bin | VectorQuantType::XBinI8 | VectorQuantType::XBinU8 => {
            provider.quantization_needed()
        }
        _ => false,
    };

    Ok((
        Arc::new(Index {
            inner: Box::new(DiskANNIndex::new_with_current_thread_runtime(
                config, provider,
            )),
            quant_type,
            state,
        }),
        quant_needed,
    ))
}

/// Create an index.
///
/// Constructs a type-erased DiskANN index object as a `Arc<Index>` and return a pointer
/// to the leaked Arc. This pointer must be freed with `drop_index()`.
///
/// Returns `ptr::null()` if there is an error. Sets the `quantization_needed` outvar if
/// the index requires quantization callbacks during its lifecycle.
///
/// Note that `quantization_needed` can be set to false even when a quantizer is used. The
/// flag controls whether supplemental control is needed from Garnet to manage quantizers
/// which require training and backfill.
///
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
    log_callback: LogCallback,
    quantization_needed: *mut bool,
) -> *const c_void {
    unsafe { *quantization_needed = false };

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
        log_callback,
    );

    match quant_type {
        VectorQuantType::Invalid => ptr::null(),
        VectorQuantType::XNoQuantU8 | VectorQuantType::XBinU8 => {
            if let Ok((index, quant_needed)) = create_index_impl::<u8>(
                quant_type,
                config,
                dim as usize,
                metric_type,
                max_degree as usize,
                callbacks,
                context,
            ) {
                unsafe { *quantization_needed = quant_needed };
                Arc::into_raw(index).cast::<c_void>()
            } else {
                ptr::null()
            }
        }
        VectorQuantType::XNoQuantI8 | VectorQuantType::XBinI8 => {
            if let Ok((index, quant_needed)) = create_index_impl::<i8>(
                quant_type,
                config,
                dim as usize,
                metric_type,
                max_degree as usize,
                callbacks,
                context,
            ) {
                unsafe { *quantization_needed = quant_needed };
                Arc::into_raw(index).cast::<c_void>()
            } else {
                ptr::null()
            }
        }
        VectorQuantType::NoQuant | VectorQuantType::Bin | VectorQuantType::Q8 => {
            if let Ok((index, quant_needed)) = create_index_impl::<f32>(
                quant_type,
                config,
                dim as usize,
                metric_type,
                max_degree as usize,
                callbacks,
                context,
            ) {
                unsafe { *quantization_needed = quant_needed };
                Arc::into_raw(index).cast::<c_void>()
            } else {
                ptr::null()
            }
        }
    }
}

/// Drop an index.
///
/// This is the only valid way to free an index pointer created with `create_index()`.
///
/// # Safety
///
/// FFI
#[unsafe(no_mangle)]
pub unsafe extern "C" fn drop_index(_ctx: u64, index_ptr: *const c_void) {
    // SAFETY: Caller must pass in a valid pointer returned from `create_index`.
    let _ = unsafe { Arc::from_raw(index_ptr.cast::<Index>()) };
}

/// `Cow` type for `Poly<[u8], AlignToEight>` types.
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

/// Helper function to interpret the vector pointer and size into a usable Rust
/// type. This will return either a borrowed or owned vector depending on how
/// the pointer is aligned. Since Garnet doesn't guarantee the alignment, if it
/// is not 4-byte aligned, we must allocate an appropriately aligned buffer to
/// access it as its correct element type.
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

/// Return type for `insert()`.
///
/// `Fail` and `Success` are obvious. `SuccessStartTraining` is used when enough vectors have
/// been inserted to start training the quantizer. That return value signals to Garnet that
/// `build_quant_table` should be called.
#[derive(Debug, Clone, Copy, PartialEq)]
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

#[cfg(test)]
impl From<u8> for InsertResult {
    fn from(value: u8) -> Self {
        match value {
            1 => InsertResult::Success,
            2 => InsertResult::SuccessStartTraining,
            _ => InsertResult::Fail,
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

    let old_ready = ctx.quantizer_ready();

    // Insert the vector
    if index.inner.insert(&ctx, &id, &v).is_ok() {
        // Write attributes to garnet. These are written after insert since
        // they are keyed on internal id.
        let attr_data = if attribute_len > 0 && !attribute_data.is_null() {
            unsafe { slice::from_raw_parts(attribute_data, attribute_len) }
        } else {
            &[]
        };
        if index.inner.set_attributes(&ctx, &id, attr_data).is_err() {
            return InsertResult::Fail.into();
        }

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

/// Ensures the index is ready to be used, and if not, runs the `init` function.
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
/// # Safety
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
/// Returns true for success and false otherwise.
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
) -> bool {
    let index = unsafe { &*index_ptr.cast::<Index>() };
    let ctx = Context::new(context);
    index
        .inner
        .backfill_quant_vectors(&ctx, task_index, task_count)
}

/// Set the attributes for a vector.
///
/// Setting attributes with `attribute_len == 0` is equivalent to deleting them.
///
/// Returns true for success and false otherwise.
///
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

    if !attribute_data.is_null() {
        let attr_data: &[u8] = unsafe { slice::from_raw_parts(attribute_data, attribute_len) };
        if !attr_data.is_empty() {
            if index.inner.set_attributes(&ctx, &id, attr_data).is_err() {
                return false;
            }
        } else {
            // Empty attribute string is interpreted as deletion
            if index.inner.delete_attributes(&ctx, &id).is_err() {
                return false;
            }
        }
    }

    true
}

/// Search continuation container.
///
/// This will be boxed and a pointer given to Garnet in order to signal that more results than
/// fit in the provided buffer are available. Garnet will hand this back along with new
/// results buffers to access more results. `drain()` is used to fill those buffers and
/// update the continuation.
pub struct Continuation {
    index: usize,
    id_index: usize,
    id_buffer: Vec<u8>,
    dist_buffer: Vec<f32>,
}

impl Continuation {
    /// Construct a new `Box<Continuation>` from the overflow buffers
    pub fn new(id_buffer: Vec<u8>, dist_buffer: Vec<f32>) -> Box<Self> {
        let index = 0;
        let id_index = 0;
        Box::new(Self {
            index,
            id_index,
            id_buffer,
            dist_buffer,
        })
    }

    /// Turn a raw pointer back into `Box<Continuation>`
    ///
    /// # SAFETY
    ///
    /// This must only be called once on the pointer.
    pub unsafe fn from_ptr(ptr: *mut c_void) -> Box<Self> {
        unsafe { Box::from_raw(ptr as *mut Continuation) }
    }

    /// Turn a `Box<Continuation>` into a raw pointer, leaking the memory.
    ///
    /// To free the memory, `from_ptr()` must be used to turn it back into a Box which can then
    /// be dropped normally.
    pub fn into_ptr(self: Box<Self>) -> *mut c_void {
        Box::into_raw(self) as *mut c_void
    }

    /// Drains the continuation buffers into the provided `ids` and `dists` buffers.
    ///
    /// Returns the amount drained.
    pub fn drain(&mut self, ids: &mut [u8], dists: &mut [f32]) -> usize {
        let mut index = 0;
        let mut id_index = 0;

        // Scan `id_buffer` until we reach the first id that doesn't fit because one of the
        // buffers is full.
        let mut count = 0;
        let prefix_len = mem::size_of::<u32>();
        while index < dists.len()
            && id_index < ids.len()
            && self.index < self.dist_buffer.len()
            && self.id_index < self.id_buffer.len()
        {
            // Read length prefix
            if id_index + prefix_len > ids.len() {
                break;
            }
            let mut len = 0u32;
            bytemuck::bytes_of_mut(&mut len)
                .copy_from_slice(&self.id_buffer[self.id_index..self.id_index + prefix_len]);

            // We check there is room before advancing the indices
            if id_index + prefix_len + len as usize > ids.len() {
                break;
            }

            // Copy length prefix
            ids[id_index..id_index + prefix_len]
                .copy_from_slice(&self.id_buffer[self.id_index..self.id_index + prefix_len]);

            id_index += prefix_len;
            self.id_index += prefix_len;

            // Copy ID

            ids[id_index..id_index + len as usize]
                .copy_from_slice(&self.id_buffer[self.id_index..self.id_index + len as usize]);

            id_index += len as usize;
            self.id_index += len as usize;

            // Copy the distance
            dists[index] = self.dist_buffer[self.index];
            index += 1;
            self.index += 1;

            count += 1;
        }

        count
    }

    /// Determines whether the continuation is finished
    pub fn is_empty(&self) -> bool {
        self.index >= self.dist_buffer.len()
    }
}

/// Search the closest vectors to the given query vector.
///
/// The k value for the search is implied by `output_distances_len`. The output
/// distances buffer will be correctly sized for k, but because IDs are variable
/// length, the output_ids buffer may be too small. If that happens, a continuation
/// will be returned so that Garnet can use `continue_search` to fetch the
/// remaining results.
///
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
    bitmap_data: *const u8,
    bitmap_len: usize,
    max_filtering_effort: usize,
    output_ids: *mut u8,
    output_ids_len: usize,
    output_distances: *mut f32,
    output_distances_len: usize,
    beam_width: u32,
    continuation: *mut *mut c_void,
) -> i32 {
    let index = unsafe { &*index_ptr.cast::<Index>() };

    let v = if let Some(v) = interpret_vector(index.quant_type, &vector_data, vector_len) {
        v
    } else {
        return -1;
    };

    let ctx = Context::new(ctx);

    let mut output = SearchResults::new(
        output_distances_len,
        output_ids,
        output_ids_len,
        output_distances,
        output_distances_len,
    );

    let knn_params = match search::Knn::new(
        search_exploration_factor as usize,
        Some(beam_width as usize),
    ) {
        Ok(params) => params,
        Err(_) => return -1,
    };

    let res = if bitmap_data.is_null() || bitmap_len == 0 {
        // normal KNN search

        index.inner.search_vector(&ctx, &v, knn_params, &mut output)
    } else {
        // inline filtered search

        let adaptive_l = match AdaptiveL::new(ADAPTIVE_L_SAMPLES, max_filtering_effort as f64) {
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
            let count = output.current_len();

            if continuation.is_null() {
                index.inner.log(&ctx, "continuation argument was null");
                return -1;
            }

            if output.overflowing() {
                let (id_buffer, dist_buffer) = output.into_overflows();
                let cont = Continuation::new(id_buffer, dist_buffer);
                unsafe { continuation.write(cont.into_ptr()) };
            } else {
                unsafe { continuation.write(ptr::null_mut()) };
            }

            count as i32
        }
    } else {
        -1
    }
}

/// Search the closest vectors to the given existing vector in the index.
///
/// This is a thin wrapper around `search_vector`, so see its documentation for more details.
///
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
    bitmap_data: *const u8,
    bitmap_len: usize,
    max_filtering_effort: usize,
    output_ids: *mut u8,
    output_ids_len: usize,
    output_distances: *mut f32,
    output_distances_len: usize,
    beam_width: u32,
    continuation: *mut *mut c_void,
) -> i32 {
    let index = unsafe { &*index_ptr.cast::<Index>() };
    let id_bytes = unsafe { slice::from_raw_parts(id_data, id_len) };
    let id = GarnetId::from(id_bytes);
    let ctx = Context::new(ctx);

    let mut output = SearchResults::new(
        output_distances_len,
        output_ids,
        output_ids_len,
        output_distances,
        output_distances_len,
    );

    let knn_params = match search::Knn::new(
        search_exploration_factor as usize,
        Some(beam_width as usize),
    ) {
        Ok(knn) => knn,
        Err(_) => return -1,
    };

    let res = if bitmap_data.is_null() || bitmap_len == 0 {
        // normal KNN search

        index
            .inner
            .search_element(&ctx, &id, knn_params, &mut output)
    } else {
        // inline filtered search

        let adaptive_l = match AdaptiveL::new(ADAPTIVE_L_SAMPLES, max_filtering_effort as f64) {
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
            let count = output.current_len();

            if continuation.is_null() {
                index.inner.log(&ctx, "continuation argument was null");
                return -1;
            }

            if output.overflowing() {
                let (id_buffer, dist_buffer) = output.into_overflows();
                let cont = Continuation::new(id_buffer, dist_buffer);
                unsafe { continuation.write(cont.into_ptr()) };
            } else {
                unsafe { continuation.write(ptr::null_mut()) };
            }

            count as i32
        }
    } else {
        -1
    }
}

/// Continue getting results for a previously executed search.
///
/// Positive return values are the count of vectors returned. `-1` will be returned on errors.
/// If further continuation is needed, `new_continuation` will be set to a new continuation
/// pointer.
///
/// # Safety
///
/// FFI
#[unsafe(no_mangle)]
pub unsafe extern "C" fn continue_search(
    ctx: u64,
    index_ptr: *const c_void,
    continuation: *mut c_void,
    output_ids: *mut u8,
    output_ids_len: usize,
    output_distances: *mut f32,
    output_distances_len: usize,
    new_continuation: *mut *mut c_void,
) -> i32 {
    let index = unsafe { &*index_ptr.cast::<Index>() };
    let ctx = Context::new(ctx);

    if continuation.is_null() {
        index.inner.log(&ctx, "continuation argument was null");
        if !new_continuation.is_null() {
            unsafe { new_continuation.write(ptr::null_mut()) };
        }
        return -1;
    }

    if new_continuation.is_null() {
        index.inner.log(&ctx, "new_continuation argument was null");
        return -1;
    }

    let output_ids = unsafe { slice::from_raw_parts_mut(output_ids, output_ids_len) };
    let output_distances =
        unsafe { slice::from_raw_parts_mut(output_distances, output_distances_len) };
    let mut continuation = unsafe { Continuation::from_ptr(continuation) };
    let count = continuation.drain(output_ids, output_distances);

    if !continuation.is_empty() {
        unsafe { new_continuation.write(continuation.into_ptr()) };
    } else {
        unsafe { new_continuation.write(ptr::null_mut()) };
    }

    count as i32
}

/// Remove a vector from the index.
///
/// Returns true on success and false otherwise.
///
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

/// Return the approximate count of vectors in the index.
///
/// # Safety
///
/// FFI
#[unsafe(no_mangle)]
pub unsafe extern "C" fn card(_ctx: u64, index_ptr: *const c_void) -> u64 {
    let index = unsafe { &*index_ptr.cast::<Index>() };

    index.inner.approximate_count()
}

/// Check if a given internal ID is a valid vector.
///
/// Returns true if the vector exists, and false otherwise.
///
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

/// Check if a given external ID is a valid vector.
///
/// Returns true if the vector exists, and false otherwise.
///
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

/// Returns random vectors from the index.
///
/// This is primarily a debugging aid.
///
/// Returns true on success and false otherwise.
///
/// # Safety
///
/// FFI
#[unsafe(no_mangle)]
pub unsafe extern "C" fn random_members(
    ctx: u64,
    index_ptr: *const c_void,
    count: u32,
    output_ids: *mut u8,
    output_ids_len: usize,
) -> bool {
    let index = unsafe { &*index_ptr.cast::<Index>() };
    let ctx = Context::new(ctx);

    // Dummy buffer for distances
    let mut output_distances = vec![0f32; output_ids_len / 5];
    let mut output = SearchResults::new(
        count as usize,
        output_ids,
        output_ids_len,
        output_distances.as_mut_ptr(),
        output_distances.len(),
    );

    index.inner.random_members(&ctx, count, &mut output)
}

/// Return the neighbors for an index vector.
///
/// This is primarily a debugging aid. It returns both the neighbors' IDs and their
/// distance from the given vector.
///
/// Returns the number of results or `-1` on error.
///
/// # Safety
///
/// FFI
#[unsafe(no_mangle)]
pub unsafe extern "C" fn search_neighbors(
    ctx: u64,
    index_ptr: *const c_void,
    id_data: *const u8,
    id_len: usize,
    output_ids: *mut u8,
    output_ids_len: usize,
    output_distances: *mut f32,
    output_distances_len: usize,
    continuation: *mut *mut c_void,
) -> i32 {
    let index = unsafe { &*index_ptr.cast::<Index>() };
    let ctx = Context::new(ctx);
    let id_bytes = unsafe { slice::from_raw_parts(id_data, id_len) };
    let id = GarnetId::from(id_bytes);

    let mut output = SearchResults::new(
        index.inner.max_degree(),
        output_ids,
        output_ids_len,
        output_distances,
        output_distances_len,
    );

    let Ok(neighbors) = index.inner.neighbors(&ctx, &id) else {
        return -1;
    };

    output.extend(neighbors);

    if continuation.is_null() {
        index.inner.log(&ctx, "continuation argument was null");
        return -1;
    }

    let count = output.current_len();
    if output.overflowing() {
        let (id_buffer, dist_buffer) = output.into_overflows();
        let cont = Continuation::new(id_buffer, dist_buffer);
        unsafe { continuation.write(cont.into_ptr()) };
    } else {
        unsafe { continuation.write(ptr::null_mut()) };
    }

    count as i32
}

#[cfg(test)]
mod tests {
    use std::{mem, ptr};

    use diskann::{
        graph::{BufferState, SearchOutputBuffer},
        neighbor::Neighbor,
    };
    use diskann_vector::distance::Metric;
    use rand::Rng;

    use crate::{
        Index, IndexState, PolyCow, SearchResults, VectorQuantType, drop_index,
        garnet::{Context, GarnetId, Term},
        test_utils::Store,
    };

    #[test]
    fn index_state() {
        assert_eq!(IndexState::from(0), IndexState::NoStartPoints);
        assert_eq!(IndexState::from(1), IndexState::SettingStartPoints);
        assert_eq!(IndexState::from(2), IndexState::Ready);
    }

    #[test]
    fn search_results() {
        let mut ids = vec![0u8; 40]; // 20 bytes for 5 IDs; 20 bytes for 5 length prefixes
        let mut dists = vec![0.0f32; 5];
        let ids_buffer = ids.as_mut_ptr();
        let ids_len = ids.len();
        let dists_buffer = dists.as_mut_ptr();
        let dists_len = dists.len();

        let mut sr = SearchResults::new(5, ids_buffer, ids_len, dists_buffer, dists_len);

        assert_eq!(sr.size_hint(), Some(5));

        let test_data = [
            Neighbor::new(GarnetId::from(bytemuck::bytes_of(&1u32)), 1.1f32),
            Neighbor::new(GarnetId::from(bytemuck::bytes_of(&2u32)), 2.1),
            Neighbor::new(GarnetId::from(bytemuck::bytes_of(&3u32)), 3.1),
            Neighbor::new(GarnetId::from(bytemuck::bytes_of(&4u32)), 4.1),
            Neighbor::new(GarnetId::from(bytemuck::bytes_of(&5u32)), 5.1),
        ];

        assert_eq!(sr.current_len(), 0);

        sr.extend(test_data);

        assert_eq!(sr.current_len(), 5);

        let mut pos = 0usize;
        for (i, d) in dists.iter().enumerate() {
            let mut size = 0u32;
            bytemuck::bytes_of_mut(&mut size).copy_from_slice(&ids[pos..pos + 4]);
            pos += 4;

            assert_eq!(size, 4);

            let mut id = 0u32;
            bytemuck::bytes_of_mut(&mut id).copy_from_slice(&ids[pos..pos + 4]);
            pos += 4;

            assert_eq!(id, i as u32 + 1);

            assert_eq!(*d, i as f32 + 1.1);
        }

        assert_eq!(
            sr.push(Neighbor::new(
                GarnetId::from(bytemuck::bytes_of(&6u32)),
                6.1f32
            )),
            BufferState::Full
        );
    }

    #[test]
    fn continue_search() {
        let first_id = b"first";
        let second_id = b"second";
        let mut continuation_ids = Vec::new();
        continuation_ids.extend_from_slice(bytemuck::bytes_of(&(first_id.len() as u32)));
        continuation_ids.extend_from_slice(first_id);
        continuation_ids.extend_from_slice(bytemuck::bytes_of(&(second_id.len() as u32)));
        continuation_ids.extend_from_slice(second_id);

        let continuation = super::Continuation::new(continuation_ids, vec![1.25, 2.5]).into_ptr();
        let mut next_continuation = ptr::null_mut();
        let mut output_ids = vec![0u8; mem::size_of::<u32>() + first_id.len()];
        let mut output_distances = [0.0f32; 1];

        let count = unsafe {
            super::continue_search(
                0,
                ptr::null(),
                continuation,
                output_ids.as_mut_ptr(),
                output_ids.len(),
                output_distances.as_mut_ptr(),
                output_distances.len(),
                &mut next_continuation,
            )
        };

        assert_eq!(count, 1);
        assert_eq!(
            output_ids[..mem::size_of::<u32>()],
            (first_id.len() as u32).to_le_bytes()
        );
        assert_eq!(&output_ids[mem::size_of::<u32>()..], first_id);
        assert_eq!(output_distances, [1.25]);
        assert!(!next_continuation.is_null());

        let mut final_continuation = ptr::null_mut();
        let mut output_ids = vec![0u8; mem::size_of::<u32>() + second_id.len()];
        let mut output_distances = [0.0f32; 1];
        let count = unsafe {
            super::continue_search(
                0,
                ptr::null(),
                next_continuation,
                output_ids.as_mut_ptr(),
                output_ids.len(),
                output_distances.as_mut_ptr(),
                output_distances.len(),
                &mut final_continuation,
            )
        };

        assert_eq!(count, 1);
        assert_eq!(
            output_ids[..mem::size_of::<u32>()],
            (second_id.len() as u32).to_le_bytes()
        );
        assert_eq!(&output_ids[mem::size_of::<u32>()..], second_id);
        assert_eq!(output_distances, [2.5]);
        assert!(final_continuation.is_null());
    }

    fn check_create_index(quant_type: VectorQuantType) {
        let store = Store::new();
        let mut quant_needed = false;
        let index_ptr = unsafe {
            super::create_index(
                0,
                2,
                0,
                quant_type,
                Metric::L2.into(),
                10,
                8,
                store.callbacks().read_callback(),
                store.callbacks().write_callback(),
                store.callbacks().delete_callback(),
                store.callbacks().rmw_callback(),
                store.callbacks().filter_callback(),
                store.callbacks().log_callback(),
                &mut quant_needed,
            )
        };
        assert!(!index_ptr.is_null());
        let index = unsafe { &*index_ptr.cast::<Index>() };
        assert_eq!(index.quant_type, quant_type);
        assert_eq!(index.inner.approximate_count(), 0);

        unsafe {
            drop_index(0, index_ptr);
        }
    }

    #[test]
    fn create_index() {
        let store = Store::new();
        let mut quant_needed = false;

        let index_ptr = unsafe {
            super::create_index(
                0,
                2,
                0,
                VectorQuantType::Invalid,
                Metric::L2.into(),
                10,
                8,
                store.callbacks().read_callback(),
                store.callbacks().write_callback(),
                store.callbacks().delete_callback(),
                store.callbacks().rmw_callback(),
                store.callbacks().filter_callback(),
                store.callbacks().log_callback(),
                &mut quant_needed,
            )
        };
        assert!(index_ptr.is_null());

        check_create_index(VectorQuantType::NoQuant);
        check_create_index(VectorQuantType::Bin);
        check_create_index(VectorQuantType::Q8);
        check_create_index(VectorQuantType::XNoQuantU8);
        check_create_index(VectorQuantType::XBinU8);
        check_create_index(VectorQuantType::XNoQuantI8);
        check_create_index(VectorQuantType::XBinI8);
    }

    #[test]
    fn interpret_vector() {
        // f32; correctly aligned
        let v = vec![0.0f32; 2];
        let v_ptr = bytemuck::cast_slice::<f32, u8>(&v).as_ptr();
        let res = super::interpret_vector(VectorQuantType::NoQuant, &v_ptr, v.len());
        assert!(matches!(res, Some(PolyCow::Borrowed(_))));

        // f32; unaligned
        let real_v = vec![0.0f32; 2];
        let mut v = vec![0u8; 2 * mem::size_of::<f32>() + 1];
        v[1..].copy_from_slice(bytemuck::cast_slice::<f32, u8>(&real_v));
        let v_ptr = unsafe { v.as_ptr().offset(1) };
        let res = super::interpret_vector(VectorQuantType::NoQuant, &v_ptr, real_v.len());
        assert!(matches!(res, Some(PolyCow::Owned(_))));

        // i8
        let v = vec![0i8; 2];
        let v_ptr = bytemuck::cast_slice::<i8, u8>(&v).as_ptr();
        let res = super::interpret_vector(VectorQuantType::XNoQuantI8, &v_ptr, v.len());
        assert!(matches!(res, Some(PolyCow::Borrowed(_))));

        let res = super::interpret_vector(VectorQuantType::Invalid, &ptr::null() as &*const u8, 0);
        assert!(res.is_none());
    }

    #[test]
    fn set_and_delete_attributes() {
        let store = Store::new();
        let mut quant_needed = false;

        let index_ptr = unsafe {
            super::create_index(
                0,
                2,
                0,
                VectorQuantType::NoQuant,
                Metric::L2.into(),
                10,
                8,
                store.callbacks().read_callback(),
                store.callbacks().write_callback(),
                store.callbacks().delete_callback(),
                store.callbacks().rmw_callback(),
                store.callbacks().filter_callback(),
                store.callbacks().log_callback(),
                &mut quant_needed,
            )
        };

        assert!(!index_ptr.is_null());

        let id = 0u32;
        let eid = GarnetId::from(bytemuck::bytes_of(&id));
        let metadata = b"{'foo': 0}";
        let ctx = Context::new(0);
        let v = [0.0f32, 0.0f32];

        assert_eq!(
            unsafe {
                super::insert(
                    ctx.get(),
                    index_ptr,
                    eid.as_ptr(),
                    eid.len(),
                    bytemuck::cast_slice::<f32, u8>(&v).as_ptr(),
                    v.len(),
                    metadata.as_ptr(),
                    metadata.len(),
                )
            },
            1
        );
        let iid = store.get(ctx.term(Term::IntMap).get(), &eid).unwrap();
        assert_eq!(
            store.get(ctx.term(Term::Attributes).get(), &iid),
            Some(metadata.as_slice().to_owned())
        );

        assert!(unsafe {
            super::set_attribute(
                ctx.get(),
                index_ptr,
                eid.as_ptr(),
                eid.len(),
                b"".as_ptr(),
                0,
            )
        });
        assert!(store.get(ctx.term(Term::Attributes).get(), &iid).is_none());

        unsafe {
            drop_index(0, index_ptr);
        }
    }

    #[test]
    fn random_members() {
        let store = Store::new();
        let mut quant_needed = false;

        let index_ptr = unsafe {
            super::create_index(
                0,
                2,
                0,
                VectorQuantType::NoQuant,
                Metric::L2.into(),
                10,
                8,
                store.callbacks().read_callback(),
                store.callbacks().write_callback(),
                store.callbacks().delete_callback(),
                store.callbacks().rmw_callback(),
                store.callbacks().filter_callback(),
                store.callbacks().log_callback(),
                &mut quant_needed,
            )
        };

        assert!(!index_ptr.is_null());

        let ctx = Context::new(0);
        let mut rng = rand::rng();

        for id in 0..100 {
            let mut v = vec![0u8; 2];
            rng.fill(v.as_mut_slice());
            let v = v.into_iter().map(|i| i as f32).collect::<Vec<f32>>();

            let eid = GarnetId::from(bytemuck::bytes_of(&id));
            assert_eq!(
                unsafe {
                    super::insert(
                        ctx.get(),
                        index_ptr,
                        eid.as_ptr(),
                        eid.len(),
                        bytemuck::cast_slice::<f32, u8>(&v).as_ptr(),
                        v.len(),
                        ptr::null(),
                        0,
                    )
                },
                1
            );
        }

        // Check basic correctness
        let mut output_ids = vec![u32::MAX; 20];
        assert!(unsafe {
            super::random_members(
                ctx.get(),
                index_ptr,
                10,
                bytemuck::cast_slice_mut::<u32, u8>(output_ids.as_mut_slice()).as_mut_ptr(),
                output_ids.len() * mem::size_of::<u32>(),
            )
        });
        assert!(
            output_ids
                .iter()
                .enumerate()
                .all(|(i, e)| if i.is_multiple_of(2) {
                    *e == 4
                } else {
                    *e < 100
                })
        );

        // Check undersized buffer
        output_ids.fill(u32::MAX);
        assert!(unsafe {
            super::random_members(
                ctx.get(),
                index_ptr,
                20,
                bytemuck::cast_slice_mut::<u32, u8>(output_ids.as_mut_slice()).as_mut_ptr(),
                output_ids.len() * mem::size_of::<u32>(),
            )
        });
        assert!(
            output_ids
                .iter()
                .enumerate()
                .all(|(i, e)| if i.is_multiple_of(2) {
                    *e == 4
                } else {
                    *e < 100
                })
        );

        // Check oversized buffer
        output_ids.fill(u32::MAX);
        assert!(unsafe {
            super::random_members(
                ctx.get(),
                index_ptr,
                5,
                bytemuck::cast_slice_mut::<u32, u8>(output_ids.as_mut_slice()).as_mut_ptr(),
                output_ids.len() * mem::size_of::<u32>(),
            )
        });
        assert!(output_ids.iter().enumerate().all(|(i, e)| if i < 10 {
            if i.is_multiple_of(2) {
                *e == 4
            } else {
                *e < 100
            }
        } else {
            *e == u32::MAX
        }));

        // Delete 50 vectors at random
        let ids = rand::seq::index::sample(&mut rng, 100, 50);
        for id in ids {
            let id = id as u32;
            let eid = GarnetId::from(bytemuck::bytes_of(&id));

            assert!(unsafe { super::remove(ctx.get(), index_ptr, eid.as_ptr(), eid.len()) });
        }

        // Check basic correctness
        output_ids.fill(u32::MAX);
        assert!(unsafe {
            super::random_members(
                ctx.get(),
                index_ptr,
                10,
                bytemuck::cast_slice_mut::<u32, u8>(output_ids.as_mut_slice()).as_mut_ptr(),
                output_ids.len() * mem::size_of::<u32>(),
            )
        });
        assert!(
            output_ids
                .iter()
                .enumerate()
                .all(|(i, e)| if i.is_multiple_of(2) {
                    *e == 4
                } else {
                    *e < 100
                })
        );

        // Check undersized buffer
        output_ids.fill(u32::MAX);
        assert!(unsafe {
            super::random_members(
                ctx.get(),
                index_ptr,
                20,
                bytemuck::cast_slice_mut::<u32, u8>(output_ids.as_mut_slice()).as_mut_ptr(),
                output_ids.len() * mem::size_of::<u32>(),
            )
        });
        assert!(
            output_ids
                .iter()
                .enumerate()
                .all(|(i, e)| if i.is_multiple_of(2) {
                    *e == 4
                } else {
                    *e < 100
                })
        );

        // Check oversized buffer
        output_ids.fill(u32::MAX);
        assert!(unsafe {
            super::random_members(
                ctx.get(),
                index_ptr,
                5,
                bytemuck::cast_slice_mut::<u32, u8>(output_ids.as_mut_slice()).as_mut_ptr(),
                output_ids.len() * mem::size_of::<u32>(),
            )
        });
        assert!(output_ids.iter().enumerate().all(|(i, e)| if i < 10 {
            if i.is_multiple_of(2) {
                *e == 4
            } else {
                *e < 100
            }
        } else {
            *e == u32::MAX
        }));
    }

    #[test]
    fn search_neighbors() {
        let store = Store::new();
        let mut quant_needed = false;

        let index_ptr = unsafe {
            super::create_index(
                0,
                2,
                0,
                VectorQuantType::NoQuant,
                Metric::L2.into(),
                10,
                8,
                store.callbacks().read_callback(),
                store.callbacks().write_callback(),
                store.callbacks().delete_callback(),
                store.callbacks().rmw_callback(),
                store.callbacks().filter_callback(),
                store.callbacks().log_callback(),
                &mut quant_needed,
            )
        };

        assert!(!index_ptr.is_null());

        let ctx = Context::new(0);
        let mut rng = rand::rng();

        for id in 0..100 {
            let mut v = vec![0u8; 2];
            rng.fill(v.as_mut_slice());
            let v = v.into_iter().map(|i| i as f32).collect::<Vec<f32>>();

            let eid = GarnetId::from(bytemuck::bytes_of(&id));
            assert_eq!(
                unsafe {
                    super::insert(
                        ctx.get(),
                        index_ptr,
                        eid.as_ptr(),
                        eid.len(),
                        bytemuck::cast_slice::<f32, u8>(&v).as_ptr(),
                        v.len(),
                        ptr::null(),
                        0,
                    )
                },
                1
            );
        }

        let mut output_ids = vec![u32::MAX; 20];
        let mut output_dists = vec![f32::MAX; 10];
        let good_id = GarnetId::from(bytemuck::bytes_of(&25u32));
        let bad_id = GarnetId::from(bytemuck::bytes_of(&250u32));

        // check the good case
        let mut continuation = ptr::null_mut();
        let count = unsafe {
            super::search_neighbors(
                ctx.get(),
                index_ptr,
                good_id.as_ptr(),
                good_id.len(),
                bytemuck::cast_slice_mut(&mut output_ids).as_mut_ptr(),
                output_ids.len() * mem::size_of::<u32>(),
                output_dists.as_mut_ptr(),
                output_dists.len(),
                &mut continuation,
            )
        };

        assert!(count > 0 && count <= 8, "count = {count}");
        assert!(continuation.is_null());

        for i in 0..count as usize {
            assert_eq!(output_ids[i * 2], 4);
            assert!(output_ids[i * 2 + 1] < 100);
            assert!(output_dists[i] < f32::MAX);
        }

        let mut continuation = ptr::null_mut();
        let count = unsafe {
            super::search_neighbors(
                ctx.get(),
                index_ptr,
                bad_id.as_ptr(),
                bad_id.len(),
                bytemuck::cast_slice_mut(&mut output_ids).as_mut_ptr(),
                output_ids.len() * mem::size_of::<u32>(),
                output_dists.as_mut_ptr(),
                output_dists.len(),
                &mut continuation,
            )
        };

        assert!(count < 0);
        assert!(continuation.is_null());
    }
}
