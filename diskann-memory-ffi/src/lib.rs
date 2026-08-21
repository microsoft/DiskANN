/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

#![warn(missing_debug_implementations)]

use std::{
    collections::{HashMap, HashSet},
    ffi::{c_char, c_void, CStr, CString},
    num::NonZeroUsize,
    panic::{catch_unwind, AssertUnwindSafe},
    path::{Path, PathBuf},
    ptr,
    sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        Arc, Mutex, OnceLock, RwLock, RwLockReadGuard, RwLockWriteGuard,
    },
};

use diskann::{
    graph::{
        config::{Builder as GraphConfigBuilder, MaxDegree},
        search::Knn,
        search_output_buffer, InplaceDeleteMethod,
    },
    provider::DataProvider,
    utils::ONE,
};
use diskann_inmem::{
    layers::Full, load_streaming_snapshot, save_streaming_snapshot, Context, Provider, Strategy,
    StreamingSnapshotConfig, StreamingSnapshotError, StreamingTag, Tag128,
};
use diskann_providers::{
    index::wrapped_async::DiskANNIndex,
    model::{
        graph::provider::async_::{
            common::{FullPrecision as LegacyStrategy, NoStore},
            inmem::FullPrecisionProvider,
        },
        IndexConfiguration as ProviderIndexConfiguration,
    },
    storage::FileStorageProvider,
};
use diskann_vector::distance::Metric as VectorMetric;

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiskANNError {
    None = 0,
    NullPointer = 1,
    InvalidPath = 2,
    InvalidUtf8 = 3,
    InvalidBuffer = 4,
    LoadFailed = 5,
    SearchFailed = 6,
    InvalidConfig = 7,
    InvalidTag = 8,
    NotFound = 9,
    CapacityExceeded = 10,
    Unsupported = 11,
    Panic = 12,
    InvalidHandle = 13,
    OperationFailed = 14,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    L2 = 0,
    InnerProduct = 1,
    Cosine = 2,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TagType {
    U32 = 0,
    U64 = 1,
    U128 = 2,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeleteMethod {
    OneHop = 0,
    TwoHopAndOneHop = 1,
    VisitedAndTopK = 2,
}

#[repr(C)]
#[derive(Debug)]
pub struct DiskANNResult {
    pub error: DiskANNError,
    pub error_message: *mut c_char,
    pub handle: *mut c_void,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct IndexConfiguration {
    pub dist_metric: u32,
    pub dim: usize,
    pub search_list_size: u32,
    pub num_threads: u32,
    pub index_path: *const c_char,
    pub tag_type: u32,
    pub max_insert_percentage: f32,
    pub build_search_list_size: u32,
    pub graph_degree: u32,
    pub consolidate_enabled: u8,
    pub consolidate_threshold: f32,
    pub consolidate_threads: u32,
    pub data_path: *const c_char,
    pub tag_path: *const c_char,
    pub is_streaming: u8,
    pub delete_method: u32,
    pub delete_num_to_replace: u32,
    pub delete_search_k: u32,
    pub delete_search_l: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SearchParams {
    pub k: u32,
    pub search_list_size: u32,
    pub beam_width: u32,
}

#[repr(C)]
#[derive(Debug)]
pub struct SearchResult {
    pub indices: *mut u32,
    pub distances: *mut f32,
    pub result_count: usize,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ByteSlice {
    pub ptr: *const u8,
    pub len: usize,
}

#[repr(C)]
#[derive(Debug)]
pub struct DiskANNStatus {
    pub error: DiskANNError,
    pub error_message: *mut c_char,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct TableStats {
    pub tag_memory_bytes: usize,
    pub active_count: usize,
    pub insert_count: u64,
    pub delete_count: u64,
}

type LegacyProvider = FullPrecisionProvider<u8, NoStore>;
type LegacyIndex = DiskANNIndex<LegacyProvider>;
type StreamingProvider<T> = Provider<Full<u8>, T>;
type StreamingDiskIndex<T> = DiskANNIndex<StreamingProvider<T>>;

#[derive(Debug)]
struct Failure {
    code: DiskANNError,
    message: String,
}

impl Failure {
    fn new(code: DiskANNError, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

type Result<T> = std::result::Result<T, Failure>;

struct StreamingIndex<T: StreamingTag> {
    index: StreamingDiskIndex<T>,
    context: Context,
    strategy: Strategy,
    dim: usize,
    search_list_size: usize,
    frozen_internal_id: u32,
    delete_method: InplaceDeleteMethod,
    delete_num_to_replace: usize,
    capacity: usize,
    pending_delete_count: AtomicUsize,
    consolidate_enabled: bool,
    consolidate_threshold: f32,
    insert_count: AtomicU64,
    delete_count: AtomicU64,
    search_count: AtomicU64,
    consolidation_count: AtomicU64,
    snapshot_gate: RwLock<()>,
}

enum IndexHandle {
    Memory(LegacyIndex),
    StreamingU32(StreamingIndex<u32>),
    StreamingU64(StreamingIndex<u64>),
    StreamingU128(StreamingIndex<Tag128>),
}

impl IndexHandle {
    fn memory(&self) -> Result<&LegacyIndex> {
        match self {
            Self::Memory(index) => Ok(index),
            Self::StreamingU32(_) | Self::StreamingU64(_) | Self::StreamingU128(_) => {
                Err(Failure::new(
                    DiskANNError::InvalidHandle,
                    "streaming handle cannot be used for a memory-index operation",
                ))
            }
        }
    }

    fn kind(&self) -> HandleKind {
        match self {
            Self::Memory(_) => HandleKind::Memory,
            Self::StreamingU32(_) | Self::StreamingU64(_) | Self::StreamingU128(_) => {
                HandleKind::Streaming
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum HandleKind {
    Memory,
    Streaming,
}

static NEXT_HANDLE: AtomicUsize = AtomicUsize::new(1);
static HANDLES: OnceLock<Mutex<HashMap<usize, Arc<IndexHandle>>>> = OnceLock::new();
static ERROR_MESSAGES: OnceLock<Mutex<HashSet<usize>>> = OnceLock::new();

fn handles() -> &'static Mutex<HashMap<usize, Arc<IndexHandle>>> {
    HANDLES.get_or_init(|| Mutex::new(HashMap::new()))
}

fn error_messages() -> &'static Mutex<HashSet<usize>> {
    ERROR_MESSAGES.get_or_init(|| Mutex::new(HashSet::new()))
}

fn next_handle() -> Result<usize> {
    let handle = NEXT_HANDLE.fetch_add(1, Ordering::Relaxed);
    if handle == 0 || handle == usize::MAX {
        Err(Failure::new(
            DiskANNError::OperationFailed,
            "handle space exhausted",
        ))
    } else {
        Ok(handle)
    }
}

fn handle_ptr(handle: usize) -> *mut c_void {
    handle as *mut c_void
}

fn handle_key(handle: *mut c_void) -> Result<usize> {
    if handle.is_null() {
        Err(Failure::new(DiskANNError::NullPointer, "handle is null"))
    } else {
        Ok(handle as usize)
    }
}

fn insert_handle(index: IndexHandle) -> Result<*mut c_void> {
    let handle = next_handle()?;
    lock_registry(handles())?.insert(handle, Arc::new(index));
    Ok(handle_ptr(handle))
}

fn get_handle(handle: *mut c_void) -> Result<Arc<IndexHandle>> {
    let key = handle_key(handle)?;
    lock_registry(handles())?
        .get(&key)
        .cloned()
        .ok_or_else(|| Failure::new(DiskANNError::InvalidHandle, "invalid or stale index handle"))
}

fn remove_handle(handle: *mut c_void, expected: Option<HandleKind>) -> Result<()> {
    let key = handle_key(handle)?;
    if expected.is_none() {
        return lock_registry(handles())?
            .remove(&key)
            .map(|_| ())
            .ok_or_else(|| {
                Failure::new(DiskANNError::InvalidHandle, "invalid or stale index handle")
            });
    }

    let entry = lock_registry(handles())?
        .get(&key)
        .cloned()
        .ok_or_else(|| {
            Failure::new(DiskANNError::InvalidHandle, "invalid or stale index handle")
        })?;
    if Some(entry.kind()) != expected {
        return Err(Failure::new(
            DiskANNError::InvalidHandle,
            "index handle type does not match the free operation",
        ));
    }

    let mut registry = lock_registry(handles())?;
    if !registry
        .get(&key)
        .is_some_and(|current| Arc::ptr_eq(current, &entry))
    {
        return Err(Failure::new(
            DiskANNError::InvalidHandle,
            "invalid or stale index handle",
        ));
    }
    registry.remove(&key);
    Ok(())
}

fn lock_registry<T>(mutex: &Mutex<T>) -> Result<std::sync::MutexGuard<'_, T>> {
    mutex.lock().map_err(|_| {
        Failure::new(
            DiskANNError::OperationFailed,
            "internal synchronization lock is poisoned",
        )
    })
}

fn metric(value: u32) -> Result<VectorMetric> {
    match value {
        0 => Ok(VectorMetric::L2),
        1 => Ok(VectorMetric::InnerProduct),
        2 => Ok(VectorMetric::Cosine),
        _ => Err(Failure::new(
            DiskANNError::InvalidConfig,
            "unknown distance metric",
        )),
    }
}

fn tag_width(value: u32) -> Result<usize> {
    match value {
        0 => Ok(4),
        1 => Ok(8),
        2 => Ok(16),
        _ => Err(Failure::new(
            DiskANNError::InvalidConfig,
            "unknown tag type",
        )),
    }
}

fn success_result(handle: *mut c_void) -> DiskANNResult {
    DiskANNResult {
        error: DiskANNError::None,
        error_message: ptr::null_mut(),
        handle,
    }
}

fn legacy_failure(code: DiskANNError) -> DiskANNResult {
    DiskANNResult {
        error: code,
        error_message: ptr::null_mut(),
        handle: ptr::null_mut(),
    }
}

fn success_status() -> DiskANNStatus {
    DiskANNStatus {
        error: DiskANNError::None,
        error_message: ptr::null_mut(),
    }
}

fn owned_message(message: &str) -> *mut c_char {
    let sanitized = message.replace('\0', "\\0");
    let value = match CString::new(sanitized) {
        Ok(value) => value,
        Err(_) => return ptr::null_mut(),
    };
    let raw = value.into_raw();
    if let Ok(mut messages) = error_messages().lock() {
        messages.insert(raw as usize);
        raw
    } else {
        // SAFETY: `raw` was produced by `CString::into_raw` immediately above.
        unsafe { drop(CString::from_raw(raw)) };
        ptr::null_mut()
    }
}

fn failure_status(failure: Failure) -> DiskANNStatus {
    DiskANNStatus {
        error: failure.code,
        error_message: owned_message(&failure.message),
    }
}

fn panic_status() -> DiskANNStatus {
    failure_status(Failure::new(
        DiskANNError::Panic,
        "panic caught at the FFI boundary",
    ))
}

fn ffi_status(f: impl FnOnce() -> Result<()> + std::panic::UnwindSafe) -> DiskANNStatus {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => success_status(),
        Ok(Err(failure)) => failure_status(failure),
        Err(_) => panic_status(),
    }
}

fn validate_len(len: usize) -> Result<()> {
    if len > isize::MAX as usize {
        Err(Failure::new(
            DiskANNError::InvalidBuffer,
            "buffer length exceeds isize::MAX",
        ))
    } else {
        Ok(())
    }
}

fn buffer_range<T>(value: *const T, len: usize, name: &str) -> Result<Option<(usize, usize)>> {
    let bytes = len
        .checked_mul(std::mem::size_of::<T>())
        .ok_or_else(|| Failure::new(DiskANNError::InvalidBuffer, "buffer size overflow"))?;
    validate_len(bytes)?;
    if len == 0 {
        return Ok(None);
    }
    if value.is_null() {
        return Err(Failure::new(
            DiskANNError::NullPointer,
            format!("{name} is null"),
        ));
    }
    let start = value as usize;
    let end = start
        .checked_add(bytes)
        .ok_or_else(|| Failure::new(DiskANNError::InvalidBuffer, "buffer address overflow"))?;
    Ok(Some((start, end)))
}

fn ensure_disjoint(ranges: &[Option<(usize, usize)>]) -> Result<()> {
    for (position, left) in ranges.iter().enumerate() {
        let Some((left_start, left_end)) = left else {
            continue;
        };
        for (right_start, right_end) in ranges[position + 1..].iter().flatten() {
            if left_start < right_end && right_start < left_end {
                return Err(Failure::new(
                    DiskANNError::InvalidBuffer,
                    "foreign buffers overlap",
                ));
            }
        }
    }
    Ok(())
}

unsafe fn input_slice<'a, T>(value: *const T, len: usize, name: &str) -> Result<&'a [T]> {
    let _ = buffer_range(value, len, name)?;
    if len == 0 {
        return Ok(&[]);
    }
    if value.is_null() {
        return Err(Failure::new(
            DiskANNError::NullPointer,
            format!("{name} is null"),
        ));
    }
    if !(value as usize).is_multiple_of(std::mem::align_of::<T>()) {
        return Err(Failure::new(
            DiskANNError::InvalidBuffer,
            format!("{name} is misaligned"),
        ));
    }
    // SAFETY: The caller contract requires a readable, aligned buffer of `len` elements.
    Ok(unsafe { std::slice::from_raw_parts(value, len) })
}

unsafe fn output_slice<'a, T>(value: *mut T, len: usize, name: &str) -> Result<&'a mut [T]> {
    let _ = buffer_range(value.cast_const(), len, name)?;
    if len == 0 {
        return Ok(&mut []);
    }
    if value.is_null() {
        return Err(Failure::new(
            DiskANNError::NullPointer,
            format!("{name} is null"),
        ));
    }
    if !(value as usize).is_multiple_of(std::mem::align_of::<T>()) {
        return Err(Failure::new(
            DiskANNError::InvalidBuffer,
            format!("{name} is misaligned"),
        ));
    }
    // SAFETY: The caller contract requires a writable, aligned buffer of `len` elements.
    Ok(unsafe { std::slice::from_raw_parts_mut(value, len) })
}

unsafe fn byte_slice<'a>(value: ByteSlice, name: &str) -> Result<&'a [u8]> {
    // SAFETY: The function inherits the foreign buffer contract from its caller.
    unsafe { input_slice(value.ptr, value.len, name) }
}

unsafe fn byte_path(value: ByteSlice, name: &str) -> Result<PathBuf> {
    // SAFETY: The function inherits the foreign buffer contract from its caller.
    let bytes = unsafe { byte_slice(value, name)? };
    if bytes.is_empty() {
        return Err(Failure::new(
            DiskANNError::InvalidPath,
            format!("{name} is empty"),
        ));
    }
    let text = std::str::from_utf8(bytes).map_err(|_| {
        Failure::new(
            DiskANNError::InvalidUtf8,
            format!("{name} is not valid UTF-8"),
        )
    })?;
    Ok(PathBuf::from(text))
}

unsafe fn c_path(value: *const c_char, name: &str) -> Result<PathBuf> {
    if value.is_null() {
        return Err(Failure::new(
            DiskANNError::NullPointer,
            format!("{name} is null"),
        ));
    }
    // SAFETY: The public FFI contract requires `value` to be NUL-terminated and readable.
    let value = unsafe { CStr::from_ptr(value) }.to_str().map_err(|_| {
        Failure::new(
            DiskANNError::InvalidUtf8,
            format!("{name} is not valid UTF-8"),
        )
    })?;
    if value.is_empty() {
        return Err(Failure::new(
            DiskANNError::InvalidPath,
            format!("{name} is empty"),
        ));
    }
    Ok(PathBuf::from(value))
}

unsafe fn label_is_empty(value: ByteSlice) -> Result<()> {
    // SAFETY: The function inherits the foreign buffer contract from its caller.
    let bytes = unsafe { byte_slice(value, "label")? };
    if bytes.is_empty() {
        Ok(())
    } else {
        Err(Failure::new(
            DiskANNError::Unsupported,
            "nonempty labels are unsupported",
        ))
    }
}

fn graph_config(
    metric: VectorMetric,
    graph_degree: usize,
    max_degree: usize,
    build_search_list_size: usize,
) -> Result<diskann::graph::Config> {
    if graph_degree == 0 || max_degree == 0 || build_search_list_size == 0 {
        return Err(Failure::new(
            DiskANNError::InvalidConfig,
            "graph degree and build search list size must be nonzero",
        ));
    }
    GraphConfigBuilder::new(
        graph_degree.min(max_degree),
        MaxDegree::new(max_degree),
        build_search_list_size,
        metric.into(),
    )
    .build()
    .map_err(|error| Failure::new(DiskANNError::InvalidConfig, error.to_string()))
}

fn load_streaming_typed<T: StreamingTag>(
    config: IndexConfiguration,
    metric: VectorMetric,
    graph_path: &Path,
    data_path: &Path,
    tag_path: &Path,
) -> Result<StreamingIndex<T>> {
    let snapshot = load_streaming_snapshot::<T>(
        Full::<u8>::new(config.dim, metric),
        StreamingSnapshotConfig {
            dim: config.dim,
            max_insert_percentage: config.max_insert_percentage,
            graph_degree: config.graph_degree as usize,
        },
        graph_path,
        data_path,
        tag_path,
    )
    .map_err(|error| {
        let code = match &error {
            StreamingSnapshotError::Read { .. } => DiskANNError::InvalidPath,
            StreamingSnapshotError::Invalid(_) => DiskANNError::InvalidBuffer,
            StreamingSnapshotError::Provider(_) => DiskANNError::LoadFailed,
            StreamingSnapshotError::Write { .. } | StreamingSnapshotError::Transaction(_) => {
                DiskANNError::OperationFailed
            }
        };
        Failure::new(code, error.to_string())
    })?;
    let max_degree = snapshot.max_degree;
    let graph_config = graph_config(
        metric,
        config.graph_degree as usize,
        max_degree,
        config.build_search_list_size as usize,
    )?;
    let thread_hint = NonZeroUsize::new(config.num_threads as usize)
        .ok_or_else(|| Failure::new(DiskANNError::InvalidConfig, "num_threads must be nonzero"))?;
    let index =
        DiskANNIndex::new_with_multi_thread_runtime(graph_config, snapshot.provider, thread_hint);
    let delete_method = delete_method(config)?;
    Ok(StreamingIndex {
        index,
        context: Context,
        strategy: Strategy,
        dim: config.dim,
        search_list_size: config.search_list_size as usize,
        frozen_internal_id: snapshot.frozen_internal_id,
        delete_method,
        delete_num_to_replace: config.delete_num_to_replace as usize,
        capacity: snapshot.capacity,
        pending_delete_count: AtomicUsize::new(0),
        consolidate_enabled: config.consolidate_enabled != 0,
        consolidate_threshold: config.consolidate_threshold,
        insert_count: AtomicU64::new(0),
        delete_count: AtomicU64::new(0),
        search_count: AtomicU64::new(0),
        consolidation_count: AtomicU64::new(0),
        snapshot_gate: RwLock::new(()),
    })
}

fn load_streaming_index(config: IndexConfiguration) -> Result<*mut c_void> {
    if config.dim == 0
        || config.search_list_size == 0
        || config.num_threads == 0
        || config.consolidate_threads == 0
        || !config.max_insert_percentage.is_finite()
        || config.max_insert_percentage < 0.0
        || !config.consolidate_threshold.is_finite()
        || !(0.0..=1.0).contains(&config.consolidate_threshold)
        || config.consolidate_enabled > 1
    {
        return Err(Failure::new(
            DiskANNError::InvalidConfig,
            "streaming configuration is invalid",
        ));
    }

    let metric = metric(config.dist_metric)?;
    let _ = tag_width(config.tag_type)?;
    // SAFETY: Path buffers are validated according to the public FFI contract.
    let graph_path = unsafe { c_path(config.index_path, "index_path")? };
    // SAFETY: Path buffers are validated according to the public FFI contract.
    let data_path = unsafe { c_path(config.data_path, "data_path")? };
    // SAFETY: Path buffers are validated according to the public FFI contract.
    let tag_path = unsafe { c_path(config.tag_path, "tag_path")? };

    let handle = match config.tag_type {
        0 => IndexHandle::StreamingU32(load_streaming_typed(
            config,
            metric,
            &graph_path,
            &data_path,
            &tag_path,
        )?),
        1 => IndexHandle::StreamingU64(load_streaming_typed(
            config,
            metric,
            &graph_path,
            &data_path,
            &tag_path,
        )?),
        2 => IndexHandle::StreamingU128(load_streaming_typed(
            config,
            metric,
            &graph_path,
            &data_path,
            &tag_path,
        )?),
        _ => {
            return Err(Failure::new(
                DiskANNError::InvalidConfig,
                "unknown tag type",
            ));
        }
    };
    insert_handle(handle)
}

fn delete_method(config: IndexConfiguration) -> Result<InplaceDeleteMethod> {
    if config.delete_num_to_replace == 0 {
        return Err(Failure::new(
            DiskANNError::InvalidConfig,
            "delete_num_to_replace must be nonzero",
        ));
    }
    match config.delete_method {
        0 => Ok(InplaceDeleteMethod::OneHop),
        1 => Ok(InplaceDeleteMethod::TwoHopAndOneHop),
        2 if config.delete_search_k != 0
            && config.delete_search_l != 0
            && config.delete_search_l >= config.delete_search_k =>
        {
            Ok(InplaceDeleteMethod::VisitedAndTopK {
                k_value: config.delete_search_k as usize,
                l_value: config.delete_search_l as usize,
            })
        }
        2 => Err(Failure::new(
            DiskANNError::InvalidConfig,
            "VisitedAndTopK requires nonzero L >= K",
        )),
        _ => Err(Failure::new(
            DiskANNError::InvalidConfig,
            "unknown delete method",
        )),
    }
}

fn load_memory_index(config: IndexConfiguration) -> Result<*mut c_void> {
    if config.dim == 0 || config.search_list_size == 0 || config.num_threads == 0 {
        return Err(Failure::new(
            DiskANNError::InvalidConfig,
            "legacy configuration is invalid",
        ));
    }
    let metric = metric(config.dist_metric)?;
    // SAFETY: The caller guarantees `index_path` is a readable NUL-terminated string.
    let path = unsafe { c_path(config.index_path, "index_path")? };
    let path = path
        .to_str()
        .ok_or_else(|| Failure::new(DiskANNError::InvalidUtf8, "index path is not valid UTF-8"))?;
    let graph_config = GraphConfigBuilder::new(
        1,
        MaxDegree::same(),
        config.search_list_size as usize,
        metric.into(),
    )
    .build()
    .map_err(|error| Failure::new(DiskANNError::InvalidConfig, error.to_string()))?;
    let provider_config = ProviderIndexConfiguration::new(
        metric,
        config.dim,
        0,
        ONE,
        config.num_threads as usize,
        graph_config,
    );
    let index =
        LegacyIndex::load_with_multi_thread_runtime(&FileStorageProvider, &(path, provider_config))
            .map_err(|_| Failure::new(DiskANNError::LoadFailed, "failed to load index"))?;
    insert_handle(IndexHandle::Memory(index))
}

fn search_list(params: SearchParams, default: usize) -> Result<(usize, usize)> {
    let k = params.k as usize;
    if k == 0 {
        return Err(Failure::new(
            DiskANNError::InvalidConfig,
            "search k must be nonzero",
        ));
    }

    let search_list = if params.search_list_size == 0 {
        default
    } else {
        params.search_list_size as usize
    };
    if search_list < k {
        return Err(Failure::new(
            DiskANNError::InvalidConfig,
            "search list size must be at least k",
        ));
    }
    Ok((k, search_list))
}

fn beam_width(params: SearchParams) -> usize {
    if params.beam_width == 0 {
        1
    } else {
        params.beam_width as usize
    }
}

/// # Safety
/// `index_path` must be a readable NUL-terminated string.
/// `is_streaming` must be 0 for read-only loading or 1 for streaming loading.
#[no_mangle]
pub unsafe extern "C" fn diskann_load_memory_index_u8(config: IndexConfiguration) -> DiskANNResult {
    let streaming = match config.is_streaming {
        0 => false,
        1 => true,
        _ => return legacy_failure(DiskANNError::InvalidConfig),
    };
    let load = || {
        if streaming {
            load_streaming_index(config)
        } else {
            load_memory_index(config)
        }
    };
    match catch_unwind(AssertUnwindSafe(load)) {
        Ok(Ok(handle)) => success_result(handle),
        Ok(Err(failure)) if streaming => DiskANNResult {
            error: failure.code,
            error_message: owned_message(&failure.message),
            handle: ptr::null_mut(),
        },
        Ok(Err(failure)) => legacy_failure(failure.code),
        Err(_) if streaming => DiskANNResult {
            error: DiskANNError::Panic,
            error_message: owned_message("panic caught at the FFI boundary"),
            handle: ptr::null_mut(),
        },
        Err(_) => legacy_failure(DiskANNError::SearchFailed),
    }
}

/// # Safety
/// All pointers must be aligned, valid for their declared lengths, and pairwise non-overlapping.
/// On entry, `result.result_count` must be the capacity of both output buffers in results.
unsafe fn search_memory_index(
    query: *const u8,
    query_len: usize,
    params: SearchParams,
    handle: *mut c_void,
    result: *mut SearchResult,
) -> DiskANNResult {
    match catch_unwind(AssertUnwindSafe(|| -> Result<()> {
        if result.is_null() {
            return Err(Failure::new(DiskANNError::NullPointer, "result is null"));
        }
        if !(result as usize).is_multiple_of(std::mem::align_of::<SearchResult>()) {
            return Err(Failure::new(
                DiskANNError::InvalidBuffer,
                "result is misaligned",
            ));
        }
        // SAFETY: The caller supplies writable storage for one `SearchResult`.
        let result = unsafe { &mut *result };
        let (k, search_list) = search_list(params, params.search_list_size as usize)?;
        let capacity = result.result_count;
        if capacity < k {
            return Err(Failure::new(
                DiskANNError::InvalidBuffer,
                "legacy output capacity is smaller than k",
            ));
        }
        let identifier_bytes = capacity
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or_else(|| {
                Failure::new(DiskANNError::InvalidBuffer, "index output size overflow")
            })?;
        ensure_disjoint(&[
            buffer_range(query, query_len, "query")?,
            buffer_range(
                result.indices.cast_const().cast::<u8>(),
                identifier_bytes,
                "indices",
            )?,
            buffer_range(result.distances.cast_const(), capacity, "distances")?,
        ])?;
        // SAFETY: The caller supplies a readable query buffer.
        let query = unsafe { input_slice(query, query_len, "query")? };
        // SAFETY: The caller supplies writable output buffers for `result_count` entries.
        let indices = unsafe { output_slice(result.indices, capacity, "indices")? };
        // SAFETY: The caller supplies writable output buffers for `result_count` entries.
        let distances = unsafe { output_slice(result.distances, capacity, "distances")? };
        let index = get_handle(handle)?;
        let index = index.memory()?;
        let internal_capacity = k.checked_add(1).ok_or_else(|| {
            Failure::new(DiskANNError::InvalidConfig, "search result size overflow")
        })?;
        let mut internal_ids = vec![0u32; internal_capacity];
        let mut internal_distances = vec![0.0f32; internal_capacity];
        let mut output =
            search_output_buffer::IdDistance::new(&mut internal_ids, &mut internal_distances);
        let internal_search_list = search_list.checked_add(1).ok_or_else(|| {
            Failure::new(DiskANNError::InvalidConfig, "search list size overflow")
        })?;
        let kind = Knn::new(internal_search_list, Some(beam_width(params)))
            .map_err(|error| Failure::new(DiskANNError::InvalidConfig, error.to_string()))?;
        let stats = index
            .search(
                kind,
                &LegacyStrategy,
                &diskann::provider::DefaultContext,
                query,
                &mut output,
            )
            .map_err(|_| Failure::new(DiskANNError::SearchFailed, "search failed"))?;
        let count = (stats.result_count as usize).min(k);
        indices[..count].copy_from_slice(&internal_ids[..count]);
        distances[..count].copy_from_slice(&internal_distances[..count]);
        result.result_count = count;
        Ok(())
    })) {
        Ok(Ok(())) => success_result(handle),
        Ok(Err(failure)) => legacy_failure(failure.code),
        Err(_) => legacy_failure(DiskANNError::SearchFailed),
    }
}

/// # Safety
/// All pointers must be aligned, valid for their declared lengths, and pairwise non-overlapping.
/// On entry, `result.result_count` must be the capacity of both output buffers in results.
/// Streaming handles require `result_count * tag_width` writable bytes behind `indices`.
/// Streaming failures may return an owned message released by `diskann_free_error_message`.
#[no_mangle]
pub unsafe extern "C" fn diskann_search_memory_index_u8(
    query: *const u8,
    query_len: usize,
    params: SearchParams,
    handle: *mut c_void,
    result: *mut SearchResult,
) -> DiskANNResult {
    let kind = match catch_unwind(AssertUnwindSafe(|| -> Result<HandleKind> {
        let index = get_handle(handle)?;
        Ok(index.kind())
    })) {
        Ok(Ok(kind)) => kind,
        Ok(Err(failure)) => return legacy_failure(failure.code),
        Err(_) => return legacy_failure(DiskANNError::SearchFailed),
    };

    match kind {
        HandleKind::Memory => {
            // SAFETY: The unified export inherits the original memory-search contract.
            unsafe { search_memory_index(query, query_len, params, handle, result) }
        }
        HandleKind::Streaming => {
            // SAFETY: The unified export requires streaming callers to provide tag-width storage
            // behind `indices`, as documented by the public contract.
            let status =
                unsafe { search_streaming_index(query, query_len, params, handle, result) };
            DiskANNResult {
                error: status.error,
                error_message: status.error_message,
                handle: if status.error == DiskANNError::None {
                    handle
                } else {
                    ptr::null_mut()
                },
            }
        }
    }
}

/// # Safety
/// `handle` must be null or a token returned by the load function in either mode.
/// Null and stale handles are silently ignored.
#[no_mangle]
pub unsafe extern "C" fn diskann_free_memory_index(handle: *mut c_void) {
    if handle.is_null() {
        return;
    }
    let _ = remove_handle(handle, None);
}

/// # Safety
/// Query and output buffers must be valid, writable where applicable, and non-overlapping.
/// On entry, `result.result_count` must be the capacity of both output buffers in results.
/// The `indices` buffer must contain `result_count * tag_width` writable bytes.
unsafe fn search_streaming_index(
    query: *const u8,
    query_len: usize,
    params: SearchParams,
    handle: *mut c_void,
    result: *mut SearchResult,
) -> DiskANNStatus {
    ffi_status(|| {
        if result.is_null() {
            return Err(Failure::new(DiskANNError::NullPointer, "result is null"));
        }
        if !(result as usize).is_multiple_of(std::mem::align_of::<SearchResult>()) {
            return Err(Failure::new(
                DiskANNError::InvalidBuffer,
                "result is misaligned",
            ));
        }
        // SAFETY: The caller supplies writable storage for one result structure.
        let result = unsafe { &mut *result };
        let handle = get_handle(handle)?;
        match &*handle {
            IndexHandle::StreamingU32(streaming) => {
                // SAFETY: Inherited from the FFI search contract.
                unsafe { search_streaming_typed(streaming, query, query_len, params, result) }
            }
            IndexHandle::StreamingU64(streaming) => {
                // SAFETY: Inherited from the FFI search contract.
                unsafe { search_streaming_typed(streaming, query, query_len, params, result) }
            }
            IndexHandle::StreamingU128(streaming) => {
                // SAFETY: Inherited from the FFI search contract.
                unsafe { search_streaming_typed(streaming, query, query_len, params, result) }
            }
            IndexHandle::Memory(_) => Err(Failure::new(
                DiskANNError::InvalidHandle,
                "memory handle cannot be used for a streaming-index operation",
            )),
        }
    })
}

unsafe fn search_streaming_typed<T: StreamingTag>(
    streaming: &StreamingIndex<T>,
    query: *const u8,
    query_len: usize,
    params: SearchParams,
    result: &mut SearchResult,
) -> Result<()> {
    let (k, search_list) = search_list(params, streaming.search_list_size)?;
    let capacity = result.result_count;
    if capacity < k {
        return Err(Failure::new(
            DiskANNError::InvalidBuffer,
            "streaming output capacity is smaller than k",
        ));
    }
    if query_len != streaming.dim {
        return Err(Failure::new(
            DiskANNError::InvalidBuffer,
            "query length does not match index dimension",
        ));
    }
    let identifier_bytes = capacity
        .checked_mul(T::WIDTH)
        .ok_or_else(|| Failure::new(DiskANNError::InvalidBuffer, "tag output size overflow"))?;
    ensure_disjoint(&[
        buffer_range(query, query_len, "query")?,
        buffer_range(
            result.indices.cast_const().cast::<u8>(),
            identifier_bytes,
            "indices",
        )?,
        buffer_range(result.distances.cast_const(), capacity, "distances")?,
    ])?;
    // SAFETY: The caller supplies a readable query buffer.
    let query = unsafe { input_slice(query, query_len, "query")? };
    // SAFETY: The caller supplies a writable tag buffer.
    let tags = unsafe { output_slice(result.indices.cast::<u8>(), identifier_bytes, "indices")? };
    // SAFETY: The caller supplies a writable distance buffer.
    let distances = unsafe { output_slice(result.distances, capacity, "distances")? };
    let internal_capacity = k
        .checked_add(1)
        .ok_or_else(|| Failure::new(DiskANNError::InvalidConfig, "search result size overflow"))?;
    let mut ids = vec![T::default(); internal_capacity];
    let mut internal_distances = vec![0.0f32; internal_capacity];
    let mut output = search_output_buffer::IdDistance::new(&mut ids, &mut internal_distances);
    let internal_search_list = search_list
        .checked_add(1)
        .ok_or_else(|| Failure::new(DiskANNError::InvalidConfig, "search list size overflow"))?;
    let kind = Knn::new(internal_search_list, Some(beam_width(params)))
        .map_err(|error| Failure::new(DiskANNError::InvalidConfig, error.to_string()))?;
    let stats = streaming
        .index
        .search(
            kind,
            &streaming.strategy,
            &streaming.context,
            query,
            &mut output,
        )
        .map_err(|error| Failure::new(DiskANNError::SearchFailed, error.to_string()))?;
    let count = (stats.result_count as usize).min(k);
    for (position, tag) in ids.into_iter().take(count).enumerate() {
        let start = position * T::WIDTH;
        tag.write_le_bytes(&mut tags[start..start + T::WIDTH]);
    }
    distances[..count].copy_from_slice(&internal_distances[..count]);
    result.result_count = count;
    streaming.search_count.fetch_add(1, Ordering::Relaxed);
    Ok(())
}

/// # Safety
/// Vector and tag buffers must be readable for their declared lengths.
#[no_mangle]
pub unsafe extern "C" fn diskann_insert_streaming_index_u8(
    vector: *const u8,
    vector_len: usize,
    tag: *const u8,
    tag_len: usize,
    label: ByteSlice,
    handle: *mut c_void,
) -> DiskANNStatus {
    ffi_status(|| {
        // SAFETY: The caller supplies a valid label buffer.
        unsafe { label_is_empty(label)? };
        let handle = get_handle(handle)?;
        match &*handle {
            IndexHandle::StreamingU32(streaming) => {
                // SAFETY: Inherited from the FFI insert contract.
                unsafe { insert_streaming_typed(streaming, vector, vector_len, tag, tag_len) }
            }
            IndexHandle::StreamingU64(streaming) => {
                // SAFETY: Inherited from the FFI insert contract.
                unsafe { insert_streaming_typed(streaming, vector, vector_len, tag, tag_len) }
            }
            IndexHandle::StreamingU128(streaming) => {
                // SAFETY: Inherited from the FFI insert contract.
                unsafe { insert_streaming_typed(streaming, vector, vector_len, tag, tag_len) }
            }
            IndexHandle::Memory(_) => Err(Failure::new(
                DiskANNError::InvalidHandle,
                "memory handle cannot be used for a streaming-index operation",
            )),
        }
    })
}

unsafe fn insert_streaming_typed<T: StreamingTag>(
    streaming: &StreamingIndex<T>,
    vector: *const u8,
    vector_len: usize,
    tag: *const u8,
    tag_len: usize,
) -> Result<()> {
    let _gate = snapshot_read(streaming)?;
    if vector_len != streaming.dim {
        return Err(Failure::new(
            DiskANNError::InvalidBuffer,
            "vector length does not match index dimension",
        ));
    }
    if tag_len != T::WIDTH {
        return Err(Failure::new(
            DiskANNError::InvalidTag,
            "tag length does not match configured tag type",
        ));
    }
    if streaming.index.inner.provider().active_count() >= streaming.capacity {
        return Err(Failure::new(
            DiskANNError::CapacityExceeded,
            "streaming index is at insert capacity",
        ));
    }
    // SAFETY: The caller supplies a readable vector buffer.
    let vector = unsafe { input_slice(vector, vector_len, "vector")? };
    // SAFETY: The caller supplies a readable tag buffer.
    let tag = T::from_le_bytes(unsafe { input_slice(tag, tag_len, "tag")? })
        .map_err(|error| Failure::new(DiskANNError::InvalidTag, error.to_string()))?;
    if streaming
        .index
        .inner
        .provider()
        .to_internal_id(&streaming.context, &tag)
        .is_ok()
    {
        return Err(Failure::new(DiskANNError::InvalidTag, "tag already exists"));
    }
    streaming
        .index
        .insert(&streaming.strategy, &streaming.context, &tag, vector)
        .map_err(|error| Failure::new(DiskANNError::OperationFailed, error.to_string()))?;
    streaming.insert_count.fetch_add(1, Ordering::Relaxed);
    Ok(())
}

/// # Safety
/// `value` must point to writable storage for one bool.
#[no_mangle]
pub unsafe extern "C" fn diskann_is_max_insert_streaming_index(
    handle: *mut c_void,
    value: *mut bool,
) -> DiskANNStatus {
    ffi_status(|| {
        let _ = buffer_range(value.cast_const(), 1, "value")?;
        if !(value as usize).is_multiple_of(std::mem::align_of::<bool>()) {
            return Err(Failure::new(
                DiskANNError::InvalidBuffer,
                "value is misaligned",
            ));
        }
        let handle = get_handle(handle)?;
        let is_max = match &*handle {
            IndexHandle::StreamingU32(streaming) => {
                streaming.index.inner.provider().active_count() >= streaming.capacity
            }
            IndexHandle::StreamingU64(streaming) => {
                streaming.index.inner.provider().active_count() >= streaming.capacity
            }
            IndexHandle::StreamingU128(streaming) => {
                streaming.index.inner.provider().active_count() >= streaming.capacity
            }
            IndexHandle::Memory(_) => {
                return Err(Failure::new(
                    DiskANNError::InvalidHandle,
                    "memory handle cannot be used for a streaming-index operation",
                ));
            }
        };
        // SAFETY: The caller supplies writable storage for one bool and alignment was checked.
        unsafe { value.write(is_max) };
        Ok(())
    })
}

/// # Safety
/// Tag bytes must be readable for `tag_len`.
#[no_mangle]
pub unsafe extern "C" fn diskann_delete_streaming_index(
    tag: *const u8,
    tag_len: usize,
    handle: *mut c_void,
) -> DiskANNStatus {
    ffi_status(|| {
        let handle = get_handle(handle)?;
        match &*handle {
            IndexHandle::StreamingU32(streaming) => {
                // SAFETY: Inherited from the FFI delete contract.
                unsafe { delete_streaming_typed(streaming, tag, tag_len) }
            }
            IndexHandle::StreamingU64(streaming) => {
                // SAFETY: Inherited from the FFI delete contract.
                unsafe { delete_streaming_typed(streaming, tag, tag_len) }
            }
            IndexHandle::StreamingU128(streaming) => {
                // SAFETY: Inherited from the FFI delete contract.
                unsafe { delete_streaming_typed(streaming, tag, tag_len) }
            }
            IndexHandle::Memory(_) => Err(Failure::new(
                DiskANNError::InvalidHandle,
                "memory handle cannot be used for a streaming-index operation",
            )),
        }
    })
}

unsafe fn delete_streaming_typed<T: StreamingTag>(
    streaming: &StreamingIndex<T>,
    tag: *const u8,
    tag_len: usize,
) -> Result<()> {
    let _gate = snapshot_read(streaming)?;
    if tag_len != T::WIDTH {
        return Err(Failure::new(
            DiskANNError::InvalidTag,
            "tag length does not match configured tag type",
        ));
    }
    // SAFETY: The caller supplies a readable tag buffer.
    let tag = T::from_le_bytes(unsafe { input_slice(tag, tag_len, "tag")? })
        .map_err(|error| Failure::new(DiskANNError::InvalidTag, error.to_string()))?;
    streaming
        .index
        .inner
        .provider()
        .to_internal_id(&streaming.context, &tag)
        .map_err(|_| Failure::new(DiskANNError::NotFound, "tag was not found"))?;
    streaming
        .index
        .inplace_delete(
            streaming.strategy,
            &streaming.context,
            &tag,
            streaming.delete_num_to_replace,
            streaming.delete_method,
        )
        .map_err(|error| Failure::new(DiskANNError::OperationFailed, error.to_string()))?;
    streaming
        .pending_delete_count
        .fetch_add(1, Ordering::Relaxed);
    streaming.delete_count.fetch_add(1, Ordering::Relaxed);
    Ok(())
}

/// # Safety
/// `value` must point to writable storage for one bool.
#[no_mangle]
pub unsafe extern "C" fn diskann_should_consolidate_delete_streaming_index(
    handle: *mut c_void,
    value: *mut bool,
) -> DiskANNStatus {
    ffi_status(|| {
        let _ = buffer_range(value.cast_const(), 1, "value")?;
        if !(value as usize).is_multiple_of(std::mem::align_of::<bool>()) {
            return Err(Failure::new(
                DiskANNError::InvalidBuffer,
                "value is misaligned",
            ));
        }
        let handle = get_handle(handle)?;
        let should = match &*handle {
            IndexHandle::StreamingU32(streaming) => should_consolidate(streaming),
            IndexHandle::StreamingU64(streaming) => should_consolidate(streaming),
            IndexHandle::StreamingU128(streaming) => should_consolidate(streaming),
            IndexHandle::Memory(_) => {
                return Err(Failure::new(
                    DiskANNError::InvalidHandle,
                    "memory handle cannot be used for a streaming-index operation",
                ));
            }
        };
        // SAFETY: The caller supplies writable storage for one bool and alignment was checked.
        unsafe { value.write(should) };
        Ok(())
    })
}

fn should_consolidate<T: StreamingTag>(streaming: &StreamingIndex<T>) -> bool {
    let pending = streaming.pending_delete_count.load(Ordering::Relaxed);
    let ratio = if streaming.capacity == 0 {
        0.0
    } else {
        pending as f32 / streaming.capacity as f32
    };
    streaming.consolidate_enabled && pending != 0 && ratio >= streaming.consolidate_threshold
}

/// # Safety
/// `handle` must be a live streaming token.
#[no_mangle]
pub unsafe extern "C" fn diskann_consolidate_delete_streaming_index(
    handle: *mut c_void,
) -> DiskANNStatus {
    ffi_status(|| {
        let handle = get_handle(handle)?;
        match &*handle {
            IndexHandle::StreamingU32(streaming) => consolidate_streaming(streaming),
            IndexHandle::StreamingU64(streaming) => consolidate_streaming(streaming),
            IndexHandle::StreamingU128(streaming) => consolidate_streaming(streaming),
            IndexHandle::Memory(_) => Err(Failure::new(
                DiskANNError::InvalidHandle,
                "memory handle cannot be used for a streaming-index operation",
            )),
        }
    })
}

fn consolidate_streaming<T: StreamingTag>(streaming: &StreamingIndex<T>) -> Result<()> {
    let _gate = snapshot_read(streaming)?;
    if !streaming.consolidate_enabled {
        return Err(Failure::new(
            DiskANNError::Unsupported,
            "delete consolidation is disabled",
        ));
    }
    let claimed = streaming.pending_delete_count.swap(0, Ordering::AcqRel);
    let work = (|| {
        streaming
            .index
            .consolidate_vector(
                &streaming.strategy,
                &streaming.context,
                streaming.frozen_internal_id,
            )
            .map_err(|error| Failure::new(DiskANNError::OperationFailed, error.to_string()))?;
        let external_ids = streaming.index.inner.provider().external_ids();
        for external_id in external_ids {
            let internal_id = streaming
                .index
                .inner
                .provider()
                .to_internal_id(&streaming.context, &external_id)
                .map_err(|error| Failure::new(DiskANNError::OperationFailed, error.to_string()))?;
            streaming
                .index
                .consolidate_vector(&streaming.strategy, &streaming.context, internal_id)
                .map_err(|error| Failure::new(DiskANNError::OperationFailed, error.to_string()))?;
        }
        Ok(())
    })();
    match work {
        Ok(()) => {
            streaming
                .consolidation_count
                .fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
        Err(error) => {
            streaming
                .pending_delete_count
                .fetch_add(claimed, Ordering::Release);
            Err(error)
        }
    }
}

/// # Safety
/// Paths must be readable for their declared lengths.
/// Output paths must be distinct. Capture failures publish no partial snapshot files.
#[no_mangle]
pub unsafe extern "C" fn diskann_dump_streaming_index(
    handle: *mut c_void,
    index_path: ByteSlice,
    data_path: ByteSlice,
    tag_path: ByteSlice,
) -> DiskANNStatus {
    ffi_status(|| {
        let handle = get_handle(handle)?;
        // SAFETY: The caller supplies valid path buffers.
        let index_path = unsafe { byte_path(index_path, "index_path")? };
        // SAFETY: The caller supplies valid path buffers.
        let data_path = unsafe { byte_path(data_path, "data_path")? };
        // SAFETY: The caller supplies valid path buffers.
        let tag_path = unsafe { byte_path(tag_path, "tag_path")? };
        match &*handle {
            IndexHandle::StreamingU32(streaming) => {
                dump_streaming(streaming, &index_path, &data_path, &tag_path)
            }
            IndexHandle::StreamingU64(streaming) => {
                dump_streaming(streaming, &index_path, &data_path, &tag_path)
            }
            IndexHandle::StreamingU128(streaming) => {
                dump_streaming(streaming, &index_path, &data_path, &tag_path)
            }
            IndexHandle::Memory(_) => {
                return Err(Failure::new(
                    DiskANNError::InvalidHandle,
                    "memory handle cannot be used for a streaming-index operation",
                ));
            }
        }
    })
}

fn snapshot_read<T: StreamingTag>(
    streaming: &StreamingIndex<T>,
) -> Result<RwLockReadGuard<'_, ()>> {
    streaming.snapshot_gate.read().map_err(|_| {
        Failure::new(
            DiskANNError::OperationFailed,
            "snapshot coordination lock is poisoned",
        )
    })
}

fn snapshot_write<T: StreamingTag>(
    streaming: &StreamingIndex<T>,
) -> Result<RwLockWriteGuard<'_, ()>> {
    streaming.snapshot_gate.write().map_err(|_| {
        Failure::new(
            DiskANNError::OperationFailed,
            "snapshot coordination lock is poisoned",
        )
    })
}

fn dump_streaming<T: StreamingTag>(
    streaming: &StreamingIndex<T>,
    index_path: &Path,
    data_path: &Path,
    tag_path: &Path,
) -> Result<()> {
    let _gate = snapshot_write(streaming)?;
    save_streaming_snapshot(
        streaming.index.inner.provider(),
        index_path,
        data_path,
        tag_path,
    )
    .map_err(|error| Failure::new(DiskANNError::OperationFailed, error.to_string()))
}

/// # Safety
/// `stats` must point to writable storage.
#[no_mangle]
pub unsafe extern "C" fn diskann_get_table_stats(
    handle: *mut c_void,
    stats: *mut TableStats,
) -> DiskANNStatus {
    ffi_status(|| {
        let _ = buffer_range(stats.cast_const(), 1, "stats")?;
        if !(stats as usize).is_multiple_of(std::mem::align_of::<TableStats>()) {
            return Err(Failure::new(
                DiskANNError::InvalidBuffer,
                "stats is misaligned",
            ));
        }
        let handle = get_handle(handle)?;
        let value = match &*handle {
            IndexHandle::Memory(_) => TableStats::default(),
            IndexHandle::StreamingU32(streaming) => table_stats(streaming),
            IndexHandle::StreamingU64(streaming) => table_stats(streaming),
            IndexHandle::StreamingU128(streaming) => table_stats(streaming),
        };
        // SAFETY: The caller supplies writable storage and alignment was checked.
        unsafe { stats.write(value) };
        Ok(())
    })
}

fn table_stats<T: StreamingTag>(streaming: &StreamingIndex<T>) -> TableStats {
    TableStats {
        tag_memory_bytes: streaming.index.inner.provider().external_id_memory_bytes(),
        active_count: streaming.index.inner.provider().active_count(),
        insert_count: streaming.insert_count.load(Ordering::Relaxed),
        delete_count: streaming.delete_count.load(Ordering::Relaxed),
    }
}

/// # Safety
/// `handle` must be a live streaming token. Memory, stale, and null handles return
/// `InvalidHandle`/`NullPointer` and are not removed.
#[no_mangle]
pub unsafe extern "C" fn diskann_free_streaming_index(handle: *mut c_void) -> DiskANNStatus {
    ffi_status(|| remove_handle(handle, Some(HandleKind::Streaming)))
}

/// # Safety
/// Message must be null or returned by this library and not previously freed.
#[no_mangle]
pub unsafe extern "C" fn diskann_free_error_message(value: *mut c_char) {
    if value.is_null() {
        return;
    }
    let owned = error_messages()
        .lock()
        .is_ok_and(|mut messages| messages.remove(&(value as usize)));
    if owned {
        // SAFETY: Membership in `ERROR_MESSAGES` proves this pointer came from
        // `CString::into_raw` and has not previously been freed.
        unsafe { drop(CString::from_raw(value)) };
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::unwrap_used)]

    use super::*;
    use diskann_providers::model::graph::provider::async_::{
        common::NoDeletes,
        inmem::{CreateFullPrecision, DefaultProviderParameters},
    };
    use std::{fs, io::Write};

    fn assert_status(status: DiskANNStatus, expected: DiskANNError) {
        assert_eq!(status.error, expected);
        if expected == DiskANNError::None {
            assert!(status.error_message.is_null());
        } else {
            assert!(!status.error_message.is_null());
        }
        // SAFETY: Null is accepted and non-null status messages are library-owned.
        unsafe { diskann_free_error_message(status.error_message) };
    }

    fn assert_result(result: DiskANNResult, expected: DiskANNError, handle: *mut c_void) {
        assert_eq!(result.error, expected);
        if expected == DiskANNError::None {
            assert!(result.error_message.is_null());
            assert_eq!(result.handle, handle);
        } else {
            assert!(!result.error_message.is_null());
            assert!(result.handle.is_null());
        }
        // SAFETY: Null is accepted and non-null streaming errors are library-owned.
        unsafe { diskann_free_error_message(result.error_message) };
    }

    fn byte_slice_from_path(path: &Path) -> ByteSlice {
        let value = path.to_str().expect("temporary paths are UTF-8").as_bytes();
        ByteSlice {
            ptr: value.as_ptr(),
            len: value.len(),
        }
    }

    fn c_path(path: &Path) -> CString {
        CString::new(path.to_str().expect("temporary paths are UTF-8")).expect("path has no NUL")
    }

    fn empty_label() -> ByteSlice {
        ByteSlice {
            ptr: ptr::null(),
            len: 0,
        }
    }

    fn write_snapshot(root: &Path, width: usize) -> (PathBuf, PathBuf, PathBuf, Vec<Vec<u8>>) {
        let graph_path = root.join("index");
        let data_path = root.join("index.data");
        let tag_path = root.join("index.tags");
        let vectors = vec![vec![0, 0, 0, 0], vec![20, 20, 20, 20], vec![10, 10, 10, 10]];

        let mut data = fs::File::create(&data_path).expect("create data");
        data.write_all(&(vectors.len() as u32).to_le_bytes())
            .expect("write count");
        data.write_all(&4u32.to_le_bytes()).expect("write dim");
        for vector in &vectors {
            data.write_all(vector).expect("write vector");
        }

        let adjacency = [vec![1u32, 2], vec![0u32, 2], vec![0u32, 1]];
        let file_size = 24
            + adjacency
                .iter()
                .map(|neighbors| 4 + neighbors.len() * 4)
                .sum::<usize>();
        let mut graph = fs::File::create(&graph_path).expect("create graph");
        graph
            .write_all(&(file_size as u64).to_le_bytes())
            .expect("write graph size");
        graph.write_all(&2u32.to_le_bytes()).expect("write degree");
        graph.write_all(&2u32.to_le_bytes()).expect("write start");
        graph.write_all(&1u64.to_le_bytes()).expect("write frozen");
        for neighbors in adjacency {
            graph
                .write_all(&(neighbors.len() as u32).to_le_bytes())
                .expect("write neighbor count");
            for neighbor in neighbors {
                graph
                    .write_all(&neighbor.to_le_bytes())
                    .expect("write neighbor");
            }
        }

        let tags = match width {
            4 => vec![
                0xaabb_ccddu32.to_le_bytes().to_vec(),
                0x1122_3344u32.to_le_bytes().to_vec(),
                vec![0; width],
            ],
            8 => vec![
                0x0000_0001_aabb_ccddu64.to_le_bytes().to_vec(),
                0x0000_0002_aabb_ccddu64.to_le_bytes().to_vec(),
                vec![0; width],
            ],
            16 => {
                let make = |high: u64| {
                    let mut value = Vec::with_capacity(16);
                    value.extend_from_slice(&0x0000_0000_aabb_ccddu64.to_le_bytes());
                    value.extend_from_slice(&high.to_le_bytes());
                    value
                };
                vec![make(1), make(2), vec![0; width]]
            }
            _ => unreachable!("test supports only configured tag widths"),
        };
        let mut tag_file = fs::File::create(&tag_path).expect("create tags");
        tag_file
            .write_all(&(tags.len() as u32).to_le_bytes())
            .expect("write tag count");
        tag_file
            .write_all(&1u32.to_le_bytes())
            .expect("write tag dim");
        for tag in &tags {
            tag_file.write_all(tag).expect("write tag");
        }
        (graph_path, data_path, tag_path, tags)
    }

    fn load_test_streaming(
        root: &Path,
        tag_type: u32,
        width: usize,
    ) -> (*mut c_void, Vec<Vec<u8>>) {
        let (graph_path, data_path, tag_path, tags) = write_snapshot(root, width);
        let graph_path = c_path(&graph_path);
        let data_path = c_path(&data_path);
        let tag_path = c_path(&tag_path);
        let config = IndexConfiguration {
            dist_metric: Metric::L2 as u32,
            dim: 4,
            search_list_size: 8,
            num_threads: 1,
            index_path: graph_path.as_ptr(),
            tag_type,
            max_insert_percentage: 100.0,
            build_search_list_size: 8,
            graph_degree: 2,
            consolidate_enabled: 1,
            consolidate_threshold: 0.2,
            consolidate_threads: 1,
            data_path: data_path.as_ptr(),
            tag_path: tag_path.as_ptr(),
            is_streaming: 1,
            delete_method: DeleteMethod::OneHop as u32,
            delete_num_to_replace: 3,
            delete_search_k: 10,
            delete_search_l: 64,
        };
        // SAFETY: Every path buffer remains live for the duration of the call.
        let result = unsafe { diskann_load_memory_index_u8(config) };
        assert_eq!(result.error, DiskANNError::None);
        assert!(result.error_message.is_null());
        assert!(!result.handle.is_null());
        (result.handle, tags)
    }

    fn create_test_memory_handle() -> *mut c_void {
        let parameters = DefaultProviderParameters::simple(1, 4, VectorMetric::L2, 2);
        let provider = LegacyProvider::new_empty(
            parameters,
            CreateFullPrecision::<u8>::new(4, None),
            NoStore,
            NoDeletes,
        )
        .expect("create memory provider");
        let config = graph_config(VectorMetric::L2, 1, 2, 2).expect("create graph config");
        let index = LegacyIndex::new_with_current_thread_runtime(config, provider);
        insert_handle(IndexHandle::Memory(index)).expect("register memory handle")
    }

    #[test]
    fn legacy_error_values_are_stable() {
        assert_eq!(DiskANNError::None as i32, 0);
        assert_eq!(DiskANNError::NullPointer as i32, 1);
        assert_eq!(DiskANNError::InvalidPath as i32, 2);
        assert_eq!(DiskANNError::InvalidUtf8 as i32, 3);
        assert_eq!(DiskANNError::InvalidBuffer as i32, 4);
        assert_eq!(DiskANNError::LoadFailed as i32, 5);
        assert_eq!(DiskANNError::SearchFailed as i32, 6);
    }

    #[test]
    fn invalid_foreign_enums_are_rejected() {
        assert_eq!(
            metric(u32::MAX).unwrap_err().code,
            DiskANNError::InvalidConfig
        );
        assert_eq!(
            tag_width(u32::MAX).unwrap_err().code,
            DiskANNError::InvalidConfig
        );
    }

    #[test]
    fn invalid_load_mode_is_rejected() {
        let config = IndexConfiguration {
            dist_metric: Metric::L2 as u32,
            dim: 4,
            search_list_size: 8,
            num_threads: 1,
            index_path: c"unused".as_ptr(),
            tag_type: TagType::U32 as u32,
            max_insert_percentage: 0.0,
            build_search_list_size: 0,
            graph_degree: 0,
            consolidate_enabled: 0,
            consolidate_threshold: 0.0,
            consolidate_threads: 0,
            data_path: ptr::null(),
            tag_path: ptr::null(),
            is_streaming: 2,
            delete_method: DeleteMethod::OneHop as u32,
            delete_num_to_replace: 3,
            delete_search_k: 0,
            delete_search_l: 0,
        };
        // SAFETY: The invalid mode is rejected before any path is accessed.
        let result = unsafe { diskann_load_memory_index_u8(config) };
        assert_eq!(result.error, DiskANNError::InvalidConfig);
        assert!(result.error_message.is_null());
        assert!(result.handle.is_null());
    }

    #[test]
    fn delete_method_configuration_is_validated() {
        let base = IndexConfiguration {
            dist_metric: Metric::L2 as u32,
            dim: 4,
            search_list_size: 8,
            num_threads: 1,
            index_path: ptr::null(),
            tag_type: TagType::U32 as u32,
            max_insert_percentage: 1.0,
            build_search_list_size: 8,
            graph_degree: 2,
            consolidate_enabled: 1,
            consolidate_threshold: 0.1,
            consolidate_threads: 1,
            data_path: ptr::null(),
            tag_path: ptr::null(),
            is_streaming: 1,
            delete_method: DeleteMethod::OneHop as u32,
            delete_num_to_replace: 3,
            delete_search_k: 0,
            delete_search_l: 0,
        };
        assert!(matches!(
            delete_method(base).unwrap(),
            InplaceDeleteMethod::OneHop
        ));
        assert!(matches!(
            delete_method(IndexConfiguration {
                delete_method: DeleteMethod::TwoHopAndOneHop as u32,
                ..base
            })
            .unwrap(),
            InplaceDeleteMethod::TwoHopAndOneHop
        ));
        assert!(matches!(
            delete_method(IndexConfiguration {
                delete_method: DeleteMethod::VisitedAndTopK as u32,
                delete_search_k: 10,
                delete_search_l: 64,
                ..base
            })
            .unwrap(),
            InplaceDeleteMethod::VisitedAndTopK {
                k_value: 10,
                l_value: 64
            }
        ));
        assert!(delete_method(IndexConfiguration {
            delete_num_to_replace: 0,
            ..base
        })
        .is_err());
        assert!(delete_method(IndexConfiguration {
            delete_method: DeleteMethod::VisitedAndTopK as u32,
            delete_search_k: 10,
            delete_search_l: 9,
            ..base
        })
        .is_err());
        assert!(delete_method(IndexConfiguration {
            delete_method: 3,
            ..base
        })
        .is_err());
    }

    #[test]
    fn empty_and_nonempty_labels_are_distinguished() {
        // SAFETY: A zero-length null buffer is explicitly accepted.
        unsafe {
            label_is_empty(ByteSlice {
                ptr: ptr::null(),
                len: 0,
            })
            .unwrap();
        }
        let label = b"x";
        // SAFETY: `label` is readable for one byte.
        let error = unsafe {
            label_is_empty(ByteSlice {
                ptr: label.as_ptr(),
                len: label.len(),
            })
            .unwrap_err()
        };
        assert_eq!(error.code, DiskANNError::Unsupported);
    }

    #[test]
    fn null_pointer_validation_is_stable() {
        // SAFETY: The null pointer is rejected before creating a slice.
        let error = unsafe { input_slice::<u8>(ptr::null(), 1, "test").unwrap_err() };
        assert_eq!(error.code, DiskANNError::NullPointer);
    }

    #[test]
    fn legacy_search_validates_result_capacity() {
        let handle = create_test_memory_handle();
        let query = [0u8; 4];
        let mut indices = [0u32; 2];
        let mut distances = [0.0f32; 2];
        let mut result = SearchResult {
            indices: indices.as_mut_ptr(),
            distances: distances.as_mut_ptr(),
            result_count: 1,
        };
        // SAFETY: Every supplied buffer is valid; `result_count` deliberately reports too little
        // capacity and is rejected before search execution.
        let status = unsafe {
            diskann_search_memory_index_u8(
                query.as_ptr(),
                query.len(),
                SearchParams {
                    k: 2,
                    search_list_size: 2,
                    beam_width: 0,
                },
                handle,
                &mut result,
            )
        };
        assert_eq!(status.error, DiskANNError::InvalidBuffer);
        assert!(status.error_message.is_null());
        // SAFETY: The handle is live and owned by this test.
        unsafe { diskann_free_memory_index(handle) };
    }

    #[test]
    fn common_and_compatibility_frees_preserve_handle_types() {
        // SAFETY: The checked streaming wrapper explicitly rejects null handles.
        assert_status(
            unsafe { diskann_free_streaming_index(ptr::null_mut()) },
            DiskANNError::NullPointer,
        );

        let memory = create_test_memory_handle();

        let mut stats = TableStats {
            tag_memory_bytes: 1,
            active_count: 1,
            insert_count: 1,
            delete_count: 1,
        };
        // SAFETY: A memory handle returns a successful zero/default table status.
        assert_status(
            unsafe { diskann_get_table_stats(memory, &mut stats) },
            DiskANNError::None,
        );
        assert_eq!(stats.tag_memory_bytes, 0);
        assert_eq!(stats.active_count, 0);
        assert_eq!(stats.insert_count, 0);
        assert_eq!(stats.delete_count, 0);

        // SAFETY: A memory handle is deliberately passed to the checked streaming wrapper.
        assert_status(
            unsafe { diskann_free_streaming_index(memory) },
            DiskANNError::InvalidHandle,
        );
        assert!(get_handle(memory).is_ok());

        // SAFETY: The common void API accepts and removes a live memory handle.
        unsafe { diskann_free_memory_index(memory) };
        assert!(get_handle(memory).is_err());
        // SAFETY: Stale and null handles remain silent no-ops.
        unsafe { diskann_free_memory_index(memory) };
        // SAFETY: Null handles are explicitly accepted.
        unsafe { diskann_free_memory_index(ptr::null_mut()) };
    }

    #[test]
    fn registry_free_keeps_inflight_arc_alive() {
        let temp = tempfile::tempdir().expect("create temp directory");
        let (raw, _) = load_test_streaming(temp.path(), TagType::U32 as u32, 4);
        let inflight = get_handle(raw).expect("clone in-flight handle");

        // SAFETY: The registry owns this live handle token.
        unsafe { diskann_free_memory_index(raw) };
        assert!(get_handle(raw).is_err());
        assert!(matches!(
            &*inflight,
            IndexHandle::StreamingU32(streaming)
                if streaming.index.inner.provider().active_count() == 2
        ));
    }

    #[test]
    fn same_handle_search_insert_delete_run_concurrently() {
        let temp = tempfile::tempdir().expect("create temp directory");
        let (raw, snapshot_tags) = load_test_streaming(temp.path(), TagType::U32 as u32, 4);
        let token = raw as usize;

        std::thread::scope(|scope| {
            for _ in 0..4 {
                scope.spawn(|| {
                    let query = [0u8; 4];
                    let mut tags = [0u32; 2];
                    let mut distances = [0.0f32; 2];
                    let mut result = SearchResult {
                        indices: tags.as_mut_ptr(),
                        distances: distances.as_mut_ptr(),
                        result_count: 2,
                    };
                    // SAFETY: Each task owns disjoint output buffers and the shared token remains
                    // registered for the scope.
                    let status = unsafe {
                        diskann_search_memory_index_u8(
                            query.as_ptr(),
                            query.len(),
                            SearchParams {
                                k: 2,
                                search_list_size: 8,
                                beam_width: 0,
                            },
                            token as *mut c_void,
                            &mut result,
                        )
                    };
                    assert_eq!(status.error, DiskANNError::None);
                });
            }
            scope.spawn(|| {
                let vector = [3u8; 4];
                let tag = 0xfeed_beefu32.to_le_bytes();
                // SAFETY: Input buffers remain live for this call and the token is registered.
                let status = unsafe {
                    diskann_insert_streaming_index_u8(
                        vector.as_ptr(),
                        vector.len(),
                        tag.as_ptr(),
                        tag.len(),
                        empty_label(),
                        token as *mut c_void,
                    )
                };
                assert_eq!(status.error, DiskANNError::None);
            });
            scope.spawn(|| {
                let tag = &snapshot_tags[1];
                // SAFETY: The tag remains live for this call and the token is registered.
                let status = unsafe {
                    diskann_delete_streaming_index(tag.as_ptr(), tag.len(), token as *mut c_void)
                };
                assert_eq!(status.error, DiskANNError::None);
            });
        });

        // SAFETY: The handle remains live after all scoped operations finish.
        unsafe { diskann_free_memory_index(raw) };
    }

    #[test]
    #[allow(clippy::panic)]
    fn dump_gate_blocks_updates_and_other_dumps() {
        use std::{
            sync::{mpsc, Barrier},
            time::Duration,
        };

        let temp = tempfile::tempdir().expect("create temp directory");
        let (raw, snapshot_tags) = load_test_streaming(temp.path(), TagType::U32 as u32, 4);
        let handle = get_handle(raw).expect("clone handle");
        let streaming = match &*handle {
            IndexHandle::StreamingU32(streaming) => streaming,
            _ => unreachable!("test loaded u32 streaming index"),
        };
        let gate = snapshot_write(streaming).expect("claim dump gate");
        let barrier = Arc::new(Barrier::new(5));
        let (sender, receiver) = mpsc::channel();
        let token = raw as usize;

        std::thread::scope(|scope| {
            let insert_sender = sender.clone();
            let insert_barrier = Arc::clone(&barrier);
            scope.spawn(move || {
                insert_barrier.wait();
                let vector = [3u8; 4];
                let tag = 0xfeed_beefu32.to_le_bytes();
                // SAFETY: Inputs remain live and the registered handle is kept alive by `handle`.
                let status = unsafe {
                    diskann_insert_streaming_index_u8(
                        vector.as_ptr(),
                        vector.len(),
                        tag.as_ptr(),
                        tag.len(),
                        empty_label(),
                        token as *mut c_void,
                    )
                };
                insert_sender.send(status.error).unwrap();
            });
            let delete_sender = sender.clone();
            let delete_barrier = Arc::clone(&barrier);
            scope.spawn(move || {
                delete_barrier.wait();
                let tag = &snapshot_tags[1];
                // SAFETY: Inputs remain live and the registered handle is kept alive by `handle`.
                let status = unsafe {
                    diskann_delete_streaming_index(tag.as_ptr(), tag.len(), token as *mut c_void)
                };
                delete_sender.send(status.error).unwrap();
            });
            let root = temp.path();
            for suffix in ["first", "second"] {
                let dump_sender = sender.clone();
                let dump_barrier = Arc::clone(&barrier);
                scope.spawn(move || {
                    dump_barrier.wait();
                    let graph = root.join(suffix);
                    let data = root.join(format!("{suffix}.data"));
                    let tags = root.join(format!("{suffix}.tags"));
                    // SAFETY: Paths remain live and the registered handle is kept alive by `handle`.
                    let status = unsafe {
                        diskann_dump_streaming_index(
                            token as *mut c_void,
                            byte_slice_from_path(&graph),
                            byte_slice_from_path(&data),
                            byte_slice_from_path(&tags),
                        )
                    };
                    dump_sender.send(status.error).unwrap();
                });
            }
            barrier.wait();
            assert!(receiver.recv_timeout(Duration::from_millis(20)).is_err());
            drop(gate);
            for _ in 0..4 {
                assert_eq!(
                    receiver.recv_timeout(Duration::from_secs(5)).unwrap(),
                    DiskANNError::None
                );
            }
        });

        // SAFETY: The handle remains live after all scoped operations finish.
        unsafe { diskann_free_memory_index(raw) };
    }

    #[test]
    fn panic_is_converted_to_status() {
        #[allow(clippy::panic)]
        let status = ffi_status(|| -> Result<()> { panic!("test panic") });
        assert_eq!(status.error, DiskANNError::Panic);
        assert!(!status.error_message.is_null());
        // SAFETY: The message was returned by this library.
        unsafe { diskann_free_error_message(status.error_message) };
    }

    #[test]
    fn abi_layout_is_stable_on_64_bit_targets() {
        assert_eq!(std::mem::size_of::<DiskANNResult>(), 24);
        assert_eq!(std::mem::size_of::<IndexConfiguration>(), 104);
        assert_eq!(std::mem::offset_of!(IndexConfiguration, dist_metric), 0);
        assert_eq!(std::mem::offset_of!(IndexConfiguration, dim), 8);
        assert_eq!(
            std::mem::offset_of!(IndexConfiguration, search_list_size),
            16
        );
        assert_eq!(std::mem::offset_of!(IndexConfiguration, num_threads), 20);
        assert_eq!(std::mem::offset_of!(IndexConfiguration, index_path), 24);
        assert_eq!(
            std::mem::offset_of!(IndexConfiguration, consolidate_threshold),
            52
        );
        assert_eq!(std::mem::offset_of!(IndexConfiguration, data_path), 64);
        assert_eq!(std::mem::offset_of!(IndexConfiguration, tag_path), 72);
        assert_eq!(std::mem::offset_of!(IndexConfiguration, is_streaming), 80);
        assert_eq!(std::mem::offset_of!(IndexConfiguration, delete_method), 84);
        assert_eq!(
            std::mem::offset_of!(IndexConfiguration, delete_num_to_replace),
            88
        );
        assert_eq!(
            std::mem::offset_of!(IndexConfiguration, delete_search_k),
            92
        );
        assert_eq!(
            std::mem::offset_of!(IndexConfiguration, delete_search_l),
            96
        );
        assert_eq!(std::mem::size_of::<SearchParams>(), 12);
        assert_eq!(std::mem::offset_of!(SearchParams, k), 0);
        assert_eq!(std::mem::offset_of!(SearchParams, search_list_size), 4);
        assert_eq!(std::mem::offset_of!(SearchParams, beam_width), 8);
        assert_eq!(std::mem::size_of::<SearchResult>(), 24);
        assert_eq!(std::mem::offset_of!(SearchResult, indices), 0);
        assert_eq!(std::mem::offset_of!(SearchResult, distances), 8);
        assert_eq!(std::mem::offset_of!(SearchResult, result_count), 16);
        assert_eq!(std::mem::size_of::<TableStats>(), 32);
        assert_eq!(std::mem::offset_of!(TableStats, tag_memory_bytes), 0);
        assert_eq!(std::mem::offset_of!(TableStats, active_count), 8);
        assert_eq!(std::mem::offset_of!(TableStats, insert_count), 16);
        assert_eq!(std::mem::offset_of!(TableStats, delete_count), 24);
    }

    #[test]
    fn legacy_failures_retain_null_messages() {
        let config = IndexConfiguration {
            dist_metric: u32::MAX,
            dim: 4,
            search_list_size: 8,
            num_threads: 1,
            index_path: c"unused".as_ptr(),
            tag_type: TagType::U32 as u32,
            max_insert_percentage: 0.0,
            build_search_list_size: 0,
            graph_degree: 0,
            consolidate_enabled: 0,
            consolidate_threshold: 0.0,
            consolidate_threads: 0,
            data_path: ptr::null(),
            tag_path: ptr::null(),
            is_streaming: 0,
            delete_method: DeleteMethod::OneHop as u32,
            delete_num_to_replace: 0,
            delete_search_k: 0,
            delete_search_l: 0,
        };
        // SAFETY: `index_path` is a static readable byte slice.
        let result = unsafe { diskann_load_memory_index_u8(config) };
        assert_eq!(result.error, DiskANNError::InvalidConfig);
        assert!(result.error_message.is_null());
        assert!(result.handle.is_null());
    }

    #[test]
    fn streaming_round_trip_all_tag_widths() {
        for (tag_type, width) in [
            (TagType::U32 as u32, 4),
            (TagType::U64 as u32, 8),
            (TagType::U128 as u32, 16),
        ] {
            let temp = tempfile::tempdir().expect("create temp directory");
            let (handle, snapshot_tags) = load_test_streaming(temp.path(), tag_type, width);

            let query = [0u8; 4];
            let mut output_tags = vec![0xff; width * 2];
            let mut distances = vec![f32::MAX; 2];
            let mut result = SearchResult {
                indices: output_tags.as_mut_ptr().cast::<u32>(),
                distances: distances.as_mut_ptr(),
                result_count: 2,
            };
            // SAFETY: All input and output buffers are valid for their declared lengths.
            let search = unsafe {
                diskann_search_memory_index_u8(
                    query.as_ptr(),
                    query.len(),
                    SearchParams {
                        k: 2,
                        search_list_size: 0,
                        beam_width: 0,
                    },
                    handle,
                    &mut result,
                )
            };
            assert_result(search, DiskANNError::None, handle);
            assert_eq!(result.result_count, 2);
            assert_eq!(&output_tags[..width], snapshot_tags[0].as_slice());
            assert_eq!(distances[0], 0.0);

            let inserted_tag = vec![3u8; width];
            let inserted_vector = [1u8; 4];
            // SAFETY: Vector and tag buffers are readable for their declared lengths.
            let status = unsafe {
                diskann_insert_streaming_index_u8(
                    inserted_vector.as_ptr(),
                    inserted_vector.len(),
                    inserted_tag.as_ptr(),
                    inserted_tag.len(),
                    empty_label(),
                    handle,
                )
            };
            assert_status(status, DiskANNError::None);
            // SAFETY: Reusing the same valid tag must be rejected by the provider mapping.
            assert_status(
                unsafe {
                    diskann_insert_streaming_index_u8(
                        inserted_vector.as_ptr(),
                        inserted_vector.len(),
                        inserted_tag.as_ptr(),
                        inserted_tag.len(),
                        empty_label(),
                        handle,
                    )
                },
                DiskANNError::InvalidTag,
            );

            let mut inserted_search_tags = vec![0xff; width * 3];
            let mut inserted_search_distances = vec![f32::MAX; 3];
            let mut inserted_search = SearchResult {
                indices: inserted_search_tags.as_mut_ptr().cast::<u32>(),
                distances: inserted_search_distances.as_mut_ptr(),
                result_count: 3,
            };
            // SAFETY: All input and output buffers are valid for their declared lengths.
            assert_result(
                unsafe {
                    diskann_search_memory_index_u8(
                        inserted_vector.as_ptr(),
                        inserted_vector.len(),
                        SearchParams {
                            k: 3,
                            search_list_size: 8,
                            beam_width: 0,
                        },
                        handle,
                        &mut inserted_search,
                    )
                },
                DiskANNError::None,
                handle,
            );
            assert!(inserted_search_tags[..inserted_search.result_count * width]
                .chunks_exact(width)
                .any(|tag| tag == inserted_tag));

            let fourth_tag = vec![4u8; width];
            let fourth_vector = [2u8; 4];
            // SAFETY: Vector and tag buffers are readable for their declared lengths.
            let status = unsafe {
                diskann_insert_streaming_index_u8(
                    fourth_vector.as_ptr(),
                    fourth_vector.len(),
                    fourth_tag.as_ptr(),
                    fourth_tag.len(),
                    empty_label(),
                    handle,
                )
            };
            assert_status(status, DiskANNError::None);

            let mut is_max = false;
            // SAFETY: `is_max` is writable and the handle is live.
            assert_status(
                unsafe { diskann_is_max_insert_streaming_index(handle, &mut is_max) },
                DiskANNError::None,
            );
            assert!(is_max);

            // SAFETY: The tag buffer is readable and the handle is live.
            assert_status(
                unsafe {
                    diskann_delete_streaming_index(
                        inserted_tag.as_ptr(),
                        inserted_tag.len(),
                        handle,
                    )
                },
                DiskANNError::None,
            );
            let mut should_consolidate = false;
            // SAFETY: The bool output is writable and the handle is live.
            assert_status(
                unsafe {
                    diskann_should_consolidate_delete_streaming_index(
                        handle,
                        &mut should_consolidate,
                    )
                },
                DiskANNError::None,
            );
            assert!(should_consolidate);
            // SAFETY: The handle is live.
            assert_status(
                unsafe { diskann_consolidate_delete_streaming_index(handle) },
                DiskANNError::None,
            );

            inserted_search_tags.fill(0xff);
            inserted_search_distances.fill(f32::MAX);
            inserted_search.result_count = 3;
            // SAFETY: All input and output buffers are valid for their declared lengths.
            assert_result(
                unsafe {
                    diskann_search_memory_index_u8(
                        inserted_vector.as_ptr(),
                        inserted_vector.len(),
                        SearchParams {
                            k: 3,
                            search_list_size: 8,
                            beam_width: 0,
                        },
                        handle,
                        &mut inserted_search,
                    )
                },
                DiskANNError::None,
                handle,
            );
            assert!(inserted_search_tags[..inserted_search.result_count * width]
                .chunks_exact(width)
                .all(|tag| tag != inserted_tag));

            let mut stats = TableStats::default();
            // SAFETY: `stats` is writable and the handle is live.
            assert_status(
                unsafe { diskann_get_table_stats(handle, &mut stats) },
                DiskANNError::None,
            );
            assert_eq!(stats.active_count, 3);
            assert_eq!(stats.insert_count, 2);
            assert_eq!(stats.delete_count, 1);
            assert_eq!(stats.tag_memory_bytes, 4 * width);

            let dump = temp.path().join("dump");
            let data = temp.path().join("dump.data");
            let tags = temp.path().join("dump.tags");
            // SAFETY: Path buffers are readable for each call.
            assert_status(
                unsafe {
                    diskann_dump_streaming_index(
                        handle,
                        byte_slice_from_path(&dump),
                        byte_slice_from_path(&data),
                        byte_slice_from_path(&tags),
                    )
                },
                DiskANNError::None,
            );
            let dump = c_path(&dump);
            let data = c_path(&data);
            let tags = c_path(&tags);
            let reload = IndexConfiguration {
                dist_metric: Metric::L2 as u32,
                dim: 4,
                search_list_size: 8,
                num_threads: 1,
                index_path: dump.as_ptr(),
                tag_type,
                max_insert_percentage: 100.0,
                build_search_list_size: 8,
                graph_degree: 2,
                consolidate_enabled: 1,
                consolidate_threshold: 0.2,
                consolidate_threads: 1,
                data_path: data.as_ptr(),
                tag_path: tags.as_ptr(),
                is_streaming: 1,
                delete_method: DeleteMethod::OneHop as u32,
                delete_num_to_replace: 3,
                delete_search_k: 10,
                delete_search_l: 64,
            };
            // SAFETY: All snapshot paths remain live for the duration of the call.
            let reloaded = unsafe { diskann_load_memory_index_u8(reload) };
            assert_eq!(reloaded.error, DiskANNError::None);
            let mut reloaded_stats = TableStats::default();
            // SAFETY: The stats output is writable and the reloaded handle is live.
            assert_status(
                unsafe { diskann_get_table_stats(reloaded.handle, &mut reloaded_stats) },
                DiskANNError::None,
            );
            assert_eq!(reloaded_stats.active_count, stats.active_count);
            let mut reloaded_tags = vec![0xff; width * 3];
            let mut reloaded_distances = vec![f32::MAX; 3];
            let mut reloaded_result = SearchResult {
                indices: reloaded_tags.as_mut_ptr().cast(),
                distances: reloaded_distances.as_mut_ptr(),
                result_count: 3,
            };
            // SAFETY: Query and output buffers are valid and the reloaded handle is live.
            assert_result(
                unsafe {
                    diskann_search_memory_index_u8(
                        fourth_vector.as_ptr(),
                        fourth_vector.len(),
                        SearchParams {
                            k: 3,
                            search_list_size: 8,
                            beam_width: 0,
                        },
                        reloaded.handle,
                        &mut reloaded_result,
                    )
                },
                DiskANNError::None,
                reloaded.handle,
            );
            assert!(reloaded_tags[..reloaded_result.result_count * width]
                .chunks_exact(width)
                .any(|tag| tag == fourth_tag));
            // SAFETY: The reloaded handle is live.
            unsafe { diskann_free_memory_index(reloaded.handle) };

            let label = [1u8];
            // SAFETY: All buffers are valid; a nonempty label is deliberately unsupported.
            assert_status(
                unsafe {
                    diskann_insert_streaming_index_u8(
                        inserted_vector.as_ptr(),
                        inserted_vector.len(),
                        inserted_tag.as_ptr(),
                        inserted_tag.len(),
                        ByteSlice {
                            ptr: label.as_ptr(),
                            len: label.len(),
                        },
                        handle,
                    )
                },
                DiskANNError::Unsupported,
            );

            // SAFETY: The common void free accepts a live streaming handle.
            unsafe { diskann_free_memory_index(handle) };
            assert!(get_handle(handle).is_err());
            // SAFETY: The typed compatibility wrapper reports the now-stale handle.
            assert_status(
                unsafe { diskann_free_streaming_index(handle) },
                DiskANNError::InvalidHandle,
            );
        }
    }
}
