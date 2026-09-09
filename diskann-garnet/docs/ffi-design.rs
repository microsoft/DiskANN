/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Reference listing of the FFI surface exported to Garnet.
//!
//! This file is not compiled. It mirrors the `#[unsafe(no_mangle)] extern "C"` symbols in
//! `src/lib.rs` and the callback types in `src/garnet.rs`, and must be updated alongside them.

/// Element type of the vector data passed across the FFI. Must match the definition on the
/// C# side.
#[repr(C)]
enum VectorValueType {
    Invalid = 0,
    FP32,
    XB8,
}

/// Quantizer selection for an index. Must match the definition on the C# side.
///
/// `NoQuant`, `Bin`, and `Q8` map to the quantizations Redis exposes and take `f32` vector
/// data. The `X`-prefixed variants are DiskANN extensions taking `u8`/`i8` vector data.
#[repr(C)]
enum VectorQuantType {
    Invalid = 0,
    NoQuant,
    Bin,
    Q8,
    XNoQuantU8,
    XNoQuantI8,
    XBinI8,
    XBinU8,
}

/// Status returned by `insert`, encoded as a `u8`.
///
/// `SuccessStartTraining` signals that the insert crossed the threshold at which the quantizer
/// can be trained, and that Garnet should call `build_quant_table`.
enum InsertResult {
    Fail = 0,
    Success = 1,
    SuccessStartTraining = 2,
}

/// Read one or more keys from Garnet.
///
/// `keys` holds `key_count` keys, each 4-byte length prefixed. `value_length_hint` is the
/// expected size in bytes of a single value, and may be an overestimate.
///
/// For every key that is present, Garnet invokes `read_data` with the index of that key within
/// the batch, the opaque `read_data_state`, and the value bytes. Missing keys are skipped, so
/// a key's existence is determined by whether `read_data` fires for it. Values must be aligned
/// to at least 8 bytes.
type ReadCallback = unsafe extern "C" fn(
    context: u64,
    key_count: u32,
    value_length_hint: u32,
    keys: *const u8,
    keys_len: usize,
    read_data: ReadDataCallback,
    read_data_state: *mut c_void,
);

/// Delivers a single value to the caller of `ReadCallback`.
type ReadDataCallback =
    unsafe extern "C" fn(index: u32, state: *mut c_void, value: *const u8, value_len: usize);

/// Write a value for a key. Returns true on success.
type WriteCallback = unsafe extern "C" fn(
    context: u64,
    key: *const u8,
    key_len: usize,
    value: *const u8,
    value_len: usize,
) -> bool;

/// Delete a key. Returns true on success.
type DeleteCallback = unsafe extern "C" fn(context: u64, key: *const u8, key_len: usize) -> bool;

/// Atomically read, modify, and write the value for a key.
///
/// Garnet invokes `modify` with the opaque `modify_state` and a mutable view of the current
/// value. If the key does not exist, a zero-initialized value of `write_len` bytes is created
/// and passed instead. Returns true on success.
type ReadModifyWriteCallback = unsafe extern "C" fn(
    context: u64,
    key: *const u8,
    key_len: usize,
    write_len: usize,
    modify: RmwDataCallback,
    modify_state: *mut c_void,
) -> bool;

/// Mutates the value in place on behalf of `ReadModifyWriteCallback`.
type RmwDataCallback = unsafe extern "C" fn(state: *mut c_void, value: *mut u8, value_len: usize);

/// Evaluate the filter of the in-flight search against a vector's attribute data.
///
/// `attributes` is null when the vector has no attributes. Returns true if the vector passes.
type FilterCallback =
    unsafe extern "C" fn(context: u64, attributes: *const u8, attributes_len: usize) -> bool;

/// Emit a UTF-8 log message. The `Term` bits of the context indicate which area of the index
/// the message concerns.
type LogCallback = unsafe extern "C" fn(context: u64, message: *const u8, message_len: usize);

/// Create a new empty index
/// Takes the params of VADD (see: https://redis.io/docs/latest/commands/vadd/), maps to a reasonable interpretation
///
/// (context % 4) == 0, xxx_callbacks add 0/1/2/3 depending on data stored
///
/// Expectation is any state necessary to recover an index is stored via read/write callbacks - including quantizers.
///
/// reduce_dim == 0 to indicate no reduction requested. Dimensionality reduction is not
/// implemented, so this parameter is currently ignored.
///
/// metric_type is passed as a raw i32. Valid values are:
/// - 0: Cosine
/// - 1: InnerProduct
/// - 2: L2 (Euclidean distance)
/// - 3: CosineNormalized
///
/// Returns an opaque handle that conceals all the generics, or null on error. The
/// handle must be released with `drop_index`.
///
/// Sets the `quantization_needed` out-param if the index requires Garnet to drive the quantizer
/// lifecycle via `build_quant_table` and `backfill_quant_vectors`. This can be false even when a
/// quantizer is in use, since not every quantizer requires training and backfill.
#[unsafe(no_mangle)]
extern "C" fn create_index(
    context: u64,
    dim: u32,
    reduce_dim: u32,
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
) -> *const c_void;

/// Drop a previously created index
///
/// This is the only valid way to release a handle returned by `create_index`.
///
/// Not called if any other operation against the index may be in flight or started.
#[unsafe(no_mangle)]
extern "C" fn drop_index(context: u64, index_ptr: *const c_void);

/// Insert a vector into an index.
///
/// Returns an `InsertResult` discriminant. `Fail` may result from the vector already being in
/// the index, or from writes failing.
///
/// vector_len is a count of elements, not bytes; the element type follows from the index's
/// `quant_type`. The pointer need not be aligned for that element type.
///
/// Note that insert has to be aware of quantizer weirdness, if buffering has to happen it happens here.  If we transition from not-quantizing to quantizing, it also has to happen here.
///
/// Attributes are optional; pass a null pointer or a zero length to insert without them.
#[unsafe(no_mangle)]
extern "C" fn insert(
    context: u64,
    index_ptr: *const c_void,
    id_data: *const u8,
    id_len: usize,
    vector_data: *const u8,
    vector_len: usize,
    attribute_data: *const u8,
    attribute_len: usize,
) -> u8;

/// Train the quantizer.
///
/// Garnet calls this once per `insert` that returned `InsertResult::SuccessStartTraining`.
/// Because inserts are concurrent it may be invoked more than once, and the implementation
/// ensures the tables are only built once.
///
/// Returns true once the tables are built, after which Garnet issues `backfill_quant_vectors`
/// calls from a thread pool. Returns false on failure, in which case it may be retried.
#[unsafe(no_mangle)]
extern "C" fn build_quant_table(context: u64, index_ptr: *const c_void) -> bool;

/// Quantize vectors that were inserted before the quantizer was trained.
///
/// Once `build_quant_table` succeeds, Garnet invokes this an arbitrary number of times from a
/// thread pool. Each invocation receives its own `task_index` and the total `task_count` so
/// that it can select and size its share of the work.
///
/// Returns true on success and false otherwise.
#[unsafe(no_mangle)]
extern "C" fn backfill_quant_vectors(
    context: u64,
    index_ptr: *const c_void,
    task_index: usize,
    task_count: usize,
) -> bool;

/// Update attribute data on a vector already in the index.
///
/// To implement VSETATTR (https://redis.io/docs/latest/commands/vsetattr/).
///
/// An empty attribute value deletes the attributes.
///
/// Return true if vector was in index and attribute was updated (even if attribute did not change), false otherwise.
#[unsafe(no_mangle)]
extern "C" fn set_attribute(
    context: u64,
    index_ptr: *const c_void,
    id_data: *const u8,
    id_len: usize,
    attribute_data: *const u8,
    attribute_len: usize,
) -> bool;

/// Find similar vectors, takes parameters of VSIM (https://redis.io/docs/latest/commands/vsim/) and maps to a reasonable interpretation.
///
/// Works with vector values.
///
/// vector_data is unquantized, vector_len will always match dimensions from create_index. As
/// with `insert`, it is a count of elements whose type follows from the index's `quant_type`.
///
/// delta is not implemented and is currently ignored.
///
/// Maximum number of results is indicated by output_distances_len, elements are i32 length prefixed in byte blobs in output_ids.
///
/// distances are [0, 1].
///
/// search_exploration_factor is the search list size and must be non-zero. beam_width is the
/// number of nodes explored per hop and must also be non-zero.
///
/// Passing a non-null `bitmap_data` with a non-zero `bitmap_len` selects filtered search, with
/// `max_filtering_effort` bounding the extra work spent satisfying the filter. Attribute data
/// is evaluated through the `filter_callback` supplied to `create_index`.
///
/// Returns the number of results, or -1 on error. Continuations are not implemented, so the
/// `continuation` parameter is currently ignored.
#[unsafe(no_mangle)]
extern "C" fn search_vector(
    context: u64,
    index_ptr: *const c_void,
    vector_data: *const u8,
    vector_len: usize,
    delta: f32,
    search_exploration_factor: u32,
    bitmap_data: *const u8,
    bitmap_len: usize,
    max_filtering_effort: usize,
    output_ids: *mut u8,
    output_ids_len: usize,
    output_distances: *mut f32,
    output_distances_len: usize,
    beam_width: u32,
    continuation: *mut c_void,
) -> i32;

/// Find similar vectors, takes parameters of VSIM (https://redis.io/docs/latest/commands/vsim/) and maps to a reasonable interpretation.
///
/// Works with item id. Parameters and return value are otherwise as documented on
/// `search_vector`.
#[unsafe(no_mangle)]
extern "C" fn search_element(
    context: u64,
    index_ptr: *const c_void,
    id_data: *const u8,
    id_len: usize,
    delta: f32,
    search_exploration_factor: u32,
    bitmap_data: *const u8,
    bitmap_len: usize,
    max_filtering_effort: usize,
    output_ids: *mut u8,
    output_ids_len: usize,
    output_distances: *mut f32,
    output_distances_len: usize,
    beam_width: u32,
    continuation: *mut c_void,
) -> i32;

/// Continues fetching results if not all were available after a call to search_xxx
///
/// Returns the number of results placed in output_xxx, or -1 on error, and sets
/// new_continuation to non-zero if even more results are available.
///
/// NOTE: This is not implemented and always returns -1.
#[unsafe(no_mangle)]
extern "C" fn continue_search(
    context: u64,
    index_ptr: *const c_void,
    continuation: *mut c_void,
    output_ids: *mut u8,
    output_ids_len: usize,
    output_distances: *mut f32,
    output_distances_len: usize,
    new_continuation: *mut c_void,
) -> i32;

/// Remove vector from index.
///
/// For implementing VREM (https://redis.io/docs/latest/commands/vrem/).
///
/// Returns true if element was removed from index.
#[unsafe(no_mangle)]
extern "C" fn remove(
    context: u64,
    index_ptr: *const c_void,
    id_data: *const u8,
    id_len: usize,
) -> bool;

/// Return number of vectors stored in index.
///
/// Equivalent to VCARD (https://redis.io/docs/latest/commands/vcard/) can be approximate, must be fast.
#[unsafe(no_mangle)]
extern "C" fn card(context: u64, index_ptr: *const c_void) -> u64;

/// Check whether an internal ID refers to a live vector.
///
/// `internal_id_data` must be exactly 4 bytes holding a native-endian u32; any other length
/// returns false.
///
/// Returns true if the vector exists in the index, false otherwise.
#[unsafe(no_mangle)]
extern "C" fn check_internal_id_valid(
    context: u64,
    index_ptr: *const c_void,
    internal_id_data: *const u8,
    internal_id_len: usize,
) -> bool;

/// Check if a vector exists in the index.
///
/// For implementing VISMEMBER - checks whether a vector with the given id is present in the index.
///
/// Returns true if the vector exists in the index, false otherwise.
#[unsafe(no_mangle)]
extern "C" fn check_external_id_valid(
    context: u64,
    index_ptr: *const c_void,
    id_data: *const u8,
    id_len: usize,
) -> bool;

/// Return up to `count` random members of the index.
///
/// For implementing VRANDMEMBER (https://redis.io/docs/latest/commands/vrandmember/). No
/// distances are produced; ids are written to output_ids as i32 length prefixed byte blobs.
///
/// Returns true on success and false otherwise.
#[unsafe(no_mangle)]
extern "C" fn random_members(
    context: u64,
    index_ptr: *const c_void,
    count: u32,
    output_ids: *mut u8,
    output_ids_len: usize,
) -> bool;

/// Return the neighbor list of a vector, with the distance to each neighbor.
///
/// For implementing VLINKS (https://redis.io/docs/latest/commands/vlinks/). Output buffers are
/// filled as they are for search_xxx.
///
/// Returns the number of neighbors written, or -1 on error. Continuations are not implemented,
/// so the `continuation` parameter is currently ignored.
#[unsafe(no_mangle)]
extern "C" fn search_neighbors(
    context: u64,
    index_ptr: *const c_void,
    id_data: *const u8,
    id_len: usize,
    output_ids: *mut u8,
    output_ids_len: usize,
    output_distances: *mut f32,
    output_distances_len: usize,
    continuation: *mut c_void,
) -> i32;
