/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

/// Create a new empty index
/// Takes the params of VADD (see: https://redis.io/docs/latest/commands/vadd/), maps to a reasonable interpretation
///
/// (context % 4) == 0, xxx_callbacks add 0/1/2/3 depending on data stored
///
/// Expectation is any state necessary to recover an index is stored via read/write callbacks - including quantizers.
///
/// reduce_dims == 0 to indicate no reduction requested (and can be ignored even if provided, if that is reasonable).
///
/// quant_type needs option that map from NoQuant, Bin, and Q8 (as that's what provided in Redis) in addition to any custom index.  They don't need to be exact, just reasonable.
///
/// metric_type is passed as a raw i32. Valid values are:
/// - 0: Cosine
/// - 1: InnerProduct
/// - 2: L2 (Euclidean distance)
/// - 3: CosineNormalized
/// Invalid values will cause the function to return null.
///
/// Returning a single pointer conceal all the generics behind an opaque handle
#[unsafe(no_mangle)]
extern "C" fn create_index(
    context: u64,
    dimensions: u32,
    reduce_dims: u32,
    quant_type: SomeCStyleEnumeration,
    metric_type: i32,
    build_exploration_factor: u32,
    num_links: u32,
    read_callback: unsafe extern "C" fn(u64, *const u8, usize, *mut u8, usize) -> i32,
    write_callback: unsafe extern "C" fn(u64, *const u8, usize, *const u8, usize) -> bool,
    delete_callback: unsafe extern "C" fn(u64, *const u8, usize) -> bool,
) -> *mut c_void;

/// Drop a previously created index
///
/// Not called if any other operation against the index may be in flight or started.
#[unsafe(no_mangle)]
extern "C" fn drop_index(
    context: u64,
    index: *const c_void
);

/// Insert a vector into an index.
///
/// Returns true if the vector is added, false if it is not.
///
/// False may result from the vector already being in the index, or writes failing.
///
/// Note that insert has to be aware of quantizer weirdness, if buffering has to happen it happens here.  If we transition from not-quantizing to quantizing, it also has to happen here.
///
/// For now, attribute_data/attribute_len can be ignored - just want space for them.
#[unsafe(no_mangle)]
extern "C" fn insert(
    context: u64,
    index: *const c_void,
    id_data: *const u8,
    id_len: usize,
    vector_data: *const f32,
    vector_len: usize,
    attribute_data: *const u8,
    attribute_len: usize
) -> bool;

/// Update attribute data on a vector already in the index.
///
/// To implement VSETATTR (https://redis.io/docs/latest/commands/vsetattr/).
///
/// We can skip implementing this for now since we don't need filters yet, just needs to be spec'd.
///
/// Return true if vector was in index and attribute was updated (even if attribute did not change), false otherwise.
#[unsafe(no_mangle)]
extern "C" fn set_attribute(
    context: u64,
    index: *const c_void,
    id_data: *const u8,
    id_len: usize,
    attribute_data: *const u8,
    attribute_len: usize
) -> bool;

/// Find similar vectors, takes parameters of VSIM (https://redis.io/docs/latest/commands/vsim/) and maps to a reasonable interpretation.
///
/// Works with vector values.
///
/// vector_data is unquantized, vector_len will always match dimensions from create_index.
///
/// delta will be [0, 1].
///
/// Maximum number of results is indicated by output_distances_len, elements are i32 length prefixed in byte blobs in output_ids.
///
/// distances are [0, 1].
///
/// Returns number of results, sets continuation to non-zero if there are more to fetch.
///
/// Filtering can be ignored for now, just reserving space in FFI.
///
/// Various search & effort values can be ignored for now, but will eventually be mapped to something sensible.  Exist for compat with Redis.
#[unsafe(no_mangle)]
extern "C" fn search_vector(
    context: u64,
    index_ptr: *const c_void,
    vector_data: *const f32,
    vector_len: usize,
    delta: float,
    search_exploration_factor: i32,
    filter_data: *const u8,
    filter_len: usize,
    max_filtering_effort: usize,
    output_ids: *mut u8,
    output_ids_len: usize,
    output_distances: *mut f32,
    output_distances_len: usize,
    continuation: *mut c_void,
) -> i32;


/// Find similar vectors, takes parameters of VSIM (https://redis.io/docs/latest/commands/vsim/) and maps to a reasonable interpretation.
///
/// Works with item id
///
/// delta will be [0, 1].
///
/// Maximum number of results is indicated by output_distances_len, elements are i32 length prefixed in byte blobs in output_ids.
///
/// distances are [0, 1].
///
/// Returns number of results.
///
/// Filtering can be ignored for now, just reserving space in FFI.
///
/// Various search & effort values can be ignored for now, but will eventually be mapped to something sensible.  Exist for compat with Redis.
#[unsafe(no_mangle)]
extern "C" fn search_element(
    context: u64,
    index_ptr: *const c_void,
    id_data: *const u8,
    id_length: usize,
    delta: float,
    search_exploration_factor: i32,
    filter_data: *const u8,
    filter_len: usize,
    max_filtering_effort: i32,
    output_ids: *mut u8,
    output_ids_len: usize,
    output_distances: *mut f32,
    output_distances_len: usize,
    continuation: *mut c_void
) -> i32;

/// Continues fetching results if not all were available after a call to search_xxx
///
/// Returns number of results placed in output_xxx
///
/// Sets new_continuation to non-zero if even more results are available.
#[unsafe(no_mangle)]
extern "C" fn continue_search(
    context: u64,
    index_ptr: *const c_void,
    continuation: usize,
    output_ids: *mut u8,
    output_ids_len: usize,
    output_distances: *mut f32,
    output_distances_len: usize,
    new_continuation: *mut c_void
) -> i32;

/// Remove vector from index.
///
/// For implementing VREM (https://redis.io/docs/latest/commands/vrem/).
///
/// Returns true if element was removed from index.
#[unsafe(no_mangle)]
extern "C" fn delete(
    context: u64,
    index_ptr: *const c_void,
    vector_data: *const u8,
    vector_len: usize
) -> bool;

/// Return number of vectors stored in index.
///
/// Equivalent to VCARD (https://redis.io/docs/latest/commands/vcard/) can be approximate, must be fast.
#[unsafe(no_mangle)]
extern "C" fn card(
    context: u64,
    index_ptr: *const c_void,
) -> u64;


/// Check if a vector exists in the index.
///
/// For implementing VISMEMBER - checks whether a vector with the given id is present in the index.
///
/// Returns true if the vector exists in the index, false otherwise.
#[unsafe(no_mangle)]
extern "C" fn has_vector(
    context: u64,
    index_ptr: *const c_void,
    id_data: *const u8,
    id_len: usize,
) -> bool;

// To inspect neighbor lists and vector data, Garnet just has to be aware of the format - not a big deal, no need for FFI.