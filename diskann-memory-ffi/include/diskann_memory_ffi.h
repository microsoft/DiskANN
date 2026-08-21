#ifndef DISKANN_MEMORY_FFI_H
#define DISKANN_MEMORY_FFI_H

#pragma once

#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <ostream>
#include <new>

namespace DiskANN {

enum class DiskANNError {
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
};

enum class Metric {
  L2 = 0,
  InnerProduct = 1,
  Cosine = 2,
};

enum class TagType {
  U32 = 0,
  U64 = 1,
  U128 = 2,
};

enum class DeleteMethod {
  OneHop = 0,
  TwoHopAndOneHop = 1,
  VisitedAndTopK = 2,
};

struct DiskANNResult {
  DiskANNError error;
  char *error_message;
  void *handle;
};

struct IndexConfiguration {
  Metric dist_metric;
  size_t dim;
  uint32_t search_list_size;
  uint32_t num_threads;
  const char *index_path;
  uint32_t tag_type;
  float max_insert_percentage;
  uint32_t build_search_list_size;
  uint32_t graph_degree;
  uint8_t consolidate_enabled;
  float consolidate_threshold;
  uint32_t consolidate_threads;
  const char *data_path;
  const char *tag_path;
  uint8_t is_streaming;
  uint32_t delete_method;
  uint32_t delete_num_to_replace;
  uint32_t delete_search_k;
  uint32_t delete_search_l;
};

struct SearchParams {
  uint32_t k;
  uint32_t search_list_size;
  uint32_t beam_width;
};

struct SearchResult {
  uint32_t *indices;
  float *distances;
  size_t result_count;
};

struct DiskANNStatus {
  DiskANNError error;
  char *error_message;
};

struct ByteSlice {
  const uint8_t *ptr;
  size_t len;
};

struct TableStats {
  size_t tag_memory_bytes;
  size_t active_count;
  uint64_t insert_count;
  uint64_t delete_count;
};

extern "C" {

/// # Safety
/// `index_path` must be a readable NUL-terminated string.
/// `is_streaming` must be 0 for read-only loading or 1 for streaming loading.
DiskANNResult diskann_load_memory_index_u8(IndexConfiguration config);

/// # Safety
/// All pointers must be aligned, valid for their declared lengths, and pairwise non-overlapping.
/// On entry, `result.result_count` must be the capacity of both output buffers in results.
/// Streaming handles require `result_count * tag_width` writable bytes behind `indices`.
/// Streaming failures may return an owned message released by `diskann_free_error_message`.
DiskANNResult diskann_search_memory_index_u8(const uint8_t *query,
                                             size_t query_len,
                                             SearchParams params,
                                             void *handle,
                                             SearchResult *result);

/// # Safety
/// `handle` must be null or a token returned by the load function in either mode.
/// Null and stale handles are silently ignored.
void diskann_free_memory_index(void *handle);

/// # Safety
/// Vector and tag buffers must be readable for their declared lengths.
DiskANNStatus diskann_insert_streaming_index_u8(const uint8_t *vector,
                                                size_t vector_len,
                                                const uint8_t *tag,
                                                size_t tag_len,
                                                ByteSlice label,
                                                void *handle);

/// # Safety
/// `value` must point to writable storage for one bool.
DiskANNStatus diskann_is_max_insert_streaming_index(void *handle, bool *value);

/// # Safety
/// Tag bytes must be readable for `tag_len`.
DiskANNStatus diskann_delete_streaming_index(const uint8_t *tag, size_t tag_len, void *handle);

/// # Safety
/// `value` must point to writable storage for one bool.
DiskANNStatus diskann_should_consolidate_delete_streaming_index(void *handle, bool *value);

/// # Safety
/// `handle` must be a live streaming token.
DiskANNStatus diskann_consolidate_delete_streaming_index(void *handle);

/// # Safety
/// Paths must be readable for their declared lengths.
/// Output paths must be distinct. Capture failures publish no partial snapshot files.
DiskANNStatus diskann_dump_streaming_index(void *handle,
                                           ByteSlice index_path,
                                           ByteSlice data_path,
                                           ByteSlice tag_path);

/// # Safety
/// `stats` must point to writable storage.
DiskANNStatus diskann_get_table_stats(void *handle, TableStats *stats);

/// # Safety
/// `handle` must be a live streaming token. Memory, stale, and null handles return
/// `InvalidHandle`/`NullPointer` and are not removed.
DiskANNStatus diskann_free_streaming_index(void *handle);

/// # Safety
/// Message must be null or returned by this library and not previously freed.
void diskann_free_error_message(char *value);

}  // extern "C"

}  // namespace DiskANN

#endif  // DISKANN_MEMORY_FFI_H
