# DiskANN memory FFI

This crate builds `diskann_memory_ffi` as static and dynamic libraries and regenerates
`include/diskann_memory_ffi.h` with cbindgen. Read-only and streaming loads share one
`IndexConfiguration`. Its original 0.1.3 fields remain the exact public prefix; streaming fields
and the final `is_streaming` mode byte are appended. `diskann_load_memory_index_u8` is the sole
load API and selects read-only (`0`) or streaming (`1`) behavior. Rust receives the metric
representation as raw `u32` and validates it safely.

`diskann_search_memory_index_u8` is the sole search API and uses the original `SearchResult`
unchanged. The handle selects read-only or streaming behavior. Before a call, `result_count` is
the capacity of both output buffers in results; afterward it is the number written. Streaming
treats `indices` as opaque tag storage, requiring `result_count * tag_width` writable bytes.
Read-only failures retain null messages. Streaming failures may return an owned message that must
be released with `diskann_free_error_message`; successful searches return the same live handle.
`SearchParams::beam_width` uses `1` when set to zero; a nonzero value is forwarded unchanged to
DiskANN while the existing internal search-list `+1` adjustment remains independent.

The streaming ABI is limited to operations used by Ads.SimilaritySearchEngine. Search, insert,
delete, and consolidation may run concurrently on the same handle; the FFI holds no operation-level
lock and relies on DiskANN/provider adjacency, mapping, and epoch synchronization. Registry removal
prevents new operations while in-flight `Arc` owners keep the index alive. Streaming error messages
are owned and must be released with `diskann_free_error_message`; legacy errors retain null messages.
Streaming `num_threads` is forwarded as DiskANN's scratch-resource thread hint; it does not set
the Tokio runtime worker count.
Streaming load converts raw `delete_method` values `0`/`1`/`2` to OneHop, TwoHopAndOneHop, or
VisitedAndTopK. `delete_num_to_replace` is required. VisitedAndTopK additionally requires nonzero
`delete_search_l >= delete_search_k`; other methods ignore K/L.

`diskann_free_memory_index` is the common void free for either handle kind and silently ignores
null or stale handles. The compatibility streaming free rejects null, stale, or memory handles
with an error and never removes a mismatched handle.

Streaming load accepts canonical C++ streaming graph/data/tags snapshots with one final frozen
point. Tags use `[u32 count][u32 dimensions=1][payload]`; a final all-zero frozen placeholder is
accepted and excluded. Configuration paths are NUL-terminated strings. `u32`, `u64`, and `u128`
tag payloads remain opaque bytes at the ABI boundary. One public `diskann-inmem::StreamingTag`
trait owns width, little-endian decode/encode, and frozen-placeholder semantics for all widths.
The typed in-memory provider owns each
external-tag-to-internal-`u32` mapping; graph adjacency IDs remain `u32`. `diskann-inmem` owns
the path-level C++ snapshot parser, validation, frozen-point restoration, and adjacency remapping.

Filtered streaming search is unsupported and the unified search API has no label argument. Dump
captures a compact active-tag set, filters/remaps graph edges, writes three temporary C++ snapshot
files, and publishes them only after all writes succeed. A per-table snapshot gate lets insert,
delete, and consolidation remain mutually concurrent under shared guards, while dump holds the
exclusive guard across capture and publication. Search and instantaneous status/stat calls do not
take the gate and remain concurrent with dump.
Generic table stats expose tag memory, active count, insert count, and delete count. Memory handles
currently return a successful zero/default `TableStats`; streaming handles return live values.
Counters use relaxed atomic snapshots for telemetry. Consolidation atomically claims pending deletes;
deletes arriving during consolidation remain pending, and a failed consolidation restores its claim.

## NuGet

On Windows, run `nuget/pack.ps1` to build Debug and Release x64 binaries and create
`target/nuget/RustDiskANNFFI.Library.0.2.0.nupkg`. The package includes the compatibility header,
`build/RustDiskANNFFI.Library.targets`, DLL/PDB files under `x64/bin/debug|release`, and import
libraries under `x64/lib/debug|release`.

CoreXT consumers should reference package ID `RustDiskANNFFI.Library` version `0.2.0`; importing
the package targets adds the include/library paths, links `diskann_memory_ffi.lib`, and copies the
matching DLL/PDB to the output directory.
