# Unified Index Format

**Status:** Draft v1
**Scope:** static (non-streaming) indices; no tags; no disk-side secondary PQ
**Audience:** DiskANN maintainers, third-party loader implementers (e.g. `rust/` crate, Python tools)

---

## 1. Motivation

Today, building an index for SSD-served (`PQFlashIndex`) versus in-memory (`Index<T,TagT,LabelT>`) serving requires two distinct build pipelines that produce different on-disk artifacts:

- **In-memory build** (`Index::save` in `src/index.cpp`) writes a variable-width graph file, a `.data` file (full-precision vectors), `.tags`, `_labels.txt`, `_labels_to_medoids.txt`, `_labels_map.txt`, `_bitmask_labels.bin`, `_integer_labels.bin`, etc.
- **Disk build** (`build_disk_index` in `src/disk_utils.cpp`) writes `_disk.index` (4 KiB sector-packed graph + full-precision coords), `_pq_pivots.bin`, `_pq_compressed.bin`, `_medoids.bin`, `_centroids.bin`, `_max_base_norm.bin`, plus the same family of label files.

Both pipelines build the same underlying Vamana graph (shared `build_merged_vamana_index` call), but the serialized artifacts diverge. As a result an index built for one serving mode cannot be loaded in the other, and users must commit to a serving mode at build time.

**Goal:** define a single self-describing container file that can be:

- produced once by a unified build pipeline, and
- loaded as either an in-memory index (`Index::load_unified`) or an SSD-served index (`PQFlashIndex::load_unified`),

with PQ data simply ignored on the in-memory path.

### 1.1 Non-goals (v1)

- **No tags.** `Index<T,TagT,LabelT>` template instantiations stay (existing code untouched), but the unified writer/reader does not emit or consume tags.
- **No frozen points / streaming.** No dynamic-index support in this version. No `num_frozen_pts` field.
- **No disk-side secondary PQ.** The optional `_disk.index_pq_pivots.bin` / `_use_disk_index_pq` path at `src/pq_flash_index.cpp:828-835, 1534/1542` is not supported. Users needing very-high-dim disk PQ keep using the legacy format.
- **No centroids region.** `_centroids.bin` today is a load-time optimization that pre-populates `_centroid_data` (used as query-expansion seeds at `src/pq_flash_index.cpp:1327`). It is always derivable from the medoid node records via `use_medoids_data_as_centroids()` (`src/pq_flash_index.cpp:401-438`). The unified loader calls that fallback at startup, paying `_num_medoids` extra disk reads.
- **No legacy→unified conversion tool.** New format lives alongside legacy. Existing indices keep loading via their legacy code paths.

### 1.2 Design principles

1. **One file, self-describing.** All sidecar files merge into one container with a fixed-layout header that declares which optional sections are present.
2. **Disk and memory share one graph encoding.** The graph + embeddings region is byte-for-byte identical regardless of which loader consumes it.
3. **Region-level 4 KiB alignment, intra-region packing.** Major regions begin at 4 KiB-aligned file offsets to preserve the `AlignedFileReader` invariants (`include/aligned_file_reader.h:73-90`, `src/windows_aligned_file_reader.cpp:125-127`). Inside a region, payload is packed without per-record padding.
4. **No redundant per-node metadata.** Embedding size is constant (`dim * sizeof(T)`, from header) and neighbor IDs are fixed-width `uint32_t`. Per-node degree is *derived* from the offset table, not stored.

### 1.3 Supporting facts from existing code

- In-memory graph load is sequential per-node `[degree:u32][nbrs:u32*degree]` (`src/in_mem_graph_store.cpp:138-202`). The in-memory search loop is degree-oblivious, so a fixed-stride or offset-indexed layout works equally well.
- Disk search keeps full-precision coords *inside each sector record* (`src/pq_flash_index.cpp:1651`) alongside the adjacency list. PQ codes live in memory (`_pq_compressed.bin`) and approximate distances during traversal.
- In-memory `Index<T>::search` never references PQ (`grep -n _pq_table src/index.cpp` returns zero hits).
- `load_bin_impl` already accepts a `file_offset` parameter (`include/utils.h:412-426`), so embedded sub-files can be read from a byte range with no API change.

---

## 2. Format Specification (normative)

All multi-byte integers are little-endian. All offsets and lengths are in bytes from the start of the file.

### 2.1 File layout

```
+--------------------------------------------------+
| Header (4 KiB)                                   |  offset 0
+--------------------------------------------------+
| Node Offset Table: uint64[npts + 1]              |  offset = header.offset_table_off
| (padded to next 4 KiB boundary)                  |
+--------------------------------------------------+
| Graph + Embeddings Region                        |  offset = header.graph_region_off
|   Per node N: [coords:T*dim][nbrs:u32*degree]    |
|   No per-node degree field.                      |
|   Variable-width packing, no sector padding.     |
|   (region padded to next 4 KiB boundary)         |
+--------------------------------------------------+
| Medoids Region (always present)                  |
|   uint32[num_medoids] of node IDs.               |
|   num_medoids = medoids_len / sizeof(uint32_t).  |
|   (padded to 4 KiB)                              |
+--------------------------------------------------+
| PQ Pivots Region                  [optional]     |  present iff HAS_PQ
|   Mirrors current _pq_pivots.bin payload byte    |
|   for byte. (padded to 4 KiB)                    |
+--------------------------------------------------+
| PQ Compressed Codes Region        [optional]     |  present iff HAS_PQ
|   Mirrors current _pq_compressed.bin payload     |
|   byte for byte. (padded to 4 KiB)               |
+--------------------------------------------------+
| Max Base Norm Region              [optional]     |  present iff HAS_MAX_BASE_NORM
|   float[1]. MIPS preprocessing only.             |
|   (padded to 4 KiB)                              |
+--------------------------------------------------+
| Labels Region                     [optional]     |  present iff HAS_LABELS
|   Three sub-sections — see §2.4.                 |
+--------------------------------------------------+
```

Optional regions whose flag is unset have both their `off` and `len` header fields set to `0`. Regions appear in the order above; readers MUST locate each region via its header offset, not by position.

### 2.2 Header (fixed 4 KiB)

```cpp
// include/unified_index_format.h
namespace diskann {

constexpr uint32_t UNIFIED_FORMAT_MAGIC   = 0x444E4E55; // "UNND" in little-endian ASCII
constexpr uint32_t UNIFIED_FORMAT_VERSION = 1;

enum class DataTypeTag   : uint32_t { Float = 1, Uint8 = 2, Int8 = 3 };
enum class MetricTag     : uint32_t { L2 = 1, InnerProduct = 2, Cosine = 3 };
enum class LabelEncoding : uint32_t { None = 0, Bitmask = 1, Integer = 2 };

enum UnifiedFormatFlags : uint32_t {
    HAS_PQ              = 1u << 0,
    HAS_LABELS          = 1u << 1,
    HAS_MAX_BASE_NORM   = 1u << 2,
};

struct UnifiedIndexHeader {            // total reserved 4096 bytes (one sector)
    uint32_t      magic;
    uint32_t      version;
    DataTypeTag   data_type;
    MetricTag     metric;
    uint64_t      npts;
    uint64_t      dim;
    uint64_t      aligned_dim;
    uint32_t      max_degree;
    uint32_t      flags;
    uint64_t      start_node;

    // Section pointers. (off = 0, len = 0) means the optional region is absent.
    uint64_t      offset_table_off,        offset_table_len;
    uint64_t      graph_region_off,        graph_region_len;
    uint64_t      medoids_off,             medoids_len;      // always present
    uint64_t      pq_pivots_off,           pq_pivots_len;    // optional
    uint64_t      pq_codes_off,            pq_codes_len;     // optional
    uint64_t      max_base_norm_off,       max_base_norm_len;// MIPS only

    // Labels (when HAS_LABELS)
    LabelEncoding label_encoding;          // Bitmask or Integer
    uint64_t      universal_label;         // 0 if none; else the integer label value
    uint64_t      total_labels;            // distinct label count; derives bitmask row width
    uint64_t      label_dictionary_off,        label_dictionary_len;
    uint64_t      per_point_labels_off,        per_point_labels_len;
    uint64_t      per_point_label_offsets_off, per_point_label_offsets_len; // Integer encoding only

    uint64_t      file_size_bytes;       // total file size in bytes, set by writer in finalize(); 0 in v1 files

    // Implementation must pad with reserved zero bytes to reach exactly 4096 bytes.
};
static_assert(sizeof(UnifiedIndexHeader) <= 4096, "header must fit in one sector");

} // namespace diskann
```

Readers MUST:

- Reject files whose `magic != UNIFIED_FORMAT_MAGIC`.
- Reject files whose `version > UNIFIED_FORMAT_VERSION` they understand (no silent partial parsing).
- Treat reserved trailing bytes within the header as opaque (do not assume zero).
- When `file_size_bytes != 0`, reject files whose on-disk size does not match the recorded value (truncation / partial write / corruption check). The `!= 0` guard allows v1 files (which did not carry this field) to load through a v2 reader without spurious rejection.

### 2.3 Node Offset Table and Graph Region

The offset table is `uint64[npts + 1]` values, packed contiguously. For node `N` (0 ≤ N < npts):

- record start (in file): `header.graph_region_off + offset_table[N]`
- record end (in file): `header.graph_region_off + offset_table[N + 1]`
- record size: `offset_table[N + 1] - offset_table[N]`

The trailing sentinel `offset_table[npts]` equals `header.graph_region_len` (the size of the graph region payload, not counting trailing 4 KiB padding).

Each node record contains, in order:

1. `coords`: exactly `dim * sizeof(T)` bytes of vector data, where `T` corresponds to `header.data_type`.
2. `neighbors`: zero or more `uint32_t` neighbor node IDs.

There is no per-node degree field. The degree is derived:

```
degree = (record_size - dim * sizeof(T)) / sizeof(uint32_t)
```

The graph region is otherwise unstructured. Implementations MUST pad with zero bytes from `header.graph_region_off + header.graph_region_len` to the next 4 KiB-aligned file offset, so that subsequent regions begin sector-aligned. Padding bytes are not part of `graph_region_len`.

### 2.4 Labels Region

Present iff `flags & HAS_LABELS`. Three sub-sections:

#### 2.4.1 Label dictionary

Replaces today's `_labels_map.txt` + `_labels_to_medoids.txt`. One row per distinct label, packed contiguously:

```
[label_string_len:u32][label_string bytes (label_string_len bytes, no nul terminator)]
[label_integer:u32][medoid_node_id:u32]
```

`label_integer` is always written as a 4-byte little-endian unsigned integer, independent of the build-time `LabelT` template parameter (`uint16_t` values are zero-extended). This makes the on-disk dictionary self-describing and uniform across writer instantiations. Row count is implicit: read rows until `label_dictionary_len` bytes are consumed.

If `header.universal_label != 0`, the dictionary MAY contain a row whose `label_integer` matches it; otherwise the universal label has no explicit dictionary entry.

#### 2.4.2 Per-point labels

The payload format depends on `header.label_encoding`:

- **`Bitmask`**: row width is fixed at `simple_bitmask::get_bitmask_size(total_labels) * sizeof(uint64_t)` bytes (see `include/label_bitmask.h:57`). Random access: point N starts at offset `N * row_width` within the region. Each row's payload is the equivalent of today's `_bitmask_labels.bin` row.
- **`Integer`**: payload bytes are raw `uint32_t` label integers packed in point order, equivalent to `integer_label_vector::_data` (`include/integer_label_vector.h:38`). To locate point N's labels, use the per-point label offsets sub-section (§2.4.3).

#### 2.4.3 Per-point label offsets

Present iff `header.label_encoding == Integer`. Format: `uint64[npts + 1]` offsets into the per-point labels region. Point N's labels span the range `_data[offsets[N] : offsets[N+1]]` (each element a `uint32_t`). Mirrors `integer_label_vector::_offset` (`include/integer_label_vector.h:37`).

For `header.label_encoding == Bitmask`, `per_point_label_offsets_off` and `per_point_label_offsets_len` MUST both be `0`.

**On-disk ordering (Integer encoding):** for symmetry with the graph region's `[offset_table, graph_data]` layout, the writer emits the per-point-label *offsets* first, then the per-point-label *payload*. Since both regions are addressed by absolute file offsets from the header, readers are unaffected by the ordering.

### 2.5 Medoids Region (always present)

A packed `uint32_t` array of node IDs. Length: `medoids_len / sizeof(uint32_t)`. Unfiltered indices write exactly one entry; filtered indices write one entry per label-bound medoid (semantics identical to today's `_medoids.bin`).

### 2.6 PQ Regions (optional)

When `HAS_PQ` is set, both `pq_pivots_off` and `pq_codes_off` MUST be non-zero. Each region's payload is byte-identical to today's `_pq_pivots.bin` / `_pq_compressed.bin`, including the in-bin metadata header that `load_bin_impl` expects (`include/utils.h:412-426`). Loaders read these via `load_bin_impl(path, pq_pivots_off)` and `load_bin_impl(path, pq_codes_off)`.

When `HAS_PQ` is unset, both fields MUST be zero, and an SSD loader MUST reject the file with a clear error (SSD serving requires PQ).

### 2.7 Max Base Norm Region (optional)

Present iff `HAS_MAX_BASE_NORM` (MIPS preprocessing only). Payload: byte-identical to today's `_max_base_norm.bin`.

---

## 3. Load Paths (informative)

### 3.1 In-memory load — `Index::load_unified(path)`

1. Open file, read first 4 KiB → parse `UnifiedIndexHeader`. Validate magic and version.
2. Read the offset table (`npts + 1` `uint64`s starting at `header.offset_table_off`).
3. Read the graph region into a buffer (or stream it in chunks).
4. For each node N in `[0, npts)`:
   - `record = region_buf[offset_table[N] : offset_table[N+1]]`
   - `coords = record[0 : dim * sizeof(T)]` → copy into `_data_store`
   - `degree = (len(record) - dim * sizeof(T)) / sizeof(uint32_t)`
   - `neighbors = record[dim * sizeof(T) :]` interpreted as `uint32_t[degree]` → copy into `InMemGraphStore::_graph[N]`
5. If `flags & HAS_LABELS`:
   - Read the dictionary; reconstruct in-memory `label_map` and `labels_to_medoids`.
   - Read `per_point_labels`; dispatch on `header.label_encoding`:
     - `Bitmask`: feed bytes into `simple_bitmask_buf` with row width derived from `total_labels`.
     - `Integer`: also read `per_point_label_offsets`; feed both into `integer_label_vector`.
   - If `header.universal_label != 0`, apply it to the label holder.
6. Read the medoids region (always present) into the in-memory medoid list (used by filtered search).
7. **PQ regions are skipped entirely.**

### 3.2 SSD load — `PQFlashIndex::load_unified(num_threads, path)`

1. Open the file via `AlignedFileReader` plus a sync `ifstream` for the small bits.
2. Read header and offset table synchronously. Keep the offset table in memory as `_node_offsets` (`8 * npts` bytes — same order of magnitude as the existing `_medoids` / cache overhead).
3. Set `_disk_index_file = path` and `_graph_region_base = header.graph_region_off`.
4. Load PQ pivots and PQ codes via `load_bin_impl(path, header.pq_pivots_off)` and `load_bin_impl(path, header.pq_codes_off)`. SSD load fails fast if `HAS_PQ` is unset.
5. Load medoids (always present) and `max_base_norm` (if `HAS_MAX_BASE_NORM`) from their `(off, len)`. Centroids are populated by calling `use_medoids_data_as_centroids()` (`src/pq_flash_index.cpp:401`) after the medoid list is known — this reads each medoid's full-precision vector from the graph region.
6. Load labels (when `HAS_LABELS`) by the same dispatch as §3.1 step 5.
7. At search time, replace the implicit per-node sector arithmetic (`get_node_sector(N) * SECTOR_LEN`, currently at `src/pq_flash_index.cpp:1430-1431`) with an offset-table lookup:
   ```
   start_byte    = graph_region_base + node_offsets[N]
   end_byte      = graph_region_base + node_offsets[N + 1]
   aligned_start = start_byte & ~(SECTOR_LEN - 1)
   aligned_end   = (end_byte + SECTOR_LEN - 1) & ~(SECTOR_LEN - 1)
   ```
   Issue the aligned read; advance the in-buffer pointer by `(start_byte - aligned_start)` to land on the node record. Degree is `(end_byte - start_byte - dim * sizeof(T)) / 4`.

This change is encapsulated in a single helper (`node_read_window(N)`) so the bulk of `cached_beam_search` is unchanged.

---

## 4. Build Path (informative)

`build_unified_index` reuses the existing pipeline (preprocess → optional PQ training → `build_merged_vamana_index`) up to the point where the legacy code would write separate files or call `create_disk_layout`. From there:

1. Train PQ if requested (same as today; skip entirely for in-memory-only builds).
2. Stream each node from the in-memory Vamana graph + base vector file into `UnifiedIndexWriter`. The writer:
   - Reserves the 4 KiB header.
   - Reserves space for the offset table (`8 * (npts + 1)` bytes, rounded up to 4 KiB).
   - Streams node records into the graph region, recording each record's offset in the offset-table buffer.
   - Pads to 4 KiB, writes the medoids region.
   - If PQ trained, pads and writes pivots + codes.
   - If MIPS, pads and writes `max_base_norm`.
   - If labels present, pads and writes the dictionary, per-point label offsets (Integer encoding only), and per-point labels, in that order.
   - Seeks back to the start of the offset table and writes it.
   - Seeks back to byte 0 and writes the final populated `UnifiedIndexHeader`.

PQ-less builds simply leave `HAS_PQ = 0` and omit the PQ regions.

---

## 5. Implementation Roadmap

### 5.1 New files

| Path | Purpose |
|------|---------|
| `include/unified_index_format.h` | `UnifiedIndexHeader`, magic/version/flag constants, `DataTypeTag`/`MetricTag`/`LabelEncoding` enums, alignment helpers (`align_up_4k`). |
| `include/unified_index_io.h` + `src/unified_index_io.cpp` | `UnifiedIndexWriter` (assembles container with correct alignment, accumulates offset table as it streams nodes) and `UnifiedIndexReader` (parses header, exposes region `(off, len)` pairs, plus a `read_node(N)` helper for in-memory loaders). |

### 5.2 Modified files (additive only)

| File | Change |
|------|--------|
| `src/disk_utils.cpp` | Add `build_unified_index(...)` next to `build_disk_index`. Same pipeline, but the post-Vamana repack step calls `UnifiedIndexWriter` instead of `create_disk_layout`, and label/medoid emission writes into the container instead of sidecar files. `build_disk_index` is untouched. |
| `include/index.h`, `src/index.cpp` | Add `Index::save_unified(path)` and `Index::load_unified(path)`. `save_unified` walks `_data_store` + `InMemGraphStore::_graph` + label holders into `UnifiedIndexWriter`. `load_unified` parses the header and populates `_data_store` + `InMemGraphStore::_graph` from the graph region. Existing `save`/`load` paths are untouched. |
| `include/pq_flash_index.h`, `src/pq_flash_index.cpp` | Add `PQFlashIndex::load_unified(num_threads, path)`. Replaces the load path; search path adds `node_read_window(N)` helper and routes the existing async read through it. Existing `load` / `load_from_separate_paths` are untouched. |
| `src/in_mem_graph_store.cpp` | Add `set_graph_from_unified(npts, max_degree, start, per_node_adjacency_view)` so `Index::load_unified` can populate the graph without going through the file-based `load_impl`. No change to `load`/`save`/`get_neighbours`. |
| `src/abstract_index.cpp` | (Optional, follow-up.) Expose `save_unified` / `load_unified` through the virtual dispatch (`_save_unified`, `_load_unified`), mirroring the recently added `_debug_search` pattern. |

### 5.3 Phasing

The implementation is broken into phases so that each lands as a reviewable unit and can be reverted without affecting legacy paths.

1. **Phase 1 — Format primitives.** Add `include/unified_index_format.h` and the `UnifiedIndexWriter`/`UnifiedIndexReader` library. Unit tests: round-trip header, round-trip a few graph regions, round-trip both label encodings.
2. **Phase 2 — In-memory save/load.** Add `Index::save_unified` and `Index::load_unified`. Test: build a small in-memory index the legacy way, `save_unified`, `load_unified` into a fresh `Index`, run search, compare top-K against the original.
3. **Phase 3 — Disk build (unified).** Add `build_unified_index` reusing the existing PQ training and Vamana code. Test: build dataset twice (legacy vs unified) with the same parameters; compare PQ pivots/codes/medoids/labels byte-for-byte where the legacy bins are payload-identical to the corresponding unified regions.
4. **Phase 4 — SSD load (unified).** Add `PQFlashIndex::load_unified` and the `node_read_window` helper. Test: cross-load — `build_unified_index` → `PQFlashIndex::load_unified` → search → compare recall and latency against legacy disk-build + legacy disk-load.
5. **Phase 5 — Optional virtual dispatch.** Expose `save_unified` / `load_unified` on `AbstractIndex`.

Each phase keeps legacy paths fully working and adds no caller-side migration burden.

---

## 6. Verification

1. **Build symmetry.** Build a small dataset (~10 K vectors) the legacy way and the unified way with identical parameters. The unified file's PQ pivots, PQ codes, medoids, max-norm, and label payload bytes should match the corresponding legacy bin payloads byte-for-byte (modulo any in-bin headers that `load_bin_impl` handles).
2. **Cross-load (memory).** Build unified → load with `Index::load_unified` → run search; compare recall@10 against legacy in-memory build + legacy load over the same dataset. The graph is identical so recall should match within a tight margin.
3. **Cross-load (disk).** Build unified → load with `PQFlashIndex::load_unified` → run search; compare recall@10 *and* latency against legacy disk build + legacy disk load. Flag if the unaligned-slice read amplification regresses by more than ~10 % (this is a known "test later" item).
4. **PQ-less unified.** Build unified without PQ (in-memory-only). Confirm: file is smaller; `PQFlashIndex::load_unified` rejects it with a clear "missing PQ" error; `Index::load_unified` succeeds.
5. **Legacy regression.** Run the existing test suite (`tests/`, `tests/utils/`). All legacy load/build paths must continue to pass unchanged.
6. **Forward-compat.** Hand-craft a unified file with `version = UNIFIED_FORMAT_VERSION + 1` and confirm both loaders fail fast with an "unsupported version" error rather than silently misinterpreting.

---

## 7. Open Questions and Follow-ups

- **Read amplification.** Dropping per-node sector padding means SSD reads slice from a 4 KiB-aligned window that may be up to 2 × `(node_record_size + SECTOR_LEN)` bytes. This is the regression the user has flagged for measurement. If unacceptable, a follow-up can add an opt-in `pad_nodes_to_sector` build flag whose payload format is a strict subset of v1 (same header, same offset table, just larger `offset_table[N+1] - offset_table[N]` deltas).
- **`AbstractIndex` virtual dispatch.** Whether `save_unified`/`load_unified` need to be exposed through the type-erased base depends on caller demand; deferred to Phase 5.
- **Conversion tool.** Not in v1. If needed later, a small `legacy_to_unified` utility can be added that calls `UnifiedIndexReader`/`UnifiedIndexWriter` and reads legacy bins via existing helpers; no format change required.

---

## 8. Glossary

| Term | Definition |
|------|------------|
| `SECTOR_LEN` | 4096 bytes. Sector size required by `AlignedFileReader` on Windows (`FILE_FLAG_NO_BUFFERING`) and by libaio at 512-byte minimum on Linux. The unified format uses 4096 throughout for cross-platform compatibility. |
| `T` | The vector element type, one of `float`, `uint8_t`, `int8_t`, encoded as `DataTypeTag`. |
| `LabelT` | Label integer type, `uint16_t` or `uint32_t`, fixed at build time by the template instantiation. |
| `medoid` | Graph entry node for search. Unfiltered indices have one; filtered indices have one per label. |
| `universal_label` | A label value that matches every point unconditionally. Sentinel `0` means none. |
