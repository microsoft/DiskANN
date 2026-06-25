# diskann-graphivf — Performance Notes & Optimization Log

A running document for tracking performance-sensitive areas, current behavior,
deliberate omissions, and investigation results for the `diskann-graphivf`
crate (hybrid graph + clustered-IVF index).

> Status legend: 🔴 not started · 🟡 in progress · 🟢 done · ⚪ deferred (seam left in code)

---

## Search path overview

The query path in [src/index.rs](src/index.rs):

1. Normalize the query (cosine only).
2. Graph-search the centroid index for the nearest `nlist` centroids.
3. Fetch those `nlist` inverted lists from disk.
4. Score the query against every fetched vector (`SquaredL2`).
5. Select top-`k`.

Step 3 is the only part that touches storage and dominates query latency. The
notes below are organized around the components that determine how fast that
read — and the work around it — can be.

---

## Performance-sensitive components

### 1. Read amplification (bytes fetched per useful byte) — 🟢 addressed

The biggest lever for IVF-on-disk. Every list read drags in whole 512-byte
sectors. With target density (~1 centroid per 10–20 points, f32 vectors) a list
is often smaller than one or two sectors, so padding read is a real tax.

- **Current behavior:** lists are packed **contiguously with no per-list
  padding** ([src/storage.rs](src/storage.rs) — `write_lists`). Each read uses
  the *smallest enclosing* 512-aligned window (`cluster_window`), not a fixed
  per-list sector.
- **Rejected alternative:** sector-aligning every list start would waste up to
  511 bytes per probed list per query (×`nlist` per query).

### 2. Number of I/O operations per query — 🟢 addressed

This workload is IOPS-bound, not bandwidth-bound: `nlist` small scattered reads
per query. Serializing them would pay device latency `nlist` times.

- **Current behavior:** all probed lists are gathered into a **single batched**
  `reader.read(&mut reads)` call ([src/index.rs](src/index.rs)). `AlignedFileReader`
  submits the whole batch to the backend (io_uring on Linux / IOCP on Windows),
  giving a queue depth of `nlist` instead of 1. This is the most important
  structural decision in the read path.

### 3. Alignment correctness (avoid buffered fallback / unaligned errors) — 🟢 addressed

Direct I/O (`O_DIRECT`) requires sector-aligned offset, length, **and** buffer
address. Any miss => error or silent slow buffered fallback.

- **Current behavior:**
  - Read windows are 512-aligned on both ends (`cluster_window`).
  - Buffers are `Poly<[u8], AlignedAllocator::A512>`, so the buffer base address
    is 512-aligned.
  - The file is zero-padded up to a 512 multiple so an aligned tail read never
    runs past EOF ([src/storage.rs](src/storage.rs)).
  - `record_bytes` is always a multiple of 4, so `inner_offset` is 4-byte
    aligned and `bytemuck::cast_slice` to `u32`/`f32` is sound and zero-copy.

### 4. Parse cost (deserialization on the hot path) — 🟢 addressed

- **Current behavior:** `parse_cluster` is **zero-copy** — `bytemuck::cast_slice`
  reinterprets the read buffer in place, no allocation or memcpy. The on-disk
  layout *is* the in-memory layout.

### 5. Scoring / distance compute — 🟢 reuses shared SIMD

- **Current behavior:** `search` takes the query already in the stored element
  type `T`. It is preprocessed once into a `T::QueryDistance` scorer and every
  candidate is scored directly over `T` with `evaluate_similarity` — no
  per-candidate decode. For f16 the scorer keeps an `f32` copy of the query
  internally and uses the SIMD `f32 × f16` kernel. The only `T → f32` decode is a
  single `as_f32` on the query for the full-precision centroid graph (a no-op
  when `T == f32`). All paths dispatch to the SIMD primitives in
  `diskann-vector`. Not reimplemented.

### 6. Top-k selection — 🟢 addressed

- **Current behavior:** `select_nth_unstable_by` (quickselect, O(n)) partitions,
  then only the surviving `k` are sorted — avoids a full sort of all candidates.

---

## Deferred optimizations (seams left in code, marked `TODO(perf)`)

These are conscious omissions for v1, kept easy to find via inline
`TODO(perf)` comments.

### A. Per-query buffer allocation — ⚪ deferred

Each query allocates fresh `Poly` buffers ([src/index.rs](src/index.rs)). Under
load this is allocator churn on the hot path.

- **Proposed fix:** a reusable per-`Searcher` buffer pool. `Searcher` already
  owns scratch (`cids`/`cdist`), so it's the natural home.
- **Expected win:** removes per-query allocations; clearest self-contained win.
- **Risk/why deferred:** kept v1 correct and readable first.

### B. Window de-duplication / coalescing — ⚪ deferred

Adjacent clusters can share a 512-byte sector; today they may be read twice.

- **Proposed fix:** coalesce overlapping/adjacent windows into one read.
- **Expected win:** fewer IOPS + less amplification.
- **Risk/why deferred:** only helps when probed clusters are neighbors on disk.

### C. Parallel scoring across lists — ⚪ deferred

The score loop is embarrassingly parallel across lists but is currently serial.

- **Proposed fix:** parallel scan over fetched lists.
- **Expected win:** matters only at large `nlist` or high `dim`.
- **Risk/why deferred:** for `nlist` in the dozens with small lists, the scan is
  cheap relative to the disk wait; parallelism would add overhead.

### D. Async overlap of compute and I/O / prefetch during the graph walk — ⚪ deferred

The read currently **blocks** until all lists arrive, then scoring starts.

- **Proposed fix:** begin prefetching lists for promising centroids *during* the
  graph walk, before it terminates, so disk latency hides behind graph
  traversal. Needs a prefetch hook in the accessor.
- **Expected win:** potentially the largest latency reduction.
- **Risk/why deferred:** largest design change; explicitly out of scope for v1.

---

## Open questions / unknowns

- Interplay of (A) buffer reuse and (D) prefetch under real concurrency.
- Is `nlist` small reads the right granularity vs. larger coalesced reads?
- None of the above is **measured** — there is no benchmark backend yet (per
  scope choice). All current reasoning is analytical, not profiled.

### Next step to enable data-driven tuning

A microbenchmark harness that sweeps `nlist`, list density (points/centroid),
`dim`, and I/O queue depth — so optimizations are justified by numbers rather
than intuition.

---

## Investigation log

> Append dated entries here as optimizations are explored or benchmarked.

### 2026-06-24 — Generic stored element type (f16 inverted lists)

The search/build path is generic over the inverted-list element type via the
repo-canonical [`diskann::utils::VectorRepr`](../diskann/src/utils/vector_repr.rs)
trait, which already implements `f32`, `Half` (f16), `i8`, `u8`, and quantized
`MinMax8`. `GraphIvfIndex<T = f32>` / `Searcher<T = f32>` carry the element type;
the centroid graph stays full-precision `f32`.

- **Why it helps read amplification (component #1):** storing list vectors as
  f16 halves the per-vector byte cost (`dim*2` vs `dim*4`), so each sector-sized
  read carries roughly twice as many candidates — directly reducing bytes
  fetched per useful candidate and, for larger lists, the number of sectors per
  list.
- **Preprocessed `T`-space scoring:** `search` takes a `&[T]` query and turns it
  once into a `T::QueryDistance` scorer (a `PreprocessedDistanceFunction`) reused
  across every candidate. Query and corpus are both scored as `T` with no
  per-candidate decode. For f16 the scorer holds an `f32` copy of the query
  internally and uses the SIMD `f32 × f16` `SquaredL2` kernel, so the stored f16
  vectors are read directly. Cosine is reduced to squared-L2; the caller
  normalizes the query (the corpus is normalized at build time). The only
  `T → f32` conversion is a single `as_f32` on the query for the centroid graph.
- **Storage change:** each on-disk record is padded to 4 bytes (`RECORD_ALIGN`)
  so `u32` ids stay 4-byte aligned even with 2-byte f16 vectors. This is a
  **no-op for f32** (records were already 4-multiples), so the f32 layout and
  existing files are unchanged. The metadata persists `size_of::<T>()` and
  `load::<T>` rejects an element-size mismatch (a size check — it distinguishes
  f32 from f16, but not equally sized types such as `i8` vs `u8`).
- **Why `VectorRepr` over a bespoke trait:** it is the idiomatic abstraction used
  across the codebase (bftree, garnet, disk), brings `i8`/`u8`/quantized support
  for free, and its preprocessed `T::QueryDistance` scores query and corpus in
  the stored `T` representation with no per-candidate decode. Tradeoff:
  `VectorRepr` carries no on-disk format tag, so the crate persists
  `size_of::<T>()` itself for load validation.
- **Status:** correctness validated — f16 L2/cosine recall meets the same bars as
  f32 on the synthetic well-separated test data. **Not yet benchmarked** for
  latency/throughput; the amplification win is analytical pending the
  microbenchmark harness.
- **Future formats:** because the seam is `VectorRepr`, `i8`/`u8` and quantized
  encodings already type-check; what remains is query-side quantization and (for
  quantized) an optional full-precision rerank file.


