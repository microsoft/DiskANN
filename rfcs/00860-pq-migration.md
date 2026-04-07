# PQ Migration & Optimization Design Document

**Date**: 2026-04-07
**Status**: Design (pre-implementation)

---

## 1. Problem Statement

The PQ (Product Quantization) code in `diskann-providers` is a direct port from C++ with
accumulated cruft, suboptimal performance in hot paths, and poor separation of concerns.
Key building blocks (distance table creation, quant-quant distances, lookup functions) should
move to `diskann-quantization` which has a much higher code quality bar. The remaining
providers-level code should be cleaned up and modernized.

---

## 2. Architecture: Two-Table Design

### 2.1 New `Table` (padded layout, quant-quant optimized)

Replaces `BasicTable`. Stores pivots with chunks **padded to SIMD width** so that every
chunk is a fixed-size full-vector SIMD operation. No masking, no epilogues, no Resumable
machinery.

- Padding cost is minimal (a few extra zeros per chunk, computed once at construction)
- SIMD kernel per chunk: load left, load right, diff, FMA — 3 instructions
- Outer loop has no variable-length complexity

### 2.2 `TransposedTable` (existing, compression + distance table creation)

Keeps its current role. Block-transposed layout is optimized for `process_into` (computing
query-to-centroid partial distances) and `compress_batch`. Its layout is structurally
hostile to quant-quant distances — that's `Table`'s job.

### 2.3 Relationship

- Two separate data structures with complementary APIs
- Constructed from the same source pivot data
- A builder function `fn build_tables(pivots, offsets) -> (TransposedTable, Table)` to
  ensure construction coordination
- Future: may be bundled into a `PQScheme` one-stop-shop type (spherical quantization
  precedent), serving as a vessel for further optimizations (e.g., SQ-compressed lookup
  tables)

---

## 3. Distance Table & Lookup Architecture

### 3.1 Unified Lookup Table Approach — All Metrics

All four metrics (L2, IP, Cosine, CosineNormalized) use the same table-based lookup
architecture, differing only in table contents and final reduction:

| Metric | Table contents per (chunk, centroid) | Final reduction |
|--------|--------------------------------------|-----------------|
| L2 | squared partial distance | `Σ partials` |
| IP | inner product partial | `-Σ partials` |
| Cosine | (IP partial, norm partial) pair | `1 - ip_sum / sqrt(q_norm² × norm_sum)` |
| CosineNorm | same as Cosine (or deferred to L2 proxy) | same as Cosine |

This eliminates `DirectCosine`'s current approach (raw query stored, full SIMD distance
per evaluation) and gives cosine the same table-lookup performance characteristics as
L2/IP.

### 3.2 `pq_dist_lookup_single` Optimization

The hot lookup function. Current implementation:

```rust
for (&value, distances) in iter {
    accum += distances[value.into_usize()];  // bounds-checked, serial dependency
}
```

**Optimization axes:**

1. **Unchecked accesses**: `u8` indexing into a 256-entry table is always in-bounds.
   Safe (provable from invariants), low risk, 5-15% improvement in the hot loop.

2. **Break the FP-add dependency chain**: Use 4-8 independent scalar accumulators.
   Reduces the serial add chain from N iterations to N/4 or N/8. ~2-3× improvement
   on the hot loop. (Note: LLVM will NOT auto-vectorize this due to FP associativity
   — explicit reordering is required.)

3. **AVX-512 gather (v4 specialization)**: `vpgatherdps zmm` processes 16 lookups per
   instruction. For 384 chunks: 24 iterations vs ~48 for scalar-unrolled-8. Solid
   1.6-2× speedup over well-optimized scalar on Ice Lake+/Sapphire Rapids. Not worth
   doing on AVX2 where gather throughput is too low.

4. **Cosine gather variant**: Two 16-wide gathers per iteration (IP + norm) with shared
   index vector. ~48 cycles for 384 chunks. Messier but still far better than current
   DirectCosine.

---

## 4. Quant-Quant Distance Optimization

Current `self_distance` on `FixedChunkPQTable` has ~6:1 overhead:compute ratio per chunk:
5 bounds checks + Resumable SIMD bookkeeping vs 2-4 FMAs of actual work (for typical
2-4 dim chunks).

### 4.1 New approach with padded `Table`

- Chunks padded to SIMD width → no masking/epilogues
- Unchecked accesses (safety provable from table invariants)
- No Resumable machinery — direct SIMD kernel per chunk
- Per-chunk SIMD kernel: load left pivot, load right pivot, diff, FMA into accumulator

### 4.2 Precomputed chunk distance tables — rejected

256×256×4 bytes = 256KB per chunk. For 384 chunks = 96MB (×2 for cosine). Memory cost
is untenable for large PQ configurations.

---

## 5. Metric Handling

### 5.1 Metric at construction time

`PQScheme` (or equivalent) takes a `Metric` at construction. This allows:
- Metric-specific auxiliary data in the padded table (e.g., centroid norms for cosine)
- Monomorphized SIMD kernels per metric
- Eliminates "call the wrong distance method" bugs

### 5.2 `dyn` at computer boundary

Use the runtime `Metric` enum and create trait objects at computer construction:

```rust
fn query_computer(&self, query: &[f32]) -> Box<dyn QueryDistance> {
    match self.metric {
        Metric::L2 => Box::new(self.build_typed::<SquaredL2>(query)),
        ...
    }
}
```

The `match` is the only place where all specializations are instantiated. Inner loops
stay monomorphized. `Box<dyn>` dispatch is one indirect call per `evaluate` — amortized
to nothing over thousands of candidates per query.

Reduces monomorphization bloat vs a type-parameter approach. Trade-off: locked to metrics
in `diskann_vector::distance::Metric`.

---

## 6. Object Pool Migration

### 6.1 Move `ObjectPool` from `diskann` to `diskann-utils`

The `ObjectPool` module (`diskann/src/utils/object_pool.rs`) has **zero crate-internal
dependencies** — only `std`. Clean, well-tested, handles poison recovery correctly.

Includes: `ObjectPool<T>`, `PooledArc<T>`, `PooledRef<'_, T>`, `PoolOption<T>`,
`AsPooled`, `TryAsPooled`, `Undef`.

### 6.2 Motivation

- `diskann-quantization` already depends on `diskann-utils` — no new dependency edge
- Distance table allocation is a hot-path concern; pool avoidance of malloc matters
- `PooledArc<Vec<f32>>` is `'static` — satisfies the paged search constraint where
  query computers must be `'static`
- Pointer stability: `Vec<f32>` has stable-address guarantee (no realloc without `&mut`),
  resolving the `Deref` idempotency concern for unsafe SIMD code
- Pool capacity becomes a tunable parameter on the final table/scheme type

### 6.3 Migration path

- Move module to `diskann-utils`
- `diskann` re-exports from `diskann-utils`
- `diskann-providers` switches imports
- `diskann-quantization` gains pool-aware distance table allocation

---

## 7. Code Cleanup in `diskann-providers/src/model/pq/`

### 7.1 Remove / relocate

| File | Lines | Action | Reason |
|------|-------|--------|--------|
| `distance/multi.rs` | 1,167 | Move to CDB internal crate | Versioned multi-table is purely a CDB online-reindexing concern |
| `pq_construction.rs` chunk offset functions | ~50 | Move to `diskann-quantization` | Pure math, belongs alongside `ChunkOffsets` |

### 7.2 Clean up

| Item | Action | Reason |
|------|--------|--------|
| 6 redundant `DistanceFunction` impls in `dynamic.rs` | Remove, use `Accessor::ElementRef` | `ElementRef` eliminates the need for `&[u8]`/`&&[u8]`/`&Vec<u8>` impls |
| `DirectCosine` in `cosine.rs` | Replace with table-based cosine (IP+norm pairs) | Unifies all metrics under lookup-table architecture |
| `NUM_PQ_CENTROIDS = 256` magic constant | Make a type-level constraint | Currently used in safety arguments as a runtime value |
| `PQCompressedData` using `AlignedBoxWithSlice` | Update to current `AlignedSlice<T>` API | `AlignedBoxWithSlice` is deprecated |
| `#[allow(dead_code)]` on `PQCompressedData` | Audit and remove dead fields | Code smell |
| CosineNormalized → L2 fallback comments | Centralize the rationale | Currently copy-pasted across `dynamic.rs` and `quantizer_preprocess.rs` |

### 7.3 Split `pq_construction.rs` (1,956 lines)

Currently mixes: pure math, k-means training, file I/O, and compression.

Proposed split:
- Chunk offset calculation → `diskann-quantization`
- K-means training (pivot generation) → stays in providers but as a focused module
- Compression (PQ data generation) → separate module, uses `TransposedTable::compress_batch`
- Matrix helpers (`accum_row_inplace`, `get_chunk_from_training_data`) → `diskann-utils`
  or inline

---

## 8. `iface.rs` Equivalent for PQ — Deferred

Spherical quantization's `iface.rs` compresses four dispatch axes (bit-width, query
layout, metric, µarch) into single `dyn` dispatch via `Curried` + `Reify` + `Poly`.

PQ has fewer axes (no bit-width, no query layout), so an equivalent would be simpler.
However, the right design depends on integration experience with the search
infrastructure and CDB's migration to `QueryComputer`.

**Decision**: Build the `quantizer.rs`-level building blocks first. Defer the type-erasure
facade until integration needs force it.

---

## 9. Future Optimizations (not in initial scope)

- **SQ-compressed lookup tables**: For 384 chunks, the f32 lookup table is 384KB (barely
  L2). Compressing to u8 brings it to 96KB (solid L1). Bounded quantization error is
  acceptable for already-approximate PQ distances. Composes with gather (smaller cache
  footprint).

- **AVX-512 gather for cosine**: Two gathers per iteration (IP + norm) with shared index
  vector. Messier but ~2× better than scalar for high chunk counts.

- **Batch quant-quant**: Process multiple (left, right) pairs simultaneously to amortize
  loop overhead and improve ILP.

---

## 10. Implementation Phases

### Phase 1: Mechanical wins (no API design decisions needed)
- Move `ObjectPool` to `diskann-utils`
- Remove redundant deref impls in `dynamic.rs` (use `Accessor::ElementRef`)
- Optimize `pq_dist_lookup_single` (unchecked + unrolled accumulators)
- Move `multi.rs` to CDB internal crate
- Update `PQCompressedData` to use current allocation APIs

### Phase 2: New table types in `diskann-quantization`
- New padded `Table` type for quant-quant SIMD
- Builder function for (TransposedTable, Table) pair
- Move `calculate_chunk_offsets` to `diskann-quantization`
- Cosine lookup table (IP + norm pairs) implementation

### Phase 3: Integration
- `PQScheme` one-stop-shop type (if warranted by usage patterns)
- Pool-aware distance table allocation in `diskann-quantization`
- Migration bridge for `diskann-providers`
- CDB migration to `QueryComputer` pattern (external dependency)

### Phase 4: Advanced optimizations
- AVX-512 gather specialization for lookup
- SQ-compressed lookup tables
- Split `pq_construction.rs`
