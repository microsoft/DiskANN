# Multi-Vector Distance Functions

| | |
|---|---|
| **Status** | InReview |
| **Authors** | Suryansh Gupta |
| **Contributors** | Suryansh Gupta, Mark Hildebrand |
| **Created** | 2026-01-06 |
| **Updated** | 2026-02-06 |

## Summary

This RFC proposes a high-performance Chamfer distance implementation for multi-vector (ColBERT-style late interaction) representations in DiskANN. The design uses a **query-transposed tiling** approach that transposes queries into a block layout while keeping documents in row-major format, achieving up to **2.67x speedup** over SIMD baseline. The implementation builds on existing types from `diskann-quantization` (`Mat`, `MaxSim`, `Chamfer`) and implements `DistanceFunctionMut` from `diskann-vector` for ecosystem compatibility.

## Motivation

### Background

Traditional vector search represents each document as a single embedding. Multi-vector representations (used in models like ColBERT) encode each document/query as a **bag of embeddings** — typically one per token. This enables:

- **Fine-grained matching**: Token-level similarity captures nuanced semantic relationships
- **Late interaction**: Document embeddings are pre-computed; only lightweight aggregation at query time
- **Better recall**: If any query token matches any document token well, the document scores high

### Problem Statement

Chamfer distance for multi-vector search requires O(Q × D × Dim) operations per query-document pair, where:

- Q = number of query tokens
- D = number of document tokens
- Dim = embedding dimensionality

For typical configurations (Q=32, D=128, Dim=384), this is ~1.5M floating-point operations per pair. Naive implementations become a bottleneck for large-scale search.

### Goals

1. Implement high-performance Chamfer distance starting with `f32` embeddings, with future support for `f16` and `u8` types
2. Achieve 2x+ speedup over baseline SIMD through memory layout optimization
3. Maintain compatibility with DiskANN's `DistanceFunctionMut` trait
4. Provide a clean API that enables standalone distance function usage without full index integration
5. Achieve performance within 10–20% of `faer` SGEMM-based Chamfer computation, when both our implementation and `faer` are restricted to AVX2 (no AVX-512 on either side)

## Proposal

### Approach: Query-Transposed Tiling

We propose the **query-transposed tiling** approach as the primary Chamfer distance implementation for DiskANN integration. This approach transposes the query into a block-transposed layout, keeps documents in row-major format, and processes pairs of document vectors together to amortize query memory loads. A pre-allocated scratch buffer tracks per-query max similarities and is reused across distance calls.

This is the recommended default because it preserves the existing document storage format (no index migration), while still achieving significant speedups through SIMD tiling.

### Chamfer Distance Definition

For query multi-vector Q and document multi-vector D:

```
Chamfer(Q, D) = Σᵢ minⱼ -IP(qᵢ, dⱼ)
```

Since `InnerProduct::evaluate` in `diskann-vector` returns the negated inner product (`-IP`), the kernel finds the minimum negated IP per query vector (equivalent to finding the maximum similarity), then sums across all query vectors. The result is a distance compatible with DiskANN's min-heap.

### Types

The design builds on the multi-vector matrix types already defined in `diskann_quantization::multi_vector`:

#### Query and Document (from `diskann-quantization`)

```rust
use diskann_quantization::multi_vector::{Mat, MatRef, Standard, QueryMatRef};

/// Owning row-major matrix: rows = tokens, cols = dimensions
type MultiVector = Mat<Standard<f32>>;

/// Borrowed view into a multi-vector
type MultiVectorRef<'a> = MatRef<'a, Standard<f32>>;
```

`Standard<f32>` provides contiguous row-major storage with `as_slice()` for BLAS compatibility and zero-copy views via `MatRef`. `QueryMatRef` (a newtype over `MatRef`) distinguishes query from document matrices for asymmetric distance functions.

#### TransposedMultiVector (new type)

A block-transposed representation for SIMD-optimized inner product computation. The full transposed type will be added in separate PRs; the below is a simplified depiction.

This will implement a `Repr` trait similar to `Standard<T>` defined in `diskann_quantization::multi_vector`, backed by a block-transposed layout analogous to `BlockTranspose<16>`:

```rust
/// Block-transposed layout for SIMD-parallel inner products.
/// Groups 16 vectors and stores dimensions contiguously:
///
///   Block 0: [v0_d0..v15_d0], [v0_d1..v15_d1], ...
///   Block 1: [v16_d0..v31_d0], ...
///
/// This enables loading 8 values (f32x8) from 8 different vectors
/// in a single SIMD load.
pub struct TransposedMultiVector { ... }

impl TransposedMultiVector {
    pub fn from_view(view: MatRef<'_, Standard<f32>>) -> Self;
    pub fn num_vectors(&self) -> usize;
    pub fn vector_dim(&self) -> usize;
}

impl From<&MultiVector> for TransposedMultiVector { ... }
```

#### MaxSim (from `diskann-quantization`)

The existing `MaxSim` type in `diskann_quantization::multi_vector::distance` provides a mutable buffer for per-query-vector maximum similarities. The query-transposed tiling approach uses an analogous scratch pattern internally:

```rust
/// Per-query-vector maximum similarity scores.
pub struct MaxSim<'a> {
    scores: &'a mut [f32],
}
```

#### Chamfer (from `diskann-quantization`)

The existing unit type `Chamfer` in `diskann_quantization::multi_vector::distance` identifies the Chamfer distance function:

```rust
/// Unit type identifying the Chamfer (MaxSim) distance function.
pub struct Chamfer;

impl PureDistanceFunction for Chamfer { ... }
```

`Chamfer` implements `PureDistanceFunction` from `diskann-vector`, enabling it to be used as a distance identifier throughout the DiskANN ecosystem without carrying computation state.

### Algorithm

The query-transposed tiling approach:

1. Transpose the query into block-transposed layout (once, before distance calls)
2. For each pair of document vectors `(d1, d2)`:
   - Load each query block once, compute inner products against both documents simultaneously
   - Update per-query min negated IPs in the score array in `MaxSim` struct
3. Handle remainder if document count is odd
4. Sum scores to produce the final Chamfer distance

```
// `scores` corresponds to the scores array inside MaxSim
// InnerProduct::evaluate returns -IP(q, d), so we track min (= best match)
scores = [f32::MAX; num_queries]  // reused across calls

for i in (0..num_docs step 2):
    d1, d2 = doc[i], doc[i+1]
    for each query block:
        // Load query block ONCE, use for BOTH documents
        for each query vector q:
            scores[q] = min(scores[q], -IP(q, d1))
            scores[q] = min(scores[q], -IP(q, d2))

if num_docs is odd:
    for each query vector q:
        scores[q] = min(scores[q], -IP(q, last_doc))

score = sum(scores[i] for i in 0..num_queries)
```

**Key optimizations**:

- Query blocks loaded once, reused for 2 document vectors — reduces memory bandwidth by ~50%
- Scratch buffer stored in `MaxSim` struct and reused across distance calls — zero allocation on hot path
- 4 SIMD accumulators per document vector to hide FMA latency
- Uses `SIMDMinMax` from `diskann-wide` for hardware-accelerated SIMD reductions

### Trait Implementation

`MaxSim` implements `DistanceFunctionMut` to populate its per-query-vector max similarities. The caller then sums the scores to produce the Chamfer distance. We propose adding a `chamfer_score()` convenience method on `MaxSim` for this:

```rust
// DistanceFunctionMut populates the MaxSim scores buffer with per-query max similarities
impl DistanceFunctionMut<&TransposedMultiVector, MatRef<'_, Standard<f32>>, Result<(), MaxSimError>> for MaxSim<'_> {
    fn evaluate(&mut self, query: &TransposedMultiVector, doc: MatRef<'_, Standard<f32>>) -> Result<(), MaxSimError>;
}

impl MaxSim<'_> {
    /// Sums the per-query min negated IPs to produce the Chamfer distance.
    /// Call after `evaluate()` has populated the scores buffer.
    ///
    /// Returns: Σᵢ scores[i]  (each score is already minⱼ -IP(qᵢ, dⱼ))
    pub fn chamfer_score(&self) -> f32 {
        self.scores.iter().sum()
    }

    /// Resets all scores to `f32::MAX` for the next document comparison.
    pub fn reset(&mut self) {
        self.scores.fill(f32::MAX);
    }
}
```

**Thread safety**: `MaxSim` holds `&mut [f32]`, so a `DistanceFunctionMut` implementation borrowing it is `!Sync` by construction. For concurrent search, each thread should own its own `MaxSim` buffer.

### API Usage

The end-to-end flow converts a row-major query into block-transposed layout, then uses `MaxSim` for repeated distance computations against document vectors:

```rust
use diskann_quantization::multi_vector::{Mat, MatRef, Standard};
use diskann_quantization::multi_vector::distance::MaxSim;
use diskann_vector::DistanceFunctionMut;

// 1. Query arrives as row-major multi-vector (e.g., 32 tokens × 384 dims)
let query: Mat<Standard<f32>> = /* from encoder/upstream layers */;

// 2. Transpose query into SIMD-friendly block layout (once per query)
let transposed_query = TransposedMultiVector::from_view(query.as_ref());

// 3. Create MaxSim buffer (reused across all document comparisons for this query)
let mut buffer = vec![f32::MAX; query.num_vectors()];
let mut max_sim = MaxSim::new(&mut buffer).unwrap();

// 4. For each candidate document:
for doc in candidate_documents {
    // Populate per-query-vector max similarities
    max_sim.evaluate(&transposed_query, doc.as_ref());

    // Sum to get final Chamfer distance
    let score = max_sim.chamfer_score();

    // Reset for next document
    max_sim.reset();
}
```

The `TransposedMultiVector::from_view` conversion is O(Q × Dim) — a single pass that rearranges the query into block-transposed layout. This cost is amortized over all document comparisons for that query, making it negligible in practice (typically thousands of comparisons per query).

## Trade-offs

### Alternative Approaches

The experimental crate explored six approaches total. The query-transposed tiling approach was selected as the proposal, but the alternatives remain available and may be better for specific workloads.

#### Document-Transposed Approaches

Instead of transposing the query, documents can be block-transposed at index time. This is implemented as `TransposedApproach` and `TransposedWithTilingApproach`.

| Aspect | Query-Transposed (Proposed) | Document-Transposed |
|--------|---------------------------|---------------------|
| **Document Layout** | Row-major (no change) | Block-transposed |
| **Query Layout** | Block-transposed (once per query) | Row-major |
| **Index Migration** | None | Required |
| **Hot Path Allocation** | None (scratch reused) | None |
| **Best For** | Many query tokens (≥16) | Few query tokens (≤8) |

**Backwards Compatibility**: Locking documents into a transposed layout creates backward compatibility obligations — if we later discover a better layout (e.g., for AVX-512), we're stuck supporting the legacy format. Query transpositions are ephemeral and not persisted, so the query-side layout can evolve freely.

#### SGEMM Approach

Uses BLAS matrix multiplication to compute the full Q×D similarity matrix, then SIMD row-max reduction. Dominates at large scale (≥32×128) with up to 2.16x speedup, but materializes the full similarity matrix (Q×D×4 bytes). Custom tiling approaches fuse max-reduction with dot product, avoiding this materialization, which makes them faster for small/medium Q×D.

#### Baseline Approaches

- **NaiveApproach**: Scalar O(n²) implementation for correctness verification
- **SimdApproach**: SIMD-accelerated inner product via `diskann_vector::InnerProduct`, iterating documents sequentially (1.0x baseline)

## Benchmark Results

**Machine:** Intel Core i7-1365U, AVX2 supported, AVX-512 not supported, 32 GB RAM

Median over 50 measurements. Each measurement computes 10 consecutive distance evaluations across 100 points.

### Speedup vs SIMD Baseline (Median, Lower Latency = Better)

| Configuration | SIMD (µs) | transposed_simd | transposed_tiling | query_transposed_tiling | sgemm |
|--------------|-----------|-----------------|-------------------|------------------------|-------|
| dim=128, doc=32, query=8 | 2,237 | 1.34x | **1.75x** | 1.05x | 1.13x |
| dim=128, doc=64, query=16 | 9,224 | 1.42x | 2.07x | **2.35x** | 1.48x |
| dim=128, doc=128, query=32 | 47,882 | 1.32x | 1.86x | **2.64x** | 1.88x |
| dim=256, doc=32, query=8 | 4,654 | 1.26x | **1.69x** | 1.13x | 0.96x |
| dim=256, doc=64, query=16 | 25,809 | 1.56x | 1.94x | **2.40x** | 1.87x |
| dim=256, doc=128, query=32 | 101,093 | 1.41x | 1.71x | **2.67x** | 1.96x |
| dim=256, doc=16, query=32 | 8,239 | 1.22x | 1.77x | **2.02x** | 1.57x |
| dim=384, doc=32, query=8 | 8,412 | 1.41x | **1.65x** | 1.30x | 1.24x |
| dim=384, doc=64, query=16 | 38,162 | 1.30x | 1.47x | **1.70x** | 1.66x |
| dim=384, doc=128, query=32 | 171,431 | 1.53x | 1.94x | 2.04x | **2.16x** |
| **Average** | — | **1.38x** | **1.79x** | **1.93x** | **1.59x** |

### Analysis

- **query_transposed_tiling** is the best overall approach (avg **1.93x**), winning 7 of 10 configurations
- **transposed_tiling** wins for small query counts (≤8 tokens) with up to **1.75x** speedup
- **sgemm** catches up at large scale (dim=384, 32×128) where BLAS cache-blocking dominates (**2.16x**)
- All tiling approaches exceed the **2x speedup** goal at ≥16×64 token configurations

## Future Work

- [ ] **FFI types**: View types for `MultiVector` / `TransposedMultiVector` for C/C++ callers
- [ ] **Integration into `diskann` crate**: Graduate from `experimental-` prefix into the main library with a clean API and appropriate module structure
- [ ] **Quantized types**: `f16`, `u8` support for memory efficiency
- [ ] **AVX-512 support**: Larger registers could enable tile size 4 (processing 4 queries simultaneously)
- [ ] **SIMD-accelerated horizontal MinMax**: Hardware-accelerated horizontal min/max reductions across SIMD lanes for faster per-query score aggregation

## References

1. [ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction](https://arxiv.org/abs/2004.12832)
2. [Chamfer Distance](https://en.wikipedia.org/wiki/Chamfer_distance)
3. Experimental implementation: [experimental-multi-vector-bench crate (PR #730)](https://github.com/microsoft/DiskANN/pull/730)
