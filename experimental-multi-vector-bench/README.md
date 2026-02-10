# experimental-multi-vector-bench

Experimental multi-vector benchmarking support for DiskANN, enabling late interaction retrieval with token-level embeddings.

## Scope & Goals

This crate is an **experimental workspace** focused on:

1. **Fast Chamfer distance implementation for `f32`** - Develop and benchmark high-performance implementations of the Chamfer distance function for multi-vector representations using 32-bit floating point values.

2. **Multiple computation approaches** - Compare naive scalar, SIMD-accelerated, transposed, tiling, and SGEMM implementations to quantify performance gains.

3. **Benchmarking infrastructure** - Provide tooling to measure and compare different implementation strategies.

## Current Status

- ✅ `MultiVector` type alias for `Mat<Standard<f32>>` (row-major storage from diskann-quantization)
- ✅ `TransposedMultiVector` type for block-transposed storage (SIMD-optimized)
- ✅ `Chamfer<Approach>` - Generic distance calculator using Inner Product similarity
- ✅ `Chamfer<NaiveApproach>` - Scalar baseline implementation
- ✅ `Chamfer<SimdApproach>` - SIMD-accelerated implementation
- ✅ `Chamfer<TransposedApproach>` - Block-transposed SIMD with transposed documents
- ✅ `Chamfer<TransposedWithTilingApproach>` - Block-transposed SIMD with query pair tiling
- ✅ `Chamfer<QueryTransposedWithTilingApproach>` - Transposed query with doc pair tiling
- ✅ `Chamfer<SgemmApproach>` - BLAS SGEMM + SIMD row-max
- ✅ Implements `diskann_vector::DistanceFunction` trait for ecosystem compatibility
- ✅ Benchmark utility integrated with diskann-benchmark-runner

## Usage

```rust
use experimental_multi_vector_bench::{
    Chamfer, SimdApproach, TransposedWithTilingApproach, QueryTransposedWithTilingApproach,
    MultiVector, TransposedMultiVector, Standard,
};
use diskann_vector::DistanceFunction;

// Create a multi-vector (3 vectors of dimension 4)
let mv = MultiVector::new(Standard::new(3, 4), 0.0f32).unwrap();

// Basic usage with row-major vectors (NaiveApproach or SimdApproach)
let chamfer = Chamfer::<SimdApproach>::new();
let distance = chamfer.evaluate_similarity(&query, &document);

// Optimized for few query tokens (≤8): transpose documents
let chamfer = Chamfer::<TransposedWithTilingApproach>::new();
let transposed_doc = TransposedMultiVector::from(&document);
let distance = chamfer.evaluate_similarity(&query, &transposed_doc);

// Optimized for many query tokens (≥16): transpose query instead
let chamfer = Chamfer::<QueryTransposedWithTilingApproach>::new();
let transposed_query = TransposedMultiVector::from(&query);
let distance = chamfer.evaluate_similarity(&transposed_query, &document);

// For large Q×D: use SGEMM
use experimental_multi_vector_bench::{SgemmApproach, SgemmScratch};
let chamfer = Chamfer::<SgemmApproach>::new();
let mut scratch = SgemmScratch::new();
let distance = chamfer.evaluate_similarity_with_scratch(&query, &document, &mut scratch);
```

## Type Aliases

This crate uses shared types from `diskann-quantization` for multi-vector representation:

```rust
// Row-major owning matrix
pub type MultiVector = Mat<Standard<f32>>;

// Immutable view
pub type MultiVectorRef<'a> = MatRef<'a, Standard<f32>>;
```

The `Standard<f32>` representation provides:

- Contiguous row-major storage
- Direct `as_slice()` access for BLAS operations
- Zero-copy views via `MatRef`

## Future Work

- [ ] Add RFC based on findings for DiskANN integration
- [ ] Additional similarity measures (Cosine, SquaredL2)
- [ ] Support for additional element types (`f16`, `u8` quantized, etc.)

## Running Benchmarks

```bash
# Run benchmarks with example configuration
cargo run --release -p experimental-multi-vector-bench --bin multivec-bench -- run \
    --input-file experimental-multi-vector-bench/examples/bench.json \
    --output-file results.json

# Verify correctness (all approaches should produce same checksum)
cargo run --release -p experimental-multi-vector-bench --bin multivec-bench -- run \
    --input-file experimental-multi-vector-bench/examples/verify.json \
    --output-file verify_results.json
```

See [examples/bench.json](examples/bench.json) for benchmark configuration format.

### Benchmark Configuration

The benchmark supports six approaches via the `approach` field:

- `"naive"` - Scalar baseline
- `"simd"` - SIMD-accelerated
- `"transposed_simd"` - Block-transposed SIMD
- `"transposed_with_tiling"` - Block-transposed SIMD with query pair tiling
- `"query_transposed_with_tiling"` - Transposed query with doc pair tiling
- `"sgemm"` - BLAS SGEMM + SIMD row-max

## Module Structure

```text
src/
├── lib.rs                       # Crate root with re-exports and type aliases
├── multi_vector.rs              # TransposedMultiVector type (block-transposed storage)
├── distance/
│   ├── mod.rs                   # Chamfer<Approach> generic struct
│   ├── naive.rs                 # Scalar implementation (NaiveApproach)
│   ├── simd.rs                  # SIMD-accelerated (SimdApproach)
│   ├── transposed.rs            # Transposed docs (TransposedApproach)
│   ├── transposed_tiling.rs     # Transposed docs + query tiling (TransposedWithTilingApproach)
│   ├── query_transposed_tiling.rs # Transposed query + doc tiling (QueryTransposedWithTilingApproach)
│   └── sgemm.rs                 # BLAS SGEMM + row-max (SgemmApproach)
└── bench/
    ├── mod.rs                   # Benchmark registration and dispatch
    ├── input.rs                 # Benchmark input types
    └── runner.rs                # Benchmark execution logic
```

## Contributing

This work is experimental and will be submitted as separate PRs.
