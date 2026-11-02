# DiskANN Disk Index Crate

This crate provides disk-based indexing capabilities for DiskANN.

## Overview

The `disk-index` crate contains all the components specifically needed for building and searching disk-based indices:

## Structure

```text
src/
├── build/             # Disk index building pipeline
│   ├── builder/       # Core disk index builder and logic
│   └── configuration/ # Build parameters and quantization configuration
├── search/            # Disk index search infrastructure
│   ├── provider/      # Disk vertex providers and caching implementations
│   └── traits/        # Core traits for vertex providers and factories
├── data_model/        # Core data structures for disk indices
├── storage/           # Disk I/O operations and quantization
└── utils/             # Disk-specific utilities
```

## Implementation Status

This crate has been populated with the core disk index functionality from the main `diskann` crate. The refactor is complete with the following modules implemented:

### Build Module

- **Builder**: Core disk index builder, quantizer, and build operations
- **Configuration**: Disk build parameters, filter parameters, and quantization types

### Search Module

- **Provider**: Disk vertex providers, caching implementations, and factory patterns
- **Traits**: Core traits for vertex providers and provider factories

### Data Model Module

- Graph headers, metadata, layout versioning, and caching structures

### Storage Module

- Disk I/O operations with reader and writer APIs
- Quantization compression and generation utilities

### Utils Module

- Disk-specific partitioning utilities

## Search quantization

Disk graph traversal uses in-memory quantized vectors, independently of the
quantizer used to build the graph. PQ remains the default: existing build
parameters, `DiskIndexReader`, and `DiskIndexSearcher::new` retain their PQ behavior.

For SphericalQ3 search, set
`DiskIndexBuildParameters::with_search_quantization(SearchQuantization::SphericalQ3)`.
Load the sidecars with `SearchData::load(prefix, SearchQuantization::SphericalQ3, storage)`
and pass them in an `Arc` to `DiskIndexSearcher::new_with_search_data`.
For the provider loading API, use `DiskProvider::load_with_quantization`;
the existing `LoadWith<AsyncDiskLoadContext>` implementation still selects PQ.

`generate_spherical_q3` can also generate sidecars for an existing disk graph.
The input dataset must have exactly the graph's original row order and metric.
It replaces Q3 sidecars but does not modify the graph or PQ sidecars.

Q3 stores its serialized quantizer in `<prefix>_spherical_q3_quantizer.bin`
(an eight-byte envelope: `DSQ3` followed by the source row length as a little-endian u32,
then the quantizer FlatBuffer)
and complete compressed records in `<prefix>_spherical_q3_compressed.bin`.
The latter uses the binary matrix header with the record size in bytes as its
column count; records include compensation metadata. Loading validates the
quantizer schema and bit width, code-file length, record size, graph point count,
source row length, decoded input dimension, and metric. The decoded dimension is
read from a disk graph vector, so encoded representations such as MinMax are
validated independently of their stored row length. Missing or incompatible Q3 data
is an error, not a reason to fall back to PQ. Spherical search supports L2, inner
product, and cosine.

The disk layout, caching, filters, and final reranking are unchanged. Q3 queries
use `QueryLayout::FullPrecision` with search-time rescaling enabled. Queries remain
FP32 after preprocessing; only dataset vectors use Q3 codes. On AVX2-capable x86-64
CPUs, this enables the FP32-by-3-bit FMA kernel. PQ likewise keeps queries in
floating point, computing query-to-centroid lookup tables rather than PQ codes.

In disk-index benchmark JSON, set `"search_quantization": "SphericalQ3"` on either
the `Build` or `Load` source. Omitting this field selects `"PQ"`. The existing
`quantization_type` field still controls graph construction only.
The `num_pq_chunks` field is ignored for SphericalQ3 search.

## Dependencies

This crate depends on:

- `diskann`: Core types and utilities
- `diskann-providers`: Main DiskANN library (including storage abstractions)
- `diskann-utils`: Utility functions
- `diskann-vector`: Vector operations
- `diskann-linalg`: Linear algebra operations
- `diskann-quantization`: Vector quantization
