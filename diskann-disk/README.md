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

## Search postprocessing tracing

When enabled, each invocation of either disk search postprocessor emits one `DEBUG` span named
`diskann.disk.search.postprocess`, with target `diskann::disk::search::postprocess`.
The span covers the complete async `post_process` method: filtering and cache lookup,
loading missing nodes, recomputing distances, sorting or selecting for diversity, and
writing the postprocessed output. Arguments, vectors, candidates, and results are not
recorded.

The caller's subscriber controls whether this span is enabled. For a consumer using
`tracing_subscriber::EnvFilter`, explicitly disable the target with:

```text
RUST_LOG=info,diskann::disk::search::postprocess=off
```

Or enable it at `DEBUG`:

```text
RUST_LOG=info,diskann::disk::search::postprocess=debug
```

`DEBUG` is not automatically disabled by an unfiltered subscriber. These directives
must be applied by the consumer's subscriber; this library does not install one.
The span is a child of the caller's current span, so reuse an existing search/root
span rather than adding another total-search wrapper.

Configure the subscriber/exporter to record span lifetimes for phase-level elapsed
timing, including any time suspended within the method. This is not an internal
lookup breakdown or true CPU time, and excludes traversal and work outside
`post_process`.

## Dependencies

This crate depends on:

- `diskann`: Core types and utilities
- `diskann-providers`: Main DiskANN library (including storage abstractions)
- `diskann-utils`: Utility functions
- `diskann-vector`: Vector operations
- `diskann-linalg`: Linear algebra operations
- `diskann-quantization`: Vector quantization
