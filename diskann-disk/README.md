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

## Search phase tracing

The caller's subscriber can enable these `DEBUG` spans. Each phase invocation emits
one span, with no arguments, vectors, candidates, or results recorded:

| Target | Span name | Boundary |
| --- | --- | --- |
| `diskann::graph::search::traversal` | `diskann.graph.search.traversal` | Complete shared graph traversal, including start-point distances, node loading, full/PQ distance work, beam/neighbor handling and candidate queues. Inline-filter traversal also includes matching, adaptive-L updates and sorting matched candidates. |
| `diskann::flat::search::scan` | `diskann.flat.search.scan` | The flat visitor's distance scan and insertion into the top-k queue. |
| `diskann::disk::search::postprocess` | `diskann.disk.search.postprocess` | Complete async `post_process`: filtering/cache lookup, loading missing nodes, recomputing distances, sorting or diversity selection, and writing postprocessed output. |

Graph, inline-filter and diverse-graph searches produce a traversal span followed
by a postprocess span; flat searches produce a scan span followed by a postprocess
span. The phase ends before postprocessing starts. Accessor/query preprocessing,
starting-point count lookup and outer scratch/queue setup are outside traversal/scan;
result packaging after `post_process` is outside these spans.

The traversal and scan targets belong to the shared `diskann` crate and also apply
to non-disk users of those routines, including repeated invocations during paged
search or construction. They require its `tracing` feature (enabled by default,
including by this crate's existing dependency).

For a consumer using `tracing_subscriber::EnvFilter`, explicitly disable all three
targets with:

```text
RUST_LOG=info,diskann::graph::search::traversal=off,diskann::flat::search::scan=off,diskann::disk::search::postprocess=off
```

Or enable it at `DEBUG`:

```text
RUST_LOG=info,diskann::graph::search::traversal=debug,diskann::flat::search::scan=debug,diskann::disk::search::postprocess=debug
```

`DEBUG` is not automatically disabled by an unfiltered subscriber. These directives
must be applied by the consumer's subscriber; this library does not install one.
These are sibling spans under the caller's current span, so reuse an existing
search/root span rather than adding another total-search wrapper.

Configure the subscriber/exporter to record span lifetimes for phase-level elapsed
timing, including any time suspended within the method. This is not an internal
lookup breakdown or true CPU time. The phases do not account for all work in a search.

## Dependencies

This crate depends on:

- `diskann`: Core types and utilities
- `diskann-providers`: Main DiskANN library (including storage abstractions)
- `diskann-utils`: Utility functions
- `diskann-vector`: Vector operations
- `diskann-linalg`: Linear algebra operations
- `diskann-quantization`: Vector quantization
