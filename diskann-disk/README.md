# DiskANN Disk Index Crate

This crate provides disk-based indexing capabilities for DiskANN.

## Optional search diagnostics

The DEBUG span `diskann.disk.search.diagnostics`, on target
`diskann::disk::search::diagnostics`, records one aggregate per search. It is a child
of the caller's current span when the search is polled. Both search entry points
and all four search modes use the same counters. Fields are recorded before the
span closes, including partial totals if a search fails or is cancelled.

| Field | Cumulative elapsed nanoseconds |
| --- | --- |
| `vector_ns` | Full-vector distance, associated-data access, per-query distance-cache insertion, and optional indexed-vector copying/output. Includes full-vector loops in flat reranking and diversity preparation, but not sorting or the diversity algorithm. |
| `pq_distance_ns` | `compute_pq_distance` only, excluding its result callback and query preprocessing. |
| `adjacency_ns` | Adjacency enumeration and the visited-set predicate, including temporary neighbor-ID collection. Zero for flat scans; excludes flat-scan filtering and inline-label decisions. |

These are disjoint **wall-time** accumulations, not thread CPU measurements or
continuous child spans. They exclude node loading (already covered by existing
I/O statistics), setup, frontier selection, batch priority-queue insertion,
postprocessing outside the listed regions, and result cleanup. Queue work lives
in the shared graph algorithms, outside the disk accessor; measuring it separately
would require additional graph instrumentation. Do not interpret the unmeasured
remainder as queue time or CPU time. Existing query PQ preprocessing and I/O
statistics are unchanged.

The library installs no subscriber. An unfiltered subscriber enables DEBUG too;
applications should explicitly disable this target unless diagnostics are wanted.
For example, using `tracing-subscriber`'s `registry`/`std` support and an existing
collection layer (no `env-filter` feature required):

```rust,ignore
use tracing_subscriber::{filter::{LevelFilter, Targets}, prelude::*, reload};

let targets = |enabled| Targets::new()
    .with_default(LevelFilter::TRACE) // preserve all other targets
    .with_target("diskann::disk::search::diagnostics",
        if enabled { LevelFilter::DEBUG } else { LevelFilter::OFF });
let (filter, handle) = reload::Layer::new(targets(false));
let subscriber = tracing_subscriber::registry().with(filter).with(collection_layer);
tracing::subscriber::set_global_default(subscriber)?;
// Later, for subsequent searches:
handle.reload(targets(true))?;
```

Attach the filter before all collection layers; no layer should bypass it.
`Targets` uses prefix matching. Without `reload`, changing startup configuration
requires a restart. Each search keeps its initial span/filter decision; no
enabled state is cached across searches. With the target disabled, the added hot
path performs span-disabled checks, but no diagnostic clock reads, counter
updates, allocations, or locks. Subscriber/filter checks at span creation remain.
Collectors must retain fields from `on_record`, not just span creation. Span
lifetime includes unmeasured work; the three fields, not that lifetime, give the
breakdown. An event-only collector will not receive these fields.

Searches run on the caller's executor; the searcher does not own a runtime or
spawn tasks. If moving a search to another task/thread, propagate both subscriber
and parent using future instrumentation rather than an enter guard across await:

```rust,ignore
use tracing::{Instrument, instrument::WithSubscriber};
let task = async move { searcher.search(&query, 10, 20, Some(4), mode).await }
    .in_current_span()
    .with_current_subscriber();
let result = tokio::spawn(task).await??;
```

For a reproducible release microbenchmark using the checked-in 256-vector,
128-dimensional fixture, run
`cargo run --release -p diskann-disk --example search_diagnostics -- off cached 100000`.
Compare `off`/`on` and `cached`/`storage` with repeated, reordered runs. The example
uses the native disk reader, warms scratch and storage with 1,000 queries, and
formats span-close output to a sink (no export). `cached` requests a static BFS
cache of up to 256 nodes; a preflight checks that all medoid-reachable nodes can
be visited without native reads. `storage` disables that application cache and
verifies native reads occur. Neither is a cold-device benchmark. CSV includes
end-to-end query latency percentiles and QPS.

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

## Dependencies

This crate depends on:

- `diskann`: Core types and utilities
- `diskann-providers`: Main DiskANN library (including storage abstractions)
- `diskann-utils`: Utility functions
- `diskann-vector`: Vector operations
- `diskann-linalg`: Linear algebra operations
- `diskann-quantization`: Vector quantization
