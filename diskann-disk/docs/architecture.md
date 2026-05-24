# DiskANN Disk Index: Trait & Data Flow Architecture

**Last Updated**: 2026-04-02

This document describes how traits, structs, and strategies interact with vectors, quantized vectors, and adjacency lists in the disk index. It covers both the **build** and **search** paths, using the `diskann-benchmark::backend::disk_index` module as the entry point.

---

## Table of Contents

1. [Overview](#overview)
2. [Index Build](#index-build)
   - [Entry Point](#build-entry-point)
   - [InmemIndexBuilder Trait](#inmemindexbuilder-trait)
   - [Build-Time Data Flow (Insert)](#build-time-data-flow-insert)
   - [Build-Time DataProvider Trait Hierarchy](#build-time-dataprovider-trait-hierarchy)
   - [Build-Time Strategy Traits](#build-time-strategy-traits)
   - [Saving to Disk](#saving-to-disk)
3. [Index Search](#index-search)
   - [Entry Point](#search-entry-point)
   - [DiskProvider (DataProvider for Search)](#diskprovider-dataprovider-for-search)
   - [Search Data Flow](#search-data-flow)
   - [Search Trait Hierarchy](#search-trait-hierarchy)
   - [VertexProvider Trait Chain](#vertexprovider-trait-chain)
   - [PQ Distance Pipeline](#pq-distance-pipeline)
   - [DiskQueryComputer](#diskquerycomputer)
4. [Build vs Search Comparison](#build-vs-search-comparison)
5. [The Unifying Abstraction](#the-unifying-abstraction)

---

## Overview

The disk index architecture separates **build** (in-memory graph construction → disk layout) from **search** (disk-based ANN queries using PQ-compressed data). Both paths share the core `DiskANNIndex` (in `diskann/src/graph/index.rs`) but use different `DataProvider` implementations to access vector data and adjacency lists.

---

## Index Build

### Build Entry Point

`build_disk_index` (in `diskann-benchmark/src/backend/disk_index/build.rs`)
→ `DiskIndexBuilder::new()` (in `diskann-disk/src/build/builder/build.rs`)
→ `DiskIndexBuilder::build()`

#### Key Structs

```text
DiskIndexBuilder<Data, StorageProvider>
  ├── build_quantizer: BuildQuantizer        // decides FP vs quantized in-mem build
  └── core: DiskIndexBuilderCore
        ├── index_writer: DiskIndexWriter    // writes final disk layout
        ├── storage_provider: &StorageProvider
        └── pq_storage: PQStorage            // PQ pivot/compressed file paths
```

### InmemIndexBuilder Trait

`DiskIndexBuilder::build_inmem_index()` selects a strategy (Merged vs OneShot), then delegates to the `InmemIndexBuilder<T>` trait (in `diskann-disk/src/build/builder/inmem_builder.rs`):

```text
┌─────────────────────────────────────────────────────────┐
│              InmemIndexBuilder<T>  (trait)               │
│                                                         │
│  capacity() → usize                                     │
│  total_points() → usize                                 │
│  set_start_point(&[T])                                  │
│  insert_vector(id, &[T]) → Future<ANNResult<()>>        │
│  final_prune(Range<u32>) → Future<ANNResult<()>>        │
│  save_index(DynWriteProvider, metadata)                  │
│  save_graph(DynWriteProvider, start_point_and_path)      │
└─────────────────────────────────────────────────────────┘
         ▲                              ▲
         │                              │
   ┌─────┴──────────┐     ┌────────────┴────────────────┐
   │ DiskANNIndex    │     │ QuantInMemBuilder<T, Q>     │
   │ <FullPrecision  │     │                             │
   │  Provider<T>>   │     │  wraps DiskANNIndex         │
   │                 │     │  <DefaultProvider<NoStore,Q>>│
   └─────────────────┘     └─────────────────────────────┘
```

**Two implementations:**

1. **Full-Precision** (`DiskANNIndex<FullPrecisionProvider<T>>`): Stores raw `T` vectors in memory. Used when the memory budget allows holding all vectors at full fidelity.

2. **Quantized** (`QuantInMemBuilder<T, Q>`): Wraps a `DiskANNIndex<DefaultProvider<NoStore, Q>>`. The `Q` parameter is a quantized vector store (e.g., SQ). Vectors are inserted as `&[T]` but stored in compressed form for reduced memory during build.

### Build-Time Data Flow (Insert)

```text
insert_vector(id, &[T])
  │
  ▼
DiskANNIndex::insert(strategy, context, &id, vector)
  │
  ├── strategy: InsertStrategy<DP, &[T]>
  │     • FullPrecision — stores vector as-is, searches with full vectors
  │     • Quantized — quantizes the vector, searches in quantized space
  │
  ├── strategy.insert_search_accessor(provider, context)
  │     returns: Accessor (impl BuildQueryComputer + NeighborAccessor)
  │       │
  │       ├── build_query_computer(vector) → QueryComputer
  │       │     Preprocesses query for fast distance computation
  │       │
  │       ├── distances_unordered(neighbor_ids, computer, callback)
  │       │     Computes distances to candidate neighbors
  │       │
  │       └── neighbors(id) → neighbor list
  │             Reads adjacency list for graph traversal
  │
  ├── Greedy search finds insertion neighborhood
  │
  └── strategy (as PruneStrategy) prunes edges
        • Writes new adjacency list via NeighborAccessorMut
```

### Build-Time DataProvider Trait Hierarchy

```text
DataProvider (diskann::provider)
  ├── type InternalId = u32
  ├── type ExternalId = u32
  ├── type Context = DefaultContext
  │
  ├── trait Accessor
  │     ├── get_element(id) → vector data
  │     ├── type Id, type GetError
  │     │
  │     ├── trait BuildQueryComputer<Query>
  │     │     ├── build_query_computer(query) → QueryComputer
  │     │     └── distances_unordered(ids, computer, callback)
  │     │
  │     └── trait BuildDistanceComputer
  │           └── build_distance_computer() → DistanceComputer
  │                 (for random-access pairwise distance)
  │
  ├── trait NeighborAccessor
  │     └── neighbors(id) → &[Id]  (read adjacency list)
  │
  ├── trait NeighborAccessorMut
  │     └── set_neighbors(id, &[Id])  (write adjacency list)
  │
  └── trait SetElement<&[T]>
        └── set_element(id, vector)  (store vector data)
```

### Build-Time Strategy Traits

```text
InsertStrategy<DP, &[T]>
  ├── insert_search_accessor() → Accessor
  │     (creates the accessor used during greedy search for insert)
  └── associated PruneStrategy
        └── prune(candidates) → pruned_neighbors
              (decides which edges to keep in adjacency list)

SearchStrategy<DP, Query>  (also used during insert's greedy search)
  ├── type SearchAccessor: Accessor + BuildQueryComputer + NeighborAccessor
  └── search_accessor(provider, context) → SearchAccessor
```

### Saving to Disk

After the in-memory graph is built:

```text
save_index() ──► SaveWith<(u32, AsyncIndexMetadata)>
  │                writes: graph structure, vectors, metadata
  │
save_graph() ──► SaveWith<(u32, DiskGraphOnly)>
                   writes: adjacency lists in disk sector layout
                   (vectors + neighbor lists interleaved for locality)
```

The `DiskIndexWriter` (in `diskann-disk/src/storage`) handles the final disk layout where each sector contains a vertex's vector data adjacent to its neighbor list for cache-friendly disk reads.

---

## Index Search

### Search Entry Point

`search_disk_index` (in `diskann-benchmark/src/backend/disk_index/search.rs`)
→ creates `DiskIndexSearcher` (in `diskann-disk/src/search/provider/disk_provider.rs`)
→ calls `searcher.search()`

#### Key Structs

```text
DiskIndexSearcher<Data, ProviderFactory>
  ├── index: DiskANNIndex<DiskProvider<Data>>     // core graph + search algorithm
  ├── runtime: tokio::Runtime                      // async executor
  ├── vertex_provider_factory: ProviderFactory     // creates per-search vertex readers
  └── scratch_pool: ObjectPool<DiskSearchScratch>  // amortized allocations
```

### DiskProvider (DataProvider for Search)

`DiskProvider<Data>` (in `diskann-disk/src/search/provider/disk_provider.rs`) is the `DataProvider` implementation for disk search. Unlike in-memory providers, it reads vectors and neighbors from disk sectors:

```text
DiskProvider<Data>  (implements DataProvider)
  ├── type InternalId = u32
  ├── type ExternalId = u32
  ├── type Context = DefaultContext
  │
  ├── pq_data: PQData              // PQ codebook + compressed vectors (in memory)
  ├── config: Config                // graph parameters
  ├── starting_points: Vec<u32>    // entry points for search
  └── search_io_limit: usize       // max parallel IO ops
```

### Search Data Flow

```text
DiskIndexSearcher::search(query, return_k, search_l, beam_width, filter, is_flat)
  │
  ▼
search_strategy(query, vector_filter)
  │  Creates DiskSearchStrategy
  │
  ▼
DiskANNIndex::search(search_params, strategy, context, query, output_buffer)
  │
  ├── strategy.search_accessor(DiskProvider, DefaultContext)
  │     │
  │     ▼
  │   DiskAccessor<Data, VertexProvider>
  │     ├── provider: &DiskProvider        // PQ data, config
  │     ├── io_tracker: &IOTracker         // counts IOs, measures time
  │     ├── scratch: DiskSearchScratch     // from pool, avoids alloc
  │     └── vertex_provider: VP            // actual disk reader
  │
  ├── accessor.build_query_computer(query)
  │     │
  │     ▼
  │   DiskQueryComputer
  │     ├── num_pq_chunks: usize
  │     └── query_centroid_l2_distance: Vec<f32>
  │           (precomputed distances from query to PQ centroids)
  │
  ├── GREEDY SEARCH LOOP:
  │     │
  │     ├── accessor.starting_points() → Vec<u32>
  │     │     entry nodes for beam search
  │     │
  │     ├── accessor.distances_unordered(candidate_ids, computer, callback)
  │     │     │
  │     │     │  For each candidate:
  │     │     │  1. vertex_provider reads disk sector → gets PQ-compressed vector
  │     │     │     + neighbor list
  │     │     │  2. DiskQueryComputer computes approximate distance using PQ
  │     │     │     lookup table
  │     │     │  3. callback(distance, id) feeds into best-first queue
  │     │     │
  │     │     └── IO is batched (beam_width sectors read in parallel)
  │     │
  │     ├── accessor.neighbors(id) → neighbor IDs
  │     │     extracted from same disk sector that was read for the vector
  │     │     (this is why disk layout interleaves vectors + adjacency lists)
  │     │
  │     └── ExpandBeam trait (implemented by DiskAccessor)
  │           manages beam search expansion with IO batching
  │
  └── output_buffer receives final top-k results
```

### Search Trait Hierarchy

```text
SearchStrategy<DiskProvider<Data>, &[Data::VectorDataType]>
  │
  └── DiskSearchStrategy<Data, ProviderFactory>
        ├── io_tracker: IOTracker
        ├── vector_filter: &dyn Fn(&u32) → bool
        ├── query: &[VectorDataType]
        ├── vertex_provider_factory: &ProviderFactory
        └── scratch_pool: &ObjectPool
        │
        └── search_accessor() → DiskAccessor
```

```text
DiskAccessor<Data, VP>  (the core search accessor)
  │
  ├── impl Accessor
  │     └── get_element(id) — reads from disk via VertexProvider
  │
  ├── impl BuildQueryComputer<&[VectorDataType]>
  │     ├── build_query_computer() → DiskQueryComputer
  │     │     Preprocesses query against PQ centroids (quantizer_preprocess)
  │     │     producing a lookup table for fast approximate distance
  │     │
  │     └── distances_unordered(ids, computer, callback)
  │           Batch-reads sectors from disk, computes PQ distances
  │
  ├── impl NeighborAccessor
  │     └── neighbors(id) — extracts adjacency list from disk sector
  │
  └── impl ExpandBeam<&[VectorDataType]>
        └── Controls beam search IO batching
```

### VertexProvider Trait Chain

```text
VertexProviderFactory<Data>  (trait)
  │  Creates per-search VertexProvider instances
  │
  └── DiskVertexProviderFactory<Data, ReaderFactory>
        ├── ReaderFactory: AlignedReaderFactory (creates aligned IO readers)
        ├── caching_strategy: CachingStrategy
        │     • None — all reads from disk
        │     • StaticCacheWithBfsNodes(n) — cache n nodes closest to start
        │
        └── creates → VertexProvider<Data>  (trait)
                         │
                         ├── read_vertex(id) → (vector_data, neighbor_list,
                         │                       associated_data)
                         │     Reads a disk sector containing interleaved data
                         │
                         └── May use cached sectors or aligned IO reads
```

### PQ Distance Pipeline

```text
Query Vector: &[f32]  (full precision)
         │
         ▼
quantizer_preprocess(query, pq_pivots, pq_tables)
         │  Computes distance from query to each PQ centroid
         ▼
DiskQueryComputer.query_centroid_l2_distance: Vec<f32>
         │  This is a lookup table: for each chunk c and centroid k,
         │  stores dist(query_chunk_c, centroid_k)
         │
         ▼
compute_pq_distance(pq_compressed_vector, lookup_table) → f32
         │  For each PQ chunk, looks up the centroid assignment
         │  in the compressed vector and sums precomputed distances
         │  O(num_pq_chunks) per candidate — very fast
         │
         ▼
Approximate distance used for beam search ranking
```

### DiskQueryComputer

`DiskQueryComputer` (in `diskann-disk/src/search/provider/disk_provider.rs`) implements `PreprocessedDistanceFunction<&[u8], f32>`:

```text
impl PreprocessedDistanceFunction<&[u8], f32> for DiskQueryComputer {
    // Takes PQ-compressed bytes, returns approximate distance
    // using the precomputed centroid distance lookup table
}
```

This is the key abstraction that lets the graph search algorithm remain generic — it calls `computer.distance(compressed_vec)` without knowing about PQ internals.

---

## Build vs Search Comparison

| Aspect | Build | Search |
|--------|-------|--------|
| **DataProvider** | `FullPrecisionProvider<T>` or `DefaultProvider<NoStore, Q>` | `DiskProvider<Data>` |
| **Vector Storage** | In-memory (`Vec<T>` or quantized store) | Disk sectors (read via `VertexProvider`) |
| **Adjacency Lists** | In-memory graph (`Vec<Vec<u32>>`) | Interleaved in disk sectors |
| **Distance** | Exact (full precision or SQ) | Approximate (PQ lookup table) |
| **Strategy** | `InsertStrategy` + `PruneStrategy` | `SearchStrategy` only |
| **Accessor** | `inmem::Accessor` variants | `DiskAccessor<Data, VP>` |
| **IO** | Memory reads | Batched aligned disk reads (beam_width) |
| **Key Trait** | `InmemIndexBuilder<T>` | `VertexProviderFactory` + `VertexProvider` |

---

## The Unifying Abstraction

Both build and search go through `DiskANNIndex::search()` (in `diskann/src/graph/index.rs`) for greedy graph traversal. The `Search` trait + `SearchStrategy` + `Accessor` hierarchy makes this possible:

```text
DiskANNIndex<DP: DataProvider>::search(params, strategy, context, query, output)
     │
     │  DP = FullPrecisionProvider<T>  →  in-memory build search
     │  DP = DefaultProvider<NoStore,Q> →  quantized build search
     │  DP = DiskProvider<Data>         →  disk-based ANN search
     │
     └── All three go through the same greedy best-first algorithm,
         differing only in how vectors are fetched and distances computed.
```

The core insight is that `DataProvider` + `Accessor` + `Strategy` form a pluggable backend system. The graph algorithm (`DiskANNIndex`) is completely agnostic to:
- Whether vectors are in memory or on disk
- Whether distances are exact or approximate
- Whether adjacency lists are in `Vec`s or packed into disk sectors

This separation is what allows the same greedy search implementation to power both in-memory graph construction and disk-based approximate nearest neighbor queries.
