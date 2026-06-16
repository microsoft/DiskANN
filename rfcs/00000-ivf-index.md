# Inverted File (IVF) Index

|                  |                                |
|------------------|--------------------------------|
| **Authors**      | Aditya Krishnan                |
| **Created**      | 2026-06-10                     |
| **Updated**      | 2026-06-10                     |
| **Status**       | Draft — requirements gathering |

> **Note**: This is a *requirements* document. It deliberately stays at the
> level of "what we need and why" and defers concrete type/API design to a
> follow-up RFC. Sections are intentionally terse so we can iterate quickly.

## 1. Summary

Add an **Inverted File (IVF)** index to DiskANN. An IVF index partitions the
dataset into a fixed number of *inverted lists* (a.k.a. cells / clusters), each
summarized by a **centroid**. Each list stores the representations (full
precision or quantized) of the points assigned to it, laid out contiguously.

- **Insert**: assign a point to the list whose centroid is closest, then append
  the point's representation to that list.
- **Search**: select the `nprobe` lists whose centroids are closest to the
  query, score the query against every point in those lists, and return the
  top-`k` via the neighbor priority queue.

## 2. Motivation

### 2.1 Background

DiskANN today exposes two index families:

- **Graph search** — random-access greedy search over a proximity graph
  ([`crate::graph::DiskANNIndex`], driven by the [`crate::provider::Accessor`]
  trait).
- **Flat search** — sequential one-pass scan-and-score
  ([`crate::flat::FlatIndex`], driven by [`crate::flat::DistancesUnordered`]).

There is currently **no clustering / coarse-quantizer based index**. IVF is the
canonical such structure and a well-understood baseline in the ANN literature
(IVF, IVFPQ, etc.).

### 2.2 Problem Statement

We want a partition-based index that:

- Trades recall for latency via a single tunable knob (`nprobe`).
- Has predictable, scan-friendly memory layout (contiguous lists) that maps
  cleanly onto the existing flat-search scan-and-score primitives.
- Composes with the existing quantization stack so lists can store quantized
  representations.

### 2.3 Goals

1. Define an IVF index that supports **build**, **insert**, and **top-k
   search** with an `nprobe` parameter.
2. **Reuse existing primitives** rather than inventing parallel machinery:
   - [`crate::provider::DataProvider`] for id mapping / context.
   - The flat-search scan-and-score path ([`DistancesUnordered`]) for scoring
     points inside a list.
   - [`crate::neighbor`] priority queue for top-k accumulation.
   - The quantization crate for list representations.
3. Support both **full-precision** and **quantized** list representations behind
   a common representation abstraction.
4. Make the **coarse quantizer** (centroid set + assignment) a pluggable
   component so we can later swap k-means for alternatives. 
5. Make the representation and access of inverted lists to be pluggable 
   so it works with multiple data-backends (disk, k-v table, caching, blob). 

### 2.4 Non-Goals (for the first cut)

- IVFPQ residual encoding / per-list re-ranking (future work).
- Deletes and compaction.

## 3. Requirements

### 3.1 Functional

| ID | Requirement |
|----|-------------|
| F1 | **Build** an index from a dataset: derive `n_lists` centroids (initially k-means) and assign every point to its nearest centroid's list. |
| F2 | **Insert** a single point: find nearest inverted list (start by using centroids), append representation to that list. Insert must not require a full rebuild. |
| F3 | **Search**: given a query and parameters `k` and `nprobe`, select the `nprobe` nearest lists by centroid distance, scan-and-score their members, return top-`k`. List scoring should reuse the flat-search scan-and-score primitive [`DistancesUnordered`]; no bespoke distance loop. | |
| F4 | Lists can store any representation of vectors (e.g. **full-precision**, **quantized**), selected at build time. |
| F5 | The **coarse quantizer** (centroid training + assignment) is a distinct, swappable component. |
| F6 | The representation of inverted lists should be swappable, consumers should be able to implement it for different backends - disk, caching, inmemory, blob etc. |

### 3.2 Non-Functional

| ID | Requirement |
|----|-------------|
| N2 | Per-list storage is **contiguous** to enable sequential, SIMD-friendly scans. |
| N3 | `nprobe` is a per-query parameter (not fixed at build time). |
| N4 | Errors follow the mid-level [`ANNError`] regime (this is a `diskann`-crate algorithm). |
| N5 | Build and search are parallelizable, but must use the workspace thread-pool conventions (no global rayon pool — see `clippy.toml`). |
| N6 | Index parameters (`n_lists`, representation kind, distance function) are recorded so an index can be reloaded consistently. |

### 3.3 Open Questions

- **Crate placement**: new module under `diskann/` (alongside `flat/` and
  `graph/`) vs. a dedicated `diskann-ivf` crate? Leaning toward a module first.
  
  **Answers**: 
  - Let's add it as a modulke to `diskann` for now. 
- **Coarse quantizer ownership**: does the centroid set live inside the IVF
  index, or is it a reusable component shared with quantization?
  **Answers**: 
  - It can live within the index for now. We can refactor this later if we want it to be a reusable component.
- **Centroid representation**: full-precision centroids always, or allow
  quantized centroids for the coarse search too?
  **Answers**: 
  - Doesn't matter. The component in the index that returns the top inverted lists should be generic enough that we can implement 
  multiple algorithms/representations to perform this operation. Including full precision centroids, quantized centroids etc.
- **List growth**: per-list `Vec` vs. a single backing arena with list offsets.
  Affects insert cost and scan locality.
  **Answers**: 
  - We need to come up with a clean API for inserting a point to an inverted list. Specific implementations will implement this API 
  depending on their specific data backend.
- **Relationship to `DataProvider`**: is the IVF index a *consumer* of a
  provider, or does it own its own storage backend?
  **Answers**: 
  - Critically, the IVF index is a consumer of a provider. It does not own its backend.
- **Serialization**: reuse an existing on-disk format, or define a new one?
  **Answers**: 
  - Let's not worry about serialization for now. 

## 4. Sketch (non-binding)

A rough mental model to anchor discussion — *not* a committed design:

```text
IvfIndex
├── coarse: CoarseQuantizer        // n_lists centroids + assign(query) -> list ids
├── lists:  [InvertedList; n_lists] // each: contiguous representations + local->external id map
└── search(query, k, nprobe):
        candidate_lists = coarse.closest(query, nprobe)
        for list in candidate_lists:
            list.scan_and_score(query, &mut neighbor_queue)   // via DistancesUnordered
        return neighbor_queue.top_k(k)
```

## 5. Future Work

- [ ] IVFPQ: store PQ residual codes per list; re-rank with full precision.
- [ ] Disk-resident lists / out-of-core search.
- [ ] Deletes, tombstoning, and list compaction.
- [ ] Advanced probing (soft assignment, learned routing).
- [ ] Label / attribute filtering integration (`diskann-label-filter`).
