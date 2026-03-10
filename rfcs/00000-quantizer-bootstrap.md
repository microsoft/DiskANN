# Quantization Bootstrapping

| | |
|---|---|
| **Authors** | Jack Moffitt |
| **Contributors** |  |
| **Created** | 2026-03-10 |
| **Updated** | 2026-03-10 |

## Summary

Indexes that use quantization must have a minimum number of vectors before they
can build quantization tables and start inserting vectors. Bootstrapping is the
process of incrementally building an index that starts empty, operates on
non-quantized vectors until enough vectors are present to build quantization
tables, and then transitions to normal operation.

## Motivation

### Background

DiskANN's quantizers require some statistical information in order to build
quantization tables. For PQ, 10,000 vectors are generally required to build good
tables; for spherical, 100 are needed. In order to create an index, these
vectors must be provided at creation time in order to build the quantization
tables, at which point each vector is quantized as it is inserted.

This requirement is easy to fulfill when building indexes from existing
datasets, but when starting from scratch, there is no ability for DiskANN to
build a quantized index since the quantization tables are a required part of the
constructor.

Current deployments of DiskANN work around this issue by not allowing index
creation until a dataset is sufficient large (pg_diskann), or operating a
separate flat index until sufficient vectors are collected at which point the
quantization tables are calculated and a graph index is built with DiskANN.

### Problem Statement

This RFC proposes changing DiskANN to operate in a quantization bootstrap mode
where it operates on full precision vectors until sufficient vectors exist to
create quantization tables, and then seamlessly transitions to a quantized
index.

This means the index will operate in three different phases. In Phase 1, the
index operates in full precision mode only until sufficient vectors exist to
build quantization tables. During Phase 2, quantization tables will be built and
vectors will be quantized on insert; pre-existing vectors be quantized in the
background. Once all vectors are quantized, Phase Three begins the normal
operation of the quantized index.

### Goals

1. Allow quantized indices to start empty and use full precision data only to
   operate until sufficient vectors are inserted.
2. Aside from allowing construction of `DiskANNIndex` without providing
   quantization tables, there should user visible changes to using the index.
3. Performance should remain as high as possible during the three phases.
4. The quantization of previously inserted full vectors during Phase Two should
   be controllable by the data provider.

## Proposal

Bootstrapping needs two changes to DiskANN.

1. **Switching Strategies**: DiskANN needs to start by using full precision only
   strategies during Phase One, and switching to a hybrid strategy for Phase
   Two, and if the user's intent is to use quantized-only strategies, switching
   to quantized-only for Phase Three.
2. **Quantization Backfill**: During Phase Two, previously inserted vectors will
   need to be quantized. As background jobs are a performance concern, DiskANN
   will need hooks for customizing this behavior.

### Switching Strategies

DiskANN already has the ability to run multiple strategies including hybrid full
precision and quantized ones. These should be sufficient for purposes of
bootstrapping, but we will need to orchestrate seamless transitions between
them.

As the caller designates a strategy to use, we can implement new
`BootstrappedQuantized` and `BootstrappedHybrid` strategies that layer over
existing `FullPrecision`, `Quantized`, and `Hybrid` strategies. These new
bootstrapped strategies will delegate operation to the existing strategies
depending on the current phase.

*Open question*: How exactly do we do this?

### Quantization Backfill

After quantization tables are built, newly inserted vectors will be quantized
before insertion, but previously inserted vectors won't have quantized
representations yet. During Phase Two, these previously inserted full precision
vectors will need to be quantized before the index enters Phase Three.

Since integrators of DiskANN are sensitive to background jobs, how the index
manages backfilling quantized vectors should be controllable.

The simplest way is to backfill all missing quantized vectors immediately during
the insert that starts Phase Two. This will cause a latency spike on that single
insert, but doesn't require any background processing.

A more complicated solution would be to launch a background job that iterates
over full-precision only vectors and quantizes them. DiskANN should provide such
a job that integrators can use, but should also provide some callback that the
hosting database can pump to make incremental progress under its own control.

Both of these methods can be realized by having a new trait `QuantBackfill`:

```rust

pub enum QuantBackfillStatus {
     Incomplete,
     Complete,
}

pub trait QuantBackfill {
     type BackfillError: AsyncFriendly;

     /// Backfill quantization vectors for up to approximately `duration` amount of time.
     fn backfill(duration: Duration) -> impl Future<Output=Result<QuantBackfillStatus, QuantBackfillError>> + AsyncFriendly;
}
```

This trait would be implemented on the type that implements the
`BootstrappedQuantized` and `BootstrappedHybrid` strategies. 

*Open question*: How to implement the background task and make it overridable?

## Trade-offs

Currently the workarounds in use are either no index at all until sufficient
vectors exist or operating a side index until sufficient vectors exist and then
building a quantized graph.

pg_diskann uses the former method, which means users are confused when they try
to create indexes on empty tables or insufficiently populated tables and get an
error. Cosmos DB uses the latter strategy and operates a flat index until an
asynchronous graph build is complete enough to use the graph index. This
requires the Cosmos DB team to maintain all their own infrastructure for the
flat index and the code around transitioning to the graph index.

This proposal mitigates the downsides while still allowing the integrator to
retain control over key performance details.

## Benchmark Results

Since there is no way to build an index currently until quantization tables are
built, there is no way to benchmark the first two phases. There should be no
impact during Phase Three to performance.

## Future Work

None.

## References

None.