# Quantization Bootstrapping

| | |
|---|---|
| **Authors** | Jack Moffitt |
| **Contributors** | Mark Hildebrand |
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
creation until a dataset is sufficiently large (pg_diskann), or operating a
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
vectors will be quantized on insert; pre-existing vectors will be quantized in
the background. Once all vectors are quantized, Phase Three begins the normal
operation of the quantized index.

### Goals

1. Allow quantized indices to start empty and use full precision data only to
   operate until sufficient vectors are inserted.
2. Aside from allowing construction of `DiskANNIndex` without providing
   quantization tables, there should be no user-visible changes to using the index.
3. Performance should remain as high as possible during the three phases.
4. The quantization of previously inserted full vectors during Phase Two should
   be controllable by the data provider.

## Proposal

Bootstrapping needs two changes to a DiskANN data provider implementation.

1. **Switching Strategies**: DiskANN needs to start by using full precision only
   strategies during Phase One, and switching to a quantized-only or hybrid
   strategy for Phase Two.
2. **Quantization Backfill**: During Phase Two, previously inserted vectors will
   need to be quantized. As background jobs are a performance concern, how
   exactly this is accomplished must be customizable by the data provider.

### Switching Strategies

DiskANN already has the ability to run multiple strategies including hybrid full
precision and quantized ones. These strategies represent the high level intent,
but the data provider can choose alternate implementions depending on the data
available during the current phase.

#### Insertion and Deletion

Insertion and deletion can remain largely the same in the data provider
implementation. Inserts will need to write vectors, mappings, attributes, et al
into storage, and can track the current phase to gate writes to quantized
vectors. The search portion of these operations will return different objects
depending on the current phase.

For example, consider a `DataProvder::set_element()` implementation:

```rust
struct ExampleProvider {
     // other fields omitted
     quantizer: Option<Quantizer>,
}

impl SetElement<[f32]> for ExampleProvider {
     // associated types ommitted

     async fn set_element(
        &self,
        context: &Self::Context,
        id: &Self::ExternalId,
        element: &[T],
     ) -> Result<Self::Guard, Self::SetError> {
          let internal_id = self.new_id()?;
          self.write_vector(context, internal_id, element)?;
          self.set_internal_map(internal_id, id)?;
          self.set_external_map(id, internal_id)?;

          // Quantize and storage quant vector if we have a quantizer.
          if let Some(quantizer) = self.quantizer {
               let qv = quantizer.quantize(element)?;
               self.write_quant_vector(context, internal_id, element)?;
          } else {
               // This function will check if we are ready for Phase Two, and if so, do or schedule the quantizer intialization.
               self.maybe_initialize_quantizer()?;
          }

          Ok(NoopGuard::new(internal_id))
     }
}
```

Delete can similarly check the status of the quantizer, and delete quantized
vectors if they exist.

#### Searching

To avoid complexity of hybrid distance calculations, either full precision
distances will be used (Phase One and Two) or quantized distances will be used
(Phase Three). If a hybrid strategy is in use, then the hybrid distances will
not be used until Phase Three.

Since vector data may be in one of two representations, the `Accessor::Element`
type should be `Poly<u8>` (this should be over-aligned to the correct alignment
for the primitive element type), and the data provider should interpret based on
data size. The distance and query computers will also need modifications to
accept both vector representations, and in the case of query computer the
representation much match that of the query.


### Quantization Backfill

After quantization tables are built, newly inserted vectors will be quantized
before insertion, but previously inserted vectors won't have quantized
representations yet. During Phase Two, these previously inserted full precision
vectors will need to be quantized before the index enters Phase Three.

Since integrators of DiskANN are sensitive to background jobs, how the index
manages backfilling quantized vectors is controlled by the data provider
implementation. The data provider must have some way to track which vectors have
missing quantized representations so that it generate them.

Once Phase Two is reached, the data provider can either pause during insertion
of the phase changing vector, or schedule the work to happen asynchronously
however it likes.

One possibility is to piggy-back on deletion tracking to track quantization
status of vectors. For example, in diskann-garnet, a free space map is kept that
tracks deletes. This could be expanded from 1-bit to 2-bits, and the second bit
used to track whether the vector is quantized. Alternatively, metadata about the
allocated range can be kept and used to iterate over the unquantized set.


## Trade-offs

Currently the workarounds in use are either no index at all until sufficient
vectors exist or operating a side index until sufficient vectors exist and then
building a quantized graph.

pg_diskann uses the former method, which means users are confused when they try
to create indexes on empty or insufficiently populated tables and get an
error. Cosmos DB uses the latter strategy and operates a flat index until an
asynchronous graph build is complete enough to use the graph index. This
requires the Cosmos DB team to maintain all their own infrastructure for the
flat index and the code around transitioning to the graph index.

This proposal mitigates the downsides while still allowing the integrator to
retain control over key performance details.

This proposal also entirely encapsulates this inside the `DataProvider`
implementation. Alternatively, one could attempt to solve this with some kind of
index or strategy layering, but the complexity this would introduce seems not
worth the cost.

## Benchmark Results

Since there is no way to build an index currently until quantization tables are
built, there is no way to benchmark the first two phases. There should be no
impact during Phase Three to performance.

## Future Work

None.

## References

None.