# Iceberg Parquet Artifact Support Plan (DiskANN Side)

## Status

Proposed implementation plan. This document covers changes in DiskANN only. The companion Iceberg integration plan is `~/iceberg-rust/docs/diskann-parquet-artifacts-plan.md`.

## Goals

- Add direct Arrow/Parquet export and import for the DiskANN-native state used by the Iceberg Vamana integration.
- Cover the spherical quantizer, compressed code slots, and in-memory Vamana adjacency graph.
- Expose logical tabular schemas rather than wrapping existing FlatBuffer or canonical graph bytes in opaque Parquet binary columns.
- Preserve the existing in-memory index construction, search, mutation, and provider APIs.
- Keep Iceberg row identities, node maps, overlays, catalog metadata, and object-store policy out of DiskANN.
- Make the new support optional so normal DiskANN users do not pay for Arrow/Parquet dependencies.

## Non-goals

- Add an Iceberg dependency.
- Read or write Iceberg table data.
- Own artifact locations, catalog records, snapshot IDs, dependency IDs, or publication.
- Store full-precision vectors for the `NoStore` provider.
- Replace existing DiskANN binary save/load formats used by benchmarks or disk indexes.
- Support product-quantized disk-index artifacts, labels, filters, or other provider combinations in the first change.
- Guarantee compatibility for arbitrary future schema versions without explicit version negotiation.

## Scope

The target provider remains:

```text
DiskANNIndex<
  DefaultProvider<NoStore, SphericalStore, TableDeleteProviderAsync>
>
```

DiskANN owns three Parquet contracts:

| Logical artifact | Primary implementation owner |
| --- | --- |
| Spherical quantizer | `diskann-quantization` |
| Spherical compressed codes | `diskann-providers` spherical store |
| Vamana adjacency graph | `diskann-providers` in-memory neighbor provider |

The Iceberg integration writes its node map and overlay files separately. They are explicitly outside this repository.

## Dependency and feature design

Add workspace dependencies with one Arrow release line shared by all participating crates:

- `arrow-array`
- `arrow-schema`
- `parquet` with Arrow and async support
- any narrowly required Arrow helper crate

Use optional dependencies and a feature named `parquet`:

- `diskann-quantization/parquet` enables quantizer RecordBatch and Parquet support.
- `diskann-providers/parquet` enables provider schemas and forwards `diskann-quantization/parquet`.

Do not enable the feature by default. Existing `flatbuffers` support remains available and existing byte-oriented APIs remain intact during migration.

Before choosing versions, align with the Arrow version used by the consuming `iceberg-rust` revision to prevent duplicate incompatible Arrow types at the API boundary.

## Public I/O boundary

Use parquet-rs async traits rather than local paths:

- Export methods accept a generic `AsyncFileWriter` implementation.
- Import methods accept a generic `AsyncFileReader` implementation.
- The caller owns object-store access, buffering policy, and final artifact location.
- Methods return small summaries such as row count, file metadata, and logical checksum when useful; they do not return catalog objects.

If exposing parquet-rs traits directly proves too constraining, expose two layers:

1. Logical Arrow conversion/validation using `RecordBatch` streams.
2. Thin async Parquet helpers over those batches.

The logical layer remains the authoritative contract and makes unit testing independent of storage.

All writers must stream bounded batches. Avoid collecting another artifact-sized byte vector.

## Common format rules

- Every file records `diskann.artifact-type` and `diskann.encoding-version=1` in Parquet key-value metadata.
- Column names, Arrow types, nullability, and required metadata are validated exactly.
- Unknown metadata keys are tolerated for forward-compatible annotations.
- Rows have a canonical order.
- Numeric values are validated before mutating a provider.
- Import validates the complete stream before or while loading into a newly allocated/unpublished provider. A failed import must never expose a partially initialized index.
- Logical CRC32 is computed over a documented canonical row encoding for compatibility with the current Iceberg prototype. A stronger digest can be added independently.

## Quantizer Parquet contract

### Schema

One row represents one spherical quantizer plan.

| Column | Arrow type | Nullability | Meaning |
| --- | --- | --- | --- |
| `nbits` | `UInt8` | required | Spherical code width |
| `metric` | `Utf8` | required | `l2`, `inner_product`, or `cosine` |
| `centroid` | `List<Float32>` | required | Shift/centroid vector |
| `mean_norm` | `Float32` | required | Positive training mean norm |
| `pre_scale` | `Float32` | required | Positive pre-scaling factor |
| `transform_kind` | `Utf8` | required | Initially `double_hadamard` |
| `transform_signs_0` | `List<Boolean>` | required | First sign stage |
| `transform_signs_1` | `List<Boolean>` | required | Second sign stage |
| `transform_subsample` | `List<UInt32>` | nullable | Sorted optional output subsample |

Required metadata:

- `diskann.artifact-type=vector.quantizer`
- `diskann.encoding-version=1`
- `diskann.input-dimension`
- `diskann.transformed-dimension`
- `diskann.logical-checksum-crc32`

### Implementation

Add a stable logical state representation for the spherical plan. It should expose only reconstruction data and validation, not mutable internals. Candidate types:

- `SphericalQuantizerState`
- `SphericalTransformState`
- `DoubleHadamardState`

Add conversions equivalent to:

- `Impl::<NBITS>::export_state()`
- `Impl::<NBITS>::try_from_state(state, allocator)`
- `Impl::<NBITS>::write_parquet(writer)`
- `Impl::<NBITS>::read_parquet(reader, allocator)`

Reuse existing invariants in `SphericalQuantizer::try_unpack`, `Transform::try_unpack`, and `DoubleHadamard::try_from_parts` rather than duplicating weaker checks.

Version 1 only needs `DoubleHadamard`, because that is the transform used by the Iceberg integration. Return a precise unsupported-transform error for other transform variants. Design the state enum so additional transforms can be added in later encoding versions without changing the top-level API.

Do not store a FlatBuffer payload in a `Binary` column. Keep `export_quantizer()` and FlatBuffer deserialization for existing users and regression comparison.

### Quantizer validation

Reject:

- a row count other than one;
- unsupported bit width or a mismatch with `Impl<NBITS>`;
- empty centroid/sign arrays;
- dimension disagreement among centroid, signs, transform, and metadata;
- invalid metric strings;
- non-finite centroid values;
- nonpositive or non-finite norm/scale;
- invalid sign or subsample structure;
- checksum mismatch.

## Compressed-code Parquet contract

### Schema

One row represents one allocated code slot.

| Column | Arrow type | Nullability | Meaning |
| --- | --- | --- | --- |
| `node_id` | `UInt32` | required | Allocated slot ID |
| `code` | `FixedSizeBinary(bytes_per_code)` | required | Canonical spherical code bytes |

Rows are strictly ordered by `node_id` and cover the complete range `[0, total_capacity)`, including free mutable slots and frozen slots. This preserves the current Iceberg export/import semantics exactly.

Required metadata:

- `diskann.artifact-type=vector.codes`
- `diskann.encoding-version=1`
- `diskann.nbits`
- `diskann.bytes-per-code`
- `diskann.total-capacity`
- `diskann.mutable-capacity`
- `diskann.frozen-points`
- `diskann.logical-checksum-crc32`

The Iceberg catalog separately records the quantizer artifact dependency. DiskANN import receives the already-created compatible `SphericalStore` and validates these metadata values against it and caller-supplied capacity expectations.

### Implementation

On `SphericalStore`, add APIs equivalent to:

- `write_codes_parquet(writer, layout)`
- `read_codes_parquet(reader, expected_layout)`

Export reads each slot through the existing canonical `code(index)` view and builds bounded `UInt32Array` and `FixedSizeBinaryArray` batches.

Import validates IDs and code widths and then copies consecutive batches through `import_codes_at`. Do not create one concatenated code vector. Preserve `import_codes` and `import_codes_at` for existing users.

### Code validation

Reject:

- missing, duplicate, out-of-order, or out-of-range node IDs;
- row count/capacity mismatch;
- fixed-size-binary width mismatch;
- null IDs or codes;
- bit-width or layout metadata mismatch;
- checksum mismatch.

## Graph Parquet contract

### Schema

One row represents the outgoing adjacency list of one allocated node.

| Column | Arrow type | Nullability | Meaning |
| --- | --- | --- | --- |
| `node_id` | `UInt32` | required | Dense node ID |
| `neighbors` | `List<UInt32>` | required | Ordered outbound neighbor IDs |

Rows are strictly ordered by `node_id` and cover all points allocated in the provider, including frozen start points.

Required metadata:

- `diskann.artifact-type=vector.search-structure`
- `diskann.encoding-version=1`
- `diskann.total-points`
- `diskann.mutable-capacity`
- `diskann.num-start-points`
- `diskann.start-point-ids` as a canonical comma-separated list
- `diskann.max-degree`
- `diskann.logical-checksum-crc32`

The first version may enforce the current Iceberg invariant of exactly one frozen start point while retaining a list-valued metadata contract for future provider support.

### Implementation

Add provider APIs equivalent to:

- `SimpleNeighborProviderAsync::write_graph_parquet(writer, start_points)`
- `SimpleNeighborProviderAsync::read_graph_parquet(reader, expected_layout)`
- or provider-level wrappers on `DefaultProvider` mirroring current `export_graph`/`import_graph` placement.

Export iterates adjacency lists directly through the neighbor provider and builds bounded list arrays. It must not first call `export_graph()`.

Import validates complete metadata and row structure, then sets each adjacency list on a preallocated provider. Reuse or factor the structural checks currently exercised by `import_direct`:

- point count equals provider capacity;
- maximum degree fits allocated adjacency width;
- start-point count and IDs match runtime expectations;
- every neighbor is in range.

Keep canonical binary `export_graph`/`import_graph` and `save_graph`/`load_graph` unchanged for other users.

### Graph validation

Reject:

- missing, duplicate, out-of-order, or out-of-range node IDs;
- row count/provider-capacity mismatch;
- null neighbor lists or null neighbor values;
- adjacency length greater than configured maximum degree;
- out-of-range or self-inconsistent neighbor IDs according to existing graph rules;
- start-point mismatch;
- checksum mismatch.

Preserve neighbor ordering exactly. Do not sort adjacency lists during serialization or import.

## Resolved runtime parameters

The Iceberg integration currently persists `ResolvedBootstrapParameters` in outer Puffin envelopes. Parquet removes those envelopes, but reload still needs the parameters before allocating the provider.

Use a shared, format-versioned metadata structure associated with the graph artifact, with compatibility copies in quantizer/codes metadata where useful. It must carry:

- parameter format version;
- bit width and metric;
- input and transformed dimensions;
- transform kind/target;
- quantizer pre-scale and training seed;
- pruned and maximum degree;
- build and search complexity;
- maximum minibatch parallelism;
- alpha;
- live row count;
- mutable capacity;
- frozen point count;
- total capacity.

Because `ResolvedBootstrapParameters` is currently defined in `iceberg-rust`, choose one of these boundaries during implementation:

1. Preferred: define a DiskANN-neutral `QuantizedVamanaLayout`/`QuantizedVamanaParameters` DTO in `diskann-providers`; `iceberg-rust` converts to and from its policy-facing type.
2. Minimal fallback: DiskANN files expose only required provider-layout metadata, while `iceberg-rust` stores its additional build/search policy metadata as namespaced Parquet key-value pairs and performs the conversion.

DiskANN must never depend on the Iceberg type. The companion implementation should settle the DTO before replacing Puffin readers, because provider allocation requires this metadata.

## API and error design

- Define precise non-allocating or modest provider-layer errors with `thiserror` where practical.
- Distinguish schema mismatch, metadata mismatch, unsupported version, malformed value, checksum failure, capacity mismatch, and Parquet I/O failure.
- Chain underlying Arrow/Parquet errors through `source`.
- Avoid converting all failures into one crate-level error enum.
- Add `#[track_caller]` when converting errors into `ANNError` at public compatibility boundaries.
- Document `# Errors` for every public export/import API.

## Files expected to change

Likely areas:

- workspace `Cargo.toml` for Arrow/Parquet versions;
- `diskann-quantization/Cargo.toml` and feature declarations;
- spherical quantizer/transform modules for logical state access and reconstruction;
- a new quantizer Parquet module under `diskann-quantization`;
- `diskann-providers/Cargo.toml` and feature declarations;
- spherical in-memory store for code streaming;
- simple neighbor/default provider for graph streaming;
- one or more new provider Parquet schema/I/O modules;
- tests colocated with each state component.

Keep Parquet code out of `diskann`, `diskann-vector`, and unrelated providers.

## Validation plan

### Quantizer tests

- Round trip 1-, 2-, and 4-bit plans with L2, inner product, and cosine where supported.
- Compare distances and compressed output before and after logical/Parquet round trip.
- Test same, natural, and override transform targets used by `DoubleHadamard`.
- Reject every schema, dimension, bit-width, metric, positivity, transform, and checksum mismatch.
- Demonstrate that Parquet reload is equivalent to existing FlatBuffer reload for the same plan.

### Code tests

- Round trip complete stores for 1-, 2-, and 4-bit codes.
- Include zeroed free slots and frozen slots.
- Use multiple record batches to exercise streaming boundaries.
- Reject gaps, duplicates, reordered IDs, incorrect fixed width, capacity mismatch, nulls, and corruption.
- Verify imports do not require one artifact-sized intermediate allocation.

### Graph tests

- Round trip empty adjacency lists, full-degree lists, mutable nodes, and frozen start points.
- Use multiple record batches.
- Preserve exact neighbor ordering.
- Reject bad node IDs, missing rows, out-of-range neighbors, excessive degree, wrong point counts, and wrong start points.
- Compare Parquet reload with current canonical binary graph reload.

### Lifecycle test

Construct the exact quantized-only provider used by Iceberg, then:

1. build an index;
2. export quantizer, codes, and graph to Parquet;
3. allocate a fresh provider from decoded metadata;
4. import all three files;
5. compare candidate search results;
6. insert, delete, consolidate, export, and import again;
7. compare search results after the second reload.

Run this matrix for 1-, 2-, and 4-bit configurations.

## Commands

Run at minimum:

- `cargo fmt --all`
- `cargo test -p diskann-quantization --features parquet`
- `cargo test -p diskann-providers --features parquet`
- `cargo clippy -p diskann-quantization -p diskann-providers --all-targets --features parquet -- -D warnings`
- `cargo clippy -p diskann-quantization -p diskann-providers --all-targets --no-default-features -- -D warnings`

Also run the existing tests covering FlatBuffer quantizer serialization and canonical graph/code import to ensure backward compatibility.

## Implementation sequence

1. Select Arrow/Parquet versions compatible with the target `iceberg-rust` revision and add optional features.
2. Define shared metadata keys, schema versions, layout DTOs, and error types.
3. Add spherical logical-state export/import and quantizer Parquet round trips.
4. Add streaming compressed-code Parquet export/import.
5. Add streaming graph Parquet export/import.
6. Add mismatch/corruption tests and the full lifecycle test.
7. Document public APIs and feature flags.
8. Commit a revision for the Iceberg integration to pin.

## Completion criteria

- With the optional feature enabled, DiskANN can directly stream quantizer, complete code-slot state, and Vamana adjacency state to Parquet.
- A fresh provider can be allocated and populated from those Parquet files without parsing the legacy FlatBuffer or canonical graph formats in `iceberg-rust`.
- Logical schemas are validated strictly and corrupt/incompatible files fail before publication.
- Existing non-Parquet save/load paths and default feature builds remain unchanged.
- The companion Iceberg integration can provide object-store readers/writers without DiskANN depending on Iceberg APIs.
