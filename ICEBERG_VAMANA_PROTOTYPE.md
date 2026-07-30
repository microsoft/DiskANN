# Iceberg Vamana Prototype: DiskANN Work

## Status

Exploration and implementation plan for a prototype. This document describes
the work owned by the DiskANN repository. Iceberg snapshot scans, Puffin files,
REST index catalog commits, and demo process orchestration belong in
`iceberg-rust`.

## Goal

Make the in-memory Vamana implementation usable as a versioned, portable
component of an Iceberg vector index. The prototype must support:

- building an in-memory quantized Vamana index from a stream of vectors;
- spherical quantization with `DoubleHadamard` and 1, 2, or 4 bits per
  dimension;
- traversing the graph with a full-precision query against quantized vectors
  and returning an expanded candidate set for exact reranking;
- exporting and importing all state needed to resume search and mutation;
- inserting and lazily deleting points from an imported index; and
- exporting the updated index in the same format as a bootstrap build.

DiskANN will not depend on Iceberg, Arrow, Parquet, Puffin, a catalog client, or
an object-store SDK. Those dependencies remain in the integration crate in
`iceberg-rust`.

## Existing Building Blocks

The prototype should extend existing abstractions rather than introduce a
second Vamana implementation.

- `diskann/src/graph/index.rs`: `DiskANNIndex` provides `insert`,
  `multi_insert`, `search`, `inplace_delete`, and `consolidate_vector`.
- `diskann-providers/src/index/diskann_async.rs`:
  `new_quant_only_index` constructs a quantized-only in-memory index.
- `diskann-providers/src/model/graph/provider/async_/inmem/spherical.rs`:
  `SphericalStore` stores compressed vectors and `Quantized` supplies the
  build, prune, insert, and search strategies.
- `diskann-providers/src/model/graph/provider/async_/inmem/full_precision.rs`:
  the existing `Rerank` post-processor reads synchronously from a resident
  `FullPrecisionStore`; it is not suitable for asynchronous object-store
  reads.
- `diskann-quantization/src/spherical`: the runtime interface dispatches to
  `Impl<1>`, `Impl<2>`, or `Impl<4>` and can serialize the quantization plan.
- `diskann-quantization/src/algorithms/transforms/double_hadamard.rs`:
  `DoubleHadamard` supplies the requested transform.
- `diskann-providers/src/model/graph/provider/async_/simple_neighbor_provider.rs`:
  the neighbor provider already has graph save/load support.
- `diskann-providers/src/storage`: `SaveWith`, `LoadWith`, and the existing
  vector-store implementations establish the local persistence pattern.
- `diskann-providers/src/model/graph/provider/async_/inmem/provider.rs`:
  `TableDeleteProviderAsync` supplies the deletion bitmap used by graph search
  and consolidation.

The main gap is that `SphericalStore` does not implement the save/load support
already available to the scalar and product-quantized stores. The current
storage APIs also do not expose one complete, versioned in-memory state object
that an external container format such as Puffin can embed.

## Provider Choice

The provider choice is settled for this prototype: use the established async
in-memory implementation in `diskann-providers`. Do not use the newer
`diskann-inmem` crate or add an adapter for it.

The concrete stack is:

```text
DiskANNIndex<
  DefaultProvider<NoStore, SphericalStore, TableDeleteProviderAsync>
>
```

Construct it with `diskann_providers::index::diskann_async::new_quant_only_index`,
the trained spherical quantizer implementation, and the `TableBasedDeletes`
creator tag. Use the spherical `Quantized` strategy for build, insertion,
pruning, and graph search. `TableDeleteProviderAsync` supplies lazy deletion
and participates in graph consolidation.

This is intentionally `new_quant_only_index`, not `new_quant_index`: DiskANN
performs graph traversal and returns candidate node IDs with approximate
distances. The Iceberg integration resolves those IDs to `(_file, _pos)`, reads
the original vectors from Parquet, and computes exact distances asynchronously.
Full-precision vectors are not duplicated in DiskANN state.

All persistence work described below extends these existing
`diskann-providers` types. Bootstrap, reload, incremental mutation,
consolidation, and candidate search must not switch provider implementations.

## Boundary With Iceberg

The integration crate supplies vectors and stable external labels. DiskANN
continues to use dense `u32` node IDs internally.

```rust
struct ExternalVector {
    node_id: u32,
    values: Vec<f32>,
}
```

The mapping from `node_id` to Iceberg `(_file, _pos)` is not a DiskANN concept.
The integration crate persists it as a separate blob in the same Puffin file
as the graph/runtime state, not in the compressed-code artifact. The following
invariants cross the repository boundary:

- every live non-frozen node ID has exactly one external row key;
- frozen/start-point IDs never have external row keys;
- node IDs and compressed-code slots have the same ordering;
- the metric, input dimension, transformed dimension, bit width, and
  quantizer fingerprint agree across every state component;
- the total number of allocated IDs, including frozen points and update
  headroom, fits in `u32`; and
- import rejects truncated, duplicate, out-of-range, or incompatible state.

Search accepts a final result count `k` and a larger candidate count `R`. The
integration's initial default is `R = min(live_rows, max(4 * k, 64))`, and the
DiskANN search-list size must be at least `R`. The quantized-only search result
must expose up to `R` non-frozen, non-deleted node IDs and their approximate
distances without applying an in-memory full-precision reranker.

## Quantization Plan

Bootstrap uses a deterministic two-pass build over one fixed Iceberg snapshot.
The Iceberg adapter owns both passes.

1. Validate one non-null, fixed-length `f32` vector per accepted row.
2. Collect a bounded deterministic training sample. The initial default is
   `min(row_count, 100_000)` vectors, seeded from table UUID, snapshot ID, and
   vector field ID.
3. Train `SphericalQuantizer` with the configured metric and bit width.
4. Configure `TransformKind::DoubleHadamard` with a target dimension accepted
   by the existing transform planner. Persist the resolved target dimension,
   not only the requested setting.
5. Rescan the same snapshot, compress every vector, and build Vamana over the
   compressed representations.

The demo defaults to L2, but metric is an index property and must be validated
through the spherical quantizer's `SupportedMetric` path. `nbits` is restricted
to 1, 2, or 4. Null vectors, mixed dimensions, unsupported Arrow element types,
and non-finite values are build errors for the prototype.

The training API currently consumes a `MatrixView`; it is not streaming. The
bounded sample prevents training memory from scaling with the table, while the
second pass can remain batch-streamed.

## DiskANN State Contract

DiskANN should expose a container-neutral state contract. Puffin wrapping and
Iceberg row labels are deliberately outside this contract.

### Quantizer State

The quantizer state contains the existing serialized spherical plan, including
the transform and all trained parameters. Import dispatches by encoded bit
width through `spherical::iface::try_deserialize` and verifies the caller's
expected metric and dimensions.

Proposed public surface, with final names adapted to existing storage traits:

```rust
pub struct SphericalQuantizerState {
    pub format_version: u32,
    pub bytes: Vec<u8>,
}

impl SphericalStore {
    pub fn export_quantizer(&self) -> Result<SphericalQuantizerState, ...>;
}
```

Do not create a second quantizer encoding. The state envelope only versions and
checks the already supported spherical FlatBuffer payload.

### Compressed Vector State

The compressed-vector state contains:

- format version and byte order;
- number of populated slots and total capacity;
- bytes per vector and required alignment;
- a contiguous canonical byte representation of all populated code slots; and
- a checksum over the payload.

Import allocates `AlignedMemoryVectorStore<u8>` and copies validated canonical
bytes into it. It must not expose uninitialized capacity or serialize native
padding. A direct read-only code view may be added for efficient Puffin writing,
but an owned import path is required.

`SphericalStore` should implement the same `SaveWith<AsyncIndexMetadata>` and
`LoadWith<AsyncQuantLoadContext>` behavior as `SQStore` where that format is
appropriate. A small in-memory export/import API is also needed because the
Iceberg Puffin writer accepts blob payloads rather than DiskANN storage paths.

### Graph And Runtime State

The graph state contains:

- the existing canonical adjacency payload;
- resolved `IndexConfiguration` and Vamana build/search parameters;
- start/frozen point IDs and frozen-point vectors or codes;
- populated node count, capacity, and next available node ID;
- deletion bitmap;
- prune strategy and graph-format versions; and
- checksums for independently stored payloads.

Loading must recreate a `DiskANNIndex<DefaultProvider<...>>` that can search and
accept subsequent mutations. Persisting only adjacency lists is insufficient:
deletion state, capacity, start points, and resolved parameters affect behavior
after restart.

## Build APIs

The integration needs two high-level operations over the generic state APIs.
They may initially live in `diskann-providers`.

### Bootstrap

```text
train sample
  -> spherical quantizer
  -> compress fixed-snapshot vector stream
  -> new_quant_only_index
  -> multi_insert/build
  -> export quantizer + codes + graph/runtime state
```

The caller assigns deterministic node IDs as rows arrive. Capacity includes the
configured frozen points and update headroom. The prototype default is 10%
headroom so a base can absorb changes before the 5% consolidation threshold.

### Apply Delta

```text
import quantizer + codes + graph/runtime state
  -> resolve deleted external labels to node IDs
  -> inplace_delete
  -> assign new node IDs and compress inserts
  -> insert or multi_insert
  -> consolidate_vector as required
  -> export the same three state components
```

The API accepts node IDs, not Iceberg row keys. The integration crate owns label
lookup, duplicate detection, and allocation policy. A delta is rejected if it
would exceed imported capacity; the caller can instead build a replacement
base with greater capacity.

Applying the same logical delta twice must either be rejected before mutation
or become a no-op through caller-supplied node allocation state. Silent creation
of duplicate live nodes is not allowed.

## Base And Overlay Interpretation

The demo publishes a base plus small overlays so each table update does not
rewrite the full base. Overlays do not contain a DiskANN index:

- a base is a normal quantized Vamana index;
- each overlay is one Puffin file containing a `delete` blob of
  `(_file, _pos)` pairs and an `insert` blob of
  `(_file, _pos, full_precision_vector)` triplets;
- inserted vectors are neither compressed with the base quantizer nor assigned
  overlay-local DiskANN node IDs;
- no Vamana graph is built for overlay inserts; and
- either blob may be empty, including the `insert` blob for a delete-only
  overlay.

Search traverses only the base Vamana graph and flat-scans every overlay's
uncompressed insert vectors with the configured metric. The adapter unions
tombstones, filters base and overlay candidates, maps base node IDs through the
separate row map, and deduplicates row identities by their best distance. It
requests an expanded base candidate set rather than treating approximate base
distances as final. The adapter then reads original vectors for surviving base
candidates from Parquet and computes exact distances; overlay insert distances
are already exact. It sorts the combined exact results and returns the final
`k`, increasing base search breadth when tombstone filtering leaves too few
live candidates.

The asynchronous Parquet reads and exact-distance reranker remain outside
DiskANN. Extending `GetFullPrecision` to perform I/O would put awaits and remote
failure semantics into graph post-processing and would couple this repository
to a storage system. The existing resident `FullPrecisionStore` reranker stays
available for other users.

At consolidation, the worker imports the base, resolves overlay deletions to
base node IDs, compresses overlay insert vectors with the base quantizer, and
applies both through the incremental APIs before consolidating graph edges and
exporting one replacement base. If capacity or compatibility prevents in-place
application, it performs a bootstrap rebuild from the exact target Iceberg
snapshot. Both paths produce the same base artifact contract.

## Required DiskANN Changes

1. Add spherical-store save/load parity with the other in-memory quantized
   stores.
2. Add checked import/export of quantizer and canonical compressed-code bytes.
3. Add a versioned graph/runtime state envelope that includes deletion and
   mutable-allocation state.
4. Add construction helpers that recreate a searchable and mutable quantized
   index from those components.
5. Expose enough resolved configuration to reproduce an imported index without
   relying on defaults that may change between releases.
6. Confirm the quantized-only search API can return a caller-selected expanded
  candidate count and approximate distances before external reranking.
7. Keep all APIs independent of Iceberg, Parquet, Puffin, and object storage.

No new DiskANN crate is required initially. Quantizer and code persistence fit
in `diskann-providers` beside the existing provider storage implementations;
quantizer payload logic remains in `diskann-quantization`.

## Validation

Add focused tests before wiring the external integration:

- round-trip spherical plans for 1, 2, and 4 bits with `DoubleHadamard`;
- round-trip compressed code slots for partially filled capacity;
- round-trip graph, start points, parameters, and deletion bitmap;
- equal search results before and after a state round trip;
- candidate-only search returns up to the requested `R` IDs without requiring
  a `FullPrecisionStore`;
- import, insert, delete, consolidate, export, and second import;
- rejection of metric, dimension, bit-width, capacity, checksum, and node-count
  mismatches;
- update-capacity exhaustion without partial mutation; and
- deterministic state decoding across supported little-endian platforms.

Useful initial commands are:

```bash
cargo test -p diskann-quantization spherical
cargo test -p diskann-providers spherical
cargo test -p diskann-providers storage
cargo clippy -p diskann-providers --all-targets -- -D warnings
```

## Prototype Limitations

- Build and search state is fully resident in memory.
- Puffin writes borrow graph and code buffers and stream uncompressed payloads
  in bounded chunks, avoiding a second blob-sized allocation during export.
- Graph traversal is resident, but every uncached rerank issues remote Parquet
  reads. Query latency and availability therefore include object-store I/O.
- Overlay fan-out is acceptable only while mutation volume is small.
- The binary state contract is prototype-versioned and not yet a long-term
  compatibility promise.

## Decisions To Revisit

- Whether production persistence should use the existing DiskANN storage files
  inside Puffin blobs or a new explicitly portable state envelope.
- Whether graph consolidation should compact IDs and code slots or only remove
  deleted graph references.
- Whether a future deployment should add a full-vector cache or artifact to
  reduce rerank I/O. The prototype deliberately uses the Iceberg Parquet files
  as the full-precision source of truth.