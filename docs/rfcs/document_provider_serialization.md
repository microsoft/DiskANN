# RFC: SaveWith and LoadWith for DocumentProvider

**Status**: Draft  
**Crate(s) affected**: `diskann-label-filter`, `diskann-providers`

---

## Table of Contents

1. [Summary](#summary)
2. [Background](#background)
3. [Existing Serialization Patterns](#existing-serialization-patterns)
   - [SaveWith and LoadWith Traits](#savewith-and-loadwith-traits)
   - [BfTreeProvider](#bftreeprovider)
   - [DefaultProvider](#defaultprovider)
   - [DiskProvider](#diskprovider)
4. [Proposed Design](#proposed-design)
   - [Crate Dependency](#crate-dependency)
   - [Attribute Store Serialization Interface](#attribute-store-serialization-interface)
   - [Label File Format](#label-file-format)
   - [DocumentProvider SaveWith Implementation](#documentprovider-savewith-implementation)
   - [DocumentProvider LoadWith Implementation](#documentprovider-loadwith-implementation)
5. [Open Questions](#open-questions)

---

## Summary

This RFC proposes implementing the `SaveWith` and `LoadWith` serialization traits for `DocumentProvider`. `DocumentProvider` wraps an inner `DataProvider` (e.g. `DefaultProvider` or `BfTreeProvider`) and an `AttributeStore` that holds the per-vector label data. Serialization must handle both: delegating to the inner provider's own save/load and persisting the attribute store to a separate binary file.

---

## Background

`DocumentProvider` (in `diskann-label-filter`) is a wrapper that combines a `DataProvider` with an `AttributeStore`. It is used when the ANN index must support filtered search: every indexed vector carries a set of typed key–value attributes (e.g. `"category" = "electronics"`, `"price" = 299.99`) and queries can include attribute-based predicates.

```rust
pub struct DocumentProvider<DP, AS> {
    inner_provider: DP,
    attribute_store: AS,
}
```

The concrete attribute store in use today is `RoaringAttributeStore<IT>`, which maintains:

- An `AttributeEncoder` mapping each distinct `Attribute` (field name + typed value) to a compact `u64` ID.
- A forward index (`RoaringTreemapSetProvider<IT>`) mapping each node's internal ID to the set of its attribute IDs.
- An inverted index (`RoaringTreemapSetProvider<u64>`) mapping each attribute ID to the set of node IDs that carry it. The inverted index can be rebuilt from the forward index and so **does not need to be persisted**.

At present, `DocumentProvider` implements `DataProvider`, `SetElement`, and `Delete` but has no serialization support. This RFC proposes filling that gap.

---

## Existing Serialization Patterns

### SaveWith and LoadWith Traits

Defined in `diskann-providers::storage`:

```rust
pub trait SaveWith<T> {
    type Ok: Send;
    type Error: std::error::Error + Send;

    fn save_with<P>(&self, provider: &P, auxiliary: &T)
        -> impl Future<Output = Result<Self::Ok, Self::Error>> + Send
    where
        P: StorageWriteProvider;
}

pub trait LoadWith<T>: Sized {
    type Error: std::error::Error + Send;

    fn load_with<P>(provider: &P, auxiliary: &T)
        -> impl Future<Output = Result<Self, Self::Error>> + Send
    where
        P: StorageReadProvider;
}
```

The generic `T` parameter carries the context needed to derive file paths. In practice this is either a file-path prefix `String`, a structured `AsyncIndexMetadata` (wrapping a prefix string), or a tuple thereof.

### BfTreeProvider

| Trait | Auxiliary type | Description |
|---|---|---|
| `SaveWith` | `String` | The `String` is the file-path prefix |
| `LoadWith` | `String` | Same |

**Files written on save:**

- `{prefix}_params.json` — JSON configuration blob (`SavedParams`): dimension, metric, `max_degree`, BfTree configuration parameters, quantization parameters (if PQ is enabled), and graph hyperparameters.
- `{prefix}_vectors.bftree` — BfTree snapshot of full-precision vector data (copy of the in-memory BfTree snapshot).
- `{prefix}_neighbors.bftree` — BfTree snapshot of the graph adjacency lists.
- `{prefix}_quant.bftree` — BfTree snapshot of quantized vectors *(PQ variant only)*.
- `{prefix}_pq_pivots.bin` — PQ pivot table and centroids *(PQ variant only)*.
- `{prefix}_deleted.bin` — Serialized delete bitmap.

**On load:**

1. Reads `_params.json` to reconstruct the `BfTree` configs and index parameters.
2. Opens BfTree snapshots (using `BfTree::new_from_snapshot`) for full vectors, neighbors, and quant vectors.
3. Loads the PQ pivot table from `_pq_pivots.bin` *(PQ variant only)*.
4. Loads the delete bitmap from `_deleted.bin`, or creates an empty bitmap if the file does not exist.

### DefaultProvider

| Trait | Auxiliary type | Description |
|---|---|---|
| `SaveWith` | `(u32, AsyncIndexMetadata)` | `u32` is the start-point ID; `AsyncIndexMetadata` wraps the file-path prefix |
| `SaveWith` | `(u32, u32, DiskGraphOnly)` | Graph-only save for disk-index construction |
| `LoadWith` | `AsyncQuantLoadContext` | Prefix + frozen-point count + metric + prefetch hints |

**Files written on save (`(u32, AsyncIndexMetadata)` variant):**

- `{prefix}.data` — Full-precision vectors in the standard `.bin` format (via `MemoryVectorProviderAsync` / `FastMemoryVectorProviderAsync`).
- `{prefix}_build_pq_compressed.bin` / `{prefix}_sq_compressed.bin` — Quantized vectors *(if a quant store is present)*.
- `{prefix}` (raw prefix, no extension) — Graph adjacency list in `.bin` graph format (via `SimpleNeighborProviderAsync::save_direct()`).

> **Note:** The delete store (`TableDeleteProviderAsync`) is **not** persisted; it is reconstructed empty via `LoadWith<usize>`.

**On load (`AsyncQuantLoadContext`):**

1. Loads full-precision vectors from `{prefix}.data`.
2. Loads quant vectors from the compressed bin file *(if present)*.
3. Loads the graph from the raw prefix file via `SimpleNeighborProviderAsync::load_direct()`.
4. Constructs an empty delete store from the point count.

### DiskProvider

`DiskProvider` implements `LoadWith<AsyncDiskLoadContext>` but **has no `SaveWith` implementation**. Disk-index creation is handled externally by `DiskIndexWriter::create_disk_layout()`, which interleaves the vector data, neighbor lists, and associated data into a sector-aligned binary file. The `DiskIndexWriter` is not integrated with the `SaveWith`/`LoadWith` trait family.

---

## Proposed Design

### Attribute Store Serialization Interface

Rather than extending the `AttributeStore` trait (which would impose serialization concerns on all implementations), we propose adding `SaveWith<T>` and `LoadWith<T>` directly on `RoaringAttributeStore` for the relevant auxiliary types. The attribute store is responsible for extracting whatever path information it needs from `T`.

The `DocumentProvider` impls will then require the attribute store bound:

```rust
impl<DP, AS, T> SaveWith<T> for DocumentProvider<DP, AS>
where
    DP: SaveWith<T>,
    AS: SaveWith<T, Ok = (), Error = ANNError>,
    ...
```

```rust
impl<DP, AS, T> LoadWith<T> for DocumentProvider<DP, AS>
where
    DP: LoadWith<T>,
    AS: LoadWith<T, Error = ANNError>,
    ...
```

This is deliberately narrow: concrete implementations are only required for `RoaringAttributeStore` for now, keeping the door open for future implementations.

### Label File Format

The label data is persisted to a single binary file at `{prefix}.labels.bin`. `RoaringAttributeStore` is responsible for deriving the path from the auxiliary type `T` passed to its `SaveWith<T>` / `LoadWith<T>` implementation. The format uses little-endian byte order throughout and is designed for straightforward sequential writes and reads.

```text
┌────────────────────────────────────────────────────────────────────┐
│  Header (16 bytes)                                                 │
│  [u64: num_attribute_entries]                                       │
│  [u64: forward_index_offset]  (byte offset from file start to      │
│                                Section 2)                          │
├────────────────────────────────────────────────────────────────────┤
│  Section 1: Attribute Dictionary                                   │
│  Repeated `num_attribute_entries` times:                           │
│                                                                    │
│  [u64: attribute_id]                                               │
│  [u32: field_name_byte_len]                                        │
│  [u8 * field_name_byte_len: UTF-8 field name]                      │
│  [u8: type_tag]                                                    │
│      0 = Bool                                                      │
│      1 = Integer                                                   │
│      2 = Real                                                      │
│      3 = String                                                    │
│      4 = Empty                                                     │
│  [value bytes, depends on type_tag]:                               │
│      Bool:    1 byte  (0 = false, 1 = true)                        │
│      Integer: 8 bytes (i64, little-endian)                         │
│      Real:    8 bytes (f64, little-endian)                         │
│      String:  [u32: byte_len] + [u8 * byte_len: UTF-8 value]       │
│      Empty:   0 bytes                                              │
├────────────────────────────────────────────────────────────────────┤
│  Section 2: Forward Index                                          │
│  [u64: num_nodes_with_labels]                                      │
│  Repeated `num_nodes_with_labels` times:                           │
│                                                                    │
│  [u32: node_internal_id]                                           │
│  [u32: num_attribute_ids_for_this_node]                            │
│  [u64 * num_attribute_ids: attribute IDs (sorted ascending)]       │
└────────────────────────────────────────────────────────────────────┘
```

**Notes:**

- The inverted index is **not persisted**; it is rebuilt from the forward index during `load_with`.
- `node_internal_id` is currently `u32` (matching `VectorId = u32` throughout the system). If the type is generalised in the future, this field and the file format version must be updated.
- The attribute dictionary is written first so that a reader can build the reverse mapping (`u64` ID → `Attribute`) before scanning the forward index.

### DocumentProvider SaveWith Implementation

```rust
impl<DP, AS, T> SaveWith<T> for DocumentProvider<DP, AS>
where
    DP: DataProvider + SaveWith<T, Error = ANNError>,
    AS: AttributeStore<DP::InternalId> + SaveWith<T, Ok = (), Error = ANNError> + AsyncFriendly,
    ANNError: From<DP::Error>,
{
    type Ok = ();
    type Error = ANNError;

    async fn save_with<P>(
        &self,
        provider: &P,
        auxiliary: &T,
    ) -> Result<(), ANNError>
    where
        P: StorageWriteProvider,
    {
        // 1. Delegate to inner provider.
        self.inner_provider
            .save_with(provider, auxiliary)
            .await?;

        // 2. Persist the attribute store.
        self.attribute_store
            .save_with(provider, auxiliary)
            .await?;

        Ok(())
    }
}
```

The `RoaringAttributeStore::save_with` implementation must:

1. Acquire read locks on `attribute_map` and `index`.
2. Write the header.
3. Iterate over all `(InternalAttribute, u64)` pairs in `AttributeEncoder` via `AttributeEncoder::for_each` (already present, currently marked `dead_code`) and write the dictionary section.
4. Iterate over all `(node_id, attribute_id_set)` pairs in the forward index and write the forward index section.

### DocumentProvider LoadWith Implementation

```rust
impl<DP, AS, T> LoadWith<T> for DocumentProvider<DP, AS>
where
    DP: DataProvider + LoadWith<T, Error = ANNError>,
    AS: AttributeStore<DP::InternalId> + LoadWith<T, Error = ANNError> + AsyncFriendly,
    ANNError: From<DP::Error>,
{
    type Error = ANNError;

    async fn load_with<P>(
        provider: &P,
        auxiliary: &T,
    ) -> Result<Self, ANNError>
    where
        P: StorageReadProvider,
    {
        // 1. Load the inner provider.
        let inner_provider = DP::load_with(provider, auxiliary).await?;

        // 2. Load the attribute store.
        let attribute_store = AS::load_with(provider, auxiliary).await?;

        Ok(Self {
            inner_provider,
            attribute_store,
        })
    }
}
```

The `RoaringAttributeStore::load_with` implementation must:

1. Read the header to obtain `num_attribute_entries` and `forward_index_offset`.
2. Decode the attribute dictionary, inserting each `(u64 id, Attribute)` pair into the `AttributeEncoder`.
3. Seek to `forward_index_offset` and decode the forward index, inserting into the `RoaringTreemapSetProvider`.
4. Rebuild the inverted index from the forward index (iterate node → attribute-IDs, invert to attribute-ID → nodes).

---

## Open Questions

### DiskProvider Compatibility

`DiskProvider` uses `DiskIndexWriter::create_disk_layout()` for serialization rather than the `SaveWith` trait. There is therefore no `SaveWith` implementation through which label data could be co-written. Consequently, `DocumentProvider<DiskProvider<_>, AS>` would only support `LoadWith`, not `SaveWith`.

For `LoadWith` to work, the label file must have been written separately (e.g. by calling `attribute_store.save_with(...)` directly during the disk-index build pipeline before creating the disk layout). `DiskIndexWriter` would need to be extended or wrapped to also write the label file at `{prefix}.labels.bin` as part of `create_disk_layout` or a new `create_disk_layout_with_labels` variant.
