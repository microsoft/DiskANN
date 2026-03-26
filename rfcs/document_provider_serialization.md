# RFC: SaveWith and LoadWith for DocumentProvider

|                  |                  |
| ---------------- | ---------------- |
| **Authors**      | Sampath Rajendra |
| **Contributors** | Sampath Rajendra |
| **Created**      | 2026-03-26       |
| **Updated**      | 2026-03-26       |

## Summary

This RFC proposes implementing the `SaveWith` and `LoadWith` serialization traits for `DocumentProvider`. `DocumentProvider` wraps an inner `DataProvider` (e.g. `DefaultProvider` or `BfTreeProvider`) and an `AttributeStore` that holds the per-vector label data. Serialization must handle both: delegating to the inner provider's own save/load and persisting the attribute store to a separate binary file.

## Motivation

### Background

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

The `SaveWith` and `LoadWith` traits are defined in `diskann-providers::storage`:

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

The generic `T` parameter carries the context needed to derive file paths — in practice a file-path prefix `String`, a structured `AsyncIndexMetadata` (wrapping a prefix string), or a tuple thereof.

Existing serialization patterns for the inner provider types:

**`BfTreeProvider`** (`SaveWith<String>` / `LoadWith<String>`): writes `_params.json`, `_vectors.bftree`, `_neighbors.bftree`, `_deleted.bin`, and PQ files when applicable.

**`DefaultProvider`** (`SaveWith<(u32, AsyncIndexMetadata)>` / `LoadWith<AsyncQuantLoadContext>`): writes `{prefix}.data`, compressed vector files, and the raw graph file. The delete store is not persisted and is reconstructed empty on load.

**`DiskProvider`** (`LoadWith<AsyncDiskLoadContext>` only): has no `SaveWith` implementation. Disk-index creation is handled externally by `DiskIndexWriter::create_disk_layout()`, which is not integrated with the `SaveWith`/`LoadWith` trait family.

### Problem Statement

`DocumentProvider` implements `DataProvider`, `SetElement`, and `Delete` but has no serialization support. An index built with `DocumentProvider` cannot be persisted and reloaded; every restart requires rebuilding the index from scratch, including re-encoding all attribute data.

### Goals

1. Implement `SaveWith<T>` and `LoadWith<T>` for `DocumentProvider`, delegating to the inner provider and the attribute store respectively.
2. Define a stable binary file format (`{prefix}.labels.bin`) for persisting `RoaringAttributeStore` label data.
3. Implement `SaveWith<T>` and `LoadWith<T>` directly on `RoaringAttributeStore` without widening the `AttributeStore` trait.

## Proposal

The `diskann-label-filter` crate takes a new dependency on `diskann-providers` (for the `SaveWith`/`LoadWith` trait definitions and the `StorageReadProvider`/`StorageWriteProvider` abstractions).

### Attribute Store Serialization Interface

Rather than extending the `AttributeStore` trait (which would impose serialization concerns on all implementations), `SaveWith<T>` and `LoadWith<T>` are added directly on `RoaringAttributeStore` for the relevant auxiliary types. The attribute store is responsible for extracting whatever path information it needs from `T`.

The `DocumentProvider` impls require the attribute store bound:

```rust
impl<DP, AS, T> SaveWith<T> for DocumentProvider<DP, AS>
where
    DP: DataProvider + SaveWith<T, Error = ANNError>,
    AS: AttributeStore<DP::InternalId> + SaveWith<T, Ok = (), Error = ANNError> + AsyncFriendly,
    ANNError: From<DP::Error>,
{ ... }
```

```rust
impl<DP, AS, T> LoadWith<T> for DocumentProvider<DP, AS>
where
    DP: DataProvider + LoadWith<T, Error = ANNError>,
    AS: AttributeStore<DP::InternalId> + LoadWith<T, Error = ANNError> + AsyncFriendly,
    ANNError: From<DP::Error>,
{ ... }
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

## Trade-offs

### Keeping serialization off the `AttributeStore` trait

Adding `SaveWith`/`LoadWith` directly to `RoaringAttributeStore` rather than to the `AttributeStore` trait avoids imposing a serialization requirement on every future `AttributeStore` implementation. The cost is that `DocumentProvider<DP, AS>` can only be saved or loaded when `AS` satisfies the respective bound — callers using a hypothetical `AS` that does not implement `SaveWith` would need to handle serialization manually or outside of this trait pair.

### DiskProvider compatibility

`DiskProvider` has no `SaveWith` implementation; disk-index creation is handled by `DiskIndexWriter::create_disk_layout()`, which is not integrated with the `SaveWith`/`LoadWith` trait family. Consequently, `DocumentProvider<DiskProvider<_>, AS>` supports `LoadWith` only.

For `LoadWith` to work, the label file (`{prefix}.labels.bin`) must have been written separately — e.g. by calling `attribute_store.save_with(...)` directly during the disk-index build pipeline before the disk layout is finalized. The alternative of integrating label-file writing into `DiskIndexWriter` is deferred to future work (see below).

## Benchmark Results

Not applicable for this RFC.

## Future Work

- [ ] Extend `DiskIndexWriter` (or introduce a `create_disk_layout_with_labels` variant) to co-write `{prefix}.labels.bin` during disk-index construction, enabling a full `SaveWith`/`LoadWith` round-trip for `DocumentProvider<DiskProvider<_>, AS>`.
- [ ] If `VectorId` is ever generalised beyond `u32`, update the `node_internal_id` field width and introduce a file format version field.

## References

1. [`diskann-providers` — `SaveWith` / `LoadWith` trait definitions](../diskann-providers/src/)
2. [`diskann-label-filter` — `DocumentProvider`, `RoaringAttributeStore`](../diskann-label-filter/src/)
