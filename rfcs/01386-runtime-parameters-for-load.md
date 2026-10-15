# Runtime Parameters for Load

| | |
|---|---|
| **Authors** | Suhas Jayaram Subramanya (suhasjs) |
| **Contributors** | |
| **Created** | 2026-09-08 |
| **Updated** | 2026-09-09 |

## Summary

Add stateful loading to `diskann-record`, inspired by Serde's [`DeserializeSeed`](https://docs.rs/serde/latest/serde/de/trait.DeserializeSeed.html). A caller supplies a loader that carries construction state and declares the type it produces. Existing stateless loading remains unchanged; target types do not need an associated runtime-parameter type.

## Motivation

A saved in-memory index contains only its live nodes. Loading 100,000 saved nodes may construct an index with capacity for 200,000, one million, or ten million nodes. Capacity is not persisted because it describes the new runtime allocation rather than the logical index.

The existing `Load` API receives only persisted data. A loader supplies the missing construction state: capacity settings, shared resources, owned storage, or a mutable destination. Different loaders may construct the same target type; there is no requirement for one runtime contract per target.

## Proposal

Keep `Load` and `Loadable` unchanged and add the following trait. The examples sketch the proposed API, not the current implementation.

```rust
pub trait Loader<'a>: Sized {
    type Value;

    fn load(self, context: Context<'a>) -> Result<Self::Value>;
}
```

The loader, rather than the output type, implements the operation. `Value` need not implement `Load` or `Loadable`; a loader loading into an existing destination may return `()`. Consuming the loader allows owned resources to move into the result and mutable references to be used without requiring interior mutability.

As in Serde, `PhantomData<T>` adapts ordinary loading to the new API:

```rust
impl<'a, T: Loadable<'a>> Loader<'a> for std::marker::PhantomData<T> {
    type Value = T;

    fn load(self, context: Context<'a>) -> Result<T> {
        <T as Loadable<'a>>::load(context)
    }
}
```

Runtime state is not stored in `Context`, `Object`, or the backend and is not serialized. A loader's resource lifetimes are independent of the manifest lifetime `'a`; any resources borrowed by the output must have their lifetimes represented in the loader and output types.

### Composition

Add helpers that accept a loader:

```rust
impl<'a> Context<'a> {
    pub fn load_with<L: Loader<'a>>(&self, loader: L) -> Result<L::Value> {
        loader.load(self.clone())
    }
}

impl<'a> Object<'a> {
    pub fn field_with<L: Loader<'a>>(&self, key: &str, loader: L) -> Result<L::Value> {
        self.child(key)?.load_with(loader)
    }
}
```

Existing `Context::load`, `Object::field`, `load_fields!`, and all primitive and collection implementations remain unchanged. A parent explicitly supplies loaders for stateful children, for example `object.field_with("graph", GraphLoader { max_slots })?`, and uses ordinary field loading for stateless children.

There is no implicit propagation into `Option<T>` or `Vec<T>`. Stateful collections use explicit traversal, constructing a loader for each present value or element and reborrowing shared state as needed. Loaders need not be `Clone`; generic collection adapters and macro extensions are outside this proposal.

### Schema Versions and Errors

`Loader` receives a `Context`; it does not automatically invoke the `Loadable` version dispatcher. A simple approach is to load a persisted representation through `Context::load` and then construct the runtime value. This retains the existing object-shape check, `Load::load` / `load_legacy` dispatch, and error classification.

A loader may instead decode directly, including streaming sidecar artifacts, but is then responsible for the same shape and version checks and any legacy upgrades it supports. Persisted schema information, not runtime settings, determines field encodings and decoding paths. No persisted format changes are required.

Invalid runtime settings and allocation failures are critical errors, not reasons to try another schema. Probing remains caller-controlled and retries only recoverable errors. Since loaders are consumed and may mutate external state, each attempt needs a fresh loader or reborrow; retrying also requires unchanged or restored external state. Prefer checking schema compatibility before mutating a destination.

## Example

Use inline adjacency lists for this small example:

```rust
struct Metadata {
    name: String, // Saved.
}

struct Graph {
    adjacency: Vec<Vec<u32>>, // Saved, one row per live node.
    max_slots: usize,         // Runtime limit; not saved.
}

struct Index {
    metadata: Metadata,
    graph: Graph,
    scratch: Vec<u32>, // Runtime scratch; not saved.
}
```

Assume all three already implement `Save` and `Load`; the existing blanket implementation supplies `Loadable`. Their loaders handle schema version `0.0`, reject unsupported versions, and validate saved fields and graph edges. Ordinary loading sets `Graph::max_slots` to `adjacency.len()` and initializes `Index::scratch` with `Vec::new()`.

`IndexLoader` loads metadata normally and supplies a `GraphLoader` for the graph field. The child receives only its slot limit; scratch allocation belongs to the parent.

```rust
struct IndexLoader {
    max_slots: usize,
    scratch_capacity: usize,
}

struct GraphLoader {
    max_slots: usize,
}

impl<'a> load::Loader<'a> for IndexLoader {
    type Value = Index;

    fn load(self, context: load::Context<'a>) -> load::Result<Index> {
        let object = context.as_object().ok_or(load::error::Kind::TypeMismatch)?;
        if object.version() != <Index as load::Load<'a>>::VERSION {
            return Err(load::error::Kind::UnknownVersion.into());
        }

        let metadata: Metadata = object.field("metadata")?;
        let graph = object.field_with("graph", GraphLoader { max_slots: self.max_slots })?;
        let mut scratch = Vec::new();
        scratch
            .try_reserve_exact(self.scratch_capacity)
            .map_err(load::Error::new)?;

        Ok(Index { metadata, graph, scratch })
    }
}

impl<'a> load::Loader<'a> for GraphLoader {
    type Value = Graph;

    fn load(self, context: load::Context<'a>) -> load::Result<Graph> {
        let object = context.as_object().ok_or(load::error::Kind::TypeMismatch)?;
        if object.version() != <Graph as load::Load<'a>>::VERSION {
            return Err(load::error::Kind::UnknownVersion.into());
        }

        load_fields!(object, [adjacency: Vec<Vec<u32>>]);
        let mut adjacency = adjacency;
        let live = adjacency.len();
        if self.max_slots < live {
            return Err(load::Error::message("max_slots is smaller than the live node count"));
        }
        for &target in adjacency.iter().flatten() {
            if usize::try_from(target).map_err(load::Error::new)? >= live {
                return Err(load::Error::message("edge targets a nonexistent node"));
            }
        }
        adjacency.try_reserve_exact(self.max_slots - live).map_err(load::Error::new)?;

        Ok(Graph { adjacency, max_slots: self.max_slots })
    }
}
```

The keys `"metadata"`, `"graph"`, and `"adjacency"` match the fields written by the existing `Save` implementations. `field("metadata")` retains ordinary `Loadable` dispatch; `field_with("graph", ...)` invokes the supplied child loader. `load_fields!` derives the `"adjacency"` key from the binding name and calls `object.field("adjacency")`. Both direct loaders check their own object shape and version rather than calling `Index::load` or `Graph::load`; this example supports no legacy schemas.

`GraphLoader` decodes adjacency through ordinary collection loading, then validates and reserves capacity; this may reallocate. Reservation guarantees at least the requested capacity without adding live nodes, and `max_slots` remains the logical limit. Invalid capacity, invalid edges, and reservation failures are critical errors. Saving again writes only metadata and live adjacency rows, never capacity or scratch state. Production implementations should share schema checks and validation with ordinary loading.

### Backend Entry Points

Add a crate-private context-based `load_with` alongside the existing `load`, and a public disk convenience function:

```rust
pub fn load_from_disk_with<L, T>(
    metadata: &Path,
    artifacts: &Path,
    loader: L,
) -> Result<T>
where
    L: for<'a> Loader<'a, Value = T>;
```

The disk function creates a temporary context and invokes the loader. Fixing `Value = T` across context lifetimes prevents the result from borrowing that temporary context; it does not require the loader or result to be `'static`, so caller-owned resources may still be borrowed. The context-based entry point uses `L: Loader<'a>` for a caller-owned context borrowed for `'a`, allowing results to borrow from it.

```rust
let index: Index = load_from_disk_with(
    &metadata_path,
    &artifact_dir,
    IndexLoader { max_slots: 200_000, scratch_capacity: 64 },
)?;
```

Existing `load_from_disk::<T>` remains available without an explicit loader; stateless loading can delegate through `PhantomData<T>`. Disk and memory backends remain responsible only for manifests and artifacts. Exposing public memory execution APIs is outside this proposal.