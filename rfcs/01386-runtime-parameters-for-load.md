# Runtime Parameters for Load

| | |
|---|---|
| **Authors** | Suhas Jayaram Subramanya (suhasjs) |
| **Contributors** | |
| **Created** | 2026-09-08 |
| **Updated** | 2026-11-02 |

## Summary

Let callers pass settings and resources when loading saved data with `diskann-record`. Keep loading without these extra inputs unchanged. Both paths use the same code to read the saved data. The design is inspired by Serde's [`DeserializeSeed`](https://docs.rs/serde/latest/serde/de/trait.DeserializeSeed.html).

## Motivation

**Want**: A saved in-memory index should contain its live nodes and frozen start points (fixed search entry points), not empty slots or state used to manage concurrent access. A caller loading 100,000 saved nodes might want room for one million. That capacity is a choice made when loading and should not be part of the saved index data.

The existing `Load` API receives only saved data. A loader supplies the extra inputs: capacity, shared resources, storage, or an existing value to fill. Different loaders can produce the same type; that type does not have to declare which settings they accept.

[`diskann-inmem::Provider`](../diskann-inmem/src/provider.rs) is the motivating use case. Its [full-precision configuration](../diskann-inmem/src/repr/full.rs) combines data that should be saved, such as the distance metric and start points, with load-time settings like capacity, concurrency, and prefetching. Since it has no save/load support today, this proposal intends to add that support with caller-supplied settings.

## Proposal

Keep `Load` and `Loadable` unchanged and add a `Loader` trait. The code below sketches the proposed API.

```rust
pub trait Loader<'a>: Sized {
    type Value;

    fn load(self, context: Context<'a>) -> Result<Self::Value>;
}
```

`load` takes ownership of the loader, so it can pass storage to the result or write into an existing value. `Value` need not implement `Load`; a loader that fills an existing value may return `()`. Implement `Loader` for `PhantomData<T>` where `T: Loadable<'a>` by calling `Loadable::load`, so ordinary loading works with the new API too.

Caller-supplied settings and resources stay in the loader, not in `Context`, `Object`, or the storage backend, and are not expected to be saved. Resources used by the result must remain valid while it uses them; they are not tied to how long the saved metadata is available (`'a`).

### Shared Saved Data

Use a separate struct, aka data transfer object (DTO), to hold the saved fields. Its `Load` implementation reads those fields and converts from any supported older formats. It does not contain runtime settings or concurrency state. Both loading paths can use the same code to build the result from it:

```text
ordinary Load -> DTO decoding -> restore(dto, default options)
    Loader -> DTO decoding -> restore(dto, supplied options)
```

The result type's ordinary `Load` uses the DTO's `VERSION`, `load`, and `load_legacy`, then builds the result with default settings. A loader calls `context.load::<Dto>()?` and builds it with caller-supplied settings. Neither path needs its own copy of the code that reads and checks saved data.

For small records, the DTO can hold all the data. For large ones, it can hold metadata, file handles, and an `Object<'a>` used to open the saved files. Building the result then reads and checks those files directly, without keeping a second full copy in memory. Reading metadata alone does not check the file contents. Finish using the DTO before the disk context is dropped.

Define the DTO in the crate that defines the saved format; `diskann-record` needs no new trait for DTOs. Saving writes the same format, either from the running index or through a wrapper that reads from it without copying it. The DTO itself need not implement `Save`.

### Loading Nested Values

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

When loading a field, pass a loader with the settings that field needs. Alternatively, read its DTO with an ordinary `field` call and pass settings when building the result. Existing `load`, `field`, `load_fields!`, and support for basic types and collections stay unchanged.

Settings are not passed automatically through `Option<T>` or `Vec<T>`. Callers handle each present value or element with a loader, passing shared resources again as needed. Loaders do not have to support copying or cloning.

### Versions and Errors

`Loader` does not choose how to read a format version automatically. Calling `context.load::<Dto>()` uses the existing record checks, version handling, and error categories. A loader that reads data directly must handle these itself. Caller settings cannot change how saved fields are interpreted. The record format and storage backends stay unchanged.

Invalid settings, corrupt data, and reported memory-allocation failures are critical errors, not reasons to try another format. This proposal does not add error handling to allocations that lack it today. Callers may retry recoverable errors, but each attempt needs a new loader and must start with unchanged or reset external data. Loading into an existing value may leave it partly changed on failure.

## Example: In-Memory Provider

Start with `Provider<Full<f32>, u32>`: saved vectors use `f32` and external IDs use `u32`. These are fixed, not choices made when loading. The provider holds the ID map; `Full` holds vectors, neighbor lists, and start points. Separate the saved data from runtime settings and state:

| Saved data | Supplied or rebuilt when loading |
|---|---|
| Live and frozen counts, vector dimension | Capacity for writable slots |
| Distance measure and maximum neighbors per node | Store concurrency and prefetch settings |
| Live and frozen vectors, neighbor lists | Slot state, free-slot lists, reader tracking, and locks |
| External IDs in the same row order as saved live vectors | Two-way ID map and counters |

### Loading

A provider DTO refers to the saved external IDs and contains another DTO for the vector and graph files. Its `restore` method reads these into a new provider:

```rust
use diskann_record::load;
use crate::{Provider, num::Capacity, repr::Full, store};

type FullProvider = Provider<Full<f32>, u32>;

#[derive(Default)]
struct FullLoadOptions {
    capacity: Option<Capacity>,
    store: store::Config,
}

struct FullProviderLoader {
    options: FullLoadOptions,
}

impl<'a> load::Loader<'a> for FullProviderLoader {
    type Value = FullProvider;

    fn load(self, context: load::Context<'a>) -> load::Result<FullProvider> {
        let dto: FullProviderDto<'a> = context.load()?;
        dto.restore(self.options)
    }
}
```

`capacity: None` uses the saved live-node count; a supplied capacity must be at least that count. Capacity counts writable slots, excluding frozen points. Other defaults use `store::Config::default()` and `Full`'s default prefetch settings. Ordinary `Load` uses the same DTO and build steps with default options. This requires new internal loading code in `diskann-inmem`.

### Saved ID Layout

Let `F` be the frozen-point count, `N` the live-node count, and `C` the requested writable capacity, with `C >= N`. As in the current `Full` layout, frozen points come first:

| Kind | Saved IDs | Restored IDs |
|---|---|---|
| Frozen points | `[0, F)` | `[0, F)` |
| Live nodes | `[F, F + N)` | `[F, F + N)` |
| Spare writable slots | Not saved | `[F + N, F + C)` |

Increasing capacity only adds empty slots; no graph IDs move on load. Live-vector and external-ID rows use IDs `[F, F + N)`, frozen-vector rows use `[0, F)`, and graph rows use `[0, F + N)`. Edges may only point to saved nodes, never to empty slots in the new capacity. Other implementations of `Representation` need not use this layout.

### Saving After Deletions

Deletions leave gaps in the live IDs. Saving removes those gaps in two steps, with all writes paused:

1. Check that live slots and the ID map agree. Keep frozen IDs and assign live nodes, in their current ID order, to `[F, F + N)`.
2. Write IDs, vectors, and graph rows using that same mapping, including frozen neighbor lists. Update edge targets to the saved IDs, drop edges to known deleted nodes, and keep the remaining neighbors in order. Write the new neighbor count. Invalid targets or inconsistent mappings are errors.

Build the complete mapping before writing edges. Share it across the save code without copying or changing the graph. Keep writes paused through both steps, including graph updates; locking one row at a time does not keep the whole snapshot consistent. Do not save runtime capacity.

### Checks When Loading

- Check `C >= N` and that `F + C` fits `IdLimit` before allocating memory.
- Check dimensions, file lengths, row and neighbor counts, edges to saved nodes, and that external IDs are unique.
- Load vectors and neighbor lists, including frozen rows, at their saved IDs without rebuilding the graph. Size the ID map for `F + C`, but add entries only for live nodes.
- Set up slot state and concurrency tracking so only empty slots can be used for insertion. Return the provider only after all data has been read, checked, and initialized.

## Loading From Disk

Add an internal `load_with` function that takes a context alongside the existing `load`, plus a public function for loading from disk:

```rust
pub fn load_from_disk_with<L, T>(
    metadata: &Path,
    artifacts: &Path,
    loader: L,
) -> Result<T>
where
    L: for<'a> Loader<'a, Value = T>;
```

The disk function creates a context and drops it before returning. The `for<'a>` requirement ensures the result cannot keep references to that context. It may still use resources supplied by the caller; those do not have to be `'static`. With a caller-supplied context, `load_with` uses `L: Loader<'a>` and allows the result to reference that context.

```rust
let provider: FullProvider = load_from_disk_with(
    &metadata_path,
    &artifact_dir,
    FullProviderLoader {
        options: FullLoadOptions {
            capacity: Some(Capacity::new(200_000)),
            ..FullLoadOptions::default()
        },
    },
)?;
```

Existing `load_from_disk::<T>` remains available and can call the new API using `PhantomData<T>`.

## Testing and Scope

The initial implementation should test:

- Saving and loading at the live-node count and at larger capacities, preserving vectors, external IDs, and graph edges, including frozen rows. Later inserts must use only empty slots.
- Saving after deletions, with matching ID, vector, and graph rows and no changes to the source graph. Include multiple frozen points, live IDs at or above writable capacity, and an empty index containing only frozen points.
- The same results and errors with ordinary loading and a loader using default settings, including supported older formats.
- Rejection of too-small capacity, ID overflow, duplicate external IDs, invalid edges (including into empty slots), and corrupt files. Never return an incomplete provider or give frozen points external IDs.
- Reading large files through a DTO that references the disk context, and loaders using caller-supplied resources that are not `'static`.

Later work can add save/load support for quantized data, snapshots without pausing writers, reusable DTO and collection helpers, macro changes, public APIs for loading from memory, and loading into existing values with rollback on failure.