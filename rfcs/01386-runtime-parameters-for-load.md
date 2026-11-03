# Runtime Parameters for Load

| | |
|---|---|
| **Authors** | Suhas Jayaram Subramanya (suhasjs) |
| **Contributors** | |
| **Created** | 2026-09-08 |
| **Updated** | 2026-11-03 |

## Summary

Let callers pass settings and resources when loading saved data with `diskann-record`. Keep loading without these extra inputs unchanged. Both paths use the same code to read the saved data. The design is inspired by Serde's [`DeserializeSeed`](https://docs.rs/serde/latest/serde/de/trait.DeserializeSeed.html).

## Motivation

**Want**: A saved in-memory index should contain its live nodes and frozen start points (fixed search entry points), not empty slots or state used to manage concurrent access. A caller loading 100,000 saved nodes might want room for one million. That capacity is a choice made when loading and should not be part of the saved index data.

The existing `Load` API receives only saved data. A loader supplies the extra inputs: capacity, shared resources, storage, or an existing value to fill. Different loaders can produce the same type; that type does not have to declare which settings they accept.

[`diskann-inmem::Provider`](../diskann-inmem/src/provider.rs) is the motivating use case. Its [full-precision configuration](../diskann-inmem/src/repr/full.rs) combines data that should be saved, such as the distance metric and start points, with load-time settings like capacity, concurrency, and prefetching. Since it has no save/load support today, this proposal intends to add that support with caller-supplied settings.

## The `Loader` trait

Keep `Load` and `Loadable` unchanged and add a `Loader` trait. The code below sketches the proposed API.

```rust
pub trait Loader<'a>: Sized {
    type Value;

    fn load(self, context: Context<'a>) -> Result<Self::Value>;
}
```

`load` takes ownership of the loader, so it can pass storage to the result or write into an existing value. `Value` need not implement `Load`; a loader that fills an existing value may return `()`. Implement `Loader` for `PhantomData<T>` where `T: Loadable<'a>` by calling `Loadable::load`, so ordinary loading works with the new API too.

### Dedup `Load` + `Loader` with DTOs

Use a separate struct -- a data transfer object (DTO) -- to hold the saved fields. Its `Load` impl reads those fields and converts from any supported older formats. It does not contain runtime settings or concurrency state. Both loading paths can use the same code to build the result from it:

```text
(ordinary) Load   -> DTO decoding  -> restore(dto, default options)
(custom)   Loader -> DTO decoding  -> restore(dto, supplied options)
```

The result type's ordinary `Load` uses the DTO's `VERSION`, `load`, and `load_legacy`, then builds the result with default settings. A loader calls `context.load::<Dto>()?` and builds it with caller-supplied settings. This way, both paths can share the same codepath to read and check saved data.

**Where the DTO lives**: We don't need a new trait in `diskann-record` for DTOs, so define the DTO in the crate that defines the saved format. The DTO itself need not implement `Save` -- we just need it to understand the saved format.

### Loading Nested Values

Add `*_with*` helpers that accept a loader:

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

When loading a field, pass a loader with the settings that field needs. Alternatively, read its DTO with an ordinary `field` call and pass settings when building the result. Existing `load`, `field`, `load_fields!`, and support for basic types and collections can stay unchanged.

Settings are not passed automatically through `Option<T>` or `Vec<T>`. Callers handle each present value or element with a loader, passing shared resources again as needed. Loaders do not have to support copying or cloning.

### Versions and Errors

`Loader` will not do any version checks or sanity checks for persisted data (the DTO should keep that responsibility). Instead, it will only do _runtime_-related sanity checks (e.g., ensuring scratch space is large enough, or that capacity is sufficient). Invalid settings, corrupt data, and memory-allocation failures can throw errors from `Loader` itself (needs new `Error` types). 

## Sketch for In-Mem 2.0

Let's start with `Provider<Full<f32>, u32>` whose saved vectors are in `f32` and external IDs in `u32`. The provider holds the ID map, `Full` holds vectors, nbr lists (graph), and start points. Let's separate the saved data from runtime settings and state:

| Saved data | Supplied or rebuilt when loading |
|---|---|
| Live and frozen counts, vector dimension | Capacity for writable slots |
| Distance metric and max nbrs per node | Store concurrency, prefetch settings |
| Live and frozen vectors, nbr lists | Slot state, free-slot lists, reader tracking, and locks |
| External IDs in the same row order as the saved live vectors | Two-way ID map and counters |

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

`capacity: None` uses the saved live-node count, and if supplied, it must be at least that count. Other defaults use `store::Config::default()` and `Full`'s default prefetch settings. Ordinary `Load` will use the same DTO and build steps, but with default options. This code wil live in the `diskann-inmem` crate.

### Saved ID Layout

Let `F` be the frozen-point count, `N` the live-node count, and `C` the requested writable capacity, with `C >= N`. As in the current `Full` layout, frozen points come first:

| Kind | Saved IDs | Restored IDs |
|---|---|---|
| Frozen points | `[0, F)` | `[0, F)` |
| Live nodes | `[F, F + N)` | `[F, F + N)` |
| Spare writable slots | Not saved | `[F + N, F + C)` |

Increasing capacity only adds empty slots, so we don't need to move any graph IDs on load. This means the load path should be blazing fast for any 'capacity' value `>= N`.
    
### Taking care of 'holes'

The save path for in-mem 2.0 is a little more involved because the live ID space can contain 'holes' from deletions. So, on `Save`, live nodes must be compacted (**writes must be paused here**): 

1. Check that live slots and the ID map agree, keep frozen IDs and assign live nodes, in their current ID order, to `[F, F + N)`. This creates a temporary map from `old ID -> new ID`.
2. Write IDs, vectors, and graph rows using that same mapping, including frozen neighbor lists. Process the nbr list in a streaming manner, and update edge targets to use new IDs.

`Save` will not persist capacity.