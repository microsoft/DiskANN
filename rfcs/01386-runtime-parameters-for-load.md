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

### Sharing the Load Path

Put decoding, version handling, validation, and construction in the loader. When all required runtime inputs have meaningful defaults, implement `Loadable` for the result by delegating to a default loader:

```text
(ordinary) Loadable -> default loader ---------+
(custom)   Loader with supplied inputs -------+-> decode, validate, allocate, restore
```

This shares the entire load path without a separate saved-data type. Implement `Loadable` directly for this adapter, not `Load`: `Loadable::load` receives a `Context`, while `Load::load` receives an `Object`. Do not implement both, since `Load` already provides a blanket `Loadable` implementation.

`Loader` does not require `Default`, and its result need not implement either `Load` or `Loadable`. If loading requires caller-owned storage, a query scratch pool, or other resources without meaningful defaults, require an explicit loader instead of offering ordinary loading.

A loader can read small metadata first, resolve runtime settings, and stream large artifacts directly into final storage. It need not keep a second full copy in memory. Resources supplied by the caller are independent of the metadata lifetime `'a`; any metadata references must remain within that lifetime.

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

When loading a field, pass a loader with the settings or resources that field needs. Fields that need no extra inputs can use ordinary `field` calls. Existing `load`, `field`, `load_fields!`, and support for basic types and collections can stay unchanged.

Settings are not passed automatically through `Option<T>` or `Vec<T>`. Callers handle each present value or element with a loader, passing shared resources again as needed. Loaders do not have to support copying or cloning.

### Versions and Errors

`Loader` does not receive automatic version dispatch. A loader reading a versioned object directly must check its shape and `Object::version()`, handle supported older formats, and validate saved data as well as runtime settings. Nested values loaded through `context.load::<T>()` or ordinary `field` calls retain their existing checks and version handling. Caller settings must not change how saved fields are interpreted.

Use the existing load error types and categories. Invalid settings, corrupt data, and reported memory-allocation failures are critical errors, not reasons to try another format. This proposal does not add error handling to allocations that lack it today. Retrying a recoverable error requires a new loader and unchanged or reset external state; loading into an existing value may leave it partly changed on failure.

## Sketch for In-Mem 2.0

Let's start with `Provider<Full<f32>, u32>` whose saved vectors are in `f32` and external IDs in `u32`. The provider holds the ID map, `Full` holds vectors, nbr lists (graph), and start points. Let's separate the saved data from runtime settings and state:

| Saved data | Supplied or rebuilt when loading |
|---|---|
| Live and frozen counts, vector dimension | Capacity for writable slots |
| Distance metric and max nbrs per node | Store concurrency, prefetch settings |
| Live and frozen vectors, nbr lists | Slot state, free-slot lists, reader tracking, and locks |
| External IDs in the same row order as the saved live vectors | Two-way ID map and counters |

### Loading

A defaultable loader holds the runtime inputs and owns the shared load path:

```rust
use diskann_record::load;
use crate::{Provider, num::Capacity, repr::Full, store};

type FullProvider = Provider<Full<f32>, u32>;

#[derive(Default)]
struct FullProviderLoader {
    capacity: Option<Capacity>,
    store: store::Config,
}

impl<'a> load::Loader<'a> for FullProviderLoader {
    type Value = FullProvider;

    fn load(self, context: load::Context<'a>) -> load::Result<FullProvider> {
        let object = context.as_object().ok_or(load::error::Kind::TypeMismatch)?;
        self.load_object(object)
    }
}

impl<'a> load::Loadable<'a> for FullProvider {
    fn load(context: load::Context<'a>) -> load::Result<Self> {
        load::Loader::load(FullProviderLoader::default(), context)
    }
}
```

In this sketch, `load_object` is an internal loader method that dispatches on the saved version, reads and checks metadata and artifact handles, allocates the provider, and streams external IDs, vectors, and graph rows into their final storage. It validates file lengths, row and neighbor counts, edge targets, and external-ID uniqueness before returning a complete provider. It does not rebuild the graph or give frozen points external IDs. This code lives in `diskann-inmem`; no new saved-data trait or type is required in `diskann-record`.

`capacity: None` is a policy, not a capacity chosen before loading. After reading the saved live-node count `N` and frozen-point count `F`, resolve writable capacity `C` to the supplied value or `N`. Check `C >= N`, that `F + C` fits `IdLimit`, and allocation sizes before allocating the provider. Other defaults use `store::Config::default()` and `Full`'s default prefetch settings. Ordinary `Loadable` delegates to this same loader with defaults.

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