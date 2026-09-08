# Runtime Parameters for Load

| | |
|---|---|
| **Authors** | Suhas Jayaram Subramanya (suhasjs) |
| **Contributors** | |
| **Created** | 2026-09-08 |
| **Updated** | 2026-09-08 |

## Summary

Extend `diskann-record` so each target type declares one runtime-parameter contract for reconstruction. Loading remains type-directed: callers choose the target Rust type, while persisted schema information governs decoding, including legacy upgrades and probing. Runtime parameters only control process-local construction choices such as capacity, concurrency limits, and scratch allocation.

## Motivation

A saved in-memory index contains only its live nodes. Loading 100,000 saved nodes may construct an index with capacity for 200,000, one million, or ten million nodes. Capacity is not persisted because it describes the new runtime allocation rather than the logical index.

The existing `Load` API receives only persisted data. Extend it rather than introducing a parallel `LoadWith<T>` interface that permits multiple runtime contracts for the same target type. A generic loading trait does not inherently prevent self-describing persistence; the goal here is one explicit runtime contract within the existing loading interface.

## Proposal

Every loadable type declares one runtime-parameter type. The following examples sketch the proposed API, not the current implementation.

```rust
trait Load<'a>: Sized {
    type Runtime: ?Sized;
    const VERSION: Version;

    fn load(object: Object<'a>, runtime: &Self::Runtime) -> Result<Self>;
    fn load_legacy(object: Object<'a>, runtime: &Self::Runtime) -> Result<Self>;
}

trait Loadable<'a>: Sized {
    type Runtime: ?Sized;

    fn load(context: Context<'a>, runtime: &Self::Runtime) -> Result<Self>;
}
```

The blanket `Loadable` implementation for `T: Load` sets `Runtime = <T as Load<'a>>::Runtime`. It retains the existing object-shape check and version dispatch, forwarding the same runtime argument to either `load` or `load_legacy`. Existing legacy upgrades and recoverable probing remain supported. Implementations must use persisted schema information, not runtime parameters, to interpret field encodings and choose decoding paths. This is an implementation contract, not a restriction enforced by the associated type.

### One Calling Convention

Use the same explicit runtime argument throughout. `()` means no runtime input; it is not an implicit default. Rather than adding `field_runtime`, change `Object::field` and `Context::load`:

```rust
impl<'a> Context<'a> {
    fn load<T: Loadable<'a>>(&self, runtime: &T::Runtime) -> Result<T> {
        <T as Loadable<'a>>::load(self.clone(), runtime)
    }
}

impl<'a> Object<'a> {
    fn field<T: Loadable<'a>>(&self, key: &str, runtime: &T::Runtime) -> Result<T> {
        self.child(key)?.load::<T>(runtime)
    }
}
```

These helpers go through `Loadable`, preserving version dispatch. Runtime input is not stored in `Context`, `Object`, or the backend. A parent explicitly supplies each child's parameters; the return type or a type annotation identifies the child type, not the runtime argument alone.

**NOTE**: The runtime argument is borrowed for the call, independently of the manifest lifetime. Loaders may copy configuration or clone shared resources such as `Arc`s; retaining borrowed resources requires their lifetimes to be represented explicitly in the result and runtime types.

The bootstrap implementations must also change:

| Implementation | Runtime and forwarding |
|---|---|
| Numerics, nonzero numerics, `bool`, `String`, `&str`, `Handle` | Declare `Runtime = ()`; retain existing decoding and validation. |
| `Option<T>` | Declare `Runtime = T::Runtime`; forward it to a present value. |
| `Vec<T>` | Declare `Runtime = T::Runtime`; pass the same argument to every element. |

Both numeric macros generate the unit-runtime signature. Internal primitive loads pass `&()`. Collections requiring different parameters per element use explicit array iteration.

**`load_fields!` extension**: Extend `load_fields!` with an optional `=> runtime_expression` after each field's optional type annotation. The expression supplies the runtime reference directly; omitting it supplies `&()`, requiring that field's `Runtime` to be `()`. This is proposed macro syntax, not functionality in the current implementation.

For example, `graph: Graph => &runtime.graph` expands to `let graph: Graph = object.field("graph", &runtime.graph)?;`, while `metadata: Metadata` expands to `let metadata: Metadata = object.field("metadata", &())?;`. Type annotations remain optional when the surrounding code determines the target type. The macro only abbreviates field calls; it does not infer or inherit runtime parameters.

## Example with a nested runtime type

This small example uses inline adjacency lists for clarity; a real index streams them through sidecar artifacts.

```rust
struct Metadata {
    name: String, // Saved.
}

struct Graph {
    adjacency: Vec<Vec<u32>>, // Saved, one row per live node.
    max_slots: usize,         // Runtime allocation limit; not saved.
}

struct Index {
    metadata: Metadata,
    graph: Graph,
    scratch: Vec<u32>, // Runtime scratch; not saved.
}

struct GraphRuntime {
    max_slots: usize,
}

struct IndexRuntime {
    graph: GraphRuntime,
    scratch_capacity: usize,
}
```

The nested types own their persistence implementations:

| Type | `Save` fields | `Load::Runtime` | Construction |
|---|---|---|---|
| `Metadata` | `name` | `()` | Decode the name. |
| `Graph` | `adjacency` | `GraphRuntime` | Validate edges and `max_slots >= adjacency.len()`, then allocate the requested slot capacity. |

Both use schema version `0.0` and reject unsupported versions. Invalid capacity is a critical load error, not a reason to try another schema. Neither type serializes its runtime parameters.

`Index` implements `Save` and `Load`, but not `Loadable` directly:

```rust
impl save::Save for Index {
    const VERSION: Version = Version::new(0, 0);

    fn save(&self, context: save::Context<'_>) -> save::Result<save::Record<'_>> {
        Ok(save_fields!(self, context, [metadata, graph]))
    }
}

impl load::Load<'_> for Index {
    type Runtime = IndexRuntime;
    const VERSION: Version = Version::new(0, 0);

    fn load(object: load::Object<'_>, runtime: &IndexRuntime) -> load::Result<Self> {
        load_fields!(object, [
            metadata: Metadata,
            graph: Graph => &runtime.graph,
        ]);
        let mut scratch = Vec::new();
        scratch.try_reserve_exact(runtime.scratch_capacity).map_err(load::Error::new)?;
        Ok(Self { metadata, graph, scratch })
    }

    fn load_legacy(_: load::Object<'_>, _: &IndexRuntime) -> load::Result<Self> {
        Err(load::error::Kind::UnknownVersion.into())
    }
}
```

`Metadata` receives no runtime input. `Graph` receives only its allocation parameters, not the parent's scratch configuration. Saving the result again writes only metadata and live adjacency rows, regardless of the chosen capacity.

### Backend Entry Points

Disk and memory loads pass the runtime argument to the same dispatcher. Backend implementations remain responsible only for the manifest and artifacts; no persisted format changes are needed.

The context-based loader retains `T: Loadable<'a>` for a caller-owned context borrowed for `'a`, so results may continue to borrow from that context. The disk entry point instead creates a temporary context and requires a result that cannot borrow from it:

```rust
fn load_from_disk<T, R: ?Sized>(
    metadata: &Path,
    artifacts: &Path,
    runtime: &R,
) -> Result<T>
where
    T: for<'a> Loadable<'a, Runtime = R>;
```
and it's called like this:
```rust
let runtime = IndexRuntime {
    graph: GraphRuntime { max_slots: 200_000 },
    scratch_capacity: 64,
};
let index: Index = load_from_disk(&metadata_path, &artifact_dir, &runtime)?;
```

The equality bound fixes the associated runtime type across manifest lifetimes; `R` does not select an alternative implementation of `Load`. A runtime-independent root uses the same entry point with `&()`.

Memory context types are public, but generic save/load execution is currently crate-private. This proposal updates runtime forwarding through that path; exposing public memory execution APIs is outside its scope.