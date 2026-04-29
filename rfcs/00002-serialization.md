# Index Serialization and Backward Compatibility

| | |
|---|---|
| **Authors** | Suhas Jayaram Subramanya |
| **Contributors** | Mark Hildebrand |
| **Created** | 2026-01-15 |
| **Updated** | 2026-04-23 |

## Summary

This RFC standardizes DiskANN index serialization with a SemVer compatibility contract and a context-first manifest model. Each persistable component carries its own version. A recursive JSON manifest describes the full component tree and its artifacts. Loaders detect the manifest at the root and dispatch per-component; legacy indices without the manifest remain loadable via an explicit fallback path.

## Motivation

### Background

DiskANN is used in many different ways and below are three common patterns:
- **In-process library**: Here, the caller owns all data buffers and supplies providers to DiskANN to access vector and graph data. This pattern does not rely on file-based serialization and the caller is responsible for managing the lifecycle of all data. 
- **File-based builder**: Here, the caller invokes `DiskANNIndexBuilder::build()` with input files (e.g., `vectors.bin`) and DiskANN produces output files (e.g., `diskann.index`, `diskann_pq_pivots.bin`, `diskann_pq_compressed.bin`). The caller is responsible for managing these files and supplying storage providers to access them at query time. This pattern relies on the stability of DiskANN's file-based serialization.
- **Dynamic index with checkpointing support**: Here, the caller builds an in-memory index that is mutable (supports insert/deletes) and periodically checkpoints it to durable storage. The caller should be able to load a checkpoint without rebuilding the entire index from the original vectors. This pattern relies on a stable serialization format for in-memory indexes.

### Problem Statement

Each crate uses its own ad hoc versioning and persistence mechanism. This makes it impossible to reliably detect binary format versions, guarantee backward compatibility, or evolve components independently while maintaining compatibility with indexes built using prior library versions. Internal layout changes risk breaking existing workflows, and currently the only safe recovery is a full index rebuild starting with the full-precision vectors. This is infeasible for large indices because it requires rebuilding the entire graph and retraining PQ from scratch even for minor changes.

### Target Scenarios

This RFC is intended to solve a bounded set of persistence scenarios rather than provide a generic framework for every storage pattern DiskANN might support in the future. Here are a few concrete scenarios we want to enable:

1. **Durable immutable indices**. A service may build a DiskANN index today, store it in durable storage (like an object store) or as an index attached to Iceberg tables and may use a newer DiskANN library version to load it at a later point in time. 
2. **Artifact discovery for file-based builders**. A caller that uses `DiskANNIndexBuilder::build()` should be able to discover the complete set of files that belong to the produced index without hard-coding quantizer-specific or index-specific file naming conventions. Should the caller choose a different quantizer or index type, they need not be familiar with that component's file contract to discover and manage its artifacts. This helps surface the full set of artifacts to users and also enables new index types and quantizers to be added without requiring developers to update caller-side file management logic. Additionally, the manifest can record component-local scalar parameters (e.g., num. PQ chunks, bits-per-chunk) that can be used to setup runtime structures (e.g., buffers and scratch spaces) without needing to parse binary artifacts.
3. **Checkpoint and restore for in-memory indices**. A caller should be able to pause insert/deletes, save an in-memory index, shut down the process and/or move the checkpoint to another machine, and load the index back without rebuilding the graph or retraining quantization artifacts from the original vectors.
4. **Ease of adoption by new library clients**. A new DiskANN consumer, including public table-format integrations such as Iceberg, should be able to treat a persisted index as a self-describing unit instead of re-deriving DiskANN-specific file contracts or inventing a separate manifest format to track DiskANN artifacts.

These scenarios are specifically about cases where DiskANN owns, emits, or describes persisted artifacts. They do not change the existing storage-agnostic provider model where the client owns the physical layout and its schema evolution strategy.

### Goals

1. Detect index format versions reliably from a root manifest and per-component metadata.
2. Load indices saved by the immediate SemVer predecessor (N-1) within the same major version.
3. Keep persisted structure self-discoverable via context (callers should not need to supply file paths).
4. Define a clear transition path for users to adopt the new serialization without requiring a one-time conversion step.

## Proposal

### Version Primitive

A simple, copyable version struct in `diskann-utils`:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Version {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

impl Version {
    pub const fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self { major, minor, patch }
    }

    /// True if `self` can be loaded by code at version `other`
    /// (same major, self <= other).
    pub fn is_compatible_with(&self, other: &Version) -> bool {
        self.major == other.major && self <= other
    }
}
```

Every persistable component defines a `const VERSION: Version`. Bump it when serialized layout, fields, or field semantics change. Components version independently allowing nested components to evolve at different rates.

### Serialization Trait Architecture

Serialization uses a three-layer design:

1. **Public API**: a `save()` free function and a `save_fields!` macro. All recursive serialization goes through these. They are the only entry points for writing child values.
2. **Dispatch trait (`Persist`)**: a universal trait that all serializable types implement. The free function delegates here. Primitives (`u32`, `f32`, `String`, etc.), containers (`Vec<T>`, `Option<T>`), and versioned components all implement `Persist`, so they serialize uniformly without requiring different traits for primitives and components.
3. **Component traits (`Save` / `Load`)**: what most user-defined structs implement. Carries `VERSION`, `load_compatible`, `load_legacy`. A blanket `impl<T: Save> Persist for T` bridges components into the dispatch layer.

Primitives implement `Persist` directly. Instead, they write a scalar into the current scope and carry no version. Containers like `Vec<T>` are generic over `T: Persist`, so `Vec<u32>` and `Vec<Graph>` both work without special-casing.

#### Component traits

```rust
trait Save {
    const VERSION: Version;
    fn save(&self, component: Component<'_>) -> Written;
}

trait Load: Sized {
    const VERSION: Version;
    fn load_compatible(component: Component<'_>, stored: Version) -> Self;
    fn load_legacy(component: Component<'_>, stored: Version) -> Self;
}
```

- `save()` always writes `Self::VERSION`.
- On load, the dispatch layer compares the stored version against `Self::VERSION` and routes to `load_compatible` or `load_legacy` accordingly.
- `load_legacy()` must handle N-1 and may handle older versions.

#### Written proof tokens

Each `save()` call returns a `Written` token that carries that node's manifest data (version, scalars, file entries). A parent produces its own `Written` via `Written::new(&[("field_name", child_token), ...])`, which composes the children's manifest nodes into the parent's JSON object. This serves two purposes: it proves all children were serialized (enforced by the type system, not runtime checks), and it builds the `UUID-metadata.json` manifest recursively as the `save()` call tree unwinds. No separate manifest-generation pass is needed!

#### Example usage

```rust
impl Save for Quantization {
    const VERSION: Version = Version::new(0, 1, 3);

    fn save(&self, mut c: Component<'_>) -> Written {
        // save_fields! stamps each field name into the manifest key,
        // the struct accessor, and the context field name.
        save_fields!(c, self, [num_chunks, ncenters])
    }
}

impl Save for Graph {
    const VERSION: Version = Version::new(0, 2, 0);

    fn save(&self, mut c: Component<'_>) -> Written {
        // Binary artifact: direct context API
        let mut f = c.open_file("graph_payload");
        // write graph data into f using whatever format it wants
        f.write_chunk(&self.payload);
        // finalize allows us to record the artifact's name 
        // and size in the manifest
        f.finalize();

        // Serialize other scalar and nested fields via the macro
        save_fields!(c, self, [max_degree, alpha, quantization])
    }
}
```

`Graph` and `Quantization` get `Persist` for free via the blanket impl. `Vec<Graph>` works because each element goes through `save()` → dispatch → `Graph::save()`. `Vec<u32>` works because `u32` implements `Persist` directly.

### Context Objects

The context is represented by a `Ctx` value that is consumed (moved) on use. When the dispatch layer encounters a `Save` component, it converts `Ctx` into a `Component` scope (recording the component's version in the manifest). For primitives, `Ctx` converts into a `Primitive` scope. For containers, `Ctx` converts into a `Seq` scope. These conversions are mutually exclusive: a `Ctx` can only become one of the three.

A `Component` scope provides:
- `field(key)`: returns a child `Ctx` for a named field (used for recursion).
- `open_file(key)`: returns a `FileWriter` for binary artifacts.
- `stored_version()`: returns the version recorded in the manifest for this component (used on the load path).

Physical file layout (naming, collision disambiguation) is entirely a context concern. Components only interact with logical keys. Each `field(key)` call appends `key` to the context's internal lineage path, so when `open_file("graph_payload")` is called inside the `"graph"` component scope, the context derives the filename `UUID-graph-graph_payload.data` automatically from the accumulated lineage.

The `aux` parameter on `Load` is reserved for runtime-only inputs (allocators, execution policy, scratch factories). Anything discoverable from the manifest must *not* go in `aux`.

### Manifest Schema

A single recursive JSON manifest (`UUID-metadata.json`) describes the full component tree. Each component node carries:

- `version`: the component's format version
- `files`: owned artifacts (`key`, `name`, `size`)
- nested child components
- component-local scalar parameters

Example (in-memory index, cut short for brevity):

```json
{
  "uuid": "52b4d2a0-...",
  "index_type": "vamana",
  "version": { "major": 0, "minor": 2, "patch": 0 },
  "common": {
    "version": { "major": 0, "minor": 2, "patch": 0 },
    "num_points": 1000000,
    "dimensions": 768,
    "metric": "l2",
    "vector_type": "f32"
  },
  "graph": {
    "version": { "major": 0, "minor": 2, "patch": 0 },
    "max_degree": 64,
    "files": [
      { "key": "graph_payload", "name": "52b4d2a0-graph-graph_payload.data", "size": 1048576 }
    ]
  },
  "quantization": {
    "version": { "major": 0, "minor": 1, "patch": 3 },
    "kind": "product",
    "product": {
      "version": { "major": 0, "minor": 1, "patch": 3 },
      "num_chunks": 96,
      "ncenters": 256
    },
    "files": [
      { "key": "pq_pivots", "name": "52b4d2a0-quantization-pq_pivots.data", "size": 4096 },
      { "key": "pq_codes", "name": "52b4d2a0-quantization-pq_codes.data", "size": 96000000 }
    ]
  },
  "vector_data": {
    "version": { "major": 0, "minor": 2, "patch": 0 },
    "files": [
      { "key": "vectors_fp", "name": "52b4d2a0-vector_data-vectors_fp.data", "size": 768000000 }
    ]
  },
  "disk_layout": null
}
```

For disk indices, `index_type` is `"diskann"` and `disk_layout` is populated with graph header, block size, and layout version metadata.

The manifest is written last during save (ideally via atomic temp + rename) so it always represents a complete snapshot. Within a component scope, artifact keys are write-once by design (macro enforces this) so the manifest can be treated as a source of truth for what was written.

### Backward Compatibility Policy
Each component carries its own version and compatibility contract. The library must guarantee that any component at version N can be loaded by code that supports N and N-1. Support for older versions (N-2, N-3, etc.) is optional but encouraged, especially for stable components like quantization that are less likely to require breaking changes.

| Current Component Version | Support for Component Versions | Optional |
|--------------|------------------|----------|
| `0.5.x` |`0.5.x`, ... `0.2.x`, `0.1.x` | — |
| `1.2.x` | `1.1.x`, `1.0.x` | `0.y.x` via `load_compatible` |
| `2.2.x` | `2.1.x`, `2.0.x` | `1.x` via `load_compatible`, support for `0.y.x` not guaranteed |

Version dispatch is per-component. A root index at version `2.2.0` may contain a quantization component still at `1.0.0`and that component's `load()` dispatches to its own `load_compatible` or `load_legacy` independently.

Release notes **must** document breaking changes, especially if it impacts the manifest schema or a component's serialized representation. A breaking change to a component requires a major version bump for *that* component, but not necessarily for the entire library. This allows for more granular evolution and clearer communication of compatibility guarantees.

Any incompatible loads must panic with a clear error message indicating the expected and found versions, and which component is affected. This helps users identify when they are trying to load an index with an incompatible library version. 

### Legacy Fallback

Root detection selects the loader path:

1. **Manifest present** (`UUID-metadata.json`): use the context-first path described above.
2. **Manifest absent**: invoke the legacy loading path using existing per-crate readers.

Writers always emit the new manifest format. No one-time conversion is required — legacy artifacts are loadable as long as the legacy path is maintained.

### Incremental Adoption

Because dispatch is per-component, migration can happen one component at a time. A component's `save()` can write its binary artifact in the existing format — it just registers the file with `open_file` and returns a `Written` token. Its `load_compatible()` can internally delegate to the old reader code to parse that artifact. Meanwhile, a sibling component (e.g., quantization) can fully adopt a new format. The manifest records what each component produced without dictating the byte layout inside any artifact. This means a user can migrate PQ first, then graph, then vector data, each at its own pace, without touching the others.

### Expected Impact
- For users that do not persist DiskANN artifacts, there should be no impact. They can continue to use providers and manage their own storage as before. The new serialization and manifest are additive features that do not change the existing provider-based data access pattern.
- For users that do persist DiskANN artifacts, there will be an initial impact to adopt the new manifest-based serialization. However, the design includes a legacy fallback path, so existing indices should remain loadable without modification. New indices will include the manifest, and loaders will handle both old and new formats transparently during a rolling upgrade.


## Future Work

- [ ] Derive `save_fields!` macro for automatic `Load`/`Save` from struct annotations
- [ ] Formal JSON schema for manifest validation tooling
- [ ] Cross-major-version migration tooling
- [ ] Serialize in-memory indexes while under insert/delete operations (?)
- [ ] Packed archive storage mode (?)