# Index Serialization and Backward Compatibility

| | |
|---|---|
| **Authors** | Suhas Jayaram Subramanya |
| **Contributors** | Mark Hildebrand |
| **Created** | 2026-01-15 |
| **Updated** | 2026-03-02 |

## Summary

This RFC standardizes DiskANN index serialization with a *SemVer* compatibility contract. It requires immediate predecessor compatibility (`N-1`) and allows broader support through `load_legacy`. The model is **context-first**: context discovers persisted metadata and file relationships, while `aux` is reserved for runtime-only inputs.

## Motivation

### Background

DiskANN internals evolve over time (new fields, reordered layouts, changed semantics). Without explicit format versioning and compatibility rules, upgrades can force full index rebuilds, which is costly for large datasets.

The library currently employs divergent approaches:

| Crate | Mechanism | Notes |
|-------|-----------|-------|
| `diskann-disk` | `GraphHeader` with `GraphLayoutVersion` | Sector-aligned; validates disk layout. |
| `diskann` | `SaveWith` / `LoadWith` with provider contexts | Persists graph (`save_graph`/`load_graph`), vector data (`.data` via `.bin`), and quantization artifacts (PQ/SQ), but without a unified recursive metadata contract. |

### Problem Statement

There is no unified serialization contract across DiskANN crates. Each crate uses its own ad hoc mechanism for versioning and persistence, making it impossible to reliably detect index format versions, guarantee backward compatibility, or evolve components independently. This leads to costly full index rebuilds on upgrades and fragile loader code.

### Goals

1. Reliably detect index format versions from root + component metadata
2. Load indices saved by the immediate SemVer predecessor (major version) using current code. May also support older versions via explicit legacy converters.
3. Define a minimal, explicit contract for custom legacy conversion logic
4. Keep persisted structure self-discoverable via context (not caller-provided paths)
5. Support optional preflight (`can_load`) and summary loading without changing full-load semantics
6. Preserve a uniform API across primitive and container types (for example, `Vec<T>`)

## Proposal

### 1. Version Primitive

A simple, copyable version struct to be placed in `diskann-utils`:

```rust
/// Represents a semantic version for index file formats.
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

    /// Returns true if `self` is the immediate SemVer predecessor of `other`
    /// within the same major (e.g., 0.2.5 -> 0.3.0 or 1.4.2 -> 1.4.3).
    /// Used to validate required N-1 compatibility.
    pub fn is_prev_of(&self, other: &Version) -> bool {
        if self.major != other.major {
            return false;
        }

        if self.minor == other.minor {
            self.patch + 1 == other.patch
        } else {
            self.minor + 1 == other.minor && other.patch == 0
        }
    }

    /// Returns true if `self` is compatible with `other` by SemVer major.
    /// This allows loading any version <= `other` within the same major.
    pub fn is_compatible_with(&self, other: &Version) -> bool {
        self.major == other.major && self <= other
    }
}
```

#### 1.1 Component Version Contract

Every persistable component/struct must define a static `VERSION` constant.

Versioning rule:
- Increment the component `VERSION` whenever any of the following changes:
  - serialized layout changes,
  - fields are added/removed/renamed/retyped,
  - field semantics or interpretation changes,
  - member usage changes that affect reconstruction behavior.

This rule applies independently to each component, so nested components can evolve at different rates.

### 2. Core `Load` and `Save` Contracts (Context-First, Async)

All persistable index/component types implement async contracts that take a context. Context owns discovery, metadata traversal, and artifact resolution.

This provides a clean separation:
- `VERSION`: the version this implementation writes and natively understands.
- `load()`: dispatches to `load_compatible()` for compatible versions; otherwise to `load_legacy()`.
- `load_compatible()`: loads compatible versions and may recursively delegate legacy conversion for nested components.
- `load_legacy()`: handles on-the-fly conversion from older versions.
- `save()`: writes only current version format (`Self::VERSION`) for the current component.

```text
trait Load<Aux> {
  const VERSION

  async load(context, aux):
    version = context.current_component_version()
    if version is compatible with VERSION:
      return load_compatible(context, aux, version)
    else:
      return load_legacy(context, aux, version)

  async load_compatible(context, aux, version)
  async load_legacy(context, aux, version)
}

trait Save<Aux> {
  const VERSION
  async save(self, context, aux)
}
```

#### 2.1 Context Responsibilities

`DeserializationContext` / `SerializationContext` are the canonical orchestrators:

1. Discover root manifest and component graph (including file ownership).
2. Read/validate per-component metadata and expose per-component version.
3. Resolve file entries to provider-backed readers/writers.
4. Enforce deterministic naming and collision disambiguation for artifacts.
5. Track component path/provenance for deterministic error messages.
6. Optionally expose preflight APIs (`can_load`, summary load path).

The provider abstraction remains intact: contexts can be built on existing `StorageReadProvider` / `StorageWriteProvider` APIs.

#### 2.2 Auxiliary (`aux`) Contract

`aux` is reserved for **runtime-only** inputs that are orthogonal to persisted representation.

Allowed in `aux`:
- allocator handles / arenas
- execution policy and runtime knobs
- preallocated scratch factories
- summary-derived runtime inputs (for example, sizing decisions produced by a preflight summary pass)

Must **not** be required in `aux`:
- persisted file names
- persisted component versions
- persisted dimensions/chunk sizes that can be discovered from metadata/payload

If runtime setup needs persisted facts (for example, allocator sizing), use summary/preflight (Section 3).

#### 2.3 How JSON Ties to `Load` / `Save`

The recursive manifest JSON is the contract consumed by `Load` and produced by `Save`.

At save time:
1. `Save::save` begins component scope in `SerializationContext`.
2. `Save::save` explicitly passes the component's static `VERSION` when entering component scope.
3. Context records that version into the component JSON node.
4. Component writes fields and binary payloads through context APIs.
5. Context records owned artifacts and emits/updates that component's manifest node.
6. Parent component links child component nodes by key.

At load time:
1. `DeserializationContext` resolves current component node from manifest path.
2. Context reads that node's stored version.
3. `Load::load` compares node version with component static `VERSION` and dispatches `load_compatible`/`load_legacy`.
4. Component resolves owned artifacts through context (not caller-provided names).
5. Child loads are executed by descending into child manifest nodes.

Practical rule: component implementations only need to:
- create scoped component/file guards,
- read/write typed fields,
- open artifacts by logical key and write/read incrementally.

Physical file layout is a context concern, not component business logic.

#### 2.4 `SerializationContext` Structure

`SerializationContext` is explicit and minimal. In v1 it targets file-per-artifact output.

Component and file APIs SHOULD be scoped/RAII-style to avoid missed cleanup on early returns.

Pseudocode shape:

```text
struct SerializationContext {
  provider
  root_json
  component_stack
  uuid

  // component scope
  begin_component(name, component_version) -> ComponentScope

  // scalar metadata writes (for primitive values)
  put_scalar<T>(key, value: T)

  // artifact writes
  open_file(key) -> FileWriter
}

struct FileWriter {
  key
  write_chunk(bytes: &[u8])
  finalize() -> size
}
```

Required behavior for `begin_component` / `open_file(key)` / `write_chunk` / `finalize`:

1. `begin_component` enters a child component scope and writes that component's `version` into the scoped JSON node.
2. Component scope finalization (explicit `finalize` or scope `Drop`) performs `exit_component` behavior.
3. `open_file(key)` resolves `key` to filename deterministically as `UUID-lineage-key.data`, where `lineage` is the current component path joined by `-`.
4. File writers support incremental writes via repeated `write_chunk(&[u8])`; callers do not need to materialize full payloads up front.
5. On file finalization (explicit `finalize` or scope drop), context computes final `size`.
6. Context records key-to-file mapping (`key -> name`) and adds `{ key, name, size }` under current component's `files` array in JSON.
7. Within a single component scope, keys are write-once. A second write to a previously written key MUST return a deterministic duplicate-key error.

Concurrency rule: within a component scope, contexts MAY restrict to one active artifact writer. Implementations SHOULD document this.

Key/value write behavior:

- `put_scalar<T>` always writes into the JSON node of the **current component scope**.
- For example, if current scope is `graph`, then `put_scalar("k", v)` writes `JSON["graph"]["k"] = v`.
- Additional example: If inside `graph` and a nested component `comp` is entered, then `put_scalar<T>` writes into `JSON["graph"]["comp"][...]`.

#### 2.5 `DeserializationContext` Structure

`DeserializationContext` is a lightweight runtime object with three responsibilities:

1. **Manifest navigation:** find the current component node and move into child nodes.
2. **Validation:** enforce version checks and root-format detection rules.
3. **Artifact resolution:** open artifact content by key from the current component's `files` list.

`DeserializationContext` should mirror `SerializationContext` operations where possible.

Pseudocode shape:

```text
struct DeserializationContext {
  provider
  root_json
  component_stack
  uuid

  // component scope
  begin_component(name) -> ComponentScope

  // scalar metadata reads (for primitive values)
  get_scalar<T>(key) -> T
  current_component_version() -> Version

  // artifact reads
  open_file(key) -> FileReader
}

struct FileReader {
  key
  read_chunk() -> &[u8]
  read_to_end() -> Vec<u8>
}
```

Version behavior:
- `SerializationContext.begin_component(name, component_version)` writes `JSON[path]["version"] = component_version`.
- `DeserializationContext.current_component_version()` reads `JSON[path]["version"]`.

Required behavior for `begin_component` / `open_file(key)` / file reads:

1. `begin_component` enters a child component scope and moves manifest cursor to that node.
2. Component scope finalization performs `exit_component` behavior.
3. `open_file(key)` resolves `key` from current component's `files` list.
4. Context opens the mapped `name` via provider and returns a reader that supports incremental reads and `read_to_end`.

Lineage behavior is symmetric with save:
- For example, inside `graph`, `get_scalar("k")` resolves from `JSON["graph"]["k"]`.
- Additional example: Inside nested `comp` under `graph`, `get_scalar("x")` resolves from `JSON["graph"]["comp"]["x"]`.

#### 2.6 File-Per-Artifact Storage (v1)

For now, this RFC proposes only one storage mode: **one file per artifact**.

Rules:
- Every persisted binary artifact is emitted as its own file.
- Each file is registered under the current component node's `files` array.
- File names are resolved from logical keys via context-managed naming.
- Collision disambiguation and filename assignment remain context-managed.
- A single `uuid` is used for all files produced by one `SerializationContext` instance.
- Backends MAY still implement internal append/packing behavior; this must remain transparent to `Load`/`Save` implementors.

#### 2.7 Primitive and Generic Type Behavior (`SaveValue` / `LoadValue`)

To avoid ambiguity, this RFC separates **component contracts** from **value contracts**:

- Component/struct types use `Load` / `Save` and participate in component scope + version dispatch.
- Primitive/value-like types use `LoadValue` / `SaveValue` and read/write under an explicit key in the current scope.

Pseudocode shape:

```text
trait SaveValue {
  async save_value(context, key)
}

trait LoadValue {
  async load_value(context, key)
}
```

Rules:
1. Primitive values (`u64`, `u32`, `i32`, `i64`, `f32`, `f16`, `String`) do not create component scopes.
2. Primitive values are encoded as key/value entries within the current component scope through `put_scalar<T>` / `get_scalar<T>`.
3. Primitive values do not carry their own `version` nodes.
4. Persisted component/struct types MUST carry versions as defined in Section 1.1.
5. Generic containers (for example, `Vec<T>`) MUST work uniformly whether `T` is primitive or composite.
6. For `Vec<T>` under container key `container`, the component name for the i-th element MUST be `container.index-i`.
7. If `T` is primitive, each element uses the value contract (`SaveValue` / `LoadValue`) under the element key.
8. If `T` is composite, each element uses the component contract (`Save` / `Load`) with normal version dispatch.
9. Field/key names SHOULD be supplied by the caller or derive macro expansion (for example, from field identifiers), not inferred at runtime from primitive values.

### 3. Dispatch Logic and Optional Summary/Preflight

`Load::load(ctx, aux)` is the canonical entrypoint and performs first-level dispatch:

```text
context = DeserializationContext.open(provider, root)
index = Index.load(context, runtime_aux)
```

Callers can optionally run a summary/preflight pass before full materialization:

```text
preflight_context = DeserializationContext.open(provider, root)
summary = IndexSummary.load(preflight_context, NoAux)
runtime_aux = RuntimeAux.from_summary(summary)

full_context = DeserializationContext.open(provider, root)
index = Index.load(full_context, runtime_aux)
```

Not every component needs a summary type; summaries are optional.

Preflight contract:
- `can_load` is a lightweight check that validates root manifest readability and basic compatibility viability.
- Summary loading is optional and returns enough persisted info for runtime sizing/policy decisions (e.g., for allocators and scratch spaces) before full materialization.
- Full load semantics remain unchanged (`Load::load(ctx, aux)` is still the canonical path).

#### 3.1 Root Format Selection (No One-Time Conversion Required)

Loader behavior is strictly dual-path:

1. If the new RFC root format is present (`UUID-metadata.json` root manifest), load using the normal context-first path.
2. If the new RFC root format is absent, use the explicit legacy path that loads each component using existing legacy readers.

No one-time conversion step is required. Legacy artifacts remain loadable via explicit legacy loaders.

#### 3.2 Root Manifest Validation

When `UUID-metadata.json` is present, context-first loading MUST validate the root manifest before descending into components.

Minimum required root fields:
- `uuid`
- `index_type`
- `version`

Validation behavior:
1. If root manifest is missing, use the legacy root path (Section 3.1).
2. If root manifest is present but unreadable/invalid JSON, return a deterministic root-manifest parse error.
3. If root manifest is present but required fields are missing or malformed, return a deterministic root-manifest validation error.
4. If root manifest is valid, component-level version handling and recursive dispatch proceed normally.

### 4. Version Ownership and Precedence

Versioning is owned per component via metadata fields in the manifest hierarchy.

Precedence rules:

1. Manifest version in each component node is the source of truth for that component.
2. Component-internal metadata versions (for example, protobuf-backed metadata) must be validated according to component rules.
3. Root manifest version and component version must satisfy compatibility rules in this RFC.
4. On mismatch:
   - if safely recoverable by documented converter, route to `load_legacy`.
   - otherwise return a deterministic compatibility error with component path provenance.

This avoids ambiguity when nested components evolve at different rates.

### 5. Artifact Keys, Component Keys, and Disambiguation

Artifact naming is context-managed, not ad hoc.

Required behavior:

1. Components are addressed by a component key (manifest path segment); artifacts are addressed by an artifact key (file entry key). These are distinct concepts.
2. Components request logical artifact keys within their current component scope.
3. In v1, concrete file names follow `UUID-lineage-key.data` and **root** metadata file is `UUID-metadata.json`.
4. Within a single component scope, duplicate artifact keys are disallowed. If a second write is attempted for an existing key, context MUST return a deterministic duplicate-key error.
5. If two distinct keys still map to the same concrete filename after normalization, context MAY apply stable suffixing (for example, `-1`, `-2`) to preserve physical filename uniqueness.
6. Manifest records concrete `name`, `key`, and `size` under owning component path.
7. For collection-like components, lineage SHOULD include deterministic index segments so artifact names remain stable across load/save.

### 6. Format Divergence

This RFC does not unify the **different payload formats** for in-memory and disk indices. It simply adds a metadata-layer so that the same **manifest-driven contract** and context APIs can be used for both formats, but payload content diverges after component dispatch.

### 7. Unified Metadata JSON Schema

We propose a **single recursive manifest JSON** with **per-component versioning** model. This avoids cascading version bumps when only a smaller component (for example, PQ/SQ) evolves.

Each component node carries its own:
- `version`
- `files` (owned artifacts with `key`, `name`, `size`)
- optional nested components
- component-local config/parameters

Any persisted component MUST have a `version`. Other fields are optional unless required by the active index type.

Rule for nested objects in examples and manifests: any nested component SHOULD carry its own `version`; purely flat key/value objects (where values are not nested components) SHOULD omit `version`.

This schema is the persisted IO contract for the context APIs in Section 2.3:
- `Save` writes component state *through context*, and context emits this schema.
- `Load` reads component state *through context*, and context resolves this schema.

#### 7.1 Metadata Generation Flow

`UUID-metadata.json` is generated during `Save` finalization on the root context. The flow is:

1. Each component save writes its payload files and component metadata.
2. Context records component-local file entries `(key, name, size)` under that component node.
3. Save serializes a single metadata object to `UUID-metadata.json`.
4. Metadata write happens last (preferably atomic, e.g., temp + rename) so it represents a complete snapshot.

`load_compatible` uses this JSON manifest for component discovery and cross-file validation.

#### 7.2 Example for In-Memory Index

For in-memory indices, `index_type` is `"vamana"` and `disk_layout` is `null`:

**File:** `UUID-metadata.json`

```json
{
  "uuid": "52b4d2a0-39a0-4c2d-b832-3c3f6d2826c4",
  "index_tag": "my-index-2026-01-15",
  "created_at": "2026-01-15T10:30:00Z",

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
    "prune_kind": "triangle_inequality",
    "max_degree": 64,
    "pruned_degree": 32,
    "alpha": 1.2,
    "l_build": 100,
    "max_occlusion_size": 750,
    "max_backedges": 32,
    "max_minibatch_par": 1,
    "intra_batch_candidates": "all",
    "saturate_after_prune": true,
    "files": [
      {
        "key": "graph_payload",
        "name": "52b4d2a0-39a0-4c2d-b832-3c3f6d2826c4-graph-graph_payload.data",
        "size": 1048576
      }
    ]
  },
  
  "start_points": {
    "version": { "major": 0, "minor": 2, "patch": 0 },
    "strategy": {
      "kind": "medoid"
    },
    "count": 1
  },
  
  "quantization": {
    "version": { "major": 0, "minor": 1, "patch": 3 },
    "kind": "product",
    "product": {
      "version": { "major": 0, "minor": 1, "patch": 3 },
      "num_chunks": 96,
      "ncenters": 256,
      "chunk_offsets": [0, 8, 16, 24, 32, "...truncated...", 768],
      "trainer": {
        "version": { "major": 0, "minor": 1, "patch": 3 },
        "algorithm": "LightPQTrainingParameters",
        "lloyds_reps": 12
      }
    },
    "files": [
      {
        "key": "pq_pivots",
        "name": "52b4d2a0-39a0-4c2d-b832-3c3f6d2826c4-quantization-pq_pivots.data",
        "size": 4096
      },
      {
        "key": "pq_codes",
        "name": "52b4d2a0-39a0-4c2d-b832-3c3f6d2826c4-quantization-pq_codes.data",
        "size": 96000000
      }
    ]
  },

  "vector_data": {
    "version": { "major": 0, "minor": 2, "patch": 0 },
    "format": "bin",
    "files": [
      {
        "key": "vectors_fp",
        "name": "52b4d2a0-39a0-4c2d-b832-3c3f6d2826c4-vector_data-vectors_fp.data",
        "size": 768000000
      }
    ]
  },
  
  "disk_layout": null
}
```

Simplified save call graph for this in-memory shape:

```text
index.save()
|- common.save()        [num_points, dimensions, metric, vector_type]
|- graph.save()         [prune_kind, max_degree, pruned_degree, alpha, l_build, ...]
|  |- graph_payload.save()
|- start_points.save()  [strategy.kind, count]
|- quantization.save()
|  |- kind.save()
|  |- product.save()    [num_chunks, ncenters, chunk_offsets, trainer.{algorithm, lloyds_reps}]
|  |- pq_pivots.save()
|  |- pq_codes.save()
|- vector_data.save()   [format, vectors_fp]
```

Simplified load call graph for the same in-memory shape:

```text
index.load()
|- common.load()        [num_points, dimensions, metric, vector_type]
|- graph.load()         [prune_kind, max_degree, pruned_degree, alpha, l_build, ...]
|  |- graph_payload.load()
|- start_points.load()  [strategy.kind, count]
|- quantization.load()
|  |- kind.load()
|  |- product.load()    [num_chunks, ncenters, chunk_offsets, trainer.{algorithm, lloyds_reps}]
|  |- pq_pivots.load()
|  |- pq_codes.load()
|- vector_data.load()   [format, vectors_fp]
```

Version dispatch occurs at each component boundary (`load` → `load_compatible` or `load_legacy`).

#### 7.3 Example for Disk-Based Index

For disk indices, `index_type` is `"diskann"` and `disk_layout` is populated:

**File:** `UUID-metadata.json`

```json
{
  "uuid": "d3f90f3c-6e8d-4baf-875b-bf4c4b2c3f9f",
  "index_tag": "billion-scale-index",
  "created_at": "2026-01-15T10:30:00Z",
  
  "index_type": "diskann",

  "version": { "major": 0, "minor": 2, "patch": 0 },
  
  "common": {
    "version": { "major": 0, "minor": 2, "patch": 0 },
    "num_points": 1000000000,
    "dimensions": 768,
    "metric": "cosine",
    "vector_type": "f16"
  },
  
  "graph": {
    "version": { "major": 0, "minor": 2, "patch": 0 },
    "prune_kind": "triangle_inequality",
    "max_degree": 64,
    "pruned_degree": 32,
    "alpha": 1.2,
    "l_build": 100,
    "max_occlusion_size": 750,
    "max_backedges": 32,
    "max_minibatch_par": 1,
    "intra_batch_candidates": "all",
    "saturate_after_prune": true,
    "files": [
      {
        "key": "disk_graph_payload",
        "name": "d3f90f3c-6e8d-4baf-875b-bf4c4b2c3f9f-graph-disk_graph_payload.data",
        "size": 1008000000000
      }
    ]
  },
  
  "start_points": {
    "version": { "major": 0, "minor": 2, "patch": 0 },
    "strategy": {
      "kind": "medoid"
    },
    "count": 1
  },
  
  "quantization": {
    "version": { "major": 1, "minor": 0, "patch": 0 },
    "kind": "product",
    "product": {
      "version": { "major": 1, "minor": 0, "patch": 0 },
      "num_chunks": 96,
      "ncenters": 256,
      "trainer": {
        "version": { "major": 1, "minor": 0, "patch": 0 },
        "algorithm": "LightPQTrainingParameters",
        "lloyds_reps": 12
      }
    },
    "files": [
      {
        "key": "pq_pivots",
        "name": "d3f90f3c-6e8d-4baf-875b-bf4c4b2c3f9f-quantization-pq_pivots.data",
        "size": 4096
      },
      {
        "key": "pq_codes",
        "name": "d3f90f3c-6e8d-4baf-875b-bf4c4b2c3f9f-quantization-pq_codes.data",
        "size": 96000000
      }
    ]
  },
  
  "disk_layout": {
    "version": { "major": 1, "minor": 0, "patch": 0 },
    "graph_header": {
      "version": { "major": 1, "minor": 0, "patch": 0 },
      "layout_version": {
        "major": 1,
        "minor": 0
      },
      "block_size": 4096
    },
    "graph_metadata": {
      "version": { "major": 1, "minor": 0, "patch": 0 },
      "num_pts": 1000000000,
      "dims": 768,
      "medoid": 500000,
      "node_len": 1008,
      "num_nodes_per_block": 4,
      "vamana_frozen_num": 1,
      "vamana_frozen_loc": 1000000000,
      "disk_index_file_size": 1008000000000,
      "associated_data_length": 0
    }
  }
}
```

### 8. Backward Compatibility Policy

This RFC proposes **minimum N-1 compatibility** within the same major:

| Code Version | Required Support | Optional Support |
|--------------|------------------|------------------|
| `0.2.x`      | `0.2.x`, `0.1.x` | none             |
| `0.3.x`      | `0.3.x`, `0.2.x` | `0.1.x` via explicit converters |
| `1.4.x`      | `1.4.x`, `1.3.x` | older `1.x` via explicit converters |

**Rules:**
1. `load(ctx, auxiliary)` dispatches to `load_compatible` when `version <= VERSION` within the same major; otherwise it dispatches to `load_legacy`.
2. `load_compatible()` may recursively invoke component loaders, including component-level `load_legacy`, when nested component versions are older.
3. `load_legacy()` must support the immediate predecessor version (`N-1`). It may also support older versions via explicit conversion logic.
4. Final accept/reject for newer major or incompatible payloads is determined by `load_legacy()`; it may convert or throw `unsupported version` error.

#### 8.1 Legacy Runtime Fallback Strategy

Root detection determines the active loader path:

1. **New format present at root:** use context-first manifest path.
2. **New format missing at root:** invoke explicit legacy component loading path.

Writers always emit the new format. Readers support both paths via root detection, with no required one-time conversion.

## Trade-offs

### Context-First vs. Caller-Provided Paths

**Chosen: context-first discovery.** The context object discovers persisted metadata and resolves file relationships from the manifest, rather than requiring callers to supply file paths.

- *Pro:* Self-discoverable; reduces coupling between caller and persisted layout.
- *Pro:* Enables manifest-driven validation and deterministic error messages with component provenance.
- *Con:* Adds complexity to the context implementation.
- *Alternative:* Caller-provided paths (current `SaveWith`/`LoadWith` approach) are simpler but fragile when file layouts change (and inconsistent/ambiguous when multiple files are involved across different components).

### Recursive Per-Component Versioning vs. Single Root Version

**Chosen: per-component versioning.** Each component in the manifest carries its own `version`, allowing independent evolution.

- *Pro:* Avoids cascading version bumps when only one component (e.g., PQ) changes.
- *Pro:* Enables fine-grained `load_legacy` dispatch at the component level.
- *Con:* More metadata to maintain; version consistency across components must be validated.
- *Alternative:* Single root version — simpler but forces all components to bump in lockstep.

### File-Per-Artifact vs. Packed/Concatenated Archive

**Chosen: file-per-artifact (v1).** Each binary artifact is its own file.

- *Pro:* Simple to implement, debug, and inspect.
- *Pro:* Works well with existing provider abstractions.
- *Con:* Many small files on disk for complex indices.
- *Alternative:* Packed archive format — fewer files but adds complexity; deferred to future work.

### Dual-Path Legacy Loading vs. One-Time Migration

**Chosen: dual-path loading with no required migration.** Readers detect the root format and dispatch to either the new context-first path or the legacy path.

- *Pro:* No forced migration step; legacy indices remain loadable indefinitely.
- *Pro:* Writers always emit the new format, so the ecosystem naturally converges.
- *Con:* Must maintain two loader paths until legacy support is dropped.
- *Alternative:* One-time migration tool — simpler reader code but imposes a mandatory upgrade step on users.

## Benchmark Results

N/A — this RFC does not change binary payload formats and therefore does not require benchmark comparison.

## Future Work

- [ ] Derive macro for automatic `Load`/`Save` implementation from struct field annotations
- [ ] Formal JSON schema (e.g., JSON Schema draft) for manifest validation tooling
- [ ] Cross-major-version migration tooling
- [ ] Manifest digital signatures or checksums for integrity verification
- [ ] Packed/concatenated archive storage mode (beyond file-per-artifact)

## References

1. [Current `SaveWith`/`LoadWith` traits](../diskann-providers/src/storage/api.rs)
2. [Current file naming conventions](../diskann-providers/src/storage/path_utility.rs)
3. [Graph serialization (`save_graph`/`load_graph`)](../diskann-providers/src/storage/bin.rs)
4. [PQ storage (`PQStorage::write_pivot_data`)](../diskann-providers/src/storage/pq_storage.rs)
5. [Disk index `GraphHeader`/`GraphMetadata`](../diskann-disk/src/data_model/)
6. [Semantic Versioning 2.0.0](https://semver.org/)
