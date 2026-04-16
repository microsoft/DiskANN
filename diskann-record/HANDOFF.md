# diskann-record: Handoff Document

## Status

**Working proof-of-concept.** The crate implements a full save→JSON→load round-trip for
nested structs with binary file artifacts. The end-to-end test in `src/lib.rs` passes.
Many parts are intentionally rough — this is a design exploration, not production code.

See `rfc.md` at the repo root for the motivating RFC.

## Architecture Overview

### Two-Trait Pattern (Save and Load)

Each side has two traits:

- **`Save`** / **`Load`** — versioned component traits. Users implement these for their
  structs. Each carries a `const VERSION: Version`.
- **`Saveable`** / **`Loadable`** — universal dispatch traits used in generic bounds.
  Primitives implement these directly. `Save`/`Load` types get them via blanket impls that
  handle version tagging/dispatch automatically.

This split exists because primitives (numbers, strings) shouldn't carry versions, but
versioned structs and primitives must be storable through the same `Value` tree.

### Value Model

`Value<'a>` is a bespoke enum decoupled from `serde_json`:

```rust
pub enum Value<'a> {
    Bool(bool),
    Number(Number),       // Copy enum: U64, I64, F64
    String(Cow<'a, str>),
    Bytes(Cow<'a, [u8]>),
    Array(Vec<Value<'a>>),
    Object(Versioned<'a>),
    Handle(Handle),
}
```

The lifetime `'a` allows the save path to borrow from the data being saved (zero-copy
strings, etc.). The load path deserializes to `Value<'static>` with owned `Cow`s since
the JSON manifest is kilobytes and zero-copy deserialization isn't worth the ergonomic cost.

Custom `Serialize`/`Deserialize` impls ensure `Number` serializes as a plain JSON number,
`Handle` as `{"$handle": "name"}`, and `Versioned` as a flat object with `$version`
alongside user fields.

### Context Objects

**Save side:** `Context<'a>` wraps `&'a ContextInner` (shared reference, `Clone`). This
enables parallel saves via `rayon::join`. `ContextInner` owns the output directory path and
a `Mutex<HashSet<String>>` for file name deduplication. `Context::write(name)` returns a
`Writer` (wrapping `BufWriter<File>`), and `Writer::finish() -> Handle` provides a proof
token that the file was fully written.

**Load side:** `Context<'a>` wraps `&'a ContextInner` + `&'a Value<'a>`. `Object<'a>`
provides typed field access via `field::<T>(key)`. `Array<'a>` provides iteration.
Everything is by-reference into the deserialized `Value` tree ("parse once, probe many").

### Error Handling

**Save errors:** `save::Error` wraps `anyhow::Error`. Simple — something went wrong.

**Load errors:** `load::Error` has a light/heavy split:
- `Light(Kind)` — cheap, `Copy` enum (`TypeMismatch`, `MissingField`, etc.) for speculative
  probing (try one version, fall back to another without allocating error context).
- `Heavy(anyhow::Error)` — rich diagnostics with context chaining for actual failures.

### Macros

- `save_fields!(self, context, [x, y, inner])` — saves named fields, wraps errors with
  field name context.
- `load_fields!(object, [x, y, inner, vector: Handle])` — loads named fields with optional
  type annotations when inference fails.

### Reserved Keys

Keys starting with `$` are reserved for infrastructure (`$version`, `$handle`). Since Rust
identifiers can't start with `$`, the `save_fields!` macro is inherently safe — no runtime
check needed in the macro path.

## What Works

- Full round-trip: `save_to_disk(&t, dir, metadata)` → JSON + binary files →
  `load_from_disk(metadata, dir) -> T`
- Nested structs with recursive save/load
- Binary file artifacts (write bytes, get Handle, store in manifest, read back on load)
- Version tagging in manifest (`$version` alongside fields)
- Custom serde for `Value`, `Number`, `Handle` (plain JSON output)
- Light/heavy error split on load path

## Remaining Work

### Error Handling Cleanup

Several places still use `unwrap()` or `panic!()` instead of proper error propagation:

- `save::context.rs` — `File::create_new` and file dedup panic on failure
- `save::context.rs` — `ContextInner::finish()` has unwraps on file creation and
  serialization
- `load::context.rs` — `ContextInner::read()` panics on missing file key
  (`panic!("this should return an error instead")`)
- `load::mod.rs` — `load_number!` macro uses `.unwrap()` on `TryFrom` — should return a
  precision/overflow error
- `save::value.rs` — `Record::insert` panics on reserved keys instead of returning an error

### Writer Flush

`Writer::finish()` consumes the `Writer` but doesn't explicitly flush the inner
`BufWriter`. `BufWriter::drop` attempts to flush but **silently swallows I/O errors**.
`finish()` should call `self.io.into_inner()` or `self.io.flush()` and propagate the error.

### Value::Bytes Wiring

`Value::Bytes` exists in the enum and serializes, but there's no end-to-end pattern for
using it (no `Saveable`/`Loadable` impls for `&[u8]` / `Vec<u8>` as inline bytes). Should
follow the `Handle` pattern — schema-aware (the Rust type tells the loader it's bytes, not
a JSON array of integers).

### Enum Support

No framework-level approach for enums yet. This is a significant gap for real-world usage
(e.g., `DistanceMetric`, `QuantizationType`). Needs a convention for how enum variants map
to the `Value` tree — likely a discriminant field pattern.

### SemVer Version Dispatch

`load_legacy()` currently receives everything that isn't an exact version match. There's no
actual SemVer compatibility logic (e.g., "minor bumps are backward compatible within the
same major version"). The blanket `Loadable` impl should route based on SemVer rules, not
just equality.

### File Name Disambiguation

`save::ContextInner` currently uses raw file names as-is. The RFC envisions UUID-based
naming to prevent collisions (e.g., `{uuid}-{user_name}.bin`). The `uuid` crate is not yet
in the dependencies.

### Missing Primitive Impls

- `bool` — `Value::Bool` exists but no `Saveable`/`Loadable` impls
- `Option<T>` — no support yet. Needs a convention (omitted field? explicit null variant?)

### Test Infrastructure

The current test in `lib.rs`:
- Writes to `"."` (cwd), leaving `metadata.json` and `auxiliary.bin` behind
- Doesn't assert round-trip equality (`t == we_are_back`)
- Should use `tempfile::tempdir()` for isolation

### Platform-Dependent Types

`usize` and `isize` are platform-dependent (32-bit vs 64-bit). Saving them to a manifest
that may be loaded on a different platform could silently truncate. Needs a policy — either
forbid them in manifests or define a canonical wire width.

### ContextInner Generalization

Currently the save and load `ContextInner` types are concrete (directory-backed). Future
work includes:
- Trait object behind `Context` for backend swappability
- Packed single-file backend (sequential artifact writes, offset table)
- VFS support (`vfs` crate already in workspace)
- `write_sized(name, size)` API for pre-allocated regions in packed format

The current design has clean seams for this — `Writer::finish() -> Handle` and
`ContextInner::finish(Value)` are the extension points.

### Manifest Improvements

- Record file sizes in the manifest for integrity checking / pre-allocation on load
- Consider preserving field ordering (`HashMap` loses insertion order —
  `IndexMap` or `Vec<(K, V)>` would preserve it for human-readable JSON)

### Value Deserialize Cleanup

The `Deserialize` impl for `Value<'a>` still carries a `PhantomData<&'a ()>` even though
it always produces `Cow::Owned`. Could be simplified to impl directly for `Value<'static>`.
`visit_borrowed_str`/`visit_borrowed_bytes` always clone — can be removed (serde falls
through to `visit_str`/`visit_bytes`).

### Derive Macros

The RFC envisions `#[derive(Save)]` and `#[derive(Load)]` to eliminate boilerplate. The
current macro-rules (`save_fields!`, `load_fields!`) are a stopgap. A proc macro crate
(`diskann-record-derive`) would be the next step.

## File Map

```
diskann-record/
├── Cargo.toml              # deps: serde, serde_json, anyhow
├── HANDOFF.md              # this document
└── src/
    ├── lib.rs              # module structure, is_reserved(), round-trip test
    ├── number.rs           # Number enum with custom Serialize/Deserialize
    ├── version.rs          # Version { major, minor, patch }
    ├── save/
    │   ├── mod.rs          # Save/Saveable traits, blanket impl, save_fields! macro,
    │   │                   # primitive impls, save_to_disk entry point
    │   ├── value.rs        # Value, Record, Versioned, Handle + serde impls
    │   ├── context.rs      # ContextInner (dir-backed), Context, Writer
    │   └── error.rs        # save::Error (anyhow newtype)
    └── load/
        ├── mod.rs          # Load/Loadable traits, blanket impl, load_fields! macro,
        │                   # primitive impls, load_from_disk entry point
        ├── context.rs      # ContextInner, Context, Object, Array, Iter, Reader
        └── error.rs        # load::Error (light/heavy split), Kind enum
```

## Key Design Rationale

Decisions that may not be obvious from the code alone:

1. **Why two traits per side?** Primitives shouldn't carry versions, but versioned structs
   and primitives must coexist in the same `Value` tree. The blanket impl bridges them.

2. **Why `Cow` in `Value`?** Save path borrows from structs (`Cow::Borrowed`). Load path
   owns from JSON (`Cow::Owned`). Same type, different usage patterns.

3. **Why light/heavy errors on load?** Loading tries the current version first, falls back
   to legacy. The first attempt's failure should be near-free (just a `Kind` enum) since
   it's expected to fail for older data. Only the final failure needs rich diagnostics.

4. **Why eager tree building?** The manifest is metadata (kilobytes). Lazy/deferred
   serialization adds lifetime complexity for no practical gain. Artifacts (gigabytes)
   stream directly to files — they're never in the tree.

5. **Why separate save/load error types?** The light/heavy split only makes sense for
   loading (speculative probing). Save errors are always "something went wrong." Unifying
   them would force the save side to carry unused machinery.
