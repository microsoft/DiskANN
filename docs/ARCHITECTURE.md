## Design Patterns And Principles

### Simplicity is the foundation of code quality

 Protect the simplicity of the codebase. Correct, maintainable systems come from code that is easy to understand, reason about, and evolve.

### Keep complexity bounded

1. Deep trait hierarchies increase complexity.
   A blanket impl or supertrait bound carries all invariants and edge cases of
   its ancestors. Use trait hierarchies only when there is a genuine "is-a"
   relationship between the capabilities they represent.

2. Prefer composition and encapsulation.
   Wrap types and delegate via traits rather than building deep trait
   hierarchies.

## Patterns We Use

### Build composable units

Think SQL as a programming language. All we do is select, map, filter, join - it is easy (for the most part) to reason over a SQL script. This is possible because we have the core units that are composed of making a bigger capability stand out. Build primitives that compose together to represent the larger piece of code. Ensure that the abstraction at the high level is a composition of domain level capabilities and avoid abstraction mismatches.

### Serialization

**Keep serialization at the edges.** Domain types should expose constructors
and accessors — nothing more. The decision of *how* to persist data (format,
I/O, file layout) belongs in an adapter or bridge layer, not on the domain
types themselves.

**Principles:**

1. **Domain types stay format-agnostic.** A domain type may derive `Serialize`
   / `Deserialize` as a format-neutral marker, but must not contain
   format-specific logic (e.g., constructing Records, choosing wire encodings,
   or writing sidecar files).

2. **Adapter layers own the format.** A dedicated adapter or manager picks the
   serialization format, performs I/O, and converts errors. Swapping formats
   (bincode → protobuf, JSON → flatbuffers) should require writing a new
   adapter — not rewriting every domain type.

3. **Dependencies flow toward the domain, not away from it.** Foundation and
   algorithm crates must not depend on serialization frameworks. If a Tier 1
   enum like `Metric` needs to be persisted, the serialization code lives in a
   higher-tier crate that imports both the domain type and the serialization
   framework.

4. **Deserialization must respect construction invariants.** Loading a type from
   disk must go through the same validation path as constructing it in memory
   (e.g., via a builder or `new`). Never bypass invariant checks by directly
   populating struct fields from deserialized data.

5. **Switching formats should be a localized change.** If serialization logic
   is spread across domain types, changing the format becomes a codebase-wide
   migration. Centralizing it in an adapter layer keeps the blast radius small.
