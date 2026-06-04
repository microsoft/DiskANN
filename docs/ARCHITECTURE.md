## Design Patterns And Principles

### Simplicity is the foundation of code quality

Correct, maintainable systems come from code that is easy to understand, reason about, and evolve. We actively protect that simplicity.
Example: Code should be readable locally. If understanding a single trait bound requires chasing through a long chain of files, the abstraction has become too indirect.

### Build simple composable units

Build small blocks with simple responsibilities.
A good block has a durable API and a clear purpose, so it rarely needs changes.
Extend behavior by adding code and composing core units, not by modifying them.
If functionality grows without touching old code, the abstractions are sound.

### Keep complexity bounded

Keep code simple at every abstraction level.

1. Deep trait hierarchies increase complexity.
   A blanket impl or supertrait bound carries all invariants and edge cases of
   its ancestors. Use trait hierarchies only when there is a genuine "is-a"
   relationship between the capabilities they represent.

2. Prefer composition and encapsulation.
   Wrap types and delegate via traits rather than building deep trait
   hierarchies.

### Dependencies point inward

[Onion Architecture](https://medium.com/@dorinbaba/n-tier-vs-hexagonal-vs-onion-vs-clean-architecture-in-very-simple-terms-68f66c4dba22) solves coupling to infrastructure.
Without it, business logic depends directly on databases, serialization frameworks, file systems, etc. — making it hard to test, hard to swap technologies, and brittle to change.

Its fundamental rule is that **all dependencies must point inwards**; outer layers
can depend on inner layers, but inner layers have zero knowledge of outer
layers.

#### Example Of Layers

```
    ┌─────────────────────────────────────────────────┐
    │            4. Infrastructure                    │
    │    (DB, file I/O, serialization, UI, logging)   │
    │                                                 │
    │   ┌──────────────────────────────────↓──────┐   │
    │   │       3. Application Services           │   │
    │   │         (use-case orchestration)        │   │
    │   │                                         │   │
    │   │   ┌──────────────────────────────↓──┐   │   │
    │   │   │      2. Domain Services         │   │   │
    │   │   │   (cross-entity logic,          │   │   │
    │   │   │    interface definitions)       │   │   │
    │   │   │                                 │   │   │
    │   │   │   ┌───────────────────────↓─┐   │   │   │
    │   │   │   │    1. Domain Model      │   │   │   │
    │   │   │   │  (entities, values,     │   │   │   │
    │   │   │   │   pure business logic)  │   │   │   │
    │ ┌─────────────────────────────────↓───↓───↓──↓┐ │
    │ │      Application Core (Context, ILogger)    │ │
    │ └─────────────────────────────────────────────┘ │
    │   │   │   └─────────────────────────┘   │   │   │
    │   │   │                                 │   │   │
    │   │   └─────────────────────────────────┘   │   │
    │   │                                         │   │
    │   └─────────────────────────────────────────┘   │
    │                                                 │
    └─────────────────────────────────────────────────┘

```

0. **Application Core (cross-cutting foundation)**
   A pure, lightweight auxiliary layer that spans all other layers. It provides
   shared primitives (Request Context, Logger Abstraction) that any layer may
   depend on without violating the inward-dependency rule. Because it contains
   no business logic and no infrastructure concerns, it remains safe to
   reference from everywhere.

1. **Domain Model (the center)**
   The innermost ring contains domain entities, value objects, and domain
   events. It represents the pure state and behavior of the business and
   contains no data-access or infrastructure concerns.

2. **Domain Services**
   The ring surrounding the Domain Model. Implements logic that involves
   multiple domain entities. This layer also defines the interfaces
   (abstractions) for operations that depend on external systems (e.g.,
   database saving or emailing).

3. **Application Services (Use Cases)**
   The layer wrapping the Domain Services. Orchestrates the flow of data to
   and from the domain entities to accomplish specific use cases. It implements
   the interfaces defined by the inner Domain Services without knowing the
   exact implementation details (e.g., which specific database is used).

4. **Infrastructure (the outermost ring)**
   The outermost layer containing concrete implementations. Handles external
   concerns like database access, UI frameworks, message queues, and logging.
   It implements the interfaces defined in the inner layers to actually execute
   tasks.

#### Core Principles

- **Dependency Inversion.** Inner layers define the rules and interfaces;
  outer layers must implement them. This allows the business logic to remain
  completely isolated and easily testable.
- **Interchangeability.** Because the core does not rely on outer layers, you
  can switch out an entire database technology or UI framework with minimal
  changes to the business logic.

### Serialization

**Keep serialization at the edges.** How data is persisted (format, I/O, file layout)
belongs in an adapter layer.

1. **Domain types stay format-agnostic; adapters own the format.** A domain
   type may derive `Serialize`/`Deserialize` as a format-neutral marker, but
   must not contain format-specific logic. A dedicated adapter picks the
   format, performs I/O, and converts errors — so swapping formats
   (bincode → protobuf) means writing a new adapter, not rewriting domain
   types.

2. **Deserialization must respect construction invariants.** Loading a type from
   disk must go through the same validation path as constructing it in memory
   (e.g., via a builder or `new`). Never bypass invariant checks by directly
   populating struct fields.

3. **Inner crates must not depend on serialization frameworks.** If a Tier 1
   type like `Metric` needs to be persisted, the serialization code lives in a
   higher-tier crate.

---

### Alex's Notes:
- Keep code simple
- Separation of concerns
  - New requirement for domain-level inner circle - support Save() and destroy, so later the state can be restored
  - How to persist that information, versioning, etc. - is not responsibility of the component.

Guiding questions:
- Should Deserialization() bypass safety checks? For example,  Config::load bypasses checks in Builder
- Yes - it's ok to have separate ::load() function
- No - then we essentially having two constructors accepting same parameters in different forms (struct vs BinaryReader)
 - Alternative option is chaining - it makes the contract between DTO and public API explicit
   - The component is reponsible for meeting the requirements - it provides api to create a component, and API to get all parameter values required to recreate the component.
   - A separate component is responsible for persisting the values (incl. versioning, etc.) and controlling mapping between DTO object and the parameters.

Formats like protobuf help reduce probability of breaking changes.
- Adding optional values is not a breaking change
- Adding required values is a breaking change - very infrequent.

Example:
Checkpoint manager - restoration is a
.save() -> offset
.create(config, offset)

.save() calls internal components - they call .save() but mainly they also choose serialization format. We want to make them format agonstic - the whole save function should be format agnostic.


## Violations Found in the PR

  ❌ Principle 3: "Dependencies flow toward the domain, not away from it"

  This is the biggest violation. Two foundation/algorithm crates now depend on diskann-record:

  ┌──────────────────┬─────────────────────┬──────────────────┐
  │ Crate            │ Tier                │ New dep          │
  ├──────────────────┼─────────────────────┼──────────────────┤
  │ diskann-vector   │ Tier 1 (Foundation) │ diskann-record   │
  ├──────────────────┼─────────────────────┼──────────────────┤
  │ diskann          │ Tier 3 (Algorithm)  │ diskann-record   │
  └──────────────────┴─────────────────────┴──────────────────┘

  Metric (a Tier 1 type in diskann-vector) now has impl Save and impl Load directly on the domain type, pulling a serialization framework into a foundation crate. Same for Config, PruneKind, IntraBatchCandidates, InsertRetry, and DiskANNIndex in the core diskann crate.

  ❌ Principle 1: "Domain types stay format-agnostic"

  Every domain type (Metric, Config, DiskANNIndex, StartPoints, etc.) now contains format-specific logic — constructing Records, inserting Value::Null for enum tags, calling save_fields!, reading Object fields. This is exactly the "constructing Records, choosing wire encodings" that Principle 1 prohibits.

  ❌ Principle 5: "Switching formats should be a localized change"

  Save/Load impls are spread across 7 files in 3 crates: metric.rs, config/mod.rs, config/experimental.rs, graph/index.rs, index_configuration.rs, common.rs, plus provider files. Swapping diskann-record for another format would require touching every one of these domain files.

  ✅ Principle 2: Partially followed

  diskann-record itself is a well-designed adapter framework (context, handles, versioning). The problem is that it's used inline on domain types instead of from a separate adapter layer.

  ✅ Principle 4: Followed

  IndexConfiguration::load correctly skips transient fields (prefetch_*). DiskANNIndex::load recreates the scratch pool. Construction invariants appear respected.

  ---------------------------------------------------------------------------------------------------------------------------------------------------------

  Recommendation

  Move all impl Save/Load blocks out of the domain crates into a dedicated adapter crate (e.g. diskann-record-adapters or into diskann-providers). The adapter crate depends on both the domain types and diskann-record. The domain crates stay format-agnostic with zero serialization dependencies.

❯ yuo are saying that principle 4 followed. what about this: Config::load bypasses its own Builder

● You're right — I was wrong to say Principle 4 is followed. Here's why:

  ❌ Principle 4: Config::load bypasses Builder validation

  Config::load directly populates the struct fields (Self { pruned_degree, max_degree, ... }) from deserialized data. It completely skips Builder::build() → Config::try_from_builder(), which is the only construction path that enforces these invariants:

  ┌─────────────────────────────────┬─────────────────────────────┬─────────────────────┐
  │ Invariant                       │ Builder check (lines)       │ Load check          │
  ├─────────────────────────────────┼─────────────────────────────┼─────────────────────┤
  │ max_degree ≥ pruned_degree      │ 266–268                     │ None                │
  ├─────────────────────────────────┼─────────────────────────────┼─────────────────────┤
  │ max_backedges ≤ pruned_degree   │ 293–296                     │ None                │
  ├─────────────────────────────────┼─────────────────────────────┼─────────────────────┤
  │ alpha is finite / valid         │ 248 (TODO, but slot exists) │ None                │
  ├─────────────────────────────────┼─────────────────────────────┼─────────────────────┤
  │ backedge_ratio ∈ (0, 1]         │ 280–282                     │ None                │
  ├─────────────────────────────────┼─────────────────────────────┼─────────────────────┤
  │ slack ≥ 1.0 && finite           │ 255–257                     │ None                │
  ├─────────────────────────────────┼─────────────────────────────┼─────────────────────┤
  │ All values are non-zero u32     │ 250, 253, 270, etc.         │ (type-level only)   │
  └─────────────────────────────────┴─────────────────────────────┴─────────────────────┘

  A hand-edited or corrupted manifest with max_degree: 4, pruned_degree: 64 would load successfully and produce a Config that Builder would have rejected. The loaded Config would then silently cause excessive pruning or other algorithmic misbehavior.

  The same issue applies to IndexConfiguration::load (populates fields directly) and StartPoints::load (reconstructs { start, end } without going through StartPoints::new).

  The fix: Config::load should deserialize the raw values, feed them into a Builder, call .build(), and map the ConfigError to a load::Error. That way disk data goes through the same gate as in-memory construction.