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
