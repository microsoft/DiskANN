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
