---
name: new-crate
description: 'Scaffold a new workspace crate. Use when "add a crate", "create a new crate", "new package", or "scaffold crate".'
---

# New Crate Scaffolding

Scaffold a new Cargo workspace crate following the tiered architecture.

## When to Use

- Adding a new crate to the workspace
- Creating a new module that should be a separate crate

## Process

### Step 1: Determine Tier

<!-- TODO: Document criteria for tier placement -->
- Tier 1 (Foundation): SIMD, vector primitives, platform utils
- Tier 2 (Core Libraries): Linear algebra, utilities, quantization
- Tier 3 (Algorithm & Storage): Core algorithm, providers, disk indexing
- Tier 4 (Infrastructure & Tools): Benchmarks, CLI tools

### Step 2: Create Crate

1. Run `cargo new <crate-name> --lib` in the repo root
2. Add to `[workspace.members]` in root `Cargo.toml`
3. Add to `default-members` if it's Tier 1 or Tier 2
4. Set `version`, `edition`, `license` from `workspace.package`
5. Add `[lints] workspace = true` to inherit workspace lints

### Step 3: Configure Dependencies

- Only depend on crates from equal or lower tiers
- Add workspace dependency entry in root `Cargo.toml` if other crates will depend on it
- Follow dependency rules documented in AGENTS.md

## Constraints

- Never add Tier 3 dependencies to benchmark-runner, benchmark-core, or benchmark-simd
- Use `workspace = true` for shared dependencies

## Validation

- `cargo check -p <crate-name>`
- `cargo clippy -p <crate-name> --all-targets --config 'build.rustflags=["-Dwarnings"]'`
