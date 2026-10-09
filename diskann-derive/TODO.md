# Reflection: Remaining Work

The reflection model, derive macro, registry catalogue, CLI rendering, and benchmark-input
migration are complete. The supported Serde subset includes field and variant naming, external,
internal, and adjacent enum representations, and validation of the effective reflected shape.

## Correctness

- [ ] Verify `RequiredOption<T>` rejects an omitted field while accepting explicit `null` and a
  present value.
  - Add tests for missing, `null`, present, and serialized values.
  - Remove `#[serde(transparent)]` if it restores `Option<T>`'s missing-field behavior.

## Tests

- [ ] Add focused coverage for `Reflection::visit_with` if direct traversal behavior is not
  sufficiently clear from the registry catalogue tests.
- [ ] Review the consolidated trybuild snapshots after the final diagnostic wording settles.

## Cleanup

- [ ] Resolve warnings around unused primitive-kind metadata.
- [ ] Remove obsolete comments and temporary migration artifacts.
- [ ] Ensure user-facing reflection and CLI documentation matches the final supported behavior.

## Final Validation

- [ ] Run formatting:

  ```bash
  cargo fmt --all --check
  ```

- [ ] Run the focused suites:

  ```bash
  cargo test -p diskann-benchmark-runner-derive
  cargo test -p diskann-benchmark-runner
  cargo check -p diskann-benchmark --all-features
  ```

- [ ] Run workspace Clippy with warnings denied:

  ```bash
  cargo clippy --workspace --all-targets --config 'build.rustflags=["-Dwarnings"]'
  ```

- [ ] Run the workspace tests:

  ```bash
  cargo test --workspace
  ```

## Deliberately Out of Scope

- Directional Serde `rename` and `rename_all` forms may produce a parser error. Reflection stores
  one effective name and does not model separate serialization and deserialization names.
- Continue rejecting `default`, `alias`, `untagged`, `flatten`, `remote`, `with`,
  `deserialize_with`, `try_from`, and skipped fields until each has an explicit metadata design
  justified by a benchmark-input use case.
- Do not add `#[reflect(opaque)]` without another concrete migration need.
- Do not optimize reflection metadata allocation without evidence that registration cost is
  material.
