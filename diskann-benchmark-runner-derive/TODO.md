# Benchmark Input Discoverability

The benchmark configuration files are a portable protocol: they support sharing
configurations across machines, batching benchmarks, validating runs, and analyzing results.
The missing piece is a human-facing projection of that protocol.

Use dedicated deserialization DTOs as the public configuration boundary. Convert DTOs into
validated runtime types after deserialization. The derive macro should document and enforce a
deliberately constrained DTO language rather than attempt general Rust or Serde reflection.

## Initial contract

- [ ] Support named structs.
- [ ] Support explicitly tagged enums, including unit, tuple, and struct variants where needed.
- [ ] Require documentation for configuration types, fields, and enum variants.
- [ ] Support known primitive and domain leaf types.
- [ ] Support documented DTO composition through selected containers such as `Option<T>` and
      `Vec<T>`.
- [ ] Read representation-changing metadata from Serde attributes so Serde remains the source
      of truth.
- [ ] Support `rename` and `rename_all`.
- [ ] Support enum `tag` and `content`.
- [ ] Decide whether `default` and `alias` should be included in the initial metadata model.
- [ ] Reject unsupported types and Serde attributes with actionable `syn::Error` diagnostics.
- [ ] Reject asymmetric serialization and deserialization names unless the metadata model
      explicitly represents both.
- [ ] Reject `untagged`, `remote`, `with`, `deserialize_with`, `try_from`, and skipped input
      fields initially.
- [ ] Add an explicit escape hatch for intentional opaque or externally implemented leaf
      types.
- [ ] Evaluate `flatten` separately; prefer explicit nested DTOs unless flattening provides a
      clear authoring benefit.
- [ ] Keep validation and DTO-to-runtime conversion outside the reflection system.
- [ ] Keep complete examples curated rather than synthesizing arbitrary field values or
      Cartesian products of enum variants.

## Runtime metadata

- [ ] Replace or refine the prototype `Reflect` API around benchmark configuration
      documentation rather than general-purpose reflection.
- [ ] Choose names that communicate the constrained public role, such as `BenchmarkInput` for
      the derive and `DescribeInput` for the generated runtime trait.
- [ ] Represent type, field, and variant documentation.
- [ ] Represent effective serialized names after applying supported Serde rules.
- [ ] Represent nested DTOs and selected containers.
- [ ] Represent accepted enum variants and their tagging strategy.
- [ ] Decide whether metadata should be statically stored or constructed on demand; favor the
      simpler dynamic model unless measurements justify static storage.
- [ ] Provide getters or a renderer-facing API for all metadata.
- [ ] Avoid requiring arbitrary runtime and third-party types to implement the reflection
      trait.

## Examples

- [ ] Add an API for multiple named, curated examples per registered input.
- [ ] Include a short description with each example.
- [ ] Keep examples on the input/DTO API rather than inferring values in the derive macro.
- [ ] Decide whether the derive accepts an examples function:

  ```rust
  #[benchmark(examples = Self::examples)]
  ```

- [ ] Ensure every example serializes and deserializes successfully.
- [ ] Consider validating examples through the normal DTO-to-runtime conversion path.

## CLI integration

- [ ] Expose input descriptions through the dynamic registry.
- [ ] Extend `inputs --describe <tag>` or settle on a clearer equivalent command.
- [ ] Render the input summary, fields, nested objects, enum choices, and important defaults.
- [ ] Render one or more complete examples.
- [ ] Consider `skeleton --input <tag>` for generating a directly editable configuration.
- [ ] Consider emitting a commented JSONC-style template for authoring while retaining strict
      JSON as the canonical shared representation.
- [ ] Preserve feature-gated input behavior and diagnostics.
- [ ] Improve deserialization and validation errors with paths such as
      `jobs[2].input.search.runs[0].search_l`.

## Macro implementation

- [ ] Replace `todo!` branches with structured compile errors.
- [ ] Implement named struct generation.
- [ ] Implement the accepted enum forms.
- [ ] Generate appropriate generic bounds for nested reflected types.
- [ ] Extract and normalize literal rustdoc.
- [ ] Parse the supported Serde subset with `syn::parse_nested_meta`.
- [ ] Implement and test Serde rename rules used by benchmark DTOs.
- [ ] Preserve useful source spans in generated diagnostics.
- [ ] Add explicit diagnostics for every rejected Serde feature.
- [ ] Add documentation-specific helper attributes only for behavior Serde does not control,
      such as examples, hiding documentation, or an opaque leaf override.
- [ ] Do not generate serialization, deserialization, validation, or arbitrary example values.

## Migration

- [ ] Select one simple DTO and one representative complex DTO as the initial vertical slice.
- [ ] Wire those DTOs through derive, registry, CLI rendering, examples, and validation.
- [ ] Use the vertical slice to confirm the output format before migrating all inputs.
- [ ] Convert remaining benchmark-facing inputs to dedicated DTOs where runtime concerns are
      still mixed into deserialization types.
- [ ] Add rustdoc to all exposed DTO fields and variants.
- [ ] Replace custom deserialization patterns where a simpler DTO plus conversion can express
      the same behavior.
- [ ] Add explicit overrides only where external or specialized types make them unavoidable.
- [ ] Migrate remaining registered inputs incrementally rather than requiring an atomic
      workspace-wide conversion.

## Tests

- [ ] Add unit tests for rustdoc extraction and normalization.
- [ ] Add tests for each supported Serde naming and tagging rule.
- [ ] Add compile-fail tests for unsupported item shapes, missing documentation, unsupported
      Serde attributes, and invalid combinations.
- [ ] Compare generated names and shapes with actual `serde_json` serialization.
- [ ] Add CLI golden tests for descriptions, examples, nested DTOs, enums, feature-gated
      inputs, and error output.
- [ ] Test that curated examples round-trip.
- [ ] Run formatting, clippy with warnings denied, and targeted workspace tests.

## Suggested rollout

- [ ] Phase 1: named DTO structs, documentation, known leaves and containers, and CLI rendering.
- [ ] Phase 2: tagged enums, Serde naming, and curated examples.
- [ ] Phase 3: migrate representative inputs and refine diagnostics.
- [ ] Phase 4: add flattening or other Serde behavior only in response to concrete DTO needs.
- [ ] Phase 5: migrate the remaining benchmark inputs and stabilize the public API.

