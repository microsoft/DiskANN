# Serde Support: Resume Notes

The initial Serde attribute parser and code generation are in place. It currently handles
field and variant names, container-level `rename_all`, and external, internal, and adjacent
enum representations.

## Correctness

- [X] Use separate Serde-compatible case conversion for fields and variants.
  - Serde applies different rules in each context.
  - In particular, Serde converts an enum variant such as `XMLHttpRequest` to
    `x_m_l_http_request` under `snake_case`, while `heck` produces
    `xml_http_request`.
  - For fields, `snake_case` is identity and `kebab-case` replaces underscores.
  - Implementing the supported transformations directly should allow removal of the `heck`
    dependency.
- [ ] Reject tuple variants on internally tagged enums with an error at the variant.
- [ ] Confirm the intended treatment of internally tagged newtype variants. Serde accepts
  some forms syntactically, but compatibility depends on the wrapped value's serialized
  shape.
- [ ] Check raw identifiers such as `r#type`; reflected names must match Serde's wire names
  rather than include the raw-identifier prefix.

## Tests

- [ ] Update the enum smoke test to expect renamed variants such as `"unit"` rather than
  `"Unit"`.
- [ ] Assert the generated enum representation, including both `tag` and `content` for an
  adjacently tagged enum.
- [ ] Add naming tests that compare reflection metadata with `serde_json`, covering:
  - explicit field and variant `rename`;
  - struct field `rename_all`;
  - enum variant `rename_all`;
  - variant-level `rename_all` for struct-variant fields;
  - acronym-heavy variants such as `XMLHttpRequest`;
  - explicit `rename` taking precedence over `rename_all`.
- [ ] Add representation tests for external, internal, and adjacent tagging.
- [ ] Add compile-fail tests for duplicate attributes, `content` without `tag`, enum-only
  attributes on structs, internally tagged tuple variants, unsupported rename rules, and
  unsupported Serde attributes.

## Cleanup and validation

- [ ] Fix the `generate_type_name_body` doctest by returning the final
  `f.write_str(">")` result instead of discarding it with a semicolon.
- [ ] Run `cargo fmt --all`.
- [ ] Run the targeted derive and runner tests.
- [ ] Run Clippy with warnings denied once the implementation and tests settle.

## Deferred Serde features

- [ ] Decide how `default` and `alias` should appear in reflection metadata before accepting
  them.
- [ ] Continue rejecting asymmetric serialization/deserialization names unless the metadata
  model represents both.
- [ ] Continue rejecting `untagged`, `flatten`, `remote`, `with`, `deserialize_with`,
  `try_from`, and skipped input fields until each has an explicit metadata design.
