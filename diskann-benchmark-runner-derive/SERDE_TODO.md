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
  - The `heck` dependency has been removed.
- [X] Confirm the intended treatment of internally tagged newtype variants.
  - Newtypes remain supported because Serde accepts newtypes containing struct/map-like
    values.
  - Serde compatibility tests cover non-empty and empty struct payloads.
- [X] Strip the `r#` prefix from raw field and variant identifiers so reflected names match
  Serde's wire names.

## Attribute validation

- [ ] Reject internally tagged tuple variants at the variant span.
  - Newtype variants must remain supported because Serde permits newtypes containing
    struct/map-like values.
  - Do not classify syntactic newtypes as tuple variants.
- [ ] Reject `#[reflect(type_name = "...")]` on every generic type, including types whose
  only generic parameters are lifetimes.
  - Check `input.generics.params.is_empty()` rather than the generated list of displayed
    type and const arguments.
- [ ] Reject adjacent enum representations whose `tag` and `content` names are equal.
- [ ] Reject fields in internally tagged variants whose effective serialized name conflicts
  with the internal tag.
  - Compare names after applying field `rename` and variant-level `rename_all`.
- [ ] Reject duplicate effective serialized names:
  - enum variants after container `rename_all` and variant `rename`;
  - named struct fields after container `rename_all` and field `rename`;
  - named variant fields after variant-level `rename_all` and field `rename`.
- [ ] Reject misplaced `reflect` attributes instead of silently ignoring them.
  - Until field-level features such as `#[reflect(opaque)]` exist, any `reflect` attribute
    on a field or variant should produce an unsupported/misplaced-attribute diagnostic.
- [ ] Add deliberate diagnostics for asymmetric Serde naming syntax instead of relying on a
  lower-level parse error:
  - `rename(serialize = "...", deserialize = "...")`;
  - `rename_all(serialize = "...", deserialize = "...")`.
- [X] Keep unsupported representation-changing attributes rejected until the reflection
  model explicitly supports them, including `untagged`, `flatten`, `skip*`, `default`,
  `alias`, `with`, `remote`, `from`, and `try_from`.

## Tests

- [X] Update enum compatibility coverage to expect renamed variants such as `"unit"` rather than
  `"Unit"`.
- [X] Verify the generated enum representation, including both `tag` and `content` for an
  adjacently tagged enum, against serialized JSON.
- [ ] Add naming tests that compare reflection metadata with `serde_json`, covering:
  - [X] explicit field and variant `rename`;
  - [X] struct field `rename_all`;
  - [X] enum variant `rename_all`;
  - [X] variant-level `rename_all` for struct-variant fields;
  - [ ] acronym-heavy variants such as `XMLHttpRequest`;
  - [X] explicit `rename` taking precedence over `rename_all`.
- [X] Add representation tests for external, internal, and adjacent tagging.
- [ ] Add compile-fail tests for duplicate attributes, `content` without `tag`, enum-only
  attributes on structs, internally tagged tuple variants, unsupported rename rules, and
  unsupported Serde attributes.
  - [X] duplicate attributes;
  - [X] `content` without `tag`;
  - [X] enum-only attributes on structs;
  - [ ] internally tagged tuple variants;
  - [X] unsupported rename rules;
  - [X] unsupported Serde attributes.
  - Add fixtures for the remaining validation rules above as their implementations land.

## Cleanup and validation

- [X] Fix the `generate_type_name_body` doctest by returning the final
  `f.write_str(">")` result instead of discarding it with a semicolon.
- [X] Run `cargo fmt --all`.
- [X] Run the targeted derive and runner tests.
- [ ] Run Clippy with warnings denied once the implementation and tests settle.

## Deferred Serde features

- [ ] Decide how `default` and `alias` should appear in reflection metadata before accepting
  them.
- [ ] Continue rejecting asymmetric serialization/deserialization names unless the metadata
  model represents both.
- [ ] Continue rejecting `untagged`, `flatten`, `remote`, `with`, `deserialize_with`,
  `try_from`, and skipped input fields until each has an explicit metadata design.
