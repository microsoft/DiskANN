---
name: diskann-pr-review
description: Review pull requests, diffs, or staged changes in the DiskANN Rust workspace. Use when asked to review a PR, review code changes, check a diff against DiskANN conventions, pre-flight a PR before submitting, or address review feedback. Encodes blocking rules (patch coverage, unsafe, error-handling tiers, crate layering, baseline tests) distilled from merged PR review history.
---

# DiskANN PR Review

Review changes the way DiskANN maintainers actually review them. This skill encodes patterns
distilled from the review history of merged PRs, the repo's own written conventions and judgements/values 
from the core contributors to the repo.

**Companion file:** [rules.md](rules.md) — the full rule catalog with rationale and evidence. Read
it when you need depth on a category, or when a rule needs justification in a review comment.

**Authoritative sources this skill defers to** (read them if a rule seems to conflict):
[AGENTS.md](../../../AGENTS.md) · [rfcs/00109-unsafe-rust.md](../../../rfcs/00109-unsafe-rust.md) ·
[clippy.toml](../../../clippy.toml) · [.codecov.yml](../../../.codecov.yml) ·
[.github/copilot-instructions.md](../../copilot-instructions.md) ·
[rfcs/README.md](../../../rfcs/README.md)


## Review workflow

### 1. Orient before reading the diff

Establish scope, then read with the right lens:

- **Gather ground truth first.** Do not infer facts you can fetch. Three commands, always:

  ```powershell
  gh api "repos/microsoft/DiskANN/pulls/<PR>/files?per_page=100" --paginate   # authoritative status/renames
  gh api "repos/microsoft/DiskANN/pulls/<PR>/comments?per_page=100" --paginate
  gh api "repos/microsoft/DiskANN/issues/<PR>/comments?per_page=100" --paginate  # includes Codecov
  ```

  `gh pr diff` does **not** emit rename headers — use the `files` endpoint for `status` and
  `previous_filename`. Read existing review threads before proposing structural changes so you don't
  re-litigate a settled decision (rule `read-existing-review-threads`), and read the Codecov body
  before any coverage claim (rule `patch-coverage`). Filter humans with `login -notlike '*[bot]'` and
  `-ne 'Copilot'`.
- **Which crates/tiers are touched?** Tier 1–2 (`diskann-wide`, `diskann-vector`, `diskann-linalg`,
  `diskann-utils`, `diskann-quantization`) → expect bespoke `thiserror` errors, no allocation, SIMD
  care. Tier 3 (`diskann`, `diskann-providers`, `diskann-disk`, `diskann-label-filter`) → expect
  `ANNError`, baseline tests. Tier 4 (benchmarks, `diskann-tools`) → `anyhow` is fine.
- **What kind of change is it?** New algorithm · refactor · perf optimization · new crate · bug fix ·
  benchmark/tooling · dependency bump. Each has a different required-evidence bar (see §3).
- **Does it need an RFC?** Cross-cutting changes, new crates, new cross-crate traits, new distance
  functions / storage layouts / index formats, and anything with backward-compat implications
  should have an RFC. Routine single-crate API additions, bug fixes, and refactors do not.
- **If the PR claims to be a pure refactor**, enumerate trait impls and constructors on the affected
  types before and after. Silently dropped capabilities are the characteristic failure of large
  mechanical PRs (rule `refactor-preserves-api`).

### 2. Run the blocking checklist

These are the items that stop a merge. Verify each explicitly — do not assume.

| # | Blocking check | Where it's enforced |
|---|---|---|
| 1 | **Patch coverage ≥ 90%** on changed lines | `.codecov.yml` (`informational: false`) |
| 2 | Every `unsafe` block has a `// SAFETY:` comment naming the invariant | workspace clippy lint `undocumented_unsafe_blocks` |
| 4 | No new crate-level catch-all error enum | AGENTS.md ("Do Not") |
| 5 | Error type matches the crate's tier (bespoke / `ANNError` / `anyhow`) | AGENTS.md |
| 6 | No `unwrap()` / `expect()` / `panic!` in non-test library code | crate-level `cfg_attr` lints |
| 7 | New/changed algorithm behavior has a **baseline** test, not an eyeball test | `diskann/src/test/cache.rs` |
| 8 | Crate tier dependency rules respected (no Tier 3 → benchmark-runner/core/simd) | AGENTS.md |
| 9 | License header present on every new file | `.github/copilot-instructions.md` |
| 10 | No unit tests deleted without stated justification | `.github/copilot-instructions.md` |
| 11 | New dependencies justified; no gratuitous transitive bloat | `.github/copilot-instructions.md` |
| 12 | Public struct fields don't bypass constructor validation | recurring maintainer objection |
| 13 | Concurrency: lock ordering consistent, atomics not split across signals, races documented | recurring maintainer objection |
| 14 | Arch-specific intrinsics validated cross-platform (SDE / QEMU) | AGENTS.md, `diskann-wide/README.md` |
| 15 | `cargo fmt --all --check` and `cargo clippy --workspace --all-targets -- -D warnings` clean | CI |

### 3. Apply the evidence bar for the change type

Match the demand to the change. Ask for exactly the evidence the change type requires:

- **Perf optimization** → before/after numbers, stated dataset + hardware + parameters. A negligible
  delta is a fine answer; *no* number is not.
- **Algorithm change** → baseline test capturing **both IDs and distances**, plus invariant
  assertions (results actually filtered, IDs in range). Baselines alone are insufficient — a broken
  baseline can be checked in.
- **Concurrency change** → stress test with enough threads/iterations to surface non-determinism;
  document benign races and why they're acceptable.
- **New public API** → at least one test that calls it (guards against silent removal) and rustdoc
  for non-obvious behavior, `# Errors` / `# Safety` / `# Panics`.
- **SIMD / intrinsics** → cross-arch validation per `diskann-wide/README.md`; check coverage on the
  scalar fallback too.
- **New benchmark / example JSON** → wired into integration tests; no `#[serde(default)]` silently
  hiding run parameters.
- **Refactor** → complete, not partial. A half-flattened module hierarchy draws "it feels partially
  done."

### 4. Read for architecture, not just correctness

The highest-value review comments in this repo are structural. Ask:

- Does this duplicate logic that already exists for another index/provider type? Could a shared
  trait serve both?
- Does the new trait leak implementation details or force SemVer commitments (e.g. an open trait
  over the ISA matrix that should be sealed)?
- Is this generic where it should be `dyn`? Generics here have a real monomorphization/compile-time
  cost, especially in benchmark and provider layers.
- Does this belong in this crate? Moving code to a higher tier often removes the need for a new
  dependency entirely.
- Will this need to be undone by in-flight work? Prefer a temporary private shim over a public
  trait that a pending refactor will delete.

### 5. Write the review

Match the output shape to the surface you're writing to.

**Posting a review on GitHub** (Copilot code review, `gh`, the REST API). A review is a set of
comments individually anchored to a file and line — there is no single document, so section headings
have nowhere to render. Instead:

- One finding per comment, anchored to the narrowest line range that shows the problem.
- Carry severity in the opening words of the comment body: `blocking:`, `consider:`, `nit:`.
- Use ` ```suggestion ` blocks whenever the fix is a concrete edit — they are one click to apply.
- Put the 2–4 sentence overall assessment in the review summary body, not in a line comment.
- **Do not claim to block a merge.** A bot review is advisory; the Codecov patch gate and CI are what
  actually gate. Say "this needs X before merge", not "I am blocking this".

**Producing a review as a document** (chat, CLI, pre-flight before you push):

```markdown
## Summary
<2-4 sentences: what the PR does and your overall assessment>

## Blocking
<numbered; each with file/line reference, the rule, and a concrete suggested fix>

## Non-blocking
<numbered; improvements worth making but not merge-gating>

## nit
<cosmetic only>

## Questions
<genuine uncertainty where you need author context>
```

Conventions (both surfaces):
- Prefix cosmetic comments with `nit:` — the repo uses this to mean explicitly non-blocking, and
  reviewers approve PRs while leaving them.
- **Propose the fix**, don't just flag the problem. Sketch the trait, name the helper, link the
  existing utility.
- Wrap identifiers in backticks in prose.
- Acknowledge good choices — this is a normal and expected part of reviews here.
- Separate "must fix now" from "follow-up PR". Deferring non-critical work to a named follow-up is
  accepted practice; say so explicitly rather than blocking.
- If you're unsure whether something is a real problem, ask rather than assert.

---

## Calibration

**Do flag:** missing baseline tests on algorithm changes · unbenchmarked perf claims · public fields
that break invariants · duplicated abstractions · undocumented `unsafe` · error-type tier mismatches
· lock-ordering inversions · silent config defaults · partial refactors · docs that restate the
signature · trait impls or constructors dropped in a "mechanical" change · infallible constructors
that gained a panic · `pub(crate)` → `pub` without a reason · error types that lost a recovery
payload · PR descriptions that no longer match the diff.

**Don't flag:** derived-trait tests (`Clone`, `Debug`, `PartialEq`) — the repo explicitly does not
want them · code duplication *inside* unit tests when it aids readability · missing docs on obvious
`pub(crate)` helpers · `unwrap()` in test code · style already settled by `rustfmt` · decisions
already agreed in an existing review thread.

**Never assert a number you didn't verify at its source.** Coverage comes from the Codecov comment,
not from counting `#[test]`. File renames come from the `pulls/<PR>/files` endpoint, not from reading
a diff. Perf claims come from a benchmark, not from reasoning about the code. A confidently wrong
number destroys the credibility of every other item in the review.

**Don't over-report.** A review with 40 low-value comments is worse than one with 5 that matter.
Lead with the structural issues.
