# DiskANN Review Rules — Detailed Catalog

Companion to [SKILL.md](SKILL.md). Each rule states what to enforce and the evidence behind it.

## How to edit this file

This is a living catalog. Add, amend, and delete rules freely.

**Rule identity is the slug in the heading** — `### crate-tier-dependencies` — never a number. Slugs
are stable: keep them unchanged when you reword a rule or move it to another section, so references
from `SKILL.md` and from past review comments keep resolving. To retire a rule, delete its block;
git history is the record. Never renumber.

**Severity** is the tag after the slug: `BLOCK` stops a merge · `MAJOR` should be fixed before merge
absent a good reason · `NIT` is non-blocking · `NOTE` is context rather than an enforceable rule.

**Grounding.** Every rule shows where it came from, in one of three forms:

| Form | Who may use it |
|---|---|
| A quote from a review, cited `— #1151, @handle` | anyone |
| A repo source — `AGENTS.md`, `.codecov.yml`, `clippy.toml`, an RFC | anyone |
| `Source: maintainer judgement — @handle, YYYY-MM` | **humans only** |

The third form exists so you are never blocked on research: if you know a rule is right, write it
and sign it. An agent updating this file may use only the first two — anything it contributes must
stay independently checkable. See [MAINTENANCE.md](MAINTENANCE.md) for the refresh procedure.

---

## 1. Architecture & API design

### private-fields-invariants `BLOCK`

**Struct fields stay private; invariants live in constructors.**
Public fields silently void every check the constructor performs, and the invariant can never be
reinstated without a breaking change.

> "A `pub` field for `knn` undoes the validation check in the constructor." — #1151, @hildebrandmw
>
> "Usual feedback applies: making these fields public invalidates the invariants established by the
> constructor. Please make these private." — #1131, @hildebrandmw

### no-duplicate-abstraction `MAJOR`

**Don't duplicate an abstraction across index types.**
When a second index type (flat, graph, disk) needs behavior the first already has, the answer is a
shared trait — not a parallel implementation. This is one of the most consistent structural
objections in the repo.

> "One big concern I have is that this does not really share much with the existing graph code. Even
> though the desire is to share code, post-process routines like diversity search will still need to
> be implemented twice." — #983, @hildebrandmw

### refactors-complete `MAJOR`

**Refactors must be complete.**
A partially-flattened hierarchy is worse than either endpoint: it carries both the old and new
mental models.

> "It feels partially done… Does it make more sense to flatten everything up into the top level and
> get rid of backend entirely?" — #1168, @hildebrandmw

### question-new-crates `MAJOR`

**Question new top-level crates.**
Prefer a feature-gated module inside an existing crate. Crate proliferation hurts discoverability
and build times.

> "How many top-level crates do we actually want to have here? If we end up with 10–20
> diskann-benchmark-* crates, is that going to be a problem?" — #1027, @arrayka

Related: internal crates should be `publish = false`.

### move-up-tier-not-dep-down `MAJOR`

**Move code up a tier instead of adding a dependency downward.**
Relocating a helper to a higher-tier crate frequently removes the need for a new dependency in a
low-tier crate.

> "Moving this piece of code to `diskann-tools`… This would resolve your concern about adding `rayon`
> and `anyhow` to `diskann-label-filter`." — #1099, @hildebrandmw

### crate-tier-dependencies `BLOCK`

**Respect crate tier dependency rules.**
Per [AGENTS.md](../../../AGENTS.md): Tier 1–2 may be depended on by anything; `diskann` may be
depended on by equal-or-higher tiers except as noted; **do not** add Tier 3 crates as dependencies
of `diskann-benchmark-runner`, `diskann-benchmark-core` (`diskann` itself is allowed), or
`diskann-benchmark-simd`.

### no-module-name-in-type `NIT`

**Don't repeat the module name in the type name.**
`flat::SearchStrategy`, not `flat::FlatSearchStrategy`.

> "In general, I'm not a fan of prefixing everything with `Flat`. We already have the `flat` module so
> `flat::SearchStrategy` reads fine to me…" — #983, @hildebrandmw

### use-constructor-consistently `NIT`

**Use the constructor consistently once one exists.**
Mixed `Type { field: … }` literals and `Type::new(…)` in the same file is a smell.

> "If we're going to have a constructor, we might as well consistently use it…" — #1267, @hildebrandmw

### no-public-trait-pending-deletion `MAJOR`

**Don't add public traits that pending work will delete.**
When a refactor is in flight, a temporary private shim beats a public trait that must be removed.

> "If you want to start programming against this API, what if you introduce your own temporary
> `TemporaryBuildQueryComputer`… hold off on post processing for a short period to bottom out on
> #1067." — #983, @hildebrandmw

### refactor-preserves-api `BLOCK`

**A "pure refactor" must not silently drop a trait impl or constructor.**
When a PR claims to be mechanical, enumerate the trait impls and constructors on each affected type
before and after. Capability loss hides easily inside a large rename, especially when trait
reorganisation causes two impls to overlap and one gets cut to make the compiler happy.

> "This seems pretty bad that users can't implement a broadcasted constructor." — #1269, @hildebrandmw

The author's reply confirms the failure mode exactly: *"That broadcast constructor got dropped by
accident: on `main`, `BlockTransposed` had both `NewOwned<T>` (value-fill) and `NewOwned<Defaulted>`
(default), which overlap… the compiler only tolerated it while `Defaulted` was a local type. Moving the
framework here made `Defaulted` foreign, the impls collided, and round-1 resolved it by cutting the
value-fill one."*

Practical check: for each type whose traits were reorganised, diff the set of `impl` blocks. An impl
that vanished with no replacement is a capability regression until proven otherwise.

### no-new-panicking-constructor `MAJOR`

**Don't turn an infallible constructor into a panicking one.**
If a constructor's value proposition was that it cannot fail, adding a panic is a downgrade even when
the panic is "unreachable". Prefer encoding the precondition in the type (`NonZeroUsize`) or returning
a `Result`.

> "The advantage of the `row_vector`/`column_vector` constructors are to provide light-weight,
> infallible ways of building matrices. Adding a panicking path here seems like a step backwards in
> library design." — #1269, @hildebrandmw

> "Instead of panicking, maybe throw an error? Or better yet, use `NonZeroUsize`?" — #1269, @arkrishn94

### visibility-widening-needs-reason `MAJOR`

**Widening visibility needs a stated reason.**
`pub(crate)` → `pub` is an API commitment, not a mechanical consequence of moving code. Raw-pointer and
`unsafe` accessors deserve the most scrutiny: once public they constrain every future refactor.

> "Should this be public?" — #1269, @arkrishn94 (on `as_raw_mut_ptr`)

A legitimate answer exists — in #1269 the framework moved crates while its implementors stayed behind,
so `pub(crate)` no longer reached them — but it must be stated, and the alternatives (sealed trait,
`#[doc(hidden)]`, moving the implementors too) should be considered rather than assumed away.

---

## 2. Correctness & concurrency

### one-atomic-per-signal `BLOCK`

**One atomic per signal; don't spread state across several.**
Interactions between multiple atomics are the hardest class of bug to reason about here.

> "Try to avoid spreading signals across multiple atomics because weird things can happen that are
> very hard to reason about." — #1050, @hildebrandmw

### ordering-vs-lock-acquisition `BLOCK`

**Check read/write ordering against lock acquisition.**
A flag read before the relevant lock is acquired is a race even when each individual access is
atomic.

> "There's a race here still. This reads `quantization_enabled` before acquiring the `next_id` read
> lock, but `enable_quantization` sets `quantization_enabled` **after** acquiring the `next_id` write
> lock." — #1050, @hildebrandmw

### consistent-lock-order `BLOCK`

**Lock acquisition order must be globally consistent.**
Inversions are easy to introduce across functions and invisible locally.

> "There is one more potential deadlock - calling `expand_to` acquires the write lock for `max_block`
> after the write lock for `next_id` is held, but earlier in the code, `refill_fast_free_list` goes
> `max_block` -> `next_id` in the other direction." — #1050, @hildebrandmw

### document-benign-races `MAJOR`

**Document benign races explicitly, with the reasoning.**
Accepted races are fine. Undocumented ones are not — the next reader cannot tell the difference
between "considered and accepted" and "not noticed."

> "Mark and I considered the races that were left and convinced ourselves that they result in benign
> issues which we accept as part of not stopping the world for mutation in the index." — #1146,
> @metajack

For epoch-based reclamation, state the yield requirement so long-running readers are catchable in
review:

> "Based on discussion, readers have to yield for epoch to transition. Request updating comments for
> copilot/reviewers to catch long running readers." — #1206, @harsha-simhadri

### validate-at-construction `MAJOR`

**Validate at construction, not lazily at use.**
> "This really should have been a construction invariant on the pq-scratch" — #1097, @hildebrandmw

### fail-at-load-time `MAJOR`

**Fail at load/parse/match time, not mid-run.**
A benchmark that only discovers an unsupported ISA after starting has wasted the run.

> "Running this on a non-AVX-512 supported machine fails at runtime instead of at load or match
> time." — #1027, @hildebrandmw

> "If the generation is going to fail due to an invalid range… this should fail before creating the
> file for write" — #847, @arkrishn94

### checked-conversions `MAJOR`

**Prefer checked conversions to `as`.**
Silent truncation is a recurring source of ID-space bugs.

> "Made the conversion fallible, thanks for flagging." — #1145, @hildebrandmw

### bind-types-explicitly `NIT`

**Bind types explicitly to catch future contract changes.**
> "if you type the return value like `let _: bool = ...` then you will get a compiler error in the
> future if the return type for `delete_iid` ever changes" — #1130, @hildebrandmw

---

## 3. Unsafe code

### safety-comment-required `BLOCK`

**Every `unsafe` block carries a `// SAFETY:` comment.**
Enforced by the workspace clippy lint `undocumented_unsafe_blocks`. The comment must name the
invariant that makes the operation sound, in the same units as the operation (element counts vs.
byte offsets).

House style is short and precise:
```rust
// SAFETY: We've already checked `states[i]`.
unsafe { states.get_unchecked_mut(i) }.occlude_factor = f32::MAX;
```

> "Rewrote the SAFETY comment to refer directly to `px_u32.add(i)` and use `remainder` as a lane
> count, dropping the broken sentence." — #1045, @hildebrandmw

### prefer-safe-alternative `MAJOR`

**Look for the safe alternative first.**
> "You can probably do `BlockTransposeRef::new(v).as_slice()` to avoid the unsafe here." — #1267,
> @hildebrandmw

> "You identified the reason this works yourself: by construction `i` is in-bounds." — #1133,
> @hildebrandmw

Prefer centralizing on an existing proven implementation (e.g. `diskann-vector`) over a new local
unsafe block.

---

## 4. Performance

### perf-claims-need-numbers `BLOCK`

**Performance claims require numbers.**
State dataset, hardware, and parameters. A measured non-result is an acceptable answer; an
unmeasured claim is not.

> "Measured this before acting: the test runs in **0.21s**… keeping as-is." — #1215, @JordanMaples
>
> "To be honest, all of this needs dedicated benchmarking, though, to figure out what the right thing
> is…" — #1206, @hildebrandmw

Maintainers also ask for end-to-end recall validation on real datasets for algorithmic changes:

> "Could you please run on a few medium size datasets and check recall before merging." — #1010,
> @harsha-simhadri

### watch-monomorphization `MAJOR`

**Watch monomorphization and code bloat.**
Generic explosion is treated as a real cost, not a theoretical one. CI has an LLVM IR bloat
regression check (#1083).

> "it very carefully minimizes monomorphizations to help keep compile times under control." — #1011,
> @hildebrandmw

### no-large-stack-allocations `MAJOR`

**No large or unbounded stack allocations.**
> "This is a *huge* stack allocation. Is this showing up in profiles? We already create auxiliary
> accessors for neighbor accesses - can't we use a `Vec` inside those?" — #1106, @hildebrandmw

### amortize-allocations `BLOCK`

**Amortize allocations in hot paths.**
Reuse scratch buffers, pool them, and prefer contiguous `Matrix`-style storage for locality.

> "Maybe use a `Matrix` to store the residuals? This will provide better locality in the processing
> loop and cut down on the number of allocations." — #1011, @hildebrandmw

> "Do we know if this will cause a large allocation if batch_size is large (~1M)?" — #1097, @arkrishn94

### no-fn-pointers-in-hot-loops `MAJOR`

**Function pointers defeat inlining in hot loops.**
> "A function pointer is (usually) not inlineable, so the resulting iterator has a large number of
> undesirable properties, including an indirect call for each evaluation." — #1216, @hildebrandmw

### benchmark-measures-what-it-claims `BLOCK`

**A benchmark must measure what it claims to measure.**
> "This runs all the searches single-threaded, but the loop nest loops over the number of threads.
> Not only will this make the benchmark take forever to run, it's also misleading…" — #1011,
> @hildebrandmw

---

## 5. Error handling

### error-strategy-by-tier `BLOCK`

**Match the error strategy to the crate tier.**
Per [AGENTS.md](../../../AGENTS.md):

| Tier | Strategy |
|---|---|
| Low-level (`diskann-wide`, `-vector`, `-linalg`, `-quantization`) | Bespoke, precise, **non-allocating** types via `thiserror`; chain with `Error::source`. `ANNError` is *not* suitable here. |
| Mid-level (`diskann` algorithms) | `diskann::ANNError` + its context machinery; `#[track_caller]` on conversions. Unrecoverable errors only. |
| High-level (tools, benchmarks) | `anyhow::Error`. |

### no-catch-all-error-enum `BLOCK`

**No single crate-level catch-all error enum.**
Explicitly listed under "Do Not" in AGENTS.md. Reasons given: no per-function failure documentation,
worse messages than bespoke types, large structs that blow up the stack, and branch-heavy `Drop`
impls that bloat code.

### preserve-error-semantics `MAJOR`

**Preserve error semantics — don't collapse distinct failures.**
> "Right now this forces every determinant-diversity failure into `DimensionMismatchError`, which may
> hide other classes of failures." — #1011, @suri-kumkaran

### centralize-error-conversions `MAJOR`

**Centralize conversions; use `?` over scattered `map_err`.**
Implement `From<E> for ANNError` (the `convert_error!` macro exists for this) rather than repeating
`map_err` at call sites.

> "Could we add `impl From<DeterminantError> for ANNError` (ideally with variant-specific
> `ANNErrorKind` mapping) and use `?` here instead of `map_err(...)`?" — #1011, @suri-kumkaran

### to-ranked-for-noncritical `MAJOR`

**Use `ToRanked` when non-critical failures must be distinguishable.**
Traits with associated error types should consider `diskann::error::ToRanked` instead of
`Into<ANNError>` so callers can decide to suppress vs. escalate transient failures.

> "Don't we already have a means of communicating transient failure with the transient error types?
> Why push down the logic and invent this new machinery…" — #1106, @hildebrandmw

### no-unwrap-in-library `BLOCK`

**No `unwrap` / `expect` / `panic!` in library code.**
`diskann-vector`, `diskann-providers`, and `diskann-tools` opt in at the crate root:

```rust
#![cfg_attr(not(test), warn(clippy::panic, clippy::unwrap_used, clippy::expect_used))]
```

These are `warn`-level, but CI runs clippy with `-Dwarnings`, so they block there. In crates without
the attribute, treat it as **[MAJOR]** rather than blocking. Panics are for invariant violations
(bugs), not for bad input — reviewers will point at existing constants or fallible helpers to avoid
an `unwrap`.

### no-needless-result `NIT`

**Don't wrap in `Result` when nothing can fail.**
> "The original code was unnecessarily wrapping the intermediate value in a Result but didn't call any
> fallible methods" — #1284, @hildebrandmw

### preserve-error-recovery-payload `MAJOR`

**Don't drop an error type's recovery payload.**
Some error types deliberately carry the failed input back to the caller so a large allocation isn't
lost on a size mismatch. Replacing such a type with a simpler one is a silent capability regression —
check what the *old* error carried before accepting a "simplification".

> "This breaks what the original `TryFromError` was trying to achieve. If you pass a large `Box<[T]>`
> to something and happen to get the sizes wrong, `TryFromError` will allow you to retrieve the
> original `Box<[T]>` without deallocating. Dropping this seems like a step back in library design."
> — #1269, @hildebrandmw

Ask of any error-type replacement: what could the caller do with the old error that it can't do with
the new one?

---

## 6. Testing

### patch-coverage `BLOCK`

**Patch coverage ≥ 90% — read the Codecov comment, don't infer it.**
`.codecov.yml` sets `patch.default.target: 90%` with `threshold: 0%` and `informational: false`, so
it blocks. Project coverage is informational only. `tests/`, `benches/`, `examples/`, and
`test_data/` are excluded from measurement.

**Always fetch the actual Codecov comment before making any coverage claim:**

```powershell
gh api "repos/microsoft/DiskANN/issues/<PR>/comments" --paginate |
  ConvertFrom-Json | Where-Object { $_.user.login -like '*codecov*' } | ForEach-Object { $_.body }
```

Counting `#[test]` attributes or `assert!` occurrences in the diff is **not** a coverage measurement
and must never be reported as one. #1269 is the cautionary case: the diff showed 68 → 29 tests and
450 → 147 assertions, which reads like a catastrophe, while Codecov reported **96.18% patch coverage
and project coverage up +0.84%** — because the PR deleted a duplicate implementation whose lines were
poorly covered. Test-count deltas are a prompt to go look; they are not a finding.

What test-count deltas *do* legitimately surface is loss of **behavioural** tests — panic conditions,
ordering guarantees, zero/edge dimensions — which line coverage cannot see. Frame those as the
finding, and name the specific removed test and the specific behaviour now unasserted.

### baseline-tests-for-algorithms `BLOCK`

**Algorithm changes use the baseline system, not eyeball assertions.**
Use `get_or_save_test_results` + `assert_eq_verbose!` (see `diskann/src/test/cache.rs` and
`diskann/src/test/cmp.rs`; regenerate with `DISKANN_TEST=overwrite`).

> "These tests still use the older style 'looks kind of okay' approach to tests rather than using the
> more rigorous baseline tests… This makes it significantly harder to refactor with confidence." —
> #1131, @hildebrandmw

### baselines-plus-invariants `MAJOR`

**Pair baselines with invariant checks.**
A baseline alone can be regenerated wrong and checked in. Assert structural properties too.

> "We cannot rely solely on the baseline to protect against regression (someone could check-in a
> broken baseline in the future), but a baseline in combination with some invariant checks … will go
> a long way toward good tests." — #928, @hildebrandmw

> "Could we also check that none of the results are out of bounds (id < max_inserted_points or
> something like that)." — #1158, @harsha-simhadri

### test-exercises-new-path `BLOCK`

**The test must actually exercise the new code path.**
Coverage percentage is not proof. Verify the branch/heuristic under test is reached.

> "even though a filter with a low selection percent are used, the filter is selecting for low IDs…
> To actually test the algorithm, the initial exploration needs to see enough matched nodes before
> the decision point." — #1131, @hildebrandmw

### concurrency-stress-depth `MAJOR`

**Concurrency tests need enough threads and iterations.**
> "Could we increase the number of threads and iterations over which this test is run. I am concerned
> that a single run with 8 threads might not be enough to surface non-determinism reliably." — #1158,
> @harsha-simhadri

### test-unhappy-and-legacy-paths `MAJOR`

**Test unhappy paths and legacy/compat paths.**
> "Many of the built-in impls are not tested for round trippability, and this `load_legacy` path is
> completely uncovered. Please add tests for the unhappy paths as well." — #1188, @hildebrandmw

### smoke-test-new-public-api `MAJOR`

**New public methods get at least a smoke test.**
> "Do you mind adding a simple test that calls this method to slightly decrease the chance of
> accidental removal?" — #1240, @hildebrandmw

### no-deleting-tests `BLOCK`

**Don't delete tests without justification.**
From [.github/copilot-instructions.md](../../copilot-instructions.md): "Unit tests must not be
removed without a stated, strong reason."

### testing-non-goals `NOTE`

**Explicit non-goals.**
Per AGENTS.md — **do not** request tests for derived traits (`Clone`, `Debug`, `PartialEq`) or for
enums without explicit functionality. And duplication inside unit tests is acceptable:

> "I generally agree with DRY… However, in unit tests, it can make tests harder to read… For small,
> simple tests like these, it's fine to allow some code duplication" — #847, @arrayka

### no-dead-code-in-tests `MAJOR`

**`#[allow(dead_code)]` in tests is a smell.**
Either the helper is used (write the test) or it isn't (delete it).

> "Why are we allowing `dead_code`?" — #983, @hildebrandmw

---

## 7. Documentation & naming

### docs-never-restate-signature `MAJOR`

**Less is more — never restate the signature.**
> "I experimented with stripping down the documentation, particularly when it simply restates the
> type-signature, and found the result much easier to understand and review… Terser and more direct
> documentation means the reader spends less time filtering and more time understanding." — #1027,
> @hildebrandmw

Per AGENTS.md: don't maintain explicit lists of types/functions in module docs; don't restate what
the signature shows. Do document non-obvious behavior, errors, safety, and design intent using
`# Errors`, `# Safety`, `# Panics`, `# Example`.

### docs-not-contrastive `MAJOR`

**Document what the code does, not how it differs from alternatives.**
Contrastive docs rot as soon as the other implementation changes.

> "In general, please keep doc comments to what the kernel does and not as a contrast to what another
> kernel does. The former is more durable long-term." — #1045, @hildebrandmw

### backtick-identifiers `NIT`

**Wrap identifiers in backticks in prose.**
> "Please wrap variable identifiers in backticks." — #1069, @hildebrandmw

### names-match-vocabulary `MAJOR`

**Names should match the library's existing vocabulary.**
> "The general vocabulary of this library would call this `NeighborAccessor`." — #1106, @hildebrandmw

> "`impl glue::FilteredAccessor for FilteredAccessor` is confusing. What about naming the struct
> `LabelFilteredAccessor`…?" — #1141, @arrayka

### valid-intra-doc-links `NIT`

**Keep intra-doc links valid.**
> "Run `cargo rustdoc --package diskann-bf_tree -- -D rustdoc::broken-intra-doc-links` to help find
> outdated intra-doc links." — #1020, @hildebrandmw

---

## 8. PR hygiene & process

### pr-description-explains-why `MAJOR`

**The PR description must explain *why*, and leave breadcrumbs.**
> "Can you please add some info to the PR description describing why this feasible to do now when it
> wasn't in the past to provide some breadcrumbs in case this is ever needed again?" — #1254,
> @hildebrandmw

Also complete the checklist in [PULL_REQUEST_TEMPLATE.md](../../PULL_REQUEST_TEMPLATE.md): release-
note-worthy title, new dependencies, API modifications, backward compatibility, docs impact.

### rfc-for-significant-change `MAJOR`

**RFC required for architecturally significant change.**
Per [rfcs/README.md](../../../rfcs/README.md): new crates, cross-crate traits/abstractions, new
distance functions / storage layouts / index formats, and backward-compat-affecting changes. The PR
*is* the RFC; merging is acceptance; tag with the `RFC` label; filename is the zero-padded PR number.
Not required for single-crate API additions, bug fixes, internal refactors, or docs.

### new-dependencies-justified `BLOCK`

**New dependencies need justification.**
From [.github/copilot-instructions.md](../../copilot-instructions.md) — plus watch for transitive
bloat and build-time impact.

> "Doesn't that also pull in a whole new crate? Right now, we only depend on `anyhow`. Do you think
> it's worth adding?" — #1188, @suhasjs
>
> "Don't pull `rayon` as a dependency of `diskann`." — #1024, @hildebrandmw

### gate-experimental-work `MAJOR`

**Gate experimental work behind a feature flag.**
A recurring unblocking pattern: ship early behind `experimental`, gate the new dependencies, and
keep default builds fast.

> "To unblock algorithmic work, what if we do the following: 1. Put this behind an 'experimental'
> feature flag and in an 'experimental' module… 2. Gate the new dependencies…" — #1099, @hildebrandmw

### no-silent-config-defaults `MAJOR`

**No silent defaults in benchmark/run configuration.**
> "I recommend not using `default` for these… The benchmark runs are meant to record all the
> information we can about a run." — #1106, @hildebrandmw

### license-header `BLOCK`

**License header on every new file.**
```rust
/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
```

### verify-removals-against-usage `MAJOR`

**Verify removals against internal/test usage.**
> "These are still being referenced in some internal tests… please double check they are indeed safe
> to remove" — #1185, @hildebrandmw

### no-build-time-regression `MAJOR`

**Don't regress build times.**
Called out directly in [.github/copilot-instructions.md](../../copilot-instructions.md).
Feature-gate heavy code paths and watch generic instantiation.

### cross-platform-validation `BLOCK`

**Cross-platform validation for arch-specific code.**
Per AGENTS.md and `diskann-wide/README.md`: AVX-512 under Intel SDE, AArch64 under QEMU, and a
baseline `x86-64` run to confirm no unsupported instructions are emitted.

---

## 9. Interaction norms

### read-existing-review-threads `BLOCK`

**Read the existing review threads before proposing structural changes.**
On any PR with prior review activity, fetch the open threads first. Structural and naming decisions
are frequently negotiated in-thread, and a review that re-litigates a settled decision costs the
author time and undermines the reviewer's credibility.

```powershell
gh api "repos/microsoft/DiskANN/pulls/<PR>/comments?per_page=100" --paginate
gh api "repos/microsoft/DiskANN/pulls/<PR>/reviews?per_page=100"  --paginate
gh api "repos/microsoft/DiskANN/issues/<PR>/comments?per_page=100" --paginate
```

Filter out `*[bot]`, `Copilot`, and `codecov-commenter` for the human signal — but read the Codecov
body itself (rule `patch-coverage`) and skim bot findings for genuine correctness bugs before
discarding them.

#1269 is the cautionary case. The module is named `views` while the PR description says `matrix`,
which looks like an obvious naming defect — but @hildebrandmw had explicitly asked for it: *"is it
possible to keep the new matrix in `views` for one PR and then rename the module in a follow-up. Will
help with auditing across all the files."* Recommending the rename would have contradicted the lead
reviewer's deliberate call. The correct finding there was the **stale PR description**, not the module
name.

Corollary: when a discrepancy between description and code has an innocent explanation, the reviewable
defect is usually the stale description (rule `pr-description-explains-why`), not the code.

### review-norms `NOTE`

**Interaction norms.**

- `nit:` means explicitly non-blocking. Reviewers approve PRs with open nits.
- Reviewers propose concrete fixes — full trait sketches, links to existing helpers, exact commands.
  Do the same.
- Disagreement is reasoned and normal. Maintainers reverse their own positions when given context
  ("I take back the suggestion to remove the ground-truth"). Authors are expected to push back with
  data.
- Deferring work to a named follow-up PR is accepted for non-critical issues; say so explicitly.
- Reviewers state their stake and priority openly ("I have a vested interest in seeing something like
  this merged ASAP. That said, there is a lot of work needed…"). Transparency about trade-offs beats
  false neutrality.
- Praise for good design choices is normal and expected.
- Large PRs get proportionally harder architectural pushback. Expect it, and split when you can.
