# Opt-in stable start-point cores during online insertion

| | |
|---|---|
| **Authors** | Zhi Yang (@yangzhi-tech) |
| **Created** | 2026-11-01 |
| **Updated** | 2026-11-01 |

## Summary

Introduce opt-in stable-core search for configured synthetic graph start points.
Use a dedicated pruning alpha of 1.4 when a start-point adjacency overflows, and
let queries expand only its first `R` stored neighbors, where `R` is the configured
pruned degree. Insertion continues to traverse the complete adjacency, including
appended backedges beyond that prefix. The feature is disabled by default.

This RFC requests agreement on the algorithm and provider extension points before
an implementation is contributed. It changes no executable code. Measurements
below come from an existing implementation on an earlier codebase, not from the
current upstream revision.

## Motivation

### Background

Online insertion adds outgoing edges for new documents and backedges on existing
nodes. Adjacencies grow until the normal overflow threshold triggers pruning.
Queries can follow different paths after these mutations even when newly inserted
documents are not closer results. This is expected for an evolving approximate
index, but frequent start-point changes can amplify that instability.

Synthetic start points are high-impact sources because searches expand them near
the beginning of traversal. In the baseline experiment, the start-point degree
ranged from roughly 4 to 84 with `R = 64`. Stabilizing this entry view is an
opportunity to reduce avoidable result churn without freezing the graph.

### Problem Statement

Repeated Top-20 queries during concurrent insertion changed their member sets in
6.100-7.685% of adjacent-round comparisons at the measured 20M-60M checkpoints.
The stable-core implementation reduced these rates to 0.773-0.891%.

This is not a request for deterministic search or snapshot isolation. Mutations
of ordinary traversal nodes, start-point pruning, deletes, and legitimately better
inserted documents can still change results.

### Goals

1. Reduce repeated-query result churn attributable to start-point mutations.
2. Preserve graph construction's access to appended backedges.
3. Keep the existing overflow threshold and avoid an admission prune on every insert.
4. Preserve default behavior and make unsupported opt-in operations fail explicitly.
5. Validate stability together with recall and insertion/search latency.

## Proposal

### Graph configuration and algorithm

Add `Builder::stabilize_start_point_adjacency(bool)` and a matching `Config`
accessor, defaulting to `false`. Keep the dedicated alpha of 1.4 internal rather
than introducing another tuning parameter.

When enabled:

1. Append backedges normally, preserving adjacency order.
2. For query expansion of a configured synthetic start point, use the stored
   prefix `adjacency[..min(R, adjacency.len())]`.
3. On the existing overflow condition, use RobustPrune with alpha 1.4 and degree
   `R` for a synthetic start-point source. The result becomes the new prefix.
4. For insertion/build traversal, retain the full adjacency.
5. For ordinary sources, retain the configured graph-wide alpha and traversal.

The prefix is selected **before** visited filtering or label classification.
Otherwise a query could refill the prefix from the tail, defeating stabilization.
Providers must preserve stored adjacency order; neither distance-sorting each
query nor selecting the first `R` accepted results implements this proposal.

An alpha-1.4 prune is not guaranteed to return `R` neighbors. An underfilled
prefix can grow as later backedges arrive. Even a full prefix can change at the
next physical prune. This is a bounded query view, not an immutable cached core.

Apply the dedicated pruning alpha to the common backedge-overflow path used by
single insertion, multi-insertion, and inplace-delete edge repair. Keep the
existing delete-repair and consolidation triggers; do not introduce immediate
start-point re-pruning after every deletion.

### Provider API direction

The current `SearchAccessor::expand_beam` owns adjacency retrieval and distance
computation. `FilteredAccessor` also owns classification. The core cannot enforce
a stored prefix using the visited predicate: that predicate has no source or
adjacency-position information and may be evaluated multiple times.

The proposed direction is an additive, fallible configuration hook on both
accessor traits, plus a pruning-source classification hook. The signatures below
are sketches for discussion, not implemented APIs:

```rust,ignore
// On SearchAccessor and FilteredAccessor:
fn set_start_point_neighbor_limit(
    &mut self,
    limit: Option<std::num::NonZeroUsize>,
) -> ANNResult<()>;

// On PruneAccessor:
fn is_synthetic_start_point(
    &self,
    id: Self::Id,
) -> impl std::future::Future<Output = ANNResult<bool>> + Send;
```

Provide default implementations:

- Setting no limit succeeds and leaves existing expansion unchanged.
- Setting a limit returns an explicit unsupported-operation error unless the
  provider implements the behavior.
- Synthetic start-point classification returns an explicit unsupported-operation
  error unless implemented. Call it only when stabilization is enabled and a
  physical overflow prune is required.

Existing implementations therefore continue to compile and behave as before
with the default configuration. Enabling the feature must not silently fall back
to unstabilized traversal or graph-wide pruning alpha.

Query entry points configure the limit once on their fresh accessor, using `R`.
Insertion accessors are not configured with a limit. Provider adapters, including
the label-filtering wrapper and beta-filtering layer, must forward configuration
to their inner accessors. Errors propagate through the normal `ANNResult` path.

Providers apply the limit only to their configured synthetic main-graph start
points. Arbitrary query seed IDs and tenant medoids must not become synthetic
start points merely because a search starts from them. Pruning classification
must use the same provider metadata, not assumptions about ID ranges. This also
avoids coupling the feature to the start-point ID placement discussed in PR #1528.

Initial implementations should cover the test provider, the in-memory provider
accessors in `diskann-providers`, and `diskann-inmem`. Other backends remain
unchanged by default and report unsupported opt-in use until implemented.
An internal prune option can override alpha for the individual overflow prune
without changing alpha for ordinary candidates or nodes.

### Search surfaces and lifecycle

The initial implementation should support standard and recorded k-NN, paged
k-NN, and inline filtered search with and without adaptive L. Paged traversal
keeps its accessor configuration across pages; it does not gain snapshot
semantics across concurrent mutations.

Range, filtered-range, diverse, and multihop traversal require separate review
before opting into the prefix contract. Until supported, their entry points
should reject an enabled stabilization configuration rather than imply that all
query modes are stabilized. Custom graph/tenant entry points require their own
design and evidence.

For order-preserving storage formats, no new serialized adjacency field is
necessary. Enabling the runtime option on a loaded legacy graph immediately uses
the first `R` stored neighbors, without re-pruning at load time. The dedicated
alpha takes effect on the next normal overflow prune. Binary compatibility and
provider-specific load behavior must be tested rather than assumed universally.

### Implementation validation

The follow-up implementation should include:

- Exact prefix selection before visited filtering, including already-visited
  prefix members, underfilled lists, multiple synthetic start points, and mixed
  beams containing ordinary sources.
- Tail-only appends leaving a full query-visible prefix unchanged while the
  insertion accessor still traverses the tail.
- Dedicated alpha on single/multi-insert and inplace-delete overflow paths;
  ordinary-node and disabled behavior unchanged.
- Propagation through labeled and beta-filtered adapters, including accepted
  and rejected prefix neighbors without tail substitution.
- Recorded/paged parity, explicit query seeds, and supported filtered traversal.
- Explicit failures for unsupported providers and query modes.
- Load/enable/next-overflow behavior on existing order-preserving indexes.
- Public or synthetic end-to-end stability/recall benchmarks on current upstream.

The upstream format, clippy, tests, and coverage checks still apply to that
implementation; this documentation-only RFC does not claim to satisfy them for
code that has not been ported.

## Trade-offs

- **Higher start-point alpha only:** may improve retained degree, but appended
  tail neighbors still change the query entry view. It does not isolate a core.
- **Prefix only:** bounds the query view, but a very short pruned adjacency can
  refill frequently. The dedicated alpha is intended to retain a fuller core.
- **Freeze the start-point adjacency permanently:** stronger local stability,
  but prevents it from adapting as the graph evolves.
- **Admission pruning on each incoming backedge:** can control core membership,
  but adds pruning work on the hot insertion path. This proposal only changes
  pruning that the existing lifecycle already requires.
- **An expansion-options argument:** alternatively add per-source limits to
  `expand_beam` and filtered equivalents. This makes policy explicit at each
  call, but changing existing signatures affects all implementors. An additive
  options-aware method is also possible; maintainers may prefer it to mutable
  accessor configuration.
- **Provider-specific optimization only:** avoids extending shared traits but
  makes query/build separation and cross-provider behavior less consistent.

The additive hooks introduce accessor state and explicit unsupported-operation
errors for opt-in users. The exact API and initial provider coverage are open
design questions. No performance guarantee is made for unmeasured workloads.

## Benchmark Results

### Method and limitations

The reported comparison uses two independently built graphs, one baseline and
one stable-core graph, each grown from scratch through 64 insertion shards to
**63.94M vectors**. Despite the experiment's "70M" shorthand, the measured build
did not reach 70M. These are not repeated trials or confidence intervals.

At 20M, 30M, 40M, 50M, and 60M, two query workers ran 200 rounds concurrently
with two insertion workers. Stability compares adjacent rounds for 4,010 unique
queries. A change means a different ordered Top-20 list or, separately, a
different Top-20 member set.

Latency summarizes the checkpoint's 1M-vector insertion shard and pools 4,058
client query rows across 200 rounds (811,600 rows per solution per checkpoint).
The difference between 4,058 latency rows and 4,010 unique stability queries has
not yet been reconciled; do not treat these as identical sample populations.

Snapshot quality uses the same 4,010 unique queries after every shard, Top-20,
and search `L = 300`. Exact reference neighbors were computed by scanning
cumulative insertion shards using squared L2. Distance-threshold Recall@20
counts a returned result when its distance is no greater than the exact
20th-neighbor distance; this is tie-tolerant, not ID-intersection recall.
Distance regret is the percentage excess of the ANN Top-20 squared-L2 distance
sum over the exact sum. Both compared configurations use `R = 64`.

Hardware, complete dataset provenance, large-run vector representation, seeds,
full build settings, raw per-query records, and a public reproduction command
are not supplied here. These aggregate results motivate evaluation; they are
not a reproducible upstream benchmark or proof of general optimality.
Ordinary-node mutations and legitimate result improvements also contribute to
the measured change rate; the metric does not classify every transition's cause.

### Repeated-query stability

| Vectors | Ordered change: baseline | Stable core | Member-set change: baseline | Stable core |
|---:|---:|---:|---:|---:|
| 20M | 10.100% | 1.347% | 6.100% | 0.796% |
| 30M | 11.217% | 1.349% | 7.310% | 0.773% |
| 40M | 11.065% | 1.466% | 6.939% | 0.825% |
| 50M | 12.166% | 1.674% | 7.583% | 0.884% |
| 60M | 12.664% | 1.694% | 7.685% | 0.891% |

Member-set change rates decreased by 86.95-89.43% relative to baseline across
these checkpoints. This does not imply deterministic results.

### Concurrent latency

Each cell is mean / P99 in milliseconds.

| Vectors | Insert: baseline | Stable core | Search: baseline | Stable core |
|---:|---:|---:|---:|---:|
| 20M | 1.958 / 3.011 | 2.009 / 3.112 | 1.893 / 3.133 | 1.905 / 2.985 |
| 30M | 2.108 / 3.308 | 2.127 / 3.383 | 2.164 / 3.659 | 2.150 / 3.517 |
| 40M | 2.137 / 3.417 | 2.184 / 3.546 | 2.249 / 3.814 | 2.273 / 3.746 |
| 50M | 2.105 / 3.425 | 2.177 / 3.609 | 2.260 / 3.898 | 2.330 / 3.930 |
| 60M | 2.094 / 3.449 | 2.136 / 3.561 | 2.268 / 3.823 | 2.307 / 3.806 |

Mean insertion latency increased by 0.90-3.42%; insertion P99 increased by
2.27-5.37%. Mean search latency changed by -0.65% to +3.10%; search P99 by
-4.72% to +0.82%. Thus "close to baseline" does not mean zero overhead.

### Snapshot graph quality

Values are means over 32 snapshots in each half, not independently replicated
experiment means.

| Vectors | Recall@20: baseline | Stable core | Distance regret: baseline | Stable core |
|---|---:|---:|---:|---:|
| 1M-32M | 93.980% | 95.059% | 0.726% | 0.504% |
| 33M-63.94M | 92.349% | 93.165% | 1.143% | 0.905% |

Recall improved by 1.079 and 0.816 percentage points on these aggregates.
Improvements were not uniform: averaged over the 59M and 60M snapshots, baseline
recall was 92.703% versus 92.403% for stable core. No claim is made that every
snapshot improves.

### Why start with alpha 1.4?

A separate screening experiment used one fresh 2M-vector build for each sampled
degree, with 256-dimensional int8 L2 data, build/search `L = 300`, and four
insert/search workers. Each build used a 1M seed and four 250K stages.
The late window is the final quarter of physical start-point prune events.

| R | Late prunes retaining exactly R | Late prunes changing core membership |
|---:|---:|---:|
| 32 | 100% | 1.40% |
| 48 | 100% | 2.78% |
| 64 | 99.73% | 3.33% |
| 96 | 99.46% | 2.25% |
| 128 | 97.93% | 3.62% |
| 160 | 96.96% | 4.79% |

Membership compares sets immediately before/after pruning, ignoring order;
percentages are per prune, not per query or insert. Late minimum retained degrees
for `R = 64, 96, 128, 160` were 62, 93, 124, and 156. This screening supports
1.4 as an initial fixed value, not universal saturation or optimality across
datasets, metrics, and degrees.

## Future Work

- [ ] Agree on accessor configuration versus per-expansion options.
- [ ] Port and validate the algorithm against current upstream.
- [ ] Publish a public/synthetic reproduction with full parameters and raw metrics.
- [ ] Reconcile the latency-row and unique-stability-query populations.
- [ ] Repeat builds across seeds, datasets, degrees, and distance metrics.
- [ ] Evaluate deletion-heavy workloads, serialized index loading, and other backends.
- [ ] Review additional traversal modes and custom tenant/medoid entry points.

## References

1. [Search-accessor RFC](01067-search-accessor.md).
2. [Filtered-search algorithms RFC](01128-filtered-algorithms.md).
3. [Paged-search RFC](01078-paged-search.md).
4. [PR #1528: Move start points to the beginning](https://github.com/microsoft/DiskANN/pull/1528).
5. [Current search/prune extension traits](../diskann/src/graph/glue.rs).
6. [Current graph implementation](../diskann/src/graph/index.rs).
