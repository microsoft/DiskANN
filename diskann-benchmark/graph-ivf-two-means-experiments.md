# Graph-IVF block-transposed two-means

The active implementations are paired_point (default), transposed_l2_16 and
transposed_dot_16. The earlier separate, packed_paired, batched_4/8, SGEMM and
center_difference experiments have been removed from the active Rust and CLI
paths. Their frozen executables and original results are historical evidence.

## Code boundaries

- diskann-quantization/src/algorithms/kmeans/two_means.rs: direct packing from
  borrowed rows into BlockTransposed<f32,16>, shape/overflow validation, and
  four f32x8 accumulators (16 points, two centers, U=1). Outputs are 0/1 or INVALID.
- diskann/src/ivf/clustering/strategy.rs: one preparation per parent; assignment
  scratch reused across steps; nonfinite scores fall back to paired L2.
  Initialization, ordered f64 sums, fixed steps and empty-center retention use
  the shared fitter in clustering/two_means.rs. Final region-wide reassignment
  is unchanged.
- diskann-benchmark/src/graph_ivf/two_means/: complete-fit timing, per-iteration
  diagnostics and independent parent capture/replay. No benchmark CLI, tracing
  or timing code is added to the new low-level primitive.

The library feature is ivf-experimental; benchmark feature ivf-two-means enables
it. The new optional quantization dependency disables its default features.
The experiment does not switch the default IVF strategy.

## Numerical contract

Both transposed candidates change floating-point reduction order. L2 uses direct
squared differences. Dot compares norm(c)-dot(x,c)-dot(x,c) and omits the shared
point norm. Finite cancellation may change decisions; overflow fallback does
not fix it. Tests retain the x=10000.25, c0=10001, c1=10000 counterexample:
true squared distances are 0.5625/0.0625, but dot scores tie and choose center 0.
No epsilon hides this discrepancy.

Every valid score tie chooses center 0. Every nonfinite score triggers a paired
recalculation of that point; padding does not become an assignment. Finite input,
RNG consumption, iterations.max(1), original point order, f64 sums and empty
centers retain the existing contracts. No NaN input ordering is introduced.

transposed_lloyd_reference wraps the existing distances_in_place and computes
point norms and residuals. It is only a complete-fit reference, unavailable to
index build/search. Bad IDs, nonfinite or negative residuals are recorded as
failed/not applicable; their samples must not count as successful speedups.

## Measurements

Complete-fit schema version 2 includes validation, packing, norm preparation,
scratch allocation, seeding, all assignments/sums/updates, and scratch/packing
release. Returned centers stay alive. Data loading, FP16 conversion and synthetic
data generation are outside the timer. packing/preparation is included in total.
packed_bytes includes padded rows; scratch_bytes includes f64 sums, assignments,
and reference point norms, but excludes the returned center buffer.

Diagnostics are separate deterministic fits after the timed rounds. They compare
assignments under the baseline centers at every step and the two independent
trajectories; first divergence includes row/point ID, paired distances and an
independent f64 distance/margin. The final trace must match the timed fit.

The graph-ivf-two-means input accepts strategies and order (rotating or abba).
Rotating defaults to all four implemented fitting paths; repetitions are rounds.
ABBA requires exactly [paired_point, candidate]; repetitions and warmups count
complete four-position blocks. Raw samples and explicit invocation order retain
all values. A sample near the boundary is not evidence of universal bit equality.

The separate graph-ivf-parent-replay input performs a diagnostic baseline build,
exports predetermined zero-based parent ordinals with row IDs, and replays the
same parents for every fitting strategy and both layouts. Missing ordinals are
reported. It uses a fresh fixed fit seed, not the original build RNG trajectory.
Export build timings are discarded. Snapshot coordinates and IDs are retained.

## Current comparison protocol

The new experiment is target/ivf-transposed-20261022. A is paired_point in the
same newly frozen executable as both B candidates. One full ABBA block per
candidate on YFCC, Enron and fbv8_mimir gives 24 full build+search invocations.
All queries (100 / 1000 / 6380) are used at nprobe 4/16/64 with one full warmup
and one measured pass. Seed7, five Lloyd steps, batch4096, split2048, initial16,
graph degree16/L64, k10/L128, one thread, affinity0x10 and Normal priority are fixed.

The CLI runner accepts --reference and repeated --candidate. It checks executed
settings, complete rows/query counts, affinity/priority and frozen binary/input
hashes. Resume only reuses verified completed artifacts and refuses to overwrite
incomplete output. Each B uses its own surrounding A runs. A single ABBA block
has no confidence interval. Slow samples and changed quality/layout are retained.

The static provider is in memory with exhaustive build routing and a centroid
graph constructed afterwards. Logical scan counts are not physical SSD I/O or
Mimir integration results. Historic 288-run evidence remains paused and separate.

See graph-ivf-transposed-two-means-results.md for measured outcomes and provenance.
