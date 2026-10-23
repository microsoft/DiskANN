# Graph-IVF exact two-means experiment

The active strategies are paired_point (default) and tiled_exact. Earlier
transposed L2/dot and controlled Lloyd reference paths are removed. Their frozen
executables, reports and raw measurements remain historical evidence.

## Implementation and numerical contract

The vector primitive in diskann-vector/src/distance/paired/tiled.rs joins two
independent eight-coordinate vectors into the halves of a sixteen-lane register.
Each point retains the baseline's four accumulation chains, FMA, combine order,
partial tail and eight-lane reduction. This does not use a sixteen-lane sum.

A V3/V4-compiled binary selects the wide implementation once per parent when the
host supports V4. Scalar builds, AArch64 and V3-only hosts retain paired L2.
There is no automatic promotion of scalar arithmetic to FMA. Equality refers
to the compiled baseline on that configuration, not different compiled ISAs.

The IVF adapter borrows rows without packing, assigns pairs and sums points in
original order. Odd last rows use paired. Seeding, fixed Lloyd steps, f64 sums,
empty centers and final region-wide reassignment use the shared implementation.
No epsilon or numerical fallback relaxes exactness.

## Measurement

Complete-fit schema version 3 includes validation, ISA selection, allocation,
seeding, all distance/selection/sum/update work and internal scratch release.
Returned centers remain alive. Data generation/loading and FP16 conversion are
outside the timer. packed_bytes and strategy scratch_bytes are zero; reported
fit scratch includes the two f64 sum rows.

Separate deterministic fits compare every step at the same centers and along
each strategy's own trajectory, final centers, assignments, sum bits and RNG.
Any mismatch fails the benchmark. Tests also compare real ordered list members
and centroid bits; fingerprints alone are diagnostic evidence, not a proof.

The ABBA runner defaults to tiled_exact. Historic strategy names are read as
protocol data without reinterpreting saved results. Every candidate uses its own
surrounding A samples; all slow samples remain. The analyzer rejects a changed
index/quality/work fingerprint for an exact candidate.

## Quick verification on 2026-10-23

See [the revised plan](graph-ivf-transposed-two-means-plan.md) for the fixed protocol:
18 complete-fit shapes, one warmup and four measured ABBA blocks; three complete
dataset builds, one build+search ABBA per dataset. YFCC uses all 100 queries;
Enron and FBV8 use the first 32 original queries and corresponding groundtruth
rows for this explicitly scoped quick check. Search uses k10/L128/nprobe4/16/64.
The original full-query experiment and paused automation are not resumed.

Final artifacts are in target/ivf-tiled-exact-20261023-final. An initial exact
tile had an uninlined-closure code-generation defect; its fit samples are
retained separately in target/ivf-tiled-exact-20261023 and are not pooled. The static provider is in
memory with exhaustive build routing and a centroid graph built afterwards.
Logical scan counts do not represent physical SSD I/O or Mimir performance.
One build/search ABBA is preliminary evidence, not a stable speedup claim.

See [the measured results](graph-ivf-tiled-exact-results-20261023.md) for the
exactness checks, retained samples, build gains and unresolved search slowdown.
