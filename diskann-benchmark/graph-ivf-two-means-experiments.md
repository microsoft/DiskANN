# Graph-IVF two-means experiments

The opt-in `ivf-two-means` benchmark evaluates complete Lloyd fits and the actual
IVF insertion path. The default library keeps paired L2. The experimental
batched and dot-product strategies change floating-point evaluation order;
they are not drop-in bitwise replacements.

## Code boundaries

`diskann/src/ivf/clustering/` owns splitting, the shared Lloyd state transitions,
parent preparation and the assignment kernels. `two_means.rs` retains the
default paired-L2 loop; `strategy.rs` prepares a parent once; `assignment.rs`
and `batched.rs` implement the optional numerical alternatives. `IVFIndex`
delegates to this module. Reusable paired-L2 primitives remain in
`diskann-vector/src/distance/`.

The library's opt-in feature is `ivf-experimental`. It exposes
`clustering::{Strategy, TwoMeans}` and the index strategy selector, without CLI
names, timers or benchmark result types. The benchmark's `ivf-two-means` feature
enables it. JSON names and parsing live in `src/inputs/graph_ivf_two_means.rs`;
complete-fit timing and final diagnostics live in
`src/graph_ivf/two_means/measurement.rs`. Split-phase `InsertStats` timers remain
part of the existing index instrumentation.

The former hidden `ivf::two_means_benchmark` module and library feature
`ivf-two-means-benchmark` were removed in this draft. All seven JSON strategy
names, their order, and the paired default are unchanged.

## Implemented strategies and numerical contract

| `two_means_strategy` | Assignment computation | Packing | Contract |
| --- | --- | --- | --- |
| `separate` | Two existing SIMD squared-L2 calls per point | None | Original distance arithmetic |
| `paired_point` | Shared point loads, two squared-L2 accumulations | None | Same distance bits as separate, default |
| `packed_paired` | Paired L2 on a contiguous copy | Once per parent, reused for distances and sums | Same centers and ordered membership as paired |
| `batched_4` | Four points share two center loads | None | One accumulator per distance changes the reduction tree |
| `batched_8` | Eight points share two center loads | None | Same numerical caveat as batched_4 |
| `sgemm` | B-by-D points times transposed two-by-D centers, then center norms | Once per parent; B-by-2 output reused | Expanded norms and GEMM reductions can change assignments |
| `center_difference` | One dot product against the center difference | None | Expanded norm difference can lose deciding bits |

All strategies share seed selection, fixed Lloyd iteration count, ordered f64
point sums, center updates, and retention of an empty center. Zero iterations
still executes one step, matching the existing API. Equal computed scores prefer
center zero. A changed arithmetic expression can nevertheless turn a true tie
into unequal computed scores.

The batched kernel uses `diskann-wide` f32x8 operations, supports borrowed
noncontiguous rows, handles coordinate tails, and uses paired L2 for an incomplete
last group. It is a distinct reduction strategy; sharing center loads alone
does not guarantee bitwise equivalence.

SGEMM uses the existing `diskann-linalg` implementation. It computes
`score_j = norm(center_j)^2 - 2 * dot(point, center_j)`; point norms cancel when
ranking. Center difference compares
`norm(c1)^2 - norm(c0)^2 - 2 * dot(point, c1 - c0)` with zero.
Both fall back to paired L2 for nonfinite scores. This is **not** a finite-error
bound. The regression example `point=10000.25`, `c0=10000`, `c1=10001` shows that
finite cancellation can change the selected center. No unexplained epsilon
or quality tolerance hides that case.

## Measurement boundaries

`graph-ivf-two-means` times shape validation, scratch allocation, parent packing where needed,
seeding, every Lloyd assignment, selection, f64 accumulation, and center update.
The same packed parent is reused across iterations; no packing cost is moved
outside this interval. It reports raw sample times in execution order and
arithmetic means. Strategy order rotates and reverses across rounds.

The refactored harness reports `preparation_ns`: validation and coordinate
packing within the full timer. The former `packing_ns` included coordinate
packing and row-view creation, with shape validation outside the total timer.
Row views and kernel scratch remain inside the complete-fit timer. These phase
fields have different scopes; historical results are not renamed or pooled with
new measurements. The ongoing ABBA build/search run keeps its frozen executable
and original protocol, independent of source-file reorganization.

Final diagnostic assignments use paired L2 against the fitted centers, outside
the timer, matching the numerical distance used by final region reassignment.
These diagnostics compare final outputs, not every intermediate Lloyd decision.
The baseline-equivalent strategies must have equal center bits and assignments.
The other strategies report disagreements without relabeling them as exact.

The standalone matrix samples points without replacement from a file, or uses
synthetic uniform points. Contiguous and scattered layouts preserve values and
logical point order; scattered rows are allocated in shuffled order. These are
parent-sized samples, not captured real split parents. The full build runs
`IVFIndex::initialize` / `insert_batch`, actual splitting, final all-child
reassignment, provider publication, and final centroid-graph construction.

The full build checks that every input ID belongs to exactly one live list.
Same-executable fingerprints cover list IDs, center bits and ordered members.
Search records ordered result-ID fingerprints, recall, returned counts, and
logical vector/list/centroid work. Fingerprints are diagnostic hashes rather
than cryptographic equality proofs. The provider is in memory: these counts
do not establish unchanged SSD I/O or Mimir write amplification.

## Reproduce

```powershell
cargo build --release -p diskann-benchmark --features ivf-two-means --offline
./target/release/diskann-benchmark.exe run --input-file diskann-benchmark/perf_test_inputs/graph-ivf-two-means-strategies.json --output-file target/two-means-matrix.json
```

For a real corpus, set `data`, `data_type`, and `search_directories` in the matrix
input. `float16` values are promoted exactly to FP32 before timing, without
renormalization. The FBV8 inputs use 958,152 base rows and 6,380 query rows of
896 dimensions; their supplied files contain FP16 values, not 448d FP32.
The examples search `data/fbv8_mimir` relative to the process working directory.
Set `search_directories` to the actual corpus directory in a local copy of each
input. The corpus itself is not included in this repository.

To measure full builds, set `build.two_means_strategy` in the static input:

```powershell
./target/release/diskann-benchmark.exe run --input-file diskann-benchmark/perf_test_inputs/graph-ivf-fbv8-build.json --output-file target/fbv8-build.json
./target/release/diskann-benchmark.exe run --input-file diskann-benchmark/perf_test_inputs/graph-ivf-fbv8-search.json --output-file target/fbv8-search.json
```

Use unique output names and identical corpus, seed, iteration count, thread
count, CPU affinity, compiler flags and index/search settings. Compare full
two-means and total build time, rather than distance-only timings. Retain slow
samples and changes in partitioning, recall and scan work. A changed partition
can alter the subsequent amount of algorithm work and is a separate contract.

On Windows, the [single-process runner](scripts/run_ivf_experiment.py) pins the
benchmark before it starts, verifies the inherited affinity and Normal priority,
and stores a log plus `.metadata.json` beside the result. Metadata includes the
executable/input SHA256, process CPU time, wall time and exit code. It refuses
to overwrite existing artifacts. Mask `0x10` selects logical processor 4; use
a valid mask for the host. It does not control other applications or CPU clocks.

```powershell
python diskann-benchmark/scripts/run_ivf_experiment.py --exe target/release/diskann-benchmark.exe --input target/fbv8-build-local.json --output target/fbv8-paired-run-0.json --affinity 0x10
```

For the [FBV8 comparison](graph-ivf-two-means-fbv8.md), run one complete build
per process for each of the seven strategies, in the table's order. Repeat seven
rounds, rotating the first strategy to the end each round, so each strategy
occupies each position once. Preserve every sample, with no discarded build
warmup, and average the seven full fits/builds per strategy. The retained JSON
contains the exact order and inputs for the reported experiment. Run the
standalone matrix separately. Keep compilation and tests outside measurement
runs, and compare quality with all queries and identical graph/search settings.

## Validation

Tests cover independent integer distance/mean oracles, exact paired/packed
results and RNG state, ties/empty clusters, coordinate and row tails, scattered
rows, overflow fallback, finite cancellation, FP16 conversion, file sampling,
and real IVF build/search for every strategy. No approximate candidate is
silently accepted using a loose output tolerance.

```powershell
cargo test -p diskann-vector paired_l2
cargo test -p diskann --features ivf-experimental ivf::clustering
cargo test -p diskann-benchmark --features ivf-two-means graph_ivf
cargo clippy --workspace --all-targets --features diskann-benchmark/ivf-two-means -- -D warnings
```

The full IVF suite has a separately reproduced baseline failure:
`invalid_initialization_and_uninitialized_insertion_fail` expects initialization
to reject a NaN centroid, which the upstream implementation accepts. The test is
unchanged. Focused passing tests are not a claim that the full suite is green.
Scalar, V3 and V4 paths ran on this host; AArch64 Windows library/tests compile,
but Neon execution and old-x86 emulation were unavailable.

The clustering/benchmark refactor was checked against the original source at
`c7e54d08` in one process: 7,812 complete-fit comparisons across seven strategies,
coordinate/row tails, contiguous/scattered rows, cancellation-prone values and
finite inputs that overflow distances. Center bits, f64 sum bits, RNG state and
final assignments matched. A separate release check matched all seven strategies'
YFCC build/search fingerprints, recall and logical scan counts (all 100 queries),
plus 448 standalone strategy/workload diagnostic comparisons. These are
correctness checks, not new performance estimates.

The refactor passes 14 clustering tests, 12 feature-enabled benchmark tests and
eight default benchmark tests; the full IVF suite remains 20 passed and the one
NaN-initialization failure above. Formatting, default/feature-enabled workspace
Clippy, a release build and AArch64 Windows test compilation pass. Separate LLVM
profiles report 95.33% line coverage over selected library files and 93.63% over
selected benchmark/input files; these local figures are not Codecov patch status.

## Decision boundary

The implementation provides reproducible experiments, not a hardware-independent
winner. Keep the production default paired until a candidate shows repeatable
end-to-end gains under an agreed numerical contract. Useful follow-ups are
replaying actual parent working sets, preserving the existing reduction tree in
a multi-point kernel, and vectorizing f64 accumulation across coordinates while
retaining the per-coordinate point order. Packing must continue to pay for
itself within the full fit. An approximate dot-product path needs a justified
error bound and tested paired fallback before a behavior-preserving claim.
