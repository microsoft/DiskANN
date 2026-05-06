# diskann-benchmark-multi-vector

Benchmarks and regression detection for the **multi-vector distance
operations** exposed by `diskann-quantization` — `Chamfer` and `MaxSim` —
across `f32` and `f16` element types.

## Layout

- `src/lib.rs` — benchmark library: input/tolerance schemas, kernel
  dispatch, regression checker.
- `src/bin.rs` — `benchmark-multi-vector` CLI entry point.
- `examples/multi-vector.json` — full benchmark matrix covering both
  operations across the registered kernels and a representative range of
  shapes.
- `examples/test.json` — minimal smoke configuration consumed by the
  integration tests.
- `examples/tolerance.json` — default regression thresholds.

## Registered kernels

The crate registers four kernels — one per `(element_type, implementation)`
pair:

| Tag                              | Element | Implementation       |
| -------------------------------- | ------- | -------------------- |
| `multi-vector-op-f32-optimized`  | `f32`   | `QueryComputer`      |
| `multi-vector-op-f16-optimized`  | `f16`   | `QueryComputer`      |
| `multi-vector-op-f32-reference`  | `f32`   | `Chamfer` / `MaxSim` |
| `multi-vector-op-f16-reference`  | `f16`   | `Chamfer` / `MaxSim` |

The **optimized** path constructs a `QueryComputer` once per shape (which
internally selects the best available SIMD kernel for the host) and calls
`chamfer` / `max_sim` inside the timed loop. The **reference** path drives
the `Chamfer` / `MaxSim` fallback used by the `multi_vector` unit tests —
useful both as a numerical ground truth and as a baseline to measure SIMD
speedups against.

## Time normalization

Per-measurement latency is normalized to **nanoseconds per inner-product
call**, abbreviated `ns/IP`:

```
ns/IP = min_latency_µs * 1000 / (Q * D * loops_per_measurement)
```

Two important properties:

- **Independent of `Q`, `D`, and `loops_per_measurement`.** Reshaping the
  benchmark or scaling the loop budget leaves the metric unchanged, so
  cache-residency effects and SIMD utilization show up directly.
- **Approximately linear in `Dim`.** Each inner-product call is itself an
  O(`Dim`) operation, so `ns/IP` grows with `Dim` — that is why the table
  headers read `ns/IP @ Dim`. Compare across rows with the same `Dim`; to
  compare across different `Dim`s, divide further by `Dim` to recover ns
  per scalar multiply.

This is the right metric for the two things this crate cares about:
detecting per-shape regressions (the `Dim` factor cancels) and comparing
optimized vs. reference at a fixed shape.

## Usage

All examples below assume you are inside the crate directory and use a
small shell function for brevity:

```bash
bench() { cargo run --release -p diskann-benchmark-multi-vector --bin benchmark-multi-vector -- "$@"; }
```

### Run benchmarks

`run` executes every job in the input file and writes per-measurement
latencies plus percentiles to the output file:

```bash
bench run --input-file examples/multi-vector.json --output-file before.json
```

### Regression check workflow

The check workflow is **two-phase**: validate the tolerance file once, then
compare two recorded result files.

**Phase 1 — preflight.** No benchmarks are executed. The verifier confirms
that every entry in `tolerance.json` matches at least one job in the input
file, and that every job is matched by exactly one entry. Run it whenever
you edit `tolerance.json`:

```bash
bench check verify \
  --tolerances examples/tolerance.json \
  --input-file examples/multi-vector.json
```

**Phase 2 — comparison.** Record results before and after a code change,
then compare. The command exits non-zero if any run regresses past its
tolerance:

```bash
# On the baseline commit
bench run --input-file examples/multi-vector.json --output-file before.json

# On the change commit
bench run --input-file examples/multi-vector.json --output-file after.json

# Compare
bench check run \
  --tolerances examples/tolerance.json \
  --input-file examples/multi-vector.json \
  --before before.json --after after.json \
  --output-file checks.json
```

A run **fails** when its post-change `ns/IP` minimum exceeds the
baseline minimum by more than `min_time_regression` (default `0.05` =
5%). Improvements (negative change) always pass.

### How tolerances are matched to jobs

Each entry in `tolerance.json` has the shape `{ input, tolerance }`. The
`input` block acts as a **partial template** against the jobs in the input
file: any field present must match; missing fields are wildcards.

The shipped `tolerance.json` uses an empty `"content": {}`, which matches
every `multi-vector-op` job — so a single 5% threshold applies to all four
kernels. To apply different thresholds per implementation, add more
specific entries, e.g.:

```json
{ "input":     { "type": "multi-vector-op", "content": { "implementation": "reference" } },
  "tolerance": { "type": "multi-vector-tolerance", "content": { "min_time_regression": 0.10 } } }
```

`check verify` will reject the file if entries overlap or leave any job
unmatched.
