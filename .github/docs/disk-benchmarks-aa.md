# DiskANN A/A Benchmark Stability Test

> Companion documentation for
> [`.github/workflows/disk-benchmarks-aa.yml`](../workflows/disk-benchmarks-aa.yml).

## Purpose

The A/A benchmark runs **main vs. main** — the same code is built once and
run twice. Its purpose is to detect **environment noise** on the CI runners,
not code regressions. If the results between two identical runs differ beyond
configured thresholds, it indicates that the runner environment is too noisy
for reliable benchmarking.

## Schedule

| Trigger | Details |
|---------|---------|
| **Cron** | Daily at **9:00 AM UTC** (`0 9 * * *`) |
| **Manual** | Can be triggered via `workflow_dispatch` for debugging |

Only one run is allowed at a time (`cancel-in-progress: true`).

## Datasets

Two datasets are benchmarked in parallel via a matrix strategy
(with `fail-fast: false`, so both always run):

| Dataset | Config | Archive |
|---------|--------|---------|
| `wikipedia-100K` | `wikipedia-100K-disk-index.json` | `wikipedia-100K.tar.gz` |
| `openai-100K` | `openai-100K-disk-index.json` | `openai-100K.tar.gz` |

Config and tolerance files live in
[`diskann-benchmark/perf_test_inputs/`](../../diskann-benchmark/perf_test_inputs/).

## Tolerance Thresholds

Defined in
[`disk-index-tolerances.json`](../../diskann-benchmark/perf_test_inputs/disk-index-tolerances.json):

| Metric | Allowed Regression |
|--------|--------------------|
| Build time | 10 % |
| QPS | 10 % |
| Recall | 1 % |
| Mean I/Os | 1 % |
| Mean comparisons | 1 % |
| Mean latency | 15 % |
| P95 latency | 15 % |

## Failure Notification

When any matrix job fails, a `notify-on-failure` job creates a GitHub issue
tagged `@microsoft/diskann-disk-maintainers` with labels `benchmark` and
`A/A-failure`. The team should inspect the uploaded artifacts (retained 30
days) to determine whether thresholds need tuning or there is a runner
environment issue.

## Comparison with A/B Benchmarks

The A/A test should be distinguished from the **A/B benchmark** workflow
([`disk-benchmarks.yml`](../workflows/disk-benchmarks.yml)), which compares
a **PR branch vs. a baseline** (usually `main`). The A/B workflow builds two
separate binaries and is designed to catch performance regressions in code
changes. In contrast, the A/A test builds only once and runs twice, so any
differences are purely due to environment variability.
