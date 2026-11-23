# DiskANN

Stay tuned for more updates!

## Developer Docs

### Test Baselines

Developers are strongly encouraged to consider the [caching infrastructure](src/test/cache.rs)
when writing index tests to provide an early warning of algorithmic changes.

This infrastructure serializes test results into a file in [`test/generated`](test/generated)
that serves as the baseline in the normal test flow. Any difference between the baseline
result and a test value gets flagged as a test failure for further review.

To regenerate baselines, run the test with the environment variable:
```
DISKANN_TEST=overwrite
```
Since tests are in a (somewhat) human readable JSON form, regenerated results can be inspected
during the review process to flag regressions early.

Before checking in new test results, it's a good idea to completely delete [`test/generated`](test/generated)
to ensure that unused baselines get removed from the repository.

The API for registering and retrieving test results is in [cache.rs](src/test/cache.rs)
and consists of:

* `fn get_or_save_test_results<R>(test_name: &str, results: &R) -> R`: Get the results for
  `test_name` in normal testing mode, or save `results` as a baseline when in overwrite mode.

  Argument `test_name` consists of paths separated by a `/` like `a/b/test` that will get
  saved beneath [`test/generated`](test/generated).

* `TestRoot` and `TestPath`: Utilities for efficiently incrementally building `test_name`.

The above API will return the previously saved baseline in the normal test mode, which can
be compared with the `results` argument.

For structural comparison diagnostics and assertion helpers, see [cmp.rs](src/test/cmp.rs).

## Dense flat search with GEMM

The optional `flat-gemm` feature enables blocked FP32 inner-product search through
`flat::gemm::knn_inner_product`. It uses an explicit Rayon thread pool to scan database
partitions, reuses each database tile across multiple queries, and merges local Top-k
results. Floating-point accumulation can differ from the scalar kernels.

Each request defaults to one worker, even in a larger shared pool. Set
`GemmOptions::max_workers` explicitly to allow more database partitions to run in
parallel; the effective limit is capped by the pool size and database row count.
Batching still benefits from GEMM with one worker. Individual GEMMs stay sequential
to avoid nested parallelism.

The paired Wikipedia Cohere benchmark compares it with the existing scalar-scoring
visitor at the same thread budget, and checks scores, Top-k overlap and groundtruth recall:

```powershell
cargo run -p diskann --release --features flat-gemm --example flat_gemm -- data/wikipedia_cohere target/flat_gemm.csv 128 32 4 100 4
```

The numeric arguments are sampled query count, query batch size, threads, k and repetitions.
The example explicitly sets both the pool size and `max_workers` to the thread argument.
An optional final argument sets the database block size (default 1024). CSV durations
measure entire query batches; time divided by the query count is amortized cost, not
individual query latency. Data loading, pool creation and validation are outside timing.

The DiskANN team thanks [INFINI Labs](https://github.com/infinilabs/) for transferring ownership of the [diskann](https://crates.io/crates/diskann) crate!
