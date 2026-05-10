### Test Baseline Caching System

DiskANN uses a baseline caching system for regression detection. 
Test results are serialized as JSON into `diskann/tests/generated/` and compared against on subsequent runs. 
Any difference is flagged as a test failure.

- To regenerate baselines: run tests with `DISKANN_TEST=overwrite`
- Before checking in: delete `diskann/tests/generated/` first, then regenerate to prune unused baselines
- Regenerated JSON files should be inspected via `git diff` during review

The APIs are **`pub(crate)`** (internal to the `diskann` crate only):
- [`diskann/src/test/cache.rs`](diskann/src/test/cache.rs) — `get_or_save_test_results`, `TestRoot`, `TestPath`
- [`diskann/src/test/cmp.rs`](diskann/src/test/cmp.rs) — `VerboseEq` trait, `verbose_eq!` macro, `assert_eq_verbose!`

See [`diskann/README.md`](diskann/README.md) for additional details.