# DiskANN

Stay tuned for more updates!

## Search phase tracing

With the default `tracing` feature, shared graph traversal routines emit a `DEBUG`
span named `diskann.graph.search.traversal` with target
`diskann::graph::search::traversal`. Flat search emits `diskann.flat.search.scan`
with target `diskann::flat::search::scan` around distance scanning and candidate
queue insertion. Both exclude outer accessor/scratch setup and postprocessing,
and record no arguments. These routines can also be called during construction
or repeatedly during a paged search.

The caller's subscriber controls filtering and records span lifetimes for elapsed
phase timing, not CPU time. An unfiltered subscriber can enable `DEBUG`; explicitly
filter these targets `off` to disable them. See the
[disk search tracing documentation](../diskann-disk/README.md#search-phase-tracing)
for exact boundaries and example filter directives.

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

The DiskANN team thanks [INFINI Labs](https://github.com/infinilabs/) for transferring ownership of the [diskann](https://crates.io/crates/diskann) crate!
