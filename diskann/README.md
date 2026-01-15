# DiskANN

Stay tuned for more updates!

## Developer Docs

### Test Baselines

Developers are strongly encouraged to consider the [caching infrastructure]()
when writing index tests to provide an early warning of algorithmic changes.

This infrastructure serializes test results into a file in `diskann/tests/generated`
that serves as the baseline in the normal test flow. Any difference between the baseline
result and a test value gets flagged as a test failure for further review.

To regenerate baselines, run the test with the environment variable:
```
DISKANN_TEST=overwrite
```
Since tests are in a (somewhat) human readable JSON form, regenerated results can be inspected
during the review process to flag regressions early.

Before checking in new test results, it's a good idea to completely delete `diskann/tests/generated`
to ensure that unused baselines get removed from the repository.

The API for registering and retrieving test results is in `diskann/src/tests/cache`
and consists of:

* `fn get_or_save_test_results<R>(test_name: &str, results: &R) -> R`: Get the results for
  `test_name` in normal testing mode, or save `results` as a baseline when in overwrite mode.

  Argument `test_name` consists of paths separated by a `/` like `a/b/test` that will get
  saved to `diskann/tests/generated/a/b/test.json`.

* `TestRoot` and `TestPath`: Utilities for efficiently incrementally building `test_name`.

The above API will return the previously saved baseline in the normal test mode, which can
be compared with the `results` argument.

When comparing baselines, developers should use the `diskann::tests::cmp::VerboseEq`
which provides more diagnostics regarding the source of structural inequality than the
standard libraries `PartialEq` trait. Additional utilities include

* `diskann::tests::cmp::verbose_eq!`: A trait for automatically implementing `VerboseEq`.
  This macro can be used until a proper `derive` macro is implemented:
  ```rust
  use diskann::test::cmp::verbose_eq;

  struct MyStruct {
      a: String,
      b: f32,
      c: usize,
  }

  // Implement the `VerboseEq` trait for `MyStruct`.
  verbose_eq!(MyStruct { a, b, c });
  ```

* `diskann::test::cmp::assert_eq_verbose!`: The equivalent of the standard library's
  `assert_eq` but using `VerboseEq` to provide more information on the test failure.

The DiskANN team thanks [INFINI Labs](https://github.com/infinilabs/) for transferring ownership of the [diskann](https://crates.io/crates/diskann) crate!
