# DiskANN Core Testing

Improve the state of DiskANN core testing by:

1. Implementing pedantic memory providers to verify function contract APIs such as:

    * Neighbors passed to `set_neighbors` are unique.
    * Neighbors passed to `append_neighbors` are unique.
    * Number of distance/neighbor retrievals is tracked.
    * Presence or absence of deleted neighbors is properly documented and handled.
    * ID roll back contract (e.g. invoking guard completion handlers or not) is properly
      implemented.
    * Filter API contracts are documented and observed.
    * etc.

2. Augmenting the `diskann-core` test suite to generate expected test results when requested and use these generated test results to verify no change in behavior.

## Pedantic Providers

Our current test infrastructure relies on the in-memory providers which are not suitable for testing purposes because they lack fine-grained metric gathering (e.g. number of vector retrievals) and invariant checking behavior such as verifying that a vector with a given ID has actually been inserted into the provider.
The latter can be understood by recognizing that any ID between 0 and the maximum configured points is "fair game".
These decisions were made for performance reasons and thus we do not want to bloat the design of these providers with optional telemetry.

Instead, a test provider can be developed in tandem with the algorithm implementations to both ensure documented algorithmic guarantees are observed (and tested) and gather fine-grained metric gathering to assist with CI testing.

### Feature Gating

Since these providers should eventually be incorporated into the benchmark to allow testing and analysis of non-trivial workloads, they cannot be solely gated by Rust's `cfg(test)` feature.

Instead, this RFC proposes adding a new `testing` feature to `diskann-core` that makes the test providers available to third-party applications.
An additional benefit is that various testing utilities (e.g. synthetic test generation) can also be made available under the same feature to provide one source of testing utilities.

Since pedantic providers will be available to third party crates, they must be subject to full `clippy` runs.
In particular, we allow `unwrap`, `expect` and `panic` in code provided it is gated behind `cfg(test)`.
This does not apply to general feature gates.
As such, there is an expectation that code available under the `testing` feature be of sufficient quality to be consumed by third party crates.

### Existing Infrastructure

The `DebugProvider` currently provides some of these features, but integrates PQ (which should not be included in `diskann-core`).

This RFC essentially proposes making the `DebugProvider` the official pedantic provider by ripping out PQ support and improving the invariant checks.

## Test Result Caching

Our current "integration" tests consist of somewhat loosely defined checks for recall and graph connectivity.
Further, they do not track metrics like number of comparisons, number of hops, etc.
This is for good reason since coming up with values for these metrics analytically is not feasible, nor is manually updating the expected values for these metrics every time the implementation changes.

Instead, this RFC proposes better test coverage by allowing tests to cache their expected results in a human readable form (i.e. JSON) that are then checked in to the repository.
When executing unit tests, individual tests will attempt to retrieve the cached results and compare the cached results with those that are newly generated - reporting any discrepancy.
Missing cached tests will be treated as an error.

To allow test and algorithmic development, an environment variable `DISKANN_TEST=overwrite` will be used.
When this variable is present and set, tests will instead save their results to JSON.
This allows us to:

* Easily generate results for new tests.
* Update existing tests with more or fewer metrics.
* Update existing tests to respond to algorithmic changes.

In the last case, `git diff` can be used to sanity check the new results.
Test result changes that do not pass the sanity check require further investigation and justification to merge.

### Expected Workflow

#### Location of Test Results

Test results will be stored in `diskann-core/tests/generated`.
**Only** automatically generated results should go here.
This allows developers to wholesale regenerate tests by deleting `diskann-core/tests/generated` and rerunning the entire test suite with `DISKANN_TEST=overwrite`.
The ability to completely blow away the cache is important to enable garbage collection of results that are no longer used.

#### Test Registration

Tests will register their results as a relative path (which will be turned into an absolute path rooted in `diskann-core/tests/generated` by the caching infrastructure.
Enabling hierarchical directories in this way will help keep file names under control, prevent collision, and keep things organized.

There is a risk that two tests will register themselves with the same name and conflict at run time.
Conflict is not something we can prevent in general; tests can run in their own processes, making conflict resolution very difficult.
Fortunately, I do not think this is a big problem because we can use `#[track_caller]` to tag test results with the file that generated them.
When two tests **do** conflict, one of the tests will panic and we'll hopefully be able to find and fix the conflict.

#### Serialization Strategy

After spending longer than I care to admit looking for alternatives, I think that [`serde`](https://docs.rs/serde/latest/serde/) and [`serde_json`](https://docs.rs/serde_json/latest/serde_json/) are still our best bets, despite the potential compile time issues with `serde`.
However, we should strictly keep `serde` and `serde_json` as `dev` dependencies.
I do not think we want to (yet) provide `serde` compatibility with `diskann-core` data structures because doing so implies some level of stability that I do not believe we want to guarantee yet.

Test data structure can be made to optionally implement `serde::Serialize` and `serde::Deserialize` using [`cfg_attr`](https://doc.rust-lang.org/reference/conditional-compilation.html#the-cfg_attr-attribute).

New tests should judiciously derive `Serialize` and `Deserialize` in an attempt to limit compile time bloat.

#### Writing Checks

To improve developer experience when debugging test failures, a `VerboseEq` trait can be used that includes the full path to items causing structural inequality.
This is a quality of life improvement over Rust's default `PartialEq` trait which only provides a boolean signal of success or failure.
The initial implementation:

* Uses the `ANNError` context chain to reconstruct the path to the failing item.
* Provides implementations for standard library types like primitives (integers/floating point numbers) and vectors.
* A fairly simple `macro_rules!` macro `verbose_eq!` for implementing `VerboseEq` for custom types.

Future extensions can add a proper [derive macro](https://doc.rust-lang.org/reference/procedural-macros.html#r-macro.proc.derive) for better ergonomics.

When writing tests, developers should be encouraged to use `VerboseEq` (with its helper check `assert_verbose_eq!`) when checking cached results.
