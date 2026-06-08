# Benchmarking Infrastructure

The goal of the benchmarking infrastructure is to make performance testing and development
easier by providing a "one click" runner for benchmarks with machine readable output.

## Usage

To get started, run
```sh
cargo run --release --package diskann-benchmark -- skeleton
```
which will print to `stdout` the following JSON schema:
```json
{
  "search_directories": [
    "directory/a",
    "directory/b"
  ],
  "jobs": []
}
```
This is a skeleton of the input used to run benchmarks.

* `search_directories`: A list of directories that can be searched for input files.
* `output_directory`: A *single* output directory where index files may be saved to, and where the benchmark tool will look for any loaded indices that aren't specified by absolute paths.
* `jobs`: A list of benchmark-compatible inputs. Benchmarking will run each job sequentially,
  and write the outputs to a file.

`jobs` should contain objects that look like the following:
```json
{
  "type": <the benchmark type to run>,
  "content": {
    "source":{
      "index-source": "Build" < or "Load", described below>,
      "data_type": <the data type of the workload>,
      "data": <the data file>,
      "distance": <the distance metric>,
      "max_degree": <the max degree of the graph>,
      "l_build": <the search length to use during inserts>,
      "alpha": <alpha to use during inserts>,
      "backedge_ratio": <the ratio of backedges to add during inserts>,
      "num_threads": <the number of threads to use during graph construction>,
      "num_start_points": <the number of starting points in the graph>,
      "num_insert_attempts": <the number of times to increase the build_l in the case that not enough edges can be found during the insertion search>,
      "retry_threshold" <the multiplier of R that when an insert contains less edges will trigger an insert retry with a longer build_l>
      "saturate_inserts": <In the case that we cannot find enough edges, and have expended our search, whether we should add occluded edges>,
      "save_path": <Optional path where the index and data will be saved to>
      },
    "search_phase": {
      "search_type": "topk" <other search types and their requisite arguments can be found in the `examples` directory>,
      "queries": <query file>,
      "groundtruth": <ground truth file>,
      "reps": <the number of times to repeat the search>,
      "num_threads": <the number of threads to use for search>,
      "runs": [
        {
          "search_n": <the number of elements to consider for top k (useful for quantization)>,
          "search_l": <length of search queue>,
          "target_recall": // this is an optional argument that is used for sample based declarative recall
          {
            "target": <a list of positive integers describing the target recall value>,
            "percentile": <a list floats describing the percentiel that the target recall is refering to>,
            "max_search_l": <how long search_l should be for calibrating target recall. This should be large (1000+)>,
            "calibration_size": <how many queries to run to calculate the hops required for our target>
          },
          "recall_k": <how many ground truths to serach for>
        }
      ]
    }
  }
}
```

In the case of loading an already constructed index rather than building, the "source" field should look like:
```json
{
  "source":{
    "index-source": "Load",
    "data_type": <the data type of the workload>,
    "distance": <the distance metric>,
    "load_path": <Path to the loaded index. Must be either contained at most one level deep in "output_directory" or an absolute path.>
  },
}
```

### Finding Inputs

Registered inputs are queried using
```sh
cargo run --release --package diskann-benchmark -- inputs
```
which will list something like
```
Available input kinds are listed below:
    graph-index-build
    graph-index-build-pq
```
To obtain the JSON schema for an input, add its name to the query like
```sh
cargo run --release --package diskann-benchmark -- inputs graph-index-build
```
which will generate something like
```json
{
  "type": "graph-index-build",
  "content": {
    "search_phase": {
      "groundtruth": "path/to/groundtruth",
      "num_threads": [
        1,
        2,
        4,
        8
      ],
      "queries": "path/to/queries",
      "reps": 5,
      "runs": [
        {
          "recall_k": 10,
          "search_l": [
            10,
            20,
            30,
            40
          ],
          "search_n": 10
        }
      ],
      "search-type": "topk"
    },
    "source": {
      "alpha": 1.2000000476837158,
      "backedge_ratio": 1.0,
      "data": "path/to/data",
      "data_type": "float32",
      "distance": "squared_l2",
      "index-source": "Build",
      "insert_retry": null,
      "l_build": 50,
      "max_degree": 32,
      "multi_insert": {
        "batch_parallelism": 32,
        "batch_size": 128,
        "intra_batch_candidates": "none"
      },
      "num_threads": 1,
      "save_path": null,
      "start_point_strategy": "medoid"
    }
  }
}
```
The above can be placed in the `jobs` array of the skeleton file.
Any number of inputs can be used.

> **_NOTE:_**: The contents of each JSON file may (and in some cases, must) be modified.
  In particular, files paths such as `"data"`, `"queries"`, and `"groundtruth"` must be
  edited and point to valid `.bin` files or the correct type. These paths can be kept as
  relative paths, benchmarking will look for relative paths among the `search_directories`
  in the input skeleton.

> **_NOTE:_**: Target recall is a more advanced feature than `search_l`. If it is defined, `search_l` does
  not need to be, but both are compatible together. This feature works by taking a sample of
  of the query set and using it to determine search_l prior to running the main query set.
  This is a way of performing automating tuning for a workload. The target is the recall target
  you wish to achieve. The percentile is the hops percnetile to achieve the target recall i.e.
  0.95 indicates 95% of the queries in the sampled set will be above the recall target. max_serach_l
  is the maximum time we will serach to find our tuned recall target. This value should be relatively
  large to prevent failure. If you notice that you your tuned search_l is close to max_search_l it
  is advised to run again with a larger value. Finally, calibration_size is the number of qureies
  that are sampled to calculate recall values during the tuning process. Note that these will be reused
  for benchmarking later.

### Finding Benchmarks

Registered benchmarks are queries using the following.
```sh
cargo run --release --package diskann-benchmark -- benchmarks
```
Example output is shown below:
```
Registered Benchmarks:
    graph-index-full-precision-f32:
        tag "graph-index-build"
        Data/Query Type: float32
        Search Kinds: "topk", "range", "topk-beta-filter", and "topk-multihop-filter"
    graph-index-full-precision-f16:
        tag "graph-index-build"
        Data/Query Type: float16
        Search Kinds: "topk"
    graph-index-pq-f32:
        tag "graph-index-build-pq"
        Data/Query Type: float32
        Search Kinds: "topk" and "range"
    ...
```
The keyword after "tag" corresponds to the type of input that the benchmark accepts.

#### Adding Search Kinds

Be aware that by default, not all benchmark types support all flavors of search.
This is a deliberate choice to keep the compile time for `diskann-benchmark` mostly reasonable.
If you are doing experiments and need (in the example above) range search for the `f16` index,
this is usually easily done with a small code change.

With the example of adding Range search to the `f16` index, the registration site:
```rust
registry.register(
    "async-full-precision-f16",
    FullPrecision::<f16>::new()
        .search(plugins::Topk),
)?;
```
Can be updated to:
```rust
registry.register(
    "async-full-precision-f16",
    FullPrecision::<f16>::new()
        .search(plugins::Topk)
        .search(plugins::Range),
)?;
```
This will both compile the range search implementation and make it available for benchmark
matching.

### Running Benchmarks

Benchmarks are run with
```sh
cargo run --release --package diskann-benchmark -- run --input-file ./diskann-benchmark/example/graph-index.json --output-file output.json
```

A benchmark run happens in several phases.
First, the input file is parsed and simple data invariants are checked such as matching with
valid input types, verifying the numeric range of some integers, and more. After successful
deserialization, more pre-flight checks are conducted. This consists of:

1. Checking that all input files referenced exist on the file system as files.
    Input file paths that aren't absolute paths will also be searched for among the list of
    search directories in order. If any file cannot be resolved, an error will be printed
    and the process aborted.

2. Any additional data invariants that cannot be checked at deserialization time will also
   be checked.

3. Matching inputs to benchmarks happens next.
   To help with compile times, we only compile a subset of the supported data types and
   compression schemes offered by DiskANN. This means that each registered benchmark may
   only accept a subset of values for an input. Backend validation makes sure that each input
   can be matched with a benchmark and if a match cannot be found, we attempt to provide a
   list of close matches.

Once all checks have succeeded, we begin running benchmarks. Benchmarks are executed
sequentially and store their results in an arbitrary JSON format. As each benchmark completes,
all results gathered so far will be saved to the specified output file. Note that
long-running benchmarks can also opt-in to incrementally saving results while the benchmark
is running. This incremental saving allows benchmarks to be interrupted without data loss.

In addition to the machine-readable JSON output files, a (hopefully) helpful summary of the
results will be printed to `stdout`.

### Streaming Runs
Running the benchmark on a streaming workload is similar to other registered benchmarks,
relying on the file formats and streaming runbooks of `big-ann-benchmarks`

First, set up the runbook and ground truth for the desired workload. Refer to the `README` in
`big-ann-benchmarks/neurips23` and the runbooks in `big-ann-benchmarks/neurips23/streaming`.

Benchmarks are run with
```sh
cargo run --release --package diskann-benchmark -- run --input-file ./diskann-benchmark/example/graph-index-dynamic.json --output-file dynamic-output.json
```
Note in the example json that the benchmark is registered under `graph-index-dynamic-run`,
instead of `graph-index-build` etc..

A streaming run happens in several phases.
First, the input file is parsed and data is checked for its validity. The check consists of
1. All input files referenced can be found in the file system.
2. The ground truth files required by the search stages in the runbook exist in `gt_directory`,
which will be searched under `search_directories`. For each search stage x (1-indexed),
the gt directory should contain exactly one `step{x}.gt{k}`.

The input file will then be matched to the proper dispatcher, similar to the static case of the
benchmark. At the end of the benchmark run, structured results will be saved to `output-file`
and a summary of the statistics will be pretty-printed to `stdout`.

The streaming benchmark implements the user layer of the index. Specifically, it tracks the tags
of vectors (`ExternalId` in the rust codebase) and matches agains the slots (`InternalId` in the
rust codebase), looking up correct vectors in the raw data by its _sequential id_ for `Insert` and
`Replace` operations. If the index will run out of its allocated number of slots, the streaming
benchmark calls `drop_deleted_neighbors` (with `only_orphans` currently set to false) across all update
threads, then calls `release` on the Delete trait of the `DataProvider` to release the slots. On
`Search` operations, the streaming benchmark takes care of translating the slots that the index returns
to tags stored in the ground truth. These user logic are guarded by invariant checking in the benchmark.
This is designed to be compatible with the fact that `ExternalId` and `InternalId` are the same in the
barebone rust index and is separately handled by its users at the time when the streaming benchmark is
added. See `benchmark/src/utils/streaming.rs` for details. The integration tests for this
can be run by `cargo test -p benchmark streaming`.

## Adding New Benchmarks

The benchmarking infrastructure works in two phases: first a raw JSON file is parsed into a
collection of registered `diskann_benchmark_runner::Input`s. Then, each input is matched
with a `diskann_benchmark_runner::Benchmark`. A `diskann_benchmark_runner::Registry` contains
the collection of all registered inputs and benchmarks.

New benchmarks must implement the `diskann_benchmark_runner::Benchmark` trait, which has its
input as an associated type. Registering a benchmark via `Registry::register` will
automatically register the associated input.

At run time, the front end will discover benchmarks in the input JSON file and use the tag
string in the `type` field to select the correct input deserializer. Benchmarks will
be matched to inputs using `Benchmark::try_match`, with the best candidate being selected
to be run.

### Example

#### Defining a new Input Type

Here, we will walk through adding a very simple "compute\_groundtruth" set of benchmarks.
First, define an input type in `src/benchmark/inputs`.
This may look something like the following.
```rust
use diskann_benchmark_runner::{utils::datatype::DataType, files::InputFile};

// We derive from `Serialize` and `Deserialize` to be JSON compatible.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct ComputeGroundTruth {
    // The data type of the dataset, such as `f32`, `f16`, etc.
    pub(crate) data_type: DataType,
    // The location of the input dataset.
    //
    // The type `InputFile` is used to opt-in to file path checking and resolution.
    pub(crate) data: InputFile,
    pub(crate) queries: InputFile,
    pub(crate) num_nearest_neighbors: usize,
}
```
We need to implement `diskann_benchmark_runner::Input` for the type. This trait associates
a tag name used for deserialization and benchmark matching, a `Raw` type for JSON
serialization/deserialization, a `from_raw` constructor that performs post-deserialization
validation (e.g., resolving file paths via the `Checker`), and an `example` that supplies
sample JSON layouts for the CLI.

In the context of the `ComputeGroundTruth` type, we use `from_raw` to check that the input
files are valid.

```rust
impl diskann_benchmark_runner::Input for ComputeGroundTruth {
    // The raw form is just `Self` since the struct is directly deserializable.
    type Raw = Self;

    // This gets associated with the JSON representation returned by `example` and at run
    // time, inputs tagged with this value will be given to `from_raw`.
    fn tag() -> &'static str {
        "compute_groundtruth"
    }

    // Construct from the raw deserialized form, performing file path resolution.
    fn from_raw(
        mut raw: Self::Raw,
        checker: &mut diskann_benchmark_runner::Checker,
    ) -> anyhow::Result<Self> {
        raw.data.resolve(checker)?;
        raw.queries.resolve(checker)?;
        Ok(raw)
    }

    // Serialize `self` to JSON.
    fn serialize(&self) -> anyhow::Result<serde_json::Value> {
        Ok(serde_json::to_value(self)?)
    }

    // Return an example input to help users create an input file.
    fn example() -> Self {
        Self {
            data_type: DataType::Float32,
            data: InputFile::new("path/to/data"),
            queries: InputFile::new("path/to/queries"),
            num_nearest_neighbors: 100,
        }
    }
}
```

#### Benchmark Registration

With the new input type ready, we register a benchmark that uses it with the
`diskann_benchmark_runner::Registry`. Input registration happens automatically as a
side-effect. Registration can fail if a different input type with the same `tag` was already
registered; duplicate registrations of the same tag and type are allowed.

When a benchmark is registered, the input will be available using
```sh
cargo run --release --package diskann-benchmark -- inputs
```
and
```sh
cargo run --release --package diskann-benchmark -- inputs compute-groundtruth
```
will display an example JSON input for our type.

To implement benchmarks, we register them with the `diskann_benchmark_runner::Registry`.
The simplest thing we can do is something like this:
```rust
use diskann_benchmark_runner::{
    benchmark::{MatchScore, FailureScore},
    Benchmark, Checkpoint, Output,
};

// Benchmarks can be stateful.
struct RunGroundTruth;

impl Benchmark for RunGroundTruth {
    // The input that will be registered along with the benchmark.
    type Input = ComputeGroundTruth;

    // Real benchmarks should have output that will be saved. For this example, there
    // is no meaningful output.
    type Output = ();

    // Always match the input.
    fn try_match(&self, input: &Self::Input) -> Result<MatchScore, FailureScore> {
        Ok(MatchScore::new(0))
    }

    // Describe the benchmark for CLI display and debugging.
    fn description(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        _input: Option<&Self::Input>,
    ) -> std::fmt::Result {
        write!(f, "compute groundtruth")
    }

    // Run the benchmark (for this example, nothing happens).
    fn run(
        &self,
        input: &Self::Input,
        checkpoint: Checkpoint<'_>,
        output: &mut dyn Output,
    ) -> anyhow::Result<Self::Output> {
        Ok(())
    }
}

fn register(registry: &mut diskann_benchmark_runner::Registry) -> anyhow::Result<()> {
    // Register the benchmark and its associated input.
    Ok(registry.register("compute-groundtruth", RunGroundTruth)?)
}
```

What is happening here is that the implementation of `Benchmark::try_match` checks if the
benchmark matches the runtime parameters in the associated input. For the case of the example,
this always succeeds. If the `try_match` is successful, then the benchmarking infrastructure
will call `Benchmark::run`. This mechanism allows multiple backend benchmarks to exist and
pull input from the deserialized inputs present in the current run. If multiple benchmarks
match an input, then the benchmark with the lowest `MatchScore` will be selected.

The argument `checkpoint: diskann_benchmark_runner::Checkpoint<'_>` allows long-running
benchmarks to periodically save incremental results to file by calling the `.checkpoint`
method. This function creates a new snapshot every time it is invoked, so benchmarks do not
need to worry about redundant data.

The argument `output: &mut dyn diskann_benchmark_runner::Output` is a dynamic type where
all output should be written to. Additionally, it provides a
[`ProgressDrawTarget`](https://docs.rs/indicatif/latest/indicatif/struct.ProgressDrawTarget.html)
for use with [indicatif](https://docs.rs/indicatif/latest/indicatif/index.html) progress bars.
This supports output redirection for integration tests and piping to files.

With the benchmark registered, that is all that is needed.

#### Matching with `try_match`

The functionality offered by `Benchmark::try_match` is much more powerful than what was
described in the simple example. In particular, careful implementation will allow your
benchmarks to be more easily discoverable from the command-line and can also assist in
debugging by providing "near misses".

**Fine Grained Matching**

The method `Benchmark::try_match` returns both a successful `MatchScore` and an
unsuccessful `FailureScore`. The registry will only invoke methods where all arguments
return successful `MatchScores`. Additionally, it will call the method with the "best"
overall score. So, you can make some registered benchmarks "better fits" for inputs
returning a better match score.

When the registry cannot find any matching method for an input, it begins a process of
finding the "nearest misses" by inspecting and ranking methods based on their `FailureScore`.
Benchmarks can opt-in to this process by returning meaningful `FailureScores` when an input is
close, but not quite right.

**Benchmark Description and Failure Description**

The trait `Benchmark` has another method:
```rust
fn description(f: &mut std::fmt::Formatter<'_>, from: Option<&Self::Input>);
```
This is used for self-documenting the matching rule: If `from` is `None`, then
implementations should write to the formatter `f` a description of the benchmark and what
inputs it can work with. If `from` is `Some`, then implementation should write the reason
for a successful or unsuccessful match with the enclosed value. Doing these two steps make
error reporting in the event of a dispatch fail much easier for the user to understand and fix.

Refer to implementations within the benchmarking framework for what some of this may look like.

### Adding a Storage Provider

When adding an entirely new storage provider (e.g., a new `DiskANNIndex<DP>` backend), use the
bf_tree implementation (`src/backend/index/bftree/`) as a reference.

#### Files to Create

| File | Purpose |
|------|---------|
| `src/inputs/<provider>.rs` | Input structs (JSON schema), `Display`, `Checker`, `Example` impls |
| `src/backend/index/<provider>/mod.rs` | Module root, `register_benchmarks()`, shared helpers |
| `src/backend/index/<provider>/*.rs` | One file per benchmark variant |
| `example/graph-index-<provider>*.json` | Example configs for each variant |
| `src/main.rs` (test module) | Integration tests for each example JSON |

#### Files to Modify

| File | Change |
|------|--------|
| `Cargo.toml` | Add optional dependency on your provider crate |
| `src/inputs/mod.rs` | Feature-gated `pub(crate) mod <provider>` |
| `src/backend/index/mod.rs` | Feature-gated `mod <provider>` + call `register_benchmarks()` |

#### Checklist

**Input structs:**
- Define input structs with all fields your provider needs
- Consider reusing shared types from `graph_index` where they fit — but only include fields your provider actually uses
- Create separate structs for static vs streaming variants
- Streaming struct includes `DynamicRunbookParams`
- Implement `validate()` for path resolution and sanity checks

**Static benchmark:**
- Implement `Benchmark` trait (see above for the full trait walkthrough)
- `try_match` should reject unsupported configurations early
- Implement `QueryType` for your provider type (associates the vector element type)

**Streaming benchmark:**
- Implement `ManagedStream<T>` on a stream struct:
  - `search` — run KNN with ground truth comparison
  - `insert` / `replace` — insert vectors at given slots
  - `delete` — delete vectors at given slots
  - `maintain` — provider-specific maintenance (cache clearing, consolidation, etc.)
- Wrap in `Managed<T, StreamStats>` (handles slot management, GT translation, maintenance scheduling)
- Implement the `Benchmark` trait for the streaming entry point

**Registration:**
- Choose descriptive tag strings (e.g., `graph-index-<provider>-full-precision-f32`)
- Feature-gate with `#[cfg(feature = "...")]`

**Integration tests:**
- Add an integration test for each example JSON in `src/main.rs` (under `#[cfg(test)]`)
- Feature-gate with `#[cfg(feature = "...")]` matching the provider feature
- Use the `run_integration_test` helper (see existing tests for the pattern)

#### Notes

- `Managed` triggers `maintain()` based on the BigANN runbook's explicit consolidate operations
- `StreamStats` has variants for each operation type
- Matching on literals to dispatch const-generic parameters (e.g., `num_bits`) is fine — it
  effectively dispatches to const generics while keeping the `Benchmark` impl monomorphic
- Check IR growth with `cargo llvm-lines --package diskann-benchmark --all-features --release`;
  each new `DiskANNIndex<DP>` instantiation adds ~150-300K IR lines

