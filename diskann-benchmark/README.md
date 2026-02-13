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
    async-index-build
    async-index-build-pq
```
To obtain the JSON schema for an input, add its name to the query like
```sh
cargo run --release --package diskann-benchmark -- inputs async-index-build
```
which will generate something like
```json
{
  "type": "async_index_build",
  "content": {
    "data_type": "float32",
    "data": "path/to/data",
    "distance": "squared_l2",
    "max_degree": 32,
    "l_build": 50,
    "alpha": 1.2,
    "backedge_ratio": 1.0,
    "num_threads": 1,
    "num_start_points": 10,
    "num_insert_attempts": 2,
    "saturate_inserts": true,
    "multi_insert": {
      "batch_size": 128,
      "batch_parallelism": 32
    },
    "search_phase": {
      "search-type": "topk",
      "queries": "path/to/queries",
      "groundtruth": "path/to/groundtruth",
      "reps": 5,
      "num_threads": [
        1,
        2,
        4,
        8
      ],
      "runs": [
        {
          "enhanced_metrics": false,
          "search_k": 10,
          "search_l": [
            10,
            20,
            30,
            40
          ],
          "target_recall": [
                {
                  "target": [50, 90, 95, 98, 99],
                  "percentile": [0.5, 0.75, 0.9, 0.95],
                  "max_search_l": 1000,
                  "calibration_size": 1000,
                },
          ],
          "recall_n": 10
        }
      ]
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
    async-full-precision-f32: tag: "async-index-build", float32
    async-full-precision-f16: tag: "async-index-build", float16
    async-pq-f32: tag: "async-index-build-pq", float32
```
The keyword after "tag" corresponds to the type of input that the benchmark accepts.

### Running Benchmarks

Benchmarks are run with
```sh
cargo run --release --package diskann-benchmark -- run --input-file ./diskann-benchmark/example/async.json --output-file output.json
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
cargo run --release --package diskann-benchmark -- run --input-file ./diskann-benchmark/example/async-dynamic.json --output-file dynamic-output.json
```
Note in the example json that the benchmark is registered under `async-dynamic-index-run`,
instead of `async-index-build` etc..

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

The benchmarking infrastructure uses a loosely-coupled method for dispatching benchmarks
broken into the front end (inputs) and the back end (benchmarks). Inputs can be any `serde`
compatible type. Input registration happens by registering types implementing
`diskann_benchmark_runner::Input` with the `diskann_benchmark_runner::registry::Inputs`
registry. This is done in `inputs::register_inputs`. At run time, the front end will discover
benchmarks in the input JSON file and use the tag string in the "contents" field to select
the correct input deserializer.

Benchmarks need to be registered with `diskann_benchmark_runner::registry::Benchmarks` by
registering themselves in `benchmark::backend::load()`. To be discoverable by the front-end
input, a `DispatchRule` from the `dispatcher` crate (via
`diskann_benchmark_runner::dispatcher`) needs to be defined matching a back-end type to
`diskann_benchmark_runner::Any`. The dynamic type in the `Any` will come from one of the
registered `diskann_benchmark_runner::Inputs`.

The rule can be as simple as checking a down cast or as complicated such as lifting run-time
information to the type/compile time realm, as is done for the async index tests for the data
type.

Once this is complete, the benchmark will be reachable by its input and can live peacefully
with the other benchmarks.

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
We need to implement a few traits related to this input type:

* `diskann_benchmark_runner::Input`: A type-name for this input that is used to identify it for
  deserialization and benchmark matching. To make this easier, `benchmark` defines
  `benchmark::inputs::Input` that can be used to express type level implementation (shown
  below)

* `CheckDeserialization`: This trait performs post-deserialization invariant checking.
  In the context of the `ComputeGroundTruth` type, we use this to check that the input
  files are valid.

```rust
impl diskann_benchmark_runner::Input for crate::inputs::Input<ComputeGroundTruth> {
    // This gets associated with the JSON representation returned by `example` and at run
    // time, inputs tagged with this value will be given to `try_deserialize`.
    fn tag() -> &'static str {
        "compute_groundtruth"
    }

    // Attempt to deserialize `Self` from raw JSON.
    //
    // Implementos can assume that `serialized` looks similar in structure to what is
    // returned by `example`.
    fn try_deserialize(
        &self,
        serialized: &serde_json::Value,
        checker: &mut diskann_benchmark_runner::Checker,
    ) -> anyhow::Result<diskann_benchmark_runner::Any> {
        checker.any(ComputeGroundTruth::deserialize(serialized)?)
    }

    // Return a serialized representation of `self` to help users create an input file.
    fn example() -> anyhow::Result<serde_json::Value> {
        serde_json::to_value(Self {
            data_type: DataType::Float32,
            data: InputFile::new("path/to/data"),
            queries: InputFile::new("path/to/queries"),
            num_nearest_neighbors: 100,
        })
    }
}

impl CheckDeserialization for ComputeGroundTruth {
    fn check_deserialization(&mut self, checker: &mut Checker) -> Result<(), anyhow::Error> {
        // Forward the deserializaiton check to the input files.
        self.data.check_deserialization(checkt)?;
        self.queries.check_deserialization(checkt)?;
        Ok(())
    }
}
```

#### Front End Registration

With the new input type ready, we can register it with the
`diskann_benchmark_runner::registry::Inputs` registry. This can be as simple as:
```rust
fn register(registry: &mut diskann_benchmark_runner::registry::Inputs) -> anyhow::Result<()> {
    registry.register(crate::inputs::Input::<ComputeGroundTruth>::new())
}
```
Note that registration can fail if multiple inputs have the same`tag`.

When these steps are completed, our new input will be available using
```sh
cargo run --release --package diskann-benchmark -- inputs
```
and
```sh
cargo run --release --package diskann-benchmark -- inputs compute-groundtruth
```
will display an example JSON input for our type.

#### Back End Benchmarks

So far, we have created a new input type and registered it with the front end, but there are
not any benchmarks that use this type. To implement new benchmarks, we need register them
with the `diskann_benchmark_runner::registry::Benchmarks` returned from
`benchmark::backend::load()`. The simplest thing we can do is something like this:
```rust
use diskann_benchmark_runner::{
    dispatcher::{DispatchRule, MatchScore, FailureScore, Ref},
    Any, Checkpoint, Output
};

// Allows the dispatcher to try to match a value with type `CentralDispatch` to the receiver
// type `ComputeGroundTruth`.
impl<'a> DispatchRule<&'a Any> for &'a ComputeGroundTruth {
    type Error = anyhow::Error;

    // Will return `Ok` if the dynamic type in `Any` matches
    //
    // Otherwise, returns a failure.
    fn try_match(from: &&'a Any) -> Result<MatchScore, FailureScore> {
        from.try_match::<ComputeGroundTruth, Self>(from)
    }

    // Will return `Ok` if `from`'s active variant is `ComputeGroundTruth`.
    //
    // This just forms a reference to the contained value and should only be called if
    // `try_match` is successful.
    fn convert(from: &'a Any) -> Result<Self, Self::Error> {
        from.convert::<ComputeGroundTruth, Self>(from)
    }
}

fn register(benchmarks: &mut diskann_benchmark_runner::registry::Bencymarks) {
    benchmarks.register::<Ref<ComputeGroundTruth>>(
        "compute-groundtruth",
        |input: &ComputeGroundTruth, checkpoint: Checkpoint<'_>, output: &mut dyn Output| {
            // Run the benchmark
        }
    )
}
```

What happening here is that the implementation of `DispatchRule` provides a valid conversion
from `&Any` to `&ComputeGroundTruth`, which is only applicable if runtime value in the `Any`
is the `ComputeGroundTruth` struct. If this happens, the benchmarking infrastructure will
call the closure passed to `benchmarks.register()` after calling `DispatchRule::convert()`
on the `Any`. This mechanism allows multiple backend benchmarks to exist and pull input from
the deserialized inputs present in the current run.

There are three more things to note about closures (benchmarks) that get registered with the dispatcher:

1. The argument `checkpoint: diskann_benchmark_runner::Checkpoint<'_>` allows long-running
   benchmarks to periodically save incremental results to file by calling the `.checkpoint`
   method. Benchmark results are anything that implements `serde::Serialize`. This function
   creates a new snapshot every time it is invoked, so benchmarks to not need to worry about
   redundant data.

2. The argument `output: &mut dyn diskann_benchmark_runner::Output` is a dynamic type where
   all output should be written too. Additionally, it provides a
   [`ProgressDrawTarget`](https://docs.rs/indicatif/latest/indicatif/struct.ProgressDrawTarget.html)
   for use with [indicatif](https://docs.rs/indicatif/latest/indicatif/index.html) progress bars.
   This supports output redirection for integration tests and piping to files.

3. The return type from the closure should be `anyhow::Result<serde_json::Value>`. This
   contains all data collected from the benchmark and will be collected and saved along with
   all other runs. Benchmark implementations do not need to worry about saving their input
   as well as this is automatically handled by the benchmarking infrastructure.

With the benchmark registered, that is all that is needed.

#### Expanding `DispatchRule`

The functionality offered by `DispatchRule` is much more powerful than what was described in
the simple example. In particular, careful implementation will allow your benchmarks to be
more easily discoverable from the command-line and can also assist in debugging by providing
"near misses".

**Fine Grained Matching**

The method `DispatchRule::try_match` returns both a successful `MatchScore` and an
unsuccessful `FailureScore`. The dispatcher will only invoke methods where all arguments
return successful `MatchScores`. Additionally, it will call the method with the "best"
overall score, determined by lexicographic ordering. So, you can make some registered
benchmarks "better fits" for inputs returning a better match score.

When the dispatcher cannot find any matching method for an input, it begins a process of
finding the "nearest misses" by inspecting and ranking methods based on their `FailureScore`.
Benchmarks can opt-in to this process by returning meaning `FailureScores` when an input is
close, but not quite right.

**Benchmark Description and Failure Description**

The trait `DispatchRule` has another method:
```rust
fn description(f: &mut std::fmt::Formatter<'_>, from: Option<&&'a Any>);
```
This is used for self-documenting the dispatch rule: If `from` is `None`, then
implementations should write to the formatter `f` a description of the benchmark and what
inputs it can work with. If `from` is `Some`, then implementation should write the reason
for a successful or unsuccessful match with the enclosed value. Doing these two steps make
error reporting in the event of a dispatch fail much easier for the user to understand and fix.

Refer to implementations within the benchmarking framework for what some of this may look like.

## Autotuner Tool

The `autotuner` tool builds on top of the benchmark framework to automatically sweep over parameter combinations and identify the best configuration based on optimization criteria (QPS, latency, or recall).

The autotuner uses a **path-based configuration system** that doesn't hardcode JSON structure, making it robust to changes in the benchmark framework. You specify which parameters to sweep by providing JSON paths.

See [diskann-tools/AUTOTUNER.md](../diskann-tools/AUTOTUNER.md) for detailed documentation.

### Quick Start

```sh
# Generate an example sweep configuration
cargo run --release --package diskann-tools --bin autotuner -- example --output sweep_config.json

# Run parameter sweep to find optimal configuration
cargo run --release --package diskann-tools --bin autotuner -- sweep \
  --base-config base_config.json \
  --sweep-config sweep_config.json \
  --output-dir ./autotuner_results \
  --criterion qps \
  --target-recall 0.95
```

The sweep configuration uses JSON paths to specify parameters:
```json
{
  "parameters": [
    {"path": "jobs.0.content.source.max_degree", "values": [16, 32, 64]},
    {"path": "jobs.0.content.source.l_build", "values": [50, 75, 100]}
  ]
}
```

This design makes the autotuner adaptable to any benchmark configuration format without requiring code changes when the benchmark framework evolves.

