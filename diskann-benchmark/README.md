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

### Graph-IVF

Graph-IVF (see [`diskann-graphivf`](../diskann-graphivf/README.md)) is behind the
off-by-default `graph-ivf` feature, so a build that does not want it does not pay for it:
```sh
cargo run --release --package diskann-benchmark --features graph-ivf -- \
    run --input-file ./diskann-benchmark/example/graph-ivf-build-static.json \
        --output-file output.json
```
Without the feature the `graph-ivf` input kind is still parsed and validated — only the
backends (`graph-ivf-f32`, `-f16`, `-u8`, `-i8`, `-minmax8`) are absent, so a config that
uses it fails at benchmark-matching rather than at deserialization.

A job is one `source` (how the index comes to exist) plus an optional final
`search_phase`. The four sources are tagged by `graph-ivf-source`; the tag matches the
`build_kind` reported in the output, so a config and its results name the same builder:

| `graph-ivf-source` | What it does | Key fields |
| --- | --- | --- |
| `Static` | Batch build: fit `k` centroids by k-means over a corpus sample, then assign every point. | `num_clusters`, `sample_size`, `kmeans_iters`, `assign_method`, `empty_clusters`, `save_path` |
| `Online` | Streaming build: insert points, splitting a cluster whenever it overflows. The cluster count emerges from the data. | `split_threshold`, `batch_size`, `max_clusters`, `reassign_neighbors`, `routing`, `normalize`, `save_path`, `telemetry_csv` |
| `OnlineRunbook` | Replay BigANN insert/delete/search stages against a live online index, then flush it once. | nested `build`, `runbook`, and `search` objects; see below |
| `Load` | Search an index built by an earlier job. | `load_path` |

`batch_size` (default `1`) controls how many points an `Online` build consumes at
a time. There is one write path and a single insert is a batch of one, so `1` is
the reference semantics: route a point, split its cluster if it overflowed. A
larger value — a few thousand matches how a real writer arrives — defers splitting
to the end of each batch, which lets the batch be routed in parallel and lets
every cluster that overflowed be re-clustered by one joint k-means instead of one
bisection at a time. That last part changes the partition, so compare recall
before switching. See
[`diskann-graphivf/ONLINE.md`](../diskann-graphivf/ONLINE.md#3b-batched-inserts).

Static, online, and load sources take `data_type` (`float32` \| `float16` \| `uint8` \|
`int8` \| `minmax8`), which is the on-disk element type of the inverted lists and selects
the backend — a `Load` job must name the same type the index was built with. An
`OnlineRunbook` puts the same online build fields inside its `build` object. Static and
online builds additionally take the corpus (`data`, `dim`, `distance`) and a `routing`
block.

`routing` selects how the index finds nearest centroids, and carries only the knobs that
mode uses:

```jsonc
// Navigate a graph over the centroids. Every key is optional.
"routing": {
  "graph": {
    "assign_l": 64,      // beam for routing a point to its cluster
    "reassign_l": 64,    // online only; beam for split/merge neighbor selection
    "graph_degree": 32,
    "graph_slack": 1.2,
    "graph_l_build": 64,
    "graph_alpha": 1.2
  }
}

// Score every live centroid with a batched matrix multiply. Exact, linear in the
// cluster count, and takes no parameters — supplying one is a parse error.
"routing": "exact"
```

Omitting `reassign_l` resolves it to `max(reassign_neighbors, assign_l)`, and validation
writes the effective value back so the job's serialized input records it.
`reassign_neighbors` is a candidate *count* rather than a beam width, so it stays outside
`routing` and applies under either mode.

The static and online build schemas are deliberately disjoint and use
`deny_unknown_fields`: a
config that mixes k-means knobs into an online build is a hard error rather than a set of
silently ignored keys. `save_path` and `load_path` are index *prefixes* — the
`.graphivf_meta`, `.graphivf_lists` and `.graphivf_centroids.fbin` suffixes are added by
the backend. A relative `save_path` is resolved against the working directory, not the
runner's `output_directory`; setting `output_directory` alongside a graph-IVF build is
rejected rather than silently ignored.

The search phase sweeps `cluster_fractions`, running one search per requested share of
the index's clusters:
```json
"search_phase": {
  "queries": "disk_index_sample_query_10pts.fbin",
  "groundtruth": "disk_index_10pts_idx_uint32_truth_search_res.bin",
  "num_threads": 1,
  "cluster_fractions": [0.0625, 0.125, 0.25, 0.5, 1.0],
  "centroid_search_alpha": 4.0,
  "recall_at": [10, 100],
  "distance": "squared_l2"
}
```
Each fraction must be finite and in `(0.0, 1.0]`. For an index with `C` clusters the
benchmark passes `ceil(fraction * C)` as the library-level `nlist`, so every positive
fraction probes at least one cluster and `1.0` is exhaustive. An `OnlineRunbook`
recomputes that value from the current live cluster count at every search stage; inserts,
deletes, splits, and dissolves therefore do not change the requested share of clusters.
Results retain both `cluster_fraction` and the effective concrete `nlist`.

`centroid_search_alpha` (optional, default `4.0`, must be `>= 1.0`) sizes the centroid
graph beam as `max(128, ceil(alpha * nlist))`. Because `nlist` is derived per fraction and
per stage, the beam follows the sweep instead of being pinned to whatever the widest
fraction or largest index will eventually need. Set
`"measure_centroid_recall": true` on a runbook search to score selection against an exact
scan of every live centroid; it costs a full centroid pass per query and is measured
outside the reported latency. Alpha is the parameter recall is most sensitive to — at 1.5
only ~63% of the probed clusters are genuinely the nearest, against ~98% at 4.0.

An online runbook source has this shape:

```json
"source": {
  "graph-ivf-source": "OnlineRunbook",
  "build": {
    "data_type": "float32",
    "data": "corpus.fbin",
    "distance": "squared_l2",
    "dim": 100,
    "split_threshold": 120,
    "merge_threshold": 40,
    "reassign_neighbors": 10,
    "routing": { "graph": { "graph_degree": 32, "graph_l_build": 64 } },
    "num_threads": 16,
    "seed": 0,
    "save_path": "/absolute/path/to/output-prefix"
  },
  "runbook": {
    "runbook_path": "runbooks/final_runbook.yaml",
    "dataset_name": "dataset-key-in-runbook",
    "gt_directory": "groundtruth/final_runbook.yaml"
  },
  "search": {
    "queries": "queries.fbin",
    "cluster_fractions": [0.01, 0.05, 0.10, 0.15],
    "centroid_search_alpha": 4.0,
    "recall_at": [50]
  }
}
```

`batch_size` sub-batches each runbook insert/delete range; it does not alter stage
boundaries. Search runs only at explicit runbook search stages. `gt_directory` must contain
exactly one `step<stage>.gt<depth>` file for every search stage. Replace stages are rejected
because graph-IVF external ids are corpus row ids and cannot be remapped in place. The
runbook result records every mutation and search stage; split and merge CSVs are written
after replay and the final index flush succeeds.

Each sweep reports recall, QPS, mean/p95/p999 latency, bytes read and IOs per query, plus
a per-stage latency breakdown (preprocess, centroid search, plan I/O, disk read, score,
top-k). `recall_at` takes a single `k` or a list of them: a sweep searches once to the
largest and scores every listed `k` from that one result set, so comparing recall@50 and
recall@1000 costs one job rather than two. Online builds can also write `telemetry_csv`,
one row per split, which is a complete timeline of cluster growth and split cost.

Four constraints are worth calling out because they are checked up front:

- `cluster_fractions` must contain at least one value, all in `(0.0, 1.0]`.
- The groundtruth must carry at least the largest `recall_at` neighbors per query, since
  scoring deeper than it reaches would silently read another query's row.
- Online builds store corpus rows verbatim and cannot normalize them, so `cosine` is
  rejected — pre-normalize the corpus and use `cosine_normalized`.
- For `minmax8` indexes the corpus and queries must both already be quantized; see
  [`compress_minmax`](../diskann-graphivf/scripts/README.md).

Three runnable examples cover static, online, and load jobs, all against the checked-in
`test_data` corpus so they work from a fresh clone. Online runbooks require a workload with
insert/delete-only stages and matching per-stage groundtruth, so their portable schema is
shown above rather than tied to a machine-local dataset:

| Config | What it shows |
| --- | --- |
| [`example/graph-ivf-build-static.json`](example/graph-ivf-build-static.json) | k-means build + search sweep in one job |
| [`example/graph-ivf-build-online.json`](example/graph-ivf-build-online.json) | streaming build with split telemetry + search sweep |
| [`example/graph-ivf-search.json`](example/graph-ivf-search.json) | re-sweeping an index built by an earlier job |

Run them from the repository root; the two build configs write their index prefix into the
working directory, and `graph-ivf-search.json` loads the one the static config produced, so
run that one first.

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

