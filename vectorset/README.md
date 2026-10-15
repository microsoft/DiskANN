# vectorset

Garnet client for benchmarking vector set workloads.

Since Garnet speaks Redis's RESP protocol, this uses the official Redis Rust
client and can additionally be used to run workloads on Redis. For maximum
performance, it uses multiple threads and pipelining.

Currently it supports an ingestion workload which inserts vectors as fast as possible, and a search workload which queries a vectorset as fast as possible while calculating the recall.

## Element Types, Metrics & Quantizers

Garnet supports additional element types, distance metrics, and quantizers than exist in Redis.  Redis vector sets only support 32-bit float elements, cosine distance, and the NOQUANT (full precision), BIN (binary), and Q8 (scalar 8-bit) quantizers. All Garnet extensions to vectorsets are prefixed with `X` (e.g. `XI8` element type).

### Additional Element Types

Garnet supports signed 8-bit (XI8) and unsigned 8-bit (XU8) vector elements.

### Additional Metrics

Garnet supports specifying the distance metric with `XDISTANCE_METRIC <metric>`. The default is `L2`, and the following metrics are available:

- `COSINE`
- `XCOSINE_NORMALIZED`
- `IP` (inner product)
- `L2` (euclidean)

### Additional Quantizers

Redis supports `NOQUANT`, `BIN`, and `Q8` quantizers for 32-bit float elements. These are supported in Garnet, but use more advanced quantizers than Redis for better performance. `BIN` uses the spherical 1-bit quantizer from DiskANN, and `Q8` uses the minmax 8-bit quantizer from DiskANN.

For `XI8` and `XU8` vectors, the full precision quantizers are `XNOQUANT_I8` and `XNOQUANT_U8` respectively. The spherical 1-bit quantizer is also available via `XBIN_I8` and `XBIN_U8`.

## Usage

You will need a dataset to import in bin format. You can find these on the
internet, but the easiest way to get some is to use [Big ANN
Benchmarks](https://github.com/harsha-simhadri/big-ann-benchmarks)'s
`create_dataset.py` to download datasets. Once you have a dataset and Garnet is
running, copy config.toml.example to config.toml and modify as necessary.

Since vectorset has its own workspace in the DiskANN repo, the following
examples should be run from the `vectorset` directory in the repo. Running
`cargo run --release -- --help` and `cargo run --release -- <subcommand> --help`
will enumerate all the various subcommands and arguments.

### Ingest

The `ingest` subcommand will ingest vectors and build the index. The following example builds a full precision only index on the wikipedia-10M dataset from big-ann-benchmarks.

`cargo run --release -- --config path/to/config.toml --quantizer no-quant ingest --tasks 32 --degree 48 --l-build 256 --metric inner-product path/to/data/wikipedia_cohere/wikipedia_base.bin.crop_nb_10000000`

### Query

The `query` subcommand queries the database while calculating recall against precomputed ground truth. By default, it will measure 10-recall@10. The following example queries the index created above and reports queries per second (QPS) and recall results.

`cargo run --release -- --config path/to/config.toml --quantizer no-quant query --tasks 32 -k 100 -n 100 --l-search 192 path/to/data/wikipedia_cohere/wikipedia_query.bin path/to/data/wikipedia_cohere/wikipedia-10M`

### Run

The `run` subcommand executes [Big ANN
Benchmark](https://github.com/harsha-simhadri/big-ann-benchmarks) style
runbooks. It requires a dataset catalog, describing metadata about available
datasets. You can find an example in the `datasets` directory. Aside from
console output, it will write reports to disk in the `reports` directory by
default.

Unlike the `query` command, searches executed from runbooks are not pipelined in
order to get more accurate latency metrics.

The following example command executes a runbook:

`cargo run --release -- --config path/to/config.toml --quantizer no-quant --threads 8 run --tasks 8 --degree 48 --l-build 300 --l-search 100 path/to/runbook.yaml path/to/datasets`
