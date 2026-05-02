# vectorset

Garnet client for benchmarking vector set workloads.

Since Garnet speaks Redis's RESP protocol, this uses the offical Redis Rust
client. For maximum performance, it uses multiple threads and pipelining.

## Usage

You will need dataset to import in bin format. You can find these on the
internet, but the easiest way to get some is to use [Big ANN
Benchmarks](https://github.com/harsha-simhadri/big-ann-benchmarks)'s
`create_dataset.py` to download these. Once you have a dataset and Garnet is
running, cp config.toml.example to config.toml and modify as necessary. Then:

`cargo run --release -- --config config.toml --data-type float32 ingest --tasks 32 --degree 48 --l-build 256 /data/wikipedia_cohere/wikipedia_base.bin.crop_nb_10000000`

will ingest vectors and:

`cargo run --release -- --config config.toml --data-type float32 query --tasks 32 -k 100 -n 100 --l-search 192 /data/wikipedia_cohere/wikipedia_query.bin /data/wikipedia_cohere/wikipedia-10M-cosine`

will run static search workloads on them reporting QPS and recall.