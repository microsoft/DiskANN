# PiPNN graph builder

`diskann-pipnn` implements Pick-in-Partitions Nearest Neighbors graph construction. It
partitions vectors with randomized ball carving, builds local k-NN edges, merges them
with HashPrune or exact deduplication, and optionally runs DiskANN's shared Vamana
RobustPrune implementation.

## Responsibility boundary

The crate's public build boundary is `builder::build_typed`. It receives a dense matrix
and produces adjacency lists. Every row is an ordinary graph node
to the PiPNN core; the core does not select medoids, create frozen points, search the
graph, or serialize it.

The existing DiskANN wrappers own those responsibilities:

- In-memory builds create start/frozen points through the same provider and
  `StartPointStrategy` path used by Vamana, then include those rows in the PiPNN build.
- Disk builds select a medoid, append its vector as the frozen row, and include it in the
  PiPNN build. When writing the disk graph, the wrapper omits the frozen row, remaps its
  edges to the real medoid ID, and writes that real ID in the header, matching Vamana.

PiPNN graph construction currently supports full-precision build data only. Disk-search
PQ generation remains independent and is controlled by `num_pq_chunks`.
PiPNN disk builds run in one pass and do not read or write build checkpoint state.

## Benchmark runner

Enable `pipnn` and choose PiPNN with the standard benchmark runner's
`build_algorithm` field. The same field works for graph-index and disk-index builds.

```json
{
  "build_algorithm": {
    "algorithm": "PiPNN",
    "num_hash_planes": 14,
    "c_max": 512,
    "c_min": 64,
    "p_samp": 0.01,
    "fanout": [10, 3],
    "k": 2,
    "replicas": 1,
    "l_max": 72,
    "final_prune": true,
    "skip_hash_prune": false
  }
}
```

Final-prune `alpha` comes from the enclosing graph configuration, alongside
`max_degree`, `l_build`, and the distance metric. The same graph parameters are
used if a disk build falls back to Vamana.

Set `skip_hash_prune` to `true` to accumulate exact-deduplicated leaf candidates
directly and pass them to the same shared RobustPrune kernel. This mode requires
`final_prune: true`; `num_hash_planes` and `l_max` may then be zero.

For a disk build, `build_ram_limit_gb` is required. The wrapper estimates the
one-shot PiPNN peak from the dataset shape/type and PiPNN configuration; if it
exceeds the limit, the existing full-precision Vamana path is selected. Omit the
unsupported PiPNN build `quantization_type`. Run an input derived from the existing
`diskann-benchmark/example/graph-index.json` or `disk-index.json`:

```bash
cargo run --release -p diskann-benchmark --features disk-index,pipnn -- \
  run --input-file input.json --output-file output.json
```

## Core build benchmark

The remaining example measures only the PiPNN core and intentionally excludes provider
loading, frozen-point handling, persistence, and search. Use `/usr/bin/time -v` for peak
RSS; its own `RSS_post` value is only the resident set after the build.

```bash
cargo build --release -p diskann-pipnn --example inmem_build_bench
/usr/bin/time -v target/release/examples/inmem_build_bench DATA.fbin [NPOINTS] [skip-hash-prune]
```

## Tests

```bash
cargo test -p diskann-pipnn --lib
cargo test -p diskann-benchmark --features disk-index,pipnn pipnn_cached_cli_fixtures
```
