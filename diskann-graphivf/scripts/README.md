<!--
 Copyright (c) Microsoft Corporation.
 Licensed under the MIT license.
-->

# `diskann-graphivf` scripts catalog

> **Building, searching and profiling a graph-IVF index now runs through the shared
> benchmark harness.** See the [graph-IVF section of
> `diskann-benchmark/README.md`](../../diskann-benchmark/README.md#graph-ivf) and the
> ready-made configs in [`diskann-benchmark/example/`](../../diskann-benchmark/example).

What is left here are the tools the harness does not cover: preparing a corpus, and a
centroid-graph quality diagnostic.

The Rust entries are standalone `cargo` examples (binary name == file stem, kept stable
via explicit `[[example]]` targets in [`../Cargo.toml`](../Cargo.toml)):

```text
cargo run --release --example <name> -- <args...>
```

The compiled binaries are also emitted to `target/release/examples/<name>[.exe]` and
can be invoked directly. Both read the DiskANN binary matrix format
(`[npoints u32][ndims u32][row-major data]`).

The Python entries are run directly and need only `numpy`.

## `dataprep/` — corpus & query preparation

| Tool | What it produces |
| --- | --- |
| [`vecprep.py`](dataprep/vecprep.py) | Everything between a pile of embeddings and a runnable benchmark: format inspection, `.npy` conversion, normalization, aligned subsets, degenerate-row filtering, and exact groundtruth **with a built-in audit**. Dataset-agnostic. |
| [`compress_minmax`](dataprep/compress_minmax.rs) | 8-bit MinMax-quantized `.bin` from an `fp16`/`f32` `.bin`. Run on **both** the corpus and the queries before any `minmax8` build or search. |

```text
python dataprep/vecprep.py --help            # subcommands
python dataprep/vecprep.py info corpus_fp16.bin
python dataprep/vecprep.py groundtruth --corpus corpus_fp16.bin \
    --queries queries_fp16.bin --out groundtruth_recall_1000.bin -k 1000
```

`groundtruth` refuses to emit labels it believes are untrustworthy. The check that
matters most is for all-zero corpus rows: under squared L2 a zero vector sits at
distance `||q||^2` from *every* query -- exactly 1.0 for a unit-norm corpus -- so it
lands inside the top-K band and occupies real groundtruth slots, capping achievable
recall at a value no amount of search effort can beat. Use `vecprep.py filter --drop zero`
to remove them, then recompute. `vecprep.py audit` re-runs the check on existing files.

```text
cargo run --release --example compress_minmax -- corpus_f32.bin  corpus_minmax8.bin  f32
cargo run --release --example compress_minmax -- queries_f32.bin queries_minmax8.bin f32
```

## `analysis/` — index-quality diagnostics

| Example | What it isolates |
| --- | --- |
| [`centroid_graph_ablation`](analysis/centroid_graph_ablation.rs) | Centroid-graph recall: the graph's top-`nlist` centroids against the exact nearest ones. This caps end-to-end recall no matter how the lists are scored, so it separates a routing problem from a clustering problem. Reads a build's `*.graphivf_centroids.fbin`. |
