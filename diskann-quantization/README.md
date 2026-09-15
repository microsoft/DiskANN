# Quantization

Refer to the crate level documentation for usage of the `quantization` crate.

## Reducing late-interaction multi-vectors

`algorithms::hac::spherical_ward` accepts a `diskann_utils::views::MatrixView<f32>`
and returns a matrix of at most the requested number of unit-normalized centroids.
It is intended for roughly **250–2,000 vectors per document**, whereas k-means is
suited to training sets of up to roughly **100,000 vectors**. These are intended
workload scales, not enforced limits.

This is size-weighted **spherical Ward-style HAC**, not classical Euclidean Ward
linkage. Centroids are renormalized after every merge. Exact global-minimum selection
uses a packed quadratic cost cache (about 7.6 MiB for 2,000 vectors); it does not use
a nearest-neighbor-chain approximation. See the function's rustdoc for numerical
behavior, errors, tie-breaking, and worst-case complexity. Calls are sequential;
parallelize across documents with a bounded pool when appropriate.

A deterministic synthetic compute-only benchmark includes input copying, normalization,
clustering, and output allocation, but excludes input generation and I/O:

```shell
cargo test -p diskann-quantization --release algorithms::hac::tests::benchmark -- --ignored --exact --nocapture
```

Use `HAC_BENCH_ROWS`, `HAC_BENCH_DIM`, `HAC_BENCH_K`, and `HAC_BENCH_REPEATS`
to change the defaults (1,250 rows, 128 dimensions, 32 centers, 3 timed runs).
`HAC_BENCH_REFERENCE=1` measures uncached full recomputation with the same numerical
kernels and checks that the results agree before timing. Report CPU, build flags,
and dimensions alongside measurements; synthetic timing does not establish retrieval
quality or throughput on real documents.

## Generating FlatBuffers bindings

The Rust bindings generated from the files in `schemas` are checked into
`src/flatbuffers`. Regenerate them after changing a schema:

```shell
./tools/generate-flatbuffers.sh
```

The script finds `flatc` on `PATH` by default. Pass a path to use another executable:

```shell
./tools/generate-flatbuffers.sh /path/to/flatc
```

Use `flatc` version 25.2.10, available from the
[official release page](https://github.com/google/flatbuffers/releases/tag/v25.2.10).

SHA-512 sums of the zip directories for `v25.2.10` are as follows.

* 6a20c2fc4e4e094574a0fd064f79a374eb9e6abba9e49d4543ec384b056725f6ca9f7823ba5952fcfa40e31a56a4e25baa659415d94edd69a7a978942577c579  Linux.flatc.binary.clang++-18.zip
* 8784aae9f7984fdf5685e3944787bc547ca3a8bccefa4ba33efbe73960ebb6c94c2061d251dcc00e683133e65c8f47833195e0293415bc8abbd7b5aab4419714  Windows.flatc.binary.zip
