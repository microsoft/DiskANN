# Static Graph-IVF benchmark

This benchmark measures **build, then search** using the new `diskann::ivf`
implementation. There are no deletes, mixed update/query stages, or runbooks.
The default example reuses the SIFT small FP32 vectors, queries, and groundtruth
under `test_data/disk_index_search`, matching the other benchmark examples.
The metric is squared L2 and both phases run on one thread.

## Run

From the repository root:

```powershell
git lfs pull
cargo run --release -p diskann-benchmark -- run --input-file diskann-benchmark\example\graph-ivf.json --output-file target\graph-ivf-baseline.json
```

The example builds 256 vectors of dimension 128 three times, inserting batches
of 64 with a split threshold of 64. Each build is followed by 10 queries at
`nprobe = 1, 4, 16`, with one unmeasured warmup pass and five measured passes
per setting. This small example verifies the pipeline and does not represent
production recall or throughput. Use the YFCC workloads below for larger runs.

Set `search` to `null`, or omit it, to measure build only. Search always operates
on the freshly built index; saving/loading an index is not included. Every build
repetition starts with a new provider, graph, and the same random seeds.

## What is measured

The library currently exposes IVF initialization and batched insertion, but no
IVF search API or reusable IVF provider. This benchmark therefore supplies
private adapters under `src/graph_ivf/`. An opt-in library insertion observer
exposes the two split-phase boundaries so every optimization layer uses the
same measurements without adding clock reads to ordinary insertions.

The module entry point only registers the benchmark. Execution orchestration,
build timing, search measurement, data preparation, and result serialization are
separate concerns, following the other index benchmarks. File loading,
percentiles, and recall reuse the existing benchmark utilities; the optional
in-memory groundtruth oracle retains deterministic row-ID tie breaking.

```text
FP32 rows
  -> IVFIndex::initialize + insert_batch
       exact centroid routing + library two-means split/reassignment
       benchmark-local in-memory point/list provider
  -> build a DiskANN graph over the final live centroids
  -> query the centroid graph for up to nprobe lists
  -> flat::knn_search over those lists only -> top-k point IDs
```

The centroid graph is real `DiskANNIndex` search, not exhaustive centroid search
disguised as graph search. During ingestion, however, centroid routing is
exhaustive, and the graph is built **once after all batches**. This is a static
measurement adapter, not the eventual graph-backed online IVF provider. It
includes no SSD I/O, quantization, replication, or Mimir integration.

| Output field | Scope |
| --- | --- |
| `builds[].ivf_seconds` | Seed selection, initialization, input preparation/copying, routing, splitting/reassignment, and publication through the last batch |
| `builds[].two_means_seconds` | Sum of split fitting intervals: centroid/sum scratch allocation, initialization, all Lloyd steps, and centroid updates |
| `builds[].reassignment_seconds` | Sum of all-child assignment intervals: assignment scratch, distance/top-1 selection, counting, and member-list materialization |
| `builds[].other_ivf_seconds` | IVF time minus the two disjoint split phases; includes routing, working-set fill, provider updates/publication, and orchestration |
| `builds[].centroid_graph_seconds` | Construction of the graph over final live centroids |
| `builds[].total_seconds`, `vectors_per_second` | Both build stages, including allocations |
| `builds[].splits`, `lists`, `empty_lists`, `max_list_size` | Amount of split work and resulting list layout |
| `builds[].split_points` | Sum of split-working-set point visits; a point may contribute on multiple insertions |
| `builds[].split_parent_histogram` | Split parents per nonempty insertion mapped to the number of insertions; includes zero-split batches. An insertion with P parents reassigns across 2P children |
| `searches[].qps`, `total_seconds` | One complete sequential query pass, including result collection |
| `searches[].latency_ns` | Per-query minimum, mean, median, p90, p99; includes graph traversal, posting scan, top-k, and query allocations |
| `searches[].recall_at_k` | Unique correct returned IDs divided by `queries * k`; missing results count as misses |
| `searches[].mean_points_scanned`, `scanned_fraction` | Posting vectors scored per query and fraction of the dataset |
| `searches[].mean_lists_scanned`, `mean_centroid_comparisons` | Actual coarse-routing/search work |
| `searches[].mean_results_returned` | Detects query settings that return fewer than k results |

Dataset loading, groundtruth, result validation, and recall
calculation are outside the timers. Output contains individual build and search
repetitions, and the runner records the input configuration alongside results.

`initial_centroids` samples distinct input rows; it is not a full k-means training
phase. Each insertion only splits touched oversized parents once, following the
current library algorithm. `split_threshold` is therefore a trigger, not a
guaranteed maximum final list size.

`effective_nprobe = min(requested_nprobe, live_lists)`. Graph search may return
fewer lists; `mean_lists_scanned` records what was actually scanned. Even asking
for every list does not guarantee that an approximate centroid graph reaches
all lists. There is no hidden exhaustive fallback.

## Use another dataset

Replace the example's `dataset` object with:

```json
{
  "data": "base.fbin",
  "queries": "queries.fbin",
  "groundtruth": "groundtruth.bin"
}
```

Add the data directory to the job file's top-level `search_directories`, or use
absolute paths. Vector files contain two little-endian uint32 values `(rows,
dimensions)`, followed by row-major float32 values. Groundtruth uses the same
header followed by uint32 IDs; any trailing distance section is ignored. IDs
must be zero-based row numbers into this exact base file. At least `k` unique
valid IDs per query are required.

Groundtruth is optional: if omitted, the benchmark computes exhaustive squared-L2
top-k against the input corpus, outside all timers. This costs O(queries *
vectors * dimensions); use a precomputed file for large datasets. Recall is
ID-based, so alternative equal-distance neighbors at the k boundary can count
as misses. Generated groundtruth breaks equal-distance ties by row ID.

## Compare an optimization

Run the same JSON before and after the change, writing different output files.
Keep CPU, compiler, target ISA, thread count, input files, seeds, batch size,
split threshold, Lloyd iterations, graph settings, and search settings fixed.
Compare `two_means_seconds` between the baseline and the two-means layer, and
`reassignment_seconds` between the two-means and reassignment layers. Also retain
`other_ivf_seconds`, `ivf_seconds`, total build time, and search quality so a local
improvement cannot hide an end-to-end regression. The console prints each split
phase separately. These are phase wall times, including each phase's setup and
output work, rather than distance-kernel-only CPU time.

The benchmark measures the real split call through
`IVFIndex::insert_batch_observed`; ordinary `insert_batch` does not read a clock.
An insertion containing one or more split parents adds four clock reads, with no
per-point timers. No-split insertions report zero phase time. `InsertStats` keeps
only deterministic work counts; the benchmark-local observer owns the
nondeterministic durations. `other_ivf_seconds` is a residual, not a separately
profiled routing measurement. Compare work and quality fields separately.

A faster run is not sufficient evidence by itself. Check inserted count, splits,
list sizes, split point visits, split-parent histograms, recall, scan counts, and
result counts. Every completed build checks
that each input point belongs to exactly one live list. Recall deduplicates
returned IDs and retains the full k denominator for short results. Unit tests
exercise splitting, full-probe agreement with an independent exhaustive oracle,
partial probing, file input, build-only mode, and invalid inputs.

This is an operation/end-to-end baseline with split phase breakdowns. It has no
automatic before/after regression threshold.
After the library gains an IVF search API and a graph-backed provider, replace
the private adapters while retaining the workload and reporting format.

```powershell
cargo test -p diskann-benchmark graph_ivf
```

## Batch size for phase comparisons

The [build](perf_test_inputs/graph-ivf-stages-build.json) and
[search](perf_test_inputs/graph-ivf-stages-search.json) workloads reuse YFCC's
10,000 FP32 vectors of dimension 192 from `test_data/yfcc`. The search workload
also uses its 100 queries and precomputed squared-L2 groundtruth. Both use
**4,096** points per insert, 16 initial centroids, split threshold 512, and five
Lloyd steps. Batch size is explicit, not a library default. The SIFT small
example remains a pipeline smoke test. With the pinned seed and fixture, the
three insertions each split multiple parents, exercising reassignment across
more than two children. Tests require that shape and verify that every repeated
build has the same split-work and final-layout signature; they do not pin one
signature across future implementation or ISA changes.

This choice follows the inspected PoC branch `u/adkrishnan/graph-ivf` at
`90cb5cafc75199d407873141b3029610e33f653d`:

- The [online build loop](https://github.com/microsoft/DiskANN/blob/90cb5cafc75199d407873141b3029610e33f653d/diskann-benchmark/src/graph_ivf/online.rs#L241)
  passes chunks of the explicitly required `batch_size` to `insert_batch`.
- The [256-point smoke example](https://github.com/microsoft/DiskANN/blob/90cb5cafc75199d407873141b3029610e33f653d/diskann-benchmark/example/graph-ivf-build-online.json#L16)
  uses 64; the [larger-corpus online example](https://github.com/microsoft/DiskANN/blob/90cb5cafc75199d407873141b3029610e33f653d/diskann-graphivf/ONLINE.md#L113)
  uses 4,096. There is no single hard-coded PoC insert batch size.
- The PoC's static clustering build uses a separate training/assignment path,
  without this online insertion loop.
- Its internal route work units (256 points) and reassignment scratch tiles
  (4,096 points) are separate constants; neither sets the insert batch size.

4,096 is a source-grounded starting workload, not an optimal batch-size claim.
It leaves three insertion boundaries in YFCC 10K, including the final partial
batch. The threshold is chosen to exercise multiple simultaneous split parents,
not as a production tuning recommendation; this remains a small-corpus
measurement rather than a stress test. A change in batch size or threshold can
change partitions and recall; compare each optimization at the **same** settings
and on the same input files.
The PoC and new provider-based algorithm also differ, so matching this parameter
does not make their full algorithms or timings equivalent.

```powershell
cargo run --release -p diskann-benchmark -- run --input-file diskann-benchmark\perf_test_inputs\graph-ivf-stages-build.json --output-file target\graph-ivf-stages-build.json
cargo run --release -p diskann-benchmark -- run --input-file diskann-benchmark\perf_test_inputs\graph-ivf-stages-search.json --output-file target\graph-ivf-stages-search.json
```
