# Static Graph-IVF benchmark

This benchmark measures **build, then search** using the new `diskann::ivf`
implementation. There are no deletes, mixed update/query stages, or runbooks.
The default example generates seeded FP32 vectors and independent queries, so
it runs without external datasets or Git LFS fixtures. The metric is squared L2
and both phases run on one thread.

## Run

From the repository root:

```powershell
cargo run --release -p diskann-benchmark -- run --input-file diskann-benchmark/example/graph-ivf.json --output-file target/graph-ivf-baseline.json
```

The example builds 8,192 vectors of dimension 128 three times. Each build is
followed by 128 queries at `nprobe = 1, 4, 16`, with one unmeasured warmup pass
and five measured passes per setting. Increase the dataset and query counts
for performance studies; this small synthetic example verifies the pipeline
and does not represent production recall or throughput.

Set `search` to `null`, or omit it, to measure build only. Search always operates
on the freshly built index; saving/loading an index is not included. Every build
repetition starts with a new provider, graph, and the same random seeds.

## What is measured

The library currently exposes IVF initialization and batched insertion, but no
IVF search API or reusable IVF provider. This benchmark therefore supplies
private adapters under `src/graph_ivf/`; it does not add a library API.

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
| `builds[].centroid_graph_seconds` | Construction of the graph over final live centroids |
| `builds[].total_seconds`, `vectors_per_second` | Both build stages, including allocations |
| `builds[].splits`, `lists`, `empty_lists`, `max_list_size` | Amount of split work and resulting list layout |
| `searches[].qps`, `total_seconds` | One complete sequential query pass, including result collection |
| `searches[].latency_ns` | Per-query minimum, mean, median, p90, p99; includes graph traversal, posting scan, top-k, and query allocations |
| `searches[].recall_at_k` | Unique correct returned IDs divided by `queries * k`; missing results count as misses |
| `searches[].mean_points_scanned`, `scanned_fraction` | Posting vectors scored per query and fraction of the dataset |
| `searches[].mean_lists_scanned`, `mean_centroid_comparisons` | Actual coarse-routing/search work |
| `searches[].mean_results_returned` | Detects query settings that return fewer than k results |

Dataset generation/loading, groundtruth, result validation, and recall
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

## Use real data

Replace the example's `dataset` object with:

```json
{
  "kind": "files",
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
Compare repeated `ivf_seconds` values for clustering/split optimizations; also
check total build time and search quality. Graph construction time is reported
separately so it cannot obscure a clustering improvement.

A faster run is not sufficient evidence by itself. Check inserted count, splits,
list sizes, recall, scan counts, and result counts. Every completed build checks
that each input point belongs to exactly one live list. Recall deduplicates
returned IDs and retains the full k denominator for short results. Unit tests
exercise splitting, full-probe agreement with an independent exhaustive oracle,
partial probing, file input, build-only mode, and invalid inputs.

This is an operation/end-to-end baseline. It does not isolate the time inside
each numerical kernel, and has no automatic before/after regression threshold.
After the library gains an IVF search API and a graph-backed provider, replace
the private adapters while retaining the workload and reporting format.

```powershell
cargo test -p diskann-benchmark graph_ivf
```
