# Indexed Vector Search

`DiskIndexSearcher::search_with_indexed_vectors` returns each valid ANN hit with the native vector stored in the graph and used for exact scoring. This vector may differ from the original input vector and is not a PQ code.

The implementation is in [`src/search/provider/disk_provider.rs`](src/search/provider/disk_provider.rs). The existing `search()` API is unchanged and does not capture or fetch result vectors.

## API

```rust
pub struct SearchResultItemWithIndexedVector<A, V> {
    pub vertex_id: u32,
    pub data: A,
    pub distance: f32,
    pub indexed_vector: Box<[V]>,
}

pub struct SearchResultWithIndexedVectors<A, V> {
    pub results: Vec<SearchResultItemWithIndexedVector<A, V>>,
    pub stats: SearchResultStats,
}
```

Each item owns its vector, so IDs, metadata, distances, and vectors cannot become positionally misaligned. The output contains valid results only:

```text
results.len() == stats.result_count
```

`indexed_vector.len()` is the graph's stored element count, which may differ from the logical dimension for quantized representations such as MinMax8.

## Capture behavior

The request-local collector uses the first-version design:

```rust
HashMap<u32, Box<[T]>>
```

It is separate from pooled search scratch. Exact-distance caching and indexed-vector caching are independent.

| Search mode | Capture policy | Maximum retained vector payload |
|---|---|---|
| Graph / InlineFilter | Capture during traversal and finalization | `L x vector_size` |
| FlatScan / DiverseGraph | Capture final winners only | `K x vector_size` |

`K <= L` is validated before capture begins. Graph and InlineFilter quietly stop traversal capture after L distinct vectors; search continues normally.

After reranking, the collector removes non-winners. Missing winner vectors are then:

1. reused from the provider's current batch when available;
2. otherwise loaded together in one final graph batch.

A final fetch emits an `info!` event with counts only:

```text
uncached_vertices=<candidate exact-distance misses>
final_fetch_vertices=<winner vectors loaded by the final batch>
```

## Ownership and allocation

Each captured vector is copied once into an owned `Box<[T]>`. Final assembly removes that Box from the collector and moves it into the result item without copying its elements again.

This design is intentionally simple, but Graph and InlineFilter may perform up to L per-vector allocations per request. FlatScan and DiverseGraph may perform up to K.

## Integration notes

- Iterate over `result.results`; every item contains its matching vector.
- Convert `Data::VectorDataType` to an explicit stable encoding before FFI or wire serialization.
- Carry both stored element count and logical dimension when they differ.
- Do not use Rust type names as a wire contract.
- The legacy `search()` path retains its existing IDs, padding behavior, and allocation profile.

## Benchmark integration

Disk search inputs select the public API and opt in to boundary metrics independently:

```json
{
  "search_phase": {
    "search_api": "indexed-vectors",
    "collect_api_metrics": true
  }
}
```

`search_api` accepts `legacy` and `indexed-vectors` and defaults to `legacy`. `collect_api_metrics` defaults to `false`. With both defaults, existing JSON uses the legacy hot loop without per-query metric arrays, clocks, or result `black_box` calls. The selected API is emitted in result JSON and the text summary.

When collection is enabled, these optional output fields are present:

- `mean_public_api_call_latency_us`
- `p95_public_api_call_latency_us`
- `p999_public_api_call_latency_us`
- `mean_returned_vector_payload_bytes`
- `max_returned_vector_payload_bytes`

Call latency covers only `search()` or `search_with_indexed_vectors()`; the clock stops immediately when the API returns. Copying IDs and distances, observing each returned vector slice once through `black_box`, computing payload bytes, and dropping results are outside this latency. Batch QPS still covers the complete per-query closures.

Returned payload bytes are exact: the sum of `indexed_vector.len() * size_of::<V>()` for the returned vectors. They exclude `Box`, `Vec`, result-item, allocator, and serialization overhead. Collected legacy runs report a payload of zero because that API returns no vectors. When collection is disabled, all five fields are absent rather than reported as zero.

The paired examples are `diskann-benchmark/example/disk-index-api-call-metrics-legacy.json` and `diskann-benchmark/example/disk-index-api-call-metrics-indexed-vectors.json`. They are identical except for `search_api`, use the checked-in load fixture, and measure one L. Run them from the repository root: the load path is repository-relative and is not resolved through `search_directories`.

### Comparison method

1. Use a Load-only job, a fresh process, and one L per process. Keep the index, queries, K, L, beam width, threads, filters, cache settings, and distance identical.
2. Repeat in ABBA order (`legacy`, `indexed-vectors`, `indexed-vectors`, `legacy`) and report every repeat.
3. Compare optional public API call latency and exact returned payload bytes. Do not add these diagnostics to the existing regression tolerance gate.
4. Disable per-query info logging. The indexed-vector final-fetch event is diagnostic and can materially distort latency.
5. For memory, the existing `PerfLogger` peak is an OS process-lifetime peak, not returned-vector allocation or per-L growth. Use it only with the Load-only, fresh-process, one-L protocol. For independent process peak measurements, use Windows Performance Monitor (PerfMon) or `/usr/bin/time -v` on Linux. Neither replaces the exact payload-byte metric.

## Optional future optimization

If profiling shows per-vector allocation is a bottleneck, the collector could use:

```text
HashMap<vertex_id, slot> + flat Vec<T>
```

Queue-aware eviction could further keep only vectors that remain candidates. This can reduce allocations and improve locality, but requires slot management, row compaction, a final `K x D` copy, and a more complex result API. The current first version deliberately uses `Box<[T]>` until benchmarks justify that complexity.
