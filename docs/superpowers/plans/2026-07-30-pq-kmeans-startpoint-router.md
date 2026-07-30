# PQ-kmeans Start-point Router Implementation Plan

> For agentic workers: REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

Goal: Optimize DiskANN disk-index start points with no recall regression, tiny extra memory, and lower hops, I/O, and latency.

Architecture: Build a compact router artifact from existing disk-index PQ codes by clustering into k = ceil(sqrt(N)) representative start nodes. At query time, scan only representative PQ codes, choose the closest bounded set of starts, seed DiskANN graph traversal from those starts, and distribute the static BFS cache across the same representative roots.

Tech Stack: Rust 2021, diskann-disk, diskann-benchmark, diskann-benchmark-runner JSON inputs, serde, Cargo tests.

## Global Constraints

- Preserve baseline behavior when start_point_router is absent.
- No recall regression at the target recall_at and search_list settings.
- Extra router memory should stay approximately k * (4 + num_pq_chunks) bytes, where k = ceil(sqrt(N)); this excludes ordinary object/vector overhead.
- Router work must be counted separately from graph traversal in benchmark metrics.
- Do not run large MSTuringANN benchmarks as part of implementation verification; only use the experiment bundle for manual runs.

---

## Files and responsibilities

- diskann-disk/src/search/pq_kmeans_router.rs: PQ-kmeans artifact format, build-time representative selection, query-time route selection, save/load validation.
- diskann-disk/src/search/provider/disk_provider.rs: multi-start graph traversal integration and router statistics.
- diskann-disk/src/search/provider/disk_vertex_provider_factory.rs: BFS cache construction from either the default medoid or multiple representative roots.
- diskann-benchmark/src/inputs/disk.rs: JSON input schema for pq-kmeans-router-build and start_point_router.
- diskann-benchmark/src/disk_index/build.rs: benchmark job that builds the router artifact from an existing disk index.
- diskann-benchmark/src/disk_index/search.rs: benchmark job that loads the router, wires cache roots, emits warmup/repetition/router metrics.
- experiment_bundles/msturingann-10m-l200-r100-pq-kmeans-startpoint-20260730/: reproducible MSTuringANN 10M configs and result/log locations.

## Implementation tasks

### Task 1: Disk router build artifact

- [ ] Add or review the benchmark input type pq-kmeans-router-build with fields load_path, artifact, optional num_representatives, optional training_sample_size, and max_iterations defaulting to 4.
- [ ] Load the existing disk index sidecars from load_path: PQ pivots, compressed PQ codes, and disk graph header/medoid.
- [ ] When num_representatives is omitted, compute k = ceil(sqrt(N)).
- [ ] Run PQ-code k-means for at most max_iterations, seeded deterministically and including the graph medoid as a safe fallback representative.
- [ ] Persist a compact binary artifact containing enough metadata to reject mismatched index/query PQ data and a table of representative_id u32 plus representative PQ code bytes.
- [ ] Emit build stats: build_time, num_points, num_pq_chunks, num_representatives, artifact_bytes.

### Task 2: Query-time PQ-code flat scan

- [ ] Load the router artifact once per search job and validate num_pq_chunks against the index PQ data.
- [ ] For each query, reuse the existing query PQ preprocessing and scan the representative PQ code table, not the full corpus.
- [ ] Select the closest max_start_points representative IDs; reject max_start_points = 0 during input validation.
- [ ] If routing fails or produces no starts, fall back to the original medoid start.
- [ ] Record per-query router_time_us, router_scanned_codes, and routed_start_points_count.

### Task 3: Multi-start DiskANN traversal

- [ ] Replace the single initial start node with a bounded vector of start nodes supplied by the router.
- [ ] Seed the candidate queue with all routed starts, deduplicate start IDs, and preserve the existing visited set semantics.
- [ ] Keep beam_width, search_list, filters, post processors, and flat/graph search mode behavior unchanged outside of start-node initialization.
- [ ] Ensure scratch capacity cannot underflow when routed start count exceeds the PQ scratch maximum; truncate starts deterministically.
- [ ] Preserve baseline medoid traversal when no router is configured.

### Task 4: BFS cache distributed across representative starts

- [ ] When num_nodes_to_cache is set and a router is loaded, seed static BFS cache construction from router representative IDs.
- [ ] Distribute cache slots across representatives with deterministic multi-source BFS; deduplicate nodes before inserting into cache.
- [ ] Keep the legacy single-root BFS cache behavior when no router is loaded.
- [ ] Validate that requested representatives themselves are preferred in the cache when the cache budget is large enough.

### Task 5: Benchmark warmup, repetitions, and router metrics

- [ ] Keep warmup_runs separate from measured repetitions so OS/cache warmup does not pollute reported latency/QPS.
- [ ] Aggregate recall, QPS, mean/p95/p999 latency, mean I/Os, mean comparisons, mean hops, cache hit percentage, mean_router_time, mean_router_scanned_codes, and mean_routed_start_points per search_list value.
- [ ] Add configs for baseline, max_start_points = 16, and max_start_points = 8 with identical search parameters.
- [ ] Compare only after the router artifact is built and the same index/query/groundtruth files are resolved by all runs.

## Testing and verification plan

Use focused tests first, then workspace checks in proportion to the change risk.

- [ ] Format check:

    cargo fmt --all --check

- [ ] Router artifact and routing unit tests:

    cargo test -p diskann-disk pq_kmeans_router

- [ ] Multi-start traversal and statistics tests:

    cargo test -p diskann-disk disk_search_uses_pq_kmeans_router_start_points

- [ ] Multi-root BFS cache tests:

    cargo test -p diskann-disk static_cache_from_multiple_roots

- [ ] Benchmark input parsing/build/search wiring tests:

    cargo test -p diskann-benchmark --features disk-index pq_kmeans

- [ ] CI-style lint after focused tests pass:

    cargo clippy --workspace --all-targets -- -D warnings

Do not claim benchmark performance wins from unit tests. Unit tests only prove artifact validation, routing behavior, cache wiring, and metrics plumbing.

## Experiment plan

Run the MSTuringANN bundle from the repository root after building the benchmark binary in release mode with the disk-index feature.

1. Build the router artifact once with configs/build_pq_kmeans_router.json.
2. Run the baseline search config with warmup_runs = 2 and repetitions = 5.
3. Run max_start_points = 16 with the same warmup and repetitions.
4. Run max_start_points = 8 to check the latency/recall tradeoff curve.
5. Collect output JSON and logs under the bundle results/ and logs/ directories.
6. Compare recall first; reject the router if recall drops at L=200, recall@100.
7. If recall is stable, compare mean hops, mean I/Os, mean latency, p95 latency, QPS, cache hit percentage, mean_router_time, mean_router_scanned_codes, and mean_routed_start_points.
8. Report extra memory as approximately k * (4 + num_pq_chunks) bytes. For N = 10,000,000 and num_pq_chunks = 64, k = ceil(sqrt(N)) = 3163, so the compact table is roughly 3163 * 68 = 215,084 bytes before file/header/container overhead.

Warmup note: warmup_runs must execute before measured repetitions for every config. This minimizes one-time page cache, static cache construction, and router artifact loading effects so the measured comparison reflects steady-state query behavior.
