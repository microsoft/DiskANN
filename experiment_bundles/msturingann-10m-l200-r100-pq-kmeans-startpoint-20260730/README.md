# MSTuringANN 10M PQ-kmeans start-point router bundle

This bundle compares the current disk-index baseline against the PQ-kmeans start-point router on the MSTuringANN 10M index at L=200, recall@100.

Run commands from the repository root after ensuring the index files exist under outputs/ and the MSTuringANN query/groundtruth files exist under /Users/xiaoweijiang/Documents/diskann/bigann10Mdatasets/MSTuringANNS.

1. Build the router artifact:

    cargo run --release -p diskann-benchmark --features disk-index -- run --input-file experiment_bundles/msturingann-10m-l200-r100-pq-kmeans-startpoint-20260730/configs/build_pq_kmeans_router.json --output-file experiment_bundles/msturingann-10m-l200-r100-pq-kmeans-startpoint-20260730/results/build_pq_kmeans_router.json

2. Run the baseline:

    cargo run --release -p diskann-benchmark --features disk-index -- run --input-file experiment_bundles/msturingann-10m-l200-r100-pq-kmeans-startpoint-20260730/configs/search_baseline.json --output-file experiment_bundles/msturingann-10m-l200-r100-pq-kmeans-startpoint-20260730/results/search_baseline.json

3. Run the router with max_start_points = 16:

    cargo run --release -p diskann-benchmark --features disk-index -- run --input-file experiment_bundles/msturingann-10m-l200-r100-pq-kmeans-startpoint-20260730/configs/search_pq_kmeans_msp16.json --output-file experiment_bundles/msturingann-10m-l200-r100-pq-kmeans-startpoint-20260730/results/search_pq_kmeans_msp16.json

4. Run the router with max_start_points = 8:

    cargo run --release -p diskann-benchmark --features disk-index -- run --input-file experiment_bundles/msturingann-10m-l200-r100-pq-kmeans-startpoint-20260730/configs/search_pq_kmeans_msp8.json --output-file experiment_bundles/msturingann-10m-l200-r100-pq-kmeans-startpoint-20260730/results/search_pq_kmeans_msp8.json

Recommended log capture is to pipe terminal output to the matching file in logs/ while keeping the JSON output file in results/.

Metrics to compare:

- recall at L=200 and recall@100; this is the gate and must not regress.
- mean_hops and mean_ios; these should fall if routing starts closer to query neighborhoods.
- mean_latency, p95_latency, p999_latency, and QPS.
- cache_hit_percentage with num_nodes_to_cache = 50000.
- mean_router_time, mean_router_scanned_codes, and mean_routed_start_points.
- router artifact size and estimated extra memory, approximately k * (4 + num_pq_chunks) bytes with k = ceil(sqrt(N)).

Warmup/repetition policy:

- Each search config uses warmup_runs = 2 and repetitions = 5.
- Do not compare a run that changed index, query file, groundtruth file, num_threads, beam_width, search_list, recall_at, or cache budget.
- Run baseline and router configs close together on the same machine to reduce environmental noise.
