# Wikipedia/Cohere PQ-kmeans Multi-start Router Scaling Prediction Plan

目标：在最多只能跑到 10M 向量的限制下，用 Wikipedia/Cohere official inner-product 数据集验证 PQ-kmeans query-aware multi-start router + multi-source BFS cache 是否有继续 scale 到 100M / 1B 的可能。

核心读数不是单点最优值，而是 treatment 相对 baseline 的 delta-metrics 随数据规模 N、routing 粒度 k、以及每个 representative 覆盖规模 N/k 的趋势。如果 1M 到 10M 的趋势仍然稳定改善，并且 10M 上不同 N/k 的曲线能外推覆盖 100M/1B 的合理 N/k 区间，就认为这个架构“有 scale 信号”；这不是 100M/1B 的证明。

---

## 1. Codebase

Repository:

~~~text
https://github.com/microsoft/DiskANN
~~~

PR:

~~~text
https://github.com/microsoft/DiskANN/pull/1300
~~~

Branch:

~~~text
codex/pq-kmeans-bfs-cache-router-clean
~~~

Required code commit:

~~~text
dee21bac5100ee97cbc18cdf1b047a823c0145e5
~~~

This is the last code-changing commit required by the experiment plan. Docs-only descendants on the same branch are acceptable, but every run must record the actual `git rev-parse HEAD`.

Before running, verify the exact code:

~~~bash
git fetch origin codex/pq-kmeans-bfs-cache-router-clean
git checkout codex/pq-kmeans-bfs-cache-router-clean
git rev-parse HEAD
git status --short --branch
~~~

Accept a run only if `git rev-parse HEAD` is `dee21bac5100ee97cbc18cdf1b047a823c0145e5` or a docs-only descendant of it. If a newer code commit is used, the report must explicitly record the replacement commit and explain what changed.

Build command:

~~~bash
cargo build --release -p diskann-benchmark --features disk-index --bin diskann-benchmark
~~~

Recommended pre-run checks:

~~~bash
cargo test -p diskann-disk pq_kmeans_router -- --test-threads=1
cargo test -p diskann-disk quantizer_preprocess -- --test-threads=1
cargo check -p diskann-benchmark --features disk-index --bins
~~~

Known caveat for this PR line: full `cargo test -p diskann-disk` may include unrelated pre-existing failures; record them separately and do not use them to invalidate this experiment unless they touch PQ router/search/cache behavior.

---

## 2. Dataset and metric contract

Use the Wikipedia/Cohere dataset from `harsha-simhadri/big-ann-benchmarks`.

Source:

~~~text
https://github.com/harsha-simhadri/big-ann-benchmarks
~~~

Dataset family:

~~~text
Wikipedia Cohere
dtype: float32
dimension: 768
query count: 5,000
official distance / ground truth metric: inner_product
vector state: unnormalized; do not normalize
~~~

Required scales:

| Dataset ID | N | Role |
|---|---:|---|
| `wikipedia-1M` | 1,000,000 | sanity scale and fixed-k comparison |
| `wikipedia-10M` | 10,000,000 | primary largest local scale |

Optional smoke scale:

| Dataset ID | N | Role |
|---|---:|---|
| `wikipedia-100K` | 100,000 | pipeline validation only; do not use for final scale claim |

The 10M scale is treated as the same Wikipedia/Cohere family crop as the official benchmark definitions. Algorithmic parameters must match the dataset family: `distance=inner_product`, `dim=768`, `num_pq_chunks=192`. If a local benchmark reference only lists 100K/1M, record 10M as a scale extension of that family, not as a different dataset.

### Hard metric gate

All distance-sensitive stages must use the same metric:

| Stage | Required value |
|---|---|
| Disk index build `source.distance` | `inner_product` |
| Disk search `search_phase.distance` | `inner_product` |
| PQ-kmeans router build `distance` | `inner_product` |
| Router artifact metric | `inner_product` |
| Ground truth | official Wikipedia/Cohere IP GT |

Do not build a `squared_l2` Wikipedia index and evaluate it against the official `wikipedia-1M` or `wikipedia-10M` ground truth. If an L2 architecture-only run is ever needed, generate a separate L2 ground truth for the exact shard and label it as a different experiment. It must not be mixed with the official-IP results in this plan.

Why this matters: L2 and IP can both use PQ lookup tables, but the table contents and ranking semantics differ. For Wikipedia/Cohere, the router must score representatives with IP ADC, i.e. the PR path that populates chunk inner-products and then scores representative PQ codes. Reusing the L2 router path would select start points under the wrong objective.

---

## 3. Data download and validation

Example setup:

~~~bash
git clone https://github.com/harsha-simhadri/big-ann-benchmarks.git
cd big-ann-benchmarks
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements_py3.10.txt
python create_dataset.py --dataset wikipedia-1M
python create_dataset.py --dataset wikipedia-10M
~~~

Expected data directory:

~~~text
big-ann-benchmarks/data/wikipedia_cohere/
~~~

Expected files:

~~~text
wikipedia_base.bin.crop_nb_1000000
wikipedia_base.bin.crop_nb_10000000
wikipedia_query.bin
wikipedia-1M
wikipedia-10M
~~~

Expected base/query sizes:

| File | Expected bytes |
|---|---:|
| `wikipedia_base.bin.crop_nb_1000000` | 3,072,000,008 |
| `wikipedia_base.bin.crop_nb_10000000` | 30,720,000,008 |
| `wikipedia_query.bin` | 15,360,008 |

Validate before building:

~~~bash
python3 - <<'PY'
from pathlib import Path
import struct

data_dir = Path("data/wikipedia_cohere")
checks = [
    ("wikipedia_base.bin.crop_nb_1000000", 1_000_000, 768),
    ("wikipedia_base.bin.crop_nb_10000000", 10_000_000, 768),
    ("wikipedia_query.bin", 5_000, 768),
]

for name, rows, dim in checks:
    path = data_dir / name
    actual = path.stat().st_size
    expected = 8 + rows * dim * 4
    with path.open("rb") as f:
        header = struct.unpack("<II", f.read(8))
    print(name, "header=", header, "size=", actual, "expected=", expected)
    assert header == (rows, dim), (name, header, rows, dim)
    assert actual == expected, (name, actual, expected)

for name in ["wikipedia-1M", "wikipedia-10M"]:
    path = data_dir / name
    print(name, "size=", path.stat().st_size)
    assert path.is_file()
PY
~~~

If the Wikipedia files are malformed or truncated:

1. Delete only the failed dataset files under `big-ann-benchmarks/data/wikipedia_cohere/`, then rerun `create_dataset.py`.
2. Re-run the header/size validation above.
3. If Wikipedia still cannot be made usable, use BigANN 10M or MSTuringANNS 10M only as a larger-scale proxy for the architecture; do not present those proxy results as Wikipedia/Cohere IP results.
4. A smaller Wikipedia shard is acceptable only for pipeline debugging. It does not replace the required 1M/10M results unless it has matching official or regenerated ground truth.

---

## 4. Experiment hypothesis

Primary hypothesis:

PQ-kmeans routing reduces wasted disk traversal by choosing query-aware start points. As N grows, the graph becomes larger and the single-medoid entry point becomes increasingly expensive for many queries, so the routed-start treatment should reduce hops, physical IO, vertices loaded, and comparisons relative to baseline. The router scan overhead should remain small because it scans k representative PQ codes, not N data points.

Secondary hypotheses:

- Recall@100 should stay flat or improve slightly because multi-start traversal enters graph regions closer to the query.
- The best k should stay below `sqrt(N)`; increasing k beyond `sqrt(N)` is intentionally excluded.
- At fixed k, treatment benefit should not disappear when moving from 1M to 10M.
- At comparable N/k, 10M behavior can provide a directional proxy for 100M/1B behavior.
- Router artifact size should stay negligible relative to the disk index, approximately O(k × pq_chunks).

---

## 5. Baseline and treatment

Baseline:

~~~text
DiskANN disk search
start_point_router = null
entry point = original single medoid behavior
num_nodes_to_cache = 50,000
~~~

Treatment:

~~~text
DiskANN disk search
start_point_router.type = "pq_kmeans"
router artifact built from the same disk index PQ data
router build distance = inner_product
search distance = inner_product
multi-source BFS cache seeded from routed representatives
num_nodes_to_cache = 50,000
~~~

Only the start-point routing/cache seeding should differ between baseline and treatment. Keep index, data, search list, beam width, thread count, recall depth, and cache budget fixed.

---

## 6. Global implementation parameters

Dataset-specific build/search contract:

| Parameter | Value |
|---|---|
| `data_type` | `float32` |
| `distance` | `inner_product` |
| `dim` | 768 |
| `max_degree` | 59 |
| `l_build` | 80 |
| `num_threads` for build | 8 |
| `num_pq_chunks` | 192 |
| `quantization_type` | `SQ_1_2.0` |
| `recall_at` | 100 |
| `search_list` | [100, 160, 200] |
| `beam_width` | 64 |
| `num_threads` for search | 4 |
| `warmup_runs` | 2 |
| `repetitions` | 5 |
| `num_nodes_to_cache` | 50,000 |
| `vector_filters_file` | null |
| `post_processor` | null |
| `search_io_limit` | null |

`search_l` must never be below `recall_at`. For Recall@100, do not run `search_l=80`.

Resource parameter:

| Scale | `build_ram_limit_gb` |
|---|---:|
| 100K / 1M | 4.0 |
| 10M | 16.0 recommended resource override |

The 10M RAM value is a resource override for practical local builds; record it in every result row. It is not an algorithmic comparison knob.

Router build parameters:

| Parameter | Value |
|---|---|
| `distance` | `inner_product` |
| `num_representatives` | k from the matrix below |
| `max_iterations` | 4 |
| `training_sample_size` | code default, N/10, unless a run explicitly records a different value |

If router build time is too high on Wikipedia 10M, use a two-pass plan:

1. Sweep k with `training_sample_size=262144`.
2. Rebuild only the best two k values with the default N/10 sample and mark those as confirmation rows.

Do not silently mix sample sizes in the same result table; include `training_sample_size` as a column.

---

## 7. k and max-start-points matrix

Rules:

~~~text
k values are powers of two only.
k must be <= sqrt(N).
Do not test k > sqrt(N).
Primary max_start_points = 8.
Top-start ablation = [4, 8, 16] only at selected best k values.
~~~

Required k matrix:

| Scale | sqrt(N) | Required k values |
|---|---:|---|
| 1M | 1,000 | [128, 256, 512] |
| 10M | 3,162 | [128, 256, 512, 1024, 2048] |

Optional smoke:

| Scale | sqrt(N) | Optional k values |
|---|---:|---|
| 100K | 316 | [64, 128, 256] |

Optional 1B coarse-regime proxy:

| Scale | Optional k | Why |
|---|---:|---|
| 10M | 64 | Tests N/k ≈ 156K, closer to 1B with k=8192 than the required 10M matrix. Use only if the required matrix is already complete. |

Top-start ablation:

| Scale | k choice | `max_start_points` |
|---|---|---|
| 1M | best k from required matrix | [4, 8, 16] |
| 10M | best k from required matrix | [4, 8, 16] |

Primary treatment rows use `max_start_points=8`; ablation rows explain sensitivity and are not part of the main N trend unless explicitly marked.

---

## 8. How this predicts 100M and 1B with only 10M data

Use two x-axes:

1. N: compare 1M vs 10M at fixed k values [128, 256, 512].
2. N/k: compare routing granularity across k values, especially 10M [128, 256, 512, 1024, 2048].

Required derived columns:

~~~text
N
k
N_over_k = N / k
delta_mean_latency_pct
delta_p95_latency_pct
delta_p99_latency_pct
delta_p999_latency_pct
delta_mean_ios_pct
delta_mean_hops_pct
delta_mean_comparisons_pct
router_overhead_pct = mean_router_time_us / treatment_mean_latency_us
net_time_saved_us = baseline_mean_latency_us - treatment_mean_latency_us
router_payback = net_time_saved_us / mean_router_time_us
~~~

100M / 1B proxy mapping:

| Target scale | Plausible k under sqrt(N) | Target N/k | Closest 10M proxy |
|---|---:|---:|---|
| 100M | 2048 | 48,828 | 10M k=256, N/k=39,063 |
| 100M | 4096 | 24,414 | 10M k=512, N/k=19,531 |
| 100M | 8192 | 12,207 | 10M k=1024, N/k=9,766 |
| 1B | 8192 | 122,070 | 10M k=64 optional, N/k=156,250 |
| 1B | 16384 | 61,035 | 10M k=128, N/k=78,125 |

Interpretation:

- If 10M benefits persist across k=128..2048 and the best region is broad, the router is less likely to be a narrow 10M artifact.
- If 1M improves but 10M regresses at fixed k, that is a negative scaling signal.
- If 10M improves at comparable N/k values that map to 100M, that is a positive 100M signal.
- 1B remains a weaker extrapolation because 10M cannot fully reproduce 1B graph structure, cache pressure, or storage behavior.

---

## 9. Benchmark input templates

These are runner payload templates. Replace paths and save names per machine. Keep all metric fields as `inner_product`.

Run command pattern:

~~~bash
cargo run --release -p diskann-benchmark --features disk-index -- \
  run --input-file configs/<input>.json --output-file results/<output>.json
~~~

### 9.1 Build disk index

~~~json
{
  "search_directories": [
    "/ABS/PATH/big-ann-benchmarks/data/wikipedia_cohere"
  ],
  "jobs": [
    {
      "type": "disk-index",
      "content": {
        "source": {
          "disk-index-source": "Build",
          "data_type": "float32",
          "data": "wikipedia_base.bin.crop_nb_10000000",
          "distance": "inner_product",
          "dim": 768,
          "max_degree": 59,
          "l_build": 80,
          "num_threads": 8,
          "build_ram_limit_gb": 16.0,
          "num_pq_chunks": 192,
          "quantization_type": "SQ_1_2.0",
          "save_path": "/ABS/PATH/outputs/wiki10m_ip_R59_L80_pq192_sq1"
        },
        "search_phase": {
          "queries": "wikipedia_query.bin",
          "groundtruth": "wikipedia-10M",
          "search_list": [100],
          "beam_width": 64,
          "recall_at": 100,
          "num_threads": 4,
          "is_flat_search": false,
          "distance": "inner_product",
          "vector_filters_file": null,
          "num_nodes_to_cache": 50000,
          "start_point_router": null,
          "warmup_runs": 1,
          "repetitions": 1,
          "search_io_limit": null,
          "post_processor": null
        }
      }
    }
  ]
}
~~~

The build job includes a minimal search phase because the benchmark input type requires it. Do not use the build-job search result as a final measurement; run the baseline/treatment load jobs below for final metrics.

### 9.2 Build PQ-kmeans router artifact

Use one job per k.

~~~json
{
  "search_directories": [
    "/ABS/PATH/big-ann-benchmarks/data/wikipedia_cohere"
  ],
  "jobs": [
    {
      "type": "pq-kmeans-router-build",
      "content": {
        "load_path": "/ABS/PATH/outputs/wiki10m_ip_R59_L80_pq192_sq1",
        "artifact": "/ABS/PATH/outputs/wiki10m_ip_R59_L80_pq192_sq1.k1024.pq_kmeans_router.bin",
        "distance": "inner_product",
        "num_representatives": 1024,
        "training_sample_size": 1000000,
        "max_iterations": 4
      }
    }
  ]
}
~~~

For 1M, use `training_sample_size=100000` if following the code default N/10. For 10M, use `training_sample_size=1000000` if following the code default N/10.

### 9.3 Baseline search

~~~json
{
  "search_directories": [
    "/ABS/PATH/big-ann-benchmarks/data/wikipedia_cohere"
  ],
  "jobs": [
    {
      "type": "disk-index",
      "content": {
        "source": {
          "disk-index-source": "Load",
          "data_type": "float32",
          "load_path": "/ABS/PATH/outputs/wiki10m_ip_R59_L80_pq192_sq1"
        },
        "search_phase": {
          "queries": "wikipedia_query.bin",
          "groundtruth": "wikipedia-10M",
          "search_list": [100, 160, 200],
          "beam_width": 64,
          "recall_at": 100,
          "num_threads": 4,
          "is_flat_search": false,
          "distance": "inner_product",
          "vector_filters_file": null,
          "num_nodes_to_cache": 50000,
          "start_point_router": null,
          "warmup_runs": 2,
          "repetitions": 5,
          "search_io_limit": null,
          "post_processor": null
        }
      }
    }
  ]
}
~~~

### 9.4 Treatment search

~~~json
{
  "search_directories": [
    "/ABS/PATH/big-ann-benchmarks/data/wikipedia_cohere"
  ],
  "jobs": [
    {
      "type": "disk-index",
      "content": {
        "source": {
          "disk-index-source": "Load",
          "data_type": "float32",
          "load_path": "/ABS/PATH/outputs/wiki10m_ip_R59_L80_pq192_sq1"
        },
        "search_phase": {
          "queries": "wikipedia_query.bin",
          "groundtruth": "wikipedia-10M",
          "search_list": [100, 160, 200],
          "beam_width": 64,
          "recall_at": 100,
          "num_threads": 4,
          "is_flat_search": false,
          "distance": "inner_product",
          "vector_filters_file": null,
          "num_nodes_to_cache": 50000,
          "start_point_router": {
            "type": "pq_kmeans",
            "artifact": "/ABS/PATH/outputs/wiki10m_ip_R59_L80_pq192_sq1.k1024.pq_kmeans_router.bin",
            "max_start_points": 8
          },
          "warmup_runs": 2,
          "repetitions": 5,
          "search_io_limit": null,
          "post_processor": null
        }
      }
    }
  ]
}
~~~

---

## 10. Metrics to record

Record both raw benchmark output and a flattened CSV/Parquet table.

Required search metrics:

| Metric | Source / meaning |
|---|---|
| `recall@100` | benchmark `recall` for `recall_at=100` |
| `qps` | benchmark `qps` |
| `mean_latency_us` | benchmark `mean_latency` |
| `p95_latency_us` | benchmark `p95_latency` |
| `p99_latency_us` | compute from per-query latencies if available; otherwise mark missing and add instrumentation before final report |
| `p999_latency_us` | benchmark `p999_latency` |
| `mean_hops` | benchmark `mean_hops` |
| `mean_ios` | physical IO operations / uncached reads per query |
| `mean_vertices_loaded` | logical graph loads per query |
| `mean_traversal_ios` | traversal physical IO |
| `mean_traversal_vertices_loaded` | traversal logical loads |
| `mean_rerank_ios` | rerank physical IO |
| `mean_rerank_vertices_loaded` | rerank logical loads |
| `cache_hit_percentage` | total cache hit rate |
| `traversal_cache_hit_percentage` | traversal cache hit rate |
| `rerank_cache_hit_percentage` | rerank cache hit rate |
| `mean_comparisons` | total comparisons per query |
| `mean_pq_preprocess_time_us` | query PQ preprocessing overhead |
| `mean_router_time_us` | query-aware router overhead |
| `mean_router_scanned_codes` | should be close to k |
| `mean_routed_start_points` | should be close to `max_start_points` |

Required artifact/build metrics:

| Metric | How to record |
|---|---|
| `index_build_wall_time_sec` | wall clock around build job |
| `router_build_wall_time_sec` | wall clock around router build job |
| `index_artifact_bytes` | `du -sb` or `du -sk` on index path |
| `router_artifact_bytes` | file size of `.pq_kmeans_router.bin` |
| `max_rss_mb` | `/usr/bin/time -l` on macOS or `/usr/bin/time -v` on Linux |
| `run_commit` | actual `git rev-parse HEAD` |
| `required_code_commit` | `dee21bac5100ee97cbc18cdf1b047a823c0145e5` unless intentionally replaced |
| `dataset_file_sha256` | optional but recommended for shared machines |

Router artifact size sanity:

~~~text
Approximate payload = k * (4 bytes representative id + 192 bytes PQ code) + metadata.
For k=2048, this is roughly 0.4 MB before serialization overhead.
~~~

---

## 11. Result table schemas

Raw run table:

| Column | Example |
|---|---|
| `run_id` | `wiki10m_ip_treat_k1024_msp8_l160_rep5` |
| `run_commit` | actual `git rev-parse HEAD` |
| `required_code_commit` | `dee21bac5100ee97cbc18cdf1b047a823c0145e5` |
| `dataset_id` | `wikipedia-10M` |
| `N` | 10000000 |
| `distance` | `inner_product` |
| `dim` | 768 |
| `num_pq_chunks` | 192 |
| `max_degree` | 59 |
| `l_build` | 80 |
| `build_ram_limit_gb` | 16.0 |
| `variant` | `baseline` or `pq_kmeans` |
| `k` | null for baseline, 1024 for treatment |
| `N_over_k` | null for baseline, 9765.625 for k=1024 |
| `max_start_points` | null for baseline, 8 for treatment |
| `num_nodes_to_cache` | 50000 |
| `training_sample_size` | null for baseline, 1000000 for router |
| `search_l` | 160 |
| `beam_width` | 64 |
| `search_threads` | 4 |
| `warmup_runs` | 2 |
| `repetitions` | 5 |
| `recall_at_100` | 0.9321 |
| `qps` | 123.4 |
| `mean_latency_us` | 8100 |
| `p95_latency_us` | 15000 |
| `p99_latency_us` | 24000 |
| `p999_latency_us` | 60000 |
| `mean_ios` | 42.0 |
| `mean_vertices_loaded` | 62.0 |
| `cache_hit_percentage` | 32.3 |
| `mean_hops` | 41.0 |
| `mean_comparisons` | 12000 |
| `mean_pq_preprocess_time_us` | 20 |
| `mean_router_time_us` | 35 |
| `mean_router_scanned_codes` | 1024 |
| `mean_routed_start_points` | 8 |
| `index_build_wall_time_sec` | 40000 |
| `router_build_wall_time_sec` | 3600 |
| `index_artifact_bytes` | 0 |
| `router_artifact_bytes` | 420000 |
| `notes` | `default training sample` |

Delta table, joined by `dataset_id + search_l + cache + build params`:

| Column | Formula |
|---|---|
| `delta_recall_pp` | `100 * (treatment_recall - baseline_recall)` |
| `delta_mean_latency_pct` | `100 * (treatment_mean_latency / baseline_mean_latency - 1)` |
| `delta_p95_latency_pct` | same |
| `delta_p99_latency_pct` | same |
| `delta_p999_latency_pct` | same |
| `delta_qps_pct` | `100 * (treatment_qps / baseline_qps - 1)` |
| `delta_mean_ios_pct` | `100 * (treatment_mean_ios / baseline_mean_ios - 1)` |
| `delta_mean_hops_pct` | same |
| `delta_mean_comparisons_pct` | same |
| `router_overhead_pct` | `100 * mean_router_time_us / treatment_mean_latency_us` |
| `router_payback` | `(baseline_mean_latency_us - treatment_mean_latency_us) / mean_router_time_us` |

---

## 12. Staged execution plan

### Stage 0: environment and data gate

Expected product:

- actual run commit recorded, with required code commit `dee21bac5100ee97cbc18cdf1b047a823c0145e5`
- benchmark binary built
- data header/size validation log
- one dry-run or tiny smoke input generated

Stopping criteria:

- stop if any distance field cannot be set to `inner_product`
- stop if official GT files are missing
- stop if base/query headers or sizes do not match expected values

### Stage 1: 100K smoke, optional

Run:

- build `wikipedia-100K`
- baseline search with cache=50K
- treatment k=[64,128,256], `max_start_points=8`

Expected product:

- pipeline proof that IP build/search/router artifact loading works
- no final scaling claim

Stopping criteria:

- stop and fix pipeline if router artifact metric mismatches search metric
- stop if Recall@100 cannot be computed with `search_l >= 100`

### Stage 2: 1M sanity and fixed-k baseline

Run:

- build `wikipedia-1M`
- baseline search
- treatment k=[128,256,512], `max_start_points=8`
- top-start ablation [4,8,16] only for the best 1M k

Expected product:

- first delta-vs-baseline table
- fixed-k anchors for [128,256,512]
- router overhead and artifact-size sanity

Stopping criteria:

- if all treatment rows have worse recall and worse latency at all search_l values, do not proceed to full 10M sweep until the metric/config path is audited
- if router_time_us is a large fraction of total latency, verify k and sample size before continuing

### Stage 3: 10M primary scale

Run:

- build `wikipedia-10M`
- baseline search
- treatment k=[128,256,512,1024,2048], `max_start_points=8`
- top-start ablation [4,8,16] only for the best 10M k

Expected product:

- 10M main delta table
- N trend at fixed k=[128,256,512]
- N/k trend for 10M
- best-k recommendation for Wikipedia/Cohere under 10M limit

Stopping criteria:

- stop if build/search uses any non-IP metric
- stop if `search_l < 100` appears in any Recall@100 row
- stop early if k=128/256/512 all regress badly and router overhead does not explain it; investigate metric and cache behavior before running k=1024/2048

### Stage 4: optional 1B proxy extension

Run only after Stage 3 is complete:

- treatment k=64, `max_start_points=8` on 10M

Expected product:

- one coarse N/k point that maps closer to 1B k=8192

Stopping criteria:

- skip if Stage 3 already shows no positive 10M scale signal

### Stage 5: analysis and report

Expected product:

- raw JSON output archive
- flattened raw table
- delta table
- plots or tables for:
  - delta latency vs N at fixed k
  - delta IO/hops/comparisons vs N at fixed k
  - delta latency/IO/hops/comparisons vs N/k on 10M
  - router_time_us and router_overhead_pct vs k
  - recall delta vs k
  - artifact size vs k

Final report wording:

- Say “positive/negative scaling signal”, not “proves 100M/1B”.
- Separate Wikipedia/Cohere IP results from any BigANN/MSTuring proxy results.
- Always include commit, dataset IDs, metric, cache budget, k matrix, and search_l.

---

## 13. Runtime estimate

The real time depends heavily on CPU, RAM, SSD, and whether the data is already local. Use this as planning guidance, not a promise.

| Stage | Work | Rough time on 16-core NVMe workstation | Rough time on laptop / shared machine |
|---|---|---:|---:|
| Stage 0 | checkout, build, data validation | 15-45 min excluding download | 30-90 min excluding download |
| Data download | 1M + 10M Wikipedia/Cohere | 30-120 min | 1-6 h |
| Stage 1 | optional 100K smoke | 10-30 min | 20-60 min |
| Stage 2 | 1M build + baseline + 3 router k + searches | 1-4 h | 3-10 h |
| Stage 3 | 10M build + baseline + 5 router k + searches | 12-36 h | 1.5-5 days |
| Stage 4 | optional 10M k=64 | 30-120 min | 1-4 h |
| Stage 5 | flattening and report | 1-3 h | 1-3 h |

Minimum useful pass:

~~~text
Stage 0 + Stage 2 + Stage 3 with 10M k=[512,1024,2048] only
Expected time: roughly 1-2 days on a strong workstation, 2-4 days on a laptop/shared box.
~~~

Full plan:

~~~text
Stage 0 through Stage 5, including all required k values and top-start ablations
Expected time: roughly 2-4 days on a strong workstation, 4-7 days on a laptop/shared box.
~~~

If runtime becomes the bottleneck, reduce only in this order:

1. Skip optional 100K smoke if Stage 0 validation is enough.
2. Skip Stage 4 k=64.
3. Delay top-start ablation until the primary k sweep is complete.
4. Use `training_sample_size=262144` for exploratory router builds, then confirm best two k values with default N/10.

Do not reduce `search_l` below 100, do not change `num_nodes_to_cache=50000`, and do not switch Wikipedia/Cohere away from `inner_product`.

---

## 14. Criteria for “scales”

Positive signal:

- Recall@100 is flat or improves: `delta_recall_pp >= -0.05 pp`, preferably positive.
- Mean and tail latency improve at 10M for at least one primary k and do not regress badly at p99/p999.
- Hops, physical IO, vertices loaded, and comparisons decrease relative to baseline.
- Router overhead is much smaller than traversal savings:
  - `router_overhead_pct < 5%` is strong.
  - `router_payback > 2` is a useful minimum.
- Benefit does not vanish from 1M to 10M at fixed k.
- 10M N/k curve has a broad useful region, not a single fragile k.
- Router artifact size remains sublinear and operationally negligible.

Negative signal:

- 10M treatment has worse latency and worse recall for all k.
- Router_time_us grows enough with k that it consumes the traversal savings.
- Cache hit rate drops or physical IO rises despite routed starts.
- Best k is unstable across search_l or requires a value near/exceeding `sqrt(N)`.
- Tail latency p99/p999 regresses materially even when mean latency improves.

Final conclusion levels:

| Label | Meaning |
|---|---|
| Strong positive | 10M improves recall-neutral latency/IO/hops/comparisons, router overhead is small, and N/k curve maps well to 100M. |
| Weak positive | Mean improves but tails or recall need more work; worth trying 30M/50M/100M on larger hardware. |
| Inconclusive | Pipeline works but results depend on one fragile k/search_l point. |
| Negative | 10M does not beat baseline after metric/config audit. |

---

## 15. One-page run checklist

Before each result row, record:

~~~text
run_commit = actual git rev-parse HEAD
required_code_commit = dee21bac5100ee97cbc18cdf1b047a823c0145e5 or documented replacement
dataset_id = wikipedia-1M or wikipedia-10M
distance = inner_product at build, search, router, and GT
dim = 768
num_pq_chunks = 192
max_degree = 59
l_build = 80
quantization_type = SQ_1_2.0
num_nodes_to_cache = 50000
recall_at = 100
search_l in [100,160,200]
k is power of two and <= sqrt(N)
max_start_points = 8 unless ablation
vector_filters_file = null
post_processor = null
~~~

Never accept a row that violates the metric gate or uses `search_l < recall_at`.
