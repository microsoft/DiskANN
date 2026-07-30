# Windows devbox rerun context: fixed PQ-geometry MSTuringANN Top-8 experiment

把这份文档直接发给 Windows devbox 里的 AI。目标是在 Windows devbox 上复跑 fixed PQ-geometry 版本的 MSTuringANN 10M Baseline vs PQ-kmeans Top-8 实验，并产出同样格式的结果表。

## 1. 关键背景

旧实验有一个 bug：PQ-kmeans router 把 PQ label ID 当作坐标做距离，等价于：

```text
sum((lhs_label_id - rhs_label_id)^2)
```

这是错的，因为 PQ centroid ID 的编号顺序没有几何意义。PR 现在已经修复为：

- build 阶段 inflate PQ code 到 reconstructed vector，在真实 `f32` 几何空间做 kmeans；
- build 阶段用 `FixedChunkPQTable::l2_distance(centroid, code)` 分配样本和选择代表点；
- query 阶段用 ADC：`populate_chunk_distances(query)` + `pq_dist_lookup_single(representative_code)`；
- BFS cache 仍然是 multi-source，把 `num_nodes_to_cache=50000` 分散到 routed starts 上。

## 2. 目标结果表

复跑下面 fixed-geometry setting：

| Metric | Baseline | k=1024 top8 | Delta |
| --- | ---: | ---: | ---: |
| Recall@100 | 73.4835 | 74.0571 | +0.574 pp |
| Mean latency | 1243.59 us | 1071.73 us | -171.86 us (-13.82%) |
| QPS | 3212.43 | 3725.45 | +513.02 (+15.97%) |
| IO / Hops | 491.183 | 387.552 | -103.632 (-21.10%) |
| Comparisons | 23,360.0 | 17,936.1 | -5,423.9 (-23.22%) |
| P95 | 1569 us | 1381 us | -188 us (-11.98%) |
| P99.9 | 2519 us | 1761 us | -758 us (-30.09%) |

Baseline 是普通 DiskANN disk search，使用默认单 start point。Top-8 是 fixed PQ-geometry PQ-kmeans start-point router，`num_representatives=1024`，`max_start_points=8`。

## 3. 使用代码

使用 draft PR：

- PR: https://github.com/microsoft/DiskANN/pull/1300
- Branch: `codex/pq-kmeans-bfs-cache-router-clean`
- Fixed geometry commit: `d81c8c79 fix: use pq geometry for kmeans router`

Windows devbox 上 checkout：

```powershell
git clone https://github.com/microsoft/DiskANN.git
cd DiskANN

# 如果有 gh
gh pr checkout 1300

# 或者 pure git
git fetch origin pull/1300/head:codex/pq-kmeans-bfs-cache-router-clean
git switch codex/pq-kmeans-bfs-cache-router-clean
```

Build benchmark binary：

```powershell
cargo build --release -p diskann-benchmark --features disk-index
```

Sanity checks：

```powershell
cargo fmt --all --check
cargo test -p diskann-disk pq_kmeans_router
cargo check -p diskann-benchmark --features disk-index
cargo clippy --workspace --all-targets -- -D warnings
```

## 4. 数据和 index 要求

MSTuringANN 10M 数据集用 [harsha-simhadri/big-ann-benchmarks](https://github.com/harsha-simhadri/big-ann-benchmarks) 里的 script 下载/裁剪。Windows devbox 上建议这样准备：

```powershell
git clone https://github.com/harsha-simhadri/big-ann-benchmarks.git D:\repos\big-ann-benchmarks
cd D:\repos\big-ann-benchmarks
python create_dataset.py --dataset msturing-10M
```

需要的数据文件：

```text
D:\repos\big-ann-benchmarks\data\MSTuringANNS\base1b.fbin.crop_nb_10000000
D:\repos\big-ann-benchmarks\data\MSTuringANNS\query100K.fbin
D:\repos\big-ann-benchmarks\data\MSTuringANNS\msturing-gt-10M
```

需要有一个已 build 好的 MSTuringANN 10M disk index，命名可以按 devbox 实际路径调整。Mac 本地实验使用：

```text
outputs/msturingann10m_user_m59_l80_sq1_pq64_index
```

## 5. Config 模板

### Baseline search

```json
{
  "search_directories": [
    ".",
    "D:/repos/big-ann-benchmarks/data/MSTuringANNS"
  ],
  "jobs": [
    {
      "type": "disk-index",
      "content": {
        "source": {
          "disk-index-source": "Load",
          "data_type": "float32",
          "load_path": "outputs/msturingann10m_user_m59_l80_sq1_pq64_index"
        },
        "search_phase": {
          "queries": "query100K.fbin",
          "groundtruth": "msturing-gt-10M",
          "search_list": [200],
          "beam_width": 64,
          "recall_at": 100,
          "num_threads": 4,
          "is_flat_search": false,
          "distance": "squared_l2",
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
```

### Router build, k=1024

```json
{
  "search_directories": [
    "."
  ],
  "jobs": [
    {
      "type": "pq-kmeans-router-build",
      "content": {
        "load_path": "outputs/msturingann10m_user_m59_l80_sq1_pq64_index",
        "artifact": "outputs/msturingann10m_user_m59_l80_sq1_pq64.pq_kmeans_router_k1024_geometry.bin",
        "max_iterations": 4,
        "num_representatives": 1024
      }
    }
  ]
}
```

### Router search, k=1024 top8

```json
{
  "search_directories": [
    ".",
    "D:/repos/big-ann-benchmarks/data/MSTuringANNS"
  ],
  "jobs": [
    {
      "type": "disk-index",
      "content": {
        "source": {
          "disk-index-source": "Load",
          "data_type": "float32",
          "load_path": "outputs/msturingann10m_user_m59_l80_sq1_pq64_index"
        },
        "search_phase": {
          "queries": "query100K.fbin",
          "groundtruth": "msturing-gt-10M",
          "search_list": [200],
          "beam_width": 64,
          "recall_at": 100,
          "num_threads": 4,
          "is_flat_search": false,
          "distance": "squared_l2",
          "vector_filters_file": null,
          "num_nodes_to_cache": 50000,
          "start_point_router": {
            "type": "pq_kmeans",
            "artifact": "outputs/msturingann10m_user_m59_l80_sq1_pq64.pq_kmeans_router_k1024_geometry.bin",
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
```

## 6. Run commands

```powershell
target\release\diskann-benchmark.exe --quiet run --input-file configs\search_baseline_after_geometry_fix.json --output-file results\search_baseline_after_geometry_fix_output.json
target\release\diskann-benchmark.exe --quiet run --input-file configs\build_pq_kmeans_router_k1024_geometry.json --output-file results\build_pq_kmeans_router_k1024_geometry_output.json
target\release\diskann-benchmark.exe --quiet run --input-file configs\search_pq_kmeans_k1024_msp8_geometry.json --output-file results\search_pq_kmeans_k1024_msp8_geometry_output.json
```

## 7. 输出要求

最后给一个表，至少包含：

- recall@100
- QPS
- mean latency
- P95 / P99.9 latency
- mean hops
- mean IOs
- mean comparisons
- mean router time
- scanned router codes
- routed start points
- artifact bytes

注意：`cache_hit_percentage` 当前 benchmark 输出仍是 0.0%，不要把它当真实 cache hit rate。这个实验主要看 hops/IOs、comparisons 和 latency。
