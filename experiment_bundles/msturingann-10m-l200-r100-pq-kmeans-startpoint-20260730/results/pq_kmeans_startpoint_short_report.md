# PQ-kmeans start-point router 短版实验报告

完整实验记录在同目录的 `pq_kmeans_startpoint_summary.md`。这份短版只保留 fixed PQ-geometry 修复后的结论和关键数据。

## 结论

推荐配置仍然是 `k=1024, max_start_points=8`，但结论现在基于正确的 PQ/codebook geometry，而不是旧的 PQ label-ID ordinal distance。

MSTuringANN 10M 上，`k=1024/top8` 相比同 setting baseline：

- recall 没有 regression：73.4835 → 74.0571（+0.574 pp）
- mean latency：1243.59 us → 1071.73 us（-13.82%）
- hops / IOs：491.183 → 387.552（-21.10%）
- comparisons：23360.0 → 17936.1（-23.22%）
- artifact 只有 69.7 KB

BigANN 10M 上，`k=1024/top8` 也有稳定收益：

- recall：96.6686 → 96.7224（+0.054 pp）
- mean latency：902.12 us → 817.64 us（-9.37%）
- hops / IOs：451.764 → 341.307（-24.45%）
- comparisons：10593.2 → 7262.2（-31.44%）

## 关键修复

旧实现的问题是把 PQ code byte 当作坐标做距离：

```text
sum((lhs_label_id - rhs_label_id)^2)
```

这是错的，因为 PQ centroid ID 的编号顺序没有几何意义。

修复后：

- build 阶段把 PQ code inflate 成 reconstructed vector，在真实 `f32` 几何空间做 kmeans centroid update；
- build 阶段用 `FixedChunkPQTable::l2_distance(centroid, code)` 分配样本和选择代表点；
- query 阶段用 ADC：`populate_chunk_distances(query)` + `pq_dist_lookup_single(code)` 给 representatives 打分；
- BFS cache 仍然是 multi-source，把 `num_nodes_to_cache=50000` 分散到 routed starts 上。

## 实验设置

MSTuringANN 和 BigANN 都使用：

| Setting | Value |
| --- | ---: |
| Search L | 200 |
| recall@K | 100 |
| beam width | 64 |
| threads | 4 |
| num_nodes_to_cache | 50,000 |
| warmup runs | 2 |
| measured repetitions | 5 |
| router max_start_points | 8 |

## MSTuringANN 10M

| Variant | Recall@100 | Mean latency | P95 | P99.9 | Hops / IOs | QPS | Router time | Artifact | Delta vs baseline |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Baseline | 73.4835 | 1243.59 us | 1569 us | 2519 us | 491.183 | 3212.43 | 0.0 us | — | — |
| k=256, top8 | 74.3409 | 1138.69 us | 1441 us | 2549 us | 404.977 | 3506.10 | 36.7 us | 17.4 KB | recall +0.857 pp, latency -8.43%, IO -17.55% |
| k=512, top8 | 74.1320 | 1104.49 us | 1418 us | 2720 us | 397.728 | 3614.01 | 44.6 us | 34.9 KB | recall +0.649 pp, latency -11.18%, IO -19.03% |
| k=1024, top8 | 74.0571 | 1071.73 us | 1381 us | 1761 us | 387.552 | 3725.45 | 58.6 us | 69.7 KB | recall +0.574 pp, latency -13.82%, IO -21.10% |

## BigANN 10M

| Variant | Recall@100 | Mean latency | P95 | P99.9 | Hops / IOs | Comparisons | QPS | Delta vs baseline |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Baseline | 96.6686 | 902.12 us | 1043 us | 1184 us | 451.764 | 10593.2 | 4414.30 | — |
| k=1024, top8 | 96.7224 | 817.64 us | 963 us | 1149 us | 341.307 | 7262.2 | 4867.83 | recall +0.054 pp, latency -9.37%, IO -24.45% |

BigANN 的收益仍然明显，主要因为 routed starts 大幅减少 traversal：hops/IOs -24.45%，comparisons -31.44%。这说明 PQ codebook geometry 对 BigANN/SIFT-like uint8 数据的 coarse locality 捕捉得比较好。

## Caveat

- 旧 label-ID 结果只能作为 invalid reference；不能再作为有效实验结论。
- `cache_hit_percentage` 仍然显示 0.0%。当前 counter 没有准确反映 multi-source BFS cache hit attribution，所以这里用 hops/IOs、comparisons 和 latency 衡量实际效果。
- Wikipedia/Cohere 本地 base 文件损坏/截断；Enron 本地缺少可用数据和预建 index，因此没有有效 rerun。
