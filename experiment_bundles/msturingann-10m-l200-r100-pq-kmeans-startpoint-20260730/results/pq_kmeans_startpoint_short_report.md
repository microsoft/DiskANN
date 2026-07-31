# PQ-kmeans start-point router 短版实验报告

完整实验记录在同目录的 `pq_kmeans_startpoint_summary.md`。这份短版只保留 fixed PQ-geometry 修复后、10%N training sample rerun 的结论和关键数据。

## 结论

推荐配置仍然是 `k=1024, max_start_points=8`。这次结论基于正确的 PQ/codebook geometry，并且使用修正后的 IO 计数：physical disk IOs、logical vertex loads、overall/traversal/rerank cache hit 分开统计。

MSTuringANN 10M 上，`k=1024/top8` 相比同 setting baseline：

- recall 没有 regression：73.4835 → 73.6540（+0.171 pp，+0.23%）
- QPS：2929.19 → 3719.83（+26.99%）
- mean latency：1362.45 us → 1073.47 us（-21.21%）
- corrected mean physical IOs：336.243 → 329.393（-2.04%）
- total logical vertex loads：692.183 → 603.332（-12.84%）
- traversal loads / hops：491.183 → 395.332（-19.51%）
- comparisons：23360.0 → 17952.1（-23.15%）

这版报告只把 MSTuringANN 作为有效实验数据集。BigANN 10M 暂时不纳入结论，因为当前本地 BigANN index/PQ 配置不适合作为这个 PR 的可比实验。

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

| Setting | Value |
| --- | ---: |
| Dataset size | N=10,000,000 |
| training_sample_size | 1,000,000 (10%N) |
| router max_start_points | 8 |
| Search L | 200 |
| recall@K | 100 |
| beam width | 64 |
| threads | 4 |
| num_nodes_to_cache | 50,000 |
| warmup runs | 2 |
| measured repetitions | 5 |

## MSTuringANN 10M corrected IO metrics

| Variant | Recall@100 | QPS | Mean latency | P99.9 | Mean IOs | Mean loads | Cache hit | Trav IOs | Trav loads / hops | Trav hit | Rerank IOs | Rerank loads | Rerank hit | Comparisons |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 73.4835 | 2929.19 | 1362.45 us | 13013 us | 336.243 | 692.183 | 51.42% | 336.243 | 491.183 | 31.54% | 0.000 | 201.000 | 100.00% | 23360.0 |
| k=256, top8 | 73.6356 | 3391.69 | 1176.94 us | 3443 us | 348.125 | 640.108 | 45.61% | 348.125 | 432.108 | 19.44% | 0.000 | 208.000 | 100.00% | 19714.6 |
| k=512, top8 | 73.6069 | 3488.05 | 1143.81 us | 3481 us | 336.109 | 624.177 | 46.15% | 336.109 | 416.177 | 19.24% | 0.000 | 208.000 | 100.00% | 18919.2 |
| k=1024, top8 | 73.6540 | 3719.83 | 1073.47 us | 1858 us | 329.393 | 603.332 | 45.40% | 329.393 | 395.332 | 16.68% | 0.000 | 208.000 | 100.00% | 17952.1 |

## Delta vs baseline

| Variant | Recall | QPS | Mean latency | Mean IOs | Mean loads | Cache hit | Trav loads / hops | Rerank loads | Comparisons |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| k=256, top8 | +0.152 pp (+0.21%) | +462.50 (+15.79%) | -185.51 us (-13.62%) | +11.883 (+3.53%) | -52.076 (-7.52%) | -5.81 pp (-11.30%) | -59.076 (-12.03%) | +7.000 (+3.48%) | -3645.4 (-15.60%) |
| k=512, top8 | +0.123 pp (+0.17%) | +558.86 (+19.08%) | -218.64 us (-16.05%) | -0.133 (-0.04%) | -68.007 (-9.82%) | -5.27 pp (-10.25%) | -75.007 (-15.27%) | +7.000 (+3.48%) | -4440.8 (-19.01%) |
| k=1024, top8 | +0.171 pp (+0.23%) | +790.65 (+26.99%) | -288.98 us (-21.21%) | -6.850 (-2.04%) | -88.852 (-12.84%) | -6.02 pp (-11.70%) | -95.852 (-19.51%) | +7.000 (+3.48%) | -5407.9 (-23.15%) |

## Caveat

- 旧 label-ID 结果只能作为 invalid reference；不能再作为有效实验结论。
- 修正后的 `mean_ios` 是 physical disk IOs，不再等同于 hops / logical loads。
- Routed rows 的 overall/traversal cache-hit percentage 低于 baseline，但它们访问的 traversal vertex 更少，latency/comparisons 也更低；所以这里应该同时看 corrected IO、loads、latency、comparisons。
- Wikipedia/Cohere 本地 base 文件损坏/截断；Enron 本地缺少可用数据和预建 index，因此没有有效 rerun。
