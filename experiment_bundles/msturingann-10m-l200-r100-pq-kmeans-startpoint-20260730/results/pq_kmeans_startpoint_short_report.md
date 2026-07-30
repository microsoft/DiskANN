# PQ-kmeans start-point router 短版实验报告

完整实验记录在同目录的 `pq_kmeans_startpoint_summary.md`。这份短版只保留结论和能支撑结论的关键数据。

## 结论

推荐配置是 `k=1024, max_start_points=8`。

这个配置达到了实验目标：recall 没有 regression，额外内存只有 69.7 KB，mean hops / IOs 降低 8.91%，mean latency 降低 1.39%。在 BigANN 10M 上收益更明显：mean latency 降低 28.54%，QPS 提升 40.25%，P99.9 latency 降低 90.56%。

核心判断：

- PQ-kmeans router 能选到比单 medoid start point 更好的 graph 入口点；直接效果是减少 traverse 中的 hops / IOs。
- `k=sqrt(N)=3163` 不是最优 tradeoff；`k=512` 和 `k=1024` 已经捕获主要收益，且 router 扫描成本和 artifact 都更小。
- BigANN 收益显著，说明这个数据集上 baseline 单入口更容易从较差区域启动；多入口 router 更容易命中 query 所在簇附近，降低无效扩展和长尾。
- 当前实验没有可靠 cache hit rate 计数；代码路径使用了 multi-source BFS cache，但现有 benchmark 输出的 `cache_hit_percentage=0.0%` 不能用于判断真实 hit rate。这里用 hops / IOs / comparisons / latency 作为实际效果指标。

## 实验设置

MSTuringANN 和 BigANN 都使用相同 search setting：

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

cache 策略使用 multi-source BFS：把 `num_nodes_to_cache=50000` 分散到 routed start points 上，而不是只围绕单 medoid 做 BFS cache。

## MSTuringANN 10M：k sweep 关键数据

baseline 是同一 setting 下的 single-start DiskANN。delta 均相对 baseline rerun。

| Variant | Recall@100 | Mean latency | Hops / IOs | QPS | Router time | Artifact | Key delta vs baseline |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Baseline | 73.4835 | 1282.58 us | 491.183 | 3113.24 | 0 us | — | — |
| k=256, msp8 | 74.6622 | 1274.07 us | 448.734 | 3133.79 | 4.754 us | 17.4 KB | recall +1.1787 pp, latency -0.66%, IO -8.64% |
| k=512, msp8 | 74.6366 | 1270.87 us | 448.262 | 3140.59 | 5.371 us | 34.9 KB | recall +1.1531 pp, latency -0.91%, IO -8.74% |
| k=1024, msp8 | 74.6721 | 1264.76 us | 447.408 | 3157.21 | 6.637 us | 69.7 KB | recall +1.1887 pp, latency -1.39%, IO -8.91% |
| k=2048, msp8 | 74.6387 | 1290.39 us | 447.898 | 3086.34 | 9.851 us | 139.3 KB | recall +1.1552 pp, latency +0.61%, IO -8.81% |
| k=3163, msp8 | 74.6030 | 1265.80 us | 448.346 | 3155.72 | 11.933 us | 215.1 KB | recall +1.1195 pp, latency -1.31%, IO -8.72% |
| k=4096, msp8 | 74.6565 | 1305.34 us | 447.698 | 3046.44 | 15.113 us | 278.6 KB | recall +1.1730 pp, latency +1.77%, IO -8.85% |

为什么推荐 `k=1024`：

- recall 最高：74.6721，比 baseline +1.1887 pp。
- mean latency 最低：1264.76 us，比 baseline -17.8 us / -1.39%。
- hops / IOs 最低：447.408，比 baseline -43.775 / -8.91%。
- router overhead 低：每 query 6.637 us，artifact 只有 69.7 KB。

如果更关注 P99.9，`k=512` 也值得考虑：P99.9 是 2826 us，比 baseline 3248 us 低 422 us，同时 artifact 只有 34.9 KB。

## BigANN 10M：跨数据集复验

BigANN 使用 `uint8` vectors + squared L2。配置为 `k=1024, max_start_points=8`。

| Variant | Recall@100 | Mean latency | P95 | P99.9 | Hops / IOs | Comparisons | QPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 96.6686 | 1242.96 us | 1608 us | 22514 us | 451.76 | 10593.16 | 3192.90 |
| PQ-kmeans k=1024 msp8 | 96.7447 | 888.19 us | 1126 us | 2126 us | 400.05 | 8715.79 | 4477.91 |
| Delta | +0.0761 pp | -354.77 us / -28.54% | -482 us / -29.98% | -20388 us / -90.56% | -51.71 / -11.45% | -1877.37 / -17.72% | +1285.01 / +40.25% |

BigANN 的收益比 MSTuringANN 更大，主要信号有三个：

- hops / IOs 下降 11.45%，说明更好的 start points 让 traverse 少走了很多磁盘节点。
- comparisons 下降 17.72%，说明候选扩展更集中，减少了无效距离计算。
- P99.9 从 22514 us 降到 2126 us，说明 baseline 存在明显长尾；router 多入口显著降低了从差入口启动导致的长路径。

这和 BigANN/SIFT-like `uint8` 数据本身有关：数据有较强局部簇结构，PQ code 的 coarse distance 对“query 属于哪个区域”有足够分辨率。因此扫描 1024 个 representative PQ codes 就能找到更接近 query 的多个入口点。

## Wikipedia/Cohere 10M 状态

Wikipedia/Cohere 没有得到可用结果。原因是本地 base vector 文件损坏或被截断：

```text
wikipedia_base.bin.crop_nb_10000000
header: 35000000 x 768
actual size: 8725200896 bytes
expected by header: 107520000008 bytes
```

因此这组没有 baseline、router build 或 search 数据。

## 验证

代码侧已跑过：

- `cargo fmt --all --check`
- `cargo test -p diskann-disk pq_kmeans_router`
- `cargo check -p diskann-benchmark --features disk-index`

实验侧所有有效搜索结果都使用 2 次 warmup + 5 次 measured repetitions，以尽量消除 cold cache 对 latency 的影响。
