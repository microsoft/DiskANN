# Graph-IVF exact two-means：实现与快速验证

2026-10-23。已移除改变浮点归约的旧转置候选，改为 `tiled_exact`：两个点并行计算，但每个点保留原四组八 lane、FMA、合并树和尾处理。三套完整数据构建后的索引指纹、统计和本次查询的 recall/返回数/逻辑扫描/结果指纹全部与 paired 一致。

**正确性修复通过；性能结果仍只支持保留为实验候选。** 完整 fit 18 种 shape 均较快，实际 two-means 阶段也较快；Enron 总 build 略慢，search 实测全部较慢，因此不能声称整体 build/search 都获得稳定加速。

## 实现为什么不再改变索引

- 不做逐维单链累加，也不使用 norm/dot 展开。`f32x16` 的两半分别是两个点的八坐标 lane，最后分别使用原八 lane 归约；绝不对整个 16 lane 求和。
- 一次父桶准备只选择 evaluator，不打包坐标，不分配 N 条 assignment。每两点的选择完成后，仍按原顺序做 f64 求和；奇数末点调用 paired。固定 seed、轮数、tie、空中心和最终跨 children reassignment 共用原路径。
- x86 V3/V4 编译且 CPU 支持 V4 时，运行 AVX-512 双点内核；其它情况保留本编译目标的 paired 行为。scalar 编译不会升级到不同的 FMA/归约。
- 默认不切换，CLI 只保留 `paired_point` 和 `tiled_exact`。quantization 中旧 two-means 实验模块和 diskann 对它的可选依赖已移除。

详细设计与代码位置见 [更新后的方案](graph-ivf-transposed-two-means-plan.md)。

## 快速实验范围

A/B 使用同一冻结 executable，只切换策略。每套数据一组 ABBA，各两次 A、两次 B，所有 base 行均参与 build；每个 nprobe 一次查询 warmup、一次实测。YFCC 使用全部 100 条查询；Enron 和 FBV8 各用原文件前 32 条及对应 groundtruth。这是固定范围的快速验证，不是之前全查询实验的续跑。

| 数据 | base 行数 | D | 本次查询 / 原始查询 |
| --- | ---: | ---: | ---: |
| YFCC | 10,000 | 192 | 100 / 100 |
| Enron | 1,087,932 | 384 | 32 / 1,000 |
| fbv8_mimir | 958,152 | 896 | 32 / 6,380 |

initial16、batch4096、split2048、Lloyd5、seed7；graph degree16/L64；k10/L128/nprobe4、16、64。Windows、AMD Ryzen AI MAX+395、单线程、affinity0x10、Normal；电源和频率由 OS 管理。

## Build 实测

下表为算术均值。减少率 = 100 × (1 − B/A)，正值更快，负值更慢；每个 B 只配自己的 ABBA 内两个 A。所有慢样本保留。

| 数据 | two-means A → B (s) | two-means 减少 | 总 build A → B (s) | 总 build 减少 |
| --- | ---: | ---: | ---: | ---: |
| yfcc | 0.000844 → 0.000653 | 22.69% | 0.007614 → 0.007392 | 2.91% |
| enron | 1.632754 → 1.225278 | 24.96% | 17.990873 → 18.197957 | -1.15% |
| fbv8_mimir | 2.944279 → 2.015296 | 31.55% | 26.466389 → 25.217047 | 4.72% |

Enron 的 two-means 节省约 0.407 s，但其它 IVF 工作均值增加约 0.609 s，抵消收益。FBV8 的 A 总 build 从 29.328 s 到 23.605 s，显示时段变化明显；单个 block 不提供可靠的置信区间。YFCC 的 two-means 只有亚毫秒级，尤其容易受时钟/调度影响。以上数字不作为稳定端到端收益承诺。

## Search：质量/工作量一致，耗时没有通过无退化判断

每个设置中 recall、返回数量、points/lists/centroid comparisons 以及结果指纹相同，平均每查询都返回 k=10。以下给出实际搜索耗时，不能因为结果相同就隐去变慢的时间。

| 数据 | nprobe (实际) | recall@10 | 平均扫描 points | search A → B (s) | B 耗时变化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| yfcc | 4 (4) | 87.100% | 3808.64 | 0.012959 → 0.018324 | +41.41% |
| yfcc | 16 (16) | 100.000% | 9731.48 | 0.027992 → 0.053276 | +90.33% |
| yfcc | 64 (17) | 100.000% | 10000.00 | 0.027193 → 0.059672 | +119.44% |
| enron | 4 (4) | 43.125% | 5736.91 | 0.103526 → 0.137568 | +32.88% |
| enron | 16 (16) | 66.875% | 23050.88 | 0.381601 → 0.505996 | +32.60% |
| enron | 64 (64) | 89.375% | 91511.03 | 1.539265 → 2.140137 | +39.04% |
| fbv8_mimir | 4 (4) | 30.000% | 4741.22 | 0.118963 → 0.141204 | +18.70% |
| fbv8_mimir | 16 (16) | 50.625% | 20455.28 | 0.534546 → 0.546415 | +2.22% |
| fbv8_mimir | 64 (64) | 77.812% | 85417.84 | 2.258856 → 2.320126 | +2.71% |

搜索路径未调用新的 two-means 内核，索引指纹和逻辑工作也未变；但这些证据不能解释耗时差异。频率、调度、缓存/分配状态或 AVX-512 的后续影响均未用计数器验证，不能直接归因于噪声，也不能宣称 search 性能不受影响。若要切默认，需在独立、重复的协议下定位并排除该退化。

## 完整 two-means 与回归

完整 fit 包含准备、ISA 选择、scratch 分配、seeding、5 轮分配/f64 求和/中心更新及内部 scratch 释放，独立诊断在计时后运行。N={128,1024,4096}、D={192,384,896}、连续/分散布局共 18 种 shape，每种 1 warmup ABBA + 4 实测 ABBA，共 288 个实测完整 fit。减少率范围为 4.19%—23.88%。

| N | D | 连续布局减少 | 分散布局减少 |
| ---: | ---: | ---: | ---: |
| 128 | 192 | 12.43% | 23.15% |
| 128 | 384 | 15.74% | 20.36% |
| 128 | 896 | 12.52% | 11.00% |
| 1024 | 192 | 15.24% | 18.37% |
| 1024 | 384 | 14.40% | 16.66% |
| 1024 | 896 | 15.50% | 7.33% |
| 4096 | 192 | 11.23% | 20.01% |
| 4096 | 384 | 12.38% | 17.87% |
| 4096 | 896 | 4.19% | 23.88% |

所有中间 assignment/中心、最终 sums/中心/assignment 和 RNG 检查均一致。原合成分歧配置 seed7 N16384 D384、seed42 N4096 D384、seed2026 N4096 D896 在两种布局也全部通过。

本地另外重放了四个保存的真实分歧点，以及它们所在父桶的原始行序和初始中心行：Enron parent75/77、FBV8 parent3/11，共 56,400 个 point-step decisions。两距离 bits 和五轮分组全部一致。该回归从已保存初始中心继续，不是重新复现原 build 全部 RNG 历史；坐标未提交。

开发初版还发现一个代码生成问题：局部 closure 未内联到 V4 函数，逐片段调用/寄存器 spill 导致完整 fit 慢 9—18 倍。已改为显式 inline(always) helper，生成热循环为 zmm FMA、无调用。初版样本和汇编证据全部单独保留，未删除慢样本或与最终 binary 混算。

## 检查与边界

- 7 个 paired/tile 距离测试（含 Scalar/V3/V4、错位、全部尾维、near-tie、溢出、signed zero/subnormal）、13 个 clustering、16 个 experimental benchmark、8 个默认 benchmark、10 个 Python protocol 测试通过。
- 默认 x86-64 编译的 7 个距离测试通过；AArch64 Windows vector/diskann all-targets 编译检查通过。Neon 真机运行和旧 x86 SDE 模拟未执行。
- 默认和 experimental workspace Clippy `-D warnings`、cargo fmt、release build 通过。
- 默认 IVF 全套仍有一个原有 NaN 初始化失败：`invalid_initialization_and_uninitialized_insertion_fail`，11 通过/1 失败，相关源码未改，历史日志已复现；不把它计为通过。
- 本地 selected-source 行覆盖：paired/tile 96.57%（新 tile 97.53%）、clustering 91.57%、benchmark 90.27%。这不是远端 Codecov patch 通过结论。coverage 的 benchmark suite 耗时 4,138 s，发生在正式实验完成后，不属于性能样本。

冻结 binary SHA256：`7beea60c66cd4128bb58ee571d98cc6a02568d28212a8dda55b1a99abe842dbd`。源起点 `4f84618561e0208199a963cb8cb0a99ddfe6f67b` 加本轮 diff；测前/测后核对 640 个源文件、binary、输入、完整数据及截取 query/GT hashes。12 次调用都成功、固定参数/affinity/priority/query 数一致。

正式 build/search 在 01:54:06—01:58:07 UTC 完成；期间未编译或测试。runner 未记录 cargo/rustc/benchmark 重叠，但其 30 秒观察不覆盖全部桌面后台活动；未保证独占 CPU 或固定频率。旧每小时自动化仍暂停。

[完整证据与原始样本索引](results/tiled-exact-20261023/README.md)。本地完整文件在 `target/ivf-tiled-exact-20261023-final`；公共 JSON 规范化本地路径，原始文件 hashes 另有索引。

建议保留本次数值修复和 opt-in 候选，不切默认。当前结论是“保持构建结果的 two-means 加速已实现并验证”，不是“完整 build/search 在所有数据集均加速”。这里使用内存 provider，逻辑扫描相同不能扩展为物理 SSD I/O 或 Mimir 服务性能结论。
