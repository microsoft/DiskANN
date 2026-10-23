# Graph-IVF two-means：保持构建结果的精确批量计算

更新：2026-10-23。本文取代上一版允许浮点归约变化的转置方案；旧实现、原始实验和失败分析保存在 Git 历史及 [旧结果](graph-ivf-transposed-two-means-results.md)。本轮在现有第二层分支实现，PR #1503 的 base 保持 `u/xiajiang/graph-ivf-static-benchmark`。

## 1. 先修正验收契约

本次是计算加速，不能靠改变聚类结果减少后续工作。相同编译基线、输入顺序、seed 和参数下，要求两距离 bits、每轮 assignment、counts、f64 sums、中心 bits、RNG 状态以及最终索引布局一致。不同 ISA 的原始 L2 本身可能采用不同归约，因此不承诺跨不同编译目标的索引相同。

旧实现把每个点的“四组 × 八坐标 lane 累加，再树形归约”变成“每 lane 一个点，沿维度顺序 FMA”。数学公式相同，浮点运算树不同；Enron 和 FBV8 的真实 near-tie 已证明会翻转 assignment。只保留 seed、f64 sums 或非有限 fallback，不能修复。norm/dot 展开和任意 epsilon 判断也不满足本契约。

## 2. 本轮只保留一个候选

A 为 `paired_point`，B 为新标识 `tiled_exact`。旧 `transposed_l2_16`、`transposed_dot_16` 和 controlled Lloyd reference 从活动 Rust/CLI 路径移除，旧结果不改名、不覆盖、不混入新样本。默认 IVF 仍是 paired；实验入口显式选择 B。

本版不继续逐维全转置，而是在寄存器里并排放两个点的八维片段。不需要打包完整父桶，也不需要 N 条 assignment scratch。

```text
原 paired（一个点、两个中心）：
  s0[0..7], s1[0..7], s2[0..7], s3[0..7]，每个中心各一套

新 tile（两个点、两个中心）：
  每个 f32x16 = [point 0 的八坐标 lane | point 1 的八坐标 lane]
  每个中心仍有 s0 / s1 / s2 / s3
  中心片段只加载一次，复制到寄存器两半
  两半各自执行原 FMA 链，不在两个点之间求和
```

对每个点及中心，严格保留以下步骤：

1. 每 32 个坐标分给 s0、s1、s2、s3，每个都是八 lane。
2. 不足 32、但完整的八维片段依次交给 s0、s1、s2。
3. 合并顺序为 `(s0 + s1) + (s2 + s3)`。
4. 拆出两个点，分别调用原 L2 的部分尾向量处理和八 lane `sum_tree`。
5. 原 `total_cmp` 严格小于选择第二中心；相等选择第一个。

SIMD 改变同时执行的点数，不改变每个点的运算树。索引层按原点序处理两个输出并做 f64 求和；奇数末点走 paired。迭代次数、初始化抽样、空簇保留和最后跨 children 的重分配共用原实现。

## 3. ISA 和回退边界

在编译基线为 x86 V3/V4 且运行机器通过 V4 检查时，用 AVX-512 的 f32x16 做双点计算；当前 V3/V4 L2 都是相同的四组八 lane/FMA 归约，逐点 bits 应一致。每父桶准备阶段选择一次 evaluator。

V3-only 机器、scalar 编译和 AArch64 本轮调用原 paired。尤其 scalar 编译不能因机器有 AVX-512 就改变无 FMA、两组归约语义。按 diskann-wide 架构 witness 调用，未检查 ISA 时不执行对应指令。

这是有硬件条件的候选。寄存器拼接、函数调用和布局访问可能抵消收益；不提前承诺加速，不把机器支持 AVX-512 等同于任意 workload 更快。

## 4. 代码职责

- `diskann-vector/src/distance/paired/tiled.rs`：精确双点两中心 primitive；维度检查、架构选择、原归约与尾处理。
- `diskann/src/ivf/clustering/strategy.rs`：借用父桶点、成对分配、按原点序求和；仍用共同 Lloyd 状态机。
- `diskann-benchmark/src/graph_ivf/two_means/`：完整 fit 计时和独立轨迹诊断；任一中间/最终分歧使实验失败。
- 删除仅为旧候选引入的 quantization two-means 模块及 diskann 对它的可选依赖；不改通用 Lloyd 算法。

## 5. 本轮验证顺序

先检查距离 bits（Scalar/V3/V4、全部尾维、错位 slice、近 tie、有限溢出、signed zero、subnormal），再检查每轮 assignment/counts/sum bits/中心/RNG。用之前保存的四个真实分歧点本地回归；原数据坐标不提交到公共仓库。合成 seed7 N16384 D384、seed42 N4096 D384、seed2026 N4096 D896 再跑完整轨迹验证。

然后冻结同一个新 executable，A/B 只切换策略。快速验证协议在测量前固定：

| 项目 | 本轮范围 |
| --- | --- |
| 完整 fit | N={128,1024,4096}，D={192,384,896}，5 轮，连续/分散行；seed7，1 warmup ABBA + 4 实测 ABBA |
| build | YFCC 10000、Enron 1087932、FBV8 958152，全部 base 数据 |
| search | YFCC 全部 100 查询；Enron、FBV8 各原文件前 32 条固定查询，按相同行截取 groundtruth |
| build/search 重复 | 每数据集 1 组 ABBA，共 12 个完整 build+search 进程 |
| 参数 | initial16、batch4096、split2048、5 轮、seed7；graph16/L64；k10、L128、nprobe4/16/64 |
| 测量 | 单线程、affinity16、Normal；每 nprobe 1 次查询 warmup + 1 次实测；保留所有样本 |

查询缩小是本轮“快速 verify”的明确范围，不能称为全查询质量/吞吐结论；A/B 使用完全相同查询，不通过减少 B 工作量制造加速。完整 build 范围和算法参数不变。原小时自动化与旧长实验继续暂停。

完整 fit 主计时包含准备/ISA 选择、scratch 分配、seeding、所有迭代分配/求和/更新及内部对象释放；I/O、FP16 转换和独立诊断在计时外。build 记录 two-means 阶段和总耗时，质量记录 recall、返回数、逻辑扫描量和结果/索引指纹。指纹是诊断证据，不能替代逐元素与逐轮测试。

## 6. 如何判断

任一 bits/轨迹/布局差异均为失败，先修实现，不用“recall 差不多”放行。通过后才报告各自 ABBA 的算术均值与原始样本；一个 build/search block 只支持快速观察，不给显著性或稳定收益保证。无收益也如实记录，不切默认。

当前 provider 是内存实现，扫描量指逻辑 points/lists，不是物理 SSD I/O 或 Mimir 服务性能。编译/测试不与正式测量并行；固定 executable、输入和 source hashes，保存环境干扰信息。新结果放独立目录，与旧失败实验分开。

实现阶段补充：第一版精确 tile 的局部 closure 未内联到 V4 函数，出现逐片段调用与 spill。已改成显式 inline(always) helper，检查生成的 zmm FMA 热循环没有调用。初版完整-fit 原始数据保留在 target/ivf-tiled-exact-20261023；修正后的独立实验目录为 target/ivf-tiled-exact-20261023-final，不混合两个 binary 的时间。
