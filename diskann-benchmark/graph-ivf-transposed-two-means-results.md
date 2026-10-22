# Graph-IVF 转置 two-means：实现与完整对照实验

本次按方案实现 16 点块转置的跨点 SIMD，保留 paired L2 baseline，删除此前 packed、batched 4/8、SGEMM 和 center-difference 等活动实现。两个新候选均为显式 opt-in；默认仍是 paired L2。

完整实验已完成：YFCC、Enron、fbv8_mimir，每个候选各一组 ABBA，共 24 次完整 build+search。每个 search 设置使用全部 100 / 1,000 / 6,380 条查询。以下数字是各自 ABBA 中 A 两次和 B 两次的算术平均值，所有样本保留。

## Build 结果

“耗时变化”正值表示更慢，负值表示更快；索引布局改变时，端到端差异还包含 split、routing 和 reassignment 工作量变化。

| 数据集 | 候选 | baseline build (s) | 候选 build (s) | build 耗时变化 | two-means 耗时变化 | index 指纹相同 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| yfcc | 转置 L2 | 0.007015 | 0.007295 | +3.99% | +55.73% | 是 |
| yfcc | 转置 dot | 0.006666 | 0.006946 | +4.20% | +42.84% | 是 |
| enron | 转置 L2 | 14.189561 | 13.991162 | -1.40% | +2.81% | 否 |
| enron | 转置 dot | 14.793441 | 15.381489 | +3.98% | +8.34% | 否 |
| fbv8_mimir | 转置 L2 | 25.121153 | 25.166882 | +0.18% | +41.73% | 否 |
| fbv8_mimir | 转置 dot | 24.651607 | 25.557506 | +3.67% | +44.55% | 否 |

每一对只有一个 ABBA block；没有独立 block 足以估计稳定置信区间，不能把小幅变化当成已证实收益。完整结果包含两次原始耗时和 split/list 工作量，见 summary.json。

## Search、recall 和逻辑扫描量

Recall 变化的单位是百分点（pp）。Scan 是每条查询扫描的向量数，不是物理 SSD I/O。Search 代码没有改动，但中心和列表可能随浮点决策改变。

| 数据集 | 候选 | nprobe | A / B recall@10 | recall 变化 (pp) | 扫描量变化 | search 耗时变化 | 返回指纹相同 |
| --- | --- | ---: | --- | ---: | ---: | ---: | --- |
| yfcc | 转置 L2 | 4 | 0.8710 / 0.8710 | +0.000 | +0.000% | +6.50% | 是 |
| yfcc | 转置 L2 | 16 | 1.0000 / 1.0000 | +0.000 | +0.000% | +4.19% | 是 |
| yfcc | 转置 L2 | 64 | 1.0000 / 1.0000 | +0.000 | +0.000% | -8.95% | 是 |
| yfcc | 转置 dot | 4 | 0.8710 / 0.8710 | +0.000 | +0.000% | -0.14% | 是 |
| yfcc | 转置 dot | 16 | 1.0000 / 1.0000 | +0.000 | +0.000% | +0.54% | 是 |
| yfcc | 转置 dot | 64 | 1.0000 / 1.0000 | +0.000 | +0.000% | +1.39% | 是 |
| enron | 转置 L2 | 4 | 0.3649 / 0.3643 | -0.060 | +1.119% | +16.91% | 否 |
| enron | 转置 L2 | 16 | 0.5694 / 0.5701 | +0.070 | +0.257% | -4.03% | 否 |
| enron | 转置 L2 | 64 | 0.7702 / 0.7826 | +1.240 | +0.003% | -3.97% | 否 |
| enron | 转置 dot | 4 | 0.3649 / 0.3677 | +0.280 | +2.242% | +5.93% | 否 |
| enron | 转置 dot | 16 | 0.5694 / 0.5673 | -0.210 | +1.840% | +2.88% | 否 |
| enron | 转置 dot | 64 | 0.7702 / 0.7767 | +0.650 | +1.669% | +3.31% | 否 |
| fbv8_mimir | 转置 L2 | 4 | 0.2392 / 0.2447 | +0.552 | +10.534% | +17.31% | 否 |
| fbv8_mimir | 转置 L2 | 16 | 0.4590 / 0.4589 | -0.003 | +5.368% | +3.88% | 否 |
| fbv8_mimir | 转置 L2 | 64 | 0.7139 / 0.7102 | -0.367 | +2.212% | +3.20% | 否 |
| fbv8_mimir | 转置 dot | 4 | 0.2392 / 0.2546 | +1.544 | +11.818% | +11.31% | 否 |
| fbv8_mimir | 转置 dot | 16 | 0.4590 / 0.4595 | +0.049 | +7.634% | +8.34% | 否 |
| fbv8_mimir | 转置 dot | 64 | 0.7139 / 0.7140 | +0.019 | +3.602% | +3.42% | 否 |

每次完整插入全部 base vectors；每条查询返回 k=10。相同策略和输入的重复运行，其成员、中心、recall、scan 和返回指纹一致。各候选与 baseline 的差异仍按实际结果保留，不能用“代码只改 build”推断 search 结果不变。

## 完整 two-means：计入一次转置的成本

E1 核心网格为 18 种 shape：N=128/1024/4096、D=128/384/896、连续/分散存储，5 轮，seed7；每策略 1 次 warmup、7 次实测，共 504 条计时样本。另有 27 种去重后的边界 shape，每候选 4 组 ABBA，共 864 条实测。打包、分配、norm、seeding、全部 assignment/f64 sums/中心更新、scratch 与打包释放均在完整-fit timer 内。

下面固定 N=1024、连续存储、5 轮；比例 >1 表示候选更慢。

| D | paired (μs) | 转置 L2 (μs / 倍数) | 转置 dot (μs / 倍数) | 既有 Lloyd reference (μs / 倍数) |
| ---: | ---: | --- | --- | --- |
| 128 | 122.4 | 131.0 / 1.07× | 124.1 / 1.01× | 129.8 / 1.06× |
| 384 | 276.8 | 563.3 / 2.03× | 550.0 / 1.99× | 593.6 / 2.14× |
| 896 | 569.0 | 1319.1 / 2.32× | 1280.9 / 2.25× | 1318.0 / 2.32× |

核心网格中两个新候选各只有 1/18 个配置略快；27 个边界 shape 的 ABBA 均未获得平均加速。即使 10 轮可摊薄打包成本，本次也没有稳定优于 paired。真实父桶 replay 也显示明显额外开销。

## 真实父桶 replay

独立 baseline build 固定抽取第 0、128、512 个 split 父桶（从零开始），保留坐标和成员顺序，然后用相同 seed7 重新 fit。这里没有恢复原 build 的完整 RNG 轨迹。YFCC 全程只有 1 个父桶，因此后两个序号明确缺失；Enron 和 FBV8 各保留 3 个。所有候选共享同一组父桶。

| 数据集 | 父桶序号 | N | 已插入数量 | 转置 L2 / paired | 转置 dot / paired |
| --- | ---: | ---: | ---: | ---: | ---: |
| yfcc | 0 | 2241 | 10000 | 1.78× | 1.76× |
| enron | 0 | 2305 | 12288 | 1.94× | 1.80× |
| enron | 128 | 2049 | 192512 | 2.09× | 1.96× |
| enron | 512 | 2053 | 729088 | 2.18× | 2.16× |
| fbv8_mimir | 0 | 2276 | 8192 | 2.47× | 2.41× |
| fbv8_mimir | 128 | 2071 | 167936 | 2.37× | 2.31× |
| fbv8_mimir | 512 | 2049 | 655360 | 2.59× | 2.40× |

这些父桶在 seed7 的诊断中，每一轮 assignment、最终中心与 paired 相同；这只覆盖抽样父桶。完整 build 仍可能在其他父桶或不同 RNG 状态处分歧。分散存储的全部结果和每轮诊断同样保存于 fit-summary.json。

## 数值差异已实际出现

- seed7、N=16384、D=384：两个新候选从第 4 轮（JSON iteration=3）首次分歧，最终 33 个 assignment、768 个中心坐标的 bits 不同。首个点的 f64 两距离只差约 2.18e-6；paired 的两个 f32 距离舍入为同值。
- seed42、N=4096、D=384：转置 L2 从第 3 轮分歧，最终 59 个 assignment 不同。seed2026、N=4096、D=896：最终 17 个不同。连续与分散存储的诊断一致。
- 有限消去回归测试：x=10000.25，c0=10001，c1=10000；真平方差为 0.5625/0.0625，dot 展开却出现相等分数而选中心 0。非有限 fallback 不会修正这个有限误差。

上述分歧不是通过放宽容差忽略的；测试和诊断明确记录它们。E1 和抽样父桶 replay 的 fallback 数为 0；完整 build 没有额外插桩统计 fallback。独立极值测试覆盖 overflow fallback。

## 实现和当前判断

每父桶直接打包成 BlockTransposed<f32,16>；每列存 16 个点的同一坐标，四个 f32x8 累加器计算两中心。L2 做直接平方差，dot 做两个点积加中心范数。f64 按原点序求和、seed、固定轮数、空簇和最后跨所有子中心的重分配保持原规则。

完整-fit 数据不支持将当前 U=1 转置实现替换为默认。准备阶段（shape 检查、分配和打包）在不少高维配置占总耗时约四到五成。逐维依赖链、assignment 分离和 f64 求和是后续 profile 的方向，本次没有逐项计时，无法确定各自贡献。反汇编确认用了跨点 SIMD/FMA；L2 循环没有观察到累加器 spill，但没有硬件计数器 profile，不能把原因全部归结为某一瓶颈。

如继续优化，应先测一次转置准备成本、allocation 重用和高维依赖链；U=2 应作为独立数值候选，重新验证，而不是事后删除慢样本或沿用当前结果。

## 验证与范围

34 个 quantization kmeans 测试、12 个 IVF clustering 测试、15 个实验 benchmark 测试、8 个默认 benchmark 测试和 9 个 Python 协议测试通过。fmt、默认/实验 workspace Clippy（warnings denied）、release build 和 AArch64 Windows all-targets check 通过；Scalar/V3/V4 已实机运行，Neon runtime 与旧 x86 模拟器未运行。

默认 IVF 全套为 11 通过、1 个既有 NaN 初始化测试失败：invalid_initialization_and_uninitialized_insertion_fail（tests.rs:626）。其 initialize 与测试源码未改，历史 baseline 已复现；没有将此失败称为通过。本地 selected-source coverage 为 kernel88.11%、clustering90.70%、benchmark91.17%，不是远端 Codecov patch 结果。

基准运行在 AMD Ryzen AI MAX+ 395、Windows、Rust1.97.1、release x86-64-v3，单线程 affinity0x10、Normal priority；OS 管理频率，HP Optimized 电源方案。

正式 ABBA 记录的并发 cargo/rustc/benchmark 干扰条目：0。未测量其他桌面进程、温度或频率变化，不能断言独占机器。全部原始慢样本保留。

## 可复核证据

- [summary.json](results/transposed-two-means-20261022/summary.json)：24 次全部结果、各自 ABBA 对照、质量与工作量。
- [fit-summary.json](results/transposed-two-means-20261022/fit-summary.json)：380 个策略/shape/seed/父桶组合、完整样本和每轮分歧。[raw-samples.zip](results/transposed-two-means-20261022/raw-samples.zip) 保留原 input/result/log/metadata；不发布数据集坐标。
- [provenance.json](results/transposed-two-means-20261022/provenance.json)、[sha256.json](results/transposed-two-means-20261022/sha256.json)、[final-verification.json](results/transposed-two-means-20261022/final-verification.json)：冻结 executable、源码/输入/数据 hashes；实验结束再次逐项核对。原始坐标快照与 executable 位于本地 target/ivf-transposed-20261022。
- 冻结 executable SHA256：275661f6d0d47940b3fab2bd560dec2302d987d6ba332198f739c268a0b4b088。source_parent 为 5b790d821e3d5f7349f17f95036a142036d9cb66，加该目录 source.patch 与 source/ 快照；未拿父提交冒充新代码提交。

继续保留现有第二层 stack：u/xiajiang/graph-ivf-two-means-simd → u/xiajiang/graph-ivf-static-benchmark；不改第三层。原七策略 288-run 实验仍暂停，旧程序及所有已完成样本保留，未与本次统计混合。

本轮按用户“一组对照”执行 24 次 E2，而不是原交接方案的 96 次。实际在用户指定的当前 worktree 替换旧实现；本轮没有另开直接基于 Aditya 的移植 PR，因此也没有该移植后的额外48次验证。
