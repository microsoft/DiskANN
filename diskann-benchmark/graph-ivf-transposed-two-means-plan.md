# Graph-IVF：参考现有 Lloyd 内核的 two-means 实现与实验方案

> 2026-10-23 实施记录：已按后续指令在当前 stack 替换旧候选，完成两个转置候选、Lloyd reference、父桶 replay 及 24 次全量 build/search ABBA。实际范围与测量结果见 [实验报告](graph-ivf-transposed-two-means-results.md)。下文保留原设计时的起点、96 次方案和交接设想，不代表当前实现状态。

日期：2026-10-22。状态：设计与交接文档；新内核尚未实现，本方案没有新增加速实测。

目标是验证：把父桶打包为块转置布局，让 SIMD 的 lane 对应不同点，能否在计入打包、分配、f64 求和及中心更新后，缩短完整 two-means 和 Graph-IVF build 时间。最终代码 PR 的目标分支为 `u/adkrishnan/ivf-index-2`。新 session 使用独立 worktree 实施；本次只交付方案。

## 1. 已核实的代码起点

以下版本来自本次读取的本地 Git 状态；没有 fetch，不能据此断言它们是远端最新版本。

| 对象 | 本次核实的版本 / 状态 | 对新实验的影响 |
| --- | --- | --- |
| 当前参考 worktree | `C:/Users/xiajiang/.codex/worktrees/7dd3/DiskANN` | 已有 paired、batched、SGEMM 及 benchmark，供阅读和移植参考 |
| 当前参考分支 | `u/xiajiang/graph-ivf-two-means-simd`，`5b790d821e3d5f7349f17f95036a142036d9cb66` | clustering 与计时模块已经分离 |
| Aditya 的本地及 origin 跟踪引用 | `u/adkrishnan/ivf-index-2`，`83c01093a3a41705791a8b12803f3340967dd89b` | IVF fitter 仍在单文件 `diskann/src/ivf/clustering.rs`，使用逐点独立 L2 |
| benchmark 分支 | 本地 `1c02ec23`；origin 跟踪引用 `ec2b7afa` | 新 session 必须核对需要哪版 benchmark，不能把本地分支名当成最新代码 |
| 历史 ABBA executable | 文档记录来源为 `c7e54d08` | 与当前源码不同；旧时间不能直接当成本实验的 A 样本 |

现有 Lloyd 内核在上述 Aditya 起点与当前参考 HEAD 之间没有 diff。真正需要处理的接入差异主要在 IVF 与 benchmark。

关键入口：

- [Lloyd 分配内核：distances_in_place](C:/Users/xiajiang/.codex/worktrees/7dd3/DiskANN/diskann-quantization/src/algorithms/kmeans/lloyds.rs:28)：块转置输入，每组 16 点，每次处理两个中心。
- [四个 SIMD 点积累加器](C:/Users/xiajiang/.codex/worktrees/7dd3/DiskANN/diskann-quantization/src/algorithms/kmeans/lloyds.rs:77)：两个点组乘两个中心。
- [现有量化 k-means 的中心更新](C:/Users/xiajiang/.codex/worktrees/7dd3/DiskANN/diskann-quantization/src/algorithms/kmeans/lloyds.rs:345)：f64 求和，但空簇中心会变成零。
- [IVF 的共同 Lloyd 循环](C:/Users/xiajiang/.codex/worktrees/7dd3/DiskANN/diskann/src/ivf/clustering/two_means.rs:43)：共同初始化、迭代、求和与空簇规则。
- [当前按点展开的 batched 内核](C:/Users/xiajiang/.codex/worktrees/7dd3/DiskANN/diskann/src/ivf/clustering/batched.rs:22)：lane 对应同一个点的不同坐标。
- [完整 fit 的 benchmark 计时](C:/Users/xiajiang/.codex/worktrees/7dd3/DiskANN/diskann-benchmark/src/graph_ivf/two_means/measurement.rs:28)：包含 preparation、fit 和父桶打包内存释放。
- [历史实验数值契约与计时口径](C:/Users/xiajiang/.codex/worktrees/7dd3/DiskANN/diskann-benchmark/graph-ivf-two-means-experiments.md)、[全查询 ABBA 协议](C:/Users/xiajiang/.codex/worktrees/7dd3/DiskANN/diskann-benchmark/graph-ivf-two-means-abba.md)。

上述源码链接指向当前参考 worktree，未来可能变化。正式报告还应记录完整 commit SHA 和变更文件哈希。

## 2. 本轮要回答的问题

1. 与已有 paired L2 相比，跨点 SIMD 能否减少中心加载、逐距离横向归约和循环开销？
2. 新布局的收益能否覆盖每个父桶的一次转置成本，以及按原点顺序进行 f64 求和的成本？
3. 直接平方差与点积展开，哪一种在真实桶大小、384/896 维数据上更合适？
4. 是否出现边界点分组、最终中心、split 数量、recall 或扫描量变化？变化来自数值计算还是程序错误？
5. 候选回移到 Aditya 分支后，是否仍有相对于该分支原始实现的收益？

默认执行单线程实验。块内 SIMD 和独立累加链属于一个核内的并行；多核构建吞吐可以作为后续独立实验，不混入本轮结论。

## 3. 可以借用什么，必须保留什么

已有 Lloyd 已经是“打包 + SIMD + FMA”的专用实现，不需要调用 SGEMM 才能实现向量化。推荐复用块转置容器和内核组织方式，保留 IVF 的迭代语义。

| 事项 | 已有量化 Lloyd | 本轮 IVF 方案 |
| --- | --- | --- |
| 数据准备 | 调用 `lloyds()` 时转置一次，跨迭代复用 | 每个 split 父桶准备一次，跨该次 fit 的迭代复用 |
| 初始化 | 调用方传入中心 | 保留 IVF 的两次随机抽样及 RNG 消耗顺序 |
| 迭代次数 | `0..max_reps` | 保留 `iterations.max(1)`，不增加提前收敛 |
| 距离 | 点积加点和中心范数 | 实验比较点积展开与直接平方差 |
| 求和 | 按点序使用 f64 | 原点序、f64 累加、同一除法与 f32 转换 |
| 空簇 | 零和除以 `max(count,1)`，得到零中心 | 保留上轮中心 |
| 最近中心比较 | SIMD 严格小于；异常距离需要额外注意 | 明确 tie、无效值及 fallback 规则 |
| split 后重分配 | 不涉及 IVF 的多个父桶 | 保留现有跨所有新子中心的最终重分配 |

不要直接替换成完整 `lloyds()`：这会改变空簇与零迭代行为，也绕过当前 IVF 的共同状态转移。即使 f64 求和代码相同，assignment 改变后，中心也会改变。

## 4. 数据布局和计算结构

原始点可能分散在内存中，以 `PointRef` 借用。按逻辑点序，每 16 点组成一块，块内按坐标存放：

```text
原始：x0 = [x0[0], x0[1], ...]
      x1 = [x1[0], x1[1], ...]
      ...

转置块：
[x0[0], x1[0], ... x15[0]]
[x0[1], x1[1], ... x15[1]]
...

处理坐标 d：
Xlo = [x0[d], ... x7[d]]       // 8 lanes = 8 个点
Xhi = [x8[d], ... x15[d]]
C0  = [c0[d], ... c0[d]]       // 同一坐标广播
C1  = [c1[d], ... c1[d]]
```

```mermaid
flowchart LR
    P[父桶 PointRef 和原点序] --> T[一次块转置]
    T --> A[16 点对两个中心的 SIMD 分配]
    C[本轮两个中心] --> A
    A --> M[按原点序的 assignment]
    P --> S[f64 sums 和 counts]
    M --> S
    S --> U[更新非空中心]
    U --> C
    U --> F[固定轮数后返回两个中心]
```

点坐标不随 Lloyd 迭代变化，可以复用转置数据；中心会更新，中心范数或差值必须按轮更新。工作缓冲区绑定一次父桶 fit，不进入 provider 的持久化格式。

建议直接从 `PointRef` 打包到 `BlockTransposed<f32,16>`，避免先额外复制整份 row-major 再转置。现有 `from_matrix_view/from_strided` 接口不能直接表示任意分散的 `PointRef`；用容器的 fallible 分配和可变视图填充，或增加最小的逐行输入构造接口。

转置存储量为 $4D\cdot16\lceil N/16\rceil$ 字节，另有 assignment、两个中心和 f64 sums 等缓冲。例如 1,024 个 128 维点需要 512 KiB 转置空间；一个 16 点、896 维的块为 56 KiB。不要假设整个父桶或每个块一定能留在 L1/L2。

## 5. 候选内核：先三个，再按证据扩展

以下名字是计划新增的实验标识，目前尚不存在。

| 实验标识 | 实现与目的 | 定位 |
| --- | --- | --- |
| `transposed_lloyd_reference` | 适配现有 `distances_in_place`，只替换分配步骤，继续使用 IVF 的初始化、求和和更新 | 完整 fit 的参考组；确认已有实现的实际效果 |
| `transposed_l2_16` | 相同布局，直接累加平方差，lane 跨点 | 首选实现候选 |
| `transposed_dot_16` | 相同布局，两个点积加中心范数，只输出分组 | 第二个实现候选 |

### 5.1 首选：transposed_l2_16

每个 16 点块使用四个 `f32x8` 累加器：前 8 点到 c0/c1，后 8 点到 c0/c1。逐维累加，每个 lane 最终就是一个点到一个中心的完整距离，不需要对每个距离做横向求和。

```rust
// 示意伪代码：不是可直接编译的 API。
for d in 0..dim {
    let (x_lo, x_hi) = load_16_points_at_dimension(d);
    let c0 = broadcast(center0[d]);
    let c1 = broadcast(center1[d]);
    let a = x_lo - c0;
    let b = x_hi - c0;
    let c = x_lo - c1;
    let e = x_hi - c1;
    s00 = a.mul_add_simd(a, s00);
    s01 = b.mul_add_simd(b, s01);
    s10 = c.mul_add_simd(c, s10);
    s11 = e.mul_add_simd(e, s11);
}
// 分别按 lane 比较 s10/s00、s11/s01，再按原点序写 assignment。
```

这没有减少数学上的 $2ND$ 项距离累加；改变的是访存组织、每次 SIMD 处理的对象和归约方式。源代码中的变量也不保证一直驻留物理寄存器，需要查看编译结果确认。

### 5.2 transposed_dot_16

参考原 Lloyd 的四个点积累加器，随后比较：

$$
s_j(x)=\|c_j\|^2-2x^\top c_j.
$$

同一点的 $\|x\|^2$ 对排名是公共项，因此这个候选不计算它，也不生成残差总和或完整距离矩阵。每轮只重算两个中心范数，用 SIMD 比较输出 assignment。

原 `distances_in_place` 则计算点范数、完整距离及 residual。两者数学排名一致，但浮点操作不同，不能要求它们逐 bit 一致，也不能把省掉公共项说成纯粹无数值影响的代码整理。

### 5.3 reference 的适用边界

reference 用来测原内核，不把原内核的异常处理当作 IVF 的数值保证。该 API 只返回最近中心和总 residual，不暴露逐点两列分数：双方距离为非有限值时，可能留下非法中心编号。

因此 reference 先用于受控、有限范围的完整-fit 对照。验证输出编号、residual 和中心；不满足时报告该样本不适用或失败，不能把非法编号用于求和，也不能将异常样本计为加速成功。reference 不列入本轮正式 build/search 的发布候选。

### 5.4 高维性能检查

原文件注释明确提示该组织在高维时可能表现不好。可能原因之一是逐维依赖链长，但这是待测推断。

首版固定展开度 U=1。若 profile/反汇编表明依赖链限制吞吐，再增加 U=2 的独立候选，分维度累加后用固定顺序合并。对于 16 点块，U=1/2/4 分别需要 4/8/16 个距离累加器，另需输入与中心等临时寄存器；AVX2 上盲目提高展开度容易产生 spill。

新增展开度必须使用独立策略名或记录在冻结协议中，并重新验证数值。不要同时搜索块宽、展开度、切换阈值、求和方式，再只保留最快组合。

## 6. 代码分层与改动范围

推荐先使用现有依赖层次，避免将 benchmark 逻辑放进算法库。

| 层次 | 计划变更 | 职责 |
| --- | --- | --- |
| `diskann-quantization` | 在 `algorithms/kmeans/` 下增加双中心 primitive，复用 `multi_vector::BlockTransposed` 与 `diskann-wide` | 布局、向量计算、shape 检查、逐点无效分数标识；不持有 IVF ID、RNG、计时器 |
| `diskann::ivf::clustering` | 增加转置 preparation/assignment adapter，接入共同 fitter | 每父桶准备一次，保持原点序；执行 fallback；复用 f64 sums、中心更新与最终重分配 |
| `diskann-benchmark` | 新策略名、完整-fit 对照、诊断输出、参数化 ABBA | 数据准备、计时、真实性与质量验证、结果统计 |

当前 `diskann` 没有依赖 `diskann-quantization`。此方案需要新增可选依赖，使用 `default-features = false`，放在实验 feature 后；Tier 3 依赖 Tier 2 符合当前仓库层次。benchmark 已依赖 quantization。既有范数 helper 是 `pub(crate)`，把新 primitive 放在同 crate 可复用它，不必扩大 helper 的公开范围。

低层错误使用精确的小错误类型，由 IVF adapter 转成 `ANNError`；不要让底层 primitive 反向依赖 `diskann`。若后续不希望正式 IVF 依赖 quantization，可另立共享 primitive 抽取 PR，不在性能实验中同时移动整个模块。

参考分支的接入点是 `clustering/strategy.rs`、`assignment.rs` 和 `two_means.rs`；Aditya 当前起点只有 `clustering.rs`，需要最小化抽取可替换的 assignment 步骤。不要把参考分支的完整重构当成 Aditya 已有接口。

内存与生命周期要求：

- 转置、assignment scratch 每父桶分配一次，跨迭代复用；不在逐点或逐维循环里分配。
- 最后不足 16 点的块填充可安全读取的 padding，但仅将真实行写回和计入 counts/sums。
- 检查 `ceil(N/16)*16*D*sizeof(f32)` 的整数和布局溢出，包括 padding 增量。
- f64 求和先继续按原始 `PointRef` 顺序读取；它可能成为剩余瓶颈，但本轮不同时改变求和顺序。
- 只返回分组，不在生产路径存储 $N\times2$ 分数矩阵；诊断可以额外保存，但在独立的未计时运行中进行。

## 7. 数值契约和 correctness 验证

### 7.1 必须保留的算法行为

输入点值和顺序、初始化 RNG、固定迭代数、同分优先中心 0、空簇保留原中心、按点序的 f64 求和、均值转换和 split 后所有子中心之间的最终重分配，全部沿用相应基线。

`transposed_l2_16` 也不是逐 bit 等价优化：原 L2 可能有多个坐标累加器和树形归约，新版逐维归约改变了舍入顺序。点积候选还增加范数展开的消去风险。它们均为显式 opt-in 数值候选，首个 PR 不自动切换默认实现。

### 7.2 无效分数和边界点

- 对两个正式候选，kernel 输出每个真实点的有效性；任一距离/score 非有限，adapter 用基线两距离计算重做该点。
- 当前参考分支可调用 paired L2；直接基于 Aditya 的代码使用现有两个 `SquaredL2::evaluate`，不能假设 `evaluate_pair` 已存在。
- 两中心完全相同且坐标有限时，同分选择与空簇规则必须稳定；padding 不参与统计。
- 非有限 fallback 只解决非有限计算，不解决有限消去误差。边界点可能仍分错；不引入未经证明的 epsilon 来隐藏差异。
- 首版沿用当前 finite-input 契约，不自行定义 NaN 输入排序或悄悄归一化输入。无效输入检查的计时归属要一致。

### 7.3 验证矩阵

| 类别 | 必测内容 |
| --- | --- |
| shape 与尾块 | N=2/7/8/9/15/16/17/31/32/33；D=1/7/8/9/31/32/33/128/384/896；维度错误、空输入、分配大小溢出 |
| 分组正确性 | 小整数和明显分离点，用独立 f64 直接平方差 oracle；比较真实 ID、assignment、counts，验证没有遗漏或重复 |
| 迭代语义 | 0/1/5/10 轮；随机初始化消耗；重复点、相同中心、空簇保留；scratch 重用不残留上一父桶状态 |
| 数值差异 | 近等距点、大小量级混合、正负零、次正规数、有限大坐标导致的溢出；记录每轮分组分歧和首次出现位置 |
| 已有消去例子 | x=10000.25、c0=10000、c1=10001；不得用平均误差或容忍阈值抹去反例 |
| 内存安全与 ISA | padding 不越界，分散与连续行逻辑结果一致；按 diskann-wide 指南覆盖 scalar、V3、V4、AArch64 |

比较采用两种视角：同一轮相同中心下的 assignment 差异，以及各自完整迭代后的最终中心/分区差异。只比较最终 paired-L2 assignment 会漏掉中间迭代分歧。

正常数据的近边界测试记录误差、间隔与 assignment，不把普通 f64 计算宣称为任意输入的精确实数证明。独立小整数 oracle 承担明确预期结果的测试。

## 8. 实验分层与测量边界

### E0：正确性与小规模 pipeline

先通过第 7 节测试，再运行小数据 build/search smoke，确认策略进入真实 split 调用链。编译成功或一个 kernel 单测通过，都不等于完整实现已验证。

### E1：完整 two-means 对照

A 固定为本次冻结 executable 中的 `paired_point`。先在核心网格比较现有七种策略以及三个新增策略，定位转置布局、计算公式和通用 Lloyd 开销的影响。

核心网格：N={128,1024,4096}，D={128,384,896}，5 轮，两种布局，共 18 种 shape；seed=7，1 次明确标注的 warmup、7 次实测。策略顺序轮换并反转，保存所有原始样本。

对通过正确性检查的两个新实现，再以 ABBA 确认边界与摊销效果：

- 基础切片：D=384、5 轮、连续布局，N={32,128,512,1024,2048,4096,16384}。
- 维度切片：N=1024、5 轮、连续布局，D={64,128,192,384,896,1536}。
- 迭代切片：N=1024，D={128,384,896}，轮数={1,5,10}，连续和分散布局。
- 相同配置去重；每个候选与 A 做 4 个 ABBA block。每个位置重新执行完整 fit，包括 preparation；不能复用上一位置已打包的数据。
- seed=42、2026 另做数值稳健性检查，不与 seed=7 的计时样本混成一次结果。

每次完整 fit 的主计时范围：

```text
shape 检查 + 分配 + 打包 + 必要范数准备 + seeding
+ 所有 Lloyd 轮次的距离/比较/分组/f64 求和/更新
+ 内部 scratch 和打包对象释放
```

返回给调用方的中心等输出允许继续存在，但各组的计时边界必须一致。数据文件读取、FP16 转换、合成数据生成在计时外；从已读数据转为候选专用布局必须在计时内。

主报告给完整耗时；分配内核、求和、范数和打包 breakdown 用独立 profile 或插桩运行解释原因，不在热循环里加入逐点计时再当作主成绩。

### E1b：真实父桶 replay

均匀合成数据或从全库随机抽样，不等同于 Graph-IVF 实际 split 父桶。追加一次独立、未计时的基线 build，导出预先固定抽样规则选出的父桶坐标和成员顺序，覆盖不同 N 与 build 阶段。

所有候选读取同一份父桶快照，并使用相同 seed 重新初始化。该 replay 保留真实局部点分布，但不是重放原 build 的完整 RNG 轨迹；明确记录这一区别。导出与校验不放入正式计时 build，不让候选各自挑选更容易的父桶。

## 9. E2：三个数据集的正式 build/search ABBA

下表是已有仓库协议记载的 shape；本次没有重新读取数据文件 header。新 session 在 protocol 冻结前必须检查实际文件 header、长度、类型和哈希。

| 数据集 | Base 数量 | D | 每个 search pass 的查询数 | 输入类型 |
| --- | ---: | ---: | ---: | --- |
| YFCC perf workload | 10,000 | 192 | 100 | FP32 |
| Enron | 1,087,932 | 384 | 1,000 | FP16 |
| fbv8_mimir | 958,152 | 896 | 6,380 | FP16 |

FP16 在计时外精确提升为 FP32，不增加归一化；同一数据和 groundtruth 文件用于所有组。

固定设置沿用已记录的可复现 workload：

| 设置 | 值 |
| --- | --- |
| A | 同一 executable 内的 `paired_point` |
| B | 分别为 `transposed_l2_16`、`transposed_dot_16` |
| insert batch / 初始中心 / split threshold | 4096 / 16 / 2048 |
| Lloyd 轮数 / seed | 5 / 7 |
| centroid graph | degree 16，build L64 |
| search | k10，L128，nprobe={4,16,64} |
| 并行与 ISA | 单线程，release x86-64-v3；在支持该 ISA 的机器上运行 |
| 搜索重复 | 每设置一次全查询 warmup，一次全查询实测 |
| ABBA | 每个数据集、每个 B 各 4 个 A-B-B-A block |

合计 $3\times2\times4\times4=96$ 次进程调用。每次重新完成完整 build，然后对每个 nprobe 跑全部查询。每一对的 8 个 A、8 个 B 都属于自己的时段，不借用旧实验或另一候选的 A 样本。候选次序按 block 轮换。

同一 frozen executable 包含所有候选；正式开始后不重编译、不更新代码、不修改输入。工作目录、输出目录、target 目录和二进制均与其他实验分开。

新 worktree 不隔离 CPU、内存带宽和磁盘。使用空闲机器或安排不重叠时段；检测已有 benchmark/cargo/rustc 运行时，不终止其他 session 的进程，不同时启动正式计时。Windows affinity 在核实有效后固定，记录 CPU 型号、SMT、频率/功耗环境、RUSTFLAGS、编译器版本和 Normal priority。所有异常慢样本保留并标注环境信息。

静态 benchmark 的边界也必须保留：插入期间采用 benchmark adapter 的 centroid routing，最终再建 centroid graph；provider 在内存中。结果代表此 build/search 路径，不代表 Mimir 持久化或真实 SSD I/O。

## 10. benchmark 和 runner 需要补哪些功能

以下差距以参考 commit `5b790d82` 为准。当前工作区可能同时有其他 session 更新 runner；新 session 应先检查最新已提交实现，复用已完成的能力。只有验证支持后，才能传入下列计划参数。

1. `GraphIvfTwoMeans` 增加可选策略列表与 ABBA 顺序模式。当前矩阵输入会遍历固定的七种策略，并不支持新策略子集或完整-fit ABBA。
2. 在 `TwoMeansStrategy` 增加新标识及库策略映射；更新 exhaustive match，保持旧输入兼容。
3. `run_ivf_abba.py` 增加 `--reference` 和可重复的 `--candidate`，保留旧默认行为。当前候选集合是硬编码，reference 固定 paired。
4. analyzer 从 protocol 读取 reference、候选数和预期 block 数，处理任意合法配对；不得继续假设只有旧七种策略或固定 288 次。
5. 记录 `schema_version`、full SHA、executable/input/dataset 哈希、ISA、线程/affinity、策略、块宽、展开度、实际准备字节和 fallback 数。
6. 计时结果与每轮 assignment 等详细诊断分开。中间分组诊断用另一次确定性运行，保持相同输入、seed 和策略。
7. 对 resume、失败、部分 block、新候选解析、完整查询校验增加针对性测试；不覆盖原实验输出，不把缺失结果补成零或成功。

当前完整-fit harness 的 `preparation_ns` 包含 shape 检查及打包。历史 `packing_ns` 口径不同；若本实验增加范数准备，需在 schema 中写清楚边界，不能沿用旧字段名后合并历史样本。

实现完上述扩展后，运行流程示例为：

```powershell
# 在新实验 worktree 中执行；先完成依赖/lockfile 更新，再冻结构建。
$env:RUSTFLAGS = '-C target-cpu=x86-64-v3'
$env:CARGO_TARGET_DIR = 'target/ivf-transposed-build'
cargo build --release --locked -p diskann-benchmark --features ivf-two-means

# 以下 --reference / --candidate 是计划新增的参数。
python diskann-benchmark/scripts/run_ivf_abba.py --exe target/ivf-transposed-build/release/diskann-benchmark.exe --output-directory target/ivf-transposed-e2/runs --dataset yfcc=target/ivf-transposed-e2/yfcc.json --dataset enron=target/ivf-transposed-e2/enron.json --dataset fbv8_mimir=target/ivf-transposed-e2/fbv8_mimir.json --reference paired_point --candidate transposed_l2_16 --candidate transposed_dot_16 --rounds 4

python diskann-benchmark/scripts/analyze_ivf_abba.py target/ivf-transposed-e2/runs --output target/ivf-transposed-e2/summary.json
```

模板及数据目录需要在新 worktree 中准备并验证；不要依赖当前 worktree 中未提交的 Enron 输入文件。完整查询校验必须检查真实查询数、requested nprobe 和每查询返回 k 个结果。

## 11. 如何分析，什么结果才能支持采用

对于第 r 个 block，先计算：

$$
A_r=(t_{A1}+t_{A2})/2,\qquad B_r=(t_{B1}+t_{B2})/2.
$$

主报告给原始样本、算术均值、标准差、每个 block 的 $100(1-B_r/A_r)$，以及所有完整 block 的 $100(1-\bar B/\bar A)$。不要把“减少 20% 耗时”和“快 1.2 倍”混用。4 个 block 的区间只能作描述，查询条数和 Lloyd 内循环次数不算独立重复。

同时记录：

| 类型 | 必需指标 |
| --- | --- |
| 完整 fit | total、preparation、packed/scratch 字节、fallback 数；独立 profile 的 assignment/sum/update 成本 |
| build | two_means_seconds、ivf_seconds、total_seconds、reassignment、centroid graph；墙钟与进程 CPU 时间 |
| 算法工作量 | inserted、splits、split_points、split_parent_histogram、lists、empty_lists、max_list_size |
| 质量与搜索 | recall@10、完整查询数、返回数量、扫描 points/lists、centroid comparisons、search 总时间/QPS/延迟 |
| 数值与结构 | 每轮 assignment 分歧、最终中心差异、最终分组、index/result fingerprint；重要分歧保存可重现样本 |

fingerprint 用于快速诊断，不是逐元素相等的证明；发布判断需要实际比较关心的结构与质量指标。

判断分两级：

- **实验 PR 可交付**：实现和数值契约明确，正确性通过，真实数据与完整原始结果可复现；即使没有加速也可如实形成实验结论。
- **建议替换默认实现**：完整 two-means 和 total build 都有可重复收益，且固定参数下没有隐藏的 recall/扫描量代价。相同输入和 seed 下，任何 recall 下降或扫描增加都要明确列出，不能仅以耗时下降判为通过。若允许质量容忍度，应在另一个 protocol 开始前约定。

不能承诺本候选必然保持中心 bits、分区、recall 或扫描量完全一致。若必须严格保持这些结果，需要另行设计保持原归约树的精确候选；本轮不把“差异很小”改名为“完全等价”。逻辑扫描计数不能证明物理 I/O 不变。

防止错误归因：

- 不减少迭代、跳过难点或缩短查询集来得到更快结果。
- 不把 packing/norm/scratch 搬出主计时，不让候选使用提前准备而 A 现场准备。
- 不删除慢样本，不按结果选择 seed、桶或 nprobe，不将部分完成当作完整实验。
- 候选改变分区后可能减少后续 split 工作；完整 build 的收益可以报告，但不能全部归因于内核更快。真实父桶 replay 用来分离这个因素。
- 不为追回 recall 单独增加某一候选的 nprobe 后仍称同参数比较；如探索 recall/成本曲线，另立配对实验。

## 12. 新 session / worktree 与 PR 的组织

### 12.1 区分实验依赖和代码 PR 的目标分支

最终代码 PR 按本次要求直接以 `u/adkrishnan/ivf-index-2` 为 base。当前完整 benchmark、paired primitive、strategy seam 并非该 base 已有代码，不能把当前整个 stack 的 diff 包装成一个新的独立优化 PR。

建议使用以下两条线：

```text
Aditya: u/adkrishnan/ivf-index-2 @ 固定 SHA
  ├─ 现有 benchmark / two-means stack（参考与实验宿主）
  │    └─ codex/graph-ivf-transposed-two-means-eval
  │         新 session 的实验 worktree：接入候选、完整对比与报告
  └─ codex/graph-ivf-transposed-two-means
            选定方案的最小实现 + 测试；draft PR base = Aditya 分支
```

新 session 先创建自己的实验 worktree，从经核实的 two-means/benchmark 宿主起点工作；若采用当前参考起点，就是 `5b790d82`。核对 benchmark 的后续更新是否需要整合，完成整合后才冻结 A/B executable。现有 worktree 只读参考。

低层 primitive 尽量单独提交，以便移植到干净的 Aditya 基础。共享 fitter 接入和 benchmark 更改分开提交，避免实验工具依赖混入最终代码 PR。准备发布时，再建立独立的干净代码 checkout/worktree；不在正在跑实验的分支上 rebase 或切换。

### 12.2 建议提交边界

| 顺序 | 内容 | 应进入最终代码 PR |
| --- | --- | --- |
| 1 | 两中心转置 primitive、布局与尾块、低层测试 | 是 |
| 2 | IVF 的最小 assignment 接入、feature/策略选择、原语义测试 | 是，适配 Aditya 当时的接口 |
| 3 | reference 对照、benchmark 参数、诊断和 ABBA 扩展 | 视 benchmark 依赖单独提交/PR；不强行带入代码 PR |
| 4 | 实验报告、可复现小输入、版本与结果索引 | 精简报告进入相关 PR；大型原始数据不提交 |

每个提交都有独立目的，但第一轮不需要拆成多个优化 PR。最终先提交一个选定方案的 draft PR；默认实现切换必须由实测和数值契约决定。

如果 Aditya 在实施期间合入现有依赖，重新核对 base 后简化堆叠。若最终选择直接依赖当前 two-means PR，则新 PR 的直接 base 应改为该分支并明示依赖；不能嘴上说直接基于 Aditya，实际却带着整条既有 stack。

### 12.3 移植后再验证

实验宿主的 A 是 paired；Aditya 当前 A 是两个独立 L2 调用。最终代码 PR 必须额外验证“Aditya 原实现 vs 最终候选”，不能用对 paired 的实验代替。

在包含最终代码和同一版本 benchmark adapter 的冻结验证宿主中，保留两种路径，只切换两中心分配策略。用 3 个数据集、每对 4 个 ABBA block、全部查询验证一个发布候选，共 48 次调用。记录最终候选 source SHA、adapter SHA 和 executable hash；代码移植或编译变化后的性能不得直接沿用 E2 时间。

若 base 更新改变了搜索、split、provider 或计时边界，重建共同基线并单独报告，不合并不同版本的样本。

## 13. 实施顺序和完成条件

1. **冻结起点**：新 worktree、独立分支/target/结果目录，记录 base 与宿主 SHA；核实数据、依赖和其他实验的运行安排。
2. **实现最小路径**：一次转置、U=1 的 L2 与 dot 内核、共同 fitter adapter、非有限 fallback。
3. **正确性**：第 7 节测试、不同 ISA 路径、small build/search；修复后再进入正式计时。
4. **E1/E1b**：完整-fit 对照和真实父桶 replay；用 profile 解释高维与打包成本，必要时另立 U=2 候选。
5. **E2**：冻结协议和 executable，运行 96 次全查询 ABBA，按完整 block 分析。
6. **整理结果**：输出全部成功/失败/不适用记录、速度与质量变化、推荐范围或无收益结论。
7. **包装代码 PR**：选定方案移植到 Aditya 基础，完成最终 48 次验证，再开以该分支为 base 的 draft PR。

实现后的针对性检查起点如下；新增测试名称以实际代码为准。架构模拟用于正确性，模拟器结果不用于性能排名。

```powershell
cargo test -p diskann-quantization algorithms::kmeans
cargo test -p diskann --features ivf-experimental ivf::clustering
cargo test -p diskann-benchmark --features ivf-two-means graph_ivf
cargo test -p diskann --no-default-features ivf
cargo fmt --all --check
```

最终提交前执行当时 AGENTS 要求的 fmt、workspace clippy 和相关覆盖检查。按 [diskann-wide 的跨架构指南](C:/Users/xiajiang/.codex/worktrees/7dd3/DiskANN/diskann-wide/README.md) 验证不支持 AVX 的路径与 V3/V4/AArch64；缺失的环境覆盖如实记录。

最终交付物：可审阅的实现与测试、冻结 protocol、每次 invocation 的 input/result/log/metadata、汇总 JSON、中文分析报告、明确 base 的 draft PR。报告须区分“已测到的收益”“实现导致的数值差异”和“尚未验证的 Mimir/物理 I/O 效果”。

## 14. 可复制到新 session 的交接任务

```text
请依据以下方案实现并验证 Graph-IVF 的跨点 SIMD two-means：
C:/Users/xiajiang/.codex/worktrees/7dd3/DiskANN/diskann-benchmark/graph-ivf-transposed-two-means-plan.md

在你自己的独立 worktree 中工作，先核对并记录实际 base/宿主 SHA。
只读参考现有 two-means worktree，不更改其源码、实验、自动化或运行进程。
复用 diskann-quantization 既有 Lloyd 的块转置/FMA 组织，先实现
transposed_l2_16 和 transposed_dot_16，原 distances_in_place 作为受控参考。
保留 IVF 的 RNG、固定轮数、f64 原点序求和、tie、空簇与最终重分配规则；
把浮点差异作为显式实验契约，不宣称逐 bit 等价。

按文档完成正确性、完整-fit、真实父桶 replay 和 YFCC/Enron/fbv8_mimir
的全查询 ABBA。正式性能运行与其他实验错开，保存全部原始样本。
将结果整理为报告；若无收益或有质量代价，同样明确报告，不强行切换默认。
实验宿主用于复用 benchmark；最终实现整理为最小代码 PR，直接以
u/adkrishnan/ivf-index-2 为 base，移植后补做同版本完整验证。
实验完成、代码可审阅且验证记录齐全后，再创建 draft PR。
```
