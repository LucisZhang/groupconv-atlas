# Four evidence-backed questions / 四个可追问的中英问题

Evidence version: 2026-09-08. Answers distinguish the new RTX 4090 10×30 atlas from the older RTX 4090 D 5×30 atlas and six-point confirmation. Implementation and evidence preparation used Codex assistance. These are rehearsal materials; the owner attests to independent explanation and participation in topic selection, design, code modification, experiment configuration and result checking. This OWNER_ATTESTED statement has not been independently assessed. See [contributions](CONTRIBUTIONS.md).

## 1. 为什么优化分组卷积，实际改了什么？ / Why optimize grouped convolution, and what changed?

**中文。** 分组数增加后，每个输出的通道归约缩短，索引、调度和数据复用的相对成本会变化。项目比较通用k0、专用空间映射k1、共享halo k2和每线程四输出的寄存器复用k3。新4090的80点上，k0/k3几何平均为Graph 3.8718、同步API 3.4490；相对调优框架只有0.7384/0.6448，整体仍更慢。k2只覆盖每组通道1/2/4的30点，相对k1的Graph分类为13赢、8输、4平、5不确定，共享存储并非普遍有效。

**English.** More groups shorten each output's channel reduction, changing the relative cost of indexing, scheduling and reuse. The project compares general k0, specialized spatial mapping k1, shared-halo k2 and four-output register reuse k3. Across the new 4090's 80 shapes, k0/k3 geometric means are 3.8718 for Graph and 3.4490 for synchronized API. Tuned-framework/k3 ratios are only 0.7384/0.6448, so the custom kernel loses in aggregate. k2 supports 30 shapes with one, two or four channels per group; against k1 its Graph classifications are 13 wins, 8 losses, 4 ties and 5 uncertain.

**追问 / Follow-up:** 画出cpg=2的group/channel索引，推导halo大小，解释k1同时改变几何专用化和空间映射为何不能分离两者因果。 / Draw the cpg=2 index mapping, derive halo size, and explain why k1 does not isolate geometry specialization from spatial mapping.

Evidence: [CUDA source](../csrc/cuda/groupconv.cu), [new atlas and denominator](../results/rtx4090/session-20260908-vast-03/analysis-atlas/summary.zh-CN.md), [support matrix](../README.md#architecture-and-support).

## 2. 怎样证明结果正确、速度可信？ / How are correctness and timing checked?

**中文。** 新Atlas每个受支持输出与同策略FP32框架全量比较，并逐点检查64个独立FP64位置；这不等于全输出FP64检查。TF32关闭，源码、库与配置哈希绑定，三类sanitizer收据独立保留。新80点有430 PASS、50不支持和129000原始记录，每条记录包含多个时间字段，不能重复计数。10个配对批中位数用95% bootstrap区间和5%阈值分类；它证明本轮批间区间，不证明跨会话稳定或设备独占。Graph、同步API、host-feed与NCU时间分别分析。旧4090D主表只有5批，六点复核也不能替整个旧表确认。

**English.** Every supported new-atlas output receives a full FP32 framework comparison and 64 independent FP64 point checks, not a full-output FP64 check. TF32 is disabled and source, library and configuration identities are bound; sanitizer receipts are retained separately. The new atlas has 430 passing rows, 50 unsupported rows and 129,000 sample records. Multiple timing fields in a record are not independent samples. Ten paired batch medians support 95% bootstrap intervals and a 5% practical threshold within this run; they do not prove cross-session stability or exclusive GPU access. Graph, synchronized API, host-feed and NCU timings remain separate. The older 4090 D five-batch atlas and six-point confirmation keep their own scope.

**追问 / Follow-up:** 为什么框架调优单独进程？何时同一批次可以配对？FP64点检可能漏掉什么错误？ / Why isolate tuning in another process, when are batches paired, and which errors can point checks miss?

Evidence: [checker](../bench/check.py), [FP64 points](../bench/fp64_points.py), [runner](../bench/cuda_run.py), [new audit](../results/rtx4090/session-20260908-vast-03/analysis-atlas/analysis.json), [old analysis](../results/rtx4090d/analysis-20260908/summary.zh-CN.md).

## 3. 算子变快，模块就会变快吗？ / Does a faster operator make the module faster?

**中文。** 不一定。新MobileNetV2在完整features[3] block中只替换一个固定卷积；N=4 Graph比值1.13849，N=1为1.04879，按5%阈值分别为赢和平。同步API在N=1/4为0.66465/0.67902，均退化，包含分配和必要输入物化。随机权重下输出检查不证明任务精度或整网性能。独立插桩Amdahl估计有event/hooks扰动，不与主计时混合；eager trace的目标GPU占比仍为UNKNOWN。旧ShuffleNetV2 block也未取得eager收益。模型层保留bias与原几何：72单位中36通过、28不支持、8按计划未测。

**English.** Not necessarily. The new MobileNetV2 test replaces one fixed convolution in the complete features[3] block. Graph ratios are 1.13849 at N=4 and 1.04879 at N=1, a win and a tie under the 5% threshold. Synchronized API ratios of 0.66465/0.67902 are losses, including allocation and necessary input materialization. Random-weight output checks establish neither task accuracy nor whole-network speed. Amdahl estimates use separate instrumented events/hooks and are not ordinary timing samples; target GPU attribution in the eager trace remains UNKNOWN. The older ShuffleNetV2 block also showed no eager gain. Original bias and geometry remain in the model-layer denominator: 36 passing, 28 unsupported and 8 planned unmeasured units out of 72.

**追问 / Follow-up:** Amdahl的p、s应来自哪个边界？保留bias但实现不支持时如何处理？ / Which boundary must supply Amdahl's p and s, and how should unsupported bias be handled?

Evidence: [module runner](../bench/module_run.py), [module/layer statistics and intervals](../results/rtx4090/session-20260908-vast-03/analysis-supplemental/summary.zh-CN.md), [historical module evidence](../results/rtx4090d/analysis-20260908/analysis.json).

## 4. 如何解释失败，并决定回退范围？ / How do you explain failures and choose fallback boundaries?

**中文。** 新卡真实NCU覆盖8点30个正式组合，实际SM89反汇编确认k2有共享访问与两处BAR，k3有36个静态FFMA位置、64个寄存器且没有共享访问或BAR。静态指令位置不是动态次数；单次replay/cache-all采集没有统计胜负。最小空间点中k3的profile时间慢于k1，说明复用收益可能被并行度或固定开销抵消，但尚未隔离唯一原因。持续roofline上界未确立。框架应保留在不支持几何和已观察失败区间；目前没有已选择并验证的CUDA自动dispatcher。Triton仅支持40个holdout中的15点，5批独立suite比较全为UNCERTAIN，不据此扩大替换范围。

**English.** Real NCU captures cover eight shapes and 30 formal combinations. Actual SM89 disassembly confirms shared accesses and two BAR positions in k2, and 36 static FFMA positions, 64 registers and no shared accesses or BAR in k3. Static instruction positions are not dynamic counts; single replay/cache-all profiles do not support statistical wins. On the smallest spatial point, k3 profiles slower than k1: reduced parallelism or fixed costs may offset reuse, but no unique cause has been isolated. A sustainable roofline is unestablished. Keep the framework for unsupported geometry and observed losing regions; no CUDA automatic dispatcher has been selected and validated. Triton supports only 15 of 40 holdout points, with all five-batch separate-suite comparisons UNCERTAIN.

**追问 / Follow-up:** 设计一个能推翻“shared bank conflict导致变慢”的对照；为何高occupancy不能直接排名速度？ / Design a control that could falsify a shared-bank-conflict explanation, and explain why higher occupancy need not mean lower latency.

Evidence: [mechanism and SASS review](../results/rtx4090/session-20260908-vast-03/analysis-profile/mechanism-review.zh-CN.md), [microbenchmark limits](../results/rtx4090/session-20260908-vast-03/analysis-microbench/summary.zh-CN.md), [Triton final scope](../results/rtx4090/session-20260908-vast-03/analysis-triton/summary.zh-CN.md).
