# Contributions / 个人贡献与说明边界

This is a course and performance-engineering project developed with Codex assistance. Its deliverable is the combination of implementations, explicit contracts, controlled experiments and inspectable evidence. Generic tiling, shared-memory reuse and register blocking are established techniques, not claimed algorithmic inventions.

本项目由 Codex 辅助实现、编写实验工具和整理文档。可主张的项目贡献是将 k0–k3 对照、统一正确性契约、原始样本与失败分母、Graph/API计时边界、实际模块替换和性能地图组织为可核验实验。仓库规模、通过测试或一次成功运行，均不能直接作为本人独立精通 CUDA／Triton 的证明。

| Deliverable / 工作 | Inspectable evidence / 核验入口 | Personal explanation required / 本人自述能够讲解，尚无独立考核 |
| --- | --- | --- |
| Output mapping and k0–k3 implementations | [CUDA source](../csrc/cuda/groupconv.cu) | 分组索引、空间尾部、halo体量、barrier一致到达、四输出复用 |
| Numerical and execution contract | [Native tests](../tests/native_cuda.cu), [checker](../bench/check.py) | FP32/FP64参照覆盖、非法输入、stream与graph条件 |
| Fair timing and honest comparison | [runner](../bench/cuda_run.py), [statistics](../bench/stats.py) | 配对批、置信区间、调优缓存、Graph与API为何不同 |
| Hardware and failure analysis | [mechanism review](../results/rtx4090/session-20260908-vast-03/analysis-profile/mechanism-review.zh-CN.md) | 观测与因果的区别、实际SASS、负结果和资源交换 |
| Triton and module extension | [Triton](../backends/triton/kernel.py), [module runner](../bench/module_run.py) | tuning/holdout隔离、有限支持、局部收益与模块收益 |

The user sets the project goals, course constraints and execution scope. Codex assistance includes implementation and evidence preparation; no file-by-file claim of sole human authorship is made. The owner has attested to all six areas: topic selection, design, code modification, experiment configuration, result checking and independent explanation. This is OWNER_ATTESTED testimony, not an independently administered interview or mastery assessment. Use [the four bilingual questions](interview-map.md) as a code-and-evidence rehearsal, and record future assessed mastery separately from this executable project evidence.

PyTorch, cuDNN, Triton and upstream model definitions remain their authors' work. The current record reports no transplantation of CUTLASS, TVM, RepLKNet or FCM implementations; it is a dependency/implementation record, not a legal originality certification. See [references and licenses](REFERENCES.md).

本人确认上述六项参与范围；保留Codex辅助开发事实。代码按[MIT](../LICENSE)提供，项目自有数据和图表按[CC BY 4.0](../DATA_LICENSE)提供，第三方内容见[NOTICE](../NOTICE)。
