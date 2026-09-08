# 冻结 031 profile 离线审计

状态：2026-09-08 新 RTX 4090 的 31 份真实 profile 报告已通过此入口，见 [证据索引](../evidence-index.md) 与新 session 的 `analysis-profile/`。实际采集入口为 `bench/profile_cuda.py`。本分析器不加入或补传冻结测量树，不启动 ncu 或 GPU。

```sh
python scripts/analyze_profile.py \
  --session DOWNLOADED/profile \
  --remote-root /EXACT/REMOTE/profile \
  --snapshot EXTRACTED_FROZEN_031 \
  --library DOWNLOADED/libgroupconv.so \
  --correctness DOWNLOADED/correctness.json \
  --output NEW_LOCAL_PROFILE_ANALYSIS
```

`--remote-root` 是收据中远程 profile 输出目录的精确绝对路径。分析器将该目录下的 reports/csv/targets/logs 按相对路径映射到下载目录，不改原 JSON，不以文件名搜索代替路径绑定。CLI 只读取显式输入，不执行 snapshot Python。输出 `analysis.json` 及有成功采集时的 `counters.csv`；每个引用下载文件有 SHA，另保留二进制、正确性、分析脚本和实际仓库验证依赖哈希。验证依赖须与 snapshot 对应文件一致，031 的 71 文件和树哈希固定。

正式审计要求 phase=collect、status=PASS、counter_gate=PASS：一次 counter-probe 加 30 个正式组合，另两项 UNSUPPORTED 保留。67 条命令必须闭合：五个查询、31 个采集、31 个导出；检查命令退出、起始时间、耗时、stdout/stderr SHA 和错误日志。逐项验证实际 `.ncu-rep` 非空及 SHA、CSV SHA、target SHA、目标 PID/launch ID/核名、源码/库/正确性/协议/完整环境、单次调用与十次预热、调用前后 full FP32 与 64 个合法不同位置 FP64 的数值判据。FP64 期望值不在此重算。

仅成功 probe 或失败/中断 session 返回 `audit_status=NOT_ASSESSED`，保留原 session 状态，不输出通过的矩阵结果。完全缺少必要文件或绑定不一致明确报错，不生产 PASS。未评估的 partial captures 留在原始结果，不被假造为零。

`.ncu-rep` 是不透明二进制，本离线工具核对其 SHA 及成功 export 命令与 CSV 的关联，**不能独立证明 CSV 真由该二进制导出**。需要更强核验时，由具备相应版本 ncu 的授权环境重新导出并核对；本工具不执行该动作。下载清单、执行日志以及 GPU 独占性/频温功耗审核仍是外部证据。run/target 的 UUID 一致和 PID 关联不构成独立执行证明。

计量仅使用同一 capture 的 `dram__bytes_read.sum + dram__bytes_write.sum` 与 `gpu__time_duration.sum`。DRAM 必须为 base byte(s)；duration 明确转换 second/msecond/usecond/nsecond，SM 周期必须 cycle(s)，未知单位拒绝计算。probe 没有 duration，时间和带宽标 NOT_ASSESSED。零 traffic 的算术强度为 null，不产生无穷值。可选指标保留原单位和值，L2 sector 不直接变字节。

输出 nominal FLOPs、profile 秒数、DRAM B/s、nominal FLOPs/s、nominal FLOPs/DRAM byte；名义 FLOPs 包含 padding，并非动态执行指令 FLOPs。k1/k2/k3 对 k0 和 k2 对 k1 的同 shape 对照仅为各一次 profile 的描述比值，不给 WIN/LOSS 或置信区间。cache-all/kernel-replay 条件与普通 graph/eager、微基准稳态条件不同，不合并其时间或强行解释旧卡延迟。

`roofline_status` 始终 `NOT_ASSESSED`：本实现不接收可手填的 peak，也不把传入状态字样当人工 SASS 通过。后续只有同 GPU/软件/运行条件的实测 micro 数据、对应测量 binary 的所选架构 SASS 人工审阅及适用带宽上界证据完整后，才能另行构建有边界说明的 roofline。当前结果可用于整理计数器和提出待验证的瓶颈假设。

本地测试：`python -m pytest -q tests/test_profile_analysis.py`。夹具覆盖时间单位、同 capture 字节/时间公式、错误单位、重复/额外 launch、核身份、非有限数、缺指标、零 traffic、probe 缺时间、下载路径逃逸与 FP32/FP64 证据反例。它们均不是设备测试。

本地夹具另覆盖真实 capture_command 生成的 launch/replay/target/export 关联及篡改；完整真实 session 的端到端离线审计已通过，详见证据索引。

正确性修订：当前 56 个合成 CPU 测试通过。full FP32 检查严格整数计数、epsilon=1e-12、有限非负 max_abs/max_rel/rms 及 RMS 不大于 max_abs。真实 FP64 `check_points` 收据包含 shape_id/output_dims/output_count/full_output、reference/selection/seed、点数及每点 absolute_error/relative_error/threshold，分析器逐项重算核对；完整卷积参数位于 target.shape，按 canonical Shape 及严格整数维度核验。冻结 seed 的选点集合和顺序由哈希绑定的 select_points 重建，FP64 期望输出仍不重算。duration 存在时必须在换算为秒后仍为正有限数；零和下溢值明确拒绝，probe 缺 duration 保持正常未评估状态。

原生正确性与环境补强：额外调用本地 `analyze_supplemental.native_receipt`，继续绑定冻结 bench 验证依赖，同时检查完整 16 项 ABI cases、状态期望、记录去重/汇总计数及所有 PASS 数值统计。新增 supplemental 分析依赖不属于冻结包，单列其当前 SHA，不伪装为 frozen 源码。

目标环境只接受 torch 2.8.0+cu128、CUDA 12.8、cuDNN 91002，以及真实 `gc_build_info()` 的 ABI=1 文本（允许其实际 OpenMP enabled/disabled 分支）。GPU name/UUID、正整数显存、两项整数 compute capability 和 visibility 字段均检查；从已绑定的 `nvidia-smi -q` stdout 读取唯一 GPU UUID/Product Name 及数值 Driver Version，与 target 名称及 UUID 对齐。target 没有独立 driver 字段，因此 driver 来源明确为 query 日志，不能声称做了不存在字段的双向比较。缺失或无法解析则拒绝审计，不以环境彼此相等代替完整性。当前针对性 CPU 合成测试 72 passed；原生补强负测单独 mock 冻结边界验证器，仅测试新增审计语义，仍不构成 GPU 验证。
