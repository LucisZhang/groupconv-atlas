# RTX 4090 真实 profile 离线审计

状态：audit_status=PASS；roofline_status=NOT_ASSESSED。

31 个报告包含一次 probe 和 30 个正式组合；8 个预定结构点、两项 k2 UNSUPPORTED、67 条查询/采集/导出命令闭合。452 个下载文件的大小与 SHA-256 独立重算全部一致，归档 SHA 也一致。031 冻结源码、设备库、正确性收据、目标数值检查及显式宽表解析适配器绑定通过。

设备为 NVIDIA GeForce RTX 4090，driver 575.51.03，torch 2.8.0+cu128、CUDA 12.8、cuDNN 91002；严格 FP32/TF32 关闭。GPU UUID 的原始 torch 表示与 nvidia-smi 带 GPU- 前缀表示都保存在 analysis.json。

八个点的 k0/k3 profile 时间比为 1.128、2.279、4.623、5.102、3.134、5.949、3.830、5.988。除最小空间点外，k0/k3 DRAM 字节比处于 0.9906 至 1.0086；因此在这些单次采集里，时间缩短没有伴随相近幅度的 DRAM 流量下降。这是描述性观察，不能据此单独确定瓶颈。最小空间点 k0/k3 DRAM 字节比为 0.9037。

19/30 个正式 capture 的 DRAM write 指标为零；原值保留，不能改成逻辑输出字节。registers/occupancy 的三个 launch 指标未被冻结查询集合选中，正式结果保留 unavailable 状态；宽表自动附带列作为额外原始字段存证，不冒充正式选中指标。

该轮采用 kernel replay、cache-control all、clock-control none；每组合一次正式 profile，无置信区间或统计 WIN/LOSS。不能把这些时间混入普通 graph/eager 基准，或用于解释旧 4090 D 延迟。微基准与所选 SM89 SASS 已完成后续人工审阅和计数器对照，见相邻 analysis-microbench 与 static-audit.json；可持续带宽上界仍未闭合，故不发布 roofline 或算力利用率结论。二进制报告的 SHA、原始 CSV 与成功导出命令关联通过；初次离线审计没有重新运行 ncu 导出；后续静态审计已重新导出全部31份profile报告及2份微基准报告，33份均与原CSV逐字节一致。导出和哈希关系不等于独立执行真实性证明。

本次实际验收修复了两个离线分析器兼容问题：原生正确性去重键漏掉 input_pattern，误将 zeros/impulse/group_isolation/cancellation 当重复；GPU UUID 比较未处理 torch 与 nvidia-smi 的 GPU- 前缀差异。修复保留完整覆盖/哈希/数值门禁，并新增真实形状多模式接受、真重复拒绝和严格 UUID 格式/不同身份拒绝测试。相关 116 项测试及 6 个子测试通过。未修改测量源码或重新执行 GPU。
