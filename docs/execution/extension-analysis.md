# 新轮 Atlas 离线分析

状态：入口与合成 CPU 夹具已实现；当前没有新轮 GPU 原始数据通过本入口的回执。分析脚本不进入已冻结的远程测量源码包，也不需要补传服务器。

```sh
python scripts/analyze_extensions.py \
  --atlas NEW_RUN/atlas \
  --correctness NEW_RUN/correctness.json \
  --snapshot EXTRACTED_FROZEN_SOURCE \
  --output NEW_LOCAL_ANALYSIS
```

`--snapshot` 必须指向实际测量源码包的本地解包目录。分析检查 run-key 中每个源码文件的 SHA 和 source-tree digest，不能用已继续修改的工作区替代。只接受一轮 run-key/provenance 中 UUID 一致的完整 80 个冻结形状及六实现记录；正式协议必须为冻结配置仅覆盖 batches=10，每个 PASS 有 10×30 个不重复样本。逐字段核对 canonical shape（含 dtype/layout/bias/shape_id/dims），严格核对 run-key 与每个有 PASS 的 shape 的 FP32 实际读回。用冻结 seed、shape ID 与 batch 重建实现顺序，repeats 必须为 1 到上限的整数且同一实现不变。检查逐形状记录、聚合 raw、bench 与 summary 的一致性，原始错误、OOM、UNSUPPORTED、NOT_RUN 等状态保留。统计审计通过与测量完成分成两个字段。

全输出 FP32 记录、64 个不同且合法位置的 FP64 数值判据、原生正确性完整覆盖及源码／库绑定均检查。这里不重新构造大输入或重算 FP64 期望值。收据自洽不构成独立执行证明；GPU 身份仅在 run-key/provenance 中保存，单 shape 文件本身没有独立 GPU／源码身份，因此无法凭 shard 独立排除混入。下载 manifest 和远程执行回执仍须单独审计。

主比值严格为双方各自批中位数的中位数之比。对同轮配对批索引做 10,000 次 bootstrap，95% 区间下界大于 1.05 为 WIN，上界小于 1/1.05 为 LOSS，整个区间位于这两个阈值之间为 TIE，其余 UNCERTAIN。轮内区间不证明跨轮稳定。跨形状几何平均仅纳入双方 PASS，同时输出全部 80 点状态分母及分类数；不把 ratio>1 自动记为 WIN。graph 设备与同步 eager API 分开。

产物为 `analysis.json` 和 `comparisons.csv`，含输入及本脚本 SHA。所有实际加载的 bench 验证依赖及 scripts 包初始化文件均须与测量 snapshot 的对应源码哈希匹配，否则拒绝；不执行 snapshot 中的 Python。分析回执另记录 Python 解释器哈希、Python/NumPy 版本和已加载标准库／NumPy 文件哈希。JSON 包含每形状批中位数、区间、分类及比较聚合；聚合包括双方 PASS 中的最差比值及其 shape ID、分类，仍保留 80 点全分母。比较为 k1/k2/k3 对 k0、k2 对 k1、k3 对 tuned。模型、布局边界、Frontend、Triton、模块及 microbench 的 schema 独立，当前明确 `analysis_status=NOT_IMPLEMENTED`、`measurement_status=NOT_ASSESSED`；不因未接分析器将已存在测量误标 NOT_RUN。

本地验证：`python -m pytest -q tests/test_extension_analysis.py`。测试包括确定比值、配对关系、宽区间、不足十批、批中位数与 pooled 口径差异、样本篡改、来源/GPU/协议不一致和完整分母但测量未执行。文件绑定夹具明确 mock 原生正确性验证器；它不构造虚假的设备验证通过证据。

当前针对性合成测试：42 passed，涵盖 canonical 元数据、精度读回、依赖漂移、固定顺序、repeats 非法及变化、最差形状；没有新 GPU 实测声明。
