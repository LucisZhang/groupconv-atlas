# 新轮80点 Atlas 审计

**audit_status=PASS；measurement_status=COMPLETE。**

完整80形状×6实现为480行，其中430 PASS、50 UNSUPPORTED（全部来自k2范围外），0 FAIL/OOM/TIMEOUT/NOT_RUN。每个PASS有10批×30个样本，合计129,000原始样本；每条同时保存Graph设备、同步API与host-feed字段，不能再乘三声称独立样本数。逐点全量FP32与64个独立FP64点检通过。

benchmark记录1070.45秒，外层执行1073.57秒，约17.9分钟。下载包2462个文件集合、大小及SHA独立核验，实际libgroupconv.so SHA与run-key/correctness一致。同卡RTX4090 UUID DEVICE_8fdfe034723b及031树哈希与本轮profile/Triton相同；逐形状文件仍没有独立设备身份，不据此声称独立执行真实性或设备独占。

速度比均为baseline/candidate；大于1表示candidate更快。以下分类来自本轮10个配对批中位数、10,000次bootstrap、95%区间和5%实用阈值。

| Baseline → Candidate | 边界 | 可比/80 | 几何均值 | WIN/LOSS/TIE/UNCERTAIN | 最差比值 |
|---|---|---:|---:|---|---:|
| k0 → k1 | Graph | 80/80 | 2.4275 | 79/1/0/0 | 0.8082 |
| k0 → k1 | 同步API | 80/80 | 2.2562 | 72/1/5/2 | 0.9383 |
| k0 → k2 | Graph | 30/80 | 1.8610 | 29/0/1/0 | 0.9746 |
| k0 → k2 | 同步API | 30/80 | 1.7094 | 22/0/5/3 | 1.0181 |
| k0 → k3 | Graph | 80/80 | 3.8718 | 80/0/0/0 | 1.1612 |
| k0 → k3 | 同步API | 80/80 | 3.4490 | 74/0/5/1 | 1.0213 |
| k1 → k2 | Graph | 30/80 | 1.0286 | 13/8/4/5 | 0.8317 |
| k1 → k2 | 同步API | 30/80 | 1.0309 | 11/6/11/2 | 0.8310 |
| torch_cuda_tuned → k3 | Graph | 80/80 | 0.7384 | 34/35/6/5 | 0.1145 |
| torch_cuda_tuned → k3 | 同步API | 80/80 | 0.6448 | 27/50/2/1 | 0.1153 |

k3相对k0：Graph全80点WIN，几何平均3.8718；同步API为74 WIN、5 TIE、1 UNCERTAIN，几何平均3.4490。最弱点为N1/C256/H=W7/G256（gc_78e99ae160f2dd60），Graph比值1.1612、API比值1.0213。

相对调优框架，k3整体没有超过：Graph几何平均0.7384，34 WIN、35 LOSS、6 TIE、5 UNCERTAIN；同步API几何平均0.6448，27 WIN、50 LOSS、2 TIE、1 UNCERTAIN。两边界最差均为N1/C256/H=W112/G1（gc_70fccacc501d7b1c），Graph比值0.1145、API比值0.1153。不能只展示获胜点。

k1对k0的Graph最差点为N32/C256/H=W7/G256（gc_25ff201831f41fcb），比值0.8082，LOSS；k2只支持30/80点，其余50点保持NOT_COMPARABLE。

这些是同轮内部分类，不是跨轮稳定性证明。稳态重复输入、图重放设备边界与同步API分别报告；不能混入NCU cache-all采集时间，不能与旧4090D做同环境统计检验。当前分析器只对Graph/API作比较，host-feed原始样本保留。库算法身份、首次初始化/搜索、框架workspace管理和GPU独占性仍保持原协议边界。
