[English](README.md) | [简体中文](README.zh-CN.md)

# GroupConv Atlas

这是一张分组卷积性能地图：研究空间映射、共享存储 tile 和寄存器复用何时有效，以及何时应使用调优框架。项目包含 C++ 串行、OpenMP、OpenCL、CUDA 与有限范围的 Triton 实现。

源码与实验工具由 Codex 辅助开发；[个人贡献说明](docs/CONTRIBUTIONS.md)区分可运行证据和本人独立讲解能力。最快入口：[证据索引](docs/evidence-index.md) · [四个中英面试问题](docs/interview-map.md) · [最新80点分析](results/rtx4090/session-20260908-vast-03/analysis-atlas/summary.zh-CN.md)。

课程形状集固定为36个点，Atlas扩展集固定为80个点；当前证据见[证据索引](docs/evidence-index.md)。

Apple OpenCL课程包已作为本地课程材料另行交付。

新增 RTX 4090 D 实机结果：CUDA 编译、数值、图捕获回归和三种 sanitizer 已通过，
80 点 Atlas、6 点独立复测及 ShuffleNetV2 的 N=1/N=4 完整 block 替换已测量。
环境、支持范围和全部结果见 [历史分析](results/rtx4090d/analysis-20260908/summary.zh-CN.md)。

后续 RTX 4090 已完成真实硬件 profiling、布局/缓存/往返、直接 cuDNN、模型层、
MobileNetV2 模块、Triton check/tune/holdout 和 copy/FMA 微基准。
新增全80点10×30确认保存129000个样本，完整轮内区间与支持分母已审计；见
[本轮分析](results/rtx4090/session-20260908-vast-03/analysis-atlas/summary.zh-CN.md)。

## 架构与支持范围

冻结形状与精度策略 → 原生／Triton 正确性检查 → 按批保存原始样本 → 配对统计与性能地图。C ABI 负责统一验证和分派，CPU、OpenCL、CUDA 分别实现；Python 负责框架参考、实验编排与统计。

原生 CUDA 均为 FP32、连续 NCHW、无 bias 的前向运算。B 几何指 Cin=Cout、3×3、stride/dilation=1、padding=1，合法分组和空间尾部仍须检查。

| 实现 | 契约 | 新卡覆盖 |
| --- | --- | --- |
| CUDA k0 | 通用合法卷积几何 | Atlas 80/80 |
| CUDA k1、k3 | B 几何 | 各80/80 |
| CUDA k2 | B 几何，每组通道1/2/4 | 30/80，其余50不支持 |
| Triton | B 几何，每组通道1/2/4；输入／权重元素数<2³¹ | holdout 15/40，其余25不支持 |
| CPU/OpenCL课程路径 | 独立Apple M4实验 | 36个课程点，不代表NVIDIA结果 |

[k1](csrc/cuda/groupconv.cu)改变空间映射并专用化几何；k2加载共享halo；k3让每线程四个水平输出复用输入和权重。这是实现之间的受控对照，不能将同时改变的机制当作单因素因果证明。[真实计数器与SM89机器码](results/rtx4090/session-20260908-vast-03/analysis-profile/mechanism-review.zh-CN.md)同时解释收益与失效。

## 运行入口

最快核验只需CMake 3.20+、C++17和真实OpenMP运行时：

```sh
# macOS + Homebrew LLVM：export CXX="$(brew --prefix llvm)/bin/clang++"
cmake -S . -B build -DGC_ENABLE_OPENCL=OFF -DGC_ENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
```

macOS可安装Homebrew LLVM/libomp；`./run.sh cpu`会在可用时选择该工具链。Python检查另需NumPy、PyTorch和pytest。`requirements-lock.txt`记录实测macOS环境；Linux CUDA使用`requirements-cuda.txt`及匹配的开发镜像。

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
./run.sh cpu
./run.sh check --backend cpu --output results/correctness.json
# macOS GPU：./run.sh opencl，再check --backend all
# Linux RTX 4090：./run.sh cuda -DCMAKE_CUDA_ARCHITECTURES=89
# 随后：./run.sh check --backend cuda
```

显式请求但不可用的后端会失败；`all`在CUDA缺失时记录NOT_RUN，并仍要求CPU和OpenCL可用。更多GPU流程见[验证协议](docs/execution/cuda-validation.md)。当前维护源码与实测源码分开：公共包的`evidence/measured-source-031887c/`保存精确数值源码子集，`evidence/measured-source-identity.json`保存原始哈希和导出范围。

导出包可在无GPU环境核验：

```sh
python scripts/audit_public_evidence.py --root .
```

导出器生成`docs/PUBLIC_EVIDENCE.md`说明脱敏、原始／公开哈希和证据边界。CPU CI运行原生正确性与sanitizer构建；GPU证据来自已归档的真实设备实验。

FP32 课程契约是 NCHW、无 bias、输入输出通道相等、3×3、步长 1、填充 1、膨胀率 1。
串行版以完整输出归约为单位，OpenMP 沿独立输出块分工，空间分块版复用权重。
OpenCL 分别提供朴素、16×8 共享 tile 和每线程四个相邻输出的寄存器复用版本。

结果保存逐形状原始样本、源码与配置哈希，并区分 GPU 事件时间、同步 API 时间和
显式数据复制在内的主机往返时间。不同计时边界分别比较。默认 5 批描述统计不作
已确认获胜结论，重要发现需要至少 10 批配对复核。

源码由 Codex 辅助开发。报告与答辩应依据实际代码、测试和测量，并由汇报者独立讲解。
旧 RTX 4090 D Atlas 上 k3 相对 k0 的描述性几何平均速度比为 Graph 设备时间 3.919、Eager API
时间 3.040；相对调优 PyTorch 为 0.789 和 0.645，整体未超过框架。
实际 ShuffleNet block 的 eager 调用也未获益。上述数字逐形状采用批中位数的
中位数之比；完整分母、逐点速度图及复测见
[分析摘要](results/rtx4090d/analysis-20260908/summary.zh-CN.md)。

旧 RTX 4090 D 主机的计数器权限受限；新 RTX 4090 已通过真实计数器门禁，完成
八点30组合及实际机器码审查。新全图的 k3/k0 Graph 比值为3.8718，调优框架/k3为
Graph 0.7384、API 0.6448，整体仍未超过框架。设备与采样协议分别记录，不外推跨轮稳定。
代码和硬件指标支持有边界的机制解释，持续峰值和硬件roofline尚未确立。
Triton已完成本轮GPU实验；昇腾NPU和网站／求职库发布回填继续按各自实际阶段推进。

## 最新结果与证据边界

![新RTX4090：调优PyTorch与k3完整矩阵](results/rtx4090/session-20260908-vast-03/analysis-atlas/atlas-tuned-vs-k3.png)

新卡10×30的调优框架/k3比较：Graph为34 WIN、35 LOSS、6 TIE、5 UNCERTAIN；同步API为27/50/2/1。分类使用10个配对批、95% bootstrap区间及5%实际差异阈值，只表示本轮内部变化。旧4090D的5×30主表、六点独立复核与Apple课程数据独立保留。

MobileNetV2完整block在N=4的Graph比值为1.13849；同步API在N=1/4分别为0.66465/0.67902，均更慢。随机权重输出检查不证明任务精度或整网提速。直接cuDNN完成32/32布局／调用路径单位。Triton已在新GPU完成check/tune/holdout，但只有5批且套件分进程运行，比较均保留描述性UNCERTAIN。Ascend尚无实现或NPU测量。[证据索引](docs/evidence-index.md)保留完整分母与入口。

外部资料和依赖许可见[REFERENCES](docs/REFERENCES.md)。本项目代码采用[MIT](LICENSE)，项目自有数据和图表采用[CC BY 4.0](DATA_LICENSE)；[NOTICE](NOTICE)保留归属和第三方边界。本人已自述参与选题、方案设计、代码修改、实验配置、结果核对及独立讲解；该OWNER_ATTESTED声明尚未经过独立面试验证。
