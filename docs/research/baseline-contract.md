# 框架基线契约

修订：2026-09-08。课程 CPU、PyTorch CPU、Apple M4 OpenCL 及 RTX 4090 D CUDA 已完成实测。
CUDA 默认／调优框架分别运行；一个代表形状和两个实际模块输入有独立 dispatch trace。
直接 cuDNN 计划搜索和硬件计数器采集尚未完成。

实际环境：Python 3.14.4、torch 2.14.0、NumPy 2.5.3、macOS 26.6.2、Homebrew Clang 22.1.0、Apple M4 OpenCL 1.2。torch构建记录 `USE_MKLDNN=OFF`、`BLAS_INFO=accelerate`、`USE_OPENMP=ON`，未采实际算子dispatch，因此性能表只称 PyTorch CPU，后台身份为 UNKNOWN。完整身份见 `docs/execution/environment-probe.json` 和各run的provenance。

课程主表：36形状，576成功实现／线程记录，86400原始样本；另有3个N=4形状。五个代表形状按10批GPU-only顺序复核，其中两个o2点相对混合CPU/GPU顺序主表出现方向反转，已标UNSTABLE。不同轮次不合并，原因仍未确认。见 `results/apple-m4/confirmation-20260908/reproducibility-audit.json`。

## 环境身份

本地安装版本与可用性以仓库环境探测产物为准。云端实际为 RTX 4090 D、SM 8.9、
驱动 580.76.05、CUDA 12.8、Python 3.12.3、torch 2.8.0+cu128，cuDNN 版本接口
返回 91002；GPU UUID、torch commit、实际依赖清单在 CUDA session 中。
未使用直接 cuDNN Frontend。阅读的 PyTorch 2.14 和 cuDNN Frontend 1.18.0 文档
是研究来源，不冒充该云端安装版本。

运行前保存Python/torch/torchvision版本、torch git_version、torch.version.cuda、cudnn.version、设备型号、OS、编译器、线程配额及配置哈希。PyTorch CPU后端只有构建信息和实际dispatch证实后才命名；Apple CPU不能自动标作oneDNN。

## 统一语义与精度

前向二维互相关；零填充、无bias，FP32输入/权重/输出及FP32累加；输入逻辑NCHW、权重 `[Cout,Cin/G,R,S]`，两侧通道整除groups。B范围为Cin=Cout、3×3、stride1/pad1/dilation1。其他情况按实现返回 `UNSUPPORTED`，不能计作通过。

FP32存储不证明内部FP32运算。云端按实际torch版本只选一套兼容TF32控制接口，关闭cuDNN与matmul TF32及其他降精度策略，读回设置。不得混用新 `fp32_precision` 和旧 `allow_tf32` 控制。确定性策略必须两边一致并落盘；默认主表固定允许非确定算法但禁止降精度，若新增确定性实验单独标记。参考：[CUDA semantics](https://docs.pytorch.org/docs/2.14/notes/cuda.html)。

## 要比较的路径

| 路径 | 配置/记录 | 当前状态 |
|---|---|---|
| CPU串行/OpenMP | 同算法1线程及1/2/4/8线程；另列相对清晰c0提升 | CPU实测完成 |
| PyTorch eager默认 | CPU F.conv2d；CUDA benchmark=false，与native同在形状worker、与调优进程隔离 | CPU和CUDA实测完成；CUDA代表形状有实际trace |
| PyTorch eager调优 | CUDA同精度策略、benchmark=true、limit=10；独立进程隔离缓存，首调用初始化／搜索在稳态之外 | CUDA实测完成；不称纯搜索耗时 |
| 直接cuDNN frontend | graph conv_fprop；固定workspace上限；筛选numeric notes，保留plan tag/engine/搜索集合 | NOT_RUN |
| 自写kernel | 每项真实路径、支持率、回退率；通过正确性后才入性能表 | OpenCL和CUDA实测完成；不支持点保留记录 |

具体workspace字节上限必须在GPU测试配置中冻结；未冻结前不得声称公平的直接cuDNN胜负。Frontend参考：[1.18.0 workflow](https://docs.nvidia.com/deeplearning/cudnn/v1.18.0/developer/overview.html)。现代plan搜索得到的最快值也仅限实际搜索集合。

## 布局、计时与正确性

NCHW和channels-last分别记录输入、权重、输出strides。布局转换/权重打包单列并加入完整调用口径。设备算子时间用同stream事件，同步后读取；API边界时间包含实际调用，主机往返和模块时间分别报告。预热、搜索、初始化、H2D/D2H不得悄悄混入或排除不同路径。框架分配输出与自有预分配的差异要在口径中解释。

先通过逐元素 `abs(actual-ref) <= 1e-4 + 1e-4*abs(ref)`，同时记录max_abs、max_rel（明确epsilon）、RMS、失败元素与NaN/Inf；小形状使用独立FP64 oracle。计时配置以项目测量配置文件为执行来源，保存原始样本；profile独立运行，其cache/replay条件不能当普通延迟。

任何 `FAIL/UNSUPPORTED/OOM/UNSTABLE/NOT_RUN` 保留原因及分母。框架回退结果不能归到专用kernel名下；源码推断和内部_select_conv_backend只作辅助，真实dispatch需trace/日志支持。

CUDA 主表将 graph 内部 CUDA event 的 `t_device_op_graph_us` 与另一轮 eager 同步
API 时间分开。短调用采用重复输入的图内事件，避免 Python 供给间隙进入设备时间；
该图条件不能外推 eager 时延。算子表的 native 输出预分配、框架输出分配差异已落盘，
模块对照则两侧都包含输出分配和必要输入整理。

80 点 Atlas 主轮是 5×30 描述性统计；6 个结构代表点有独立 10×30 确认轮，
其中 3 个 k3 相对调优框架的 eager 速度比跨轮方向反转。轮内区间不证明跨轮稳定。
N=1/N=4 的 ShuffleNetV2 随机权重模块测试保持原权重、BN、激活与 eval 状态，
只替换 `stage2.1` 中实际筛选出的 depthwise 层；完整 block 计时未显示 eager 收益。
不报告模型精度或完整网络加速。详见 `docs/execution/cuda-session-20260908.md`。
