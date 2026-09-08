# 代表性相关工作与采用决策

核验日期：2026-09-08。本文完成 PLAN 第6节中5组代表工作的定向核验，服务于已有实现和解释边界；不是系统综述、复现排行榜或新颖性证明。先执行 anysearch Node CLI 的两项官方源码查询，返回 `ENOTFOUND api.anysearch.com`；随后改用网页工具读取官方版本文档、作者仓库及论文。未使用二手博客、历史 issue 或搜索摘要作为性能结论。

本项目状态不再统称 PLANNED：RTX 4090 D 的旧 FP32 NCHW 前向 Atlas 已实测；direct cuDNN、布局、Triton及新 profiling 入口已有源码，但本次文献核验没有执行 GPU。外部 CUTLASS、TVM、RepLKNet、FCM 代码均未移植或运行。调用 PyTorch/cuDNN API 属于依赖使用，不能写成“代码复用为0且不依赖外部库”。

## 1. PyTorch 2.8.0、cuDNN 9.10.2 与 Frontend 1.18.0：参考算子不等于固定实现

**问题。** `F.conv2d` 在 depthwise 和一般 grouped 上是否走同一个 CUDA 实现？“FP32”能否排除 TF32？直接库基线能否限定搜索与 workspace？

**来源与发现。** [PyTorch 2.8 Conv2d](https://docs.pytorch.org/docs/2.8/generated/torch.nn.Conv2d.html)规定2D互相关、输入与输出通道均须被 groups 整除；depthwise 可含 multiplier，不只本项目的 m1。其[2.8 CUDA语义](https://docs.pytorch.org/docs/2.8/notes/cuda.html)将 matmul 和 cuDNN 的 TF32开关分开。因而本项目同时关闭两者，并记录读回值；只写 tensor dtype=float32 不够。

固定 [v2.8.0 Convolution.cpp](https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/aten/src/ATen/native/Convolution.cpp) 的 `use_cudnn_depthwise`：channels-last 在满足 `use_cudnn` 时有优先路径；普通 contiguous 的快速分支检查输入和权重为 Half、4维、无dilation等。另有大型索引例外。因此旧 FP16 workload 启发式不能直接解释本项目 FP32 NCHW 的所有点；这是源码分支判断，仍须实际 trace 确认安装构建的 dispatch。

[cuDNN 9.10.2 grouped说明](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.10.2/developer/misc.html#grouped-convolutions)要求张量描述完整算子尺寸，权重每组通道为 C/G；支持前向、输入梯度和权重梯度，并沿用各函数支持的格式。页面后面的3D best-practice表不是整个2D grouped API的支持矩阵，不能据其推荐集合缺少 cpg2 就判本项目 cpg2 unsupported。

[Frontend 1.18卷积接口](https://docs.nvidia.com/deeplearning/cudnn/v1.18.0/operations/Convolutions.html)及[graph流程](https://docs.nvidia.com/deeplearning/cudnn/v1.18.0/developer/overview.html)提供描述、验证、heuristics、支持检查、plan构建、numeric notes和workspace筛选。graph合法不等于存在满足指定精度、布局和显存预算的plan；Python异常和无候选应保留。

**硬件、比较与决定。** 这是 NVIDIA CUDA 库路径，不是某张卡上的速度保证。本地锁为 torch2.8.0+cu128、cuDNN包9.10.2.21（runtime91002）、Frontend1.18.0，见 [requirements-cuda.txt](../../requirements-cuda.txt)。旧框架基线是已测 API 调用；[新增 direct协议](../../configs/cudnn-frontend-protocol.json)固定FP32计算、NCHW/channels-last分开、256MiB workspace、候选及时间上限、排除降低精度的notes。它是待设备验证的有限搜索，不能称库的全局最优。继续使用已有正确性、layout转换和双计时边界，不增加新benchmark。

**版本、许可、复用。** PyTorch按v2.8.0文档和源码核验；其[LICENSE](https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/LICENSE)含BSD式条款及组件声明。Frontend[官方许可证](https://raw.githubusercontent.com/NVIDIA/cudnn-frontend/main/LICENSE.txt)为MIT，但此许可证页面是访问日main，未冒充v1.18文件快照；cuDNN backend另受NVIDIA许可约束。项目调用上述API并依赖安装包，没有复制或宣称拥有其内部CUDA kernel。

## 2. CUTLASS例46：SIMT并不自动意味着FP32同口径

**问题与一级来源。** [官方例46源码](https://raw.githubusercontent.com/NVIDIA/cutlass/main/examples/46_depthwise_simt_conv2dfprop/depthwise_simt_conv2dfprop.cu)完整可读；v4.3.0及v3.5.1固定tag页面本轮工具未取得正文，故本条版本明确为访问日main，提交SHA未冻结，不保留旧文档对4.3.0的含混绑定。

**发现。** 该具体实例为depthwise 2D fprop：输入、权重、输出、累加均 `half_t`；NHWC；`OpClassSimt`、Sm60；固定3×3、stride1、dilation1、groups=C=K，通道需满足8元素向量对齐检查。它没有证明任意 FP32 grouped cpg2/4 可直接编译运行。源码还明确提醒固定stride/dilation、大filter可能增加寄存器spill；示例性能路径及可选reference检查均在源码中，而不是我们跑出的结果。

**决定与验证映射。** 借鉴“减少地址计算与增加寄存器压力存在交换关系”这一问题设置，用本项目既有k1/k2/k3及8点硬件采集检验；不复制模板，也不把FP16/NHWC示例数值加入FP32/NCHW主表。本轮不新增CUTLASS依赖或移植。实际SM60编译能力、现代卡表现和完整shape支持未实测。

**许可与复用。** 此源码自身含 `SPDX-License-Identifier: BSD-3-Clause`；该文件级声明不外推到CUTLASS仓库中所有DSL或第三方组件。源码复用：无；概念参照：tile、地址开销、寄存器压力。固定commit缺口在真正移植前才需关闭，本轮没有作移植承诺。

## 3. TVM v0.14.0 TOPI：表达支持、目标调度与搜索成本分开

**问题与版本。** 编译器能表达grouped卷积，是否就等于为指定GPU得到高性能实现？核验 [v0.14.0 topi/nn/conv2d.py](https://raw.githubusercontent.com/apache/tvm/v0.14.0/python/tvm/topi/nn/conv2d.py)，尤其 `group_conv2d_nchw`、`group_conv2d_nhwc` 和 `conv`。

**发现。** 前向定义包含groups、stride、padding、dilation；NCHW配OIHW权重，NHWC配HWIO，输出dtype显式控制，并检查输入/输出通道整除groups。FP32输入、FP32输出可在这一表达层指定；它不是某个目标设备的二进制，也没有凭这一文件证明所有GPU schedule、尾部或特殊布局都可用。当前版本提供算子表达，与2018 TVM论文中某组硬件速度并非同一实测版本；本条不转录历史加速倍数或声称本机已跑TVM。

**决定与验证映射。** 本项目保留原始shape与layout语义、将编译/搜索成本从稳态时间剥离；现有Triton有限候选只在tuning集合选择，holdout之前冻结。这个实验纪律并非TVM算法移植，也不是新的自动调度算法。本轮只沿用已写协议，不启动TVM工程。

**许可与复用。** 文件头声明Apache-2.0，另见[项目LICENSE](https://raw.githubusercontent.com/apache/tvm/main/LICENSE)。核验的计算定义是v0.14.0，许可证总览是访问日main；没有复制TOPI、TVM调度器或成本模型。GPU型号/性能：本项目未测，不能由IR支持推定。

## 4. RepLKNet／MegEngine大核depthwise：适用域与模块语义不同

**问题与来源。** [RepLKNet作者README](https://github.com/DingXiaoH/RepLKNet-pytorch)指向MegEngine的外部CUTLASS扩展；[作者replknet.py](https://raw.githubusercontent.com/DingXiaoH/RepLKNet-pytorch/main/replknet.py) 的 `get_conv2d` 只在设置环境变量、核大于5、Cin=Cout=groups、stride1、对称same padding、dilation1时选择该实现。3×3课程核心不触发这条替换路径。[MegEngine官方实现](https://github.com/MegEngine/RepLKNet)注明优化大核depthwise至少需要MegEngine1.8.2；这不是本项目已安装版本。

[扩展Python包装器](https://raw.githubusercontent.com/MegEngine/cutlass/master/examples/19_large_depthwise_conv2d_torch_extension/depthwise_conv2d_implicit_gemm.py)公开FP32/FP16的forward、backward-data、backward-filter入口。自测输入N64/C384/H=W32、31×31、groups384，沿用PyTorch NCHW形状，并有独立bias加法；不能外推到一般grouped或任意strides。包装器自测比较输出和权重梯度，不能当作我们的实测或完整kernel支持审计。实际CUDA架构、每个大核尺寸与连续性限制未从完整C++扩展逐项核实。

**决定与比较边界。** RepLKNet训练、BN重参数化和大核模块收益不是单个3×3前向kernel收益。其bias、BN合并前后差异也说明“替换卷积”必须保留模块语义。本项目继续已有真实模型形状及完整block检查；大核、反向、RepLKNet重参数化不进入本轮实现范围。

**版本、许可、复用。** 作者main及MegEngine/cutlass master均为访问日可变分支，未锁定提交；[作者仓库MIT许可证](https://raw.githubusercontent.com/DingXiaoH/RepLKNet-pytorch/main/LICENSE)已读，不能据此给外部MegEngine/CUTLASS扩展整体重新授权，其依赖许可审计未完成。未复制、安装或运行这两套实现。

## 5. Qararyah等，2024：融合收益不能冒充单算子收益

**论文与代码。** [Fusing Depthwise and Pointwise Convolutions for Efficient Inference on GPUs，arXiv:2404.19331v1](https://arxiv.org/html/2404.19331v1)，作者[实现仓库](https://github.com/fqararyah/Fusing_DW_and_PW_on_GPUs)对应ICPP 2024 workshop论文。它研究前向推理的DW/PW及融合模块，不是本项目一般grouped反向实现。

**已核实验边界。** 论文列GTX1660、RTX A4000、Jetson AGX Orin及CUDA11.6；FP32与INT8，来自MobileNet等网络的不同融合点，不能把两精度的点逐一视为同一shape。以CUDA events计时、Nsight Compute分析流量；比较自己的逐层实现、三个cuDNN算法及以cuDNN为backend的TVM整网路径。本文不把它与cuDNN9.10.2/Frontend1.18的有限搜索混为同一基线。

论文有单模块与整网两种边界，且有融合减速点。收益依赖流量减少、冗余计算和片上资源；这些是作者实验的发现，不是我们尚未取得计数器时的瓶颈证明。论文正文未在本轮明确核实所有layout、完整N/C/H/W列表及cuDNN精确版本；其表1部分规格不能替代设备探测。

**决定与复用。** 采用其“保持逐层对照、区分融合与单算子”的比较原则，支持本项目已有完整block测量及Amdahl边界；不因此新增fusion kernel或整网性能承诺。作者仓库有FP32/INT8 kernels及FusePlanner目录，但访问日根目录未发现明确许可证，提交也未冻结；公开可见不等于取得再分发许可。代码复用与复现实验：均无。

## 本项目能够主张的内容

官方库的优化、CUTLASS模板、TVM搜索和作者融合方法属于各自工作。本项目的贡献范围是自行实现的k0–k3对照、统一shape/正确性契约、保留失败和unsupported分母、分离图内设备与eager API计时，以及可追溯的实验组合；通用的合并访问、shared halo和寄存器复用本身不构成算法首创。没有逐行源码相似性审计，因此“未引入外部实现”是依赖/移植记录，不是法律上的原创性鉴定。

旧Atlas的80形状、5×30实测见 [分析摘要](../../results/rtx4090d/analysis-20260908/summary.zh-CN.md)及[原始样本](../../results/rtx4090d/session-20260908-v2/atlas/samples.jsonl)。k0/k3几何平均为图内3.919、eager3.040；调优PyTorch/k3为0.789、0.645。后两项小于1，不能写成整体战胜调优库；5批只作描述。新增源码、CPU协议通过和外部文献核验均不替代新的GPU回执。

本轮完成的是上述5项的决策核验；保留的未知项已就地列出，不再用“全部PLANNED”覆盖已有实测。没有扩大GPU实现范围，也没有将外部速度数字设为本项目验收目标。
