# Triton 优先子集：执行与证据边界

2026-09-08。源码已实现；GPU 编译、设备正确性与性能在实机回执产生前均为 `NOT_RUN`。这是 PLAN D 的 Triton 子集，不包含 Ascend/昇腾迁移。

## 内核与支持范围

`backends/triton/kernel.py` 是真正的 Triton JIT direct grouped cross-correlation：一条 lane 计算一个输出，按 ci/r/s 顺序 FP32 `tl.fma`，带空间与向量尾部 mask。NCHW、Cin=Cout、3×3、stride=pad=dilation=1、cpg∈{1,2,4}，包括 depthwise multiplier 1 和一般 grouped。支持非方形和 N=3/C=6 的 cpg1/2；C=6/G=2 的 cpg3 明确 unsupported。所有元素索引限定在有符号32位范围内。包装器检查 tensor 设备、FP32、形状、连续布局和缓冲区重叠。

候选仅 block128/256/512，固定4 warps、1 stage。禁用隐式 FP fusion，显式使用正确舍入 FP32 FMA，不使用 dot、TF32、Tensor Core 或近似数学指令。源代码能证明表达的算法；实际 PTX 与运行结果须由 GPU 阶段保存后确认。

## 集成命令

要求 Linux NVIDIA CUDA、PyTorch 的 CUDA 版以及 `triton==3.4.0`。程序精确检查 Triton 版本，不自动安装或改环境。首先按当前完整源码快照重跑 native CUDA 正确性，得到与 library/source hash 匹配的完整报告；旧 session 报告不能给新源码冒名背书。以下占位路径由根流程替换，输出目录必须不存在：

```sh
python -m bench.triton_run --help
python -m bench.triton_run check --library build/libgroupconv.so \
  --correctness NEW_NATIVE_CORRECTNESS.json --output NEW_RUN/triton-check
python -m bench.triton_run tune --library build/libgroupconv.so \
  --correctness NEW_NATIVE_CORRECTNESS.json \
  --check-report NEW_RUN/triton-check/receipt.json --output NEW_RUN/triton-tune
python -m bench.triton_run evaluate --library build/libgroupconv.so \
  --correctness NEW_NATIVE_CORRECTNESS.json \
  --frozen NEW_RUN/triton-tune/frozen.json --output NEW_RUN/triton-holdout
```

每阶段有总预算和单 job 超时，子任务独立进程组，正常、异常和 SIGTERM 退出清理当前组（包含库 tuned 子进程）。云端总控还应设硬超时和停机门槛；本程序退出不代表云端停止计费。中途失败保留已完成 JSON/PTX 和失败回执，不生成可用 frozen；不自动覆盖或恢复旧输出。

## 正确性与冻结

check 使用现有 boundary_shapes，并补 N3/C6/H5/W9/G3、G6。所有合法点执行全部3个 block和3个seed，对完整输出做 CUDA PyTorch FP32 比较，以及独立 Python FP64 全输出对照。unsupported 是支持契约分类，不冒充设备执行通过。包装器拒绝逻辑有 CPU 协议测试；GPU 数值结论以回执为准。

tune 只读取 `configs/split.json` 的 tuning ID：全部40形状保留分母，其中15个合法，每个试3候选。每候选5批×30个原始样本，每批候选顺序固定seed随机化。每个 cpg 选择在该组所有合法 tuning 形状上的图内时间几何平均最小 block；每形状先取批内中位数，再取批中位数的中位数。精确平手取小 block。缺失、重复、FAIL、OOM 或不完整样本均禁止冻结。未读取旧 Atlas holdout 性能。

只有完整 tune 成功才生成 frozen，绑定源码树、协议、split、native library、native correctness、GPU UUID/架构、torch/CUDA/Triton版本、tuning原始文件及PTX hash。evaluate 校验这些绑定和证据文件后，仅用冻结 block；holdout40形状中15个合法，25个 unsupported 全部保留，不重新选择 block。对每个合法 holdout输出再做全量 CUDA FP32和64点独立 FP64检查。holdout的新形状仍会 JIT 专门化，这属于编译，不是搜索。

冻结校验同时绑定实际 cuDNN 版本、torch git版本及 `nvidia-smi` 返回的 GPU UUID/driver版本；身份读取失败不继续执行。生成和重新读取冻结选择时，每个调优候选都必须具备完整种子、全输出 FP32 统计、64点 FP64、编译选项/首调用时间/metadata和非空PTX文件，逐文件核验SHA。FP64点位和期望值根据固定输入及seed重新独立计算，不能只凭passed标记。边界检查使用相同验证器并要求全部输出。冻结证据集合必须精确等于全部调优job、check回执及合法候选PTX集合，不允许空列表、漏项、额外项或重复路径。调优job自身的shape、suite、binding、environment、状态以及全分母也逐项核验；tuning IDs和排除holdout标记必须与split一致，再从raw samples重算选择和scores。文件哈希和结构校验用于发现错配或缺失，不是外部签名的执行证明。

## 计时和对照

复用 `bench.cuda_run.precision/timing/validate_correctness`。设备时间是图内外部 event；API 是分开的同步 eager墙钟。capture、warm replay、JIT和搜索均在稳态时间外。Triton复用输出，与native一致；torch输出分配保持现有口径。每阶段使用全新的 TRITON_CACHE_DIR，记录每次首调用 JIT+launch 墙钟，不能叫纯编译时间；tune总耗时包括正确性、进程和计时，不叫纯搜索内核时间。每次生成的PTX、hash和编译 metadata独立保存。

holdout每个合法形状另运行现有 CUDA suite：k0/k1/k2/k3、torch default/tuned，继续使用独立 tuned缓存进程和相同5×30协议。Triton suite与CUDA suite按seed随机先后执行，**不是三者逐批交错配对实验**；支持形状可以做相同口径描述性ratio，不把批号相同解读为时间配对或给出跨轮稳定性置信结论。尚未绘图/汇总的原始JSON保留全部raw samples、状态、正确性、PTX及绑定。失败即停止后续job，未运行部分不算PASS。

## 官方API来源

访问2026-09-08，均为官方主来源：

- [Triton JIT API](https://triton-lang.org/main/python-api/generated/triton.jit.html)：JIT编译与constexpr。
- [官方向量加法示例](https://github.com/triton-lang/triton/blob/v3.4.0/python/tutorials/01-vector-add.py)：grid/lane mask/load/store和调用方式。
- [tl.fma](https://triton-lang.org/main/python-api/generated/triton.language.fma.html)：逐元素 fused multiply-add。
- [v3.4.0 compiler源码](https://github.com/triton-lang/triton/blob/v3.4.0/python/triton/compiler/compiler.py)：CompiledKernel asm及metadata产物。此类编译产物接口按实际版本检查，缺PTX则失败。

在线main API可能后续变化；本执行协议锁定3.4.0。CPU协议测试和 `--help` 能核验输入/冻结流程，不能证明GPU编译或性能。
