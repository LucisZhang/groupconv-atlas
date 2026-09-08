# Triton 实际结果离线审计

最终状态以 validated.json 为准：analysis_status=PASS，measurement_status=PASS。首轮 analysis.json 的 NOT_ASSESSED 保留，原因是 torch 与 nvidia-smi 的 GPU- UUID 前缀表示不同；离线分析器严格规范化后通过，原始 GPU 字段保持不变。加入不同/非法 UUID、重复设备行和非数值驱动拒绝测试，54 个 Triton 针对性测试通过。

1817 个下载文件的大小和 SHA-256 独立重算全部一致，完整归档 SHA 为 424f0ab490e1985150a8d56bb0e0b36960e87428362dc467e5942efe2cd4e6f0。冻结031源码、库、原生正确性和三阶段环境绑定通过；实际 GPU 为同一 RTX4090、CUDA12.8、torch2.8.0+cu128、cuDNN91002、Triton3.4.0。

正确性阶段25个边界形状：18支持、7不支持；三个block候选形成54 PASS和21 UNSUPPORTED，每个支持候选验证三个固定seed，共162份独立导出的PTX。调优集40点中15支持、25不支持；45支持候选、75不支持候选，5×30样本。只在调优集选择后，cpg1/2/4均冻结block256；记录的调优搜索耗时72.60秒。

Holdout另40个不重叠点：15支持、25不支持。55个job闭合，包括40个Triton job和15个同新卡CUDA六实现job；形成15个Triton PASS、90个CUDA/框架PASS和75条不支持候选。汇总units折叠不支持候选后为105 PASS、25 UNSUPPORTED。CUDA原生与框架对照均来自本轮，没有拿旧4090D性能替代。完整FP32及独立FP64收据、冻结选择、PTX哈希和样本门禁通过。

速度比为 baseline/Triton，逐点使用批中位数的中位数后跨15个支持点取几何平均：对k3，Graph为1.077、同步API为1.122；对调优PyTorch，Graph为1.446、同步API为0.921、host-feed为0.550。调优框架Graph比值15/15大于1，但API仅6/15大于1；时间边界改变会改变方向。完整逐点和分母见validated.json，聚合见descriptive-aggregates.csv。

所有支持点仍为UNCERTAIN：CUDA与Triton分开进程跑suite，不能当配对批次，只有5批，不给配对置信区间或统计WIN。25个不支持点始终保留，不能外推整个Atlas或模型收益。

编译材料包含222份显式导出PTX（check162、tune45、evaluate15）；另有114个JIT缓存cubin及对应缓存PTX。全部336个PTX文本含sm_89目标与fma.rn.f32，属于文本存在性检查；离线性能审计仅按协议校验显式PTX/metadata绑定，不因此宣称最终cubin/SASS语义或调度配置已人工审阅。JIT缓存cubin被下载manifest保护，但尚未逐个建立到测量launch的独立执行绑定。metadata_semantic_review保持NOT_ASSESSED。first-call JIT+launch保留，不能解释为纯编译耗时；不证明未记录的holdout访问从未发生。

### 追加静态证据核验

完整包中 114 份 Triton cache cubin 均有哈希对应的 SASS 与资源导出，属于静态检查，不产生新的 GPU 性能样本。对支持 holdout 的 15 份显式保存 PTX，逐一在 triton-evaluate/jit-cache 找到且仅找到一个字节 SHA 相同的 direct.ptx；该组相邻 direct.json 与 direct.cubin 均存在，cubin 哈希对应静态检查清单。元数据为 CUDA arch89、kernel direct、enable_fp_fusion=false。逐项匹配与资源文本保存于 static-review.json。

因此可以将显式 PTX 关联到唯一保存缓存组，再关联其已反汇编 cubin。该关系仍依赖缓存目录组织，未独立认证编译链或运行时真正发射的 cubin。仅凭同目录、哈希及 SASS 不能补成 launched-binary 证明，也不能证明全部加速因果。元数据中的默认 dot 精度 tf32 不能证明实际存在 dot 指令；enable_fp_fusion=false 本身也不是完整数值语义验证。
