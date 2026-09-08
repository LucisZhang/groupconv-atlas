# References and licenses / 引用与许可

The project uses the following dependency and research records. Versioned findings and access-date limitations are documented in [related-work.md](research/related-work.md), reviewed on 2026-09-08. That document is a historical literature review; current GPU completion is recorded in [the evidence index](evidence-index.md).

| Source | Version / scope used | Use and attribution |
| --- | --- | --- |
| [PyTorch Conv2d](https://docs.pytorch.org/docs/2.8/generated/torch.nn.Conv2d.html), [CUDA semantics](https://docs.pytorch.org/docs/2.8/notes/cuda.html) | PyTorch 2.8.0; measured package 2.8.0+cu128 | Reference outputs, tuned/default framework calls; [BSD-style license and component notices](https://github.com/pytorch/pytorch/blob/v2.8.0/LICENSE) |
| [cuDNN grouped convolution](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.10.2/developer/misc.html#grouped-convolutions) | cuDNN 9.10.2, runtime 91002 | Backend dependency; NVIDIA's cuDNN terms apply separately from Frontend |
| [cuDNN Frontend](https://docs.nvidia.com/deeplearning/cudnn/v1.18.0/operations/Convolutions.html) | Frontend 1.18.0 | Direct plan/workspace experiment; [MIT license](https://github.com/NVIDIA/cudnn-frontend/blob/main/LICENSE.txt), linked main is not a frozen release license snapshot |
| [Triton](https://github.com/triton-lang/triton/tree/v3.4.0) | Measured Triton 3.4.0 | JIT/compiler dependency for project kernel; [MIT license](https://github.com/triton-lang/triton/blob/v3.4.0/LICENSE) |
| [CUTLASS example 46](https://github.com/NVIDIA/cutlass/blob/main/examples/46_depthwise_simt_conv2dfprop/depthwise_simt_conv2dfprop.cu) | Access-date main, no frozen commit | Conceptual reference; example BSD-3-Clause notice; no code transplant or measured comparison |
| [TVM TOPI](https://github.com/apache/tvm/blob/v0.14.0/python/tvm/topi/nn/conv2d.py) | v0.14.0 | Expression/schedule boundary and search-cost reference; Apache-2.0; not run |
| [RepLKNet](https://github.com/DingXiaoH/RepLKNet-pytorch), [MegEngine implementation](https://github.com/MegEngine/RepLKNet) | Access-date branches, commits not frozen | Large-kernel/module comparison; no transplant; author repository MIT does not relicense external dependencies |
| [Qararyah et al., Fusing Depthwise and Pointwise Convolutions for Efficient Inference on GPUs](https://arxiv.org/html/2404.19331v1) | arXiv v1, 2024 | Fusion versus single-operator scope; [author implementation](https://github.com/fqararyah/Fusing_DW_and_PW_on_GPUs) not copied or run; license not established in the recorded review |

Synthetic inputs and randomly initialized model weights are used for numerical and timing experiments. They do not establish dataset accuracy. Course templates and teaching materials are external inputs to document generation; possession is not a redistribution license. The standalone generated source package and public candidate must retain their own file manifests and exclude unapproved source materials.

**Repository licenses:** the owner selected [MIT](../LICENSE) for project code and [CC BY 4.0](../DATA_LICENSE) for project-owned data and figures. [NOTICE](../NOTICE) records attribution and third-party boundaries. These licenses do not relicense upstream binaries, course templates or external code. Publication is a separate action from licensing and local export preparation.

代码与项目自有数据／图表的许可已由用户选择；第三方材料继续适用各自条款。有关本人与Codex的贡献边界，见[CONTRIBUTIONS](CONTRIBUTIONS.md)。
