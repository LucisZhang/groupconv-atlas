# Direct cuDNN Frontend baseline preparation

Prepared 2026-09-08. CPU protocol/CLI checks do not establish GPU compatibility or performance. The historical environment is Python 3.12.3, torch 2.8.0+cu128, cuDNN 91002, RTX 4090 D; the new run records its own environment and source identity. The entry point requires torch 2.8.0+cu128, CUDA 12.8 and cuDNN 91002 in both libraries, together with Frontend 1.18.0; mismatches are UNSUPPORTED. Existing results remain separate.

## Audited API and dependency

The entry point uses `cudnn.pygraph` from **nvidia-cudnn-frontend 1.18.0**, explicitly selects a built plan, and calls `execute_plan_at_index`. PyTorch supplies CUDA storage, a strict-FP32 correctness reference and CUDA Graph/event facilities. PyTorch does not implement the measured convolution. Missing Frontend, another Frontend version, no CUDA, or differing Frontend/PyTorch cuDNN versions yields `UNSUPPORTED`; no replacement backend is timed.

Official sources inspected at the v1.18.0 tag:

- [Convolution Python API](https://docs.nvidia.com/deeplearning/cudnn/v1.18.0/operations/Convolutions.html): `conv_fprop` image/weight/padding/stride/dilation arguments.
- [Python wrapper](https://github.com/NVIDIA/cudnn-frontend/blob/v1.18.0/python/cudnn/__init__.py): package version, pointer-map execution, backend loading.
- [Graph bindings](https://github.com/NVIDIA/cudnn-frontend/blob/v1.18.0/python/pygraph/pygraph.cpp): graph construction, per-index build/execute, plan tags, workspace and behavior notes.
- [Plan implementation](https://github.com/NVIDIA/cudnn-frontend/blob/v1.18.0/include/cudnn_frontend/plans.h): excluded numerical notes bar matching candidates; building each index checks support and workspace restrictions. Plan names are vendor engine/knob tags, retained verbatim. No separate unverified engine-id parser is used.
- [Convolution node](https://github.com/NVIDIA/cudnn-frontend/blob/v1.18.0/include/cudnn_frontend/node/conv_fprop.h): explicit logical dimensions/strides and FLOAT compute; grouped convolution uses input/filter channel dimensions. Full-output checks cover the actual group semantics.
- [Python installation guide](https://docs.nvidia.com/deeplearning/cudnn/installation/latest/python-frontend-install.html): package name and import convention. Installation instructions are not execution evidence.

For a prepared GPU environment, install the exact Frontend package while preserving the existing torch/cuDNN runtime (inspect the installer dependency plan before execution):

```sh
python -m pip install 'nvidia-cudnn-frontend==1.18.0'
python -m bench.cudnn_frontend_run --output results/new-session/frontend-smoke --shape-set course --shape-ids gc_48ead864bb00a89a --batches 1 --samples 2 --smoke
python -m bench.cudnn_frontend_run --output results/new-session/frontend-course --shape-set course
python -m bench.cudnn_frontend_run --output results/new-session/frontend-batch4 --shape-set course-batch4
python -m bench.cudnn_frontend_run --output results/new-session/frontend-atlas --shape-set atlas
```

No dependency installation or GPU run was performed during preparation. First-run runtime errors must be resolved or reported before interpreting the baseline. Source changes require a new output directory; resume rejects changed source/protocol/environment identity.

## Numerical and search contract

The runner has its own per-candidate oracle gate and does not accept a native-correctness receipt as a substitute. Once per shape/layout input case, it prepares 64 deterministic independent scalar FP64 output points (all outputs when fewer than 64). The torch FP32 reference must pass those points first. Each candidate then passes a full-output torch FP32 comparison followed by the same prepared FP64 points before warmup/timing; both selected layout paths repeat that gate. Point coverage is sampled FP64 evidence, not a full FP64 output claim. Receipts contain positions, expected/actual values and pass/fail details, the seed and canonical FP32 input/weight byte hashes. Source identity version 2 includes new/untracked source and backend files; mapped cuDNN libraries and Frontend Python/extension files are hashed separately.

All I/O, intermediates and convolution compute use FLOAT. The shared `precision()` helper sets and reads back IEEE FP32 / TF32-disabled controls for the torch reference. Direct cuDNN additionally excludes TENSOR_CORE, DOWN_CONVERT_INPUTS and REDUCED_PRECISION_REDUCTION numerical notes before building plans. Missing enum symbols, rejected filter calls or incompatible APIs fail closed; no execution follows an unapplied filter. Actual heuristic/numerical enum strings and completed filter application are recorded. The Python API does not expose per-plan numerical-note readback; receipts state that limitation and record the applied exclusions. This conservative subset can have no supported candidate for some layouts/shapes; that is an explicit result, not evidence that all cuDNN algorithms are unavailable. Static v1.18.0 bindings and `plans.h` confirm A/FALLBACK and per-index candidate-slot filtering/building; actual cuDNN 9.10.2 runtime support remains subject to the first GPU smoke.

The fixed search uses A plus FALLBACK heuristics, at most 32 candidate slots, a 256 MiB workspace cap, and a 60-second soft tuning budget checked between candidates. Each candidate includes support/build, full-output comparison to torch FP32, ten warmups and five tuning samples outside final sampling. Setup and tuning costs are recorded separately. A single candidate can exceed the soft budget; a process per shape/layout and a 600-second hard timeout contain CUDA stalls. Total wall budget is 10,800 seconds. Unvisited, rejected, failed and timed-out candidates/shape-layouts remain visible. Unexpected execution/capture failure stops that shape/layout so a poisoned context cannot silently pick a later plan. Selection means fastest successfully validated and measured candidate within the stated budget, not exhaustive cuDNN optimality.

## Timing and layout interpretation

Both logical NCHW and channels-last physical storage use explicit strides, including singleton dimensions. Input and weight values are checked after layout materialization. Output, workspace and all conversion buffers are preallocated. This direct API allocation policy differs from the existing torch Conv baseline's output allocation and is recorded explicitly.

`resident` times only execution with already prepared buffers. `caller_nchw` separately includes NCHW-to-channels-last input and weight copies on **every invocation**, direct execution, and output restoration; for NCHW all transforms are identities. This is a preallocated conversion-inclusive boundary, not a deployment throughput or persistent-weight-cache claim. Initial layout materialization has a separate synchronized host timer. Full-output checks run on every built/timed candidate and both final paths; captured output is checked against eager output before sampling.

`t_device_op_graph_us` uses the existing timing helper's external CUDA event nodes surrounding each logical operation inside a CUDA Graph on one non-default stream. Capture and initial graph validation are outside samples. `t_api_us` is a separate eager loop through stream synchronization; `t_host_feed_us` stops before that final synchronization and includes Python/Frontend wrapper feed overhead. These are distinct boundaries. Host/device transfers are `NOT_RUN`; setup uploads and correctness downloads are outside timing. Samples use five batches of thirty paired, randomized-path observations with bounded repeat counts. Reports are descriptive batch summaries, with no confidence interval or cross-session claim.

Each output directory contains the run key, per-shape/layout incremental receipts, `bench.json`, `bench.csv`, `raw.jsonl` and `summary.json`. Receipts retain raw tuning samples, final samples, graph JSON/hash, selected engine/knob tag, behavior notes, candidate count, workspace, versions, stream readback, source hashes and mapped cuDNN library hashes. Graph JSON describes graph construction; it is not a serialized execution-plan proof. The actual per-index plan tag and workspace provide plan evidence.
