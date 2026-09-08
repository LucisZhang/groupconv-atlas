# Offline Triton analysis

`scripts/analyze_triton.py` audits the frozen `031887ca96243dc6a8aa27fc4b9089123fae9efd` production schema. It reads existing artifacts and performs CPU reference calculations. It does not import snapshot code or execute CUDA/Triton. The new RTX 4090 production check/tune/holdout artifacts passed this audit on 2026-09-08; see [the execution receipt](../evidence-index.md) and `analysis-triton/validated.json` in the new session. The tests below remain synthetic rejection/contract checks.

## Inputs and invocation

Supply the downloaded extension-results directory containing `triton-check/`, `triton-tune/` and `triton-evaluate/`, its original absolute remote directory, the full extracted measured source snapshot, the exact native shared library and original native correctness JSON. Keep the remote hierarchy inside the downloaded directory, including PTX files. The mapper translates recorded paths for reading only; receipt bytes and hashes remain unchanged. Paths outside the declared results root and escaping symlinks are rejected.

```sh
.venv/bin/python scripts/analyze_triton.py \
  --results /local/downloaded/extensions \
  --remote-root /original/remote/extensions \
  --snapshot /local/extracted-031887c \
  --library /local/downloaded/libgroupconv.so \
  --correctness /local/downloaded/correctness.json \
  --output /local/analysis/triton-analysis.json
```

Output must be new. Exit 0 means evidence audit PASS, not independent execution attestation. Exit 2 writes `NOT_ASSESSED`, the rejection reason and an explicitly unaudited complete expected-job inventory preserving observed FAIL/OOM/UNSUPPORTED states and missing jobs. Invalid evidence never generates performance comparisons.

## Evidence gates

The full 71-file source-identity-v2 manifest must equal frozen tree `5b499cd77b1594e501fce8c8d457ec147ffe410ae5764fa6b1df1a4566934cbd`; every file is checked against the snapshot. Loaded local bench and Triton contract validators are bound to those file hashes. The native library, native correctness, protocol and split bytes must match all three phase bindings. The existing full native correctness validator is reused with complete coverage and ABI checks. This cannot be satisfied by a shortened one-row PASS receipt or by a public source subset.

All phases share exact recorded source, protocol and environment. Required runtime: torch 2.8.0+cu128, CUDA 12.8, cuDNN 91002 and Triton 3.4.0, plus recorded torch build, GPU name/UUID/capability/memory/visibility and matching driver UUID/version. The contract allows the newly authorized GPU; it does not substitute old RTX 4090 D evidence.

The existing source-bound Triton validator checks every boundary shape and all blocks 128/256/512 against FP32 full output and independent full-output FP64 for seeds 0, 7 and 20260908, plus per-seed PTX artifact hashes and recorded compile-option fields. Opaque metadata is required to be nonempty; its generated-code semantics are not independently reviewed. Tuning includes 40 shapes: 15 supported shapes × 3 candidates, and 25 unsupported shapes × 3 explicit unsupported rows. The frozen evidence set must close exactly over tuning jobs, check receipt and tuning PTX. Selection and geometric-mean scores are recomputed from all candidate samples, per cpg 1/2/4. The selected blocks, tuning-only IDs and `holdout_performance_accessed=false` must match.

Holdout contains another 40 disjoint shapes, 15 supported Triton jobs plus 15 six-implementation CUDA-suite jobs and 25 unsupported Triton jobs: 55 jobs total. It checks the seeded suite order, one frozen block per supported shape, full output FP32, independently regenerated 64 FP64 points, precise sample identity/order/repeats, strict precision and the 5×30 sample grid. All 25 unsupported shapes remain in the result and comparison denominator; their 75 explicit unsupported candidate rows are checked. FP32 metrics must be finite and internally consistent. CUDA baselines retain their actual `k0`/`k1`/`k2`/`k3`/`torch_cuda_default`/`torch_cuda_tuned` names and benchmark/cache-isolation evidence.

## Interpretation and limits

Each metric uses median(batch medians). Graph device time, synchronized eager API time and host feed time remain separate. Ratios use baseline median divided by Triton median. All supported comparisons are `UNCERTAIN`, with `paired=false` and `ci95=null`: the production runner measures separate suites in separate processes, so shared batch numbers are not paired timing evidence. No statistical WIN, cross-session CI or whole-model acceleration is inferred.

First-call JIT/launch metadata and native/framework first-call initialization/search fields are retained separately from steady-state samples; frozen search elapsed time is separate too. JIT first call may include within-run cache reuse and is not pure compiler cost. The actual timing schema has no per-sample graph-status flag: timing semantics and capture/eager full-output checks derive from the hash-bound `bench.cuda_run.timing` implementation. The CUDA child-job schema has no independent GPU/runtime attestation; it inherits the common receipt binding. PTX/native binary hashes establish artifact identity, not independent recompilation. Receipts cannot prove that no unrecorded holdout access occurred. Analysis itself records its source and supplemental helper hashes and Python version.

## Local validation

`PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/test_triton_analysis.py tests/test_triton_protocol.py -q -p no:cacheprovider`: **52 passed**. CLI help also passes. New tests cover median-of-batch-medians, missing/duplicate/foreign/nonfinite/bool samples, wrong order/repeats, runtime/driver drift, unsafe artifact paths and hash changes, incomplete source and candidate denominators, missing three-seed compile evidence, and preserved OOM/missing-job inventory. Existing Triton protocol tests exercise frozen selection/check/evidence rejection. These checks are not a production end-to-end receipt acceptance or GPU result; that remains pending real downloaded artifacts.

The audit explicitly returns `metadata_semantic_review=NOT_ASSESSED`: PTX hash identity, nonempty opaque metadata and recorded launch options do not prove generated configuration semantics. Every Triton sample must carry a strict integer block matching its enclosing candidate row; cross-candidate mixtures are rejected even when their order matches another valid candidate. Relative `--results` paths are normalized before artifact hashing. Focused regression checks cover both cases.
