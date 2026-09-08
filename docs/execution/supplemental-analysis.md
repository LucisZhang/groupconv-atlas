# Supplemental offline analysis

This analyzer consumes existing receipts; it runs no GPU work, executes no code from the supplied snapshot, and does not load old RTX 4090 D measurements as substitute results. Its output is an audited description, not proof that a remote process ran beyond the supplied receipts. Missing execution, numerical or source context produces `NOT_ASSESSED`, not a fabricated performance `FAIL`.

## Frozen plan and scope

`supplemental-analysis-plan-031887c.json` derives the complete denominator from the frozen `031887ca96243dc6a8aa27fc4b9089123fae9efd` archive and `scripts/run_extensions.py` default commands. It contains no PASS assumption. The plan is written before this supplemental analysis; it does not claim a historical timestamped preregistration service or execution receipt. Override commands require a separately reviewed plan, never a plan reconstructed from successful shards.

| Entry | Actual runner schema | Predefined units | Batches |
| --- | --- | ---: | ---: |
| boundaries | provenance.json; shapes/*.json rows, samples, conversions | 8 shapes × 6 implementations × 3 residency boundaries × 2 cache conditions = 288 | 5 |
| models | run-key.json; model-shapes.json; 12 shape shards | 12 original layers × 6 implementations = 72 | 5 |
| module-n1 / module-n4 | summary.json; samples.jsonl; random-state.pt | 2 complete MobileNetV2 features[3] block implementations per input batch size | 10 |
| frontend | run-key.json; bench.json records with embedded samples | 8 shapes × 2 layouts × resident/caller_nchw = 32 | 5 |

The models denominator includes four original biased operations: 16 native UNSUPPORTED rows and 8 planned unmeasured framework rows (24 rows across these four layers). Eight unbiased operations supply the remaining 48 planned rows. No bias is removed to make a shape eligible. Layer timings do not establish full-model speedup or accuracy.

The model entry also pins the exact verified CPU export SHA and its protocol SHA; a changed report cannot be accepted by changing its adjacent run-key hash.

## Statistical boundaries

Raw samples must contain exactly the planned unique `(batch, sample)` grid for a PASS unit. Summaries are recomputed from raw values: first each within-batch median, then the median of batch medians. Reported cached medians are not treated as ground truth. The ratio is baseline median / candidate median. Non-PASS units and missing shards remain in the complete denominator; partial PASS or duplicate/invalid samples are rejected.

Only MobileNet block comparisons receive paired bootstrap confidence intervals, and only with at least ten batches and a verified seeded randomized implementation order in every raw sample. Resampling uses the same batch index on both sides; classification uses a predefined 5% practical margin. WIN requires the entire 95% interval above 1.05; LOSS requires it below 1/1.05; TIE requires it entirely within that band. These are within-run exploratory intervals, not independent-session confirmation, and no multiple-comparison correction is claimed.

Boundaries execute implementations in fixed outer order. Frontend layouts execute in separate processes. Model tuned-framework samples are subprocess-fed. This version conservatively gives these comparisons descriptive ratios and `UNCERTAIN` only, even if extra batches arrive; equal numeric batch labels alone do not establish paired experiments. Five batches never produce statistical WIN.

Graph-internal event time, synchronized eager API time, host-feed time and pinned host-to-host time remain separate fields. Only like timing boundaries are compared. Eviction is an attempted same-stream buffer fill before each timed interval; the runner explicitly reports that clearing all caches is NOT_VERIFIED. Conversion samples have no batch dimension, so they remain raw descriptive evidence without CI. Layout initialization/search, first variant call, Frontend tuning/setup/workspace/selected-plan evidence, and the module first call with hooks remain outside steady-state samples. The module first call includes hooks/clones/recording/synchronization and is not labeled pure search cost. Profile/Amdahl receipts are retained separately without recomputing or promoting their conclusions.

## Evidence gates and current limitations

- Require an explicit plan, exact source-identity-v2 manifest hash and every frozen snapshot file hash. The analyzer's own source SHA, Python version, loaded input artifact SHAs and plan SHA are recorded separately from the measured source.
- Native suites reuse `bench.cuda_run.validate_correctness(smoke=False)` for full boundary-shape/implementation/seed/reference coverage. Every loaded local bench validator dependency is hash-bound to the measured source manifest before invocation. Additional gates enforce all 16 mandatory ABI cases, expected/actual return codes, record-count consistency, unique record identities and finite metrics for every PASS correctness row. The supplied correctness file must also match measured source/library and show CUDA availability with no failed or limited run. This is a receipt binding gate; the analyzer does not rerun numerical kernels or execute a snapshot's correctness checker.
- PASS rows require strict TF32-off settings, full-output FP32 counters/tolerances, finite nonnegative max_abs/max_rel/rms and consistent epsilon/RMS bounds, and 64 unique independently recorded FP64 points. FP64 receipts must match the canonical current shape ID/output count, every coordinate must be four in-range integers, and numerical values plus error/threshold fields must be self-consistent. Module leaf dimensions come from selected_shape and must match the actual hook observation; full-model output count is the fixed MobileNet 1000-class output times the planned batch size. The module also requires leaf/block/model checks, graph-versus-eager checks, nondefault native stream, saved random-state hash and runner hash.
- Frontend requires runtime/library hashes, matching before/after/key version and GPU identity; runtime-file hash agreement wherever a library is present (additional lazily loaded libraries are retained), stream readback, input hashes, graph/layout checks, numerical-filter receipt, reference/candidate/path oracles, complete candidate index list, workspace bound and a validated selected tuning winner. Unsupported dependency/plan results stay UNSUPPORTED. The Python API cannot expose per-plan numerical-note readback; this analyzer preserves the runner's vendor-filter limitation.
- Canonical oracle shapes also require FP32/NCHW/no-bias metadata and exact integer input/weight/output dimensions. Frontend PASS rows bind the locally loaded Shape validator to the frozen `bench/api.py` hash even though Frontend does not consume the native correctness report.
- This version does not independently regenerate tensors and FP64 references, authenticate vendor execution, audit PTX/SASS, prove all cache levels cold, infer pure search costs, or claim whole-model performance/accuracy. Unsupported output versions and absent mandatory context are NOT_ASSESSED. In particular, unrecorded graph checks cannot be invented by the analyzer; it relies on the hash-bound runner's capture-check path plus raw timings.
- Scope is the four schemas above. Atlas/Triton/profiler hardware-counter analysis remains with its dedicated analyzer. Different devices, sessions, layouts and timing boundaries are never pooled.

## Invocation

Run from the local project checkout, after the result files arrive. `--snapshot` is the extracted frozen source tree, not the evolving working tree. Use a fresh output file.

```sh
.venv/bin/python scripts/analyze_supplemental.py --kind boundaries --entry boundaries \
  --run /path/to/extensions/boundaries \
  --plan docs/execution/supplemental-analysis-plan-031887c.json \
  --snapshot /path/to/extracted-031887c --correctness /path/to/correctness.json \
  --output /path/to/new-analysis/boundaries.json
```

Use kind `models` / entry `models`, kind `module` / entry `module-n1` or `module-n4`, or kind `frontend` / entry `frontend`. Frontend has its own candidate oracle and does not require a native correctness receipt. Missing context returns exit code 2 with NOT_ASSESSED; runtime execution statuses are retained separately from assessment.

## Local validation

` .venv/bin/python -m pytest -q tests/test_supplemental_analysis.py` passes 23 meaningful synthetic tests and 6 subtests, including a complete randomized ten-batch module fixture and corrupted order, an all-UNSUPPORTED Frontend fixture with missing-layout denominator preservation, duplicate/partial/nonfinite raw samples, strict precision, canonical metadata, numeric FP64 tampering, validator/source-snapshot drift, and five-batch/unpaired-ten-batch inference suppression. CLI help passes. No supplemental GPU data had been analyzed when this implementation was prepared.

Root verification additionally used the actual 71-file frozen archive extraction to bind all five plan entries. The historical RTX 4090 D report's 16 contract rows and 1256 PASS metric records were accepted by the relevant schema-only gates; this checks compatibility with real saved receipt structure, not acceptance as a new-device run.


The small source-manifest and module fixtures are deliberately synthetic unit tests; the module timing fixture mocks only the separate native receipt gate. They do not demonstrate a production 031887c full-tree audit or complete numerical run. The one-row deletion test separately invokes the real complete validator with hash-bound local dependencies and verifies coverage rejection. Production source acceptance still requires the reviewed frozen plan plus every file in the actual measured manifest and extracted snapshot. These checks validate reported evidence and cannot reconstruct full tensors from aggregate FP32 summaries. No new GPU result has been accepted by the test suite.

The strengthened analyzer uses the existing local NumPy-dependent validator and Shape model. Run through the project virtual environment. The default system Python on this machine lacks NumPy and is not the validated execution environment.
