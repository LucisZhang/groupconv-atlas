# Evidence index / 证据索引

Evidence date: 2026-09-08. Ratios mean baseline time / candidate time; values above one favor the candidate. Every comparison retains its device, precision, supported denominator and timing boundary. The links below address retained local evidence; availability in a public export must be checked against that export's manifest.

| Evidence set | Observed scope | Entry |
| --- | --- | --- |
| Apple M4 course delivery | CPU/OpenMP/OpenCL; 36 course shapes plus separately retained batch-4 and confirmation runs | Local course package delivered separately; outside this public evidence export |
| RTX 4090 D historical atlas | 80 points, 5×30 descriptive sampling; N=1/N=4 ShuffleNetV2 block; counter access denied on that host | [analysis](../results/rtx4090d/analysis-20260908/summary.zh-CN.md) |
| RTX 4090 D six-point confirmation | Independent smaller confirmation; not a replacement for all 80 historical points | [combined analysis with separate confirmation](../results/rtx4090d/analysis-20260908/analysis.json) |
| RTX 4090 atlas confirmation | 80×6 rows: 430 PASS, 50 UNSUPPORTED; 10×30 = 129,000 records, each carrying several timing fields | [summary](../results/rtx4090/session-20260908-vast-03/analysis-atlas/summary.zh-CN.md), [raw rows](../results/rtx4090/session-20260908-vast-03/complete-evidence/results/atlas-confirmation/atlas/bench.json), [samples](../results/rtx4090/session-20260908-vast-03/complete-evidence/results/atlas-confirmation/atlas/samples.jsonl) |
| RTX 4090 Nsight Compute and actual SM89 code | 8 shapes, 30 formal captures plus probe, 2 unsupported k2 combinations; static library/SASS audit | [profile](../results/rtx4090/session-20260908-vast-03/analysis-profile/summary.zh-CN.md), [mechanisms](../results/rtx4090/session-20260908-vast-03/analysis-profile/mechanism-review.zh-CN.md) |
| Layout/cache/transfer and direct cuDNN | Boundary units 276/288 PASS, 12 unsupported; Frontend 32/32 PASS | [supplemental summary](../results/rtx4090/session-20260908-vast-03/analysis-supplemental/summary.zh-CN.md) |
| Model layers and MobileNetV2 block | Model units 36/72 PASS, 28 unsupported, 8 planned NOT_RUN; complete blocks N=1/N=4 measured | [module/layer details](../results/rtx4090/session-20260908-vast-03/analysis-supplemental/summary.zh-CN.md) |
| Triton check/tune/holdout | 25 boundary shapes; 40 tuning and 40 disjoint holdout shapes; 15/40 holdout supported; 105 supported comparison units PASS | [summary](../results/rtx4090/session-20260908-vast-03/analysis-triton/summary.zh-CN.md), [final audit](../results/rtx4090/session-20260908-vast-03/analysis-triton/validated.json) |
| Copy/FMA microbenchmarks | 8 configurations, 1,200 samples, 2 separate counter captures | [summary](../results/rtx4090/session-20260908-vast-03/analysis-microbench/summary.zh-CN.md) |

## Main findings

The new 4090 atlas has k0/k3 geometric means of 3.8718 (Graph) and 3.4490 (synchronized API). Tuned PyTorch/k3 is 0.7384/0.6448: no aggregate framework improvement. Graph WIN/LOSS/TIE/UNCERTAIN is 34/35/6/5 against the tuned framework; API is 27/50/2/1. The [complete matrix](../results/rtx4090/session-20260908-vast-03/analysis-atlas/atlas-tuned-vs-k3.png) includes losing and uncertain points. Ten paired batch medians, 10,000 bootstrap resamples, 95% intervals and a 5% practical threshold describe within-run variation, not independent-session stability.

The older 4090 D figures remain 3.919/3.040 for k0/k3 and 0.789/0.645 for tuned PyTorch/k3 under its own five-batch protocol. Do not pool these with the new 4090 or treat new counters as explanations of old-device latency.

MobileNetV2 N=4 Graph block ratio is 1.13849 (WIN); N=1 Graph is 1.04879 (TIE). API ratios 0.66465/0.67902 are both LOSS. Instrumented Amdahl estimates are separate from ordinary timing and do not predict eager API or whole-network performance. Random-weight checks validate observed output equivalence, not classification accuracy.

Triton comparisons remain UNCERTAIN: five batches and separately executed suites do not support paired statistical wins. Supported holdout geometric means versus tuned PyTorch are 1.446 Graph, 0.921 API and 0.550 host-feed. These fifteen supported points do not establish full-atlas or model improvements.

## Completion boundaries

CUDA, real NCU, direct cuDNN and Triton have actual new-device evidence. The 64 independent FP64 points per supported atlas row complement full-output FP32 checking; they are not full-output FP64 coverage. GPU replay, synchronized API, host-feed and profiler timing are distinct. Microbenchmarks have not established a sustainable hardware roofline. Ascend has no source, simulator or NPU receipt. Website publication and job-evidence/ability acceptance require their own receipts.

See [contributions](CONTRIBUTIONS.md), [references/licenses](REFERENCES.md) and [four bilingual questions](interview-map.md) for explanation and reuse boundaries.

## Public verification and attribution

The export includes the exact measured numerical source subset at `evidence/measured-source-031887c/`, its identity at `evidence/measured-source-identity.json`, and generated scope notes at `docs/PUBLIC_EVIDENCE.md`. Run `python scripts/audit_public_evidence.py --root .` from the exported root to verify hashes and the retained atlas samples. The complete private archive and the separate local course package are outside this export.

Project code: [MIT](../LICENSE). Project-owned data/figures: [CC BY 4.0](../DATA_LICENSE). See [NOTICE](../NOTICE). The owner attests to all six contribution areas recorded in [CONTRIBUTIONS](CONTRIBUTIONS.md); this is owner testimony, not an independent interview result.
