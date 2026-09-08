# Public evidence package

This export contains actual timing samples, numerical checks, analysis tables,
Nsight CSV exports, and the exact frozen numerical source/configuration subset.
Each manifest entry records its public hash, original input hash, evidence class,
and whether bytes are identical. Identifying metadata strings are redacted;
numeric JSON values, timing samples, and result statuses remain unchanged.

The RTX 4090 D five-batch experiment and RTX 4090 ten-batch confirmation remain
separate datasets. The public source tree is a maintained version; the measured
source directory contains the numerical files used for the newer experiments.

The complete private archive additionally contains host logs, opaque profiler
reports, binaries, and authoring/operational files. Those are outside this bounded
export. Audit conclusions and original hashes support traceability; this export
does not independently prove every archived execution or reproduce the complete
private audit unchanged. The frozen Apple OpenCL course package is kept separately.
The pinned CPU model-export receipt retains its generic temporary command paths;
they are historical invocation arguments, not personal paths or runnable setup steps.

Run `python scripts/audit_public_evidence.py --root .` to validate exported hashes,
the complete 80-shape sample denominator, batch identities, and geometric ratios.
GPU execution requires compatible real hardware; see the README for scope.
