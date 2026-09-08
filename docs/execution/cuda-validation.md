# CUDA validation and support

Source implementation: `csrc/cuda/groupconv.cu`. The no-CUDA build uses
`csrc/cuda/stub.cpp`, reports availability 0, and returns `GC_UNAVAILABLE`.
Do not link the stub and CUDA translation unit together.

The following is the implemented support contract. CUDA compilation and the
declared test suite passed on a rented RTX 4090 D on 2026-09-08. Test coverage
does not establish correctness for every possible geometry or a speed advantage.

| ID | Mapping | Supported geometry |
|---|---|---|
| 0 / k0 | One output/thread, grid-stride flattened NCHW | All valid contiguous FP32 geometry accepted by the common shape validator, including unequal Cin/Cout, depthwise multipliers, 3/5/7 kernels, stride and dilation |
| 1 / k1 | 16 × 8 spatial tile per output channel | B geometry, every legal group count |
| 2 / k2 | 16 × 8 spatial tile plus 1-pixel shared input halo | B geometry with Cin/groups = 1, 2 or 4 |
| 3 / k3 | Four adjacent horizontal outputs/thread, six input values per row in registers | B geometry, every legal group count |

B geometry means Cin=Cout, 3×3 kernel, stride=1, padding=1, dilation=1.
All kernels implement cross-correlation without bias and use FP32 FMA reductions.
N and H/W need not be block multiples; positive non-square and tiny dimensions
are supported. k2 includes both `depthwise_m1` and `grouped_small_cpg` reductions.
k3 is independent of k2, providing a separate optimization comparison.
No implementation routes to another implementation when unsupported.

Descriptors are borrowed and validated through `gc_validate_tensors` for shape,
dtype, device metadata, contiguous strides, alignment, byte capacity and aliasing.
The CUDA entry point also queries pointer attributes: all three buffers must be
CUDA device allocations on their declared device and the caller's current device.
Managed and mapped-host pointers are outside this contract. The byte-capacity
field remains a caller assertion: pointer attributes establish memory kind and
device, not the allocation's remaining byte capacity. Buffer lifetime and capacity
must remain valid through stream completion. There are no allocations, copies,
device changes or host dereferences of tensor contents in the convolution call.

The supplied stream is used for every launch; null means CUDA's default stream.
The entry point checks stream capture status and launch errors but does not synchronize.
`cudaStreamIsCapturing` validates the handle during eager calls and graph capture;
the initial `cudaStreamGetFlags` check returned error 900 during CUDA 12.8 capture
on the rented RTX 4090D and was replaced after an isolated API-level reproduction.
The caller must check asynchronous errors at stream/event synchronization. A
stream must belong to the current device; test cross-device rejection on a
multi-GPU host. A prior asynchronous CUDA failure can also surface at an API
check, so run negative-pointer/stream tests separately from numerical cases.

## Required device run

1. Record GPU model, compute capability, driver, CUDA toolkit/compiler, host
   compiler, source commit and build flags. Configure the optional CUDA backend
   with an explicit CMake `CMAKE_CUDA_ARCHITECTURES` value matching the measured
   GPU (for example `86` only for a compute capability 8.6 device). Retain the
   configure/build logs; do not enable fast math. Compilation is its own gate.
2. Run full output comparisons against same-policy FP32 PyTorch and independent
   FP64 CPU reference for small cases, with `atol=rtol=1e-4`; reject NaN/Inf for
   finite test inputs. Include zero, impulse, group-isolation, cancellation and
   multiple random seeds. Save maximum absolute/relative error, RMS and failures.
3. Exercise k0 at Cin≠Cout; multiplier=2; 3×3, 5×5 and 7×7; stride=2;
   dilation=2; padding=0 and nonzero. Exercise B kernels at cpg=1/2/4 and
   k1/k3 at cpg=3/8 and G=1. Check `GC_UNSUPPORTED` for k2 cpg=3/8 and for
   k1/k2/k3 outside B; do not count these cases as numerical passes.
4. Cover `(H,W)=(1,1),(1,19),(11,1),(7,13),(9,17),(17,33)`, N=1 and N=4,
   and C=6,G=2. For k2, test both full and partial blocks for every supported
   cpg. The second barrier protects shared storage before the next grid-stride
   iteration; include more than 65,535 spatial tiles to exercise that reuse.
5. Create a nondefault nonblocking stream. Initialize/copy inputs, launch
   `gc_cuda_conv` with its handle, then copy/read outputs and record completion
   on that same stream. Synchronize it and compare full outputs. Do not rely
   on default-stream synchronization to order the work. Also cover bad dtype,
   rank, strides, device metadata, host pointers, undersized declared buffers,
   aliasing and an invalid implementation ID. Use live device allocations for
   positive tests and keep all referenced objects alive until synchronization.
6. Run the device correctness runner under each command below, including tails,
   non-square shapes and k2's multi-iteration blocks. Save exit status and logs.
   Run each command once normally and once with `--stress`.

   ```sh
   compute-sanitizer --tool memcheck --error-exitcode 99 ./build/native_cuda
   compute-sanitizer --tool racecheck --error-exitcode 99 ./build/native_cuda
   compute-sanitizer --tool synccheck --error-exitcode 99 ./build/native_cuda
   ```

Only after these gates pass should same-stream event timings enter an ablation
table. Report k0/k1/k2/k3 separately and preserve unsupported and failed shapes.
Inspect register count/spills for k3 and shared-memory/occupancy costs for k2;
the mapping changes do not establish a speedup by themselves.

## Current evidence boundary

The authoritative device run is
`results/rtx4090d/session-20260908-v2/`. The source tree digest is
`992f2ec19481398fa93a62000b37d4e0510ba19ef8c8cb9a7dcdeb179f5ccb59`;
the measured library digest is
`27103c097b3e27fa6c942d2a7a23c625315a2b50dc929331318a3e4762dbae31`.
The uploaded source snapshot records local base commit `445faf5` and the
uncommitted file hashes. The remote archive has no Git metadata.

- CUDA 12.8, SM 8.9 compilation passed. Compiler output reports registers
  k0/k1/k2/k3 = 64/46/40/64, zero spill loads/stores, and k2 shared memory
  720/1440/2880 bytes for cpg 1/2/4. These are compiler resource reports,
  not measured occupancy or memory traffic.
- All five CTest cases passed: CPU, CUDA numerical, CUDA contracts, CUDA
  shared-storage stress, and CUDA Graph regression. Python tests passed
  16 tests and 8 subtests on the real GPU.
- Memcheck, racecheck and synccheck passed both ordinary numerical and
  shared-storage stress runners. The stress runner reaches 65,536 spatial
  tiles, exercising reuse beyond the 65,535-block launch cap.
- The Python CUDA correctness report covers 36 course, 23 boundary/general
  and 3 batch-4 shapes, with 1,256 PASS and 264 UNSUPPORTED records and no
  FAIL or NOT_RUN. These are implementation/input/reference records, not
  distinct shapes: FP32 references compare every output; independent FP64
  references cover every boundary output and fixed points on larger shapes.
  The 16 common/CPU contract checks in this JSON are distinct from the
  native CUDA contract test in CTest.
- The initial eager run passed, while its graph smoke exposed error 900 from
  `cudaStreamGetFlags`. An isolated API probe reproduced the failure. The
  stream-query repair and `native_cuda --graph` regression passed, followed
  by a complete validation rerun. The earlier run remains retained as
  before-fix evidence and is excluded from the formal performance dataset.
- Nsight Compute 2025.1.1 was installed, but all four kernel profiles failed
  with `ERR_NVGPUCTRPERM`. Hardware-counter profiling remains **BLOCKED**
  on this instance. Separate PyTorch profiler traces record observed kernel
  dispatch; they do not supply DRAM traffic, occupancy or roofline evidence.

Only one GPU was rented; cross-device rejection is not tested. Native tests
use a nondefault nonblocking stream and include full-output FP64 comparisons.
The local macOS build still cannot execute CUDA. Formal timing, confirmation
and module results are recorded separately in `cuda-session-20260908.md`.

## Reproduce this CUDA session

Use the matching Linux development image and an explicit shutdown deadline.
The macOS dependency lock does not describe this CUDA environment; the session
includes the actual pip freeze and environment records.

```sh
python -m scripts.cloud_cuda_session --phase probe --output results/my-cuda-run --seconds 600
python -m scripts.cloud_cuda_session --phase validate --output results/my-cuda-run --seconds 1800
python -m scripts.cloud_cuda_session --phase measure --output results/my-cuda-run --seconds 7200
```

Each phase writes exit statuses and log hashes. Measurement requires the
matching successful correctness report, source files and built library.
Device measurements use graph-internal CUDA events; eager synchronized API
measurements are separate. Finishing the script does not shut down the instance.
