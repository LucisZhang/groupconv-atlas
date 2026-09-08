# Native CUDA runner integration

`tests/native_cuda.cu` is independent of PyTorch and links the real CUDA backend.
It uses a live nondefault nonblocking stream for input copies, NaN output
initialization, kernel launch, download and synchronization. The full downloaded
output is checked against an independent scalar FP64 cross-correlation oracle
with atol=rtol=1e-4 and finite-output checks. No timing claims are produced.

Add after the existing `enable_testing()` in the top-level CMake file:

```cmake
if(GC_ENABLE_CUDA)
  add_executable(native_cuda tests/native_cuda.cu)
  target_link_libraries(native_cuda PRIVATE groupconv CUDA::cudart)
  set_target_properties(native_cuda PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
  target_compile_options(native_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
  add_test(NAME native_cuda COMMAND native_cuda)
  add_test(NAME native_cuda_contracts COMMAND native_cuda --contracts)
  add_test(NAME native_cuda_stress COMMAND native_cuda --stress)
  add_test(NAME native_cuda_graph COMMAND native_cuda --graph)
endif()
```

The backend already receives `--fmad=false` and uses explicit `fmaf` in its
reductions. Do not add fast math. Example for a verified RTX 4090 / SM 8.9 host:

```sh
cmake -S . -B build-cuda -DGC_ENABLE_CUDA=ON -DGC_ENABLE_OPENCL=OFF -DGC_ENABLE_OPENMP=OFF -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda -j --target native_cuda
./build-cuda/native_cuda
./build-cuda/native_cuda --contracts
./build-cuda/native_cuda --graph
./build-cuda/native_cuda --stress
compute-sanitizer --tool memcheck --error-exitcode 99 ./build-cuda/native_cuda
compute-sanitizer --tool memcheck --error-exitcode 99 ./build-cuda/native_cuda --stress
compute-sanitizer --tool racecheck --error-exitcode 99 ./build-cuda/native_cuda
compute-sanitizer --tool racecheck --error-exitcode 99 ./build-cuda/native_cuda --stress
compute-sanitizer --tool synccheck --error-exitcode 99 ./build-cuda/native_cuda
compute-sanitizer --tool synccheck --error-exitcode 99 ./build-cuda/native_cuda --stress
ncu --set basic --force-overwrite -o native-k2 ./build-cuda/native_cuda --profile --only-impl 2
ncu --set basic --force-overwrite -o native-k3 ./build-cuda/native_cuda --profile --only-impl 3
```

Capture each command's combined output and exit status. A pipeline to `tee`
requires `set -o pipefail`; zero output or a missing report is not a pass.
Exit 77 is unavailable hardware, exit 1 is failure, exit 0 is completed checks.
Expected unsupported cases print `UNSUPPORTED expected=true` and have a separate
summary counter; they are not numerical passes.

Modes are mutually exclusive. The default covers all supported implementations,
cpg=1/2/4/3/8 (k2 only 1/2/4), N=1/4, every specified tiny/non-square shape,
17x33, exact 8x16 tiles, G=1, C=6/G=2, zero/impulse/group-isolation/cancellation,
and deterministic random seeds. k0 also covers unequal Cin/Cout, depthwise
multiplier 2, kernels 3/5/7, stride and dilation 1/2, padding 0/3, plus asymmetric
kernel/stride/dilation. `--only-impl 0..3` narrows numerical/profile tests.

`--stress` runs only k2 at cpg 1/2/4. Each case has 1x17 planes (both spatial axes
have tails), 65,536 total tiles, two iterations in at least one physical block,
and deterministic per-plane inputs and weights checked over every output.
The case label records its computed tile count. This specifically exercises the
barrier before a block reuses shared halo storage. It does not allocate huge
individual images and remains useful under racecheck.

`--graph` captures each k0/k1/k2/k3 launch at cpg=2, H=7, W=13 in a
nondefault stream, instantiates and launches the CUDA graph, downloads on the
same stream and checks the full output against the FP64 oracle. Upload and its
synchronization occur before capture. This is a regression guard for the ABI
stream validation being safe during capture. `--only-impl` can narrow the case.

`--contracts` isolates metadata/pointer rejection from numerical runs. It checks
dtype/rank/strides/device/index mismatches, current-device pointer mismatch,
undersized declared capacity, aliasing and implementation IDs. Pinned host memory
must return GC_INVALID. Pageable host memory may return GC_INVALID or
GC_DEVICE_ERROR depending on runtime pointer-query behavior. Any API error slot
is consumed between cases. No destroyed/fabricated stream handles are passed.
Contract cases use implementation 0 for descriptor tests even with `--only-impl`;
that option filters implementation-specific geometry checks. Multi-device stream
ownership still requires a separate multi-GPU host and is not tested here.

## Local evidence

The macOS preparation host has no `nvcc`. A host C++17 syntax-only check against
minimal temporary CUDA API declarations passed with Clang warnings enabled;
this checks runner syntax only, not CUDA header compatibility, linkage or device
behavior. Real nvcc compilation, CUDA device results, sanitizer results and NCU
reports remain NOT_RUN until the commands above execute on the target host.
