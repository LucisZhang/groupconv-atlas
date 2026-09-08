#!/usr/bin/env bash
set -euo pipefail
GC_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$GC_ROOT"
GC_PYTHON="${GC_PYTHON:-$GC_ROOT/.venv/bin/python}"
if [[ ! -x "$GC_PYTHON" ]]; then GC_PYTHON="$(command -v python3 || true)"; fi
GC_ACTION="${1:-check}"
if [[ $# -gt 0 ]]; then shift; fi
gc_build() {
  local build_dir="$1" opencl="$2" cuda="$3"
  shift 3
  local options=()
  if [[ "$(uname -s)" == Darwin ]] && command -v brew >/dev/null 2>&1; then
    local llvm_prefix
    llvm_prefix="$(brew --prefix llvm 2>/dev/null || true)"
    if [[ -x "$llvm_prefix/bin/clang++" ]]; then options+=("-DCMAKE_CXX_COMPILER=$llvm_prefix/bin/clang++"); fi
  fi
  cmake -S "$GC_ROOT" -B "$build_dir" -DCMAKE_BUILD_TYPE=Release \
    -DGC_ENABLE_OPENCL="$opencl" -DGC_ENABLE_CUDA="$cuda" "${options[@]}" "$@"
  cmake --build "$build_dir" --parallel 4
}
case "$GC_ACTION" in
  cpu) gc_build build OFF OFF "$@" ;;
  opencl) gc_build build ON OFF "$@" ;;
  cuda)
    if ! command -v nvcc >/dev/null 2>&1; then
      echo 'CUDA unavailable: nvcc is required. Use the documented Linux CUDA development image.' >&2; exit 2
    fi
    gc_build build OFF ON "$@" ;;
  check)
    ctest --test-dir build --output-on-failure
    "$GC_PYTHON" -m pytest -q tests
    "$GC_PYTHON" -m bench.check "$@" ;;
  bench) "$GC_PYTHON" -m bench.run "$@" ;;
  probe) "$GC_PYTHON" -m scripts.probe "$@" ;;
  analyze) "$GC_PYTHON" -m scripts.analyze "$@" ;;
  sanitize)
    gc_build build-sanitize OFF OFF -DGC_SANITIZE=ON "$@"
    ctest --test-dir build-sanitize --output-on-failure ;;
  *) echo 'Usage: ./run.sh cpu|opencl|cuda|check|bench|probe|analyze|sanitize [options]' >&2; exit 2 ;;
esac
