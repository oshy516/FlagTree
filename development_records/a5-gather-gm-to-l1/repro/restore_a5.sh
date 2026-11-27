#!/usr/bin/env bash
set -euo pipefail
usage() {
  echo "Usage: $0 consumer WORKDIR COMPILER_CHECKOUT CANN_ROOT [JOBS]" >&2
  echo "       $0 producer WORKDIR FLAGTREE_CHECKOUT LLVM22_SDK PYTHON CANN_ROOT [JOBS]" >&2
  exit 2
}
[[ $# -ge 4 ]] || usage
mode=$1
work=$(realpath -m "$2")
src=$(realpath "$3")
records=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
case "$mode" in
  consumer) [[ $# == 4 || $# == 5 ]] || usage; cann=$(realpath "$4"); jobs=${5:-16} ;;
  producer) [[ $# == 6 || $# == 7 ]] || usage; sdk=$(realpath "$4"); python=$(realpath -s "$5"); cann=$(realpath "$6"); jobs=${7:-16} ;;
  *) usage ;;
esac
[[ $jobs =~ ^(1[6-9]|2[0-9]|3[0-2])$ ]]
[[ $(uname -m) == aarch64 ]]
[[ -f $cann/set_env.sh ]]
set +u
source "$cann/set_env.sh"
set -u
mkdir -p "$work"
# Serialize all restores using this work directory.
exec 9>"$work/.build.lock"
flock -n 9
ulimit -c 0

case "$mode" in
  consumer)
    pin=efb7141581c009bbd1d676b18941ed15ede62a68
    llvm19=06913395ef8aca66a9814723f4ccc616570b0fed
    [[ $(git -C "$src" rev-parse HEAD) == "$pin" ]]
    git -C "$src" submodule update --init third-party/llvm-project
    llvm="$src/third-party/llvm-project"
    [[ $(git -C "$llvm" rev-parse HEAD) == "$llvm19" ]]
    for patch in pinned-compiler r7-matmul-namespace r8-arith-aptype-gcc r9-static-link r11-gather-cbuf-abi; do
      patch="$records/patches/$patch.patch"
      if ! git -C "$src" apply --reverse --check "$patch" 2>/dev/null; then
        git -C "$src" apply --check "$patch"
        git -C "$src" apply "$patch"
      fi
    done
    build="$work/llvm19-bishengir"
    cmake -S "$llvm/llvm" -B "$build" -G Ninja \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$work/llvm19-sdk" \
      -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_TARGETS_TO_BUILD=AArch64 \
      -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_ENABLE_RTTI=OFF -DLLVM_PARALLEL_LINK_JOBS=1 \
      -DLLVM_BSPUB_DAVINCI=ON -DLLVM_BSPUB_DAVINCI_BISHENGIR=ON \
      -DBSPUB_DAVINCI_BISHENGIR=ON -DLLVM_BSPUB_DAVINCI_BISHENGIR_A5=ON \
      -DLLVM_BSPUB_DAVINCI_BISHENGIR_A5_NPUIR=ON \
      -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF -DLLVM_INCLUDE_EXAMPLES=OFF \
      -DMLIR_INCLUDE_TESTS=ON -DMLIR_ENABLE_BINDINGS_PYTHON=OFF \
      -DLLVM_ENABLE_ZSTD=OFF -DLLVM_ENABLE_TERMINFO=OFF \
      -DLLVM_EXTERNAL_PROJECTS=bishengir -DLLVM_EXTERNAL_BISHENGIR_SOURCE_DIR="$src" \
      -DBISHENGIR_ENABLE_TRITON_COMPILE=ON -DBISHENGIR_BUILD_HIVMC=ON \
      -DBISHENGIR_ENABLE_A5_UNPUBLISHED_FEATURES=ON -DBISHENGIR_ENABLE_PM_CL_OPTIONS=ON \
      -DBISHENGIR_BUILD_TEMPLATE=OFF
    cmake --build "$build" --target bishengir-opt bishengir-compile llvm-config --parallel "$jobs"
    grep -q '^#define BSPUB_DAVINCI_BISHENGIR 1' "$build/tools/mlir/include/mlir/Config/mlir-config.h"
    assets=("$cann"/tools/bishengir/lib/*c310*.bc "$cann"/tools/bishengir/lib/host.*.bc)
    for asset in "${assets[@]}"; do
      [[ -f "$asset" ]]
      dest="$build/lib/${asset##*/}"
      [[ -e "$dest" ]] || ln -s -- "$asset" "$dest"
      cmp -s -- "$asset" "$dest"
    done
    "$build/bin/llvm-config" --version --host-target
    "$build/bin/bishengir-opt" --version
    sha256sum "$build/bin/bishengir-opt" "$build/bin/bishengir-compile" "${assets[@]}" >"$work/consumer.sha256"
    ;;
  producer)
    git -C "$src" merge-base --is-ancestor a21c93a71855a6f299d05e4a630c82452ebccb9f HEAD
    [[ $(cat "$src/cmake/llvm-hash.txt") == f6ded0be897e2878612dd903f7e8bb85448269e5 ]]
    [[ $("$sdk/bin/llvm-config" --version) == 22.* ]]
    [[ -f $sdk/lib/cmake/mlir/MLIRConfig.cmake ]]
    build="$work/flagtree"
    export TRITON_BUILD_NPUIR=OFF
    cmake -S "$src" -B "$build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DFLAGTREE_BACKEND=ascend -DTRITON_BUILD_PYTHON_MODULE=ON \
      -DTRITON_BUILD_PROTON=OFF -DTRITON_BUILD_FLAGPRISM=OFF -DTRITON_BUILD_UT=OFF \
      '-DTRITON_CODEGEN_BACKENDS=nvidia;amd;ascend' -DASCEND_CUSTOM_OPS_ARCH_FAMILIES=c310 \
      -DPython3_EXECUTABLE="$python" -Dpybind11_DIR="$("$python" -m pybind11 --cmakedir)" \
      -DLLVM_SYSPATH="$sdk" -DLLVM_INCLUDE_DIRS="$sdk/include" -DLLVM_LIBRARY_DIR="$sdk/lib" \
      -DLLVM_DIR="$sdk/lib/cmake/llvm" -DMLIR_DIR="$sdk/lib/cmake/mlir" \
      -DTRITON_WHEEL_DIR="$build/wheel/triton" -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="$build/wheel/triton/_C"
    cmake --build "$build" --target triton triton-opt triton-mlir-opt --parallel "$jobs"
    find "$build" -type f \( -name libtriton.so -o -name triton-opt -o -name triton-mlir-opt \) -print0 |
      sort -z | xargs -0 -r sha256sum >"$work/producer.sha256"
    test -s "$work/producer.sha256"
    ;;
esac
