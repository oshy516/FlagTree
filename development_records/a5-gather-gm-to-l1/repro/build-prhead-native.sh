#!/usr/bin/env bash
set -euo pipefail
cd /home/ysh-workspace/a5-dsa-template
exec >logs/gather-r3-prhead-native.log 2>&1
trap 'printf "%s\n" "$?" > logs/gather-r3-prhead-native.exit' EXIT
source phase0-env.sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export TRITON_BUILD_NPUIR=OFF
ulimit -c 0
llvm=/home/ysh-workspace/.isolated/a5-tla-validation/cache/.triton/llvm/llvm-f6ded0be-3f5016d1-ubuntu-arm64
ref="$PWD/FlagTree-v36-gather-r2"
build="$PWD/build-phase1/gather-r2/prhead-native"
mkdir -p "$build/wheel/triton/_C"
a5_check_disk_budget
cmake -S "$ref" -B "$build" -G Ninja -DFLAGTREE_BACKEND=ascend -DCMAKE_BUILD_TYPE=Release -DTRITON_BUILD_PYTHON_MODULE=ON -DTRITON_BUILD_PROTON=OFF -DTRITON_BUILD_UT=OFF '-DTRITON_CODEGEN_BACKENDS=nvidia;amd;ascend' -DPython3_EXECUTABLE="$PWD/.venv-phase11/bin/python" -Dpybind11_DIR="$(python3 -m pybind11 --cmakedir)" -DLLVM_SYSPATH="$llvm" -DLLVM_INCLUDE_DIRS="$llvm/include" -DLLVM_LIBRARY_DIR="$llvm/lib" -DMLIR_DIR="$llvm/lib/cmake/mlir" -DLLVM_DIR="$llvm/lib/cmake/llvm" -DTRITON_WHEEL_DIR="$build/wheel/triton" -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="$build/wheel/triton/_C" -DASCEND_CUSTOM_OPS_ARCH_FAMILIES=c310
cmake --build "$build" --target triton triton-opt triton-mlir-opt --parallel 16
a5_check_disk_budget
