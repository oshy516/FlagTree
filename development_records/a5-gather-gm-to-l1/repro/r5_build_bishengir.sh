#!/usr/bin/env bash
set -euo pipefail
cd /home/ysh-workspace/a5-dsa-template
exec >logs/gather-r5-patched-bishengir.log 2>&1
trap 'printf "%s\n" "$?" > logs/gather-r5-patched-bishengir.exit' EXIT
source phase0-env.sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh
ulimit -c 0
until [ -s logs/gather-r5-sdk-resume.exit ]; do sleep 30; done
test "$(cat logs/gather-r5-sdk-resume.exit)" = 0
d="$PWD/build-phase1/gather-r2"
dep="$PWD/dependencies/llvm19-06913395ef8aca66a9814723f4ccc616570b0fed"
sdk="$dep/sdk"
build="$d/patched-bishengir-build"
test -s "$sdk/lib/libLLVMDemangle.a"
test "$(git -C "$d/AscendNPU-IR-pinned" rev-parse HEAD)" = efb7141581c009bbd1d676b18941ed15ede62a68
git -C "$d/AscendNPU-IR-pinned" diff --check
a5_check_disk_budget
cmake -S "$d/compiler-cmake" -B "$build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLVM_DIR="$sdk/lib/cmake/llvm" -DMLIR_DIR="$sdk/lib/cmake/mlir" -DLLVM_MAIN_SRC_DIR="$dep/source/llvm" -DLLVM_NATIVE_ARCH=AArch64 -DLLVM_TABLEGEN_EXE="$sdk/bin/llvm-tblgen" -DMLIR_TABLEGEN_EXE="$sdk/bin/mlir-tblgen" -DBISHENGIR_BUILD_STANDALONE_IR_ONLY=OFF -DBISHENGIR_BUILD_HIVMC=ON -DBISHENGIR_ENABLE_A5_UNPUBLISHED_FEATURES=ON
cmake --build "$build" --target bishengir-opt --parallel 2
opt=$(find "$build" -type f -name bishengir-opt -perm /111 -print -quit)
test -x "$opt"
"$opt" --version
"$opt" "$d/mark-custom-l1.mlir" -hivm-mark-tightly-coupled-buffer >logs/gather-r5-patched-tcb.mlir
sed -n '/func.func @custom_l1/,/func.func @copy_control/p' logs/gather-r5-patched-tcb.mlir | grep 'annotation.mark.*hivm.tightly_coupled_buffer'
sha256sum "$opt" >logs/gather-r5-patched-opt.sha256
a5_check_disk_budget
cmake --build "$build" --target bishengir-compile --parallel 2
compile=$(find "$build" -type f -name bishengir-compile -perm /111 -print -quit)
test -x "$compile"
sha256sum "$opt" "$compile" >logs/gather-r5-patched-tools.sha256
"$compile" --version
a5_check_disk_budget
