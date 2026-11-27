#!/usr/bin/env bash
set -euo pipefail
cd /home/ysh-workspace/a5-dsa-template
exec >logs/gather-r6-integrated-bishengir.log 2>&1
trap 'echo $? > logs/gather-r6-integrated-bishengir.exit' EXIT
ulimit -c 0
source phase0-env.sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh
d=$PWD/build-phase1/gather-r2
dep=$(realpath dependencies/llvm19-*)
a5_check_disk_budget
cmake -S "$dep/source/llvm" -B "$dep/build" \
 -DLLVM_EXTERNAL_PROJECTS=bishengir -DLLVM_EXTERNAL_BISHENGIR_SOURCE_DIR="$d/AscendNPU-IR-pinned" \
 -DMLIR_INCLUDE_TESTS=ON -DBISHENGIR_ENABLE_TRITON_COMPILE=ON -DBISHENGIR_BUILD_HIVMC=ON \
 -DBISHENGIR_ENABLE_A5_UNPUBLISHED_FEATURES=ON -DBISHENGIR_ENABLE_PM_CL_OPTIONS=ON
cmake --build "$dep/build" --target bishengir-opt --parallel 16
opt="$dep/build/bin/bishengir-opt"
"$opt" --version
"$opt" "$d/mark-custom-l1.mlir" -hivm-mark-tightly-coupled-buffer >logs/gather-r6-patched-tcb.mlir
sed -n '/func.func @custom_l1/,/func.func @copy_control/p' logs/gather-r6-patched-tcb.mlir | grep 'annotation.mark.*hivm.tightly_coupled_buffer'
a5_check_disk_budget
cmake --build "$dep/build" --target bishengir-compile --parallel 16
sha256sum "$opt" "$dep/build/bin/bishengir-compile" >logs/gather-r6-patched-tools.sha256
