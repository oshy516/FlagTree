#!/usr/bin/env bash
set -euo pipefail
cd /home/ysh-workspace/a5-dsa-template
exec > logs/gather-r3-llvm19-build.log 2>&1
trap 'printf "%s\n" "$?" > logs/gather-r3-llvm19-build.exit' EXIT
source phase0-env.sh
ulimit -c 0
pin=06913395ef8aca66a9814723f4ccc616570b0fed
old=/home/ysh-workspace/.isolated/a5-tla-validation/src/AscendNPU-IR/third-party/llvm-project
dep="$PWD/dependencies/llvm19-$pin"
src="$dep/source"
build="$dep/build"
install="$dep/sdk"
free=$(df -PB1 . | awk 'NR==2 {print $4}')
[ "$free" -ge 35000000000 ] || { echo "insufficient space: $free"; exit 1; }
mkdir -p "$dep"
if [ ! -s "$dep/source-pin" ]; then
 test ! -e "$src"
 mkdir "$src"
 git -C "$old" archive "$pin" | tar -xf - -C "$src"
 test -f "$src/llvm/CMakeLists.txt"
 printf '%s\n' "$pin" > "$dep/source-pin"
fi
test "$(cat "$dep/source-pin")" = "$pin"
cmake -S "$src/llvm" -B "$build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$install" -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_BSPUB_DAVINCI_BISHENGIR=ON -DLLVM_BSPUB_DAVINCI_BISHENGIR_A5=ON -DLLVM_BSPUB_DAVINCI_BISHENGIR_A5_NPUIR=ON -DLLVM_TARGETS_TO_BUILD=AArch64 -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF -DLLVM_INCLUDE_EXAMPLES=OFF -DMLIR_INCLUDE_TESTS=OFF -DMLIR_ENABLE_BINDINGS_PYTHON=OFF -DLLVM_ENABLE_ZSTD=OFF -DLLVM_ENABLE_TERMINFO=OFF -DLLVM_PARALLEL_LINK_JOBS=1 -DLLVM_ENABLE_ASSERTIONS=ON
cmake --build "$build" --target install --parallel 16 &
worker=$!
while kill -0 "$worker" 2>/dev/null; do
  free=$(df -PB1 . | awk 'NR==2 {print $4}')
  if [ "$free" -lt 25000000000 ] || ! a5_check_disk_budget; then
    kill -TERM "$worker"; break
  fi
  sleep 25
done
wait "$worker"
test -s "$install/lib/libLLVMDemangle.a"
test -s "$install/lib/cmake/mlir/MLIRConfig.cmake"
"$install/bin/llvm-config" --version --host-target
