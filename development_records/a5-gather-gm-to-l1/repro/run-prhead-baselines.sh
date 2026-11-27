#!/usr/bin/env bash
set -euo pipefail
cd /home/ysh-workspace/a5-dsa-template
exec > logs/gather-r3-prhead-baseline.log 2>&1
trap 'printf "%s\n" "$?" > logs/gather-r3-prhead-baseline.exit' EXIT
source phase0-env.sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh
ulimit -c 0
until [ -s logs/gather-r3-prhead-native.exit ]; do sleep 15; done
test "$(cat logs/gather-r3-prhead-native.exit)" = 0
root="$PWD/FlagTree-v36-gather-r2"
build="$PWD/build-phase1/gather-r2/prhead-native"
pkg="$root/python/triton"
link_once() {
 if [ ! -e "$2" ] && [ ! -L "$2" ]; then ln -s -- "$1" "$2"; fi
 test "$(realpath "$1")" = "$(realpath "$2")"
}
mkdir -p "$pkg/spec/ascend" "$build/bin"
link_once "$root/third_party/ascend/backend" "$pkg/backends/ascend"
link_once "$root/third_party/ascend/language/cann" "$pkg/language/extra/cann"
link_once "$root/third_party/ascend/spec/triton" "$pkg/spec/ascend/triton"
link_once "$build/wheel/triton/_C/libtriton.so" "$pkg/_C/libtriton.so"
if [ ! -e "$pkg/FLAGTREE_BACKEND" ]; then printf 'ascend\n' > "$pkg/FLAGTREE_BACKEND"; fi
test "$(cat "$pkg/FLAGTREE_BACKEND")" = ascend
for tool in triton-opt triton-mlir-opt; do
 test -x "$build/third_party/ascend/bin/$tool"
 link_once "$build/third_party/ascend/bin/$tool" "$build/bin/$tool"
done
sha256sum "$build/bin/triton-opt" "$build/bin/triton-mlir-opt" "$build/wheel/triton/_C/libtriton.so"
export PYTHONPATH="$root/python:$(python3 -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
export FLAGTREE_BACKEND=ascend TRITON_BACKENDS_IN_TREE=1 TRITON_BUILD_DIR="$build"
export PYTHONFAULTHANDLER=1 PYTHONNOUSERSITE=1 PYTHONPYCACHEPREFIX="$XDG_CACHE_HOME/py"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/gather-r3"
for case in dot ub; do
 a5_check_disk_budget
 rc=0
 timeout -k 5 55 .venv-phase11/bin/python -u build-phase1/gather-r2/prhead_baseline.py "$case" >"logs/gather-r3-prhead-$case.log" 2>&1 || rc=$?
 printf '%s %s\n' "$case" "$rc"
done
a5_check_disk_budget
