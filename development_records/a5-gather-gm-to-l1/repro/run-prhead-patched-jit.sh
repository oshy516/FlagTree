#!/usr/bin/env bash
set -euo pipefail
cd /home/ysh-workspace/a5-dsa-template
exec > logs/gather-r3-prhead-patched-jit.log 2>&1
trap 'printf "%s\n" "$?" > logs/gather-r3-prhead-patched-jit.exit' EXIT
source phase0-env.sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh
ulimit -c 0
for stage in prhead-baseline patched-bishengir; do
 until [ -s "logs/gather-r3-$stage.exit" ]; do sleep 30; done
 test "$(cat "logs/gather-r3-$stage.exit")" = 0
done
grep -Fxq 'dot 0' logs/gather-r3-prhead-baseline.log
grep -Fxq 'ub 0' logs/gather-r3-prhead-baseline.log
sha256sum -c logs/gather-r3-patched-tools.sha256
root="$PWD/FlagTree-v36-gather-r2"
build="$PWD/build-phase1/gather-r2/prhead-native"
toolbuild="$PWD/build-phase1/gather-r2/patched-bishengir-build"
opt=$(find "$toolbuild" -type f -name bishengir-opt -perm /111 -print -quit)
compile=$(find "$toolbuild" -type f -name bishengir-compile -perm /111 -print -quit)
export PATH="$(dirname "$opt"):$(dirname "$compile"):$PATH"
export PYTHONPATH="$root/python:$(python3 -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
export FLAGTREE_BACKEND=ascend TRITON_BACKENDS_IN_TREE=1 TRITON_BUILD_DIR="$build"
export PYTHONFAULTHANDLER=1 PYTHONNOUSERSITE=1 PYTHONPYCACHEPREFIX="$XDG_CACHE_HOME/py"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/gather-r3-patched"
a5_check_disk_budget
timeout -k 5 90 .venv-phase11/bin/python -u build-phase1/gather-r2/diagnose-prhead-patched.py
a5_check_disk_budget
