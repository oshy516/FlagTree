#!/usr/bin/env bash
set -eo pipefail
cd /home/ysh-workspace/a5-dsa-template
source phase0-env.sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh
ulimit -c 0
a5_check_disk_budget
root="$PWD/FlagTree-v36-gather-r2"
build="$PWD/build-phase1/gather-r2/prhead-native"
export PYTHONPATH="$root/python:$(python3 -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
export FLAGTREE_BACKEND=ascend TRITON_BACKENDS_IN_TREE=1 TRITON_BUILD_DIR="$build"
export PYTHONFAULTHANDLER=1
export TRITON_CACHE_DIR
TRITON_CACHE_DIR=$(mktemp -d "$PWD/.cache/r6-dot-XXXXXX")
timeout -k 5 40 .venv-phase11/bin/python -u build-phase1/gather-r2/prhead_baseline.py dot
