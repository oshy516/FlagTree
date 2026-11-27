#!/usr/bin/env bash
set -eo pipefail
cd /home/ysh-workspace/a5-dsa-template
source phase0-env.sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh
ulimit -c 0
npu-smi info -t proc-mem -i 2 -p 0 | tee /dev/stderr | grep -q 'No process in device'
export ASCEND_RT_VISIBLE_DEVICES=2
export PATH="$R7_BISHENGIR_ROOT/bin:$PATH"
export PYTHONPATH="$PWD/FlagTree-v36-gather-r2/python:$(python3 -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
export FLAGTREE_BACKEND=ascend TRITON_BACKENDS_IN_TREE=1 TRITON_BUILD_DIR="$PWD/build-phase1/gather-r2/prhead-native"
export TRITON_CACHE_DIR="$PWD/.cache/gather-r7-gather-$(basename "$R7_BISHENGIR_ROOT")" PYTHONFAULTHANDLER=1
timeout -k 5 100 .venv-phase11/bin/python -u build-phase1/gather-r2/r7_gather_device.py
