#!/usr/bin/env bash
set -euo pipefail
[[ $# == 10 ]] || {
  echo "Usage: $0 FLAGTREE PRODUCER_BUILD PYTHON COMPILER_PREFIX CANN_ROOT AIC_BC GENERIC_FP16_ADAPTER OFFLINE_RESULTS LOG_DIR NPU_ID" >&2
  exit 2
}
export R11_FLAGTREE=$(realpath "$1") TRITON_BUILD_DIR=$(realpath "$2")
python=$(realpath -s "$3")
export R7_BISHENGIR_ROOT=$(realpath "$4")
cann=$(realpath "$5")
export R11_GATHER_BC=$(realpath "$6") R11_GENERIC_ADAPTER=$(realpath "$7")
offline=$(realpath "$8")
logs=$(realpath -m "$9")
device=${10}
script=$(realpath "$(dirname "$0")/r11_gather_replay.py")
mkdir -p "$logs"
exec 9>"$logs/.device-replay.lock"
flock -n 9
set +u
source "$cann/set_env.sh"
set -u
export PYTHONPATH="$R11_FLAGTREE/python${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$R7_BISHENGIR_ROOT/bin:$PATH" ASCEND_RT_VISIBLE_DEVICES="$device"
export FLAGTREE_BACKEND=ascend TRITON_BACKENDS_IN_TREE=1 PYTHONFAULTHANDLER=1
ulimit -c 0
for dtype in fp16 bf16; do
  sha256sum -c "$offline/$dtype.elf.sha256"
  sha256sum -c "$offline/$dtype.inputs.sha256"
  grep -q 'hir.custom.*i64, i64).*outs.*address_space<cbuf>' "$offline/$dtype.passes.log"
  npu-smi info -t proc-mem -i "$device" -p 0 >"$logs/$dtype.occupancy.log"
  grep -q 'No process in device' "$logs/$dtype.occupancy.log"
  export R11_DTYPE="$dtype"
  export TRITON_CACHE_DIR=$(mktemp -d "$logs/cache-$dtype.XXXXXX")
  timeout -k 5 100 "$python" -u "$script" >"$logs/$dtype.device.log" 2>&1
  test "$(grep -c '^GATHER_DEVICE_PASS True ' "$logs/$dtype.device.log")" = 3
done
