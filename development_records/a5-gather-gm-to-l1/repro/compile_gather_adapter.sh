#!/usr/bin/env bash
set -euo pipefail
[[ $# == 6 ]] || {
  echo "Usage: $0 COMPILER_PREFIX CANN_ROOT GATHER_DEVICE_BC ASCEND_LIBDEVICE ADAPTER_MLIR OUTPUT_PREFIX" >&2
  exit 2
}
tools=$(realpath "$1")
cann=$(realpath "$2")
gather=$(realpath "$3")
libdevice=$(realpath "$4")
input=$(realpath "$5")
output=$(realpath -m "$6")
mkdir -p "$(dirname "$output")"
set +u
source "$cann/set_env.sh"
set -u
ulimit -c 0
sha256sum "$tools/bin/bishengir-compile" "$gather" "$libdevice" "$input" >"$output.inputs.sha256"
"$tools/bin/bishengir-compile" "$input" --target=Ascend950DT_9582 \
  --enable-legacy-insert-load-store-for-mix-cv=False --enable-auto-multi-buffer=False \
  --enable-auto-bind-sub-block=True --enable-optimized-metaop=True --disable-ffts \
  --link-aicore-bitcode="$gather" --enable-auto-blockify-loop \
  --enable-hfusion-compile=true --enable-triton-kernel-compile=true \
  "--append-bisheng-options=-cce-link-aicore-ll-module $libdevice" --enable-vf-merge-level=1 \
  --mlir-disable-threading --mlir-print-ir-module-scope \
  --mlir-print-ir-before=hivm-insert-load-store-for-mix-cv \
  --mlir-print-ir-after=hivm-normalize-matmul,hivm-insert-load-store-for-mix-cv,hivm-combine-optimized-convert-layout,insert-workspace-for-mix-cv,hivm-mark-tightly-coupled-buffer,hivm-split-mix-kernel,hivm-plan-memory-regbase \
  -o "$output" >"$output.passes.log" 2>&1
test -s "$output.o"
sha256sum "$output.o" >"$output.elf.sha256"
