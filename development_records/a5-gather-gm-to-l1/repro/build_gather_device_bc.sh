#!/usr/bin/env bash
set -euo pipefail
[[ $# == 4 ]] || {
  echo "Usage: $0 FLAGTREE_CHECKOUT COMPILER_CHECKOUT CANN_ROOT OUTPUT_DIR" >&2
  exit 2
}
src=$(realpath "$1")/python/triton/experimental/tle/language/dsa/ascend/custom_ops/mem_ops/gather_gm_to_l1.cpp
template=$(realpath "$2")/bishengir/lib/Template
cann=$(realpath "$3")
out=$(realpath -m "$4")
asc="$cann/asc"
set +u
source "$cann/set_env.sh"
set -u
mkdir -p "$out"
ccec=$(command -v ccec)
args=(-O2 -x cce --cce-auto-sync=off --cce-aicore-only --cce-generic-addrspace=off
      -mllvm -disable-llvm-optzns --cce-aicore-arch=dav-c310-cube --cce-enable-print
      --cce-enable-sanitizer -std=c++17 --cce-mask-opt -DTILING_KEY_VAR=0
      -I "$template/lib/RegBase/Cube/include" -I "$template/include"
      -I "$asc" -I "$asc/impl/basic_api" -I "$asc/include" -I "$asc/include/basic_api")
"$ccec" "${args[@]}" "$src" -emit-llvm -c -o "$out/gather-normal.bc"
"$ccec" "${args[@]}" -cce-enable-mix -mllvm -enable-mix=true "$src" -emit-llvm -c -o "$out/gather-mix.bc"
sha256sum "$src" "$out/gather-normal.bc" "$out/gather-mix.bc" >"$out/gather.sha256"
