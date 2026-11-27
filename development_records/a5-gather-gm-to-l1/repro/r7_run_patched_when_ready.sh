#!/usr/bin/env bash
set -eu
cd /home/ysh-workspace/a5-dsa-template
dep=$(realpath dependencies/llvm19-*)
exit_file=logs/gather-r6-integrated-bishengir.exit
while test ! -s "$exit_file"; do sleep 30; done
test "$(cat "$exit_file")" = 0
sha256sum -c logs/gather-r6-patched-tools.sha256
export R7_BISHENGIR_ROOT="$dep/build"
bash build-phase1/gather-r2/r7_gather_device.sh
