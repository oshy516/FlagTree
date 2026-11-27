# Source this file before using project tools.
_a5_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
export TMPDIR="$_a5_root/.tmp" XDG_CACHE_HOME="$_a5_root/.cache"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip" TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export CMAKE_BUILD_PARALLEL_LEVEL=2 MAKEFLAGS=-j2 MAX_JOBS=2
a5_check_disk_budget() {
  local usage
  usage=$(du -s -B1 -- "$TMPDIR" "$XDG_CACHE_HOME" | awk '{n += $1} END {print n+0}') || return 1
  if (( usage >= 4000000000 )); then
    printf 'A5 project temporary files and cache reached 4 GB. Stop and review project artifacts before continuing.\n' >&2
    return 1
  fi
}
unset _a5_root
a5_check_disk_budget
