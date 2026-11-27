cd /home/ysh-workspace/a5-dsa-template
source phase0-env.sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh
ulimit -c 0
ulimit -f 2048
root=$PWD/FlagTree-v36-gather-r2
build=$PWD/build-phase1/gather-r2/prhead-native
g=$(realpath .tmp/gather-r3-gdb-*/extracted)
export PYTHONPATH="$root/python:$(python3 -c 'import sysconfig; print(sysconfig.get_path("purelib"))')" FLAGTREE_BACKEND=ascend TRITON_BACKENDS_IN_TREE=1 TRITON_BUILD_DIR="$build" PYTHONFAULTHANDLER=1
export TRITON_CACHE_DIR="$PWD/.cache/gather-r5-ub-gdb"
LD_LIBRARY_PATH="$g/usr/lib/aarch64-linux-gnu:$g/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}" timeout -k 5 45 "$g/usr/bin/gdb" -batch -ex 'set pagination off' -ex 'set print thread-events off' -ex 'run' -ex 'bt 24' --args .venv-phase11/bin/python -u "$build/../r5_typed_probe.py" ub
echo "gdb_exit=$?"
