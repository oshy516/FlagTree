# Restore A5 Gather development

## Source revisions

| Component | Source and revision |
| --- | --- |
| FlagTree | `oshy516/FlagTree`, branch `feature/a5-gather-gm-to-l1-r2`. R10 checkpoint `a21c93a71855a6f299d05e4a630c82452ebccb9f`. PR #1767 original head `e27584c7977d8f0818d50e8115a7308552042a0b`. |
| Actual AscendNPU-IR compiler | `Ascend/AscendNPU-IR` on GitCode, pin `efb7141581c009bbd1d676b18941ed15ede62a68`. |
| GitCode development branch | `Estella516/AscendNPU-IR_shyuan`, `feature/a5-customop-l1-tcb`. Base `932115cd01b709dbe37500f5828be91c6cb1146b`, TCB commit `4f9f2c0`. |
| LLVM 19 consumer | Ascend LLVM 19.1.7, `06913395ef8aca66a9814723f4ccc616570b0fed`, compiler gitlink `third-party/llvm-project`. |
| LLVM 22 producer | FlagTree pin `f6ded0be897e2878612dd903f7e8bb85448269e5`, AArch64 LLVM/MLIR SDK. |
| Other compiler gitlinks | shmem `6aad41fd93cf17c0e0a7cc87f531ecf455d622e8`, torch-mlir `5e9c503194b644ef53e3f788ade7afc76ce269e4`. The host tool build initializes only LLVM. |
| Runtime and device assets | CANN 9.2.1 AArch64, `Ascend950DT_9582` / c310. |

Use a fresh AArch64 working directory, with CMake, Ninja, GCC, Python 3.12, pybind11 and a matched torch / torch_npu environment. Set CANN_ROOT to the CANN version directory containing set_env.sh and tools/bishengir/lib. Use the LLVM 22 SDK identified by the FlagTree pin, with headers, libraries and LLVM/MLIR CMake exports. An LLVM major version check alone does not establish the SDK pin.

## Build from source

1. Check out the FlagTree branch and a separate AscendNPU-IR tree at the actual compiler pin. Run `bash repro/restore_a5.sh consumer WORK COMPILER_CHECKOUT CANN_ROOT 16` from this records directory. The script starts at the LLVM source tree, with no historical CMake cache. It applies `pinned-compiler.patch`, `r7-matmul-namespace.patch`, `r8-arith-aptype-gcc.patch`, `r9-static-link.patch`, and `r11-gather-cbuf-abi.patch` in order. The GitCode TCB patch is for its own baseline.
2. Run `bash repro/restore_a5.sh producer WORK FLAGTREE_CHECKOUT LLVM22_SDK PYTHON CANN_ROOT 16`. The script uses the explicit Python interpreter and LLVM 22 prefix. Follow the checkout's producer dependency declarations for optional features. Install the matching Ascend backend libdevice at the path reported by get_libdevice().
3. Keep the two invocations serialized or use the same work directory lock. The scripts accept 16–32 jobs. Use the two tools from the same consumer prefix. The script stages CANN c310 meta-op bitcode in its lib directory and emits separate SHA256 manifests for producer and consumer.

## Artifacts and verification

| Artifact | Relative path or source |
| --- | --- |
| LLVM 19 build and generated config | `WORK/llvm19-bishengir` and `tools/mlir/include/mlir/Config/mlir-config.h` below it |
| Patched consumer binaries | `WORK/llvm19-bishengir/bin/bishengir-opt` and `bishengir-compile` |
| Producer Python binding and tools | `WORK/flagtree/wheel/triton/_C/libtriton.so`, `triton-opt`, `triton-mlir-opt` under the producer build |
| Device assets | CANN c310 meta-op libraries, matching Ascend `libdevice.10.bc`, FP16/BF16 Gather Mix bitcode built from the FlagTree Gather source |
| Internal evidence | Captured adapter, pass dumps, compiler logs and device logs referenced in RESUME.md |

Use `repro/compile_gather_adapter.sh` to rebuild and trace a captured Gather adapter with paths supplied at run time. When moving an adapter across machines, update its CustomOp bitcode attribute to match the supplied Gather asset. Check rank-4 NZ CBUF and two i64 scalars on the final CustomOp, and verify that the MMAD consumes the same CBUF SSA value without a Gather-to-GM intermediate. Require an idle NPU and run device tests serially only after this offline check. The ordinary dot device baseline passed in R7.

The prior scripts under repro/ record historical experiments and some refer to the retired workspace. The restore script above is the cross-machine entry point. Keep binary SDKs, captured IR and raw logs in the approved internal storage area.
