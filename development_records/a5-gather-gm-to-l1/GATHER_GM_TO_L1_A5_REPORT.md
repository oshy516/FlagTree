# A5 ordinary Row Gather to L1

## Current status

A5 currently has yuansheng, not the previously deleted yuanshihan. Continued in /home/ysh-workspace/a5-dsa-template. No device test, commit, push, or PR.

PR #1767 actual diff (logs/gather-a5-pr1767-evidence.log) says CustomOp-written L1 lacks the tightly-coupled identity needed at AIC/AIV Split. Current lowering may report read before first write; legacy can expose uninitialized values. Neither is acceptable as a final execution path.

The existing v3.5 candidate in mem_ops/gather_gm_to_l1.cpp uses CANN 9.2.1 AscendC Nd2NzParams and DataCopyGM2L1ND2NZImpl, retains rank-2 FP16/BF16 GM source, rank-2 INT32 indices, rank-4 CBUF output, and both one-row and two-consecutive-row paths. It retains C310 PIPE_ALL completion synchronization. Ordinary and Mix device bitcodes in build-phase1/gather-c310 were inspected again; each defines both half and bf16 ABI entries. See logs/gather-a5-{normal,mix}-device-ir.log. DMA behavior and synchronization are still unverified on hardware.

The existing uncommitted AscendNPU-IR feature/a5-customop-l1-tcb candidate collects MTE2 hivm.custom L1 outputs with the existing allocation tracer. No new compiler changes; no demonstrated cross-core identity, allocation pairing, or tl.dot correctness. No sparse implementation used.

| Gate | FP16 | BF16 |
|---|---|---|
| c310 ordinary/Mix device bitcode definitions | confirmed | confirmed |
| v3.5 Python import | blocked: triton._C missing | blocked: triton._C missing |
| Registry/JIT, TCB/Split/PlanMemory, ELF | not run | not run |
| 950 tl.dot correctness and performance | not run | not run |

Test prepared at build-phase1/gather-c310/validation/check_row_gather_dot.py. It passed syntax checks and invokes the original v3.5 tutorial Python/JIT kernel with the current pipeline, using PyTorch reference checks for FP16/BF16, tiles 16/32, D 32/64 and consecutive/discrete/mixed indices. It has not run. The matching v3.5 native extension is absent and its 7d5de303 ARM64 archive is incomplete. Import evidence: logs/gather-a5-v35-import.log. No v3.6 binary was substituted, and no candidate was installed as an active Registry asset.

GitCode .cim from the latest request differs from the previously configured .com. The .cim account email and push permission were not confirmed; no author configuration or remote refs changed.

## R2 · v3.6 PR #1767

PR #1767 open, unmerged at inspection. Base main 7cdc4362247be87979754b5038e9f8df3961a04d; head e27584c7977d8f0818d50e8115a7308552042a0b. Actual Git branch feature/a5-gather-gm-to-l1-r2 at that head in FlagTree-v36-gather-r2, origin https://github.com/oshy516/FlagTree.git. Only gather_gm_to_l1.cpp changed, restoring the c310 ordinary-row CANN ND2NZ and completion sync. Ordinary/Mix stand-alone assets compiled cleanly from the PR Head. No registry gate change or other op change. Existing v3.5 untouched.

The PR Head has the same LLVM f6ded0be pin but different compiled C++ binding and CMake files (logs/gather-r2-pr1767-abi-comparison.log). Existing project-local v3.6 native extension matches its e8b1b75 source checkout, not the exact PR Head. That existing checkout was used only as a diagnostic. The temporary in-process Registry override used the ordinary gather Mix bitcode, the original Python tutorial kernel and disabled legacy pipeline. The FP16 C310 JIT emitted TTIR with Gather MTE2 CUBE, the intended asset path and tt.dot, then reported glibc heap corruption and exited via timeout 124. No ELF, BF16 or hardware results. See logs/gather-r2-v36-jit.log and .cache/gather-r2/*/gather_gm_to_l1_dot_kernel.ttir.

PR Head pins AscendNPU-IR efb7141581c009bbd1d676b18941ed15ede62a68. The baseline CANN 9.2.1 bishengir-opt (LLVM 19.1.7) accepted the minimal two-function MLIR and marked the CopyOp CBUF control but not the MTE2 CustomOp CBUF output. See build-phase1/gather-r2/mark-custom-l1.mlir and logs/gather-r2-stock-tcb.mlir. This proves the narrow marking gap, not downstream Split, placement or synchronization. The previous candidate applied cleanly in the separate feature/a5-gather-r2-pinned worktree of the exact pin, leaving the Estella516 clone candidate intact. Wrapper CMake for a patched compiler is blocked by the historical LLVM19 SDK's missing libLLVMDemangle.a referenced by LLVMExports.cmake (logs/gather-r2-pinned-wrapper.log). No patched binary was produced or used by JIT.

| R2 evidence | FP16 | BF16 |
|---|---|---|
| PR Head c310 normal/Mix bitcode | compiled | compiled |
| Existing v3.6 Python/JIT diagnostic | TTIR with dot; heap corruption before ELF | not run |
| Patched compiler Split/TCB/PlanMemory | not run | not run |
| 950 correctness, stability, performance | not run | not run |

No confirmed device reservation was used. Git local identities use the requested .com values. No commit, push, or PR; no write permission was reverified. The formal c310 gate and skip remain pending successful hardware verification.

## R3 · Build unblock

- PR Head checkout and Gather-only source change retained, branch feature/a5-gather-gm-to-l1-r2 at e27584c7977d8f0818d50e8115a7308552042a0b. Retired v3.5 FlagTree checkout as authorized, retained its tracked diff in logs/gather-r3-retired-v35-tracked.patch and retained other reports, Gather assets and evidence. No remote refs changed.
- PR Head native producer uses read-only f6ded0be SDK, existing venv and symlinked clean compiler/FLIR dependencies. Older e8b1b75 build artifacts are kept, but its CMake source root and compiled binding files differ. PR Head uses a distinct build tree, no global extension. TRITON_BUILD_NPUIR=OFF, optional FlagPrism disabled. Initial configuration required this explicit FlagPrism setting. Native targets triton, triton-opt and triton-mlir-opt build with 16 jobs. See logs/gather-r3-prhead-native.log/.exit.
- Both previous 19.1.7 install prefixes contained CMake LLVM exports referencing 100 static archives, all absent. No complete prebuilt SDK was located. The exact Ascend LLVM pin 06913395ef8aca66a9814723f4ccc616570b0fed was found in a local historical Git object database and exported read-only without changing that checkout. A separate AArch64 LLVM/MLIR SDK build is running at 16 jobs, no LLVM 22 substituted. Combined concurrency cap 32. See logs/gather-r3-llvm19-sdk-audit.txt and logs/gather-r3-llvm19-build.log/.exit. Dependency tree is tracked independently of .tmp/.cache; minimum 25 GB free-space guard for the SDK build.
- Exact PR Head ordinary dot and Gather UB offline probes are queued (logs/gather-r3-prhead-baseline.log/.exit and gather-r3-prhead-{dot,ub}.log). No NPU calls in these probes. SDK completion queues only the patched bishengir-opt build, minimal before/after TCB test, then bishengir-compile, with actual paths and SHA256 when produced. See logs/gather-r3-patched-bishengir.log/.exit. The existing unmodified CANN output is logs/gather-r2-stock-tcb.mlir, the patched IR is pending.
- Backend selection order is complete bundled opt/compile pair, then PATH, then TRITON_NPU_COMPILER_PATH fallback. The isolated FP16 Gather test queues only after both baseline cases succeed and the patch tools are built. It checks actual resolved paths and validates recorded SHA256 before compile. Logs: gather-r3-prhead-patched-jit.log/.exit. No claim of Split, PlanMemory, synchronization, ELF, hardware correctness, stability or performance. No device selected and no formal gate change. Local .com Git identities retained; no commit/push/PR.

### R3 native and debugger checkpoint

Exact PR Head native build returned 0. Project Python and libtriton paths verified in the baseline, with SHA256 for libtriton and both producer tools recorded in logs/gather-r3-prhead-baseline.log. Plain FP16 tl.dot reproduced glibc heap corruption independently of Gather. Used project-local GDB 12.1 from extracted Ubuntu packages, with core dumps disabled and no system installs. In logs/gather-r3-prhead-dot-gdb.log GDB stops on SIGABRT: malloc -> aligned operator new -> llvm::allocate_buffer -> mlir::AsmStateImpl -> mlir::Operation::print -> libtriton pybind wrapper. This stack locates where corruption is detected, not the memory write or Pass responsible. No ABI conclusion. The independent Gather UB probe and the eventual patched-compiler JIT are reported separately in their logs. Patched compiler cannot execute until the exact LLVM19 SDK completes. No hardware, performance, formal gate change, commit, or push.

## R4 SDK configuration recovery

LLVM19 failed in MemRefOps.cpp with no declaration matching ReinterpretCastOp::build. The pinned Ascend build-tools/build.sh passes LLVM_BSPUB_DAVINCI_BISHENGIR_A5=ON and LLVM_BSPUB_DAVINCI_BISHENGIR_A5_NPUIR=ON, missing in the initial standalone SDK configuration. Reconfiguring the existing build directory with both, no new LLVM source or build tree. Changed compiler definitions may invalidate old objects. See repro/resume-llvm19-sdk-a5.sh. The compiler candidate remains unvalidated. No new Gather ELF or hardware run.

R4 continuation: compared the eight added lines of the pushed GitCode TCB commit 4f9f2c0caa63435a197dade3f8c58309635bd4fc with the exact-pin efb7141581c0 worktree patch; identical additions, separate base histories.

After enabling the A5 flags, LLVM TableGen failed with undefined v256bf16. The parent LLVM_BSPUB_DAVINCI_BISHENGIR setting was still OFF. The pinned build-tools configuration exposes the parent feature flag along with the two A5 flags. The new resume script enables all three in the same existing SDK build directory. See evidence/llvm19-tablegen-v256bf16.txt. This is a configuration recovery, not a validated toolchain or hardware result.

## R5 · 2026-11-27 · Targeted environment diagnosis

- **LLVM 19 root cause confirmed.** The first failed command for `MLIRMemRefDialect/MemRefOps.cpp.o` reports `FoldConstantDimOfOutputShape` undeclared at line 2751. The translation unit enables the A5 caller, but `ExpandShapeUtils.h` gates the definition with `#if BSPUB_DAVINCI_BISHENGIR`. The generated `build/tools/mlir/include/mlir/Config/mlir-config.h` sets that value to 0. The actual cache enabled `LLVM_BSPUB_DAVINCI_BISHENGIR` and the A5 settings, and omitted the separate `BSPUB_DAVINCI_BISHENGIR=ON` passed by the pinned build.sh. The exported LLVM CMake, MemRef source and helper header hashes match pin `06913395ef8aca66a9814723f4ccc616570b0fed`. A command-line define alone still failed because the generated header overwrote it. Prepending an isolated include overlay that changes only the generated value to 1 passed the same TU with `-fsyntax-only` (exit 0), including the secondary PatternBenefit conversion failure. The SDK build was left unchanged. See the first-command, pin audit and config-probe evidence.
- **Dot producer boundary.** Tested with the exact PR Head project native extension at `build-phase1/gather-r2/prhead-native/wheel/triton/_C/libtriton.so`. The PR Head versus branch diff has no changes in the common producer or Gather UB code. The branch baseline therefore exercises the same checked-in compiler path; no separate pristine checkout was run. TTIR pass dumps reached the end of the recorded TTIR sequence. With GraphOptimize on and off, the dot adapter emitted MLIR and `triton-mlir-opt --emit-bytecode --emit-bytecode-version=4` rejected the `bufferization.to_tensor` syntax (expected 'to', parser diagnostic on the next line). Both processes then aborted with heap corruption, exit 134. A separate GDB run with pass dumping caught SIGABRT during MLIR printing. That frame is an allocator detection point. Neither the first invalid write nor a causal link to the parser rejection has been established.
- **Gather UB.** The traced offline run emitted TTIR and reached CANN regbase device compilation. BiSheng rejected the missing Ascend backend `lib/libdevice.10.bc` path. The process exited 1 with no GDB stack or ELF. This run used no NPU. The similarly named NVIDIA file is outside this backend and must not be substituted.
- **Toolchain.** The FlagTree producer is LLVM 22 `f6ded0be897e2878612dd903f7e8bb85448269e5`. CANN 9.2.1 ships BiShengIR 2.0.0 `efb7141581c0` built with the consumer LLVM 19.1.7 `06913395ef8a`. The two processes exchange MLIR/text or compatible bytecode and do not share an LLVM C++ ABI. Stock tool real paths, versions and SHA256 are in `evidence/gather-r5-cann-toolchain.txt`.
- **Next experiment.** Capture dot adapter output, compare exact producer tool invocations, and feed the minimal `bufferization.to_tensor` IR through the text parser/bytecode writer independently. Use a targeted memory diagnostic on the first failing in-process sequence to identify the invalid write. Obtain the matching Ascend libdevice asset and rerun only the offline Gather UB baseline with an isolated cache. Compiler TCB validation and Gather L1 hardware acceptance remain pending.

### R5 repair checkpoint

Reconfigured the existing LLVM 19 build with `-DBSPUB_DAVINCI_BISHENGIR:BOOL=ON`. CMake generated the correct header and the original MemRef object target compiled successfully. The single Ninja install job continued with two workers. The patched BiShengIR build script waits for its successful exit before building and checking TCB.
The existing Ascend 3.6.0 libdevice asset matches both setup.py's MD5 prefix and upstream Git blob `aa92cc6564bdba4635bdef1d193432dfaedf3036`. Installed into the expected Ascend backend directory. The first UB compile retry hit heap corruption and timed out (124); ELF generation remains unverified.

### R5 adapter parser isolation

Captured the original dot adapter MLIR. Matching PR Head triton-opt and triton-mlir-opt reject the old custom syntax with no result type; stock BiShengIR opt accepts it. Adding an explicit `to tensor<...>` in the two captured operations reverses the text-parser outcome. Generic operation text works with the consumer. The typed adapter also emits v4 bytecode successfully with triton-mlir-opt, and stock BiShengIR opt reads it successfully. This isolates the first parse error to the LLVM 22 producer text boundary. The standalone parse exits 1; the Python capture reproduces a separate GC-time SIGSEGV after the compilation error (139). The first illegal memory access remains unknown. The typed text edit is a diagnostic only.

The isolated typed-adapter probe reached the normal FlagTree compiler and generated a 25,672-byte Mix ELF for dot (exit 0, no NPU launch). Torch NPU remained loaded. The same probe for Gather UB encountered heap corruption before its adapter callback. The syntax workaround therefore establishes a path to a dot ELF; it does not establish the root cause of the in-process memory issue.

A bounded GDB run of the Gather UB typed probe caught SIGABRT inside glibc free during MLIR DataFlowSolver analysis-state destruction in `mlir::triton::runUseAnalysis`, called by `TritonToLinalgPass::runOnOperation`. This precedes the producer adapter parser callback and narrows the next isolation step to the in-process pass. The free is a detection site; instrument the corresponding analysis and capture pass input to locate the original write.

## R6 · structured adapter serialization and compiler build

The Ascend native binding now emits generic adapter assembly with explicit operand/result types, and the normal bytecode stage consumes that assembly. The existing native build incrementally compiled the binding and relinked libtriton. The unmodified dot regression kernel generated a 25,672-byte Mix ELF; the Python process then segfaulted during garbage collection (139). This fixes the observed text-parser error, and full dot process/runtime acceptance is pending.

The original LLVM 19 install succeeded. An installed-SDK-only BiShengIR configuration lacks required MLIR build-graph targets. The pinned external-project configuration reuses the LLVM build graph, enables built-in Triton and the upstream MLIR test targets required by hivmc, and starts with 16 jobs and one LLVM link job. The paired compiler build and TCB/Split/PlanMemory/sync proofs remain pending.

The captured input to Gather UB TritonToLinalg was replayed through the matching CLI pass with per-pass verification and exited 0. This differs from the in-process DataFlowSolver cleanup abort. Further state-lifetime diagnostics are required before changing UseAnalysis.

## R7 · A5 device validation
Physical NPU 2 was idle and mapped to runtime device 0 (Ascend950DT_9582). The existing 25,672-byte Mix dot ELF launched through its compiled kernel metadata and FlagTree driver on that device. FP16 inputs / FP32 output matched the PyTorch reference exactly (max absolute error 0.0), synchronized successfully, and exited 0. This is a device pass for the dot baseline.

The existing patched compiler build stopped on GCC's explicit-template-specialization namespace rule in Matmul.cpp. The build-pin worktree now reopens the anonymous namespace around the specialization. The incremental 16-job build passed the failed object and continued. The exact build-pin patch is saved separately from the TCB candidate.

The stock-compiler diagnostic admitted the existing Gather L1 Mix bitcode locally and entered the first FP16 consecutive test on NPU 2. Python reported heap corruption during JIT and timed out (124) before a Gather ELF or kernel launch was observed. Device permission and dot runtime had already passed. The patched-compiler replay is queued behind the successful incremental build and rechecks that NPU 2 is idle.

## R8 · GCC compatibility

ArithToHIVMAVE.cpp now uses four member partial specializations with a defaulted second template argument. The APInt/APFloat function bodies and call sites are unchanged. The original failed Ninja object compiled with GCC and returned 0 (logs/gather-r8-arith-object.log and .exit). Both compiler targets resumed in the existing LLVM 19 build with one 16-job Ninja invocation (logs/gather-r8-build.log and .exit).

## R9 · Static links and first patched Gather compile

Both host tools linked successfully with Ninja -j16 after making AdapterSanitizer an explicit A3/A5 static dependency and instantiating collapseAndReplace for CollapseShapeOp. Tool SHA256 values are in logs/gather-r9-tools.sha256. The minimal TCB check marked the CustomOp CBUF allocation and Split retained TCB ID 0 in both AIC and AIV, with Gather on AIC. Running PlanMemory directly on this fixture aborted because the Copy control still had a tensor operand. The first patched Gather JIT reached device linking, which failed with missing built-in c310 meta-op symbols. The existing CANN 9.2.1 c310 assets were linked into the isolated compiler build lib directory for the follow-up.

## R10 · Gather pipeline and A5 fault

With the c310 meta-op assets staged, the captured FP16 16x32 Gather compiled offline through TCB, Split and PlanMemoryRegBase and produced a 44,576-byte Mix ELF. In the final AIC IR the CustomOp receives a rank-2 GM workspace view as its destination, followed by a separate ND2NZ into CBUF. The linked implementation expects a rank-4 CBUF destination. This is a visible ABI mismatch and a candidate cause of the runtime fault.

The ordinary Gather JIT retry exited 139 in make_ttir. A diagnostic replay using the captured adapter with the normal device compiler and launcher launched the FP16 consecutive case on idle NPU 2. Synchronization reported 507015 with AIC 263/82 (unaligned or out-of-bounds address), before output comparison. Ordinary dot was not repeated.

## R11 · Direct Gather CBUF ABI

The first InsertLoadStoreForMixCV maps PIPE_MTE2 output to GM and inserts the store. The layout combination then materializes ND2NZ before MMAD. The A5 fix matches the known Gather symbols with one unpadded ND2NZ consumer before Mix propagation. It uses the consumer's static NZ shape for an explicit CBUF allocation, widens the two scalar operands to i64, and keeps the same buffer for MMAD. A second Mix pass now preserves the CBUF output. FP16 and a BF16 variant of the captured adapter both linked offline with the AIC asset. Final IR writes and reads the same rank-4 CBUF, and the only remaining ND2NZ converts the separate query operand. The device wrapper uses two rank-2 GM descriptors, two i64 scalars and one rank-4 CBUF descriptor. The FP16 destination has offset 0, sizes [2,1,16,16] and strides [256,256,16,1]. Final device code selects AIC-only for this kernel, so the normal AIC Gather bitcode is required for the replay.
