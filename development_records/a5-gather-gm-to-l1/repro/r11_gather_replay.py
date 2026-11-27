import ast
import os
import re
import torch
import torch_npu
from pathlib import Path
import sys
import types

import triton
from triton._C import libtriton
from triton.backends.compiler import GPUTarget
from triton.backends.ascend import compiler
from triton.compiler import ASTSource
from triton.experimental.tle.language.dsa.ascend.custom_ops import registry
from triton.language.extra.cann.extension import custom_op

src = Path(os.environ["R11_FLAGTREE"]).resolve()
assert Path(triton.__file__).resolve().is_relative_to(src)
assert Path(libtriton.__file__).resolve().is_relative_to(Path(os.environ["TRITON_BUILD_DIR"]).resolve())
from triton.backends.ascend.utils import _get_bishengir_opt_path, _get_npucompiler_path
tool_root = Path(os.environ["R7_BISHENGIR_ROOT"]).resolve()
for pick in (_get_bishengir_opt_path, _get_npucompiler_path):
    path, _env = pick()
    resolved = Path(path).resolve()
    assert resolved.is_relative_to(tool_root), resolved
    print("JIT compiler tool", resolved, flush=True)

producer_only = os.environ.get("R11_PRODUCER_ONLY") == "1"
if producer_only:
    target = GPUTarget("npu", "Ascend950DT_9582", 0)
else:
    assert torch.npu.device_count() == 1
    torch.npu.set_device(0)
    target = triton.runtime.driver.active.get_current_target()
assert target.arch.startswith("Ascend950"), target
triton.runtime.driver.set_active(types.SimpleNamespace(get_current_target=lambda: target))
class CompileOnlyDeviceQuery:
    def has_device_limit(self):
        return False
compiler.NPUUtils = CompileOnlyDeviceQuery
op = custom_op._custom_op_registry["gather_gm_to_l1"]
original_init = op.__init__
bundle = Path(os.environ["R11_GATHER_BC"]).resolve()
def diagnostic_init(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    self.bitcode = str(bundle)
op.__init__ = diagnostic_init
op.supported_arches = ("c220", "c310")
print("diagnostic-only gate override", triton.__file__, libtriton.__file__, bundle, flush=True)

tutorial = src / "python/tutorials/tle/custom/test_custom_ops.py"
tree = ast.parse(tutorial.read_text(), filename=str(tutorial))
nodes = [node for node in tree.body
         if (isinstance(node, ast.Import) and all(a.name not in {"numpy", "torch", "torch_npu"} for a in node.names))
         or isinstance(node, ast.ImportFrom)
         or (isinstance(node, ast.FunctionDef) and node.name == "gather_gm_to_l1_dot_kernel")]
mod = types.ModuleType("r2_gather_tutorial")
mod.__file__ = str(tutorial)
sys.modules[mod.__name__] = mod
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(tutorial), "exec"), mod.__dict__)
fn = mod.gather_gm_to_l1_dot_kernel
if producer_only:
    class TTIRReached(Exception):
        pass

    def stop_after_ttir(src, metadata, options):
        raise TTIRReached

    compiler.ttir_to_linalg = stop_after_ttir
    constants = {"NUM_ROWS": 64, "TILE_SIZE": 16, "D": 32,
                 "DTYPE": triton.language.float16}
    signature = {"src": "*fp16", "src_index": "*i32", "query": "*fp16",
                 "output": "*fp32", **dict.fromkeys(constants, "constexpr")}
    try:
        triton.compile(ASTSource(fn, signature, constexprs=constants), target=target,
                       options={"enable_legacy_insert_load_store_for_mix_cv": False})
    except TTIRReached:
        print("PRODUCER_TTIR_PASS", flush=True)
        sys.exit(0)
    raise RuntimeError("Producer probe did not reach the TTIR boundary")

triton.runtime.driver.reset_active()

dtype_name = os.environ["R11_DTYPE"]
adapter = Path(os.environ["R11_GENERIC_ADAPTER"])
captured_ir = adapter.read_text()
assert '"hivm.hir.custom"' in captured_ir and '"bufferization.to_tensor"' in captured_ir
captured_ir, count = re.subn(r'bitcode = "[^"]+"',
                             lambda match: f'bitcode = "{bundle}"', captured_ir)
assert count == 1
if dtype_name == "bf16":
    captured_ir = captured_ir.replace("f16", "bf16").replace(
        "custom_gather_gm_to_l1_half", "custom_gather_gm_to_l1_bf16")
assert 'mix_mode = "mix"' in captured_ir
captured_ir = captured_ir.replace('mix_mode = "mix"', 'mix_mode = "aic"')
ASTSource.make_ir = lambda *args, **kwargs: captured_ir
compiler.make_ttir = lambda src, metadata, options: src
assert "hacc.program_grid_transforms" not in captured_ir and "hacc.coalesce_" not in captured_ir
def replay_adapter(src, metadata, options):
    metadata.update(program_grid_transforms=None, program_grid_mapping_applied=False,
                    row_coalescing_applied=False, has_auto_blockify_blacklist_op=False)
    return captured_ir
compiler.ttir_to_linalg = replay_adapter
print("Captured FP16 16x32 adapter replay; standard bytecode, device compiler and launcher", flush=True)
dtypes = {"fp16": (torch.float16, triton.language.float16),
          "bf16": (torch.bfloat16, triton.language.bfloat16)}
for dtype, tl_dtype in [dtypes[dtype_name]]:
    for tile, cols in [(16, 32)]:
        rows = tile * 4
        src_cpu = ((torch.arange(rows * cols).reshape(rows, cols) % 31 - 15) / 16).to(dtype)
        query_cpu = ((torch.arange(tile * cols).reshape(tile, cols) % 13 - 6) / 8).to(dtype)
        src = src_cpu.npu()
        query = query_cpu.npu()
        for pattern, ids in [("consecutive", torch.arange(tile)+1),
                             ("discrete", torch.arange(tile)*3),
                             ("mixed", torch.arange(tile)+2*(torch.arange(tile)//2))]:
            index = ids.int().reshape(tile, 1).npu()
            result = torch.full((tile, tile), float("nan"), dtype=torch.float32, device="npu")
            expected = query_cpu.float() @ src_cpu[ids].float().T
            print("GATHER_CASE", dtype, tile, cols, pattern, flush=True)
            for _ in range(3):
                result.fill_(float("nan"))
                kernel = fn[(1,)](src, index, query, result, NUM_ROWS=rows, TILE_SIZE=tile, D=cols,
                                 DTYPE=tl_dtype, enable_legacy_insert_load_store_for_mix_cv=False)
                torch.npu.synchronize()
                actual = result.cpu()
                torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
            assert kernel.asm["npubin"].startswith(b"\x7fELF")
            print("GATHER_DEVICE_PASS", True,
                  (actual-expected).abs().max().item(), flush=True)
