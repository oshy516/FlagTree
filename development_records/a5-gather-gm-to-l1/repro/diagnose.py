import ast
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

project = Path(__file__).resolve().parents[2]
src = project / "FlagTree-v36-diagnostic"
assert Path(triton.__file__).resolve().is_relative_to(src)
assert Path(libtriton.__file__).resolve().is_relative_to(project / "build-phase1/v36-native")
target = GPUTarget("npu", "Ascend950DT_9582", 0)
triton.runtime.driver.set_active(types.SimpleNamespace(get_current_target=lambda: target))
class CompileOnlyDeviceQuery:
    def has_device_limit(self):
        return False
compiler.NPUUtils = CompileOnlyDeviceQuery
op = custom_op._custom_op_registry["gather_gm_to_l1"]
original_init = op.__init__
bundle = project / "build-phase1/gather-c310/gather-mix.bc"
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
constants = {"NUM_ROWS": 64, "TILE_SIZE": 16, "D": 32, "DTYPE": triton.language.float16}
sig = {"src": "*fp16", "src_index": "*i32", "query": "*fp16", "output": "*fp32",
       **dict.fromkeys(constants, "constexpr")}
out = triton.compile(ASTSource(fn, sig, constexprs=constants), target=target,
                     options={"enable_legacy_insert_load_store_for_mix_cv": False})
print("mode", out.metadata.mix_mode, "ELF", out.asm["npubin"][:4] == b"\x7fELF", flush=True)
