import ast
from pathlib import Path
import sys
import types

import triton
import triton.language as tl
from triton._C import libtriton
from triton.backends.ascend import compiler
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

project = Path(__file__).resolve().parents[2]
source = project / "FlagTree-v36-gather-r2"
build = project / "build-phase1/gather-r2/prhead-native"
assert Path(triton.__file__).resolve().is_relative_to(source)
assert Path(libtriton.__file__).resolve().is_relative_to(build)
target = GPUTarget("npu", "Ascend950DT_9582", 0)
triton.runtime.driver.set_active(types.SimpleNamespace(get_current_target=lambda: target))

class CompileOnlyDeviceQuery:
    def has_device_limit(self):
        return False

compiler.NPUUtils = CompileOnlyDeviceQuery

@triton.jit
def dot_baseline(a, b, out):
    rows = tl.arange(0, 16)
    k = tl.arange(0, 32)
    lhs = tl.load(a + rows[:, None] * 32 + k[None, :])
    rhs = tl.load(b + k[:, None] * 16 + rows[None, :])
    result = tl.dot(lhs, rhs)
    tl.store(out + rows[:, None] * 16 + rows[None, :], result)

def load_ub():
    path = source / "python/tutorials/tle/custom/test_custom_ops.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    nodes = [node for node in tree.body
             if (isinstance(node, ast.Import) and all(a.name not in {"numpy", "torch", "torch_npu"} for a in node.names))
             or isinstance(node, ast.ImportFrom)
             or (isinstance(node, ast.FunctionDef) and node.name == "gather_gm_to_ub_store_kernel")]
    module = types.ModuleType("r3_prhead_ub")
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), module.__dict__)
    return module.gather_gm_to_ub_store_kernel

case = sys.argv[1]
if case == "dot":
    fn = dot_baseline
    sig = {"a": "*fp16", "b": "*fp16", "out": "*fp32"}
    constants = {}
else:
    fn = load_ub()
    constants = {"NUM_ROWS": 64, "TILE_SIZE": 16, "D": 32, "DTYPE": tl.float16}
    sig = {"src": "*fp16", "src_index": "*i32", "output": "*fp16", **dict.fromkeys(constants, "constexpr")}
print("source", triton.__file__, "native", Path(libtriton.__file__).resolve(), "case", case, flush=True)
kernel = triton.compile(ASTSource(fn, sig, constexprs=constants), target=target)
binary = kernel.asm["npubin"]
print("mode", kernel.metadata.mix_mode, "ELF", binary.startswith(b"\x7fELF"), "bytes", len(binary))
assert binary.startswith(b"\x7fELF")
