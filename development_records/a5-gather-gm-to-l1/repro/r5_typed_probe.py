from pathlib import Path
import re
import runpy
import sys
from triton.backends.ascend import compiler

baseline = Path(__file__).with_name("prhead_baseline.py")
case = sys.argv[1]
original = compiler.linalg_to_bc_by_triton_mlir_opt
compiler.get_simt_stack_limit = lambda: 1152

def use_typed_adapter(linalg, metadata, opt):
    print("adapter reached", "torch_npu" in sys.modules, flush=True)
    typed = re.sub(r"(bufferization\.to_tensor[^\n]*: memref<([^>,]+)>)$", r"\1 to tensor<\2>", linalg, flags=re.M)
    return original(typed, metadata, opt)

compiler.linalg_to_bc_by_triton_mlir_opt = use_typed_adapter
sys.argv = [str(baseline), case]
runpy.run_path(str(baseline), run_name="__main__")
