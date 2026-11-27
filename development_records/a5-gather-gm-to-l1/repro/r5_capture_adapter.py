from pathlib import Path
import runpy
import sys
from triton.backends.ascend import compiler

capture = Path(sys.argv[1])
baseline = Path(__file__).with_name("prhead_baseline.py")
case = sys.argv[2]
original = compiler.linalg_to_bc_by_triton_mlir_opt

def capture_adapter(linalg, metadata, opt):
    capture.write_text(linalg)
    return original(linalg, metadata, opt)

compiler.linalg_to_bc_by_triton_mlir_opt = capture_adapter
sys.argv = [str(baseline), case]
runpy.run_path(str(baseline), run_name="__main__")
