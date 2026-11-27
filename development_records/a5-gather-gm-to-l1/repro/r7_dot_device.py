from pathlib import Path
import runpy
import sys

import torch
import torch_npu
import triton

torch.npu.set_device(0)
print("runtime", torch.npu.current_device(), torch.npu.get_device_name(0), flush=True)
baseline = Path(__file__).with_name("prhead_baseline.py")
sys.argv = [str(baseline), "dot"]
program = runpy.run_path(str(baseline), run_name="__main__")
triton.runtime.driver.reset_active()
a = ((torch.arange(512).reshape(16, 32) % 17 - 8) / 8).half()
b = ((torch.arange(512).reshape(32, 16) % 13 - 6) / 8).half()
expected = a.float() @ b.float()
output = torch.empty((16, 16), dtype=torch.float32, device="npu")
program["kernel"][(1, 1, 1)](a.npu(), b.npu(), output)
torch.npu.synchronize()
actual = output.cpu()
torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)
print("DOT_DEVICE_PASS max_abs_error", (actual - expected).abs().max().item(), flush=True)
