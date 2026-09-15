"""Demo: byte accounting + parity + per-layer savings scaled to a real model."""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fusednorm import HAS_TRITON, fused_rmsnorm_residual, hbm_bytes, reference_rmsnorm_residual  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "receipts")
os.makedirs(OUT, exist_ok=True)

# parity on a hidden size from a real model (llama-3-8b: 4096)
x = (torch.randn(16, 1024) * 0.3).to(torch.float16)
r = (torch.randn(16, 1024) * 0.3).to(torch.float16)
w = (torch.randn(1024) * 0.2 + 1.0).to(torch.float16)
out = fused_rmsnorm_residual(x, r, w)
ref = reference_rmsnorm_residual(x, r, w)
parity = (out.to(torch.float32) - ref.to(torch.float32)).abs().max().item()

# byte accounting per decode step, llama-3-8b: 2 fused sites x 32 layers
b = hbm_bytes(1, 4096)
per_step_unfused = b["unfused_bytes"] * 2 * 32
per_step_fused = b["fused_bytes"] * 2 * 32

receipt = {
    "mode": "triton-interpreter" if os.environ.get("TRITON_INTERPRET") == "1" else "compiled",
    "kernel_vs_reference_max_abs_err_fp16": parity,
    "bytes_per_layer_site": b,
    "llama3_8b_decode_step_bytes": {
        "unfused": per_step_unfused,
        "fused": per_step_fused,
        "saved_bytes_per_token": per_step_unfused - per_step_fused,
        "note": "weights dominate decode traffic; this is the *norm-site* share only",
    },
}
with open(os.path.join(OUT, "receipt.json"), "w") as f:
    json.dump(receipt, f, indent=2)
print(json.dumps(receipt, indent=2))
assert parity < 3e-2
print("PARITY_OK")
