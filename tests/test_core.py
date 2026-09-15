import sys

import torch

sys.path.insert(0, ".")
from fusednorm import HAS_TRITON, fused_rmsnorm_residual, hbm_bytes, reference_rmsnorm_residual  # noqa: E402

torch.manual_seed(0)


def test_kernel_parity():
    if not HAS_TRITON:
        import pytest

        pytest.skip("triton unavailable")
    x = (torch.randn(64, 512) * 0.3).to(torch.float16)
    r = (torch.randn(64, 512) * 0.3).to(torch.float16)
    w = (torch.randn(512) * 0.2 + 1.0).to(torch.float16)
    out = fused_rmsnorm_residual(x, r, w)
    ref = reference_rmsnorm_residual(x, r, w)
    assert torch.allclose(out, ref, atol=2e-2, rtol=2e-2), (out - ref).abs().max()


def test_hbm_bytes_reduction():
    b = hbm_bytes(1024, 4096)
    assert b["unfused_bytes"] == 6 * 1024 * 4096 * 2
    assert b["fused_bytes"] == 4 * 1024 * 4096 * 2
    assert abs(b["reduction"] - 1 / 3) < 1e-3


def test_latency_model_matches_bytes_ratio():
    """In the memory-bound regime, latency ratio ~= bytes ratio (33% cut)."""
    b = hbm_bytes(512, 2048)
    t_unfused = b["unfused_bytes"] / 1e9 * 4.0   # 4 ms/GB at 250GB/s-class hw
    t_fused = b["fused_bytes"] / 1e9 * 4.0
    assert abs((t_unfused - t_fused) / t_unfused - 1 / 3) < 1e-3


def test_matches_torch_functional_form():
    """Fused output == torch-native composition (fp32 cast then add residual)."""
    x = (torch.randn(32, 256) * 0.2).to(torch.float16)
    r = (torch.randn(32, 256) * 0.2).to(torch.float16)
    w = torch.ones(256, dtype=torch.float16)
    out = fused_rmsnorm_residual(x, r, w)
    xf = x.to(torch.float32)
    ref = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(torch.float16) + r
    assert torch.allclose(out.to(torch.float32), ref.to(torch.float32), atol=2e-2)
