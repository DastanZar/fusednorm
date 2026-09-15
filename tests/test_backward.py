"""Backward-pass tests: parity against PyTorch AUTOGRAD (the only oracle
that cannot lie, because it was written by neither of us)."""
import sys

import pytest
import torch

sys.path.insert(0, ".")
from fusednorm import HAS_TRITON, hbm_bytes, rmsnorm_residual_autograd  # noqa: E402


def _torch_ref(x, r, w, eps=1e-6):
    x = x.clone().requires_grad_(True)
    r = r.clone().requires_grad_(True)
    w = w.clone().requires_grad_(True)
    xf = x.float()
    inv = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    y = (xf * inv * w.float()).to(x.dtype) + r
    return y, (x, r, w)


def _grads(y, refs, dy):
    return torch.autograd.grad(y, refs, dy, retain_graph=True)


def _relerr(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


@pytest.mark.parametrize("hidden", [256, 512, 1024])
def test_backward_parity_all_grads(hidden):
    if not HAS_TRITON:
        pytest.skip("triton unavailable")
    torch.manual_seed(4)
    n = 64
    x = (torch.randn(n, hidden) ).half()
    r = (torch.randn(n, hidden) * 0.5).half()
    w = (torch.randn(hidden) * 0.2 + 1.0).half()
    dy = (torch.randn(n, hidden) * 0.3).half()

    # reference grads
    yref, refs = _torch_ref(x, r, w)
    gx, gr, gw = _grads(yref, refs, dy)

    xa, ra, wa = [t.clone().requires_grad_(True) for t in (x, r, w)]
    y = rmsnorm_residual_autograd(xa, ra, wa)
    assert torch.allclose(y.float(), yref.float(), atol=2e-2, rtol=2e-2)
    y.backward(dy)
    assert xa.grad is not None and ra.grad is not None and wa.grad is not None
    # fp16 storage -> 2-3e-2 rel tolerance on each grad path separately
    assert _relerr(xa.grad, gx) < 0.03, f"dx relerr {_relerr(xa.grad, gx)}"
    assert _relerr(ra.grad, gr) < 1e-6, "dr must be bit-identical to dy (identity path)"
    assert _relerr(wa.grad, gw) < 0.03, f"dw relerr {_relerr(wa.grad, gw)}"


@pytest.mark.skipif(not HAS_TRITON, reason="triton unavailable")
def test_backward_matches_chain_rule_symbolically():
    """For w=1, r=0: dx must equal the analytic RMSNorm jacobian-vector product.

    y = x*s, s=(mean(x^2)+eps)^-1/2
    dx_j = dy_j*s - s^3 * x_j * (dy·x)/N
    """
    torch.manual_seed(5)
    n, hidden = 8, 128
    x = torch.randn(n, hidden)
    w = torch.ones(hidden)
    r = torch.zeros(n, hidden)
    dy = torch.randn(n, hidden)
    xf = x.double()
    ms = xf.pow(2).mean(-1, keepdim=True)
    inv = torch.rsqrt(ms + 1e-6)
    dot = (dy.double() * xf).sum(-1, keepdim=True)
    dx_analytic = dy.double() * inv - (inv.pow(3) / hidden) * xf * dot
    xa = x.clone().requires_grad_(True)
    y = rmsnorm_residual_autograd(xa, r, w)
    y.backward(dy)
    assert _relerr(xa.grad.double(), dx_analytic) < 2e-5, _relerr(xa.grad.double(), dx_analytic)


def test_training_step_end_to_end():
    """Real optimizer steps through the fused autograd reduce loss monotonically,
    tracking the pure-torch fp32 trajectory within fp16 noise."""
    torch.manual_seed(6)
    n, hidden = 16, 256
    x = torch.randn(n, hidden).half()
    r = torch.randn(n, hidden).half()
    w = torch.ones(hidden).half()
    w.requires_grad_(True)  # after dtype, or the param isn't a leaf
    target = torch.randn(n, hidden)

    w32 = w.detach().float().requires_grad_(True)
    opt, opt32 = torch.optim.SGD([w], lr=0.05), torch.optim.SGD([w32], lr=0.05)
    fused_l, ref_l = [], []
    for _ in range(20):
        opt.zero_grad()
        y = rmsnorm_residual_autograd(x, r, w)
        loss = torch.nn.functional.mse_loss(y.float(), target)
        loss.backward()
        opt.step()
        fused_l.append(loss.item())

        opt32.zero_grad()
        xf = x.detach().float()
        y32 = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6) * w32
               + r.detach().float())
        l32 = torch.nn.functional.mse_loss(y32, target)
        l32.backward()
        opt32.step()
        ref_l.append(l32.item())

    assert all(b < a for a, b in zip(fused_l, fused_l[1:])), fused_l  # strict descent
    assert fused_l[-1] < fused_l[0]
    assert abs(fused_l[-1] - ref_l[-1]) / ref_l[-1] < 0.02, (fused_l[-1], ref_l[-1])


def test_hbm_bytes_backward_mode():
    b = hbm_bytes(1024, 4096, backward=True)
    # fwd 6->4 and bwd 10->6: unfused 16N, fused 10N -> 37.5% cut
    assert b["unfused_bytes"] == 16 * 1024 * 4096 * 2
    assert b["fused_bytes"] == 10 * 1024 * 4096 * 2
    assert abs(b["reduction"] - 0.375) < 1e-3
