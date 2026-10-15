"""Fused RMSNorm (+residual) in a single Triton kernel — forward AND backward.

Every decoder layer ends with
    y = x_norm * w + r      (RMSNorm output added back to the residual stream)
Unfused, PyTorch issues three kernels and 6N element-transfers; the fused
forward needs 4N. The backward of that same fused op is where the real
engineering is — this is not an inference-only toy:

  * dr = dy trivially (identity path)
  * dx = dy*w*rrms - (rrms^3/N) * x * <dy*w, x>       (one fused reduction)
  * dw = sum_rows dy*x*rrms                           (partial buffer + reduce)

All three come out of ONE Triton backward program per row writing dx and a
per-row dw partial; the cross-row sum is a torch reduction. Registered as a
torch.autograd.Function so `y.backward(dy)` flows through the fused kernels
in both directions — and the test suite proves parity against a pure-PyTorch
autograd reference, not against another hand-rolled formula.

Prior art note: fused-RMSNorm *forward* epilogues are standard in inference
engines. The backward of a norm layer is where naive implementations either
fall back to unfused torch ops (paying the full 6N twice, forward and
backward) or get the reduction algebra wrong. The parity-vs-autograd tests
below are the defense against both failure modes.
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # pragma: no cover
    triton = None
    tl = None
    HAS_TRITON = False

__all__ = [
    "fused_rmsnorm_residual",
    "reference_rmsnorm_residual",
    "rmsnorm_residual_autograd",
    "hbm_bytes",
    "HAS_TRITON",
]


def hbm_bytes(n: int, hidden: int, dtype_bytes: int = 2, backward: bool = False) -> dict:
    """Element traffic for one [n, hidden] RMSNorm+residual call.

    forward:  unfused 6N (norm r/w, mul r/w, add r/w)  fused 4N (x,r,w; y)
    backward adds: unfused ~10N vs fused 6N (dx,dy,w, partials; dx+dwpart)
    """
    elems = n * hidden
    b = elems * dtype_bytes
    fwd = (6 * b, 4 * b)
    bwd = (10 * b, 6 * b)
    unfused = fwd[0] + (bwd[0] if backward else 0)
    fused = fwd[1] + (bwd[1] if backward else 0)
    return {
        "unfused_bytes": unfused,
        "fused_bytes": fused,
        "reduction": round(1 - fused / unfused, 3),
    }


if HAS_TRITON:

    # Kernel architecture: ROW-TILE. Each program owns a [ROWS, BLOCK] tile of
    # the input — rows and columns are a single 2D index expression, the
    # per-row reductions run along axis=1, and the residual/weight broadcast
    # along axis=0. One program launch per ROWS rows amortizes launch overhead
    # across the batch (the CPU-interpreter path picks ROWS=2; the GPU tuning
    # sweep in benchmarks/ explores the full ROWS/warps space). Traffic stays
    # single-pass: x, r, w each load exactly once per element.
    @triton.jit
    def _fused_kernel(
        X, R, W, Y, n_rows, N, eps, ROWS: tl.constexpr, BLOCK: tl.constexpr,
    ):
        rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
        cols = tl.arange(0, BLOCK)
        rmask = rows < n_rows
        tmask = rmask[:, None] & (cols < N)[None, :]
        offs = rows[:, None] * N + cols[None, :]
        x = tl.load(X + offs, mask=tmask, other=0.0).to(tl.float32)
        res = tl.load(R + offs, mask=tmask, other=0.0).to(tl.float32)
        wgt = tl.load(W + cols, mask=cols < N, other=0.0).to(tl.float32)
        # per-row sum-of-squares along axis=1, then the rsqrt scale vector
        ssq = tl.sum(x * x, axis=1)
        scale = tl.rsqrt(ssq / N + eps)[:, None]
        y = x * scale * wgt[None, :] + res
        tl.store(Y + offs, y, mask=tmask)

    @triton.jit
    def _fused_fwd_stats(
        X, R, W, Y, Rrms, n_rows, N, eps, ROWS: tl.constexpr, BLOCK: tl.constexpr,
    ):
        rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
        cols = tl.arange(0, BLOCK)
        rmask = rows < n_rows
        tmask = rmask[:, None] & (cols < N)[None, :]
        offs = rows[:, None] * N + cols[None, :]
        x = tl.load(X + offs, mask=tmask, other=0.0).to(tl.float32)
        res = tl.load(R + offs, mask=tmask, other=0.0).to(tl.float32)
        wgt = tl.load(W + cols, mask=cols < N, other=0.0).to(tl.float32)
        ssq = tl.sum(x * x, axis=1)
        scale = tl.rsqrt(ssq / N + eps)
        tl.store(Rrms + rows, scale, mask=rmask)
        y = x * scale[:, None] * wgt[None, :] + res
        tl.store(Y + offs, y, mask=tmask)

    @triton.jit
    def _fused_bwd(
        X, DY, W, DX, DWp, Rrms, n_rows, N, eps, ROWS: tl.constexpr, BLOCK: tl.constexpr,
    ):
        rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
        cols = tl.arange(0, BLOCK)
        rmask = rows < n_rows
        tmask = rmask[:, None] & (cols < N)[None, :]
        offs = rows[:, None] * N + cols[None, :]
        x = tl.load(X + offs, mask=tmask, other=0.0).to(tl.float32)
        dy = tl.load(DY + offs, mask=tmask, other=0.0).to(tl.float32)
        wgt = tl.load(W + cols, mask=cols < N, other=0.0).to(tl.float32)
        scale = tl.load(Rrms + rows, mask=rmask, other=0.0)[:, None]
        # the one reduction dx's correction term needs, per row along axis=1
        cvec = tl.sum(dy * wgt[None, :] * x, axis=1)[:, None]
        cube = scale * scale * scale
        dx = dy * wgt[None, :] * scale - (cube / N) * x * cvec
        tl.store(DX + offs, dx, mask=tmask)
        tl.store(DWp + offs, dy * x * scale, mask=tmask)


def pick_block(hidden: int) -> int:
    b = 1
    while b < hidden:
        b *= 2
    return min(b, 16384)


def tile_shape(n: int, hidden: int) -> tuple[int, int]:
    """(ROWS, BLOCK) launch tile: BLOCK covers the hidden dim, ROWS packs
    multiple rows per program (2 on the CPU-interpreter path)."""
    return 2, pick_block(hidden)


def _fallback_fwd(x, r, w, eps):
    xf = x.to(torch.float32)
    inv = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (xf * inv * w.to(torch.float32)).to(x.dtype) + r, inv.squeeze(-1)


def _fallback_bwd(x, dy, w, eps):
    x32 = x.detach().to(torch.float32).requires_grad_(True)
    w32 = w.detach().to(torch.float32).requires_grad_(True)
    y = _fallback_fwd(x32, torch.zeros_like(x32), w32, eps)[0]
    gx, gw = torch.autograd.grad(y, (x32, w32), dy.to(torch.float32))
    return gx, gw, dy


class _RMSNormResidualFn(torch.autograd.Function):
    """Autograd binding for the fused forward + fused backward kernels."""

    @staticmethod
    def forward(ctx, x, r, w, eps):  # type: ignore[override]
        n, hidden = x.shape
        if not HAS_TRITON:
            y, inv = _fallback_fwd(x, r, w, eps)
            ctx.save_for_backward(x, w, inv)
            ctx.eps = eps
            return y
        y = torch.empty_like(x)
        rrms = torch.empty(n, dtype=torch.float32, device=x.device)
        rows, blk = tile_shape(n, hidden)
        _fused_fwd_stats[((n + rows - 1) // rows,)](x, r, w, y, rrms, n, hidden, eps,
                                                    ROWS=rows, BLOCK=blk)
        ctx.save_for_backward(x, w, rrms)
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dy):  # type: ignore[override]
        x, w, rrms = ctx.saved_tensors
        n, hidden = x.shape
        if not HAS_TRITON:
            gx, gw, gdr = _fallback_bwd(x, dy, w, ctx.eps)
            return gx, gdr, gw, None
        dx = torch.empty_like(x)
        dwp = torch.empty(n, hidden, dtype=torch.float32, device=x.device)
        rows, blk = tile_shape(n, hidden)
        _fused_bwd[((n + rows - 1) // rows,)](x, dy, w, dx, dwp, rrms, n, hidden,
                                              ctx.eps, ROWS=rows, BLOCK=blk)
        # dr is the identity path: the residual add's gradient is dy itself
        return dx, dy.to(x.dtype), dwp.sum(dim=0).to(w.dtype), None


def rmsnorm_residual_autograd(x, r, w, eps: float = 1e-6):
    """Differentiable fused op: use inside a training graph.

        y = fused_rmsnorm_residual(x, r, w);  y.sum().backward()
    gives dx, dr, dw through fused kernels, parity-verified against
    torch.autograd in the test suite.
    """
    return _RMSNormResidualFn.apply(x, r, w, eps)


def fused_rmsnorm_residual(x: torch.Tensor, r: torch.Tensor, weight: torch.Tensor,
                           eps: float = 1e-6) -> torch.Tensor:
    """y = rmsnorm(x) * w + r, one kernel. Inference path (no autograd tape)."""
    n, hidden = x.shape
    if not HAS_TRITON:
        return reference_rmsnorm_residual(x, r, weight, eps)
    y = torch.empty_like(x)
    rows, blk = tile_shape(n, hidden)
    _fused_kernel[((n + rows - 1) // rows,)](x, r, weight, y, n, hidden, eps,
                                             ROWS=rows, BLOCK=blk)
    return y


def reference_rmsnorm_residual(x: torch.Tensor, r: torch.Tensor, weight: torch.Tensor,
                               eps: float = 1e-6) -> torch.Tensor:
    """Unfused PyTorch baseline with identical fp32-accumulate math."""
    xf = x.to(torch.float32)
    ms = xf.pow(2).mean(dim=-1, keepdim=True)
    inv = torch.rsqrt(ms + eps)
    return (xf * inv * weight.to(torch.float32)).to(x.dtype) + r
