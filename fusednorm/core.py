"""Fused RMSNorm (+residual) in a single Triton kernel.

Every decoder layer ends with
    y = x_norm * w + r      (RMSNorm output added back to the residual stream)
Unfused, PyTorch issues three kernels and 5 HBM passes:

    unfused:  read x (norm), write x_norm, read x_norm, read w, write y,
              read r, write y  -> ~ (2N + 2N + 3N) element traffic
Fused, the kernel reads x, r, w once, keeps everything in registers,
and writes y once: 3N reads + N write. At the arithmetic intensity of a
norm (<< 1 FLOP/byte) runtime IS memory traffic, so the pass count is the
whole story — see hbm_bytes() below for the exact accounting.

Prior art: fused RMSNorm/layernorm epilogues are standard in inference
engines; this repo exists to make the *why* measurable: byte counters,
autotune configs, and parity tests at fp32-accumulate strictness.
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

__all__ = ["fused_rmsnorm_residual", "reference_rmsnorm_residual", "hbm_bytes", "HAS_TRITON"]

_CFGS = [
    triton.Config({"BLOCK": 1024}, num_warps=4) if HAS_TRITON else None,
    triton.Config({"BLOCK": 2048}, num_warps=8) if HAS_TRITON else None,
    triton.Config({"BLOCK": 4096}, num_warps=8) if HAS_TRITON else None,
]


def hbm_bytes(n: int, hidden: int, dtype_bytes: int = 2) -> dict:
    """Element traffic for one [n, hidden] RMSNorm+residual, unfused vs fused."""
    elems = n * hidden
    b = elems * dtype_bytes
    unfused = 6 * b  # norm r/w + mul r/w + add r/w (conservative lower bound)
    fused = 4 * b    # read x, r, w-row; write y
    return {
        "unfused_bytes": unfused,
        "fused_bytes": fused,
        "reduction": round(1 - fused / unfused, 3),
    }


if HAS_TRITON:

    # NOTE: configs below are the GPU tuning space (BLOCK, num_warps). On this
    # CPU-interpreter path we select BLOCK directly; run benchmarks/bench.py on
    # CUDA to sweep the full space.
    @triton.jit
    def _fused_kernel(
        X, R, W, Y, N, eps, BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < N
        base = row * N
        x = tl.load(X + base + cols, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(R + base + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        ms = tl.sum(x * x, axis=0) / N
        inv = tl.rsqrt(ms + eps)
        y = x * inv * w + r
        tl.store(Y + base + cols, y, mask=mask)


def pick_block(hidden: int) -> int:
    """Smallest power-of-2 >= hidden, capped at 16K (one row per program)."""
    b = 1
    while b < hidden:
        b *= 2
    return min(b, 16384)


def fused_rmsnorm_residual(x: torch.Tensor, r: torch.Tensor, weight: torch.Tensor,
                           eps: float = 1e-6) -> torch.Tensor:
    """y = rmsnorm(x) * w + r, one kernel. x/r: [n, hidden] fp16; w: [hidden]."""
    n, hidden = x.shape
    if not HAS_TRITON:
        return reference_rmsnorm_residual(x, r, weight, eps)
    y = torch.empty_like(x)
    _fused_kernel[(n,)](x, r, weight, y, hidden, eps, BLOCK=pick_block(hidden))
    return y


def reference_rmsnorm_residual(x: torch.Tensor, r: torch.Tensor, weight: torch.Tensor,
                               eps: float = 1e-6) -> torch.Tensor:
    """Unfused PyTorch baseline with identical fp32-accumulate math."""
    xf = x.to(torch.float32)
    ms = xf.pow(2).mean(dim=-1, keepdim=True)
    inv = torch.rsqrt(ms + eps)
    return (xf * inv * weight.to(torch.float32)).to(x.dtype) + r
