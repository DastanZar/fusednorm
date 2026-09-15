"""fusednorm — fused RMSNorm + residual add."""
import os

if not os.environ.get("FUSEDNORM_DISABLE_INTERPRET"):
    import torch as _torch

    if not _torch.cuda.is_available():
        os.environ.setdefault("TRITON_INTERPRET", "1")

from .core import (
    HAS_TRITON,
    fused_rmsnorm_residual,
    hbm_bytes,
    reference_rmsnorm_residual,
    rmsnorm_residual_autograd,
)

__all__ = [
    "fused_rmsnorm_residual",
    "reference_rmsnorm_residual",
    "rmsnorm_residual_autograd",
    "hbm_bytes",
    "HAS_TRITON",
]
