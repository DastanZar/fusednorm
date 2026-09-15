"""fusednorm figures: the traffic diagram, and the training-loss GIF (the showpiece)."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch, Rectangle

from fusednorm import hbm_bytes, rmsnorm_residual_autograd  # noqa: E402

FIG = os.path.join(ROOT, "assets")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 140, "font.size": 9})


# Fig 1: traffic map — unfused (3 kernels, 6N) vs fused (1 kernel, 4N)
fig, ax = plt.subplots(figsize=(8.8, 3.6))
ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 4)


def op(ax, x, label, color):
    ax.add_patch(Rectangle((x, 2.0), 1.3, .7, facecolor=color, edgecolor="k"))
    ax.text(x + .65, 2.35, label, ha="center", va="center", fontsize=7.5)


def flow(ax, x0, y, lbl, col, up=True):
    a = FancyArrowPatch((x0, y), (x0 + 1.6, y), arrowstyle="-|>", mutation_scale=10,
                        color=col, lw=1.4)
    ax.add_patch(a)
    ax.text(x0 + .8, y + (.14 if up else -.24), lbl, ha="center", fontsize=6.4, color=col)


ax.text(0, 3.5, "UNFUSED — PyTorch: three kernels meet only in HBM", fontsize=9.5, weight="bold")
op(ax, .3, "norm", "#8ea6db"); op(ax, 3.6, "·w", "#8ea6db"); op(ax, 6.9, "+r", "#8ea6db")
flow(ax, .3, 2.0 - .5, "read x → HBM write x_norm", "#c0504d")
flow(ax, 2.4, 2.0 - .5, "read x_norm, w → write y′", "#c0504d")
flow(ax, 5.7, 2.0 - .5, "read y′, r → write y", "#c0504d")
ax.text(9.0, 1.2, "6N", color="#c0504d", fontsize=13, weight="bold")
ax.text(0, .35, "FUSED — one program; x, r, w meet in registers; y is the only thing that leaves", fontsize=9.5, weight="bold")
op(ax, .3, "rmsnorm·w+r", "#70ad47")
flow(ax, .3, 2.0 - 1.9, "read x, r, w", "#2e6b2f")
flow(ax, 2.9, 2.0 - 1.9, "write y only", "#2e6b2f")
ax.text(9.0, .1, "4N", color="#2e6b2f", fontsize=13, weight="bold")
fig.tight_layout(); fig.savefig(f"{FIG}/traffic_map.png"); plt.close(fig)

# Fig 2: the gradient flow diagram of the backward (the differentiator)
fig, ax = plt.subplots(figsize=(8.4, 2.9))
ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 2.6)
ax.add_patch(Rectangle((.2, 1.0), 1.5, .6, facecolor="#f4b942", edgecolor="k"))
ax.text(.95, 1.3, "dy", ha="center", va="center")
ax.add_patch(Rectangle((2.4, 1.0), 2.4, .6, facecolor="#8ea6db", edgecolor="k"))
ax.text(3.6, 1.3, "c = tl.sum(dy·w·x)", ha="center", va="center", fontsize=8)
ax.add_patch(Rectangle((5.5, 1.35), 2.5, .55, facecolor="#70ad47", edgecolor="k"))
ax.text(6.75, 1.62, "dx fused per row", ha="center", va="center", fontsize=8)
ax.add_patch(Rectangle((5.5, .55), 2.5, .55, facecolor="#c0504d", edgecolor="k"))
ax.text(6.75, .82, "dw partial [n,N] → Σ", ha="center", va="center", fontsize=8)
ax.add_patch(Rectangle((8.8, 1.0), 1.0, .6, facecolor="#ddd", edgecolor="k"))
ax.text(9.3, 1.3, "dr=dy", ha="center", va="center", fontsize=8)
for x0, x1 in [((1.7, 2.4)), ((4.8, 5.5))]:
    ax.add_patch(FancyArrowPatch((x0, 1.3), (x1, 1.6), arrowstyle="-|>", mutation_scale=9))
ax.add_patch(FancyArrowPatch((4.8, 1.3), (5.5, .85), arrowstyle="-|>", mutation_scale=9))
ax.add_patch(FancyArrowPatch((1.7, 1.15), (8.8, 1.15), arrowstyle="-|>", mutation_scale=9,
                             ls=(0, (4, 3)), color="#555"))
ax.set_title("Backward of the fused op — the algebra that most 'fused' layers punt to unfused torch", fontsize=9.5)
fig.tight_layout(); fig.savefig(f"{FIG}/bwd_graph.png"); plt.close(fig)

# Fig 3 (GIF): REAL training — loss curve stepping through the fused autograd
torch.manual_seed(2)
n, hidden = 24, 256
xs = torch.randn(n, hidden).half()
rs = torch.randn(n, hidden).half()
target = torch.randn(n, hidden)
w = torch.ones(hidden).half(); w.requires_grad_(True)
opt = torch.optim.SGD([w], lr=0.05)
losses = []
for it in range(26):
    opt.zero_grad()
    y = rmsnorm_residual_autograd(xs, rs, w)
    loss = torch.nn.functional.mse_loss(y.float(), target)
    loss.backward(); opt.step()
    losses.append(loss.item())
fig, ax = plt.subplots(figsize=(6.4, 3.8))
for f in range(len(losses)):
    ax.clear()
    ax.plot(range(f + 1), losses[:f + 1], "o-", color="#4472c4", lw=1.4, ms=3.5)
    ax.set_xlim(0, len(losses) - 1)
    ax.set_ylim(min(losses) * .98, max(losses) * 1.02)
    ax.set_xlabel("optimizer step"); ax.set_ylabel("MSE loss")
    ax.set_title(f"fused fwd+bwd, fp16 — step {f}, loss {losses[f]:.4f}", fontsize=9.5)
    ax.annotate("all grads flow through\n2 Triton kernels + 1 reduce",
                (f, losses[f]), textcoords="offset points", xytext=(-90, 18),
                fontsize=7.5, arrowprops=dict(arrowstyle="->", lw=.8))
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(
        fig.canvas.get_width_height()[1], -1, 4)[..., :3].copy()
    matplotlib.image.imsave(f"/tmp/fn_{f:03d}.png", buf)
plt.close(fig)
import subprocess
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "5",
                "-i", "/tmp/fn_%03d.png", "-loop", "0", f"{FIG}/training_descent.gif"],
               check=True)
print("fusednorm figures ->", sorted(os.listdir(FIG)), "| loss:", round(losses[0], 3), "->", round(losses[-1], 3))
