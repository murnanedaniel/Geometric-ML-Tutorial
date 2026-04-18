"""02i — Sinkhorn: algorithm step by step (row/col normalisation)."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from .helpers import (twins_dataset, sinkhorn, setup_canvas, save_animation,
                      DEFAULT_FPS, FG, MUTE, GRID, GREY, A_COLOR, B_COLOR)

PLAN_CMAP = LinearSegmentedColormap.from_list(
    "plan", ["#FAFAF7", "#D4E2F0", "#6EA8D9", "#2260A3", "#0B2545"])


def render(out_path: Path, fps: int = DEFAULT_FPS):
    A, B, gt = twins_dataset()
    N = len(A)
    D = np.linalg.norm(A[:, None] - B[None, :], axis=-1)
    eps = 0.25

    # Run Sinkhorn and capture all iteration plans.
    _, plans = sinkhorn(D, eps=eps, n_iters=60)

    fig = plt.figure(figsize=(12.0, 7.6), dpi=100)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.8], wspace=0.12,
                          left=0.04, right=0.97, top=0.84, bottom=0.10)
    ax_p = fig.add_subplot(gs[0])   # transport plan heatmap
    ax_m = fig.add_subplot(gs[1])   # marginals bar chart

    # Plan heatmap.
    ax_p.set_xticks([]); ax_p.set_yticks([])
    for s in ("left", "right", "top", "bottom"):
        ax_p.spines[s].set_visible(False)
    ax_p.set_title("transport plan $P$", fontsize=11, color=MUTE,
                   pad=6, loc="left")
    img = ax_p.imshow(np.zeros((N, N)), cmap=PLAN_CMAP, vmin=0, vmax=0.1,
                      origin="lower", aspect="equal")
    iter_label = ax_p.text(0.98, 0.96, "", transform=ax_p.transAxes,
                           ha="right", va="top", fontsize=13, color=FG,
                           fontweight="bold",
                           bbox=dict(boxstyle="round,pad=0.3",
                                     fc="white", ec=GRID, lw=0.9))

    # Marginals.
    ax_m.set_xlim(-0.5, N - 0.5)
    ax_m.set_ylim(0, 0.15)
    ax_m.set_xticks([]); ax_m.tick_params(labelsize=9)
    for s in ("top", "right"): ax_m.spines[s].set_visible(False)
    ax_m.set_title("row sums (should \u2192 1/N)", fontsize=11,
                   color=MUTE, pad=6, loc="left")
    target = 1.0 / N
    ax_m.axhline(target, color="#2E8B57", ls="--", lw=1.2, alpha=0.7)
    bars = ax_m.bar(np.arange(N), np.zeros(N), color=B_COLOR,
                    edgecolor="white", lw=0.6)

    fig.text(0.5, 0.96, "Sinkhorn iterations — step by step",
             ha="center", va="top", fontsize=15, fontweight="bold")
    sub = fig.text(0.5, 0.92, "", ha="center", va="top",
                   fontsize=11, color=MUTE)
    cap = fig.text(0.5, 0.025, "", ha="center", va="bottom",
                   fontsize=11, color=FG)

    PER_ITER = 6     # frames per Sinkhorn iteration
    HOLD = 50
    n_iters_show = len(plans)
    TOTAL = n_iters_show * PER_ITER + HOLD

    def update(f):
        if f < n_iters_show * PER_ITER:
            it = f // PER_ITER
        else:
            it = n_iters_show - 1
        P = plans[it]
        img.set_data(P)
        img.set_clim(0, max(P.max() * 0.85, 1e-6))
        iter_label.set_text(f"iter {it + 1}")

        row_sums = P.sum(axis=1)
        for j, b in enumerate(bars):
            b.set_height(row_sums[j])
        ax_m.set_ylim(0, max(0.12, row_sums.max() * 1.2))

        dev = np.abs(row_sums - target).mean()
        phase = "row norm" if it % 2 == 0 else "col norm"
        if f < n_iters_show * PER_ITER:
            sub.set_text(f"iteration {it+1}: {phase}")
            cap.set_text(f"marginal deviation = {dev:.4f}")
        else:
            sub.set_text("converged \u2014 plan is doubly stochastic")
            cap.set_text(f"\u03b5 = {eps}   \u00b7   "
                         f"{n_iters_show} iterations")

    anim = animation.FuncAnimation(fig, update, frames=TOTAL,
                                   interval=1000/fps, blit=False)
    save_animation(anim, out_path, fps=fps)
    plt.close(fig)
    print(f"wrote {out_path}  ({TOTAL} frames, {fps} fps)")
