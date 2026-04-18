"""02j — Sinkhorn: epsilon sweep (blurry \u2192 sharp)."""

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

    eps_values = np.concatenate([
        np.linspace(3.0, 0.08, 80),
        np.full(25, 0.08),
    ])
    plans = []
    for eps in eps_values:
        P, _ = sinkhorn(D, eps=eps, n_iters=80)
        plans.append(P)

    fig = plt.figure(figsize=(12.0, 7.6), dpi=100)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.10,
                          left=0.04, right=0.97, top=0.84, bottom=0.10)
    ax = fig.add_subplot(gs[0])
    ax_p = fig.add_subplot(gs[1])
    setup_canvas(ax)

    ax.scatter(A[:, 0], A[:, 1], s=130, c=[GREY]*N, marker="o",
               edgecolors="white", linewidths=1.4, zorder=4)
    ax.scatter(B[:, 0], B[:, 1], s=130, c=[GREY]*N, marker="D",
               edgecolors="white", linewidths=1.4, zorder=4)

    MAX_LINES = 50
    flow_lines = []
    for _ in range(MAX_LINES):
        ln, = ax.plot([], [], color="#363D45", lw=1.5, alpha=0.0, zorder=3)
        flow_lines.append(ln)

    ax_p.set_xticks([]); ax_p.set_yticks([])
    for s in ("left", "right", "top", "bottom"):
        ax_p.spines[s].set_visible(False)
    ax_p.set_title("transport plan $P$", fontsize=11, color=MUTE,
                   pad=6, loc="left")
    img = ax_p.imshow(np.zeros((N, N)), cmap=PLAN_CMAP, vmin=0, vmax=0.1,
                      origin="lower", aspect="equal")
    eps_label = ax_p.text(0.98, 0.96, "", transform=ax_p.transAxes,
                          ha="right", va="top", fontsize=13, color=FG,
                          fontweight="bold",
                          bbox=dict(boxstyle="round,pad=0.3",
                                    fc="white", ec=GRID, lw=0.9))

    fig.text(0.5, 0.96,
             "Sinkhorn: \u03b5 sweep (blurry \u2192 sharp)",
             ha="center", va="top", fontsize=15, fontweight="bold")
    sub = fig.text(0.5, 0.92, "", ha="center", va="top",
                   fontsize=11, color=MUTE)
    cap = fig.text(0.5, 0.025, "", ha="center", va="bottom",
                   fontsize=11, color=FG)

    HOLD = 30
    TOTAL = len(eps_values) + HOLD

    def draw_plan(k):
        P = plans[k]
        eps = eps_values[k]
        img.set_data(P)
        img.set_clim(0, max(P.max() * 0.8, 1e-6))
        eps_label.set_text(rf"$\varepsilon = {eps:.2f}$")
        # Flow lines.
        flat = np.argsort(P.ravel())[::-1][:MAX_LINES]
        for li, fi in enumerate(flat):
            i, j = divmod(int(fi), N)
            flow_lines[li].set_data([A[i, 0], B[j, 0]],
                                    [A[i, 1], B[j, 1]])
            mass = float(P[i, j])
            flow_lines[li].set_alpha(min(1.0, mass * N * 3.0))
            flow_lines[li].set_linewidth(max(0.5, min(3.5, mass * N * 8)))

    def update(f):
        k = min(f, len(eps_values) - 1)
        draw_plan(k)
        eps = eps_values[k]
        if eps > 1.0:
            sub.set_text("large \u03b5 \u2192 blurry (nearly uniform)")
        elif eps > 0.3:
            sub.set_text("medium \u03b5 \u2192 structure emerging")
        else:
            sub.set_text("small \u03b5 \u2192 sharp, permutation-like")
        if f >= len(eps_values):
            cap.set_text("\u03b5\u21920 recovers Hungarian   \u00b7   "
                         "\u03b5\u2192\u221e recovers uniform")
        else:
            cap.set_text(f"\u03b5 = {eps:.2f}")

    anim = animation.FuncAnimation(fig, update, frames=TOTAL,
                                   interval=1000/fps, blit=False)
    save_animation(anim, out_path, fps=fps)
    plt.close(fig)
    print(f"wrote {out_path}  ({TOTAL} frames, {fps} fps)")
