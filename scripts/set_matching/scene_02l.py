"""02l — Synthesis: 4-panel comparison of optimised B positions."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from .helpers import (twins_dataset, optimise_B, setup_canvas, save_animation,
                      DEFAULT_FPS, FG, MUTE, GRID, GREY,
                      A_COLOR, B_COLOR, OK_COLOR, BAD_COLOR)


def render(out_path: Path, fps: int = DEFAULT_FPS):
    A, B, gt = twins_dataset()
    N = len(A)

    # Run all three optimisations.
    ch_snaps = optimise_B(A, B, "chamfer",   n_steps=40, lr=0.08)
    hu_snaps = optimise_B(A, B, "hungarian", n_steps=40, lr=0.08)
    sk_snaps = optimise_B(A, B, "sinkhorn",  n_steps=80, lr=0.50,
                          eps_sink=0.08)

    results = [
        ("ground truth", gt, B),
        ("Chamfer-optimised",   ch_snaps[-1][1], ch_snaps[-1][0]),
        ("Hungarian-optimised", hu_snaps[-1][1], hu_snaps[-1][0]),
        ("Sinkhorn-optimised",  sk_snaps[-1][1], sk_snaps[-1][0]),
    ]

    fig = plt.figure(figsize=(12.0, 13.0), dpi=90)
    gs = fig.add_gridspec(2, 2, wspace=0.08, hspace=0.14,
                          left=0.04, right=0.96, top=0.88, bottom=0.07)

    for idx in range(4):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        setup_canvas(ax)
        label, matching, B_cur = results[idx]

        ax.scatter(A[:, 0], A[:, 1], s=90, c=A_COLOR, marker="o",
                   edgecolors="white", linewidths=1.2, zorder=4)
        ax.scatter(B_cur[:, 0], B_cur[:, 1], s=90, c=B_COLOR, marker="D",
                   edgecolors="white", linewidths=1.2, zorder=4)

        if matching is not None:
            n_ok = 0
            for i in range(N):
                j = matching[i] if idx > 0 else gt[i]
                is_gt = (idx == 0)
                correct = (j == gt[i]) if not is_gt else True
                col = OK_COLOR if correct else BAD_COLOR
                if is_gt:
                    col = "#888888"
                if correct:
                    n_ok += 1
                ax.plot([A[i, 0], B_cur[j, 0]], [A[i, 1], B_cur[j, 1]],
                        color=col, lw=1.3, alpha=0.85, zorder=3)
            score = f"({n_ok}/{N})" if idx > 0 else "(20/20)"
        else:
            score = ""
        ax.set_title(f"{label}  {score}", fontsize=12, color=FG, pad=6,
                     loc="left", fontweight="bold")

    fig.text(0.5, 0.965, "Twin matching \u2014 four methods compared",
             ha="center", va="top", fontsize=16, fontweight="bold")
    fig.text(0.5, 0.93,
             "green = correct twin   red = wrong   grey = ground truth",
             ha="center", va="top", fontsize=11, color=MUTE)
    fig.text(0.5, 0.03,
             "Chamfer: greedy.  Hungarian: exact but discrete.  "
             "Sinkhorn: soft & differentiable.",
             ha="center", va="bottom", fontsize=11, color=FG)

    TOTAL = 70
    anim = animation.FuncAnimation(fig, lambda f: None, frames=TOTAL,
                                   interval=1000/fps, blit=False)
    save_animation(anim, out_path, fps=fps)
    plt.close(fig)
    print(f"wrote {out_path}  ({TOTAL} frames, {fps} fps)")
