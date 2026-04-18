"""02c — Chamfer: full dataset matching + green/red verdict."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch
from .helpers import (twins_dataset, chamfer_matches, setup_canvas,
                      save_animation, DEFAULT_FPS, FG, MUTE, GRID,
                      A_COLOR, B_COLOR, OK_COLOR, BAD_COLOR, GREY)


def render(out_path: Path, fps: int = DEFAULT_FPS):
    A, B, gt = twins_dataset()
    N = len(A)
    a_to_b, _, _ = chamfer_matches(A, B)
    correct = int((a_to_b == gt).sum())
    distinct = len(set(a_to_b.tolist()))

    fig = plt.figure(figsize=(7.6, 8.0), dpi=100)
    ax = fig.add_axes([0.04, 0.06, 0.92, 0.78])
    setup_canvas(ax)

    ax.scatter(A[:, 0], A[:, 1], s=130, c=GREY, marker="o",
               edgecolors="white", linewidths=1.4, zorder=4)
    ax.scatter(B[:, 0], B[:, 1], s=130, c=GREY, marker="D",
               edgecolors="white", linewidths=1.4, zorder=4)

    arrows = []
    for i in range(N):
        ar = FancyArrowPatch((0, 0), (0, 0), arrowstyle="-|>",
                             mutation_scale=16, lw=1.6, color=A_COLOR,
                             alpha=0.0, zorder=3, shrinkA=4, shrinkB=4)
        ax.add_patch(ar)
        arrows.append(ar)

    fig.text(0.5, 0.96, "Chamfer matching — full dataset",
             ha="center", va="top", fontsize=15, fontweight="bold")
    sub = fig.text(0.5, 0.92, "", ha="center", va="top",
                   fontsize=11, color=MUTE)
    cap = fig.text(0.5, 0.025, "", ha="center", va="bottom",
                   fontsize=11, color=FG)

    TICK = 4
    SWEEP = N * TICK
    VERDICT = 50
    HOLD = 30
    TOTAL = SWEEP + VERDICT + HOLD

    def update(f):
        if f < SWEEP:
            idx = f // TICK
            for k in range(N):
                if k <= idx:
                    arrows[k].set_positions(A[k], B[a_to_b[k]])
                    arrows[k].set_alpha(0.8)
                else:
                    arrows[k].set_alpha(0.0)
            sub.set_text(f"A\u2192B nearest neighbour: {idx+1}/{N}")
            cap.set_text(""); return

        # Colour by correctness.
        for k in range(N):
            arrows[k].set_positions(A[k], B[a_to_b[k]])
            col = OK_COLOR if a_to_b[k] == gt[k] else BAD_COLOR
            arrows[k].set_color(col)
            arrows[k].set_alpha(0.85)

        if f < SWEEP + VERDICT:
            sub.set_text("green = correct twin, red = wrong")
            cap.set_text(f"correct: {correct}/{N}   "
                         f"unmatched B's: {N - distinct}"); return

        sub.set_text("Chamfer: cheap but greedy")
        cap.set_text(f"{correct}/{N} correct   \u00b7   "
                     f"no 1-to-1 constraint")

    anim = animation.FuncAnimation(fig, update, frames=TOTAL,
                                   interval=1000/fps, blit=False)
    save_animation(anim, out_path, fps=fps)
    plt.close(fig)
    print(f"wrote {out_path}  ({TOTAL} frames, {fps} fps)")
