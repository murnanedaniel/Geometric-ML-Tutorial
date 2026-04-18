"""02a — pair-coloured twins dataset: colours fade to grey challenge."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from .helpers import (twins_dataset, pair_colors, b_pair_colors,
                      fade_to_grey, setup_canvas, save_animation,
                      DEFAULT_FPS, GREY, FG, MUTE, GRID)


def render(out_path: Path, fps: int = DEFAULT_FPS):
    A, B, gt = twins_dataset()
    N = len(A)
    pcols = pair_colors(N)
    bcols = b_pair_colors(gt, pcols)

    fig = plt.figure(figsize=(7.6, 8.0), dpi=100)
    ax = fig.add_axes([0.04, 0.06, 0.92, 0.78])
    setup_canvas(ax)

    pair_lines = []
    for i in range(N):
        ln, = ax.plot([A[i, 0], B[gt[i], 0]], [A[i, 1], B[gt[i], 1]],
                      color=pcols[i], lw=1.6, alpha=0.0, zorder=2)
        pair_lines.append(ln)

    sa = ax.scatter(A[:, 0], A[:, 1], s=140, c=pcols, marker="o",
                    edgecolors="white", linewidths=1.5, zorder=4)
    sb = ax.scatter(B[:, 0], B[:, 1], s=140, c=bcols, marker="D",
                    edgecolors="white", linewidths=1.5, zorder=4)

    fig.text(0.5, 0.96, "Two groups of (non-identical) twins",
             ha="center", va="top", fontsize=15, fontweight="bold")
    sub = fig.text(0.5, 0.92, "", ha="center", va="top",
                   fontsize=11, color=MUTE)
    cap = fig.text(0.5, 0.025, "", ha="center", va="bottom",
                   fontsize=11, color=FG)
    fig.text(0.06, 0.87,
             "same colour = same pair\n"
             "circle \u25cf = A   diamond \u25c6 = B",
             ha="left", va="top", fontsize=9, color=FG,
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=GRID))

    HOLD1, REVEAL, HOLD2, FADE, HOLD3 = 40, 50, 40, 40, 35
    TOTAL = HOLD1 + REVEAL + HOLD2 + FADE + HOLD3

    def update(f):
        if f < HOLD1:
            sub.set_text("each circle–diamond pair of the same colour are twins")
            cap.set_text(""); return
        if f < HOLD1 + REVEAL:
            t = (f - HOLD1 + 1) / REVEAL
            for i, ln in enumerate(pair_lines):
                ln.set_alpha(0.85 if i < int(t * N) else 0.0)
            sub.set_text("ground truth"); cap.set_text(""); return
        if f < HOLD1 + REVEAL + HOLD2:
            for ln in pair_lines: ln.set_alpha(0.85)
            sub.set_text("twins are close but not co-located")
            cap.set_text(f"{N} pairs"); return
        if f < HOLD1 + REVEAL + HOLD2 + FADE:
            t = (f - HOLD1 - REVEAL - HOLD2 + 1) / FADE
            sa.set_facecolor(fade_to_grey(pcols, t))
            sb.set_facecolor(fade_to_grey(bcols, t))
            for ln in pair_lines: ln.set_alpha(0.85 * (1 - t))
            sub.set_text("now forget the colours \u2026")
            cap.set_text(""); return
        sa.set_facecolor([GREY] * N); sb.set_facecolor([GREY] * N)
        for ln in pair_lines: ln.set_alpha(0.0)
        sub.set_text("the challenge")
        cap.set_text("given only positions, recover the pairing")

    anim = animation.FuncAnimation(fig, update, frames=TOTAL,
                                   interval=1000/fps, blit=False)
    save_animation(anim, out_path, fps=fps)
    plt.close(fig)
    print(f"wrote {out_path}  ({TOTAL} frames, {fps} fps)")
