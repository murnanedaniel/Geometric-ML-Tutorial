"""02f — Hungarian: full dataset matching + green/red verdict."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from .helpers import (twins_dataset, hungarian_matches, setup_canvas,
                      save_animation, DEFAULT_FPS, FG, MUTE, GRID,
                      OK_COLOR, BAD_COLOR, GREY)


def render(out_path: Path, fps: int = DEFAULT_FPS):
    A, B, gt = twins_dataset()
    N = len(A)
    row, col, D = hungarian_matches(A, B)
    pred = np.empty(N, dtype=np.int64); pred[row] = col
    correct = int((pred == gt).sum())

    fig = plt.figure(figsize=(7.6, 8.0), dpi=100)
    ax = fig.add_axes([0.04, 0.06, 0.92, 0.78])
    setup_canvas(ax)

    ax.scatter(A[:, 0], A[:, 1], s=130, c=[GREY]*N, marker="o",
               edgecolors="white", linewidths=1.4, zorder=4)
    ax.scatter(B[:, 0], B[:, 1], s=130, c=[GREY]*N, marker="D",
               edgecolors="white", linewidths=1.4, zorder=4)

    match_lines = []
    for i in range(N):
        ln, = ax.plot([], [], lw=1.6, alpha=0.0, zorder=3)
        match_lines.append(ln)

    fig.text(0.5, 0.96, "Hungarian matching — verdict",
             ha="center", va="top", fontsize=15, fontweight="bold")
    sub = fig.text(0.5, 0.92, "", ha="center", va="top",
                   fontsize=11, color=MUTE)
    cap = fig.text(0.5, 0.025, "", ha="center", va="bottom",
                   fontsize=11, color=FG)

    DRAW = N * 4
    HOLD = 60
    TOTAL = DRAW + HOLD

    def update(f):
        if f < DRAW:
            n_show = f // 4 + 1
        else:
            n_show = N
        for i in range(N):
            if i < n_show:
                col_c = OK_COLOR if pred[i] == gt[i] else BAD_COLOR
                match_lines[i].set_data([A[i, 0], B[pred[i], 0]],
                                        [A[i, 1], B[pred[i], 1]])
                match_lines[i].set_color(col_c)
                match_lines[i].set_alpha(0.85)
            else:
                match_lines[i].set_alpha(0.0)
        n_ok = sum(1 for i in range(min(n_show, N)) if pred[i] == gt[i])
        sub.set_text("green = correct twin, red = wrong")
        cap.set_text(f"{n_ok}/{min(n_show, N)} correct")

    anim = animation.FuncAnimation(fig, update, frames=TOTAL,
                                   interval=1000/fps, blit=False)
    save_animation(anim, out_path, fps=fps)
    plt.close(fig)
    print(f"wrote {out_path}  ({TOTAL} frames, {fps} fps)")
