"""02g — Hungarian: why the assignment is not differentiable.

Nudge one B point by a small epsilon and show the assignment jumping."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from .helpers import (twins_dataset, hungarian_matches, setup_canvas,
                      save_animation, DEFAULT_FPS, FG, MUTE, GRID,
                      A_COLOR, B_COLOR, GREY, OK_COLOR, BAD_COLOR)


def render(out_path: Path, fps: int = DEFAULT_FPS):
    A, B, gt = twins_dataset()
    N = len(A)

    # Find a B point whose assignment flips when nudged.
    # Try nudging each B in a few directions until we find a flip.
    flip_j = None
    for j in range(N):
        _, col_orig, _ = hungarian_matches(A, B)
        for angle in np.linspace(0, 2*np.pi, 12, endpoint=False):
            B_nudge = B.copy()
            B_nudge[j] += 0.6 * np.array([np.cos(angle), np.sin(angle)])
            _, col_new, _ = hungarian_matches(A, B_nudge)
            if not np.array_equal(col_orig, col_new):
                flip_j = j
                flip_angle = angle
                flip_eps = 0.6
                break
        if flip_j is not None:
            break

    if flip_j is None:
        flip_j, flip_angle, flip_eps = 0, 0.0, 0.5

    # Compute assignments at a sweep of nudge magnitudes.
    n_nudge = 60
    nudge_mags = np.linspace(-flip_eps, flip_eps, n_nudge)
    direction = np.array([np.cos(flip_angle), np.sin(flip_angle)])
    assignments = []
    costs = []
    for mag in nudge_mags:
        B_n = B.copy()
        B_n[flip_j] += mag * direction
        row, col, D = hungarian_matches(A, B_n)
        pred = np.empty(N, dtype=np.int64); pred[row] = col
        assignments.append(pred.copy())
        costs.append(D[row, col].sum())

    fig = plt.figure(figsize=(12.0, 7.6), dpi=100)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.0], wspace=0.10,
                          left=0.04, right=0.97, top=0.84, bottom=0.12)
    ax = fig.add_subplot(gs[0])
    ax_c = fig.add_subplot(gs[1])
    setup_canvas(ax)

    sa = ax.scatter(A[:, 0], A[:, 1], s=130, c=[GREY]*N, marker="o",
                    edgecolors="white", linewidths=1.4, zorder=4)
    sb = ax.scatter(B[:, 0], B[:, 1], s=130, c=[GREY]*N, marker="D",
                    edgecolors="white", linewidths=1.4, zorder=4)
    # Highlight the nudged B.
    nudge_ring = ax.scatter([], [], s=300, facecolors="none",
                            edgecolors="#DC2626", linewidths=2.5, zorder=6)

    match_lines = []
    for _ in range(N):
        ln, = ax.plot([], [], color="#363D45", lw=1.5, alpha=0.0, zorder=3)
        match_lines.append(ln)

    # Cost vs nudge plot.
    ax_c.set_xlim(nudge_mags[0], nudge_mags[-1])
    ax_c.set_ylim(min(costs)*0.95, max(costs)*1.05)
    ax_c.set_xlabel("nudge magnitude", fontsize=10)
    ax_c.set_ylabel("total assignment cost", fontsize=10)
    ax_c.tick_params(labelsize=9)
    for s in ("top", "right"): ax_c.spines[s].set_visible(False)
    ax_c.set_title("cost is smooth, but assignment jumps",
                   fontsize=11, color=MUTE, pad=6, loc="left")
    cost_line, = ax_c.plot([], [], lw=2.2, color=A_COLOR)
    cost_dot = ax_c.scatter([], [], s=60, c=A_COLOR, edgecolors="white",
                            lw=1.2, zorder=5)

    fig.text(0.5, 0.96, "Hungarian is not differentiable",
             ha="center", va="top", fontsize=15, fontweight="bold")
    sub = fig.text(0.5, 0.92, "", ha="center", va="top",
                   fontsize=11, color=MUTE)
    cap = fig.text(0.5, 0.025, "", ha="center", va="bottom",
                   fontsize=11, color=FG)

    SWEEP = n_nudge * 2   # 2 frames per nudge step
    HOLD = 50
    TOTAL = SWEEP + HOLD

    prev_assignment = None

    def update(f):
        nonlocal prev_assignment
        if f < SWEEP:
            k = f // 2
        else:
            k = n_nudge - 1
        mag = nudge_mags[k]
        B_cur = B.copy()
        B_cur[flip_j] += mag * direction
        pred = assignments[k]

        sb.set_offsets(B_cur)
        nudge_ring.set_offsets([B_cur[flip_j]])

        # Draw matching lines.  Colour lines that CHANGED red.
        for i in range(N):
            match_lines[i].set_data([A[i, 0], B_cur[pred[i], 0]],
                                    [A[i, 1], B_cur[pred[i], 1]])
            changed = (prev_assignment is not None and
                       pred[i] != prev_assignment[i])
            match_lines[i].set_color("#DC2626" if changed else "#363D45")
            match_lines[i].set_alpha(0.85)

        prev_assignment = pred.copy()

        # Cost curve.
        cost_line.set_data(nudge_mags[:k+1], costs[:k+1])
        cost_dot.set_offsets([[mag, costs[k]]])

        if f < SWEEP:
            sub.set_text(f"nudging diamond {flip_j} "
                         f"(red ring) by {mag:+.2f}")
            cap.set_text("red lines = assignments that just changed")
        else:
            sub.set_text("the assignment is a discrete argmin \u2014 "
                         "its gradient is zero almost everywhere")
            cap.set_text("you can\u2019t backprop through Hungarian "
                         "in a neural network")

    anim = animation.FuncAnimation(fig, update, frames=TOTAL,
                                   interval=1000/fps, blit=False)
    save_animation(anim, out_path, fps=fps)
    plt.close(fig)
    print(f"wrote {out_path}  ({TOTAL} frames, {fps} fps)")
