"""02e — Hungarian: the assignment problem (cost matrix + optimal vs random)."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from .helpers import (twins_dataset, hungarian_matches, setup_canvas,
                      save_animation, DEFAULT_FPS, FG, MUTE, GRID,
                      A_COLOR, B_COLOR, GREY)


COST_CMAP = LinearSegmentedColormap.from_list(
    "cost", ["#FFF6E5", "#FFCFA5", "#F28D35", "#C2417C", "#5B2A86", "#1B0F3B"])


def render(out_path: Path, fps: int = DEFAULT_FPS):
    A, B, gt = twins_dataset()
    N = len(A)
    row, col, D = hungarian_matches(A, B)
    opt_cost = D[row, col].sum()

    # A random assignment for comparison.
    rng = np.random.default_rng(42)
    rand_perm = rng.permutation(N)
    rand_cost = sum(D[i, rand_perm[i]] for i in range(N))

    fig = plt.figure(figsize=(12.0, 7.6), dpi=100)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.95], wspace=0.12,
                          left=0.04, right=0.97, top=0.84, bottom=0.10)
    ax = fig.add_subplot(gs[0])
    ax_m = fig.add_subplot(gs[1])
    setup_canvas(ax)

    ax.scatter(A[:, 0], A[:, 1], s=130, c=[GREY]*N, marker="o",
               edgecolors="white", linewidths=1.4, zorder=4)
    ax.scatter(B[:, 0], B[:, 1], s=130, c=[GREY]*N, marker="D",
               edgecolors="white", linewidths=1.4, zorder=4)

    match_lines = []
    for _ in range(N):
        ln, = ax.plot([], [], color="#363D45", lw=1.6, alpha=0.0, zorder=3)
        match_lines.append(ln)

    img = ax_m.imshow(np.zeros((N, N)), cmap=COST_CMAP, vmin=0, vmax=D.max(),
                      origin="lower", aspect="equal")
    ax_m.set_xticks([]); ax_m.set_yticks([])
    for s in ("left", "right", "top", "bottom"):
        ax_m.spines[s].set_visible(False)
    ax_m.set_title(r"cost matrix $C_{ij} = \|a_i - b_j\|$",
                   fontsize=11, color=MUTE, pad=6, loc="left")
    sel_scat = ax_m.scatter([], [], s=90, marker="s", facecolors="none",
                            edgecolors="#1F1F1F", linewidths=2.0,
                            zorder=5, alpha=0.0)

    fig.text(0.5, 0.96, "Hungarian matching \u00b7 exact 1-to-1",
             ha="center", va="top", fontsize=15, fontweight="bold")
    sub = fig.text(0.5, 0.92, "", ha="center", va="top",
                   fontsize=11, color=MUTE)
    cap = fig.text(0.5, 0.025, "", ha="center", va="bottom",
                   fontsize=11, color=FG)

    FILL = 30
    SHOW_RAND = 50
    SHOW_OPT = N * 4 + 40
    HOLD = 30
    TOTAL = FILL + SHOW_RAND + SHOW_OPT + HOLD

    sel_pairs = list(zip(row, col))

    def update(f):
        # Phase 1: fill cost matrix.
        if f < FILL:
            t = (f + 1) / FILL
            nr = int(t * N)
            partial = np.full((N, N), np.nan)
            partial[:nr] = D[:nr]
            img.set_data(partial)
            sel_scat.set_alpha(0.0)
            for ln in match_lines: ln.set_alpha(0.0)
            sub.set_text("building the pairwise cost matrix")
            cap.set_text(f"row {nr}/{N}"); return

        img.set_data(D)

        # Phase 2: random assignment.
        if f < FILL + SHOW_RAND:
            sel_scat.set_offsets([[rand_perm[i], i] for i in range(N)])
            sel_scat.set_edgecolors("#C62828")
            sel_scat.set_alpha(0.9)
            for i in range(N):
                match_lines[i].set_data([A[i, 0], B[rand_perm[i], 0]],
                                        [A[i, 1], B[rand_perm[i], 1]])
                match_lines[i].set_color("#C62828")
                match_lines[i].set_alpha(0.5)
            sub.set_text("a random 1-to-1 assignment")
            cap.set_text(f"total cost = {rand_cost:.2f}"); return

        # Phase 3: optimal (Hungarian) assignment lights up one by one.
        f3 = f - FILL - SHOW_RAND
        if f3 < N * 4 + 40:
            n_show = min(N, f3 // 4 + 1) if f3 < N * 4 else N
            xs = [c for (_, c) in sel_pairs[:n_show]]
            ys = [r for (r, _) in sel_pairs[:n_show]]
            sel_scat.set_offsets(list(zip(xs, ys)))
            sel_scat.set_edgecolors("#1F1F1F")
            sel_scat.set_alpha(0.95)
            for i in range(N):
                if i < n_show:
                    r_, c_ = sel_pairs[i]
                    match_lines[i].set_data([A[r_, 0], B[c_, 0]],
                                            [A[r_, 1], B[c_, 1]])
                    match_lines[i].set_color("#363D45")
                    match_lines[i].set_alpha(0.85)
                else:
                    match_lines[i].set_alpha(0.0)
            partial_cost = D[row[:n_show], col[:n_show]].sum()
            sub.set_text("optimal: one per row, one per column")
            cap.set_text(f"{n_show}/{N} matched \u00b7 "
                         f"cost = {partial_cost:.2f} "
                         f"(random was {rand_cost:.2f})"); return

        # Hold.
        sub.set_text("Hungarian: exact, optimal, combinatorial")
        cap.set_text(f"total cost = {opt_cost:.2f}")

    anim = animation.FuncAnimation(fig, update, frames=TOTAL,
                                   interval=1000/fps, blit=False)
    save_animation(anim, out_path, fps=fps)
    plt.close(fig)
    print(f"wrote {out_path}  ({TOTAL} frames, {fps} fps)")
