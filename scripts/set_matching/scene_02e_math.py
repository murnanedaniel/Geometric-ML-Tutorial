"""02e_math — Hungarian: building intuition with real numbers on a 4×4 subset."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch, Rectangle
from .helpers import (twins_dataset, setup_canvas, save_animation,
                      DEFAULT_FPS, FG, MUTE, GRID, GREY,
                      A_COLOR, B_COLOR, OK_COLOR, BAD_COLOR)
from scipy.optimize import linear_sum_assignment


def render(out_path: Path, fps: int = DEFAULT_FPS):
    A_all, B_all, gt_all = twins_dataset()

    # Pick a confusing 4-pair subset (pairs 0,1,6,7 share a cluster).
    sub_a = [0, 1, 6, 7]
    sub_b = [gt_all[i] for i in sub_a]
    M = len(sub_a)
    A = A_all[sub_a]
    B = B_all[sub_b]
    D = np.linalg.norm(A[:, None] - B[None, :], axis=-1)

    # Hungarian on the 4×4.
    h_row, h_col = linear_sum_assignment(D)
    h_cost = D[h_row, h_col].sum()
    greedy_col = D.argmin(axis=1)

    # ---- Layout: scatter left, matrix right ----
    fig = plt.figure(figsize=(13.0, 7.6), dpi=100)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.1], wspace=0.08,
                          left=0.04, right=0.97, top=0.82, bottom=0.10)
    ax = fig.add_subplot(gs[0])
    ax_m = fig.add_subplot(gs[1])

    # Scatter — zoom to the cluster.
    pts = np.concatenate([A, B])
    lo, hi = pts.min(0) - 1.0, pts.max(0) + 1.0
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ("left", "right", "top", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_title("twin scatter (zoomed)", fontsize=11, color=MUTE, pad=6)

    sa = ax.scatter(A[:, 0], A[:, 1], s=220, c=A_COLOR, marker="o",
                    edgecolors="white", linewidths=2.0, zorder=4)
    sb = ax.scatter(B[:, 0], B[:, 1], s=220, c=B_COLOR, marker="D",
                    edgecolors="white", linewidths=2.0, zorder=4)
    # Index labels on points.
    for i in range(M):
        ax.text(A[i, 0]+0.12, A[i, 1]+0.12, f"a{i}", fontsize=9,
                color=A_COLOR, fontweight="bold", zorder=5)
        ax.text(B[i, 0]+0.12, B[i, 1]+0.12, f"b{i}", fontsize=9,
                color=B_COLOR, fontweight="bold", zorder=5)

    # Scatter connection lines (drawn during various phases).
    scat_lines = []
    for _ in range(M * M):
        ln, = ax.plot([], [], lw=1.5, alpha=0.0, zorder=3)
        scat_lines.append(ln)

    # Highlight rings.
    ring_a = ax.scatter([], [], s=400, facecolors="none",
                        edgecolors=FG, linewidths=2.5, zorder=6)
    ring_b = ax.scatter([], [], s=400, facecolors="none",
                        edgecolors=FG, linewidths=2.5, zorder=6, marker="D")

    # ---- Matrix panel ----
    ax_m.set_xlim(-0.5, M - 0.5); ax_m.set_ylim(-0.5, M - 0.5)
    ax_m.set_aspect("equal")
    ax_m.set_xticks(range(M)); ax_m.set_yticks(range(M))
    ax_m.set_xticklabels([f"b{j}" for j in range(M)], fontsize=10,
                         color=B_COLOR)
    ax_m.set_yticklabels([f"a{i}" for i in range(M)], fontsize=10,
                         color=A_COLOR)
    ax_m.tick_params(length=0, pad=6)
    for s in ("left", "right", "top", "bottom"):
        ax_m.spines[s].set_visible(False)
    ax_m.set_title(r"cost matrix  $C_{ij} = \|a_i - b_j\|$",
                   fontsize=11, color=MUTE, pad=6)

    # Cell backgrounds (light grid).
    cell_bgs = []
    for i in range(M):
        row_bgs = []
        for j in range(M):
            rect = Rectangle((j-0.5, i-0.5), 1, 1, facecolor="#F0EDE5",
                              edgecolor="white", lw=1.5, zorder=0)
            ax_m.add_patch(rect)
            row_bgs.append(rect)
        cell_bgs.append(row_bgs)

    # Cell text objects (numbers appear during animation).
    cell_texts = []
    for i in range(M):
        row_texts = []
        for j in range(M):
            txt = ax_m.text(j, i, "", ha="center", va="center",
                            fontsize=13, fontweight="bold", color=FG,
                            zorder=2)
            row_texts.append(txt)
        cell_texts.append(row_texts)

    # Selection markers on the matrix.
    sel_rects = []
    for _ in range(M):
        rect = Rectangle((0, 0), 1, 1, facecolor="none",
                          edgecolor=FG, lw=3.0, zorder=3, alpha=0.0)
        ax_m.add_patch(rect)
        sel_rects.append(rect)

    fig.text(0.5, 0.96, "Hungarian matching — the math",
             ha="center", va="top", fontsize=15, fontweight="bold")
    sub = fig.text(0.5, 0.92, "", ha="center", va="top",
                   fontsize=11, color=MUTE)
    cap = fig.text(0.5, 0.025, "", ha="center", va="bottom",
                   fontsize=11, color=FG)

    # ---- helpers ----
    def clear_all():
        for ln in scat_lines: ln.set_alpha(0.0)
        ring_a.set_alpha(0.0); ring_b.set_alpha(0.0)
        for r in sel_rects: r.set_alpha(0.0)
        for i in range(M):
            for j in range(M):
                cell_texts[i][j].set_text("")
                cell_bgs[i][j].set_facecolor("#F0EDE5")

    def show_number(i, j, highlight=False, color=FG):
        cell_texts[i][j].set_text(f"{D[i,j]:.2f}")
        cell_texts[i][j].set_color(color)
        if highlight:
            cell_bgs[i][j].set_facecolor("#FFE0B2")

    # ---- Frame schedule ----
    P1 = 70          # compute one entry
    P2 = 70          # fill the row
    P3 = 80          # greedy row-min + conflict
    P4 = 80          # optimal assignment
    P5 = 40          # hold
    TOTAL = P1 + P2 + P3 + P4 + P5

    # Pick one (i,j) to compute first.
    demo_i, demo_j = 0, 1  # A[0] → B[1], which is the greedy-min entry

    def update(f):
        clear_all()

        # ===== Phase 1: compute one entry =====
        if f < P1:
            i, j = demo_i, demo_j
            t = min(1.0, (f + 1) / 30)
            # Highlight A[i] and B[j].
            ring_a.set_offsets([A[i]]); ring_a.set_alpha(t)
            ring_b.set_offsets([B[j]]); ring_b.set_alpha(t)
            # Draw dashed line between them.
            scat_lines[0].set_data([A[i, 0], B[j, 0]],
                                   [A[i, 1], B[j, 1]])
            scat_lines[0].set_color(FG)
            scat_lines[0].set_linestyle("--")
            scat_lines[0].set_alpha(0.8 * t)
            # Show number in the cell.
            show_number(i, j, highlight=True)
            sub.set_text(f"$C[a_0, b_1] = \\|a_0 - b_1\\| = {D[i,j]:.2f}$")
            cap.set_text("each entry = Euclidean distance between one "
                         "circle and one diamond")
            return

        # ===== Phase 2: fill row 0 =====
        if f < P1 + P2:
            i = demo_i
            t = (f - P1 + 1) / P2
            n_cols = max(1, int(t * M))
            ring_a.set_offsets([A[i]]); ring_a.set_alpha(0.9)
            ring_b.set_alpha(0.0)
            for j in range(M):
                if j < n_cols:
                    show_number(i, j, highlight=(j < n_cols))
                    scat_lines[j].set_data([A[i, 0], B[j, 0]],
                                           [A[i, 1], B[j, 1]])
                    scat_lines[j].set_color(MUTE)
                    scat_lines[j].set_linestyle("-")
                    scat_lines[j].set_alpha(0.5)
            # Also fill remaining rows with numbers (all at once).
            if t > 0.6:
                for ii in range(1, M):
                    for jj in range(M):
                        show_number(ii, jj)
            sub.set_text(f"fill the whole matrix — row $a_0$ shown")
            cap.set_text("each row = distances from one circle to every "
                         "diamond")
            return

        # ===== Phase 3: greedy row-min + conflict =====
        if f < P1 + P2 + P3:
            # Show all numbers.
            for ii in range(M):
                for jj in range(M):
                    show_number(ii, jj)
            ring_a.set_alpha(0.0); ring_b.set_alpha(0.0)

            t = (f - P1 - P2 + 1) / P3
            n_rows_done = max(1, int(t * M))
            # Highlight the greedy min in each row.
            for ii in range(min(n_rows_done, M)):
                jj = int(greedy_col[ii])
                cell_bgs[ii][jj].set_facecolor("#B8E6C8")
                sel_rects[ii].set_xy((jj - 0.5, ii - 0.5))
                sel_rects[ii].set_alpha(0.9)
                # Draw matching line on scatter.
                ln = scat_lines[ii]
                ln.set_data([A[ii, 0], B[jj, 0]],
                            [A[ii, 1], B[jj, 1]])
                ln.set_color(OK_COLOR)
                ln.set_linestyle("-")
                ln.set_alpha(0.8)

            # Detect conflict: rows 0 and 1 both pick col 1.
            if n_rows_done >= 2 and greedy_col[0] == greedy_col[1]:
                cj = int(greedy_col[0])
                cell_bgs[0][cj].set_facecolor("#FFCDD2")
                cell_bgs[1][cj].set_facecolor("#FFCDD2")
                sel_rects[0].set_edgecolor(BAD_COLOR)
                sel_rects[1].set_edgecolor(BAD_COLOR)
                sub.set_text("greedy: pick cheapest per row — "
                             f"but rows 0 and 1 both want b{cj}!")
                cap.set_text("conflict! greedy fails when two rows "
                             "claim the same column")
            else:
                sub.set_text(f"greedy row-min: {n_rows_done}/{M} rows")
                cap.set_text("")
            return

        # ===== Phase 4: optimal assignment =====
        if f < P1 + P2 + P3 + P4:
            for ii in range(M):
                for jj in range(M):
                    show_number(ii, jj)
            ring_a.set_alpha(0.0); ring_b.set_alpha(0.0)

            t = (f - P1 - P2 - P3 + 1) / P4
            n_show = max(1, int(t * M))
            for k in range(min(n_show, M)):
                ii, jj = int(h_row[k]), int(h_col[k])
                cell_bgs[ii][jj].set_facecolor("#C8E6C9")
                sel_rects[k].set_xy((jj - 0.5, ii - 0.5))
                sel_rects[k].set_edgecolor("#1B5E20")
                sel_rects[k].set_alpha(0.9)
                scat_lines[k].set_data([A[ii, 0], B[jj, 0]],
                                       [A[ii, 1], B[jj, 1]])
                scat_lines[k].set_color("#1B5E20")
                scat_lines[k].set_linestyle("-")
                scat_lines[k].set_linewidth(2.5)
                scat_lines[k].set_alpha(0.9)
            partial = D[h_row[:n_show], h_col[:n_show]].sum()
            sub.set_text("Hungarian: one per row, one per column, "
                         "minimum total")
            cap.set_text(f"{n_show}/{M} assigned — "
                         f"cost so far = {partial:.2f}")
            return

        # ===== Phase 5: hold =====
        for ii in range(M):
            for jj in range(M):
                show_number(ii, jj)
        for k in range(M):
            ii, jj = int(h_row[k]), int(h_col[k])
            cell_bgs[ii][jj].set_facecolor("#C8E6C9")
            sel_rects[k].set_xy((jj - 0.5, ii - 0.5))
            sel_rects[k].set_edgecolor("#1B5E20")
            sel_rects[k].set_alpha(0.9)
            scat_lines[k].set_data([A[ii, 0], B[jj, 0]],
                                   [A[ii, 1], B[jj, 1]])
            scat_lines[k].set_color("#1B5E20")
            scat_lines[k].set_linewidth(2.5)
            scat_lines[k].set_alpha(0.9)
        sub.set_text(f"optimal 1-to-1 assignment — total cost = {h_cost:.2f}")
        cap.set_text("Hungarian algorithm solves this exactly for any N")

    anim = animation.FuncAnimation(fig, update, frames=TOTAL,
                                   interval=1000/fps, blit=False)
    save_animation(anim, out_path, fps=fps)
    plt.close(fig)
    print(f"wrote {out_path}  ({TOTAL} frames, {fps} fps)")
