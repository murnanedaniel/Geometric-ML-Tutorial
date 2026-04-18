"""02i_math — Sinkhorn: the algorithm shown slowly with real numbers.

One row at a time, one column at a time, with long pauses so the
audience can read the numbers before and after each normalisation step.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from .helpers import (twins_dataset, setup_canvas, save_animation,
                      DEFAULT_FPS, FG, MUTE, GRID, GREY,
                      A_COLOR, B_COLOR)


def render(out_path: Path, fps: int = DEFAULT_FPS):
    A_all, B_all, gt_all = twins_dataset()

    # Same 4-pair subset as 02e_math.
    sub_a = [0, 1, 6, 7]
    sub_b = [gt_all[i] for i in sub_a]
    M = len(sub_a)
    A = A_all[sub_a]
    B = B_all[sub_b]
    D = np.linalg.norm(A[:, None] - B[None, :], axis=-1)

    EPS = 0.4
    K = np.exp(-D / EPS)
    target = 1.0 / M   # = 0.25

    # ---- Layout: scatter left, matrix + margins right ----
    fig = plt.figure(figsize=(14.0, 7.8), dpi=100)
    gs = fig.add_gridspec(1, 2, width_ratios=[0.8, 1.2], wspace=0.06,
                          left=0.03, right=0.97, top=0.82, bottom=0.10)
    ax = fig.add_subplot(gs[0])
    ax_m = fig.add_subplot(gs[1])

    # Scatter.
    pts = np.concatenate([A, B])
    lo, hi = pts.min(0) - 1.0, pts.max(0) + 1.0
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ("left", "right", "top", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_title("twin scatter", fontsize=11, color=MUTE, pad=6)
    ax.scatter(A[:, 0], A[:, 1], s=220, c=A_COLOR, marker="o",
               edgecolors="white", linewidths=2, zorder=4)
    ax.scatter(B[:, 0], B[:, 1], s=220, c=B_COLOR, marker="D",
               edgecolors="white", linewidths=2, zorder=4)
    for i in range(M):
        ax.text(A[i, 0]+0.12, A[i, 1]+0.12, f"a{i}", fontsize=9,
                color=A_COLOR, fontweight="bold", zorder=5)
        ax.text(B[i, 0]+0.12, B[i, 1]+0.12, f"b{i}", fontsize=9,
                color=B_COLOR, fontweight="bold", zorder=5)

    # Flow lines on scatter.
    flow_lines = []
    for _ in range(M * M):
        ln, = ax.plot([], [], color="#2260A3", lw=1.5, alpha=0.0, zorder=3)
        flow_lines.append(ln)

    # ---- Matrix panel (positioned manually for margin room) ----
    mat_left, mat_right = 0.52, 0.88
    mat_bot, mat_top = 0.18, 0.72
    ax_m.set_position([mat_left, mat_bot,
                       mat_right - mat_left, mat_top - mat_bot])
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

    # Cell backgrounds + text.
    cell_bgs = []
    cell_texts = []
    for i in range(M):
        rbg, rtx = [], []
        for j in range(M):
            rect = Rectangle((j-0.5, i-0.5), 1, 1, facecolor="#F0EDE5",
                              edgecolor="white", lw=1.5, zorder=0)
            ax_m.add_patch(rect)
            rbg.append(rect)
            txt = ax_m.text(j, i, "", ha="center", va="center",
                            fontsize=11, fontweight="bold", color=FG, zorder=2)
            rtx.append(txt)
        cell_bgs.append(rbg)
        cell_texts.append(rtx)

    # Row-sum texts (right margin).
    row_sum_texts = []
    for i in range(M):
        txt = fig.text(mat_right + 0.015,
                       mat_bot + (i + 0.5) / M * (mat_top - mat_bot),
                       "", ha="left", va="center", fontsize=10, color=FG,
                       fontweight="bold")
        row_sum_texts.append(txt)
    fig.text(mat_right + 0.015, mat_top + 0.02, "row\nsum",
             ha="left", va="bottom", fontsize=9, color=MUTE)

    # Col-sum texts (bottom margin).
    col_sum_texts = []
    for j in range(M):
        txt = fig.text(mat_left + (j + 0.5) / M * (mat_right - mat_left),
                       mat_bot - 0.035, "", ha="center", va="top",
                       fontsize=10, color=FG, fontweight="bold")
        col_sum_texts.append(txt)
    fig.text(mat_left - 0.04, mat_bot - 0.035, "col\nsum",
             ha="right", va="top", fontsize=9, color=MUTE)

    # Header texts.
    fig.text(0.5, 0.96, "Sinkhorn \u2014 the algorithm",
             ha="center", va="top", fontsize=15, fontweight="bold")
    sub = fig.text(0.5, 0.92, "", ha="center", va="top",
                   fontsize=11, color=MUTE)
    cap = fig.text(0.5, 0.025, "", ha="center", va="bottom",
                   fontsize=11, color=FG)

    # ---- display helpers ----
    def show_P(P, title_str, active_row=-1, active_col=-1):
        """Render matrix P with numbers, row/col sums, cell colours."""
        ax_m.set_title(title_str, fontsize=11, color=MUTE, pad=6, loc="left")
        vmax = max(P.max(), 1e-6)
        for i in range(M):
            for j in range(M):
                cell_texts[i][j].set_text(f"{P[i,j]:.3f}")
                intensity = min(1.0, P[i, j] / vmax)
                r, g, b = (1 - 0.55*intensity, 1 - 0.25*intensity,
                           1 - 0.05*intensity)
                cell_bgs[i][j].set_facecolor((r, g, b))
                # Highlight active row/col.
                if i == active_row:
                    cell_bgs[i][j].set_edgecolor(A_COLOR)
                    cell_bgs[i][j].set_linewidth(3.0)
                elif j == active_col:
                    cell_bgs[i][j].set_edgecolor(B_COLOR)
                    cell_bgs[i][j].set_linewidth(3.0)
                else:
                    cell_bgs[i][j].set_edgecolor("white")
                    cell_bgs[i][j].set_linewidth(1.5)
        rs = P.sum(axis=1)
        cs = P.sum(axis=0)
        for i in range(M):
            row_sum_texts[i].set_text(f"{rs[i]:.3f}")
            row_sum_texts[i].set_color(
                "#2E8B57" if abs(rs[i] - target) < 0.003 else FG)
        for j in range(M):
            col_sum_texts[j].set_text(f"{cs[j]:.3f}")
            col_sum_texts[j].set_color(
                "#2E8B57" if abs(cs[j] - target) < 0.003 else FG)

    def draw_flow(P):
        vmax = max(P.max(), 1e-6)
        k = 0
        for i in range(M):
            for j in range(M):
                flow_lines[k].set_data([A[i, 0], B[j, 0]],
                                       [A[i, 1], B[j, 1]])
                mass = P[i, j] / vmax
                flow_lines[k].set_alpha(min(1.0, mass * 1.5))
                flow_lines[k].set_linewidth(max(0.5, mass * 5))
                k += 1

    def hide_flow():
        for ln in flow_lines:
            ln.set_alpha(0.0)

    # ---- Frame schedule ----
    # Each "step" gets a long hold so the audience can read numbers.
    HOLD = 60       # ~4.3 s per conceptual step

    # Steps:
    # 1) Show cost matrix C
    # 2) Show kernel K = exp(-C/eps)
    # 3) Show K with row sums — "not uniform"
    # 4-7) Normalise row 0, row 1, row 2, row 3  (one at a time)
    # 8) Pause: "rows done, but columns drifted"
    # 9-12) Normalise col 0, col 1, col 2, col 3
    # 13) Pause: "one full iteration done"
    # 14-17) Second iteration: row 0-3 (faster)
    # 18-21) Second iteration: col 0-3 (faster)
    # 22) Show flow lines on scatter
    # 23) Final hold

    MED  = 40       # medium-slow for iteration 2
    FAST = 6        # quick steps for iterations 3-12

    steps = []
    steps.append(("cost_C", HOLD))
    steps.append(("kernel_K", HOLD))
    steps.append(("K_sums", HOLD))
    # Iteration 1 — slow, one row/col at a time.
    for i in range(M):
        steps.append((f"row_{i}", HOLD))
    steps.append(("rows_done", HOLD))
    for j in range(M):
        steps.append((f"col_{j}", HOLD))
    steps.append(("iter1_done", HOLD))
    # Iteration 2 — medium pace.
    for i in range(M):
        steps.append((f"row2_{i}", MED))
    steps.append(("rows2_done", MED))
    for j in range(M):
        steps.append((f"col2_{j}", MED))
    steps.append(("iter2_done", HOLD))
    # Iterations 3-12 — fast (one step per full iteration).
    for it in range(3, 13):
        steps.append((f"fast_iter_{it}", FAST))
    steps.append(("converged", HOLD))
    steps.append(("flow", HOLD))
    steps.append(("final", 50))

    # Build frame→step mapping.
    frame_to_step = []
    for step_name, dur in steps:
        for _ in range(dur):
            frame_to_step.append(step_name)
    TOTAL = len(frame_to_step)

    # Pre-compute all intermediate matrices.
    def _row_norm(P):
        for i in range(M):
            rs = P[i].sum()
            if rs > 1e-30:
                P[i] *= target / rs

    def _col_norm(P):
        for j in range(M):
            cs = P[:, j].sum()
            if cs > 1e-30:
                P[:, j] *= target / cs

    P = K.copy()
    states = {"kernel_K": K.copy(), "K_sums": K.copy()}

    # Iteration 1 — per-row, per-col snapshots.
    for i in range(M):
        rs = P[i].sum()
        if rs > 1e-30:
            P[i] *= target / rs
        states[f"row_{i}"] = P.copy()
    states["rows_done"] = P.copy()
    for j in range(M):
        cs = P[:, j].sum()
        if cs > 1e-30:
            P[:, j] *= target / cs
        states[f"col_{j}"] = P.copy()
    states["iter1_done"] = P.copy()

    # Iteration 2 — per-row, per-col snapshots.
    for i in range(M):
        rs = P[i].sum()
        if rs > 1e-30:
            P[i] *= target / rs
        states[f"row2_{i}"] = P.copy()
    states["rows2_done"] = P.copy()
    for j in range(M):
        cs = P[:, j].sum()
        if cs > 1e-30:
            P[:, j] *= target / cs
        states[f"col2_{j}"] = P.copy()
    states["iter2_done"] = P.copy()

    # Iterations 3-12 — one snapshot per full iteration.
    for it in range(3, 13):
        _row_norm(P)
        _col_norm(P)
        states[f"fast_iter_{it}"] = P.copy()
    states["converged"] = P.copy()
    states["flow"] = P.copy()
    states["final"] = P.copy()

    def update(f):
        hide_flow()
        step = frame_to_step[min(f, TOTAL - 1)]

        if step == "cost_C":
            show_P(D, r"cost matrix $C_{ij} = \|a_i - b_j\|$")
            sub.set_text(f"start from pairwise distances   "
                         f"(\u03b5 = {EPS})")
            cap.set_text(""); return

        if step == "kernel_K":
            show_P(K, r"kernel $K_{ij} = \exp(-C_{ij} / \varepsilon)$")
            sub.set_text(r"$K = \exp(-C / \varepsilon)$:  "
                         "low cost \u2192 large entry, high cost \u2192 tiny")
            cap.set_text(""); return

        if step == "K_sums":
            show_P(K, "K with row/col sums")
            sub.set_text("row sums are unequal \u2014 "
                         "not a valid transport plan yet")
            cap.set_text(f"we need every row and column to sum to "
                         f"1/N = {target:.3f}")
            return

        # Row normalisation (iteration 1).
        for i in range(M):
            if step == f"row_{i}":
                show_P(states[step], f"normalise row a{i}",
                       active_row=i)
                rs_before = (states[f"row_{i-1}"] if i > 0
                             else K)[i].sum()
                sub.set_text(f"row a{i}:  divide every entry by "
                             f"row sum ({rs_before:.3f}), "
                             f"multiply by {target:.3f}")
                cap.set_text(f"row a{i} sum is now {target:.3f} \u2714")
                return

        if step == "rows_done":
            show_P(states[step], "all rows normalised")
            sub.set_text("all row sums = 0.250 \u2714   "
                         "but column sums have drifted \u2718")
            cap.set_text("next: normalise columns"); return

        # Column normalisation (iteration 1).
        for j in range(M):
            if step == f"col_{j}":
                show_P(states[step], f"normalise column b{j}",
                       active_col=j)
                cs_before = (states[f"col_{j-1}"] if j > 0
                             else states["rows_done"])[:, j].sum()
                sub.set_text(f"col b{j}:  divide by col sum "
                             f"({cs_before:.3f}), multiply by {target:.3f}")
                cap.set_text(f"col b{j} sum is now {target:.3f} \u2714")
                return

        if step == "iter1_done":
            show_P(states[step], "iteration 1 complete")
            sub.set_text("col sums = 0.250 \u2714   "
                         "but row sums drifted slightly \u2718")
            cap.set_text("iterate again \u2014 each pass gets closer "
                         "to doubly stochastic"); return

        # Iteration 2 — medium pace, same structure.
        for i in range(M):
            if step == f"row2_{i}":
                show_P(states[step], f"iter 2: normalise row a{i}",
                       active_row=i)
                sub.set_text("iteration 2 \u2014 row normalisation")
                cap.set_text(f"row a{i} \u2192 sum = {target:.3f}")
                return
        if step == "rows2_done":
            show_P(states[step], "iter 2: rows done")
            sub.set_text("iteration 2 \u2014 rows normalised, "
                         "now columns")
            cap.set_text(""); return
        for j in range(M):
            if step == f"col2_{j}":
                show_P(states[step], f"iter 2: normalise col b{j}",
                       active_col=j)
                sub.set_text("iteration 2 \u2014 column normalisation")
                cap.set_text(f"col b{j} \u2192 sum = {target:.3f}")
                return
        if step == "iter2_done":
            show_P(states[step], "iteration 2 complete")
            sub.set_text("2 iterations done \u2014 "
                         "already much closer to doubly stochastic")
            cap.set_text("10 more quick iterations \u2026"); return

        # Iterations 3-12 — fast.
        for it in range(3, 13):
            if step == f"fast_iter_{it}":
                show_P(states[step], f"iteration {it}")
                rs = states[step].sum(axis=1)
                dev = np.abs(rs - target).max()
                sub.set_text(f"iteration {it}/12")
                cap.set_text(f"max row deviation = {dev:.5f}")
                return

        if step == "converged":
            show_P(states[step], "converged (12 iterations)")
            rs = states[step].sum(axis=1)
            cs = states[step].sum(axis=0)
            r_dev = np.abs(rs - target).max()
            c_dev = np.abs(cs - target).max()
            sub.set_text("converged \u2014 plan is doubly stochastic")
            cap.set_text(f"max row dev = {r_dev:.6f}   "
                         f"max col dev = {c_dev:.6f}"); return

        if step == "flow":
            show_P(states[step], "transport plan P (2 iterations)")
            draw_flow(states[step])
            sub.set_text("each P[i,j] = mass sent from a_i to b_j")
            cap.set_text("line thickness \u221d transport mass"); return

        # Final.
        show_P(states["final"], "transport plan P (2 iterations)")
        draw_flow(states["final"])
        sub.set_text("iterate row/col normalisation until convergence "
                     "\u2014 that\u2019s Sinkhorn")
        cap.set_text(f"\u03b5 = {EPS}   \u00b7   "
                     "fully differentiable at every step")

    anim = animation.FuncAnimation(fig, update, frames=TOTAL,
                                   interval=1000/fps, blit=False)
    save_animation(anim, out_path, fps=fps)
    plt.close(fig)
    print(f"wrote {out_path}  ({TOTAL} frames, {fps} fps)")
