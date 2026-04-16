"""02i_math — Sinkhorn: one iteration with real numbers on a 4×4 subset."""

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
    target = 1.0 / M

    # Pre-run a few Sinkhorn half-steps manually so we can show
    # row-norm → col-norm with real numbers.
    u, v = np.ones(M), np.ones(M)
    snapshots = []  # list of (P, label, row_sums, col_sums)
    P0 = (u[:, None] * K) * v[None, :]
    snapshots.append((P0.copy(), "initial $K$",
                      P0.sum(axis=1), P0.sum(axis=0)))

    # Row normalisation.
    u = (target) / (K @ v + 1e-30)
    P1 = (u[:, None] * K) * v[None, :]
    snapshots.append((P1.copy(), "after row normalisation",
                      P1.sum(axis=1), P1.sum(axis=0)))

    # Column normalisation.
    v = (target) / (K.T @ u + 1e-30)
    P2 = (u[:, None] * K) * v[None, :]
    snapshots.append((P2.copy(), "after column normalisation",
                      P2.sum(axis=1), P2.sum(axis=0)))

    # One more full iteration for good measure.
    u = (target) / (K @ v + 1e-30)
    v = (target) / (K.T @ u + 1e-30)
    P3 = (u[:, None] * K) * v[None, :]
    snapshots.append((P3.copy(), "after iteration 2",
                      P3.sum(axis=1), P3.sum(axis=0)))

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

    # ---- Matrix panel ----
    # We leave room for margin annotations (row sums right, col sums top).
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

    # Cell backgrounds.
    cell_bgs = []
    for i in range(M):
        row_bgs = []
        for j in range(M):
            rect = Rectangle((j-0.5, i-0.5), 1, 1, facecolor="#F0EDE5",
                              edgecolor="white", lw=1.5, zorder=0)
            ax_m.add_patch(rect)
            row_bgs.append(rect)
        cell_bgs.append(row_bgs)

    # Cell number texts.
    cell_texts = []
    for i in range(M):
        row_t = []
        for j in range(M):
            txt = ax_m.text(j, i, "", ha="center", va="center",
                            fontsize=11, fontweight="bold", color=FG, zorder=2)
            row_t.append(txt)
        cell_texts.append(row_t)

    # Row-sum annotations (to the right of the matrix).
    row_sum_texts = []
    for i in range(M):
        txt = fig.text(mat_right + 0.015, mat_bot + (i + 0.5) / M *
                       (mat_top - mat_bot), "", ha="left", va="center",
                       fontsize=10, color=FG, fontweight="bold")
        row_sum_texts.append(txt)
    fig.text(mat_right + 0.015, mat_top + 0.02, "row\nsum",
             ha="left", va="bottom", fontsize=9, color=MUTE)

    # Col-sum annotations (below the matrix).
    col_sum_texts = []
    for j in range(M):
        txt = fig.text(mat_left + (j + 0.5) / M * (mat_right - mat_left),
                       mat_bot - 0.035, "", ha="center", va="top",
                       fontsize=10, color=FG, fontweight="bold")
        col_sum_texts.append(txt)
    fig.text(mat_left - 0.04, mat_bot - 0.035, "col\nsum",
             ha="right", va="top", fontsize=9, color=MUTE)

    fig.text(0.5, 0.96, "Sinkhorn — the math",
             ha="center", va="top", fontsize=15, fontweight="bold")
    sub = fig.text(0.5, 0.92, "", ha="center", va="top",
                   fontsize=11, color=MUTE)
    cap = fig.text(0.5, 0.025, "", ha="center", va="bottom",
                   fontsize=11, color=FG)
    mat_title = ax_m.set_title("", fontsize=11, color=MUTE, pad=6,
                               loc="left")

    # ---- helpers ----
    def show_matrix(P, label, highlight_rows=False, highlight_cols=False):
        ax_m.set_title(label, fontsize=11, color=MUTE, pad=6, loc="left")
        vmax = max(P.max(), 1e-6)
        for i in range(M):
            for j in range(M):
                cell_texts[i][j].set_text(f"{P[i,j]:.3f}")
                # Color intensity.
                intensity = P[i, j] / vmax
                r, g, b = (1 - 0.6*intensity, 1 - 0.3*intensity,
                           1 - 0.1*intensity)
                cell_bgs[i][j].set_facecolor((r, g, b))
                if highlight_rows:
                    cell_bgs[i][j].set_edgecolor(A_COLOR)
                elif highlight_cols:
                    cell_bgs[i][j].set_edgecolor(B_COLOR)
                else:
                    cell_bgs[i][j].set_edgecolor("white")
        rs = P.sum(axis=1)
        cs = P.sum(axis=0)
        for i in range(M):
            row_sum_texts[i].set_text(f"{rs[i]:.3f}")
            ok = abs(rs[i] - target) < 0.002
            row_sum_texts[i].set_color("#2E8B57" if ok else FG)
        for j in range(M):
            col_sum_texts[j].set_text(f"{cs[j]:.3f}")
            ok = abs(cs[j] - target) < 0.002
            col_sum_texts[j].set_color("#2E8B57" if ok else FG)

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
    P_COST = 55       # show C values
    P_KERNEL = 55     # show K = exp(-C/eps)
    P_ROWNORM = 70    # row normalisation
    P_COLNORM = 70    # col normalisation
    P_FLOW = 55       # connect to twin space
    P_HOLD = 40
    TOTAL = P_COST + P_KERNEL + P_ROWNORM + P_COLNORM + P_FLOW + P_HOLD

    def update(f):
        hide_flow()

        # ===== Phase 1: show cost matrix C =====
        if f < P_COST:
            show_matrix(D, r"cost matrix $C_{ij} = \|a_i - b_j\|$")
            sub.set_text(f"start from the cost matrix   "
                         f"(\u03b5 = {EPS})")
            cap.set_text("same pairwise distances as Hungarian")
            return

        # ===== Phase 2: C → K = exp(-C/ε) =====
        if f < P_COST + P_KERNEL:
            t = (f - P_COST + 1) / P_KERNEL
            # Blend C → K visually.
            blend = D * (1 - t) + K * t
            show_matrix(blend if t < 0.5 else K,
                        r"kernel $K_{ij} = \exp(-C_{ij}/\varepsilon)$")
            if t > 0.5:
                show_matrix(K, r"kernel $K_{ij}$")
            sub.set_text(r"$K = \exp(-C / \varepsilon)$  — "
                         "high cost → small kernel, low cost → large kernel")
            cap.set_text("row sums are unequal — not a valid transport "
                         "plan yet")
            return

        # ===== Phase 3: row normalisation =====
        if f < P_COST + P_KERNEL + P_ROWNORM:
            t = (f - P_COST - P_KERNEL + 1) / P_ROWNORM
            n_rows = max(1, int(t * M))
            P_snap = snapshots[0][0].copy()  # start from K
            # Normalise rows 0..n_rows-1.
            for i in range(n_rows):
                rs = P_snap[i].sum()
                if rs > 1e-30:
                    P_snap[i] *= target / rs
            show_matrix(P_snap, f"row normalisation ({n_rows}/{M} rows)",
                        highlight_rows=True)
            sub.set_text("divide each row by its sum, "
                         f"multiply by 1/N = {target:.3f}")
            done_rows = sum(1 for i in range(M)
                            if abs(P_snap[i].sum() - target) < 0.002)
            cap.set_text(f"{done_rows}/{M} rows normalised — "
                         "columns may have drifted")
            return

        # ===== Phase 4: column normalisation =====
        if f < P_COST + P_KERNEL + P_ROWNORM + P_COLNORM:
            t = (f - P_COST - P_KERNEL - P_ROWNORM + 1) / P_COLNORM
            n_cols = max(1, int(t * M))
            P_snap = snapshots[1][0].copy()  # after row norm
            for j in range(n_cols):
                cs = P_snap[:, j].sum()
                if cs > 1e-30:
                    P_snap[:, j] *= target / cs
            show_matrix(P_snap, f"column normalisation ({n_cols}/{M} cols)",
                        highlight_cols=True)
            sub.set_text("divide each column by its sum, "
                         f"multiply by 1/N = {target:.3f}")
            done_cols = sum(1 for j in range(M)
                            if abs(P_snap[:, j].sum() - target) < 0.002)
            cap.set_text(f"{done_cols}/{M} columns normalised — "
                         "rows may have drifted slightly")
            return

        # ===== Phase 5: connect to twin space =====
        if f < P_COST + P_KERNEL + P_ROWNORM + P_COLNORM + P_FLOW:
            P_final = snapshots[2][0]  # after one full iteration
            show_matrix(P_final, "transport plan $P$ (1 iteration)")
            draw_flow(P_final)
            sub.set_text("each $P_{ij}$ = fraction of mass "
                         "sent from $a_i$ to $b_j$")
            cap.set_text("line thickness \u221d transport mass — "
                         "iterate to convergence")
            return

        # ===== Phase 6: hold =====
        P_final = snapshots[-1][0]
        show_matrix(P_final, "transport plan $P$ (2 iterations)")
        draw_flow(P_final)
        sub.set_text("iterate row/col normalisation until convergence "
                     "— that's Sinkhorn")
        cap.set_text(f"\u03b5 = {EPS}   \u00b7   "
                     "fully differentiable at every step")

    anim = animation.FuncAnimation(fig, update, frames=TOTAL,
                                   interval=1000/fps, blit=False)
    save_animation(anim, out_path, fps=fps)
    plt.close(fig)
    print(f"wrote {out_path}  ({TOTAL} frames, {fps} fps)")
