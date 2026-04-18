"""02d — Chamfer: optimise B positions (3 slow + fast convergence)."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch
from .helpers import (twins_dataset, pair_colors, b_pair_colors,
                      optimise_B, setup_canvas, save_animation,
                      DEFAULT_FPS, FG, MUTE, GRID, GREY,
                      A_COLOR, B_COLOR, OK_COLOR, BAD_COLOR)


def render(out_path: Path, fps: int = DEFAULT_FPS):
    A, B, gt = twins_dataset()
    N = len(A)
    pcols = pair_colors(N)
    bcols = b_pair_colors(gt, pcols)

    snaps = optimise_B(A, B, method="chamfer", n_steps=40, lr=0.08)

    fig = plt.figure(figsize=(7.6, 8.4), dpi=100)
    gs = fig.add_gridspec(2, 1, height_ratios=[6, 1], hspace=0.18,
                          left=0.04, right=0.96, top=0.84, bottom=0.08)
    ax = fig.add_subplot(gs[0])
    ax_l = fig.add_subplot(gs[1])
    setup_canvas(ax)

    sa = ax.scatter(A[:, 0], A[:, 1], s=130, c=GREY, marker="o",
                    edgecolors="white", linewidths=1.4, zorder=4)
    sb = ax.scatter(B[:, 0], B[:, 1], s=130, c=GREY, marker="D",
                    edgecolors="white", linewidths=1.4, zorder=4)

    # Gradient arrows for slow-step demos.
    grad_arrows = []
    for _ in range(N):
        ar = FancyArrowPatch((0, 0), (0, 0), arrowstyle="-|>",
                             mutation_scale=16, lw=1.8, color=B_COLOR,
                             alpha=0.0, zorder=5, shrinkA=4, shrinkB=4)
        ax.add_patch(ar)
        grad_arrows.append(ar)

    # Matching lines (shown during fast phase).
    match_lines = []
    for _ in range(N):
        ln, = ax.plot([], [], color=GREY, lw=1.0, alpha=0.0, zorder=2)
        match_lines.append(ln)

    # Loss strip.
    losses = []
    for B_s, m in snaps:
        if m is not None:
            from .helpers import chamfer_matches
            a2b, _, _ = chamfer_matches(A, B_s)
            D = np.linalg.norm(A - B_s[a2b], axis=1)
            losses.append(float((D**2).sum()))
        else:
            losses.append(None)
    # Fill first entry.
    from .helpers import chamfer_matches as cm
    a2b0, _, _ = cm(A, snaps[0][0])
    D0 = np.linalg.norm(A - snaps[0][0][a2b0], axis=1)
    losses[0] = float((D0**2).sum())

    ax_l.set_xlim(0, len(snaps)-1)
    ax_l.set_ylim(0, max(l for l in losses if l) * 1.05)
    ax_l.set_xticks([]); ax_l.tick_params(labelsize=9)
    for s in ("top", "right"): ax_l.spines[s].set_visible(False)
    ax_l.set_ylabel("Chamfer loss", fontsize=10)
    loss_line, = ax_l.plot([], [], lw=2.2, color=A_COLOR)
    loss_dot = ax_l.scatter([], [], s=50, c=A_COLOR, edgecolors="white",
                            lw=1.2, zorder=5)

    fig.text(0.5, 0.96, "Chamfer loss \u2192 optimise B positions",
             ha="center", va="top", fontsize=15, fontweight="bold")
    sub = fig.text(0.5, 0.92, "", ha="center", va="top",
                   fontsize=11, color=MUTE)
    cap = fig.text(0.5, 0.025, "", ha="center", va="bottom",
                   fontsize=11, color=FG)

    # Legend — updates between training and reveal phases.
    legend_text = fig.text(
        0.05, 0.86, "", ha="left", va="top", fontsize=9,
        color=FG, linespacing=1.6,
        bbox=dict(boxstyle="round,pad=0.42", fc="white", ec=GRID, lw=0.9))
    LEGEND_TRAIN = (
        "\u25cf  grey = A  (fixed)\n"
        "\u25c6  grey = B  (moves)\n"
        "\u2192  arrow = gradient\n"
        "\u2500  line = current match"
    )
    LEGEND_REVEAL = (
        "\u25cf  colour = A pair identity\n"
        "\u25c6  green = correct twin\n"
        "\u25c6  red = wrong twin\n"
        "\u25c6  grey = orphaned"
    )

    SLOW_STEPS = 3
    SLOW_HOLD = 50       # ~3.6 s per slow step
    FAST_PER = 2         # 2 frames per remaining step
    n_fast = len(snaps) - 1 - SLOW_STEPS
    REVEAL = 40          # reveal pair colours
    HOLD = 30
    TOTAL = SLOW_STEPS * SLOW_HOLD + n_fast * FAST_PER + REVEAL + HOLD

    def snap_at(step):
        return min(step, len(snaps) - 1)

    def draw_B(step):
        B_cur = snaps[snap_at(step)][0]
        sb.set_offsets(B_cur)

    def draw_loss(step):
        k = snap_at(step)
        xs = np.arange(k + 1)
        ys = [losses[i] for i in range(k + 1)]
        loss_line.set_data(xs, ys)
        loss_dot.set_offsets([[k, losses[k]]])

    def hide_arrows():
        for ar in grad_arrows: ar.set_alpha(0.0)

    def show_grad(step):
        """Show gradient arrows: each B → direction it will move."""
        B_cur, matching = snaps[snap_at(step)]
        if matching is None: return
        B_nxt = snaps[snap_at(step + 1)][0]
        for j in range(N):
            d = B_nxt[j] - B_cur[j]
            mag = np.linalg.norm(d)
            if mag > 0.01:
                grad_arrows[j].set_positions(B_cur[j],
                                             B_cur[j] + d * 3)
                grad_arrows[j].set_alpha(0.85)
            else:
                grad_arrows[j].set_alpha(0.15)

    def update(f):
        hide_arrows()
        for ln in match_lines: ln.set_alpha(0.0)

        # --- Slow steps ---
        if f < SLOW_STEPS * SLOW_HOLD:
            step = f // SLOW_HOLD       # which slow step (0, 1, 2)
            t_in = f % SLOW_HOLD
            draw_B(step)
            draw_loss(step)
            show_grad(step)
            legend_text.set_text(LEGEND_TRAIN)
            sub.set_text(f"step {step}: gradient arrows show where "
                         "each diamond will move")
            cap.set_text(f"Chamfer loss = {losses[snap_at(step)]:.2f}")
            return

        # --- Fast steps ---
        f2 = f - SLOW_STEPS * SLOW_HOLD
        if f2 < n_fast * FAST_PER:
            step = SLOW_STEPS + f2 // FAST_PER
            draw_B(step)
            draw_loss(step)
            # Draw current matching lines.
            _, matching = snaps[snap_at(step)]
            if matching is not None:
                B_cur = snaps[snap_at(step)][0]
                for i in range(N):
                    match_lines[i].set_data([A[i, 0], B_cur[matching[i], 0]],
                                            [A[i, 1], B_cur[matching[i], 1]])
                    match_lines[i].set_alpha(0.4)
            legend_text.set_text(LEGEND_TRAIN)
            sub.set_text(f"optimising (step {step})")
            cap.set_text(f"loss = {losses[snap_at(step)]:.2f}")
            return

        # --- Reveal pair colours ---
        final_step = len(snaps) - 1
        draw_B(final_step)
        draw_loss(final_step)
        f3 = f - SLOW_STEPS * SLOW_HOLD - n_fast * FAST_PER
        if f3 < REVEAL:
            t = (f3 + 1) / REVEAL
            legend_text.set_text(LEGEND_REVEAL)
            sa.set_facecolor([pcols[i] for i in range(N)])
            _, matching = snaps[final_step]
            b_show = []
            for j in range(N):
                claimers = [i for i in range(N) if matching is not None
                            and matching[i] == j]
                if claimers:
                    if any(gt[c] == j for c in claimers):
                        b_show.append(OK_COLOR)
                    else:
                        b_show.append(BAD_COLOR)
                else:
                    b_show.append(GREY)
            sb.set_facecolor(b_show)
            sub.set_text("verdict: did each diamond reach its true twin?")
            cap.set_text(""); return

        # --- Hold ---
        legend_text.set_text(LEGEND_REVEAL)
        sa.set_facecolor(pcols)
        sub.set_text("Chamfer optimisation: some B\u2019s land at wrong A")
        B_final, matching = snaps[final_step]
        n_ok = sum(1 for i in range(N) if matching is not None
                   and gt[i] == matching[i])
        cap.set_text(f"correct convergence: {n_ok}/{N}")

    anim = animation.FuncAnimation(fig, update, frames=TOTAL,
                                   interval=1000/fps, blit=False)
    save_animation(anim, out_path, fps=fps)
    plt.close(fig)
    print(f"wrote {out_path}  ({TOTAL} frames, {fps} fps)")
