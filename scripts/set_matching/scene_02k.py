"""02k — Sinkhorn: full dataset + optimise B positions."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from .helpers import (twins_dataset, pair_colors, b_pair_colors,
                      optimise_B, setup_canvas, save_animation,
                      DEFAULT_FPS, FG, MUTE, GRID, GREY, B_COLOR)


def render(out_path: Path, fps: int = DEFAULT_FPS):
    A, B, gt = twins_dataset()
    N = len(A)
    pcols = pair_colors(N)

    snaps = optimise_B(A, B, method="sinkhorn", n_steps=40,
                       lr=0.08, eps_sink=0.15)

    fig = plt.figure(figsize=(7.6, 8.0), dpi=100)
    ax = fig.add_axes([0.04, 0.06, 0.92, 0.78])
    setup_canvas(ax)

    sa = ax.scatter(A[:, 0], A[:, 1], s=130, c=[GREY]*N, marker="o",
                    edgecolors="white", linewidths=1.4, zorder=4)
    sb = ax.scatter(B[:, 0], B[:, 1], s=130, c=[GREY]*N, marker="D",
                    edgecolors="white", linewidths=1.4, zorder=4)

    match_lines = []
    for _ in range(N):
        ln, = ax.plot([], [], color=B_COLOR, lw=1.2, alpha=0.0, zorder=2)
        match_lines.append(ln)

    fig.text(0.5, 0.96,
             "Sinkhorn loss \u2192 optimise B positions",
             ha="center", va="top", fontsize=15, fontweight="bold")
    sub = fig.text(0.5, 0.92, "", ha="center", va="top",
                   fontsize=11, color=MUTE)
    cap = fig.text(0.5, 0.025, "", ha="center", va="bottom",
                   fontsize=11, color=FG)

    SLOW_STEPS = 3
    SLOW_HOLD = 50
    n_fast = len(snaps) - 1 - SLOW_STEPS
    FAST_PER = 2
    REVEAL = 40
    HOLD = 30
    TOTAL = SLOW_STEPS * SLOW_HOLD + n_fast * FAST_PER + REVEAL + HOLD

    def snap_at(s): return min(s, len(snaps) - 1)

    def draw_state(step):
        B_cur, matching = snaps[snap_at(step)]
        sb.set_offsets(B_cur)
        if matching is not None:
            for i in range(N):
                match_lines[i].set_data([A[i, 0], B_cur[matching[i], 0]],
                                        [A[i, 1], B_cur[matching[i], 1]])
                match_lines[i].set_alpha(0.45)
        else:
            for ln in match_lines: ln.set_alpha(0.0)

    def update(f):
        if f < SLOW_STEPS * SLOW_HOLD:
            step = f // SLOW_HOLD
            draw_state(step)
            sub.set_text(f"step {step}: Sinkhorn plan \u2192 "
                         "smooth gradient on every B")
            cap.set_text(f"step {step}  \u00b7  fully differentiable")
            return

        f2 = f - SLOW_STEPS * SLOW_HOLD
        if f2 < n_fast * FAST_PER:
            step = SLOW_STEPS + f2 // FAST_PER
            draw_state(step)
            sub.set_text(f"optimising (step {step})")
            cap.set_text(""); return

        final = len(snaps) - 1
        draw_state(final)
        f3 = f - SLOW_STEPS * SLOW_HOLD - n_fast * FAST_PER
        if f3 < REVEAL:
            t = (f3 + 1) / REVEAL
            sa.set_facecolor(pcols)
            B_final, matching = snaps[final]
            b_show = []
            for j in range(N):
                claimed_ok = any(matching[i] == j and gt[i] == j
                                 for i in range(N))
                if claimed_ok:
                    b_show.append(np.array([0.18, 0.55, 0.34, t]))
                else:
                    b_show.append(GREY)
            sb.set_facecolor(b_show)
            sub.set_text("restoring pair colours \u2026")
            cap.set_text(""); return

        sa.set_facecolor(pcols)
        _, matching = snaps[final]
        n_ok = sum(1 for i in range(N) if matching is not None
                   and gt[i] == matching[i])
        sub.set_text("Sinkhorn: soft, differentiable \u2192 "
                     "smooth convergence")
        cap.set_text(f"correct convergence: {n_ok}/{N}")

    anim = animation.FuncAnimation(fig, update, frames=TOTAL,
                                   interval=1000/fps, blit=False)
    save_animation(anim, out_path, fps=fps)
    plt.close(fig)
    print(f"wrote {out_path}  ({TOTAL} frames, {fps} fps)")
