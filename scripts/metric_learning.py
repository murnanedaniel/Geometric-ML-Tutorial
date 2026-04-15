"""
Metric-learning visualizations for the Geometric ML tutorial / seminar.

Generates a family of GIFs explaining pairwise contrastive hinge loss,
built around a 3-arm "spiral galaxy" dataset.

    python scripts/metric_learning.py dataset     -> 01a_dataset.gif
    python scripts/metric_learning.py loss        -> 01b_pairwise_loss.gif
    python scripts/metric_learning.py mining      -> 01d_hard_negatives.gif
    python scripts/metric_learning.py training    -> 01e_training.gif
    python scripts/metric_learning.py knn         -> 01f_learned_knn.gif
    python scripts/metric_learning.py all         -> all of the above
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from matplotlib.patches import Circle
from PIL import Image

# --------------------------------------------------------------------------
# Global animation settings
# --------------------------------------------------------------------------

# ~50% slower than a standard 24 fps gif, and every gif in this file plays
# through exactly once (no loop).
DEFAULT_FPS = 14


class PillowWriterNoLoop(PillowWriter):
    """PillowWriter that writes a GIF with no Netscape loop extension.

    matplotlib's PillowWriter hard-codes `loop=0` (infinite loop) when it
    dumps the GIF. We override `finish` to omit the loop keyword entirely,
    which produces a GIF that plays through exactly once in every major
    decoder (browsers, GitHub, Quick Look).
    """

    def finish(self):
        self._frames[0].save(
            self.outfile,
            save_all=True,
            append_images=self._frames[1:],
            duration=int(1000 / self.fps),
            # deliberately no `loop` kwarg: no Netscape extension is written
        )


def save_animation(anim, out_path: Path, fps: int = DEFAULT_FPS):
    """Save a matplotlib FuncAnimation as a single-play (non-looping) GIF."""
    anim.save(out_path, writer=PillowWriterNoLoop(fps=fps))

# --------------------------------------------------------------------------
# Style
# --------------------------------------------------------------------------

CLASS_COLORS = {
    0: "#E85D4A",   # ember
    1: "#2A9D8F",   # teal
    2: "#6E63B5",   # violet
}
BG   = "#FAFAF7"
FG   = "#2C2C2C"
MUTE = "#8A8F9B"
GRID = "#E6E3DC"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   BG,
    "savefig.facecolor": BG,
    "axes.edgecolor":   FG,
    "axes.labelcolor":  FG,
    "text.color":       FG,
    "xtick.color":      FG,
    "ytick.color":      FG,
    "font.family":      "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "grid.color":       GRID,
    "grid.linewidth":   0.8,
})

OUT_DIR = Path(__file__).resolve().parent.parent / "assets" / "metric_learning"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------

def spiral_galaxy(n_per_arm: int = 180,
                  n_arms: int = 3,
                  noise: float = 0.09,
                  turns: float = 1.9,
                  seed: int = 0):
    """Three interleaved spiral arms, labelled 0/1/2."""
    rng = np.random.default_rng(seed)
    X_parts, y_parts = [], []
    for k in range(n_arms):
        t = np.linspace(0.06, 1.0, n_per_arm)
        r = t * 3.0
        theta = turns * 2 * np.pi * t + k * 2 * np.pi / n_arms
        # Noise grows a little with radius for a galactic look.
        jitter = noise * (0.5 + 1.1 * t)
        x = r * np.cos(theta) + rng.normal(0, 1, n_per_arm) * jitter
        y = r * np.sin(theta) + rng.normal(0, 1, n_per_arm) * jitter
        X_parts.append(np.stack([x, y], axis=1))
        y_parts.append(np.full(n_per_arm, k))
    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    return X, y


# --------------------------------------------------------------------------
# 01a — the dataset
# --------------------------------------------------------------------------

def make_dataset_gif(out_path: Path,
                     fps: int = DEFAULT_FPS,
                     size: int = 720):
    X, y = spiral_galaxy(n_per_arm=220, seed=0)

    # Draw in order of radius (inside-out) for an "emerging spiral" effect.
    r_all = np.linalg.norm(X, axis=1)
    order = np.argsort(r_all)
    X_o, y_o = X[order], y[order]
    N = len(X_o)

    # Pick three "adversarial" query points (one per class) — points where
    # an epsilon-ball of radius RADIUS contains many wrong-class neighbours.
    RADIUS_FOR_PICK = 0.95
    query_idxs = []
    for k in range(3):
        mask = (y == k)
        idx_k = np.where(mask)[0]
        # only consider mid/outer radii so we have enough neighbours
        keep = idx_k[(r_all[idx_k] > 1.4) & (r_all[idx_k] < 2.7)]
        # score = number of wrong-class points within RADIUS
        scores = []
        for i in keep:
            d = np.linalg.norm(X - X[i], axis=1)
            in_ball = (d <= RADIUS_FOR_PICK) & (d > 1e-6)
            wrong = in_ball & (y != k)
            scores.append(wrong.sum())
        # pick a top-scoring point (with a touch of randomness)
        order_ = np.argsort(scores)[::-1][:6]
        rng = np.random.default_rng(11 + k)
        query_idxs.append(int(keep[order_[rng.integers(0, len(order_))]]))

    # Frame layout
    DRAW   = 110              # spiral draws itself
    HOLD1  = 20               # brief pause
    KNN    = 45               # per query
    FINAL  = 40               # final hold on caption
    total  = DRAW + HOLD1 + len(query_idxs) * KNN + FINAL

    fig = plt.figure(figsize=(size / 100, size / 100), dpi=100)
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    ax.set_xlim(-3.9, 3.9); ax.set_ylim(-3.9, 3.9)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ("left", "bottom", "top", "right"):
        ax.spines[s].set_visible(False)

    scat = ax.scatter([], [], s=22, linewidths=0.5, edgecolors="white", zorder=3)

    # Query-time overlay artists.
    knn_circle = Circle((0, 0), 0.001, fill=False, lw=1.8, ls="--",
                        color=FG, alpha=0.0, zorder=5)
    ax.add_patch(knn_circle)
    # Query: large filled dot in the class colour with a white halo.
    query_halo = ax.scatter([], [], s=360, facecolors="white",
                            edgecolors="none", zorder=6, alpha=0.0)
    query_dot = ax.scatter([], [], s=180, facecolors="white",
                           edgecolors=FG, linewidths=2.6, zorder=7)
    # Wrong-class neighbours get a red X, right-class get a subtle ring.
    right_ring = ax.scatter([], [], s=110, facecolors="none",
                            edgecolors=FG, linewidths=1.2, zorder=5, alpha=0.55)
    wrong_mark = ax.scatter([], [], s=150, marker="x",
                            color="#C62828", linewidths=2.2, zorder=6)

    title = ax.text(0.5, 0.97, "A spiral galaxy",
                    transform=ax.transAxes, ha="center", va="top",
                    fontsize=17, fontweight="bold")
    subtitle = ax.text(0.5, 0.935,
                       "three arms • three classes • nonlinearly separable",
                       transform=ax.transAxes, ha="center", va="top",
                       fontsize=11, color=MUTE)
    caption = ax.text(0.5, 0.04, "", transform=ax.transAxes,
                      ha="center", va="bottom", fontsize=12,
                      bbox=dict(boxstyle="round,pad=0.45",
                                fc="white", ec=GRID, lw=1))

    def set_caption(txt):
        caption.set_text(txt)

    def draw_prefix(end: int):
        pts  = X_o[:end]
        cols = [CLASS_COLORS[c] for c in y_o[:end]]
        scat.set_offsets(pts if len(pts) else np.empty((0, 2)))
        scat.set_color(cols if len(cols) else [])

    def clear_knn():
        knn_circle.set_alpha(0.0)
        query_dot.set_offsets(np.empty((0, 2)))
        query_halo.set_offsets(np.empty((0, 2)))
        query_halo.set_alpha(0.0)
        right_ring.set_offsets(np.empty((0, 2)))
        wrong_mark.set_offsets(np.empty((0, 2)))

    RADIUS = 0.95

    def show_query(qi, grow=1.0):
        qx, qy = X[qi]
        col = CLASS_COLORS[y[qi]]
        query_halo.set_offsets([[qx, qy]])
        query_halo.set_alpha(0.95)
        query_dot.set_offsets([[qx, qy]])
        query_dot.set_facecolors(col)
        query_dot.set_edgecolors("white")
        knn_circle.center = (qx, qy)
        knn_circle.set_radius(max(RADIUS * grow, 1e-3))
        knn_circle.set_alpha(0.9)
        d = np.linalg.norm(X - np.array([qx, qy]), axis=1)
        mask = (d <= RADIUS * grow) & (d > 1e-6)
        same_mask = mask & (y == y[qi])
        diff_mask = mask & (y != y[qi])
        right_ring.set_offsets(X[same_mask] if same_mask.any() else np.empty((0, 2)))
        wrong_mark.set_offsets(X[diff_mask] if diff_mask.any() else np.empty((0, 2)))
        return same_mask.sum(), diff_mask.sum()

    def update(f):
        # Phase 1: draw spiral.
        if f < DRAW:
            frac = (f + 1) / DRAW
            frac = 1 - (1 - frac) ** 3
            end = int(frac * N)
            draw_prefix(end)
            clear_knn()
            set_caption("")
            return
        if f < DRAW + HOLD1:
            draw_prefix(N)
            clear_knn()
            set_caption("")
            return

        # Phase 2: k-NN demo, one query at a time.
        draw_prefix(N)
        fk = f - DRAW - HOLD1
        if fk >= KNN * len(query_idxs):
            # Final hold on the last query.
            qi = query_idxs[-1]
            s, d = show_query(qi, grow=1.0)
            tot = s + d
            set_caption(
                r"In raw $\mathbb{R}^2$, nearest $\neq$ same class."
                "\nWe need to learn a metric."
            )
            return

        slot = fk // KNN
        t_in = fk % KNN
        qi = query_idxs[slot]
        grow = min(1.0, t_in / 16.0)
        s, d = show_query(qi, grow=grow)
        tot = s + d
        if tot > 0:
            set_caption(
                f"query on arm {y[qi]}   •   "
                f"{s} same-class  •  {d} wrong-class in the $\\epsilon$-ball"
            )
        else:
            set_caption(f"query on arm {y[qi]}")

    anim = animation.FuncAnimation(
        fig, update, frames=total, interval=1000 / fps, blit=False
    )
    save_animation(anim, out_path, fps=fps)
    plt.close(fig)
    print(f"wrote {out_path}  ({total} frames, {fps} fps, no loop)")


# --------------------------------------------------------------------------
# placeholder stubs — filled in after 01a is approved
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# 01b — the pairwise contrastive hinge loss, two chapters
# --------------------------------------------------------------------------

def _ease_inout(t):
    """Smooth 0→1 ease (cubic)."""
    return 3 * t ** 2 - 2 * t ** 3


def make_loss_gif(out_path: Path,
                  fps: int = DEFAULT_FPS,
                  size: int = 720):
    from matplotlib.patches import FancyArrowPatch

    M = 2.0  # margin (chapter 2 only)

    BLUE_GRAD = "#2563EB"   # attractive gradient (pull)
    RED_GRAD  = "#DC2626"   # repulsive gradient (push)

    # ----------------------------------------------------------------------
    # Layout: wide 2-D canvas on top, loss strip below.
    # ----------------------------------------------------------------------
    fig = plt.figure(figsize=(8.4, 7.2), dpi=100)
    gs = fig.add_gridspec(
        2, 1, height_ratios=[3.4, 1.0],
        hspace=0.20, left=0.05, right=0.97, top=0.83, bottom=0.08,
    )
    ax   = fig.add_subplot(gs[0])
    ax_l = fig.add_subplot(gs[1])

    # Main canvas — tight framing around the action.
    ax.set_xlim(-3.1, 3.3); ax.set_ylim(-2.1, 2.1)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ("left", "right", "top", "bottom"):
        ax.spines[s].set_visible(False)

    # Margin disk (chapter 2).
    margin_disk = Circle((0, 0), M, color="#E85D4A", alpha=0.0, zorder=0)
    margin_edge = Circle((0, 0), M, fill=False, color="#E85D4A",
                         lw=1.6, ls="--", alpha=0.0, zorder=1)
    ax.add_patch(margin_disk); ax.add_patch(margin_edge)
    margin_label = ax.text(0, M + 0.08, "", color="#B83A2B",
                           fontsize=10, ha="center", va="bottom", alpha=0.0)

    # Anchor — white outer halo so it stays visible over the margin fill.
    ax.scatter([0], [0], s=520, c="white", edgecolors="none", zorder=4)
    anchor_dot = ax.scatter([0], [0], s=320, c=CLASS_COLORS[1],
                            edgecolors="white", linewidths=2.4, zorder=5)
    ax.text(0, -0.42, "anchor", ha="center", va="top",
            fontsize=10, color=FG, zorder=5)

    # Partner.
    partner_dot = ax.scatter([], [], s=280, c="white",
                             edgecolors="white", linewidths=2.4, zorder=5)

    # Distance line + live "d = ..." badge (placed perpendicular to the line).
    dist_line, = ax.plot([], [], ls="--", lw=1.3, color=MUTE, zorder=3)
    dist_text = ax.text(0, 0, "", fontsize=10, color=FG, ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.28",
                                  fc="white", ec=GRID, lw=0.8), zorder=6)

    # Gradient arrow (attached to partner; points along −∇L).
    grad_arrow = FancyArrowPatch((0, 0), (0, 0),
                                 arrowstyle="-|>", mutation_scale=26,
                                 lw=3.2, color=BLUE_GRAD, zorder=6, alpha=0.0)
    ax.add_patch(grad_arrow)

    # "L = 0" satisfied badge (chapter 2, once outside the margin).
    # Top-centre of the canvas, out of the partner's path.
    sat_badge = ax.text(0.5, 0.96, "", transform=ax.transAxes,
                        ha="center", va="top", fontsize=12,
                        color="#1A7F74", fontweight="bold", alpha=0.0,
                        bbox=dict(boxstyle="round,pad=0.35",
                                  fc="#E8F5F2", ec="#1A7F74", lw=1.1))

    # One-line arrow caption under the chapter banner — recolours per chapter.
    arrow_caption = ax.text(0.5, 0.96, "", transform=ax.transAxes,
                            ha="center", va="top", fontsize=10.5,
                            color=BLUE_GRAD, fontweight="bold", alpha=0.9)

    # Headers — title above, formula middle, chapter below.
    fig.text(0.5, 0.965, "Pairwise contrastive hinge loss",
             ha="center", va="top", fontsize=15, fontweight="bold")
    formula = fig.text(0.5, 0.915, "",
                       ha="center", va="top", fontsize=14)
    chapter = fig.text(0.5, 0.875, "",
                       ha="center", va="top", fontsize=11, color=MUTE)

    # Loss strip.
    ax_l.set_xlim(0, 1); ax_l.set_ylim(0, 13)
    ax_l.set_xticks([])
    ax_l.tick_params(axis="y", labelsize=9)
    for s in ("top", "right"):
        ax_l.spines[s].set_visible(False)
    ax_l.set_ylabel("loss", fontsize=10)
    loss_line, = ax_l.plot([], [], lw=2.6, color=CLASS_COLORS[1])
    loss_dot = ax_l.scatter([], [], s=70, c=CLASS_COLORS[1],
                            edgecolors="white", lw=1.6, zorder=5)
    strip_note = ax_l.text(0.01, 0.92, "", transform=ax_l.transAxes,
                           ha="left", va="top", fontsize=10, color=FG)

    # ----------------------------------------------------------------------
    # Timing: two chapters + short cross-fade + final hold.
    # ----------------------------------------------------------------------
    T1     = 110    # chapter 1 frames
    TRANS  = 18     # cross-fade between chapters
    T2     = 110    # chapter 2 frames
    HOLD   = 34     # final hold on "L=0, gradient vanishes"
    TOTAL  = T1 + TRANS + T2 + HOLD

    # Partner trajectories.
    def ch1_pos(t):
        e = _ease_inout(t)
        r = 0.45 + 2.85 * e
        theta = np.deg2rad(28) + 0.22 * np.sin(np.pi * e)
        return np.array([r * np.cos(theta), r * np.sin(theta)])

    def ch2_pos(t):
        e = _ease_inout(t)
        r = 0.40 + 3.05 * e
        theta = np.deg2rad(-32) + 0.18 * np.sin(np.pi * e)
        return np.array([r * np.cos(theta), r * np.sin(theta)])

    ch1_p = np.array([ch1_pos(t) for t in np.linspace(0, 1, T1)])
    ch1_d = np.linalg.norm(ch1_p, axis=1)
    ch1_L = ch1_d ** 2

    ch2_p = np.array([ch2_pos(t) for t in np.linspace(0, 1, T2)])
    ch2_d = np.linalg.norm(ch2_p, axis=1)
    ch2_L = np.maximum(0.0, M - ch2_d) ** 2

    def _set_distance_badge(pos):
        mid = pos * 0.5
        d = np.linalg.norm(pos)
        dist_line.set_data([0, pos[0]], [0, pos[1]])
        # push badge perpendicular to the line by a fixed amount
        perp = np.array([-pos[1], pos[0]])
        perp = perp / (np.linalg.norm(perp) + 1e-9) * 0.32
        dist_text.set_position((mid[0] + perp[0], mid[1] + perp[1]))
        dist_text.set_text(f"d = {d:.2f}")

    def _draw_gradient_arrow(pos, direction_unit, magnitude, color):
        """Draw an arrow of visible length, growing with gradient magnitude."""
        if magnitude <= 1e-4:
            grad_arrow.set_alpha(0.0)
            return
        # Keep arrow visually readable: min length 0.35, max length ~1.2.
        L = min(0.35 + 0.28 * magnitude, 1.3)
        # Tail starts just outside the partner dot.
        start = pos + 0.14 * direction_unit
        head = pos + L * direction_unit
        grad_arrow.set_positions(start, head)
        grad_arrow.set_color(color)
        grad_arrow.set_alpha(0.95)

    def update(f):
        # ---------------- Chapter 1: same class (attractive) ----------------
        if f < T1:
            k   = f
            pos = ch1_p[k]
            d   = ch1_d[k]
            L   = ch1_L[k]
            col = CLASS_COLORS[1]  # teal

            chapter.set_text("chapter 1 — same class (attractive branch)")
            formula.set_text(rf"$L^+ \;=\; d(a,p)^2 \;=\; {L:.2f}$")
            arrow_caption.set_text("blue arrow  =  $-\\nabla L^+$  (pulls partner toward anchor)")
            arrow_caption.set_color(BLUE_GRAD)
            arrow_caption.set_alpha(0.95)

            # partner
            partner_dot.set_offsets([pos])
            partner_dot.set_facecolors(col)
            partner_dot.set_edgecolors("white")

            _set_distance_badge(pos)

            # gradient: −∇_p L⁺ = −2(p − a)  ⇒  points INWARD
            u = -pos / max(d, 1e-6)
            _draw_gradient_arrow(pos, u, magnitude=2 * d, color=BLUE_GRAD)

            # margin hidden
            margin_disk.set_alpha(0.0); margin_edge.set_alpha(0.0)
            margin_label.set_alpha(0.0)
            sat_badge.set_alpha(0.0)

            # loss strip
            xs = np.linspace(0, 1, T1)[:k + 1]
            loss_line.set_data(xs, ch1_L[:k + 1])
            loss_line.set_color(col)
            loss_dot.set_offsets([[xs[-1], L]])
            loss_dot.set_facecolors(col)
            ax_l.set_ylim(0, 1.1 * max(ch1_L.max(), 1))
            strip_note.set_text("loss grows with distance — no upper cap")
            return

        # ---------------- Cross-fade (chapter 1 → chapter 2) -----------------
        if f < T1 + TRANS:
            s = (f - T1 + 1) / TRANS          # 0→1
            # Fade margin in, fade partner from teal→ember, keep last pos of ch1.
            pos = ch1_p[-1] * (1 - s) + ch2_p[0] * s
            col = np.array([
                np.array(_hex_to_rgb(CLASS_COLORS[1])) * (1 - s)
                + np.array(_hex_to_rgb(CLASS_COLORS[0])) * s
            ]) / 255.0

            chapter.set_text("switching the pair's class label…")
            formula.set_text("")
            arrow_caption.set_alpha(0.0)

            partner_dot.set_offsets([pos])
            partner_dot.set_facecolors(col)
            partner_dot.set_edgecolors("white")

            _set_distance_badge(pos)

            # no gradient during transition
            grad_arrow.set_alpha(0.0)

            # margin fades in
            margin_disk.set_alpha(0.15 * s)
            margin_edge.set_alpha(0.85 * s)
            margin_label.set_alpha(s)
            margin_label.set_text(f"margin  m = {M:.1f}")
            sat_badge.set_alpha(0.0)

            # strip: fade chapter-1 curve, prepare for chapter 2
            loss_line.set_alpha(1 - s)
            loss_dot.set_alpha(1 - s)
            strip_note.set_text("")
            return

        loss_line.set_alpha(1.0)
        loss_dot.set_alpha(1.0)

        # ---------------- Chapter 2: different class (repulsive) -------------
        k_abs = f - T1 - TRANS
        if k_abs < T2:
            k   = k_abs
            pos = ch2_p[k]
            d   = ch2_d[k]
            L   = ch2_L[k]
            col = CLASS_COLORS[0]  # ember

            chapter.set_text("chapter 2 — different class (repulsive hinge)")
            formula.set_text(
                rf"$L^- \;=\; \max(0,\; m - d)^2 \;=\; {L:.2f}$"
            )

            partner_dot.set_offsets([pos])
            partner_dot.set_facecolors(col)
            partner_dot.set_edgecolors("white")

            _set_distance_badge(pos)

            # gradient only active inside the margin
            if d < M:
                u = pos / max(d, 1e-6)  # outward unit
                _draw_gradient_arrow(pos, u, magnitude=2 * (M - d),
                                     color=RED_GRAD)
                arrow_caption.set_text(
                    "red arrow  =  $-\\nabla L^-$  (pushes partner past the margin)"
                )
                arrow_caption.set_color(RED_GRAD)
                arrow_caption.set_alpha(0.95)
                sat_badge.set_alpha(0.0)
            else:
                grad_arrow.set_alpha(0.0)
                arrow_caption.set_alpha(0.0)
                sat_badge.set_alpha(0.95)
                sat_badge.set_text(r"$L^- = 0$  ·  gradient vanishes")

            margin_disk.set_alpha(0.15); margin_edge.set_alpha(0.85)
            margin_label.set_alpha(1.0)
            margin_label.set_text(f"margin  m = {M:.1f}")

            # loss strip (fresh for this chapter)
            xs = np.linspace(0, 1, T2)[:k + 1]
            loss_line.set_data(xs, ch2_L[:k + 1])
            loss_line.set_color(col)
            loss_dot.set_offsets([[xs[-1], L]])
            loss_dot.set_facecolors(col)
            ax_l.set_ylim(0, 1.1 * max(ch2_L.max(), 0.1))
            if d < M:
                strip_note.set_text(
                    "inside margin → loss is positive, gradient pushes outward"
                )
            else:
                strip_note.set_text(
                    "past the margin → loss pinned to 0, no gradient"
                )
            return

        # ---------------- Final hold ----------------------------------------
        # Freeze chapter-2 final state.
        k = T2 - 1
        pos = ch2_p[k]; d = ch2_d[k]; L = ch2_L[k]
        partner_dot.set_offsets([pos])
        partner_dot.set_facecolors(CLASS_COLORS[0])
        partner_dot.set_edgecolors("white")
        _set_distance_badge(pos)
        grad_arrow.set_alpha(0.0)
        margin_disk.set_alpha(0.15); margin_edge.set_alpha(0.85)
        margin_label.set_alpha(1.0)
        margin_label.set_text(f"margin  m = {M:.1f}")
        sat_badge.set_alpha(0.95)
        sat_badge.set_text(r"$L^- = 0$  ·  gradient vanishes")
        chapter.set_text("chapter 2 — different class (repulsive hinge)")
        formula.set_text(rf"$L^- \;=\; \max(0,\; m - d)^2 \;=\; {L:.2f}$")
        arrow_caption.set_alpha(0.0)
        strip_note.set_text("past the margin → loss pinned to 0, no gradient")

    anim = animation.FuncAnimation(
        fig, update, frames=TOTAL, interval=1000 / fps, blit=False
    )
    save_animation(anim, out_path, fps=fps)
    plt.close(fig)
    print(f"wrote {out_path}  ({TOTAL} frames, {fps} fps, no loop)")


def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
# --------------------------------------------------------------------------
# 01d — hard negative mining
# --------------------------------------------------------------------------

def make_mining_gif(out_path: Path,
                    fps: int = DEFAULT_FPS,
                    size: int = 720):
    from matplotlib.patches import FancyArrowPatch

    M = 2.0
    LR = 0.45           # gradient step size (visual)

    # ----------------------------------------------------------------------
    # Scene: anchor at origin, 24 random negatives.  We tilt the random
    # draw so ~10 land inside the margin (hard) and ~14 outside (easy).
    # ----------------------------------------------------------------------
    rng = np.random.default_rng(7)
    N_NEG = 24
    pos = rng.uniform(-3.2, 3.2, size=(N_NEG, 2))
    d0 = np.linalg.norm(pos, axis=1)
    order = np.argsort(d0)
    HARD_N = 10
    for i in range(HARD_N):
        idx = order[i]
        r_t = rng.uniform(0.55, 1.9)
        pos[idx] *= r_t / max(d0[idx], 1e-6)
    for i in range(HARD_N, N_NEG):
        idx = order[i]
        r_t = rng.uniform(2.15, 3.25)
        pos[idx] *= r_t / max(d0[idx], 1e-6)
    d = np.linalg.norm(pos, axis=1)
    L = np.maximum(0.0, M - d) ** 2

    # Pick the top-K "hardest" (highest loss) to mine.
    K = 3
    topk = np.argsort(-L)[:K]
    hard_mask = d < M

    # ----------------------------------------------------------------------
    # Figure layout — big square 2-D canvas on top, thin loss-bar strip.
    # ----------------------------------------------------------------------
    fig = plt.figure(figsize=(7.8, 8.6), dpi=100)
    gs = fig.add_gridspec(
        2, 1, height_ratios=[5.2, 1.2],
        hspace=0.22, left=0.06, right=0.97, top=0.85, bottom=0.07,
    )
    ax = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])

    ax.set_xlim(-3.6, 3.6); ax.set_ylim(-3.6, 3.6)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ("left", "right", "top", "bottom"):
        ax.spines[s].set_visible(False)

    # Margin disk + ring.
    margin_disk = Circle((0, 0), M, color="#E85D4A", alpha=0.0, zorder=0)
    margin_ring = Circle((0, 0), M, fill=False, color="#E85D4A",
                         lw=1.6, ls="--", alpha=0.0, zorder=1)
    ax.add_patch(margin_disk); ax.add_patch(margin_ring)
    margin_label = ax.text(0, M + 0.12, "margin  m = 2.0",
                           color="#B83A2B", fontsize=10,
                           ha="center", va="bottom", alpha=0.0)

    # Anchor — white halo + teal dot.
    ax.scatter([0], [0], s=520, c="white", edgecolors="none", zorder=4)
    ax.scatter([0], [0], s=300, c=CLASS_COLORS[1],
               edgecolors="white", linewidths=2.4, zorder=5)
    ax.text(0, -0.45, "anchor", ha="center", va="top",
            fontsize=10, color=FG, zorder=5)

    # Negative dots (scatter updated each frame).
    scat = ax.scatter(np.zeros(N_NEG), np.zeros(N_NEG),
                      s=np.full(N_NEG, 70.0),
                      c=[CLASS_COLORS[0]] * N_NEG,
                      edgecolors="white",
                      linewidths=1.3, zorder=4)
    # Red selection rings around the top-K mined.
    ring_scat = ax.scatter([], [], s=[], facecolors="none",
                           edgecolors="#DC2626", linewidths=2.8, zorder=5,
                           alpha=0.0)

    # Gradient arrows for the selected hard negatives (only drawn during step).
    grad_arrows = []
    for _ in range(K):
        ga = FancyArrowPatch((0, 0), (0, 0),
                             arrowstyle="-|>", mutation_scale=22,
                             lw=2.5, color="#DC2626", zorder=5, alpha=0.0)
        ax.add_patch(ga)
        grad_arrows.append(ga)

    # Loss bar strip.
    ax_b.set_xlim(-0.5, N_NEG - 0.5)
    ax_b.set_ylim(0, 1.1 * max(L.max(), 0.5))
    ax_b.set_xticks([])
    ax_b.tick_params(labelsize=9)
    for s in ("top", "right"):
        ax_b.spines[s].set_visible(False)
    ax_b.set_ylabel("loss  $L^-$", fontsize=10)
    # We sort negatives by descending loss for the bar chart x-axis.
    sort_idx = np.argsort(-L)
    bar_colors = np.array(["#D0D3D9"] * N_NEG)
    for j in topk:
        bar_colors[j] = "#DC2626"
    # For others that are "hard but not mined", use ember.
    for j in np.where(hard_mask)[0]:
        if j not in topk:
            bar_colors[j] = CLASS_COLORS[0]
    bars = ax_b.bar(np.arange(N_NEG), L[sort_idx],
                    color=bar_colors[sort_idx], edgecolor="white", lw=0.8)
    bar_note = ax_b.text(0.01, 0.92, "", transform=ax_b.transAxes,
                         ha="left", va="top", fontsize=10, color=FG)

    # Headers.
    fig.text(0.5, 0.955, "Hard negative mining",
             ha="center", va="top", fontsize=15, fontweight="bold")
    chapter = fig.text(0.5, 0.905, "",
                       ha="center", va="top", fontsize=11, color=MUTE)
    footer = fig.text(0.5, 0.02, "", ha="center", va="bottom",
                      fontsize=10, color=FG)

    # ----------------------------------------------------------------------
    # Timing.
    # ----------------------------------------------------------------------
    T_SCATTER  = 42     # negatives scatter in
    T_MEASURE  = 34     # loss encoding reveals easy vs hard
    T_MINE     = 34     # spotlight top-K
    T_STEP     = 54     # gradient step animates hards outward
    T_HOLD     = 30
    TOTAL      = T_SCATTER + T_MEASURE + T_MINE + T_STEP + T_HOLD

    # Phase-1 start positions: all scatter in from random points off-canvas.
    start_pos = rng.uniform(-5.0, 5.0, size=(N_NEG, 2))
    # Guarantee they start well outside the visible area.
    start_pos *= 1.8 / np.maximum(np.linalg.norm(start_pos, axis=1, keepdims=True),
                                  1e-6) * 4.5

    # Final post-gradient-step positions for the mined negatives.
    step_end = pos.copy()
    for j in topk:
        u = pos[j] / max(d[j], 1e-6)
        step_end[j] = pos[j] + LR * 2 * (M - d[j]) * u

    # Helpers.
    def loss_at(p):
        d_ = np.linalg.norm(p, axis=1)
        return np.maximum(0.0, M - d_) ** 2, d_

    def dot_styles(L_arr, d_arr, fade_easy=False, select_hard=False):
        """Return sizes and colors for each negative."""
        sizes = np.full(N_NEG, 70.0)
        colors = np.empty(N_NEG, dtype=object)
        for j in range(N_NEG):
            if d_arr[j] < M:            # hard
                sizes[j]  = 90 + 70 * L_arr[j]
                colors[j] = CLASS_COLORS[0]
            else:                       # easy
                sizes[j]  = 60
                colors[j] = "#C8CBD1" if fade_easy else "#8A8F9B"
        if select_hard:
            for j in topk:
                sizes[j] = 130 + 70 * L_arr[j]
                colors[j] = "#DC2626"
        return sizes, colors

    def update(f):
        # ---- Phase 1: scatter in ----
        if f < T_SCATTER:
            s = (f + 1) / T_SCATTER
            e = _ease_inout(s)
            p = start_pos * (1 - e) + pos * e
            L_cur, d_cur = loss_at(p)

            sizes, colors = dot_styles(L_cur, d_cur, fade_easy=False,
                                       select_hard=False)
            scat.set_offsets(p)
            scat.set_sizes(sizes)
            scat.set_facecolor(colors)

            margin_disk.set_alpha(0.12 * e)
            margin_ring.set_alpha(0.85 * e)
            margin_label.set_alpha(e)

            ring_scat.set_alpha(0.0)
            for ga in grad_arrows:
                ga.set_alpha(0.0)

            chapter.set_text("many candidate negatives to choose from…")
            footer.set_text("")
            # Bars follow the scatter-in progress.
            for i, b in enumerate(bars):
                b.set_height(L_cur[sort_idx][i])
                b.set_color("#D0D3D9")
            bar_note.set_text("loss per negative (unsorted, all candidates)")
            return

        # ---- Phase 2: measure loss — easies fade, hards pop ----
        if f < T_SCATTER + T_MEASURE:
            s = (f - T_SCATTER + 1) / T_MEASURE
            e = _ease_inout(s)

            sizes, colors = dot_styles(L, d, fade_easy=False,
                                       select_hard=False)
            # Interpolate easies toward grey.
            for j in range(N_NEG):
                if not hard_mask[j]:
                    # fade from mid-grey to light-grey
                    sizes[j] = 60 - 10 * e
            scat.set_offsets(pos)
            scat.set_sizes(sizes)
            scat.set_facecolor(colors)

            ring_scat.set_alpha(0.0)
            for ga in grad_arrows:
                ga.set_alpha(0.0)

            chapter.set_text("only negatives inside the margin carry gradient")
            footer.set_text(
                f"{hard_mask.sum()} of {N_NEG} candidates are hard  "
                f"(loss > 0).  The rest contribute nothing."
            )

            # Bars: sort and recolor.
            vals_sorted = L[sort_idx]
            colors_sorted = np.array(["#D0D3D9"] * N_NEG, dtype=object)
            for i, j in enumerate(sort_idx):
                if hard_mask[j]:
                    colors_sorted[i] = CLASS_COLORS[0]
            for i, b in enumerate(bars):
                b.set_height(vals_sorted[i])
                b.set_color(colors_sorted[i])
            bar_note.set_text("sorted by loss — easies (grey) give 0 signal")
            return

        # ---- Phase 3: mine the top-K ----
        if f < T_SCATTER + T_MEASURE + T_MINE:
            s = (f - T_SCATTER - T_MEASURE + 1) / T_MINE
            e = _ease_inout(s)

            sizes, colors = dot_styles(L, d, fade_easy=True,
                                       select_hard=True)
            scat.set_offsets(pos)
            scat.set_sizes(sizes)
            scat.set_facecolor(colors)

            ring_scat.set_offsets(pos[topk])
            ring_scat.set_sizes([220 + 40 * np.sin(np.pi * e)] * K)
            ring_scat.set_alpha(0.9 * e)

            for ga in grad_arrows:
                ga.set_alpha(0.0)

            chapter.set_text(f"pick the top-{K} hardest  →  mined negatives")
            footer.set_text(
                "hard-negative mining focuses the gradient on the "
                "few examples that actually drive learning."
            )

            # Bars: top-K switch to red.
            for i, j in enumerate(sort_idx):
                if j in topk:
                    bars[i].set_color("#DC2626")
                elif hard_mask[j]:
                    bars[i].set_color(CLASS_COLORS[0])
                else:
                    bars[i].set_color("#D0D3D9")
            bar_note.set_text(f"top-{K} mined (red)")
            return

        # ---- Phase 4: gradient step — mined move outward ----
        if f < T_SCATTER + T_MEASURE + T_MINE + T_STEP:
            s = (f - T_SCATTER - T_MEASURE - T_MINE + 1) / T_STEP
            e = _ease_inout(s)
            p = pos.copy()
            for j in topk:
                p[j] = pos[j] + e * (step_end[j] - pos[j])
            L_cur, d_cur = loss_at(p)

            sizes, colors = dot_styles(L_cur, d_cur, fade_easy=True,
                                       select_hard=True)
            scat.set_offsets(p)
            scat.set_sizes(sizes)
            scat.set_facecolor(colors)

            ring_scat.set_offsets(p[topk])
            ring_scat.set_sizes([220] * K)
            ring_scat.set_alpha(0.9)

            # Gradient arrows: follow each mined negative as it moves.
            for gi, j in enumerate(topk):
                # Show arrow from current position outward.
                u = pos[j] / max(d[j], 1e-6)
                tail = p[j] + 0.18 * u
                head = p[j] + (0.55 + 0.6 * (M - d[j])) * u * (1 - 0.8 * e)
                if 1 - 0.8 * e > 0.2:
                    grad_arrows[gi].set_positions(tail, head)
                    grad_arrows[gi].set_alpha(0.9 * (1 - 0.7 * e))
                else:
                    grad_arrows[gi].set_alpha(0.0)

            chapter.set_text("one gradient step — only the mined three move")
            n_now_outside = int((d_cur[topk] >= M).sum())
            footer.set_text(
                f"after the step: {n_now_outside}/{K} mined negatives have "
                "crossed the margin and their loss is now 0."
            )

            # Bars: update heights for mined ones.
            vals_cur = L_cur
            for i, j in enumerate(sort_idx):
                bars[i].set_height(vals_cur[j])
            return

        # ---- Phase 5: final hold ----
        p = pos.copy()
        for j in topk:
            p[j] = step_end[j]
        L_cur, d_cur = loss_at(p)

        sizes, colors = dot_styles(L_cur, d_cur, fade_easy=True,
                                   select_hard=True)
        scat.set_offsets(p)
        scat.set_sizes(sizes)
        scat.set_facecolor(colors)

        ring_scat.set_offsets(p[topk])
        ring_scat.set_sizes([220] * K)
        ring_scat.set_alpha(0.9)
        for ga in grad_arrows:
            ga.set_alpha(0.0)

        chapter.set_text("gradient signal, concentrated")
        footer.set_text(
            "same compute → larger effective step on the examples that matter."
        )
        for i, j in enumerate(sort_idx):
            bars[i].set_height(L_cur[j])

    anim = animation.FuncAnimation(
        fig, update, frames=TOTAL, interval=1000 / fps, blit=False
    )
    save_animation(anim, out_path, fps=fps)
    plt.close(fig)
    print(f"wrote {out_path}  ({TOTAL} frames, {fps} fps, no loop)")
# --------------------------------------------------------------------------
# 01e — training an MLP with pairwise contrastive hinge + hard mining
# --------------------------------------------------------------------------

_TRAIN_CACHE = OUT_DIR / ".train_cache.npz"


def _run_training(n_per_arm=140, n_steps=800, save_every=6,
                  batch=256, margin=0.9, lr=3e-3, seed=0,
                  warmup_steps=80, use_cache=True):
    """Train an MLP with pairwise contrastive hinge loss.

    The MLP ends in a Tanh so the embedding is bounded to [-1, 1]^2.  We
    start with random negatives (warmup) and then switch to semi-hard
    mining, which gives pedagogically-nice cluster formation without the
    collapse failure mode seen with pure hard mining on normalised
    embeddings.

    Returns (X_input, y, embeddings_per_snapshot, losses, demos).
    Results are cached to disk; pass use_cache=False to retrain.
    """
    import torch
    import torch.nn as nn

    cache_key = (n_per_arm, n_steps, save_every, batch,
                 margin, lr, seed, warmup_steps)
    cache_key_str = repr(cache_key)
    if use_cache and _TRAIN_CACHE.exists():
        data = np.load(_TRAIN_CACHE, allow_pickle=True)
        if str(data["key"]) == cache_key_str:
            print("  (using cached training result)")
            return (data["X"].copy(), data["y"].copy(),
                    [e for e in data["embeddings"]],
                    list(data["losses"]),
                    [tuple(d) for d in data["demos"]])

    torch.manual_seed(seed)
    np_rng = np.random.default_rng(seed)

    # Normalise the inputs so the MLP doesn't have to fight scale.
    X_np_raw, y_np = spiral_galaxy(n_per_arm=n_per_arm, seed=seed)
    X_np = X_np_raw / np.abs(X_np_raw).max()   # roughly in [-1, 1]^2
    N = len(X_np)
    X = torch.tensor(X_np, dtype=torch.float32)
    y = y_np

    net = nn.Sequential(
        nn.Linear(2,   128), nn.ReLU(),
        nn.Linear(128, 128), nn.ReLU(),
        nn.Linear(128, 128), nn.ReLU(),
        nn.Linear(128,   2),
        nn.Tanh(),
    )
    for m in net:
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            nn.init.zeros_(m.bias)
    optim = torch.optim.Adam(net.parameters(), lr=lr)

    class_idx = {k: np.where(y == k)[0] for k in np.unique(y)}
    diff_pools = {k: np.concatenate([class_idx[j] for j in class_idx if j != k])
                  for k in class_idx}

    embeddings = []
    losses = []
    demos = []

    for step in range(n_steps):
        anchor_idx = np_rng.integers(0, N, batch)
        y_a = y[anchor_idx]

        # Positives: random same-class (not the anchor itself).
        pos_idx = np.empty(batch, dtype=np.int64)
        for i in range(batch):
            same = class_idx[int(y_a[i])]
            same = same[same != anchor_idx[i]]
            pos_idx[i] = np_rng.choice(same)

        # Negatives: random for warmup, semi-hard thereafter.
        if step < warmup_steps:
            neg_idx = np.empty(batch, dtype=np.int64)
            for i in range(batch):
                neg_idx[i] = np_rng.choice(diff_pools[int(y_a[i])])
        else:
            net.eval()
            with torch.no_grad():
                emb_all = net(X).numpy()
            neg_idx = np.empty(batch, dtype=np.int64)
            for i in range(batch):
                pool = diff_pools[int(y_a[i])]
                cand = np_rng.choice(pool, size=min(64, len(pool)),
                                     replace=False)
                d_an = np.linalg.norm(
                    emb_all[cand] - emb_all[anchor_idx[i]], axis=1
                )
                # semi-hard: sample from the closer half of the pool
                closer_half = cand[np.argsort(d_an)[: len(cand) // 2]]
                neg_idx[i] = np_rng.choice(closer_half)

        net.train()
        e_a = net(X[anchor_idx])
        e_p = net(X[pos_idx])
        e_n = net(X[neg_idx])

        d_ap = torch.linalg.norm(e_a - e_p, dim=1)
        d_an = torch.linalg.norm(e_a - e_n, dim=1)
        loss = (d_ap ** 2
                + torch.clamp(margin - d_an, min=0) ** 2).mean()

        optim.zero_grad()
        loss.backward()
        optim.step()

        if step % save_every == 0:
            with torch.no_grad():
                emb = net(X).numpy()
            embeddings.append(emb)
            losses.append(float(loss.detach()))
            demos.append(
                (int(anchor_idx[0]), int(pos_idx[0]), int(neg_idx[0]))
            )

    if use_cache:
        np.savez(
            _TRAIN_CACHE,
            key=cache_key_str,
            X=X_np_raw,
            y=y,
            embeddings=np.array(embeddings),
            losses=np.array(losses),
            demos=np.array(demos),
        )
    return X_np_raw, y, embeddings, losses, demos


MARGIN = 1.0   # margin used for both 01e and 01f


def make_training_gif(out_path: Path,
                      fps: int = DEFAULT_FPS,
                      size: int = 720):
    from matplotlib.patches import FancyArrowPatch

    print("  training …")
    X, y, embeddings, losses, demos = _run_training(
        n_per_arm=140, n_steps=800, save_every=6,
        batch=256, margin=MARGIN, lr=3e-3, seed=0, warmup_steps=80,
    )
    n_snaps = len(embeddings)

    # Frame schedule: training segments interleaved with three long demo
    # pauses (~5 s each).  We deliberately freeze the embedding during a
    # demo so the audience can read the arrows.
    DEMO_LEN = 70                     # ~5 s at 14 fps
    HOLD     = 25
    # Pick where each demo lands by snapshot index.
    demo_snaps = [
        min(int(n_snaps * 0.32), n_snaps - 1),
        min(int(n_snaps * 0.65), n_snaps - 1),
        n_snaps - 1,
    ]
    # Hand-pick anchors of three different classes for variety.
    demo_anchors = []
    used_classes = set()
    for snap in demo_snaps:
        # Walk a window of 12 demos around `snap` for an anchor whose
        # class hasn't been shown yet.
        chosen = None
        for k in list(range(snap, max(snap - 12, -1), -1)) + \
                 list(range(snap + 1, min(snap + 12, n_snaps))):
            ai, pi, ni = demos[k]
            if y[ai] not in used_classes:
                chosen = (ai, pi, ni); break
        if chosen is None:
            chosen = demos[snap]
        used_classes.add(int(y[chosen[0]]))
        demo_anchors.append(chosen)

    # Build the per-frame plan.
    plan = []   # (snap_idx, in_demo: bool, demo_idx: int, demo_alpha: float)
    seg_starts = [0] + demo_snaps[:-1]
    seg_ends   = demo_snaps
    for seg_i, (s_start, s_end) in enumerate(zip(seg_starts, seg_ends)):
        for snap in range(s_start, s_end):
            plan.append((snap, False, -1, 0.0))
        # demo pause at s_end
        for j in range(DEMO_LEN):
            if j < 8:
                a = j / 8.0
            elif j > DEMO_LEN - 9:
                a = max(0.0, (DEMO_LEN - 1 - j) / 8.0)
            else:
                a = 1.0
            plan.append((s_end, True, seg_i, a))
    # Final hold (no demo, last snapshot).
    for _ in range(HOLD):
        plan.append((n_snaps - 1, False, -1, 0.0))
    TOTAL = len(plan)
    print(f"  {n_snaps} snapshots; gif = {TOTAL} frames "
          f"(~{TOTAL / fps:.1f} s)")

    # Embeddings live in tanh ∈ [-1, 1]^2; add a small border.
    emb_span = 1.15

    # ----------------------------------------------------------------------
    # Layout: two square panels side by side + thin loss strip.
    # ----------------------------------------------------------------------
    fig = plt.figure(figsize=(11.0, 6.8), dpi=100)
    gs = fig.add_gridspec(
        2, 2, width_ratios=[1.0, 1.0], height_ratios=[4.2, 1.0],
        hspace=0.28, wspace=0.10,
        left=0.05, right=0.97, top=0.83, bottom=0.09,
    )
    ax_in  = fig.add_subplot(gs[0, 0])
    ax_em  = fig.add_subplot(gs[0, 1])
    ax_ls  = fig.add_subplot(gs[1, :])

    # Left: input space with the static spiral.
    ax_in.set_xlim(-3.5, 3.5); ax_in.set_ylim(-3.5, 3.5)
    ax_in.set_aspect("equal")
    ax_in.set_xticks([]); ax_in.set_yticks([])
    for s in ("left", "right", "top", "bottom"):
        ax_in.spines[s].set_visible(False)
    ax_in.set_title("input space", fontsize=11, color=MUTE, pad=6, loc="left")
    in_colors = np.array([CLASS_COLORS[int(c)] for c in y])
    ax_in.scatter(X[:, 0], X[:, 1], s=20, c=in_colors,
                  edgecolors="white", linewidths=0.5, zorder=3)

    # Optional triplet demo lines on the input side.
    in_line_pos, = ax_in.plot([], [], color=CLASS_COLORS[1], lw=1.6,
                              alpha=0.0, zorder=5)
    in_line_neg, = ax_in.plot([], [], color=CLASS_COLORS[0], lw=1.6,
                              alpha=0.0, zorder=5)
    in_anchor_ring = ax_in.scatter([], [], s=160, facecolors="none",
                                   edgecolors=FG, linewidths=1.8, zorder=6,
                                   alpha=0.0)

    # Right: embedding space.
    ax_em.set_xlim(-emb_span, emb_span); ax_em.set_ylim(-emb_span, emb_span)
    ax_em.set_aspect("equal")
    ax_em.set_xticks([]); ax_em.set_yticks([])
    for s in ("left", "right", "top", "bottom"):
        ax_em.spines[s].set_visible(False)
    ax_em.set_title("embedding space  $f(x) \\in [-1, 1]^2$",
                    fontsize=11, color=MUTE, pad=6, loc="left")
    # A light bounding square so the tanh region is visible.
    ax_em.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1],
               color=MUTE, lw=1.0, ls="--", alpha=0.55, zorder=1)
    em_scat = ax_em.scatter(
        embeddings[0][:, 0], embeddings[0][:, 1],
        s=20, c=in_colors, edgecolors="white", linewidths=0.5, zorder=3,
    )

    # Triplet demo overlays.
    # Pull arrow: drawn ON the anchor, points TOWARD the positive
    # (anchor is being pulled into the same-class neighbour).
    pull_arrow = FancyArrowPatch((0, 0), (0, 0),
                                 arrowstyle="-|>", mutation_scale=30,
                                 lw=3.4, color="#2563EB", zorder=7, alpha=0.0)
    # Push arrow: drawn ON the negative, points AWAY from the anchor
    # (the negative is being pushed past the margin).
    push_arrow = FancyArrowPatch((0, 0), (0, 0),
                                 arrowstyle="-|>", mutation_scale=30,
                                 lw=3.4, color="#DC2626", zorder=7, alpha=0.0)
    ax_em.add_patch(pull_arrow); ax_em.add_patch(push_arrow)

    # Big rings around the three involved points (anchor / positive /
    # negative) so they are easy to track on a busy embedding.
    em_anchor_ring = ax_em.scatter([], [], s=240, facecolors="none",
                                   edgecolors=FG, linewidths=2.0, zorder=7,
                                   alpha=0.0)
    em_pos_ring    = ax_em.scatter([], [], s=170, facecolors="none",
                                   edgecolors="#2563EB", linewidths=1.7,
                                   zorder=7, alpha=0.0)
    em_neg_ring    = ax_em.scatter([], [], s=170, facecolors="none",
                                   edgecolors="#DC2626", linewidths=1.7,
                                   zorder=7, alpha=0.0)
    # Mirror rings on the input side too (so the audience sees which
    # input-space points are being pulled/pushed).
    in_pos_ring = ax_in.scatter([], [], s=140, facecolors="none",
                                edgecolors="#2563EB", linewidths=1.5,
                                zorder=6, alpha=0.0)
    in_neg_ring = ax_in.scatter([], [], s=140, facecolors="none",
                                edgecolors="#DC2626", linewidths=1.5,
                                zorder=6, alpha=0.0)
    # No separate demo banner — we promote the footer text during demos.

    # Bottom: loss strip.
    ax_ls.set_xlim(0, n_snaps - 1)
    ax_ls.set_ylim(0, max(losses) * 1.1)
    ax_ls.set_xticks([]); ax_ls.tick_params(labelsize=9)
    for s in ("top", "right"):
        ax_ls.spines[s].set_visible(False)
    ax_ls.set_ylabel("loss", fontsize=10)
    loss_line, = ax_ls.plot([], [], lw=2.2, color=FG)
    loss_dot = ax_ls.scatter([], [], s=60, c=FG, edgecolors="white",
                             lw=1.5, zorder=5)
    step_label = ax_ls.text(0.98, 0.88, "", transform=ax_ls.transAxes,
                            ha="right", va="top", fontsize=10, color=FG)

    # Headers & footer.
    fig.text(0.5, 0.955,
             "Training:  pairwise hinge + hard negative mining",
             ha="center", va="top", fontsize=15, fontweight="bold")
    fig.text(0.5, 0.905,
             "tanh-bounded MLP  $2 \\to 128^3 \\to 2$  "
             "·  margin $m = 0.9$  ·  semi-hard negatives",
             ha="center", va="top", fontsize=11, color=MUTE)
    footer = fig.text(0.5, 0.025, "", ha="center", va="bottom",
                      fontsize=10, color=FG)

    DEMO_BANNERS = [
        "demo 1 — early training",
        "demo 2 — clusters forming",
        "demo 3 — converged",
    ]

    def hide_demo():
        for art in (in_line_pos, in_line_neg, in_anchor_ring,
                    in_pos_ring, in_neg_ring, em_anchor_ring,
                    em_pos_ring, em_neg_ring):
            art.set_alpha(0.0)
        pull_arrow.set_alpha(0.0)
        push_arrow.set_alpha(0.0)

    def draw_demo(snap, demo_i, alpha):
        ai, pi, ni = demo_anchors[demo_i]
        emb = embeddings[snap]
        e_a, e_p, e_n = emb[ai], emb[pi], emb[ni]
        d_ap = float(np.linalg.norm(e_p - e_a))
        d_an = float(np.linalg.norm(e_n - e_a))
        push_mag = max(0.0, MARGIN - d_an)

        # Input-side: dashed-ish lines + rings.
        in_line_pos.set_data([X[ai, 0], X[pi, 0]], [X[ai, 1], X[pi, 1]])
        in_line_neg.set_data([X[ai, 0], X[ni, 0]], [X[ai, 1], X[ni, 1]])
        in_line_pos.set_alpha(0.55 * alpha)
        in_line_neg.set_alpha(0.55 * alpha)
        in_anchor_ring.set_offsets([X[ai]])
        in_anchor_ring.set_alpha(alpha)
        in_pos_ring.set_offsets([X[pi]])
        in_pos_ring.set_alpha(alpha)
        in_neg_ring.set_offsets([X[ni]])
        in_neg_ring.set_alpha(alpha)

        # Embedding-side: anchor + positive + negative rings.
        em_anchor_ring.set_offsets([e_a])
        em_anchor_ring.set_alpha(alpha)
        em_pos_ring.set_offsets([e_p])
        em_pos_ring.set_alpha(alpha)
        em_neg_ring.set_offsets([e_n])
        em_neg_ring.set_alpha(alpha)

        # Pull arrow: from anchor toward positive.  Minimum visible length.
        u_pull = (e_p - e_a) / max(d_ap, 1e-6)
        pull_len = float(min(0.22 + 0.45 * d_ap, 0.55))
        pull_tail = e_a + 0.06 * u_pull
        pull_head = pull_tail + pull_len * u_pull
        pull_arrow.set_positions(pull_tail, pull_head)
        pull_arrow.set_alpha(0.95 * alpha)

        # Push arrow: from negative AWAY from anchor (only if violator).
        if push_mag > 0.02:
            u_push = (e_n - e_a) / max(d_an, 1e-6)   # outward
            push_len = float(min(0.22 + 0.65 * push_mag, 0.55))
            push_tail = e_n + 0.06 * u_push
            push_head = push_tail + push_len * u_push
            push_arrow.set_positions(push_tail, push_head)
            push_arrow.set_alpha(0.95 * alpha)
        else:
            push_arrow.set_alpha(0.0)

        return d_ap, d_an, push_mag

    def update(f):
        snap, in_demo, demo_i, alpha = plan[f]
        emb = embeddings[snap]
        em_scat.set_offsets(emb)

        # Loss strip.
        xs = np.arange(snap + 1)
        loss_line.set_data(xs, losses[:snap + 1])
        loss_dot.set_offsets([[snap, losses[snap]]])
        step_label.set_text(
            f"step {snap * 4}  ·  loss {losses[snap]:.3f}"
        )

        if in_demo:
            d_ap, d_an, push_mag = draw_demo(snap, demo_i, alpha)
            head = DEMO_BANNERS[demo_i]
            if push_mag > 0.02:
                line2 = ("blue arrow pulls anchor toward same-class positive   "
                         "·   red arrow pushes hard negative past "
                         f"margin $m = {MARGIN}$")
            else:
                line2 = ("negative already past margin → no push, "
                         "only a (now small) attractive pull remains")
            footer.set_text(f"{head}\n{line2}")
        else:
            hide_demo()
            if f >= TOTAL - HOLD:
                footer.set_text(
                    "learned embedding  →  3 clean clusters; "
                    "a nearest-neighbour query now respects class."
                )
            else:
                footer.set_text("")

    anim = animation.FuncAnimation(
        fig, update, frames=TOTAL, interval=1000 / fps, blit=False
    )
    save_animation(anim, out_path, fps=fps)
    plt.close(fig)
    print(f"wrote {out_path}  ({TOTAL} frames, {fps} fps, no loop)")


# --------------------------------------------------------------------------
# 01f — k-NN under the learned metric (the payoff)
# --------------------------------------------------------------------------

def _pick_adversarial_queries(X, y, radius=0.95, seed_offset=11):
    """Same logic as 01a — pick a query per class whose Euclidean
    radius-ball contains many wrong-class neighbours."""
    r_all = np.linalg.norm(X, axis=1)
    queries = []
    for k in range(3):
        mask = (y == k)
        idx_k = np.where(mask)[0]
        keep = idx_k[(r_all[idx_k] > 1.4) & (r_all[idx_k] < 2.7)]
        scores = []
        for i in keep:
            d = np.linalg.norm(X - X[i], axis=1)
            in_ball = (d <= radius) & (d > 1e-6)
            wrong = in_ball & (y != k)
            scores.append(int(wrong.sum()))
        order_ = np.argsort(scores)[::-1][:6]
        rng = np.random.default_rng(seed_offset + k)
        queries.append(int(keep[order_[rng.integers(0, len(order_))]]))
    return queries


def make_knn_gif(out_path: Path,
                 fps: int = DEFAULT_FPS,
                 size: int = 720):
    print("  loading trained model …")
    X, y, embeddings, losses, demos = _run_training(
        n_per_arm=140, n_steps=800, save_every=6,
        batch=256, margin=MARGIN, lr=3e-3, seed=0, warmup_steps=80,
    )
    final_emb = embeddings[-1]
    R = 0.9    # radius in embedding space (matches the training margin)
    N = len(X)

    # Re-use the same three adversarial queries as 01a, so the comparison
    # is direct and obvious.
    queries = _pick_adversarial_queries(X, y)

    # Precompute, for each query, the embedded distances to every point
    # and a Euclidean-radius comparison for the on-screen counter.
    embed_dist = []
    eucl_dist  = []
    for q in queries:
        embed_dist.append(np.linalg.norm(final_emb - final_emb[q], axis=1))
        eucl_dist.append(np.linalg.norm(X - X[q], axis=1))

    # ----------------------------------------------------------------------
    # Layout: single 1:1 canvas (the spiral) + thin colorbar legend below.
    # ----------------------------------------------------------------------
    fig = plt.figure(figsize=(size / 100, (size + 96) / 100), dpi=100)
    ax = fig.add_axes([0.04, 0.10, 0.92, 0.80])
    ax.set_xlim(-3.9, 3.9); ax.set_ylim(-3.9, 3.9)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ("left", "right", "top", "bottom"):
        ax.spines[s].set_visible(False)

    # Distance colormap — perceptually uniform, low (close) bright,
    # high (far) dark; chosen to contrast with the class palette.
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.cm import ScalarMappable
    dist_cmap = LinearSegmentedColormap.from_list(
        "embed_dist",
        ["#FCE38A", "#F28D35", "#C2417C", "#5B2A86", "#1B0F3B"],
    )
    # Distance domain for the colormap.  Embeddings live in [-1, 1]^2,
    # so the largest feasible distance is ~2.83.  We clip at 1.6 because
    # past that the colour stops being meaningful for the seminar.
    dnorm = Normalize(vmin=0.0, vmax=1.6)

    scat = ax.scatter(X[:, 0], X[:, 1], s=26, c=[FG] * N,
                      edgecolors="white", linewidths=0.6, zorder=3)

    # Big query marker (always class-coloured edge so the audience knows
    # which arm we're querying from).
    query_dot = ax.scatter([], [], s=320, facecolors="white",
                           edgecolors=FG, linewidths=2.6, zorder=6)
    query_halo = ax.scatter([], [], s=620, facecolors="white",
                            edgecolors="none", zorder=5, alpha=0.0)

    # In-radius rings — gold, very visible.
    in_ring = ax.scatter([], [], s=170, facecolors="none",
                         edgecolors="#1A1A1A", linewidths=2.0,
                         zorder=7, alpha=0.0)

    # Headers + caption.
    fig.text(0.06, 0.94, "Nearest neighbours under the learned metric",
             ha="left", va="top", fontsize=15, fontweight="bold")
    fig.text(0.06, 0.905,
             rf"highlight rings show $\|f(x) - f(\mathrm{{query}})\| < {R}$",
             ha="left", va="top", fontsize=11, color=MUTE)
    caption = fig.text(0.5, 0.05, "", ha="center", va="bottom",
                       fontsize=12, color=FG)
    cap_sub = fig.text(0.5, 0.02, "", ha="center", va="bottom",
                       fontsize=10, color=MUTE)

    # Inline colour-bar legend, top-right of the figure.
    cb_ax = fig.add_axes([0.62, 0.97, 0.32, 0.014])
    sm = ScalarMappable(norm=dnorm, cmap=dist_cmap)
    cb = fig.colorbar(sm, cax=cb_ax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, length=0, pad=2)
    cb.set_ticks([0.0, R, 1.6])
    cb.set_ticklabels(["0", f"{R:g}", "1.6+"])
    cb.ax.set_xlabel("embedded distance from query", fontsize=9,
                     color=MUTE, labelpad=2)

    # ----------------------------------------------------------------------
    # Frame schedule.
    # ----------------------------------------------------------------------
    PER_QUERY = 75       # ~5.4 s per query
    FADE_IN   = 14
    FINAL_HOLD = 30
    TOTAL = PER_QUERY * len(queries) + FINAL_HOLD

    def update(f):
        if f < PER_QUERY * len(queries):
            qi = f // PER_QUERY
            t  = f %  PER_QUERY
        else:
            qi = len(queries) - 1
            t  = PER_QUERY - 1

        q = queries[qi]
        ed = embed_dist[qi]
        ed_clamped = np.clip(ed, 0.0, 1.6)

        # Fade-in animation: blend point colours from neutral grey into
        # the distance colour over the first FADE_IN frames of each query.
        s = min(1.0, t / FADE_IN)
        rgb_dist = dist_cmap(dnorm(ed_clamped))
        # neutral starting colour (cool grey, slightly transparent)
        rgb_neutral = np.tile(np.array([0.78, 0.79, 0.82, 1.0]), (N, 1))
        rgb = rgb_neutral * (1 - s) + rgb_dist * s
        scat.set_facecolor(rgb)
        # Slightly larger sizes for closer points to give an extra
        # visual lift.
        sizes = 22 + 28 * np.exp(-ed_clamped / 0.4) * s
        scat.set_sizes(sizes)

        # Query marker.
        qx, qy = X[q]
        cls_col = CLASS_COLORS[int(y[q])]
        query_dot.set_offsets([[qx, qy]])
        query_dot.set_facecolors(cls_col)
        query_dot.set_edgecolors("white")
        query_halo.set_offsets([[qx, qy]])
        query_halo.set_alpha(0.95)

        # In-radius rings appear after the fade-in finishes.
        if s >= 1.0:
            in_mask = (ed < R) & (np.arange(N) != q)
            in_ring.set_offsets(X[in_mask] if in_mask.any()
                                else np.empty((0, 2)))
            in_ring.set_alpha(0.95)

            same = int((y[in_mask] == y[q]).sum())
            total = int(in_mask.sum())
            same_pct = (100.0 * same / total) if total > 0 else 0.0
            n_eucl_in = int((eucl_dist[qi] < 0.95).sum() - 1)
            n_eucl_wrong = int(((eucl_dist[qi] < 0.95)
                                & (y != y[q])).sum())
            caption.set_text(
                f"query on arm {y[q]}    ·    "
                f"{total} pts within $r = {R}$    ·    "
                f"{same_pct:.0f}% same class"
            )
            cap_sub.set_text(
                f"01a's Euclidean ball of similar size: "
                f"{n_eucl_in} pts, "
                f"{n_eucl_wrong} of them wrong-class."
            )
        else:
            in_ring.set_alpha(0.0)
            caption.set_text(f"query on arm {y[q]}   ·   measuring …")
            cap_sub.set_text("")

    anim = animation.FuncAnimation(
        fig, update, frames=TOTAL, interval=1000 / fps, blit=False
    )
    save_animation(anim, out_path, fps=fps)
    plt.close(fig)
    print(f"wrote {out_path}  ({TOTAL} frames, {fps} fps, no loop)")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

BUILDERS = {
    "dataset":  ("01a_dataset.gif",        make_dataset_gif),
    "loss":     ("01b_pairwise_loss.gif",  make_loss_gif),
    "mining":   ("01d_hard_negatives.gif", make_mining_gif),
    "training": ("01e_training.gif",       make_training_gif),
    "knn":      ("01f_learned_knn.gif",    make_knn_gif),
}

def main(argv):
    targets = argv[1:] if len(argv) > 1 else ["dataset"]
    if targets == ["all"]:
        targets = list(BUILDERS)
    for t in targets:
        if t not in BUILDERS:
            raise SystemExit(f"unknown target {t!r}; choose from {list(BUILDERS)}")
        fname, fn = BUILDERS[t]
        fn(OUT_DIR / fname)


if __name__ == "__main__":
    main(sys.argv)
