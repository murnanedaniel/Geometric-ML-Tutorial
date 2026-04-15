"""
Metric-learning visualizations for the Geometric ML tutorial / seminar.

Generates a family of GIFs explaining pairwise contrastive hinge loss,
built around a 3-arm "spiral galaxy" dataset.

    python scripts/metric_learning.py dataset     -> 01a_dataset.gif
    python scripts/metric_learning.py loss        -> 01b_pairwise_loss.gif
    python scripts/metric_learning.py landscape   -> 01c_loss_landscape.gif
    python scripts/metric_learning.py mining      -> 01d_hard_negatives.gif
    python scripts/metric_learning.py training    -> 01e_training.gif
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
# 01c — the loss landscape (attractive bowl + repulsive margin disk)
# --------------------------------------------------------------------------

def _teal_cmap():
    """white -> teal linear cmap, for the attractive bowl."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "teal_bowl", ["#FFFFFF", "#C8E8E1", "#5BB8AA", "#1F7A6E", "#0D4A42"]
    )


def _ember_cmap():
    """white -> ember linear cmap, for the repulsive disk."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "ember_disk", ["#FFFFFF", "#FCD6CE", "#F3866F", "#C8422B", "#7A2213"]
    )


def make_landscape_gif(out_path: Path,
                       fps: int = DEFAULT_FPS,
                       size: int = 720):
    # Landscape grid (kept modest so GIF stays under ~6 MB).
    X_LIM = 3.2
    grid = 140
    xs = np.linspace(-X_LIM, X_LIM, grid)
    ys = np.linspace(-X_LIM, X_LIM, grid)
    XX, YY = np.meshgrid(xs, ys)
    D = np.sqrt(XX ** 2 + YY ** 2)

    # 1-D cross-section grid (distance d from anchor).
    ds = np.linspace(0, X_LIM, 400)

    # ----------------------------------------------------------------------
    # Layout: two panels (2-D heatmap, 1-D cross-section) + title/caption.
    # ----------------------------------------------------------------------
    fig = plt.figure(figsize=(9.0, 5.4), dpi=90)
    gs = fig.add_gridspec(
        1, 2, width_ratios=[1.0, 1.0],
        wspace=0.12, left=0.05, right=0.97, top=0.79, bottom=0.18,
    )
    ax2d = fig.add_subplot(gs[0])
    ax1d = fig.add_subplot(gs[1])

    # 2-D panel: heatmap + anchor + margin outline.
    ax2d.set_xlim(-X_LIM, X_LIM); ax2d.set_ylim(-X_LIM, X_LIM)
    ax2d.set_aspect("equal")
    ax2d.set_xticks([]); ax2d.set_yticks([])
    for s in ("left", "right", "top", "bottom"):
        ax2d.spines[s].set_visible(False)
    ax2d.set_title("loss as a function of the free point's position",
                   fontsize=11, color=MUTE, pad=6, loc="left")

    # imshow artist — we'll swap data + cmap + vmax as chapters change
    img = ax2d.imshow(
        np.zeros_like(D),
        origin="lower",
        extent=[-X_LIM, X_LIM, -X_LIM, X_LIM],
        cmap=_teal_cmap(),
        vmin=0.0, vmax=X_LIM ** 2, zorder=0,
    )
    # anchor marker
    ax2d.scatter([0], [0], s=520, c="white", edgecolors="none", zorder=2)
    ax2d.scatter([0], [0], s=300, c=CLASS_COLORS[1],
                 edgecolors="white", linewidths=2.4, zorder=3)
    ax2d.text(0, -0.42, "anchor", ha="center", va="top",
              fontsize=10, color=FG, zorder=4)

    # Margin circle (chapter 2 only).
    margin_ring = Circle((0, 0), 0.001, fill=False, color=FG,
                         lw=1.8, ls="--", alpha=0.0, zorder=4)
    ax2d.add_patch(margin_ring)
    margin_radius_label = ax2d.text(0, 0, "", ha="center", va="bottom",
                                    color=FG, fontsize=10, alpha=0.0, zorder=5)

    # 1-D panel: cross-section curve.
    ax1d.set_xlim(0, X_LIM)
    ax1d.set_ylim(0, X_LIM ** 2 * 1.05)
    ax1d.set_xlabel("distance $d$ from anchor", fontsize=10)
    ax1d.set_ylabel("pair loss  $L$", fontsize=10)
    ax1d.tick_params(labelsize=9)
    for s in ("top", "right"):
        ax1d.spines[s].set_visible(False)
    ax1d.set_title("radial cross-section",
                   fontsize=11, color=MUTE, pad=6, loc="left")

    # 1-D curves.
    curve_line, = ax1d.plot([], [], lw=3.0, color=CLASS_COLORS[1])
    curve_fill = [None]   # mutable holder for the fill_between artist

    def redraw_fill(ds_arr, L_arr, color):
        if curve_fill[0] is not None:
            curve_fill[0].remove()
        curve_fill[0] = ax1d.fill_between(ds_arr, 0, L_arr,
                                          color=color, alpha=0.18, zorder=1)

    # Vertical dashed line marking the margin on the 1D plot (chapter 2 only).
    m_vline = ax1d.axvline(0, color=FG, ls="--", lw=1.6, alpha=0.0)
    m_vline_label = ax1d.text(0, 0, "", ha="left", va="bottom",
                              color=FG, fontsize=10, alpha=0.0)
    # Moving dot showing the current (d, L) point on the cross-section.
    curve_dot = ax1d.scatter([], [], s=0, zorder=5)

    # Headers (top) & footer (bottom, clear of the xlabel).
    fig.text(0.5, 0.955, "The hinge-loss landscape",
             ha="center", va="top", fontsize=15, fontweight="bold")
    formula = fig.text(0.5, 0.905, "",
                       ha="center", va="top", fontsize=13.5)
    chapter = fig.text(0.5, 0.855, "",
                       ha="center", va="top", fontsize=11, color=MUTE)
    footer = fig.text(0.5, 0.025, "", ha="center", va="bottom",
                      fontsize=10, color=FG)

    # ----------------------------------------------------------------------
    # Timing.
    # ----------------------------------------------------------------------
    T_ATTR   = 70          # chapter 1: attractive landscape
    T_TRANS1 = 16          # cross-fade to chapter 2
    T_MSWEEP = 120         # chapter 2: m-sweep (0.2 → 3.0 → 2.0)
    T_POS    = 80          # chapter 3: free point orbits the anchor
    T_HOLD   = 28          # final hold
    TOTAL    = T_ATTR + T_TRANS1 + T_MSWEEP + T_POS + T_HOLD

    # Precompute landscape data.
    L_attr = D ** 2
    L_attr_curve = ds ** 2

    def L_rep_field(m):
        return np.maximum(0.0, m - D) ** 2

    def L_rep_curve(m):
        return np.maximum(0.0, m - ds) ** 2

    # m schedule for chapter 2 (ease out then back to 2.0)
    m_sched = np.concatenate([
        np.linspace(0.2, 3.0, int(T_MSWEEP * 0.55)),
        np.linspace(3.0, 2.0, T_MSWEEP - int(T_MSWEEP * 0.55)),
    ])

    # Chapter-3 orbit: free point circles the anchor at r_orbit, cycling m=2.0.
    M_FIXED = 2.0
    orbit_t = np.linspace(0, 2 * np.pi, T_POS)
    # Radius oscillates so we see the partner dip in and out of the margin.
    r_orbit = 1.0 + 1.7 * (0.5 + 0.5 * np.sin(orbit_t * 1.0 - np.pi / 2))

    def update(f):
        # ------- Chapter 1: attractive landscape -------
        if f < T_ATTR:
            s = (f + 1) / T_ATTR
            ease = _ease_inout(s)
            # fade in the heatmap
            img.set_cmap(_teal_cmap())
            img.set_data(L_attr * ease)
            img.set_clim(0, X_LIM ** 2)
            img.set_alpha(ease)

            # 1-D curve draws itself
            k = int(ease * len(ds))
            curve_line.set_data(ds[:k], L_attr_curve[:k])
            curve_line.set_color(CLASS_COLORS[1])
            redraw_fill(ds[:k], L_attr_curve[:k], CLASS_COLORS[1])
            if k > 0:
                curve_dot.set_offsets([[ds[k - 1], L_attr_curve[k - 1]]])
                curve_dot.set_sizes([70])
                curve_dot.set_color(CLASS_COLORS[1])
                curve_dot.set_edgecolors("white")
                curve_dot.set_linewidths(1.5)
            ax1d.set_ylim(0, X_LIM ** 2 * 1.05)

            # margin hidden
            margin_ring.set_alpha(0); margin_radius_label.set_alpha(0)
            m_vline.set_alpha(0); m_vline_label.set_alpha(0)

            formula.set_text(r"$L^+(p)\;=\;\|p - a\|^2$")
            chapter.set_text("chapter 1 — attractive landscape")
            footer.set_text("same class: a quadratic bowl, no upper cap, "
                            "no margin — always pulls.")
            return

        # ------- Cross-fade chapter 1 → chapter 2 -------
        if f < T_ATTR + T_TRANS1:
            s = (f - T_ATTR + 1) / T_TRANS1
            # Fade teal bowl out, ember field in (at m = 0.2, basically nil).
            m = 0.2
            L_mix = (1 - s) * L_attr + s * L_rep_field(m)
            img.set_cmap(_ember_cmap())
            img.set_data(L_rep_field(m))
            img.set_clim(0, 9)
            img.set_alpha(s)
            # 1-D: swap curves
            curve = L_rep_curve(m)
            curve_line.set_data(ds, curve)
            curve_line.set_color(CLASS_COLORS[0])
            curve_line.set_alpha(s)
            redraw_fill(ds, curve, CLASS_COLORS[0])
            curve_fill[0].set_alpha(0.18 * s)
            curve_dot.set_sizes([0])
            ax1d.set_ylim(0, X_LIM ** 2 * 1.05)

            margin_ring.set_alpha(0.9 * s)
            margin_ring.set_radius(m)
            margin_radius_label.set_alpha(s)
            margin_radius_label.set_position((0, m + 0.08))
            margin_radius_label.set_text(f"m = {m:.2f}")
            m_vline.set_xdata([m, m])
            m_vline.set_alpha(0.9 * s)
            m_vline_label.set_position((m + 0.06, X_LIM ** 2 * 0.92))
            m_vline_label.set_alpha(s)
            m_vline_label.set_text(r"$m$")

            formula.set_text(r"$L^-(p)\;=\;\max(0,\; m - \|p - a\|)^2$")
            chapter.set_text("switching to the repulsive branch…")
            footer.set_text("")
            return

        # ------- Chapter 2: m-sweep over repulsive landscape -------
        k_abs = f - T_ATTR - T_TRANS1
        if k_abs < T_MSWEEP:
            k = k_abs
            m = m_sched[k]
            L_field = L_rep_field(m)
            L_curve = L_rep_curve(m)

            img.set_cmap(_ember_cmap())
            img.set_data(L_field)
            # vmax scales with m so the colour range "fills" the disk.
            img.set_clim(0, max(m ** 2, 0.3))
            img.set_alpha(1.0)

            curve_line.set_data(ds, L_curve)
            curve_line.set_color(CLASS_COLORS[0])
            curve_line.set_alpha(1.0)
            redraw_fill(ds, L_curve, CLASS_COLORS[0])

            # Mark the hinge kink.
            curve_dot.set_offsets([[m, 0.0]])
            curve_dot.set_sizes([80])
            curve_dot.set_color(CLASS_COLORS[0])
            curve_dot.set_edgecolors("white")
            curve_dot.set_linewidths(1.5)
            ax1d.set_ylim(0, max(9.5, m ** 2 * 1.2))

            margin_ring.set_alpha(0.9)
            margin_ring.set_radius(m)
            margin_radius_label.set_alpha(1.0)
            margin_radius_label.set_position((0, m + 0.08))
            margin_radius_label.set_text(f"m = {m:.2f}")
            m_vline.set_xdata([m, m])
            m_vline.set_alpha(0.9)
            m_vline_label.set_position((m + 0.06, ax1d.get_ylim()[1] * 0.9))
            m_vline_label.set_alpha(1.0)
            m_vline_label.set_text(r"$m$")

            formula.set_text(r"$L^-(p)\;=\;\max(0,\; m - \|p - a\|)^2$")
            chapter.set_text(f"chapter 2 — margin sweep  ($m = {m:.2f}$)")
            footer.set_text(
                "flat-zero outside the disk, quadratic bowl inside — "
                "the hinge kink at $d = m$ is exactly the boundary."
            )
            return

        # ------- Chapter 3: free point orbits at fixed m -------
        k_abs2 = f - T_ATTR - T_TRANS1 - T_MSWEEP
        if k_abs2 < T_POS:
            k = k_abs2
            m = M_FIXED
            L_field = L_rep_field(m)
            L_curve = L_rep_curve(m)

            img.set_cmap(_ember_cmap())
            img.set_data(L_field)
            img.set_clim(0, m ** 2)
            img.set_alpha(1.0)

            # Place a "free point" orbiting the anchor.
            r = r_orbit[k]
            theta = orbit_t[k]
            px, py = r * np.cos(theta), r * np.sin(theta)
            L_here = max(0.0, m - r) ** 2

            # Remove previous orbit artist (stored on ax2d as attribute).
            if not hasattr(ax2d, "_orbit_dot"):
                ax2d._orbit_dot = ax2d.scatter(
                    [], [], s=220, c=CLASS_COLORS[0],
                    edgecolors="white", linewidths=2.0, zorder=6)
                ax2d._orbit_line, = ax2d.plot(
                    [], [], ls=":", lw=1.2, color=FG, alpha=0.55, zorder=5)
            ax2d._orbit_dot.set_offsets([[px, py]])
            ax2d._orbit_line.set_data([0, px], [0, py])

            # 1-D curve (static) with live dot.
            curve_line.set_data(ds, L_curve)
            curve_line.set_color(CLASS_COLORS[0])
            curve_line.set_alpha(1.0)
            redraw_fill(ds, L_curve, CLASS_COLORS[0])
            curve_dot.set_offsets([[r, L_here]])
            curve_dot.set_sizes([120])
            curve_dot.set_color(CLASS_COLORS[0])
            curve_dot.set_edgecolors("white")
            curve_dot.set_linewidths(1.8)
            ax1d.set_ylim(0, m ** 2 * 1.2)

            margin_ring.set_alpha(0.9)
            margin_ring.set_radius(m)
            margin_radius_label.set_alpha(1.0)
            margin_radius_label.set_position((0, m + 0.08))
            margin_radius_label.set_text(f"m = {m:.2f}")
            m_vline.set_xdata([m, m])
            m_vline.set_alpha(0.9)
            m_vline_label.set_position((m + 0.06, ax1d.get_ylim()[1] * 0.9))
            m_vline_label.set_alpha(1.0)
            m_vline_label.set_text(r"$m$")

            formula.set_text(r"$L^-(p)\;=\;\max(0,\; m - \|p - a\|)^2$")
            chapter.set_text("chapter 3 — a point orbits the anchor  "
                             "(fixed $m$)")
            inside = "inside" if r < m else "outside"
            footer.set_text(
                f"free point at $d = {r:.2f}$ ({inside} margin)  →  "
                f"$L^- = {L_here:.2f}$"
            )
            return

        # ------- Final hold -------
        m = M_FIXED
        r = r_orbit[-1]; theta = orbit_t[-1]
        px, py = r * np.cos(theta), r * np.sin(theta)
        L_here = max(0.0, m - r) ** 2
        if hasattr(ax2d, "_orbit_dot"):
            ax2d._orbit_dot.set_offsets([[px, py]])
            ax2d._orbit_line.set_data([0, px], [0, py])
        formula.set_text(r"$L^-(p)\;=\;\max(0,\; m - \|p - a\|)^2$")
        chapter.set_text("the hinge loss has a flat zero zone and a bowl")
        footer.set_text(
            "the learned metric wants different-class points in the flat zone."
        )

    anim = animation.FuncAnimation(
        fig, update, frames=TOTAL, interval=1000 / fps, blit=False
    )
    save_animation(anim, out_path, fps=fps)
    plt.close(fig)
    print(f"wrote {out_path}  ({TOTAL} frames, {fps} fps, no loop)")
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
def make_training_gif(out_path):   raise NotImplementedError


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

BUILDERS = {
    "dataset":   ("01a_dataset.gif",        make_dataset_gif),
    "loss":      ("01b_pairwise_loss.gif",  make_loss_gif),
    "landscape": ("01c_loss_landscape.gif", make_landscape_gif),
    "mining":    ("01d_hard_negatives.gif", make_mining_gif),
    "training":  ("01e_training.gif",       make_training_gif),
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
