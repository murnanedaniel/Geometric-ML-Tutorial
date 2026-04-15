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
def make_landscape_gif(out_path):  raise NotImplementedError
def make_mining_gif(out_path):     raise NotImplementedError
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
