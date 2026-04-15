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
                     fps: int = 24,
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
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"wrote {out_path}  ({total} frames, {fps} fps)")


# --------------------------------------------------------------------------
# placeholder stubs — filled in after 01a is approved
# --------------------------------------------------------------------------

def make_loss_gif(out_path):       raise NotImplementedError
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
