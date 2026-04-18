"""02b — Chamfer: local neighbourhood zoom showing many-to-one failure."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch
from .helpers import (twins_dataset, pair_colors, b_pair_colors,
                      chamfer_matches, setup_canvas, save_animation,
                      DEFAULT_FPS, FG, MUTE, GRID, GREY, A_COLOR, B_COLOR,
                      OK_COLOR, BAD_COLOR, _ease_inout)


def render(out_path: Path, fps: int = DEFAULT_FPS):
    A, B, gt = twins_dataset()
    N = len(A)

    # Pick one of the "confusing" clusters (pairs 0,1 share close centres).
    cluster_a = [0, 1]
    # Find the B indices that are their twins + any other B nearby.
    cluster_b = [gt[i] for i in cluster_a]
    # Also grab any A/B within 1.5 of the cluster centre for context.
    cx = 0.5 * (A[cluster_a[0]] + A[cluster_a[1]])
    for i in range(N):
        if i not in cluster_a and np.linalg.norm(A[i] - cx) < 1.8:
            cluster_a.append(i)
            cluster_b.append(gt[i])
    cluster_a = sorted(set(cluster_a))
    cluster_b_from_a = [gt[i] for i in cluster_a]
    # All B indices we'll show (twins of the cluster + any B near centre).
    all_b = set(cluster_b_from_a)
    for j in range(N):
        if np.linalg.norm(B[j] - cx) < 1.8:
            all_b.add(j)
    all_b = sorted(all_b)

    # Local Chamfer within this cluster.
    A_local = A[cluster_a]
    B_local = B[all_b]
    a_to_b_local, _, _ = chamfer_matches(A_local, B_local)
    # Map back to global B indices for GT check.
    a_to_b_global = [all_b[j] for j in a_to_b_local]

    n_a = len(cluster_a)
    n_b = len(all_b)

    # Bounding box for the zoom.
    pts = np.concatenate([A_local, B_local])
    lo = pts.min(axis=0) - 0.8
    hi = pts.max(axis=0) + 0.8

    # --- Figure ---
    fig = plt.figure(figsize=(7.6, 8.0), dpi=100)
    ax = fig.add_axes([0.04, 0.06, 0.92, 0.78])
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ("left", "right", "top", "bottom"):
        ax.spines[s].set_visible(False)

    sa = ax.scatter(A_local[:, 0], A_local[:, 1], s=220, c=A_COLOR,
                    marker="o", edgecolors="white", linewidths=2, zorder=4)
    sb = ax.scatter(B_local[:, 0], B_local[:, 1], s=220, c=B_COLOR,
                    marker="D", edgecolors="white", linewidths=2, zorder=4)

    # Pre-create arrow artists (one per A in cluster).
    arrows = []
    for _ in range(n_a):
        ar = FancyArrowPatch((0, 0), (0, 0), arrowstyle="-|>",
                             mutation_scale=22, lw=2.4, color=A_COLOR,
                             alpha=0.0, zorder=3, shrinkA=6, shrinkB=6)
        ax.add_patch(ar)
        arrows.append(ar)

    # Highlight ring for the "active" A being processed.
    ring = ax.scatter([], [], s=400, facecolors="none", edgecolors=FG,
                      linewidths=2.5, zorder=6, alpha=0.0)
    # Target ring on the chosen B.
    tgt = ax.scatter([], [], s=400, facecolors="none", edgecolors=FG,
                     linewidths=2.5, zorder=6, alpha=0.0, marker="D")

    fig.text(0.5, 0.96, "Chamfer matching — local view",
             ha="center", va="top", fontsize=15, fontweight="bold")
    sub = fig.text(0.5, 0.92, "", ha="center", va="top",
                   fontsize=11, color=MUTE)
    cap = fig.text(0.5, 0.025, "", ha="center", va="bottom",
                   fontsize=11, color=FG)

    # Schedule: for each A, spend TICK frames growing the arrow.
    TICK = 40     # ~2.8 s per circle
    HOLD_ALL = 50
    HOLD_VERDICT = 50
    TOTAL = n_a * TICK + HOLD_ALL + HOLD_VERDICT

    def update(f):
        if f < n_a * TICK:
            idx = f // TICK
            t = (f % TICK + 1) / TICK
            # Show arrows 0..idx-1 fully; arrow idx extending.
            for k in range(n_a):
                if k < idx:
                    arrows[k].set_positions(A_local[k], B_local[a_to_b_local[k]])
                    arrows[k].set_alpha(0.85)
                elif k == idx:
                    head = A_local[k] + t * (B_local[a_to_b_local[k]] - A_local[k])
                    arrows[k].set_positions(A_local[k], head)
                    arrows[k].set_alpha(0.85)
                else:
                    arrows[k].set_alpha(0.0)
            ring.set_offsets([A_local[idx]])
            ring.set_alpha(0.9)
            tgt.set_offsets([B_local[a_to_b_local[idx]]])
            tgt.set_alpha(0.9)
            sub.set_text(f"circle {idx+1}/{n_a}: find nearest diamond")
            # Check many-to-one.
            used = [a_to_b_local[k] for k in range(idx + 1)]
            dups = len(used) - len(set(used))
            cap.set_text(f"duplicates so far: {dups}" if dups else "")
            return

        # All arrows drawn.
        for k in range(n_a):
            arrows[k].set_positions(A_local[k], B_local[a_to_b_local[k]])
            arrows[k].set_alpha(0.85)
        ring.set_alpha(0.0); tgt.set_alpha(0.0)

        if f < n_a * TICK + HOLD_ALL:
            # Highlight many-to-one: colour arrows by correctness.
            for k in range(n_a):
                correct = (a_to_b_global[k] == gt[cluster_a[k]])
                arrows[k].set_color(OK_COLOR if correct else BAD_COLOR)
            n_correct = sum(a_to_b_global[k] == gt[cluster_a[k]]
                            for k in range(n_a))
            used = set(a_to_b_local)
            orphans = n_b - len(used)
            sub.set_text("Chamfer = nearest neighbour, no 1-to-1 constraint")
            cap.set_text(f"correct: {n_correct}/{n_a}   "
                         f"orphan diamonds: {orphans}")
            return

        # Final verdict hold.
        sub.set_text("many circles can grab the same diamond")
        cap.set_text("greedy matching \u2192 some twins lost")

    anim = animation.FuncAnimation(fig, update, frames=TOTAL,
                                   interval=1000/fps, blit=False)
    save_animation(anim, out_path, fps=fps)
    plt.close(fig)
    print(f"wrote {out_path}  ({TOTAL} frames, {fps} fps)")
