"""
Set-matching visualisations for the Geometric ML tutorial / seminar.

Builds four (+ one synthesis) GIFs around the "twins" narrative:

    you have two groups of (non-identical) twins.  Half the twins go to
    school A, half go to school B.  Forget the labels — recover the
    matching from positions alone.

    python scripts/set_matching.py dataset    -> 02a_twins_dataset.gif
    python scripts/set_matching.py chamfer    -> 02b_chamfer.gif
    python scripts/set_matching.py hungarian  -> 02c_hungarian.gif
    python scripts/set_matching.py sinkhorn   -> 02d_sinkhorn.gif
    python scripts/set_matching.py synthesis  -> 02e_synthesis.gif
    python scripts/set_matching.py all        -> all of the above
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch
from scipy.optimize import linear_sum_assignment

# Re-use the styling + writer from metric_learning so the two seminars
# look like one coherent set.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from metric_learning import (   # noqa: E402
    PillowWriterNoLoop, save_animation, DEFAULT_FPS,
    BG, FG, MUTE, GRID, _ease_inout,
)

# --------------------------------------------------------------------------
# Style for set matching specifically.
# --------------------------------------------------------------------------

A_COLOR  = "#E85D4A"  # ember (set A — circles)
B_COLOR  = "#2A9D8F"  # teal  (set B — diamonds)
GT_COLOR = "#888888"  # ground-truth pair link
OK_COLOR = "#2E8B57"  # correct match
BAD_COLOR = "#C62828" # wrong match

OUT_DIR = Path(__file__).resolve().parent.parent / "assets" / "set_matching"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Twins dataset
# --------------------------------------------------------------------------

def twins_dataset(n_pairs: int = 20, sigma: float = 0.35,
                  seed: int = 1):
    """Generate `n_pairs` of non-identical twins.

    Returns:
        A: (n, 2) positions of the A-school twins (ember circles)
        B: (n, 2) positions of the B-school twins (teal diamonds)
        gt: (n,)  ground-truth permutation:  B[gt[i]] is the twin of A[i]

    Each pair has a centre mu_k, and the two twins sit at
    mu_k + N(0, sigma).  Centres are drawn so:
      - they are well-spread (Hungarian usually finds the true matching)
      - a few pairs are close enough that Chamfer's nearest-neighbour
        rule mistakes them (so it produces visible failures)
    """
    rng = np.random.default_rng(seed)

    # Mostly well-spread centres + a few clusters of 2-3 close pairs
    # that will trip up Chamfer's nearest-neighbour rule.
    base = rng.uniform(-2.9, 2.9, size=(n_pairs, 2))
    # Light repulsion: push pairs apart unless they're "intentionally close".
    for _ in range(80):
        for i in range(n_pairs):
            for j in range(n_pairs):
                if i == j:
                    continue
                d = base[i] - base[j]
                r = np.linalg.norm(d) + 1e-9
                if r < 0.95:
                    base[i] += 0.04 * d / r
                    base[j] -= 0.04 * d / r
    # Force three "confusing" trios — pairs whose centres are <= 0.6
    # apart, well within the noise scale.
    base[1] = base[0] + np.array([0.50, -0.25])
    base[5] = base[4] + np.array([-0.40, 0.42])
    base[9] = base[8] + np.array([0.35, 0.45])
    base[13] = base[12] + np.array([-0.55, -0.20])

    mu = base
    A = mu + rng.normal(0, sigma, size=(n_pairs, 2))
    B = mu + rng.normal(0, sigma, size=(n_pairs, 2))

    # Permute B so the ground-truth matching isn't index-aligned.
    perm = rng.permutation(n_pairs)
    B = B[perm]
    # After B = B[perm], new_B[j] is the twin of A[perm[j]].  So the twin
    # of A[i] sits at j = inverse_perm[i], i.e. gt = argsort(perm).
    gt = np.argsort(perm)
    return A, B, gt


# --------------------------------------------------------------------------
# Common drawing helpers
# --------------------------------------------------------------------------

def setup_canvas(ax, span=4.0):
    ax.set_xlim(-span, span); ax.set_ylim(-span, span)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ("left", "right", "top", "bottom"):
        ax.spines[s].set_visible(False)


def draw_AB(ax, A, B, alpha_A=1.0, alpha_B=1.0, size=110):
    """Plot A as filled ember circles, B as filled teal diamonds."""
    sa = ax.scatter(A[:, 0], A[:, 1], s=size, c=A_COLOR, marker="o",
                    edgecolors="white", linewidths=1.4, zorder=4,
                    alpha=alpha_A)
    sb = ax.scatter(B[:, 0], B[:, 1], s=size, c=B_COLOR, marker="D",
                    edgecolors="white", linewidths=1.4, zorder=4,
                    alpha=alpha_B)
    return sa, sb


def legend_corner(fig, x, y):
    """Draw a tiny A/B legend on the figure (figure-fraction coords)."""
    fig.text(x + 0.030, y - 0.005,
             "  A  (ember circle)\n  B  (teal diamond)",
             ha="left", va="top",
             fontsize=9, color=FG, linespacing=1.6,
             bbox=dict(boxstyle="round,pad=0.42",
                       fc="white", ec=GRID, lw=0.9))
    # Two small marker images using axes positioned by figure-fraction.
    mk_ax = fig.add_axes([x + 0.005, y - 0.038, 0.022, 0.038])
    mk_ax.set_xlim(-1, 1); mk_ax.set_ylim(-1, 1)
    mk_ax.axis("off")
    mk_ax.scatter([0], [0.5], s=55, c=A_COLOR, marker="o",
                  edgecolors="white", linewidths=1.0)
    mk_ax.scatter([0], [-0.5], s=55, c=B_COLOR, marker="D",
                  edgecolors="white", linewidths=1.0)


# --------------------------------------------------------------------------
# 02a — the twins dataset itself
# --------------------------------------------------------------------------

def make_twins_dataset_gif(out_path: Path, fps: int = DEFAULT_FPS):
    A, B, gt = twins_dataset()
    N = len(A)

    fig = plt.figure(figsize=(7.6, 8.0), dpi=100)
    ax = fig.add_axes([0.04, 0.06, 0.92, 0.78])
    setup_canvas(ax, span=4.2)

    # Twin pair lines (drawn but hidden until phase 2).
    pair_lines = []
    for i in range(N):
        ln, = ax.plot([A[i, 0], B[gt[i], 0]],
                      [A[i, 1], B[gt[i], 1]],
                      color=GT_COLOR, lw=1.3, ls="--", alpha=0.0, zorder=2)
        pair_lines.append(ln)

    # Scatters (initially empty — we'll set offsets per frame).
    sa = ax.scatter([], [], s=140, c=A_COLOR, marker="o",
                    edgecolors="white", linewidths=1.5, zorder=4)
    sb = ax.scatter([], [], s=140, c=B_COLOR, marker="D",
                    edgecolors="white", linewidths=1.5, zorder=4)

    # Headers + caption.
    fig.text(0.5, 0.96, "Two groups of (non-identical) twins",
             ha="center", va="top", fontsize=15, fontweight="bold")
    sub = fig.text(0.5, 0.92, "", ha="center", va="top",
                   fontsize=11, color=MUTE)
    cap = fig.text(0.5, 0.025, "", ha="center", va="bottom",
                   fontsize=11, color=FG)

    legend_corner(fig, 0.04, 0.86)

    # Frame schedule.
    PAIRS_PER_BURST = 3
    BURST_FRAMES = 8           # frames to drop in one batch of pairs
    HOLD_AFTER   = 6
    n_bursts = (N + PAIRS_PER_BURST - 1) // PAIRS_PER_BURST
    SCATTER = n_bursts * (BURST_FRAMES + HOLD_AFTER)
    REVEAL  = 80               # gradient reveal of pairing lines
    HOLD2   = 30
    FADE    = 30               # lines fade out
    HOLD3   = 30               # final challenge state
    TOTAL   = SCATTER + REVEAL + HOLD2 + FADE + HOLD3

    def update(f):
        # ---- Phase 1: scatter in twin pairs ----
        if f < SCATTER:
            burst = f // (BURST_FRAMES + HOLD_AFTER)
            in_burst = f %  (BURST_FRAMES + HOLD_AFTER)
            n_visible_pairs = min(N, (burst + 1) * PAIRS_PER_BURST)
            if in_burst < BURST_FRAMES:
                # current burst is still falling in
                full = burst * PAIRS_PER_BURST
                growing = min(N, full + PAIRS_PER_BURST) - full
                ease = (in_burst + 1) / BURST_FRAMES
                pts_a = [A[k] for k in range(full)]
                pts_b = [B[gt[k]] for k in range(full)]
                # for falling-in pairs: scale position from above
                for j in range(growing):
                    k = full + j
                    if k < N:
                        # they enter from above
                        a = A[k] + np.array([0, 4.0]) * (1 - ease)
                        b = B[gt[k]] + np.array([0, 4.0]) * (1 - ease)
                        pts_a.append(a); pts_b.append(b)
                pts_a = np.array(pts_a); pts_b = np.array(pts_b)
                # B may need re-indexing because gt[k] not k for B
                # We placed in order of pair k.
            else:
                pts_a = A[:n_visible_pairs]
                pts_b = np.array([B[gt[k]] for k in range(n_visible_pairs)])
            sa.set_offsets(pts_a)
            sb.set_offsets(pts_b)
            sub.set_text("each ember–teal pair drops in together")
            cap.set_text("")
            for ln in pair_lines:
                ln.set_alpha(0.0)
            return

        # All twins in place from now on.
        sa.set_offsets(A)
        sb.set_offsets(B)

        # ---- Phase 2: reveal pairing lines progressively ----
        if f < SCATTER + REVEAL:
            t = (f - SCATTER + 1) / REVEAL
            n_lines = int(t * N)
            for i, ln in enumerate(pair_lines):
                if i < n_lines:
                    ln.set_alpha(0.85)
                else:
                    ln.set_alpha(0.0)
            sub.set_text("ground truth: who is twinned with whom")
            cap.set_text(f"{n_lines}/{N} pairs revealed")
            return

        # ---- Phase 3: hold full ground truth ----
        if f < SCATTER + REVEAL + HOLD2:
            for ln in pair_lines:
                ln.set_alpha(0.85)
            sub.set_text("ground truth: who is twinned with whom")
            cap.set_text(f"{N} pairs total — twins are CLOSE but not "
                         "co-located")
            return

        # ---- Phase 4: lines fade ----
        if f < SCATTER + REVEAL + HOLD2 + FADE:
            t = (f - SCATTER - REVEAL - HOLD2 + 1) / FADE
            for ln in pair_lines:
                ln.set_alpha(0.85 * (1 - t))
            sub.set_text("now forget the labels …")
            cap.set_text("")
            return

        # ---- Phase 5: the challenge state ----
        for ln in pair_lines:
            ln.set_alpha(0.0)
        sub.set_text("the challenge")
        cap.set_text(
            "given only the positions of A and B, recover the pairing."
        )

    anim = animation.FuncAnimation(
        fig, update, frames=TOTAL, interval=1000 / fps, blit=False,
    )
    save_animation(anim, out_path, fps=fps)
    plt.close(fig)
    print(f"wrote {out_path}  ({TOTAL} frames, {fps} fps, no loop)")


# --------------------------------------------------------------------------
# Matching algorithm helpers (used by 02b/c/d/e)
# --------------------------------------------------------------------------

def chamfer_matches(A, B):
    """For each a in A, return its nearest-neighbour index in B
    (and vice versa)."""
    Da = np.linalg.norm(A[:, None] - B[None, :], axis=-1)
    a_to_b = Da.argmin(axis=1)
    Db = Da.T
    b_to_a = Db.argmin(axis=1)
    return a_to_b, b_to_a, Da


def hungarian_matches(A, B):
    Da = np.linalg.norm(A[:, None] - B[None, :], axis=-1)
    row, col = linear_sum_assignment(Da)
    return row, col, Da


def sinkhorn(C, eps, n_iters=80, tol=1e-9):
    """Sinkhorn iterations on cost matrix C with regulariser eps and
    uniform marginals.  Returns the transport plan P and the running
    list of plans (one per iteration) for animation."""
    n, m = C.shape
    a = np.full(n, 1.0 / n)
    b = np.full(m, 1.0 / m)
    K = np.exp(-C / eps)
    u = np.ones(n)
    v = np.ones(m)
    plans = []
    for it in range(n_iters):
        u = a / (K @ v + 1e-30)
        v = b / (K.T @ u + 1e-30)
        P = (u[:, None] * K) * v[None, :]
        plans.append(P.copy())
        if it > 5 and np.linalg.norm(P.sum(axis=1) - a) < tol:
            break
    return P, plans


# --------------------------------------------------------------------------
# 02b — Chamfer matching
# --------------------------------------------------------------------------

def make_chamfer_gif(out_path: Path, fps: int = DEFAULT_FPS):
    A, B, gt = twins_dataset()
    N = len(A)
    a_to_b, b_to_a, D = chamfer_matches(A, B)
    # Per-direction nearest-neighbour distances.
    d_ab = D[np.arange(N), a_to_b]      # for each a: dist to its NN in B
    d_ba = D.T[np.arange(N), b_to_a]    # for each b: dist to its NN in A

    # Quick stats for the captions.
    chamfer_total = d_ab.sum() + d_ba.sum()
    correct_a_to_b = int((a_to_b == gt).sum())
    # how many distinct B's are picked by A→B (vs N if it were 1-1)
    distinct_b = len(set(a_to_b.tolist()))
    # b's not picked by anyone in A→B
    unpicked_b = N - distinct_b

    # ----------------------------------------------------------------------
    # Layout: square canvas with a thin "loss bar" along the bottom for the
    # running Chamfer total.
    # ----------------------------------------------------------------------
    fig = plt.figure(figsize=(7.6, 8.4), dpi=100)
    gs = fig.add_gridspec(
        2, 1, height_ratios=[6.0, 1.0],
        hspace=0.20, left=0.04, right=0.96, top=0.84, bottom=0.08,
    )
    ax = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])

    setup_canvas(ax, span=4.2)

    # Static scatters (offsets fixed).
    sa = ax.scatter(A[:, 0], A[:, 1], s=140, c=A_COLOR, marker="o",
                    edgecolors="white", linewidths=1.5, zorder=4)
    sb = ax.scatter(B[:, 0], B[:, 1], s=140, c=B_COLOR, marker="D",
                    edgecolors="white", linewidths=1.5, zorder=4)

    # Storage for arrow artists, drawn one at a time.
    a_arrows = []
    b_arrows = []
    for _ in range(N):
        ar = FancyArrowPatch((0, 0), (0, 0),
                             arrowstyle="-|>", mutation_scale=18,
                             lw=1.9, color=A_COLOR, alpha=0.0, zorder=3,
                             shrinkA=4, shrinkB=4)
        ax.add_patch(ar)
        a_arrows.append(ar)
    for _ in range(N):
        br = FancyArrowPatch((0, 0), (0, 0),
                             arrowstyle="-|>", mutation_scale=18,
                             lw=1.9, color=B_COLOR, alpha=0.0, zorder=3,
                             shrinkA=4, shrinkB=4)
        ax.add_patch(br)
        b_arrows.append(br)

    # Highlight rings: which B is currently being targeted, and a small
    # "missed B" marker for B's that no A picks.
    target_ring = ax.scatter([], [], s=260, facecolors="none",
                             edgecolors=FG, linewidths=2.0,
                             zorder=6, alpha=0.0)
    miss_ring = ax.scatter([], [], s=240, marker="o", facecolors="none",
                           edgecolors="#9C2B19", linewidths=2.4,
                           zorder=6, alpha=0.0)

    # GT-correctness rings (drawn during the verdict phase).
    ok_ring  = ax.scatter([], [], s=240, facecolors="none",
                          edgecolors=OK_COLOR, linewidths=2.4,
                          zorder=7, alpha=0.0)
    bad_ring = ax.scatter([], [], s=240, facecolors="none",
                          edgecolors=BAD_COLOR, linewidths=2.4,
                          zorder=7, alpha=0.0)

    # Headers + caption.
    fig.text(0.5, 0.96, "Chamfer matching", ha="center", va="top",
             fontsize=15, fontweight="bold")
    sub = fig.text(0.5, 0.92, "", ha="center", va="top",
                   fontsize=11, color=MUTE)
    cap = fig.text(0.5, 0.025, "", ha="center", va="bottom",
                   fontsize=11, color=FG)

    # Bar strip — running Chamfer total.
    ax_b.set_xlim(0, 2 * N)
    ax_b.set_ylim(0, max(d_ab.max(), d_ba.max()) * 1.1)
    ax_b.set_xticks([]); ax_b.tick_params(labelsize=9)
    for s in ("top", "right"):
        ax_b.spines[s].set_visible(False)
    ax_b.set_ylabel("NN dist", fontsize=10)
    ax_b.text(0.01, 0.92, "per-arrow nearest-neighbour distance",
              transform=ax_b.transAxes, fontsize=9, color=FG)
    bar_xs = np.arange(2 * N)
    bar_heights = np.zeros(2 * N)
    bar_colors  = [A_COLOR] * N + [B_COLOR] * N
    bars = ax_b.bar(bar_xs, bar_heights, color=bar_colors,
                    edgecolor="white", lw=0.8)
    bar_total = ax_b.text(0.99, 0.92, "", transform=ax_b.transAxes,
                          ha="right", va="top", fontsize=10, color=FG,
                          fontweight="bold")

    legend_corner(fig, 0.04, 0.86)

    # Frame schedule:
    #   Phase 1 - A→B sweep (one A per "tick", 8 frames per tick)
    #   Phase 2 - B→A sweep (one B per tick)
    #   Phase 3 - "many-to-one" highlight: shade duplicate targets red
    #   Phase 4 - "ground-truth verdict": green/red rings
    #   Phase 5 - hold
    TICK = 8
    PHASE1 = N * TICK
    PHASE2 = N * TICK
    PHASE3 = 36
    PHASE4 = 60
    PHASE5 = 30
    TOTAL = PHASE1 + PHASE2 + PHASE3 + PHASE4 + PHASE5

    def update(f):
        # --------- Phase 1: A→B sweep ---------
        if f < PHASE1:
            i = f // TICK             # which A
            t = (f % TICK + 1) / TICK  # ease 0→1 within tick
            # Draw arrows 0..i-1 fully, arrow i partially extending.
            for k in range(N):
                if k < i:
                    a_arrows[k].set_positions(A[k], B[a_to_b[k]])
                    a_arrows[k].set_alpha(0.85)
                elif k == i:
                    head = A[k] + t * (B[a_to_b[k]] - A[k])
                    a_arrows[k].set_positions(A[k], head)
                    a_arrows[k].set_alpha(0.85)
                else:
                    a_arrows[k].set_alpha(0.0)
            # Currently targeted B
            target_ring.set_offsets([B[a_to_b[i]]])
            target_ring.set_alpha(0.9)
            miss_ring.set_alpha(0.0)

            # Bar values for completed arrows
            for k in range(N):
                bars[k].set_height(d_ab[k] if k <= i else 0)
                bars[N + k].set_height(0)

            sub.set_text(r"step 1: each $a \in A$ picks its nearest "
                         r"$b \in B$")
            cap.set_text(f"A→B sweep:  {i + 1}/{N}   ·   "
                         f"Σ = {d_ab[:i + 1].sum():.2f}")
            bar_total.set_text(f"Σ (so far) = {d_ab[:i + 1].sum():.2f}")
            return

        # --------- Phase 2: B→A sweep ---------
        if f < PHASE1 + PHASE2:
            for k in range(N):
                a_arrows[k].set_positions(A[k], B[a_to_b[k]])
                a_arrows[k].set_alpha(0.5)   # dim
            j = (f - PHASE1) // TICK
            t = ((f - PHASE1) % TICK + 1) / TICK
            for k in range(N):
                if k < j:
                    b_arrows[k].set_positions(B[k], A[b_to_a[k]])
                    b_arrows[k].set_alpha(0.85)
                elif k == j:
                    head = B[k] + t * (A[b_to_a[k]] - B[k])
                    b_arrows[k].set_positions(B[k], head)
                    b_arrows[k].set_alpha(0.85)
                else:
                    b_arrows[k].set_alpha(0.0)
            target_ring.set_offsets([A[b_to_a[j]]])
            target_ring.set_alpha(0.9)
            miss_ring.set_alpha(0.0)

            for k in range(N):
                bars[k].set_height(d_ab[k])
                bars[N + k].set_height(d_ba[k] if k <= j else 0)

            sub.set_text(r"step 2: each $b \in B$ picks its nearest "
                         r"$a \in A$")
            running = d_ab.sum() + d_ba[:j + 1].sum()
            cap.set_text(f"B→A sweep:  {j + 1}/{N}   ·   "
                         f"Chamfer Σ = {running:.2f}")
            bar_total.set_text(f"Chamfer Σ = {running:.2f}")
            return

        # --------- Phase 3: many-to-one highlight ---------
        # All arrows visible.  Find which B's are picked multiple times in
        # A→B and which are not picked at all.
        for k in range(N):
            a_arrows[k].set_positions(A[k], B[a_to_b[k]])
            a_arrows[k].set_alpha(0.55)
            b_arrows[k].set_positions(B[k], A[b_to_a[k]])
            b_arrows[k].set_alpha(0.55)

        if f < PHASE1 + PHASE2 + PHASE3:
            target_ring.set_alpha(0.0)
            miss_idx = [k for k in range(N) if k not in a_to_b]
            miss_ring.set_offsets(B[miss_idx] if miss_idx
                                  else np.empty((0, 2)))
            t = (f - PHASE1 - PHASE2 + 1) / PHASE3
            miss_ring.set_alpha(min(1.0, t * 1.3))
            sub.set_text("Chamfer is greedy — many a's may share a single b")
            cap.set_text(
                f"distinct B's chosen by A→B: {distinct_b}/{N}   ·   "
                f"unmatched B's: {unpicked_b}"
            )
            bar_total.set_text(f"Chamfer Σ = {chamfer_total:.2f}")
            ok_ring.set_alpha(0.0); bad_ring.set_alpha(0.0)
            return

        # --------- Phase 4: GT verdict ---------
        # green ring = a's NN was its true twin, red ring = wrong.
        if f < PHASE1 + PHASE2 + PHASE3 + PHASE4:
            t = (f - PHASE1 - PHASE2 - PHASE3 + 1) / PHASE4
            target_ring.set_alpha(0.0)
            miss_ring.set_alpha(0.0)
            n_show = int(t * N)
            ok_pts  = []
            bad_pts = []
            for k in range(min(n_show, N)):
                if a_to_b[k] == gt[k]:
                    ok_pts.append(A[k])
                else:
                    bad_pts.append(A[k])
            ok_ring.set_offsets(ok_pts if ok_pts else np.empty((0, 2)))
            bad_ring.set_offsets(bad_pts if bad_pts else np.empty((0, 2)))
            ok_ring.set_alpha(0.95)
            bad_ring.set_alpha(0.95)
            sub.set_text("vs. ground truth")
            cap.set_text(f"checking matches:  {n_show}/{N}")
            bar_total.set_text(f"Chamfer Σ = {chamfer_total:.2f}")
            return

        # --------- Phase 5: final hold ---------
        target_ring.set_alpha(0.0)
        miss_ring.set_alpha(0.0)
        ok_pts  = [A[k] for k in range(N) if a_to_b[k] == gt[k]]
        bad_pts = [A[k] for k in range(N) if a_to_b[k] != gt[k]]
        ok_ring.set_offsets(ok_pts if ok_pts else np.empty((0, 2)))
        bad_ring.set_offsets(bad_pts if bad_pts else np.empty((0, 2)))
        ok_ring.set_alpha(0.95); bad_ring.set_alpha(0.95)
        sub.set_text("Chamfer:  cheap, no 1-to-1 constraint, often wrong")
        cap.set_text(
            f"correct twin matches:  {correct_a_to_b}/{N}   ·   "
            f"unmatched B's:  {unpicked_b}"
        )

    anim = animation.FuncAnimation(
        fig, update, frames=TOTAL, interval=1000 / fps, blit=False,
    )
    save_animation(anim, out_path, fps=fps)
    plt.close(fig)
    print(f"wrote {out_path}  ({TOTAL} frames, {fps} fps, no loop)")


# --------------------------------------------------------------------------
# placeholder stubs (filled in next)
# --------------------------------------------------------------------------
def make_hungarian_gif(out_path: Path, fps: int = DEFAULT_FPS):
    A, B, gt = twins_dataset()
    N = len(A)
    row, col, D = hungarian_matches(A, B)
    pred = np.empty(N, dtype=np.int64)
    pred[row] = col
    correct = int((pred == gt).sum())
    total_cost = D[row, col].sum()

    # ----------------------------------------------------------------------
    # Layout: scatter on the left, cost matrix heatmap on the right.
    # ----------------------------------------------------------------------
    fig = plt.figure(figsize=(12.0, 7.6), dpi=100)
    gs = fig.add_gridspec(
        1, 2, width_ratios=[1.0, 0.95],
        wspace=0.12, left=0.04, right=0.97, top=0.85, bottom=0.10,
    )
    ax = fig.add_subplot(gs[0])
    ax_m = fig.add_subplot(gs[1])
    setup_canvas(ax, span=4.2)

    sa = ax.scatter(A[:, 0], A[:, 1], s=140, c=A_COLOR, marker="o",
                    edgecolors="white", linewidths=1.5, zorder=4)
    sb = ax.scatter(B[:, 0], B[:, 1], s=140, c=B_COLOR, marker="D",
                    edgecolors="white", linewidths=1.5, zorder=4)

    # Match lines (drawn one by one).
    match_lines = []
    for k in range(N):
        ln, = ax.plot([], [], color="#363D45", lw=1.6, ls="-",
                      alpha=0.0, zorder=3)
        match_lines.append(ln)

    # GT verdict rings.
    ok_ring  = ax.scatter([], [], s=240, facecolors="none",
                          edgecolors=OK_COLOR, linewidths=2.4,
                          zorder=7, alpha=0.0)
    bad_ring = ax.scatter([], [], s=240, facecolors="none",
                          edgecolors=BAD_COLOR, linewidths=2.4,
                          zorder=7, alpha=0.0)

    # ---- Cost matrix axis ----
    from matplotlib.colors import LinearSegmentedColormap
    cost_cmap = LinearSegmentedColormap.from_list(
        "cost", ["#FFF6E5", "#FFCFA5", "#F28D35", "#C2417C",
                 "#5B2A86", "#1B0F3B"]
    )
    img = ax_m.imshow(np.zeros((N, N)), cmap=cost_cmap,
                      vmin=0, vmax=D.max(), origin="lower", aspect="equal")
    ax_m.set_xticks([]); ax_m.set_yticks([])
    for s in ("left", "right", "top", "bottom"):
        ax_m.spines[s].set_visible(False)
    ax_m.set_title("cost matrix  $C_{ij} = \\|a_i - b_j\\|$",
                   fontsize=11, color=MUTE, pad=6, loc="left")
    ax_m.set_xlabel("$j$  (B index)", fontsize=10, color=FG)
    ax_m.set_ylabel("$i$  (A index)", fontsize=10, color=FG)

    # Highlight squares for selected (row, col) cells.
    sel_scat = ax_m.scatter([], [], s=80, marker="s",
                            facecolors="none", edgecolors="#1F1F1F",
                            linewidths=2.0, zorder=5, alpha=0.0)

    fig.text(0.5, 0.96, "Hungarian matching  ·  exact 1-to-1",
             ha="center", va="top", fontsize=15, fontweight="bold")
    sub = fig.text(0.5, 0.92, "", ha="center", va="top",
                   fontsize=11, color=MUTE)
    cap = fig.text(0.5, 0.025, "", ha="center", va="bottom",
                   fontsize=11, color=FG)

    legend_corner(fig, 0.04, 0.86)

    # Frame schedule.
    FILL_TICK = 2          # frames per cost row reveal
    PHASE1 = N * FILL_TICK + 12   # cost matrix fills row by row
    PHASE2 = N * 5         # selected cells light up + matching lines drawn
    PHASE3 = 50            # GT verdict
    PHASE4 = 30            # final hold
    TOTAL = PHASE1 + PHASE2 + PHASE3 + PHASE4

    # The Hungarian-selected cells are (row, col) pairs.  Render them in
    # row order so the audience sees the assignment unfold left to right.
    sel_pairs = list(zip(row, col))

    def update(f):
        # ---- Phase 1: fill cost matrix row by row ----
        if f < PHASE1:
            n_rows = min(N, f // FILL_TICK + 1)
            partial = np.zeros((N, N))
            partial[:n_rows] = D[:n_rows]
            # Mask the unfilled rows by setting them to vmax+ (will display
            # as zeros via vmin/vmax — but we want them blank-white).
            # Instead, set unfilled rows to NaN after constructing.
            partial[n_rows:] = np.nan
            img.set_data(partial)
            sel_scat.set_alpha(0.0)
            for ln in match_lines:
                ln.set_alpha(0.0)
            sub.set_text(r"build the cost matrix  $C_{ij}$  row by row")
            cap.set_text(f"row {n_rows}/{N} filled")
            ok_ring.set_alpha(0.0); bad_ring.set_alpha(0.0)
            return

        img.set_data(D)

        # ---- Phase 2: selected cells light up + matching lines ----
        if f < PHASE1 + PHASE2:
            t = (f - PHASE1) // 5    # which selection step
            t = min(N - 1, t)
            # Highlight selected cells 0..t.
            xs = [c for (_, c) in sel_pairs[:t + 1]]
            ys = [r for (r, _) in sel_pairs[:t + 1]]
            sel_scat.set_offsets(list(zip(xs, ys)))
            sel_scat.set_alpha(0.95)

            # Draw matching lines for the selected pairs.
            for k in range(N):
                if k <= t:
                    r_, c_ = sel_pairs[k]
                    match_lines[k].set_data([A[r_, 0], B[c_, 0]],
                                            [A[r_, 1], B[c_, 1]])
                    match_lines[k].set_alpha(0.85)
                else:
                    match_lines[k].set_alpha(0.0)
            partial_cost = D[row[:t + 1], col[:t + 1]].sum()
            sub.set_text("Hungarian picks one cell per row, one per column")
            cap.set_text(f"{t + 1}/{N} matched   ·   "
                         f"running cost = {partial_cost:.2f}")
            ok_ring.set_alpha(0.0); bad_ring.set_alpha(0.0)
            return

        # All matches drawn.
        for k, (r_, c_) in enumerate(sel_pairs):
            match_lines[k].set_data([A[r_, 0], B[c_, 0]],
                                    [A[r_, 1], B[c_, 1]])
            match_lines[k].set_alpha(0.85)
        sel_scat.set_offsets([[c, r] for r, c in sel_pairs])
        sel_scat.set_alpha(0.95)

        # ---- Phase 3: GT verdict ----
        if f < PHASE1 + PHASE2 + PHASE3:
            t = (f - PHASE1 - PHASE2 + 1) / PHASE3
            n_show = int(t * N)
            ok_pts  = []
            bad_pts = []
            for k in range(min(n_show, N)):
                if pred[k] == gt[k]:
                    ok_pts.append(A[k])
                else:
                    bad_pts.append(A[k])
            ok_ring.set_offsets(ok_pts if ok_pts else np.empty((0, 2)))
            bad_ring.set_offsets(bad_pts if bad_pts else np.empty((0, 2)))
            ok_ring.set_alpha(0.95); bad_ring.set_alpha(0.95)
            sub.set_text("vs. ground truth")
            cap.set_text(
                f"verdict:  {n_show}/{N} checked   ·   "
                f"total cost = {total_cost:.2f}"
            )
            return

        # ---- Phase 4: final hold ----
        ok_pts  = [A[k] for k in range(N) if pred[k] == gt[k]]
        bad_pts = [A[k] for k in range(N) if pred[k] != gt[k]]
        ok_ring.set_offsets(ok_pts if ok_pts else np.empty((0, 2)))
        bad_ring.set_offsets(bad_pts if bad_pts else np.empty((0, 2)))
        ok_ring.set_alpha(0.95); bad_ring.set_alpha(0.95)
        sub.set_text("Hungarian:  exact, optimal, but combinatorial "
                     "(not differentiable)")
        cap.set_text(
            f"correct twin matches:  {correct}/{N}   ·   "
            f"total cost = {total_cost:.2f}"
        )

    anim = animation.FuncAnimation(
        fig, update, frames=TOTAL, interval=1000 / fps, blit=False,
    )
    save_animation(anim, out_path, fps=fps)
    plt.close(fig)
    print(f"wrote {out_path}  ({TOTAL} frames, {fps} fps, no loop)")
def make_sinkhorn_gif(out_path: Path, fps: int = DEFAULT_FPS):
    A, B, gt = twins_dataset()
    N = len(A)
    D = np.linalg.norm(A[:, None] - B[None, :], axis=-1)

    # ε schedule:  sweep from large (blurry) to small (sharp).
    eps_values = np.concatenate([
        np.linspace(3.0, 0.08, 100),
        np.full(30, 0.08),          # hold at converged plan
    ])
    N_EPS = len(eps_values)
    # Pre-compute converged transport plan at each ε.
    plans = []
    for eps in eps_values:
        P, _ = sinkhorn(D, eps, n_iters=80)
        plans.append(P)

    # "Hard" assignment from smallest-ε plan.
    P_final = plans[-1]
    pred_sink = P_final.argmax(axis=1)
    correct_sink = int((pred_sink == gt).sum())

    # Hungarian for comparison.
    row_h, col_h, _ = hungarian_matches(A, B)
    pred_h = np.empty(N, dtype=np.int64); pred_h[row_h] = col_h
    correct_h = int((pred_h == gt).sum())

    # ------------------------------------------------------------------
    # Layout: scatter left, transport-plan heatmap right.
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(12.0, 7.6), dpi=100)
    gs = fig.add_gridspec(
        1, 2, width_ratios=[1.0, 0.95],
        wspace=0.12, left=0.04, right=0.97, top=0.82, bottom=0.10,
    )
    ax = fig.add_subplot(gs[0])
    ax_p = fig.add_subplot(gs[1])
    setup_canvas(ax, span=4.5)

    sa = ax.scatter(A[:, 0], A[:, 1], s=140, c=A_COLOR, marker="o",
                    edgecolors="white", linewidths=1.5, zorder=4)
    sb = ax.scatter(B[:, 0], B[:, 1], s=140, c=B_COLOR, marker="D",
                    edgecolors="white", linewidths=1.5, zorder=4)

    # Transport-flow lines: one per (i, j), alpha ∝ P[i, j].
    # For efficiency, only draw the top-K entries each frame.
    MAX_LINES = 60
    flow_lines = []
    for _ in range(MAX_LINES):
        ln, = ax.plot([], [], color="#363D45", lw=1.5, alpha=0.0, zorder=3)
        flow_lines.append(ln)

    # Transport plan heatmap.
    from matplotlib.colors import LinearSegmentedColormap
    plan_cmap = LinearSegmentedColormap.from_list(
        "plan", ["#FAFAF7", "#D4E2F0", "#6EA8D9", "#2260A3", "#0B2545"]
    )
    img = ax_p.imshow(np.zeros((N, N)), cmap=plan_cmap,
                      vmin=0, vmax=0.1, origin="lower", aspect="equal")
    ax_p.set_xticks([]); ax_p.set_yticks([])
    for s in ("left", "right", "top", "bottom"):
        ax_p.spines[s].set_visible(False)
    ax_p.set_title("transport plan  $P$", fontsize=11,
                   color=MUTE, pad=6, loc="left")
    ax_p.set_xlabel("$j$  (B index)", fontsize=10, color=FG)
    ax_p.set_ylabel("$i$  (A index)", fontsize=10, color=FG)

    # GT verdict rings (final phase).
    ok_ring  = ax.scatter([], [], s=240, facecolors="none",
                          edgecolors=OK_COLOR, linewidths=2.4,
                          zorder=7, alpha=0.0)
    bad_ring = ax.scatter([], [], s=240, facecolors="none",
                          edgecolors=BAD_COLOR, linewidths=2.4,
                          zorder=7, alpha=0.0)

    fig.text(0.5, 0.96,
             "Sinkhorn matching  ·  soft, differentiable transport",
             ha="center", va="top", fontsize=15, fontweight="bold")
    sub = fig.text(0.5, 0.92, "", ha="center", va="top",
                   fontsize=11, color=MUTE)
    cap = fig.text(0.5, 0.025, "", ha="center", va="bottom",
                   fontsize=11, color=FG)
    eps_label = ax_p.text(0.98, 0.96, "", transform=ax_p.transAxes,
                          ha="right", va="top", fontsize=13, color=FG,
                          fontweight="bold",
                          bbox=dict(boxstyle="round,pad=0.3",
                                    fc="white", ec=GRID, lw=0.9))

    legend_corner(fig, 0.04, 0.86)

    # Frame schedule:
    #   Phase 1: ε-sweep (N_EPS frames)
    #   Phase 2: verdict (40 frames)
    #   Phase 3: hold (30 frames)
    PHASE1 = N_EPS
    PHASE2 = 40
    PHASE3 = 30
    TOTAL = PHASE1 + PHASE2 + PHASE3

    def draw_plan(P, eps):
        """Update scatter-flow lines and heatmap for transport plan P."""
        # Heatmap.
        img.set_data(P)
        img.set_clim(0, max(P.max() * 0.8, 1e-6))
        eps_label.set_text(rf"$\varepsilon = {eps:.2f}$")

        # Flow lines: draw the top-K entries of P.
        flat_idx = np.argsort(P.ravel())[::-1][:MAX_LINES]
        for k, fi in enumerate(flat_idx):
            i, j = divmod(int(fi), N)
            flow_lines[k].set_data([A[i, 0], B[j, 0]],
                                   [A[i, 1], B[j, 1]])
            mass = float(P[i, j])
            flow_lines[k].set_alpha(min(1.0, mass * N * 3.0))
            flow_lines[k].set_linewidth(max(0.6, min(3.5, mass * N * 8)))

    def update(f):
        ok_ring.set_alpha(0.0); bad_ring.set_alpha(0.0)
        # ---- Phase 1: ε-sweep ----
        if f < PHASE1:
            k = f
            eps = eps_values[k]
            P = plans[k]
            draw_plan(P, eps)
            # Is the plan roughly permutation-like?
            row_sum_dev = np.abs(P.sum(axis=1) - 1.0 / N).mean()
            if eps > 1.0:
                sub.set_text(r"large $\varepsilon$  →  blurry (nearly "
                             "uniform) plan")
            elif eps > 0.3:
                sub.set_text(r"medium $\varepsilon$  →  structure "
                             "emerging")
            else:
                sub.set_text(r"small $\varepsilon$  →  sharp, "
                             "permutation-like plan")
            cap.set_text(
                rf"entropic regularisation:  $\varepsilon = {eps:.2f}$   ·   "
                f"max $P_{{ij}}$ = {P.max():.3f}"
            )
            return

        # ---- Phase 2: verdict ----
        draw_plan(plans[-1], eps_values[-1])
        if f < PHASE1 + PHASE2:
            t = (f - PHASE1 + 1) / PHASE2
            n_show = int(t * N)
            ok_pts  = [A[k] for k in range(n_show)
                       if pred_sink[k] == gt[k]]
            bad_pts = [A[k] for k in range(n_show)
                       if pred_sink[k] != gt[k]]
            ok_ring.set_offsets(ok_pts if ok_pts else np.empty((0, 2)))
            bad_ring.set_offsets(bad_pts if bad_pts else np.empty((0, 2)))
            ok_ring.set_alpha(0.95); bad_ring.set_alpha(0.95)
            sub.set_text("vs. ground truth  (hard assignment from "
                         "argmax of plan)")
            cap.set_text(f"checking {n_show}/{N}")
            return

        # ---- Phase 3: hold ----
        ok_pts  = [A[k] for k in range(N) if pred_sink[k] == gt[k]]
        bad_pts = [A[k] for k in range(N) if pred_sink[k] != gt[k]]
        ok_ring.set_offsets(ok_pts if ok_pts else np.empty((0, 2)))
        bad_ring.set_offsets(bad_pts if bad_pts else np.empty((0, 2)))
        ok_ring.set_alpha(0.95); bad_ring.set_alpha(0.95)
        sub.set_text("Sinkhorn: soft, differentiable — "
                     rf"approaches Hungarian as $\varepsilon \to 0$")
        cap.set_text(
            f"correct: {correct_sink}/{N}   "
            f"(Hungarian got {correct_h}/{N})   ·   "
            f"but Sinkhorn is differentiable"
        )

    anim = animation.FuncAnimation(
        fig, update, frames=TOTAL, interval=1000 / fps, blit=False,
    )
    save_animation(anim, out_path, fps=fps)
    plt.close(fig)
    print(f"wrote {out_path}  ({TOTAL} frames, {fps} fps, no loop)")
def make_synthesis_gif(out_path: Path, fps: int = DEFAULT_FPS):
    A, B, gt = twins_dataset()
    N = len(A)
    D = np.linalg.norm(A[:, None] - B[None, :], axis=-1)

    # --- All three methods ---
    a_to_b_ch, _, _ = chamfer_matches(A, B)
    correct_ch = int((a_to_b_ch == gt).sum())

    row_h, col_h, _ = hungarian_matches(A, B)
    pred_h = np.empty(N, dtype=np.int64); pred_h[row_h] = col_h
    correct_h = int((pred_h == gt).sum())

    P_sink, _ = sinkhorn(D, eps=0.08, n_iters=80)
    pred_s = P_sink.argmax(axis=1)
    correct_s = int((pred_s == gt).sum())

    # ------------------------------------------------------------------
    # Layout: 2×2 grid of square scatter plots + footer.
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(12.0, 13.0), dpi=90)
    gs = fig.add_gridspec(
        2, 2, wspace=0.08, hspace=0.14,
        left=0.04, right=0.96, top=0.88, bottom=0.07,
    )
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    titles = [
        f"ground truth  (20/20)",
        f"Chamfer  ({correct_ch}/20 correct)",
        f"Hungarian  ({correct_h}/20 correct)",
        f"Sinkhorn  ({correct_s}/20 correct)",
    ]
    matching_data = [
        ("gt", gt),
        ("chamfer", a_to_b_ch),
        ("hungarian", pred_h),
        ("sinkhorn", pred_s),
    ]

    for idx, ax_i in enumerate(axes):
        setup_canvas(ax_i, span=4.5)
        ax_i.scatter(A[:, 0], A[:, 1], s=90, c=A_COLOR, marker="o",
                     edgecolors="white", linewidths=1.2, zorder=4)
        ax_i.scatter(B[:, 0], B[:, 1], s=90, c=B_COLOR, marker="D",
                     edgecolors="white", linewidths=1.2, zorder=4)

        mtype, pred = matching_data[idx]
        for i in range(N):
            if mtype == "gt":
                col = GT_COLOR
            elif pred[i] == gt[i]:
                col = OK_COLOR
            else:
                col = BAD_COLOR
            ax_i.plot([A[i, 0], B[pred[i], 0]],
                      [A[i, 1], B[pred[i], 1]],
                      color=col, lw=1.3, alpha=0.85, zorder=3)
        ax_i.set_title(titles[idx], fontsize=12, color=FG, pad=6,
                       loc="left", fontweight="bold")

    fig.text(0.5, 0.965, "Twin matching — four methods compared",
             ha="center", va="top", fontsize=16, fontweight="bold")
    fig.text(0.5, 0.93,
             "green = correct twin,  red = wrong twin,  "
             "grey = ground truth",
             ha="center", va="top", fontsize=11, color=MUTE)
    fig.text(0.5, 0.03,
             "Chamfer is cheap but greedy (many-to-one).  "
             "Hungarian is optimal but combinatorial.  "
             "Sinkhorn is soft and differentiable — "
             r"approaches Hungarian as $\varepsilon \to 0$.",
             ha="center", va="bottom", fontsize=11, color=FG,
             wrap=True)

    # This synthesis is a single-frame "GIF" that holds for a few seconds.
    TOTAL = 70   # ~5 s hold
    anim = animation.FuncAnimation(
        fig, lambda f: None, frames=TOTAL,
        interval=1000 / fps, blit=False,
    )
    save_animation(anim, out_path, fps=fps)
    plt.close(fig)
    print(f"wrote {out_path}  ({TOTAL} frames, {fps} fps, no loop)")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

BUILDERS = {
    "dataset":   ("02a_twins_dataset.gif", make_twins_dataset_gif),
    "chamfer":   ("02b_chamfer.gif",        make_chamfer_gif),
    "hungarian": ("02c_hungarian.gif",      make_hungarian_gif),
    "sinkhorn":  ("02d_sinkhorn.gif",       make_sinkhorn_gif),
    "synthesis": ("02e_synthesis.gif",      make_synthesis_gif),
}


def main(argv):
    targets = argv[1:] if len(argv) > 1 else ["dataset"]
    if targets == ["all"]:
        targets = list(BUILDERS)
    for t in targets:
        if t not in BUILDERS:
            raise SystemExit(f"unknown target {t!r}; "
                             f"choose from {list(BUILDERS)}")
        fname, fn = BUILDERS[t]
        fn(OUT_DIR / fname)


if __name__ == "__main__":
    main(sys.argv)
