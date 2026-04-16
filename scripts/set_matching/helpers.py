"""Shared helpers for the set-matching visualisation scenes."""

from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch
from scipy.optimize import linear_sum_assignment

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from metric_learning import (  # noqa: E402
    PillowWriterNoLoop, save_animation, DEFAULT_FPS,
    BG, FG, MUTE, GRID, _ease_inout,
)

# -- Colours -----------------------------------------------------------------
A_COLOR  = "#E85D4A"
B_COLOR  = "#2A9D8F"
GT_COLOR = "#888888"
OK_COLOR = "#2E8B57"
BAD_COLOR = "#C62828"
GREY = np.array([0.54, 0.56, 0.60, 1.0])

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "assets" / "set_matching"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -- Pair colours ------------------------------------------------------------
def pair_colors(n):
    return [np.array(plt.cm.tab20(i / n)) for i in range(n)]

def b_pair_colors(gt, pcols):
    """Return colours for B ordered by B-index (not pair-index)."""
    out = [None] * len(gt)
    for i, j in enumerate(gt):
        out[j] = pcols[i]
    return out

def fade_to_grey(cols, t):
    return [c * (1 - t) + GREY * t for c in cols]

# -- Canvas ------------------------------------------------------------------
def setup_canvas(ax, span=4.5):
    ax.set_xlim(-span, span); ax.set_ylim(-span, span)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ("left", "right", "top", "bottom"):
        ax.spines[s].set_visible(False)

# -- Dataset -----------------------------------------------------------------
def twins_dataset(n_pairs=20, sigma=0.35, seed=1):
    rng = np.random.default_rng(seed)
    base = rng.uniform(-2.9, 2.9, size=(n_pairs, 2))
    for _ in range(80):
        for i in range(n_pairs):
            for j in range(n_pairs):
                if i == j: continue
                d = base[i] - base[j]
                r = np.linalg.norm(d) + 1e-9
                if r < 0.95:
                    base[i] += 0.04 * d / r
                    base[j] -= 0.04 * d / r
    base[1]  = base[0]  + np.array([ 0.50, -0.25])
    base[5]  = base[4]  + np.array([-0.40,  0.42])
    base[9]  = base[8]  + np.array([ 0.35,  0.45])
    base[13] = base[12] + np.array([-0.55, -0.20])
    A = base + rng.normal(0, sigma, size=(n_pairs, 2))
    B = base + rng.normal(0, sigma, size=(n_pairs, 2))
    perm = rng.permutation(n_pairs)
    B = B[perm]
    gt = np.argsort(perm)
    return A, B, gt

# -- Algorithm helpers -------------------------------------------------------
def chamfer_matches(A, B):
    Da = np.linalg.norm(A[:, None] - B[None, :], axis=-1)
    return Da.argmin(axis=1), Da.T.argmin(axis=1), Da

def hungarian_matches(A, B):
    Da = np.linalg.norm(A[:, None] - B[None, :], axis=-1)
    row, col = linear_sum_assignment(Da)
    return row, col, Da

def sinkhorn(C, eps, n_iters=80, tol=1e-9):
    n, m = C.shape
    a, b = np.full(n, 1.0/n), np.full(m, 1.0/m)
    K = np.exp(-C / eps)
    u, v = np.ones(n), np.ones(m)
    plans = []
    for it in range(n_iters):
        u = a / (K @ v + 1e-30)
        v = b / (K.T @ u + 1e-30)
        P = (u[:, None] * K) * v[None, :]
        plans.append(P.copy())
        if it > 5 and np.linalg.norm(P.sum(axis=1) - a) < tol:
            break
    return P, plans

# -- Optimisation ------------------------------------------------------------
def optimise_B(A, B_init, method="chamfer", n_steps=40, lr=0.08,
               eps_sink=0.15):
    B = B_init.copy()
    N = len(A)
    snaps = [(B.copy(), None)]
    for step in range(n_steps):
        grad = np.zeros_like(B)
        if method == "chamfer":
            a_to_b, _, _ = chamfer_matches(A, B)
            matching = a_to_b
            for i in range(N):
                grad[a_to_b[i]] += 2 * (B[a_to_b[i]] - A[i])
        elif method == "hungarian":
            row, col, _ = hungarian_matches(A, B)
            matching = np.empty(N, dtype=np.int64); matching[row] = col
            for r, c in zip(row, col):
                grad[c] += 2 * (B[c] - A[r])
        elif method == "sinkhorn":
            D = np.linalg.norm(A[:, None] - B[None, :], axis=-1)
            P, _ = sinkhorn(D, eps=eps_sink)
            matching = P.argmax(axis=1)
            for j in range(N):
                for i in range(N):
                    grad[j] += 2 * P[i, j] * (B[j] - A[i])
        B = B - lr * grad
        snaps.append((B.copy(), matching.copy()))
    return snaps
