"""04c — Diffusion self-supervised training.

Shows progressive noise addition (forward process) and learned
denoising (reverse process) on a 2D point cloud / simple image.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, FancyArrowPatch
from .helpers import (save_animation, DEFAULT_FPS, FG, MUTE, BG,
                      PREDICT_COLOR)

# We'll use a simple 2D smiley face made of points.
def _make_smiley(n=200, seed=42):
    """Generate a 2D point cloud in the shape of a smiley face."""
    rng = np.random.default_rng(seed)
    pts = []
    # Outer circle (face outline).
    for _ in range(80):
        angle = rng.uniform(0, 2 * np.pi)
        r = 2.0 + rng.normal(0, 0.05)
        pts.append([r * np.cos(angle), r * np.sin(angle)])
    # Left eye.
    for _ in range(25):
        pts.append([-0.7 + rng.normal(0, 0.12),
                     0.7 + rng.normal(0, 0.12)])
    # Right eye.
    for _ in range(25):
        pts.append([0.7 + rng.normal(0, 0.12),
                     0.7 + rng.normal(0, 0.12)])
    # Mouth (arc).
    for _ in range(50):
        angle = rng.uniform(-2.5, -0.6)
        r = 1.2 + rng.normal(0, 0.06)
        pts.append([r * np.cos(angle), r * np.sin(angle)])
    # Fill remaining.
    while len(pts) < n:
        angle = rng.uniform(0, 2 * np.pi)
        r = 2.0 + rng.normal(0, 0.05)
        pts.append([r * np.cos(angle), r * np.sin(angle)])
    return np.array(pts[:n])

N_STEPS = 8  # diffusion steps to show
NOISE_SCHEDULE = np.linspace(0, 1.0, N_STEPS + 1)  # beta cumulative


def render(out_path: Path, fps: int = DEFAULT_FPS):
    x0 = _make_smiley()
    N_pts = len(x0)
    rng = np.random.default_rng(7)

    # Pre-compute noisy versions.
    noise = rng.normal(0, 1, size=(N_STEPS + 1, N_pts, 2))
    x_noisy = []
    for t in range(N_STEPS + 1):
        alpha = 1.0 - NOISE_SCHEDULE[t]
        xt = np.sqrt(alpha) * x0 + np.sqrt(1 - alpha) * noise[t] * 2.5
        x_noisy.append(xt)

    fig = plt.figure(figsize=(14.0, 7.8), dpi=100)

    # Layout: top row = forward process (small panels), bottom = big view.
    gs_top = fig.add_gridspec(1, N_STEPS + 1,
                              left=0.04, right=0.96,
                              top=0.82, bottom=0.52, wspace=0.08)
    gs_bot = fig.add_gridspec(1, 2,
                              left=0.08, right=0.92,
                              top=0.44, bottom=0.08, wspace=0.15)

    # Small panels for the forward process.
    ax_fwd = []
    for i in range(N_STEPS + 1):
        ax = fig.add_subplot(gs_top[i])
        ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ("left", "right", "top", "bottom"):
            ax.spines[s].set_visible(False)
        ax.set_facecolor(BG)
        ax_fwd.append(ax)

    # Two big panels at bottom.
    ax_big_noisy = fig.add_subplot(gs_bot[0])
    ax_big_clean = fig.add_subplot(gs_bot[1])
    for ax in (ax_big_noisy, ax_big_clean):
        ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ("left", "right", "top", "bottom"):
            ax.spines[s].set_visible(False)

    fig.text(0.5, 0.96, "diffusion — self-supervised training",
             ha="center", va="top", fontsize=15, fontweight="bold")
    sub = fig.text(0.5, 0.92, "", ha="center", va="top",
                   fontsize=11, color=MUTE)
    cap = fig.text(0.5, 0.025, "", ha="center", va="bottom",
                   fontsize=11, color=FG)

    # Pre-scatter objects (hidden initially).
    fwd_scats = []
    for i in range(N_STEPS + 1):
        sc = ax_fwd[i].scatter([], [], s=8, alpha=0.0, zorder=2)
        fwd_scats.append(sc)

    fwd_titles = []
    for i in range(N_STEPS + 1):
        txt = ax_fwd[i].set_title("", fontsize=8, color=MUTE, pad=2)
        fwd_titles.append(txt)

    # Arrow labels between small panels.
    arrow_texts = []
    for i in range(N_STEPS):
        txt = fig.text(0.04 + (i + 0.5) / (N_STEPS + 1) * 0.92 + 0.92/(N_STEPS+1)*0.5,
                       0.835, "", ha="center", fontsize=10,
                       color=MUTE, alpha=0.0)
        arrow_texts.append(txt)

    big_noisy_sc = ax_big_noisy.scatter([], [], s=20, alpha=0.0, zorder=2)
    big_clean_sc = ax_big_clean.scatter([], [], s=20, alpha=0.0, zorder=2)

    # ---- Frame schedule ----
    HOLD = 70

    steps = []
    steps.append(("intro",        HOLD))
    steps.append(("clean_data",   HOLD))
    # Forward process: add noise step by step.
    for t in range(1, N_STEPS + 1):
        steps.append((f"forward_{t}", 45))
    steps.append(("full_noise",   HOLD))
    steps.append(("reverse_intro", HOLD))
    # Reverse process: denoise step by step.
    for t in range(N_STEPS, 0, -1):
        steps.append((f"reverse_{t}", 45))
    steps.append(("recovered",    HOLD))
    steps.append(("training",     HOLD))
    steps.append(("loss_explain", HOLD))
    steps.append(("summary",      60))

    frame_to_step = []
    step_offsets = {}
    for sname, dur in steps:
        step_offsets[sname] = len(frame_to_step)
        for _ in range(dur):
            frame_to_step.append(sname)
    TOTAL = len(frame_to_step)

    def _step_t(f, step_name):
        off = step_offsets[step_name]
        dur = dict(steps)[step_name]
        return min(1.0, (f - off + 1) / dur)

    # Colour maps: clean = teal, noisy = red.
    cmap_clean = "#2A9D8F"
    cmap_noisy = "#E85D4A"
    cmap_pred  = "#6E63B5"

    def _noise_color(t_idx):
        """Blend from teal (clean) to red (noisy)."""
        frac = t_idx / N_STEPS
        r = int(42 + (232 - 42) * frac)
        g = int(157 + (93 - 157) * frac)
        b = int(143 + (74 - 143) * frac)
        return f"#{r:02x}{g:02x}{b:02x}"

    def _show_forward(up_to):
        """Show forward process panels 0..up_to."""
        for i in range(N_STEPS + 1):
            if i <= up_to:
                fwd_scats[i].set_offsets(x_noisy[i])
                fwd_scats[i].set_alpha(0.7)
                fwd_scats[i].set_color(_noise_color(i))
                ax_fwd[i].set_title(f"t={i}" if i > 0 else "t=0\n(clean)",
                                    fontsize=8, color=MUTE, pad=2)
            else:
                fwd_scats[i].set_alpha(0.0)
                ax_fwd[i].set_title("", fontsize=8, pad=2)
        for i in range(min(up_to, N_STEPS)):
            arrow_texts[i].set_text("→")
            arrow_texts[i].set_alpha(0.6)

    def _clear_big():
        big_noisy_sc.set_alpha(0.0)
        big_clean_sc.set_alpha(0.0)
        ax_big_noisy.set_title("", fontsize=11, color=MUTE, pad=6)
        ax_big_clean.set_title("", fontsize=11, color=MUTE, pad=6)

    def update(f):
        step = frame_to_step[min(f, TOTAL - 1)]
        t = _step_t(f, step)
        for a in arrow_texts:
            a.set_alpha(0.0)
        _clear_big()

        # ===== Intro =====
        if step == "intro":
            sub.set_text("idea: learn to reverse a gradual noising process")
            cap.set_text("no labels needed — the noise schedule IS "
                         "the supervision")
            return

        # ===== Clean data =====
        if step == "clean_data":
            _show_forward(0)
            big_noisy_sc.set_offsets(x0)
            big_noisy_sc.set_color(cmap_clean)
            big_noisy_sc.set_alpha(0.7)
            ax_big_noisy.set_title("clean data x₀",
                                   fontsize=11, color=MUTE, pad=6)
            sub.set_text("start with clean data (a 2D smiley face)")
            cap.set_text("this is x₀ — the data we want to learn "
                         "to generate")
            return

        # ===== Forward process steps =====
        for ti in range(1, N_STEPS + 1):
            if step == f"forward_{ti}":
                _show_forward(ti)
                big_noisy_sc.set_offsets(x_noisy[ti])
                big_noisy_sc.set_color(_noise_color(ti))
                big_noisy_sc.set_alpha(0.7)
                ax_big_noisy.set_title(f"x_{{t={ti}}} (noised)",
                                       fontsize=11, color=MUTE, pad=6)
                sub.set_text(f"forward step {ti}/{N_STEPS}: "
                             f"add Gaussian noise "
                             f"(noise level = {NOISE_SCHEDULE[ti]:.0%})")
                cap.set_text("x_t = √α_t · x₀ + √(1−α_t) · ε   "
                             "where ε ~ N(0, I)")
                return

        # ===== Full noise =====
        if step == "full_noise":
            _show_forward(N_STEPS)
            big_noisy_sc.set_offsets(x_noisy[N_STEPS])
            big_noisy_sc.set_color(cmap_noisy)
            big_noisy_sc.set_alpha(0.7)
            ax_big_noisy.set_title(f"x_{{T}} ≈ pure noise",
                                   fontsize=11, color=MUTE, pad=6)
            sub.set_text("at t=T the signal is completely destroyed "
                         "— just Gaussian noise")
            cap.set_text("forward process: deterministic, requires "
                         "no learning")
            return

        # ===== Reverse intro =====
        if step == "reverse_intro":
            _show_forward(N_STEPS)
            big_noisy_sc.set_offsets(x_noisy[N_STEPS])
            big_noisy_sc.set_color(cmap_noisy)
            big_noisy_sc.set_alpha(0.7)
            ax_big_noisy.set_title("noisy input",
                                   fontsize=11, color=MUTE, pad=6)
            ax_big_clean.set_title("model output (denoised)",
                                   fontsize=11, color=MUTE, pad=6)
            sub.set_text("the model learns to REVERSE each step: "
                         "given x_t, predict x_{t-1}")
            cap.set_text("this is the learned part — "
                         "a neural network predicts the noise to subtract")
            return

        # ===== Reverse process steps =====
        for ti in range(N_STEPS, 0, -1):
            if step == f"reverse_{ti}":
                _show_forward(N_STEPS)
                # Show current noisy state on left.
                big_noisy_sc.set_offsets(x_noisy[ti])
                big_noisy_sc.set_color(_noise_color(ti))
                big_noisy_sc.set_alpha(0.7)
                ax_big_noisy.set_title(f"x_{{t={ti}}}",
                                       fontsize=11, color=MUTE, pad=6)
                # Show denoised output on right.
                big_clean_sc.set_offsets(x_noisy[ti - 1])
                big_clean_sc.set_color(_noise_color(ti - 1))
                big_clean_sc.set_alpha(0.7)
                ax_big_clean.set_title(f"→ x_{{t={ti-1}}}  (denoised)",
                                       fontsize=11, color=MUTE, pad=6)
                done = N_STEPS - ti + 1
                sub.set_text(f"reverse step {done}/{N_STEPS}: "
                             f"denoise from t={ti} → t={ti-1}")
                cap.set_text("model predicts noise ε̂, then "
                             "x_{t-1} = (x_t − β_t · ε̂) / √α_t")
                return

        # ===== Recovered =====
        if step == "recovered":
            _show_forward(N_STEPS)
            big_noisy_sc.set_offsets(x_noisy[N_STEPS])
            big_noisy_sc.set_color(cmap_noisy)
            big_noisy_sc.set_alpha(0.4)
            ax_big_noisy.set_title("started from pure noise",
                                   fontsize=11, color=MUTE, pad=6)
            big_clean_sc.set_offsets(x0)
            big_clean_sc.set_color(cmap_clean)
            big_clean_sc.set_alpha(0.7)
            ax_big_clean.set_title("recovered clean data x₀ ✓",
                                   fontsize=11, color=MUTE, pad=6)
            sub.set_text("after T denoising steps: "
                         "noise → clean data (generation!)")
            cap.set_text("at inference: sample x_T ~ N(0,I), "
                         "then run the learned reverse chain")
            return

        # ===== Training objective =====
        if step == "training":
            _show_forward(N_STEPS)
            big_noisy_sc.set_offsets(x_noisy[3])
            big_noisy_sc.set_color(_noise_color(3))
            big_noisy_sc.set_alpha(0.7)
            ax_big_noisy.set_title("x_t  (random t, random noise)",
                                   fontsize=11, color=MUTE, pad=6)
            big_clean_sc.set_alpha(0.0)
            ax_big_clean.set_title("",
                                   fontsize=11, color=MUTE, pad=6)
            sub.set_text("training: sample random t, random noise ε, "
                         "compute x_t, predict ε")
            cap.set_text("loss = ||ε − ε̂(x_t, t)||²   "
                         "(simple MSE on the noise)")
            return

        # ===== Loss explanation =====
        if step == "loss_explain":
            _show_forward(N_STEPS)
            sub.set_text("1. sample x₀ from data\n"
                         "2. sample t ~ Uniform(1..T) and ε ~ N(0,I)\n"
                         "3. compute x_t = √ᾱ_t · x₀ + √(1−ᾱ_t) · ε\n"
                         "4. train model to predict ε from (x_t, t)")
            cap.set_text("self-supervised: no labels — "
                         "the noise ε IS the target")
            return

        # ===== Summary =====
        _show_forward(N_STEPS)
        big_clean_sc.set_offsets(x0)
        big_clean_sc.set_color(cmap_clean)
        big_clean_sc.set_alpha(0.7)
        ax_big_clean.set_title("generates new samples",
                               fontsize=11, color=MUTE, pad=6)
        big_noisy_sc.set_offsets(x_noisy[N_STEPS])
        big_noisy_sc.set_color(cmap_noisy)
        big_noisy_sc.set_alpha(0.4)
        ax_big_noisy.set_title("from pure noise",
                               fontsize=11, color=MUTE, pad=6)
        sub.set_text("learn to reverse gradual noising — "
                     "no labels needed, just data + noise schedule")
        cap.set_text("DDPM, DDIM, stable diffusion, DALL·E 2, ... — "
                     "foundation of modern generative models")

    anim = animation.FuncAnimation(fig, update, frames=TOTAL,
                                   interval=1000/fps, blit=False)
    save_animation(anim, out_path, fps=fps)
    plt.close(fig)
    print(f"wrote {out_path}  ({TOTAL} frames, {fps} fps)")
