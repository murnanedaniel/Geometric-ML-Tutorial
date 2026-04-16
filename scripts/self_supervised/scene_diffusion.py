"""04c — Flow matching (modern diffusion) self-supervised training.

Shows straight-line interpolation paths from noise to data,
the velocity field the model learns, and the connection to
denoising: predicting v = x₀ − ε is the same as predicting
the clean data x̂₀ = xₜ + (1−t)·v.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, FancyArrow
from .helpers import (save_animation, DEFAULT_FPS, FG, MUTE, BG,
                      PREDICT_COLOR)


def _make_smiley(n=150, seed=42):
    rng = np.random.default_rng(seed)
    pts = []
    for _ in range(60):
        angle = rng.uniform(0, 2 * np.pi)
        r = 2.0 + rng.normal(0, 0.05)
        pts.append([r * np.cos(angle), r * np.sin(angle)])
    for _ in range(20):
        pts.append([-0.7 + rng.normal(0, 0.12), 0.7 + rng.normal(0, 0.12)])
    for _ in range(20):
        pts.append([0.7 + rng.normal(0, 0.12), 0.7 + rng.normal(0, 0.12)])
    for _ in range(40):
        angle = rng.uniform(-2.5, -0.6)
        r = 1.2 + rng.normal(0, 0.06)
        pts.append([r * np.cos(angle), r * np.sin(angle)])
    while len(pts) < n:
        angle = rng.uniform(0, 2 * np.pi)
        r = 2.0 + rng.normal(0, 0.05)
        pts.append([r * np.cos(angle), r * np.sin(angle)])
    return np.array(pts[:n])


def _noise_color(t_val):
    """Blend teal (t=1, clean) → red (t=0, noisy)."""
    frac = 1.0 - t_val
    r = int(42 + (232 - 42) * frac)
    g = int(157 + (93 - 157) * frac)
    b = int(143 + (74 - 143) * frac)
    return f"#{r:02x}{g:02x}{b:02x}"


CMAP_CLEAN = "#2A9D8F"
CMAP_NOISY = "#E85D4A"
CMAP_VEL   = "#6E63B5"
CMAP_PRED  = "#2E8B57"


def render(out_path: Path, fps: int = DEFAULT_FPS):
    x0 = _make_smiley(n=100, seed=42)
    N_pts = len(x0)
    rng = np.random.default_rng(7)
    eps = rng.normal(0, 1, size=(N_pts, 2)) * 2.5  # noise endpoints

    # Flow path: x_t = (1-t)*eps + t*x0, t in [0,1].
    # Velocity: v = x0 - eps (constant per particle).
    vel = x0 - eps

    def x_at_t(t):
        return (1 - t) * eps + t * x0

    # Pick a small subset for path visualisation.
    show_idx = rng.choice(N_pts, size=12, replace=False)

    fig = plt.figure(figsize=(14.0, 7.8), dpi=100)

    # Layout: one big panel (left) for the flow, one right panel for
    # equations / zoomed views.
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.0], wspace=0.08,
                          left=0.04, right=0.96, top=0.82, bottom=0.08)
    ax = fig.add_subplot(gs[0])
    ax_r = fig.add_subplot(gs[1])

    span = 5.0
    for a in (ax, ax_r):
        a.set_xlim(-span, span); a.set_ylim(-span, span)
        a.set_aspect("equal")
        a.set_xticks([]); a.set_yticks([])
        for s in ("left", "right", "top", "bottom"):
            a.spines[s].set_visible(False)

    fig.text(0.5, 0.96, "flow matching — self-supervised training",
             ha="center", va="top", fontsize=15, fontweight="bold")
    sub = fig.text(0.5, 0.92, "", ha="center", va="top",
                   fontsize=11, color=MUTE)
    cap = fig.text(0.5, 0.025, "", ha="center", va="bottom",
                   fontsize=11, color=FG)

    # Persistent scatter for the point cloud.
    scat = ax.scatter([], [], s=18, alpha=0.0, zorder=3)

    # Path lines (for the subset).
    path_lines = []
    for _ in show_idx:
        ln, = ax.plot([], [], color=CMAP_VEL, lw=0.8, alpha=0.0, zorder=1)
        path_lines.append(ln)

    # Velocity arrows (quiver — drawn fresh each frame).
    quiv = [None]

    # Right panel scatter.
    scat_r = ax_r.scatter([], [], s=18, alpha=0.0, zorder=3)

    # ---- Frame schedule ----
    HOLD = 70

    steps = []
    steps.append(("intro",          HOLD))
    steps.append(("clean_data",     HOLD))
    steps.append(("noise",          HOLD))
    steps.append(("paths_idea",     HOLD))
    steps.append(("show_paths",     int(HOLD * 1.5)))
    steps.append(("velocity_intro", HOLD))
    steps.append(("velocity_field", HOLD))
    steps.append(("training_obj",   HOLD))
    steps.append(("flow_t0",        50))
    # Flow from t=0 to t=1 in smooth steps.
    N_FLOW = 10
    for fi in range(N_FLOW):
        steps.append((f"flow_{fi}", 25))
    steps.append(("flow_done",     HOLD))
    steps.append(("denoise_conn",  HOLD))
    steps.append(("denoise_show",  HOLD))
    steps.append(("equiv",         HOLD))
    steps.append(("inference",     HOLD))
    steps.append(("summary",       60))

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

    def _clear():
        scat.set_alpha(0.0)
        scat_r.set_alpha(0.0)
        for ln in path_lines:
            ln.set_alpha(0.0)
        if quiv[0] is not None:
            quiv[0].remove()
            quiv[0] = None
        ax.set_title("", fontsize=11, color=MUTE, pad=6)
        ax_r.set_title("", fontsize=11, color=MUTE, pad=6)
        # Clear right panel dynamic content.
        while len(ax_r.texts) > 0:
            ax_r.texts[-1].remove()

    def _show_cloud(t_val, on_ax=None):
        target = on_ax or ax
        xt = x_at_t(t_val)
        if target is ax:
            scat.set_offsets(xt)
            scat.set_color(_noise_color(t_val))
            scat.set_alpha(0.7)
        else:
            scat_r.set_offsets(xt)
            scat_r.set_color(_noise_color(t_val))
            scat_r.set_alpha(0.7)

    def _show_paths_up_to(t_val):
        n_seg = 40
        for k, idx in enumerate(show_idx):
            ts = np.linspace(0, t_val, n_seg)
            xs = (1 - ts) * eps[idx, 0] + ts * x0[idx, 0]
            ys = (1 - ts) * eps[idx, 1] + ts * x0[idx, 1]
            path_lines[k].set_data(xs, ys)
            path_lines[k].set_alpha(0.5)

    # Fixed subset for velocity arrows (deterministic for GIF dedup).
    vel_idx = np.linspace(0, N_pts - 1, 20, dtype=int)

    def _show_velocity(t_val, on_ax=None, subsample=20):
        """Draw velocity arrows at current positions."""
        target = on_ax or ax
        xt = x_at_t(t_val)
        idx = vel_idx[:subsample]
        if quiv[0] is not None:
            quiv[0].remove()
        quiv[0] = target.quiver(xt[idx, 0], xt[idx, 1],
                                vel[idx, 0], vel[idx, 1],
                                color=CMAP_VEL, alpha=0.5,
                                scale=25, width=0.004, zorder=2)

    def update(f):
        _clear()
        step = frame_to_step[min(f, TOTAL - 1)]
        t = _step_t(f, step)

        # ===== Intro =====
        if step == "intro":
            sub.set_text("a simpler, modern approach to generative modelling")
            cap.set_text("Lipman et al. 2022, Liu et al. 2022 — "
                         "used in Stable Diffusion 3, Flux, ...")
            return

        # ===== Clean data =====
        if step == "clean_data":
            _show_cloud(1.0)
            ax.set_title("clean data  x₀  (t = 1)",
                         fontsize=11, color=MUTE, pad=6)
            sub.set_text("start with data samples from the target "
                         "distribution")
            cap.set_text("a 2D smiley — we want to learn to "
                         "generate these")
            return

        # ===== Noise =====
        if step == "noise":
            _show_cloud(1.0)
            ax.set_title("data  x₀", fontsize=11, color=MUTE, pad=6)
            _show_cloud(0.0, on_ax=ax_r)
            ax_r.set_title("noise  ε ~ N(0, I)  (t = 0)",
                           fontsize=11, color=MUTE, pad=6)
            sub.set_text("sample noise ε from a simple Gaussian")
            cap.set_text("goal: learn a map from noise → data")
            return

        # ===== Paths idea =====
        if step == "paths_idea":
            _show_cloud(1.0)
            ax.set_title("data  x₀", fontsize=11, color=MUTE, pad=6)
            _show_cloud(0.0, on_ax=ax_r)
            ax_r.set_title("noise  ε", fontsize=11, color=MUTE, pad=6)
            sub.set_text("key idea: connect each noise point to its data "
                         "point with a straight line")
            cap.set_text("x_t = (1−t)·ε + t·x₀     "
                         "t ∈ [0, 1]")
            return

        # ===== Show paths =====
        if step == "show_paths":
            t_anim = min(1.0, t * 1.3)
            _show_paths_up_to(t_anim)
            _show_cloud(t_anim * 0.5 + 0.5 * (1 - t_anim))
            # Show endpoints.
            for k, idx in enumerate(show_idx):
                if t > 0.3:
                    ax.plot(eps[idx, 0], eps[idx, 1], "o",
                            color=CMAP_NOISY, ms=4, alpha=0.5, zorder=4)
                    ax.plot(x0[idx, 0], x0[idx, 1], "o",
                            color=CMAP_CLEAN, ms=4, alpha=0.5, zorder=4)
            ax.set_title("straight-line paths: noise → data",
                         fontsize=11, color=MUTE, pad=6)
            sub.set_text("each particle follows a straight line from "
                         "its noise sample to its data point")
            cap.set_text("x_t = (1−t)·ε + t·x₀  — "
                         "simple linear interpolation")
            return

        # ===== Velocity intro =====
        if step == "velocity_intro":
            _show_paths_up_to(1.0)
            _show_cloud(0.5)
            ax.set_title("what's the velocity along each path?",
                         fontsize=11, color=MUTE, pad=6)
            ax_r.text(0, 3.5, r"$x_t = (1-t)\,\epsilon + t\,x_0$",
                      fontsize=14, ha="center", color=FG)
            ax_r.text(0, 2.0, r"$\frac{dx_t}{dt} = x_0 - \epsilon$",
                      fontsize=16, ha="center", color=CMAP_VEL,
                      fontweight="bold")
            ax_r.text(0, 0.5, r"$v^* = x_0 - \epsilon$",
                      fontsize=14, ha="center", color=CMAP_VEL)
            ax_r.text(0, -1.0, "(constant velocity,\nstraight line)",
                      fontsize=11, ha="center", color=MUTE)
            ax_r.set_title("the velocity field",
                           fontsize=11, color=MUTE, pad=6)
            sub.set_text("derivative of x_t w.r.t. t = x₀ − ε — "
                         "constant along each path")
            cap.set_text("the model will learn to predict this "
                         "velocity from (x_t, t)")
            return

        # ===== Velocity field visualisation =====
        if step == "velocity_field":
            _show_cloud(0.5)
            _show_velocity(0.5, subsample=30)
            ax.set_title("velocity field at t = 0.5",
                         fontsize=11, color=MUTE, pad=6)
            ax_r.text(0, 3.0, "the model learns:", fontsize=12,
                      ha="center", color=MUTE)
            ax_r.text(0, 1.5, r"$v_\theta(x_t,\, t) \approx x_0 - \epsilon$",
                      fontsize=16, ha="center", color=CMAP_VEL,
                      fontweight="bold")
            ax_r.text(0, -0.5, "training loss:", fontsize=12,
                      ha="center", color=MUTE)
            ax_r.text(0, -2.0,
                      r"$\| v_\theta(x_t, t) - (x_0 - \epsilon) \|^2$",
                      fontsize=14, ha="center", color=FG)
            ax_r.set_title("learned velocity field",
                           fontsize=11, color=MUTE, pad=6)
            sub.set_text("purple arrows = true velocity at each point")
            cap.set_text("the neural network learns to predict these "
                         "arrows from (x_t, t) alone")
            return

        # ===== Training objective =====
        if step == "training_obj":
            ax_r.text(0, 4.0, "training recipe:", fontsize=13,
                      ha="center", color=FG, fontweight="bold")
            ax_r.text(0, 2.8, "1. sample x₀ from data", fontsize=11,
                      ha="center", color=FG)
            ax_r.text(0, 2.0, "2. sample ε ~ N(0, I)", fontsize=11,
                      ha="center", color=FG)
            ax_r.text(0, 1.2, "3. sample t ~ Uniform(0, 1)", fontsize=11,
                      ha="center", color=FG)
            ax_r.text(0, 0.4, "4. compute x_t = (1−t)ε + t·x₀",
                      fontsize=11, ha="center", color=FG)
            ax_r.text(0, -0.6, "5. predict v_θ(x_t, t)",
                      fontsize=11, ha="center", color=CMAP_VEL)
            ax_r.text(0, -1.8,
                      "6. loss = ||v_θ − (x₀ − ε)||²",
                      fontsize=13, ha="center", color=FG,
                      fontweight="bold")
            ax_r.text(0, -3.2, "self-supervised: no labels needed\n"
                      "x₀ and ε are the supervision",
                      fontsize=11, ha="center", color=MUTE)
            ax_r.set_title("the training loop",
                           fontsize=11, color=MUTE, pad=6)
            _show_cloud(0.4)
            _show_velocity(0.4, subsample=20)
            ax.set_title("x_t at random t = 0.4",
                         fontsize=11, color=MUTE, pad=6)
            sub.set_text("each training step: one random (x₀, ε, t) triple")
            cap.set_text("simpler than DDPM — no noise schedule, "
                         "no reverse Markov chain")
            return

        # ===== Flow animation: integrate from t=0 to t=1 =====
        if step == "flow_t0":
            _show_cloud(0.0)
            _show_velocity(0.0, subsample=25)
            ax.set_title("inference: start from pure noise (t = 0)",
                         fontsize=11, color=MUTE, pad=6)
            ax_r.text(0, 2.0, "inference:", fontsize=14,
                      ha="center", color=FG, fontweight="bold")
            ax_r.text(0, 0.5, "sample ε ~ N(0, I)\n"
                      "integrate v_θ from t=0 to t=1\n"
                      "(ODE solver)",
                      fontsize=12, ha="center", color=MUTE)
            ax_r.set_title("generation by ODE integration",
                           fontsize=11, color=MUTE, pad=6)
            sub.set_text("follow the learned velocity field from noise "
                         "to data")
            cap.set_text("no iterative denoising — just solve an ODE")
            return

        for fi in range(N_FLOW):
            if step == f"flow_{fi}":
                t_flow = (fi + 1) / N_FLOW
                _show_cloud(t_flow)
                _show_velocity(t_flow, subsample=20)
                _show_paths_up_to(t_flow)
                ax.set_title(f"flowing: t = {t_flow:.1f}",
                             fontsize=11, color=MUTE, pad=6)
                ax_r.text(0, 2.0,
                          f"t = {t_flow:.1f}",
                          fontsize=20, ha="center", color=FG,
                          fontweight="bold")
                if t_flow < 0.5:
                    ax_r.text(0, 0, "still mostly noise...",
                              fontsize=12, ha="center", color=MUTE)
                elif t_flow < 0.8:
                    ax_r.text(0, 0, "structure emerging!",
                              fontsize=12, ha="center", color=CMAP_VEL)
                else:
                    ax_r.text(0, 0, "almost clean data",
                              fontsize=12, ha="center", color=CMAP_CLEAN)
                sub.set_text("integrating the velocity field — "
                             "particles flow toward data")
                cap.set_text(f"x_{{t={t_flow:.1f}}} = "
                             f"x_{{t={max(0,t_flow-0.1):.1f}}} + "
                             f"Δt · v_θ(x_t, t)")
                return

        if step == "flow_done":
            _show_cloud(1.0)
            _show_paths_up_to(1.0)
            ax.set_title("t = 1.0 — generated data ✓",
                         fontsize=11, color=MUTE, pad=6)
            ax_r.text(0, 2.0, "noise → data", fontsize=18,
                      ha="center", color=CMAP_CLEAN, fontweight="bold")
            ax_r.text(0, 0.5, "by following straight-line\n"
                      "velocity field", fontsize=12,
                      ha="center", color=MUTE)
            ax_r.set_title("generation complete",
                           fontsize=11, color=MUTE, pad=6)
            sub.set_text("pure noise transformed into clean data "
                         "by ODE integration")
            cap.set_text("the smiley face is recovered!")
            return

        # ===== Connection to denoising =====
        if step == "denoise_conn":
            _show_cloud(0.4)
            ax.set_title("x_t at t = 0.4  (partially noisy)",
                         fontsize=11, color=MUTE, pad=6)
            ax_r.text(0, 4.0, "connection to denoising:",
                      fontsize=13, ha="center", color=FG,
                      fontweight="bold")
            ax_r.text(0, 2.5,
                      r"if $v_\theta \approx x_0 - \epsilon$, then:",
                      fontsize=12, ha="center", color=MUTE)
            ax_r.text(0, 1.0,
                      r"$\hat{x}_0 = x_t + (1-t) \cdot v_\theta$",
                      fontsize=16, ha="center", color=CMAP_PRED,
                      fontweight="bold")
            ax_r.text(0, -0.8,
                      "predicting velocity = predicting\n"
                      "where the clean data lives",
                      fontsize=12, ha="center", color=MUTE)
            ax_r.text(0, -2.5,
                      "this IS denoising!",
                      fontsize=14, ha="center", color=CMAP_PRED,
                      fontweight="bold")
            ax_r.set_title("velocity prediction = denoising",
                           fontsize=11, color=MUTE, pad=6)
            sub.set_text("from (x_t, t) and the predicted velocity, "
                         "you can jump straight to x̂₀")
            cap.set_text("x̂₀ = x_t + (1−t)·v_θ  — one-step denoising "
                         "at any t")
            return

        if step == "denoise_show":
            # Show x_t on left, x̂_0 on right.
            t_demo = 0.3
            _show_cloud(t_demo)
            _show_velocity(t_demo, subsample=25)
            ax.set_title(f"x_t  (t = {t_demo})",
                         fontsize=11, color=MUTE, pad=6)
            # Predicted x0 = x_t + (1-t)*vel.
            xt = x_at_t(t_demo)
            x0_pred = xt + (1 - t_demo) * vel
            scat_r.set_offsets(x0_pred)
            scat_r.set_color(CMAP_PRED)
            scat_r.set_alpha(0.7)
            ax_r.set_title("x̂₀ = x_t + (1−t)·v_θ  (denoised!)",
                           fontsize=11, color=CMAP_PRED, pad=6)
            sub.set_text(f"at t = {t_demo}: apply velocity to jump "
                         "directly to clean data estimate")
            cap.set_text("the velocity field gives you a denoiser "
                         "for free — at every t")
            return

        # ===== Equivalence =====
        if step == "equiv":
            ax_r.text(0, 4.0, "three equivalent views:",
                      fontsize=13, ha="center", color=FG,
                      fontweight="bold")
            ax_r.text(0, 2.5,
                      "1. predict velocity  v = x₀ − ε",
                      fontsize=12, ha="center", color=CMAP_VEL)
            ax_r.text(0, 1.3,
                      "2. predict noise  ε = x_t − t·v",
                      fontsize=12, ha="center", color=CMAP_NOISY)
            ax_r.text(0, 0.1,
                      "3. predict clean data  x₀ = x_t + (1−t)·v",
                      fontsize=12, ha="center", color=CMAP_PRED)
            ax_r.text(0, -1.5,
                      "all linear reparameterizations\nof the same thing",
                      fontsize=11, ha="center", color=MUTE)
            ax_r.text(0, -3.2,
                      "flow matching = denoising\nwith straight paths",
                      fontsize=13, ha="center", color=FG,
                      fontweight="bold")
            ax_r.set_title("the three parameterizations",
                           fontsize=11, color=MUTE, pad=6)
            _show_cloud(0.5)
            ax.set_title("x_t at t = 0.5",
                         fontsize=11, color=MUTE, pad=6)
            sub.set_text("velocity prediction, noise prediction, "
                         "and clean-data prediction are all equivalent")
            cap.set_text("flow matching chooses velocity — "
                         "simplest to train, straight paths")
            return

        # ===== Inference summary =====
        if step == "inference":
            _show_cloud(1.0)
            _show_paths_up_to(1.0)
            ax.set_title("generated samples",
                         fontsize=11, color=MUTE, pad=6)
            ax_r.text(0, 3.5, "inference:", fontsize=14,
                      ha="center", color=FG, fontweight="bold")
            ax_r.text(0, 2.0,
                      "1. sample ε ~ N(0, I)",
                      fontsize=12, ha="center", color=CMAP_NOISY)
            ax_r.text(0, 1.0,
                      "2. solve ODE: dx/dt = v_θ(x, t)",
                      fontsize=12, ha="center", color=CMAP_VEL)
            ax_r.text(0, 0.0,
                      "   from t=0 to t=1",
                      fontsize=12, ha="center", color=MUTE)
            ax_r.text(0, -1.5,
                      "3. x₁ = generated sample ✓",
                      fontsize=12, ha="center", color=CMAP_CLEAN)
            ax_r.text(0, -3.5,
                      "fewer steps than DDPM\n"
                      "(straight paths → easier to integrate)",
                      fontsize=11, ha="center", color=MUTE)
            ax_r.set_title("generation",
                           fontsize=11, color=MUTE, pad=6)
            sub.set_text("at inference: solve an ODE from noise to data")
            cap.set_text("no noise schedule, no β parameters — "
                         "just integrate v_θ")
            return

        # ===== Summary =====
        _show_cloud(1.0)
        ax.set_title("flow matching",
                     fontsize=11, color=MUTE, pad=6)
        ax_r.text(0, 3.5, "flow matching", fontsize=20,
                  ha="center", color=FG, fontweight="bold")
        ax_r.text(0, 1.8,
                  "straight paths: x_t = (1−t)ε + t·x₀",
                  fontsize=11, ha="center", color=CMAP_VEL)
        ax_r.text(0, 0.8,
                  "learn velocity: v_θ(x_t, t) ≈ x₀ − ε",
                  fontsize=11, ha="center", color=CMAP_VEL)
        ax_r.text(0, -0.2,
                  "= denoising: x̂₀ = x_t + (1−t)·v_θ",
                  fontsize=11, ha="center", color=CMAP_PRED)
        ax_r.text(0, -1.2,
                  "self-supervised: no labels needed",
                  fontsize=11, ha="center", color=MUTE)
        ax_r.text(0, -3.0,
                  "SD3, Flux, Sora, ...",
                  fontsize=11, ha="center", color=MUTE)
        sub.set_text("learn to flow from noise to data along "
                     "straight paths — that's flow matching")
        cap.set_text("simpler than DDPM · faster inference · "
                     "same denoising principle")

    anim = animation.FuncAnimation(fig, update, frames=TOTAL,
                                   interval=1000/fps, blit=False)
    save_animation(anim, out_path, fps=fps)
    plt.close(fig)
    print(f"wrote {out_path}  ({TOTAL} frames, {fps} fps)")
