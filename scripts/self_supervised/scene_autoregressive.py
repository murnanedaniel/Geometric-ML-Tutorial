"""04a — Autoregressive (GPT-style) self-supervised training.

Shows a sentence being predicted one token at a time, with the
causal attention mask revealed step by step.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, FancyArrowPatch
from .helpers import (save_animation, DEFAULT_FPS, FG, MUTE, BG,
                      TOKEN_COLORS, DEFAULT_TOKEN_COLOR, MASK_COLOR,
                      PREDICT_COLOR, CAUSAL_COLOR)


SENTENCE = ["the", "cat", "sat", "on", "a", "mat", "and", "purred"]
N = len(SENTENCE)


def _tok_color(tok):
    return TOKEN_COLORS.get(tok, DEFAULT_TOKEN_COLOR)


def render(out_path: Path, fps: int = DEFAULT_FPS):
    fig = plt.figure(figsize=(13.5, 7.8), dpi=100)

    # Two rows: top = token sequence, bottom = attention mask.
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.2],
                          hspace=0.15,
                          left=0.05, right=0.95, top=0.82, bottom=0.08)
    ax_seq = fig.add_subplot(gs[0])
    ax_mask = fig.add_subplot(gs[1])

    for ax in (ax_seq, ax_mask):
        ax.set_xticks([]); ax.set_yticks([])
        for s in ("left", "right", "top", "bottom"):
            ax.spines[s].set_visible(False)

    ax_seq.set_xlim(-0.5, N + 1.5); ax_seq.set_ylim(-1, 3)
    ax_mask.set_xlim(-0.5, N - 0.5); ax_mask.set_ylim(-0.5, N - 0.5)
    ax_mask.set_aspect("equal")

    # Title + captions.
    fig.text(0.5, 0.96, "autoregressive training (GPT-style)",
             ha="center", va="top", fontsize=15, fontweight="bold")
    sub = fig.text(0.5, 0.92, "", ha="center", va="top",
                   fontsize=11, color=MUTE)
    cap = fig.text(0.5, 0.025, "", ha="center", va="bottom",
                   fontsize=11, color=FG)

    ax_seq.set_title("", fontsize=11, color=MUTE, pad=6)
    ax_mask.set_title("", fontsize=11, color=MUTE, pad=6)

    # ---- Token boxes (top panel) ----
    box_w, box_h = 0.85, 0.9
    tok_rects = []
    tok_texts = []
    for i in range(N):
        rect = Rectangle((i - box_w/2, 0.5), box_w, box_h,
                          facecolor=_tok_color(SENTENCE[i]), alpha=0.0,
                          edgecolor="white", lw=2, zorder=2)
        ax_seq.add_patch(rect)
        tok_rects.append(rect)
        txt = ax_seq.text(i, 0.95, "", ha="center", va="center",
                          fontsize=12, fontweight="bold", color="white",
                          alpha=0.0, zorder=3)
        tok_texts.append(txt)

    # Prediction arrow + predicted token (appears one ahead).
    pred_rect = Rectangle((0, 0.5), box_w, box_h, facecolor="none",
                           edgecolor=PREDICT_COLOR, lw=3, ls="--",
                           alpha=0.0, zorder=4)
    ax_seq.add_patch(pred_rect)
    pred_text = ax_seq.text(0, 0.95, "", ha="center", va="center",
                            fontsize=12, fontweight="bold",
                            color=PREDICT_COLOR, alpha=0.0, zorder=5)
    pred_arrow = ax_seq.annotate("", xy=(0, 0), xytext=(0, 0),
                                 arrowprops=dict(arrowstyle="->",
                                 color=PREDICT_COLOR, lw=2),
                                 alpha=0.0, zorder=4)

    # Position labels below tokens.
    pos_texts = []
    for i in range(N):
        txt = ax_seq.text(i, 0.2, f"pos {i}", ha="center", va="top",
                          fontsize=8, color=MUTE, alpha=0.0)
        pos_texts.append(txt)

    # ---- Causal attention mask (bottom panel) ----
    # Static grid labels.
    mask_row_labels = []
    mask_col_labels = []
    for i in range(N):
        rl = ax_mask.text(-0.7, N - 1 - i, SENTENCE[i], fontsize=9,
                          fontweight="bold", color=_tok_color(SENTENCE[i]),
                          ha="right", va="center", alpha=0.0)
        mask_row_labels.append(rl)
        cl = ax_mask.text(i, N - 0.3, SENTENCE[i], fontsize=9,
                          fontweight="bold", color=_tok_color(SENTENCE[i]),
                          ha="center", va="bottom", alpha=0.0, rotation=45)
        mask_col_labels.append(cl)

    # Mask cells (drawn fresh each frame via _draw_mask helper).
    mask_cell_patches = []
    mask_cell_texts_list = []
    for i in range(N):
        for j in range(N):
            rect = Rectangle((j - 0.45, N - 1 - i - 0.45), 0.9, 0.9,
                              facecolor=BG, edgecolor="#E0E0E0", lw=1,
                              zorder=0, alpha=0.0)
            ax_mask.add_patch(rect)
            mask_cell_patches.append(rect)
            txt = ax_mask.text(j, N - 1 - i, "", ha="center", va="center",
                               fontsize=10, fontweight="bold", alpha=0.0,
                               zorder=1)
            mask_cell_texts_list.append(txt)

    def _get_mask_cell(i, j):
        """Get (rect, text) for row i, col j."""
        idx = i * N + j
        return mask_cell_patches[idx], mask_cell_texts_list[idx]

    def _show_mask(n_revealed, highlight_row=-1):
        """Show the causal mask for positions 0..n_revealed-1."""
        for i in range(N):
            for j in range(N):
                rect, txt = _get_mask_cell(i, j)
                if i >= n_revealed or j >= n_revealed:
                    rect.set_alpha(0.0)
                    txt.set_alpha(0.0)
                    continue
                rect.set_alpha(1.0)
                if j <= i:
                    rect.set_facecolor("#D6F5D6" if i == highlight_row
                                       else "#E8F5E9")
                    txt.set_text("1")
                    txt.set_color(PREDICT_COLOR)
                    txt.set_alpha(0.8)
                else:
                    rect.set_facecolor("#FFE0E0" if i == highlight_row
                                       else "#F5F5F5")
                    txt.set_text("0")
                    txt.set_color(CAUSAL_COLOR)
                    txt.set_alpha(0.5)
                rect.set_edgecolor("#C0C0C0")
        for i in range(N):
            mask_row_labels[i].set_alpha(1.0 if i < n_revealed else 0.0)
            mask_col_labels[i].set_alpha(1.0 if i < n_revealed else 0.0)

    # ---- Frame schedule ----
    HOLD = 70

    steps = []
    steps.append(("intro",       HOLD))
    steps.append(("show_full",   HOLD))
    steps.append(("explain_task", HOLD))
    for pos in range(N - 1):
        steps.append((f"predict_{pos}", HOLD))
    steps.append(("full_mask",   HOLD))
    steps.append(("summary",     60))

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

    def _reset():
        for r in tok_rects: r.set_alpha(0.0)
        for t in tok_texts: t.set_alpha(0.0); t.set_text("")
        for t in pos_texts: t.set_alpha(0.0)
        pred_rect.set_alpha(0.0)
        pred_text.set_alpha(0.0)
        for p in mask_cell_patches: p.set_alpha(0.0)
        for t in mask_cell_texts_list: t.set_alpha(0.0)
        for t in mask_row_labels: t.set_alpha(0.0)
        for t in mask_col_labels: t.set_alpha(0.0)

    def update(f):
        _reset()
        step = frame_to_step[min(f, TOTAL - 1)]
        t = _step_t(f, step)

        # ===== Intro: show the sentence =====
        if step == "intro":
            a = min(1.0, t * 2.5)
            for i in range(N):
                tok_rects[i].set_alpha(0.3 * a)
                tok_texts[i].set_text(SENTENCE[i])
                tok_texts[i].set_alpha(a)
                pos_texts[i].set_alpha(a * 0.6)
            ax_seq.set_title("a sentence (8 tokens)",
                             fontsize=11, color=MUTE, pad=6)
            sub.set_text("training data: sequences of tokens (words)")
            cap.set_text("the model will learn to predict each next "
                         "token from the previous ones")
            return

        # ===== Show full sentence =====
        if step == "show_full":
            for i in range(N):
                tok_rects[i].set_alpha(0.3)
                tok_texts[i].set_text(SENTENCE[i])
                tok_texts[i].set_alpha(1.0)
                pos_texts[i].set_alpha(0.6)
            ax_seq.set_title("the training objective",
                             fontsize=11, color=MUTE, pad=6)
            sub.set_text('given tokens 0..t, predict token t+1')
            cap.set_text("no labels needed — the next token IS the label "
                         "(self-supervised)")
            return

        # ===== Explain task =====
        if step == "explain_task":
            for i in range(N):
                tok_rects[i].set_alpha(0.3)
                tok_texts[i].set_text(SENTENCE[i])
                tok_texts[i].set_alpha(1.0)
                pos_texts[i].set_alpha(0.6)
            # Show first token visible, rest faded.
            for i in range(1, N):
                tok_rects[i].set_alpha(0.1)
                tok_texts[i].set_alpha(0.3)
            ax_seq.set_title("step by step: predict the next token",
                             fontsize=11, color=MUTE, pad=6)
            ax_mask.set_title("causal attention mask",
                              fontsize=11, color=MUTE, pad=6)
            _show_mask(1, highlight_row=0)
            sub.set_text('start: given "the" → predict next word')
            cap.set_text("the causal mask prevents looking at future tokens")
            return

        # ===== Predict each position =====
        for pos in range(N - 1):
            if step == f"predict_{pos}":
                # Show tokens 0..pos (context).
                for i in range(N):
                    if i <= pos:
                        tok_rects[i].set_alpha(0.3)
                        tok_texts[i].set_text(SENTENCE[i])
                        tok_texts[i].set_alpha(1.0)
                    else:
                        tok_rects[i].set_alpha(0.05)
                        tok_texts[i].set_text("?")
                        tok_texts[i].set_alpha(0.2)
                    pos_texts[i].set_alpha(0.6 if i <= pos else 0.2)

                # Show prediction for pos+1.
                pred_x = pos + 1
                pred_rect.set_xy((pred_x - box_w/2, 0.5))
                pred_rect.set_alpha(0.9)
                pred_text.set_position((pred_x, 0.95))
                pred_text.set_text(SENTENCE[pos + 1])
                pred_text.set_alpha(0.9 if t > 0.4 else 0.0)

                # Show causal mask up to this position.
                _show_mask(pos + 2, highlight_row=pos + 1)

                ctx = " ".join(SENTENCE[:pos+1])
                target = SENTENCE[pos + 1]
                ax_seq.set_title(
                    f'context: "{ctx}" → predict: "{target}"',
                    fontsize=11, color=MUTE, pad=6)
                ax_mask.set_title(
                    f"causal mask (position {pos+1} can see 0..{pos})",
                    fontsize=11, color=MUTE, pad=6)
                sub.set_text(
                    f"position {pos+1}: attend to previous {pos+1} "
                    f"tokens only (row {pos+1} of the mask)")
                cap.set_text(
                    f"loss = −log P(\"{target}\" | "
                    f"\"{ctx}\")")
                return

        # ===== Full mask revealed =====
        if step == "full_mask":
            for i in range(N):
                tok_rects[i].set_alpha(0.3)
                tok_texts[i].set_text(SENTENCE[i])
                tok_texts[i].set_alpha(1.0)
                pos_texts[i].set_alpha(0.6)
            _show_mask(N)
            ax_seq.set_title("all positions predicted in parallel",
                             fontsize=11, color=MUTE, pad=6)
            ax_mask.set_title("full causal attention mask (lower-triangular)",
                              fontsize=11, color=MUTE, pad=6)
            sub.set_text("training: all N−1 predictions computed in one "
                         "forward pass (teacher forcing)")
            cap.set_text("total loss = sum of per-position cross-entropy "
                         "losses")
            return

        # ===== Summary =====
        for i in range(N):
            tok_rects[i].set_alpha(0.3)
            tok_texts[i].set_text(SENTENCE[i])
            tok_texts[i].set_alpha(1.0)
        _show_mask(N)
        ax_seq.set_title("autoregressive self-supervised training",
                         fontsize=11, color=MUTE, pad=6)
        ax_mask.set_title("causal (lower-triangular) attention mask",
                          fontsize=11, color=MUTE, pad=6)
        sub.set_text("predict next token from previous context — "
                     "no labels needed")
        cap.set_text("GPT, GPT-2, GPT-3, GPT-4, LLaMA, ... — "
                     "foundation of modern LLMs")

    anim = animation.FuncAnimation(fig, update, frames=TOTAL,
                                   interval=1000/fps, blit=False)
    save_animation(anim, out_path, fps=fps)
    plt.close(fig)
    print(f"wrote {out_path}  ({TOTAL} frames, {fps} fps)")
