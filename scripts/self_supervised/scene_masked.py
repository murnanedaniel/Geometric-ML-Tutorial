"""04b — Masked modelling (BERT / MAE style) self-supervised training.

Shows tokens being masked at random and the model predicting the
masked positions, with the bidirectional attention mask.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from .helpers import (save_animation, DEFAULT_FPS, FG, MUTE, BG,
                      TOKEN_COLORS, DEFAULT_TOKEN_COLOR, MASK_COLOR,
                      PREDICT_COLOR, CORRECT_COLOR)


SENTENCE = ["the", "cat", "sat", "on", "a", "mat", "and", "purred"]
N = len(SENTENCE)
MASK_IDX = [1, 4, 6]  # "cat", "a", "and" — masked positions (15-40%)


def _tok_color(tok):
    return TOKEN_COLORS.get(tok, DEFAULT_TOKEN_COLOR)


def render(out_path: Path, fps: int = DEFAULT_FPS):
    fig = plt.figure(figsize=(13.5, 7.8), dpi=100)

    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.2],
                          hspace=0.15,
                          left=0.05, right=0.95, top=0.82, bottom=0.08)
    ax_seq = fig.add_subplot(gs[0])
    ax_mask = fig.add_subplot(gs[1])

    for ax in (ax_seq, ax_mask):
        ax.set_xticks([]); ax.set_yticks([])
        for s in ("left", "right", "top", "bottom"):
            ax.spines[s].set_visible(False)

    ax_seq.set_xlim(-0.5, N + 0.5); ax_seq.set_ylim(-1, 3)
    ax_mask.set_xlim(-0.5, N - 0.5); ax_mask.set_ylim(-0.5, N - 0.5)
    ax_mask.set_aspect("equal")

    fig.text(0.5, 0.96, "masked modelling (BERT / MAE style)",
             ha="center", va="top", fontsize=15, fontweight="bold")
    sub = fig.text(0.5, 0.92, "", ha="center", va="top",
                   fontsize=11, color=MUTE)
    cap = fig.text(0.5, 0.025, "", ha="center", va="bottom",
                   fontsize=11, color=FG)
    ax_seq.set_title("", fontsize=11, color=MUTE, pad=6)
    ax_mask.set_title("", fontsize=11, color=MUTE, pad=6)

    # ---- Token boxes ----
    box_w, box_h = 0.85, 0.9
    tok_rects = []
    tok_texts = []
    pos_texts = []
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
        ptxt = ax_seq.text(i, 0.2, f"pos {i}", ha="center", va="top",
                           fontsize=8, color=MUTE, alpha=0.0)
        pos_texts.append(ptxt)

    # Prediction labels (below masked tokens).
    pred_texts = []
    for i in range(N):
        txt = ax_seq.text(i, 1.8, "", ha="center", va="bottom",
                          fontsize=11, fontweight="bold",
                          color=CORRECT_COLOR, alpha=0.0, zorder=5)
        pred_texts.append(txt)

    # ---- Attention mask grid ----
    mask_cells = []
    mask_txts = []
    for i in range(N):
        for j in range(N):
            rect = Rectangle((j - 0.45, N - 1 - i - 0.45), 0.9, 0.9,
                              facecolor=BG, edgecolor="#E0E0E0", lw=1,
                              alpha=0.0, zorder=0)
            ax_mask.add_patch(rect)
            mask_cells.append(rect)
            txt = ax_mask.text(j, N - 1 - i, "", ha="center", va="center",
                               fontsize=10, fontweight="bold", alpha=0.0,
                               zorder=1)
            mask_txts.append(txt)

    # Row/col labels.
    row_labels = []
    col_labels = []
    for i in range(N):
        rl = ax_mask.text(-0.7, N - 1 - i, SENTENCE[i], fontsize=9,
                          fontweight="bold", color=_tok_color(SENTENCE[i]),
                          ha="right", va="center", alpha=0.0)
        row_labels.append(rl)
        cl = ax_mask.text(i, N - 0.3, SENTENCE[i], fontsize=9,
                          fontweight="bold", color=_tok_color(SENTENCE[i]),
                          ha="center", va="bottom", alpha=0.0, rotation=45)
        col_labels.append(cl)

    def _show_bidir_mask(show_all=True, highlight_rows=None):
        """Bidirectional: every token attends to every other."""
        if highlight_rows is None:
            highlight_rows = []
        for i in range(N):
            row_labels[i].set_alpha(1.0)
            col_labels[i].set_alpha(1.0)
            if i in MASK_IDX:
                row_labels[i].set_text("[MASK]")
                row_labels[i].set_color(MASK_COLOR)
                col_labels[i].set_text("[MASK]")
                col_labels[i].set_color(MASK_COLOR)
            else:
                row_labels[i].set_text(SENTENCE[i])
                row_labels[i].set_color(_tok_color(SENTENCE[i]))
                col_labels[i].set_text(SENTENCE[i])
                col_labels[i].set_color(_tok_color(SENTENCE[i]))

            for j in range(N):
                idx = i * N + j
                mask_cells[idx].set_alpha(1.0)
                if i in highlight_rows:
                    mask_cells[idx].set_facecolor("#D6F5D6")
                else:
                    mask_cells[idx].set_facecolor("#E8F5E9")
                mask_txts[idx].set_text("1")
                mask_txts[idx].set_color(PREDICT_COLOR)
                mask_txts[idx].set_alpha(0.6)

    # ---- Frame schedule ----
    HOLD = 70

    steps = []
    steps.append(("intro",          HOLD))
    steps.append(("show_sentence",  HOLD))
    steps.append(("mask_tokens",    HOLD))
    steps.append(("explain_mask",   HOLD))
    steps.append(("bidir_mask",     HOLD))
    for mi, midx in enumerate(MASK_IDX):
        steps.append((f"predict_{mi}", HOLD))
    steps.append(("all_predicted",  HOLD))
    steps.append(("compare_gpt",   HOLD))
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

    def _reset():
        for r in tok_rects: r.set_alpha(0.0)
        for t in tok_texts: t.set_alpha(0.0); t.set_text("")
        for t in pos_texts: t.set_alpha(0.0)
        for t in pred_texts: t.set_alpha(0.0); t.set_text("")
        for p in mask_cells: p.set_alpha(0.0)
        for t in mask_txts: t.set_alpha(0.0)
        for t in row_labels: t.set_alpha(0.0)
        for t in col_labels: t.set_alpha(0.0)

    def _show_tokens(masked=False, fade_masked=False):
        for i in range(N):
            is_masked = i in MASK_IDX
            if masked and is_masked:
                tok_rects[i].set_facecolor(MASK_COLOR)
                tok_rects[i].set_alpha(0.3)
                tok_texts[i].set_text("[MASK]")
                tok_texts[i].set_alpha(0.9)
            elif fade_masked and is_masked:
                tok_rects[i].set_facecolor(MASK_COLOR)
                tok_rects[i].set_alpha(0.15)
                tok_texts[i].set_text("[MASK]")
                tok_texts[i].set_alpha(0.4)
            else:
                tok_rects[i].set_facecolor(_tok_color(SENTENCE[i]))
                tok_rects[i].set_alpha(0.3)
                tok_texts[i].set_text(SENTENCE[i])
                tok_texts[i].set_alpha(1.0)
            pos_texts[i].set_alpha(0.6)

    def update(f):
        _reset()
        step = frame_to_step[min(f, TOTAL - 1)]
        t = _step_t(f, step)

        if step == "intro":
            a = min(1.0, t * 2.5)
            for i in range(N):
                tok_rects[i].set_facecolor(_tok_color(SENTENCE[i]))
                tok_rects[i].set_alpha(0.3 * a)
                tok_texts[i].set_text(SENTENCE[i])
                tok_texts[i].set_alpha(a)
                pos_texts[i].set_alpha(a * 0.6)
            ax_seq.set_title("a sentence (8 tokens)",
                             fontsize=11, color=MUTE, pad=6)
            sub.set_text("same sentence as before — "
                         "but a very different training strategy")
            cap.set_text("")
            return

        if step == "show_sentence":
            _show_tokens()
            ax_seq.set_title("original sentence",
                             fontsize=11, color=MUTE, pad=6)
            sub.set_text("step 1: randomly select ~15-40% of tokens to mask")
            cap.set_text(f"we'll mask positions "
                         f"{MASK_IDX} "
                         f"({', '.join(SENTENCE[i] for i in MASK_IDX)})")
            return

        if step == "mask_tokens":
            _show_tokens(masked=True)
            ax_seq.set_title("masked input",
                             fontsize=11, color=MUTE, pad=6)
            sub.set_text(f"{len(MASK_IDX)} tokens replaced with [MASK] — "
                         "the model must reconstruct them")
            cap.set_text("the labels are free: they're the original tokens "
                         "we just hid")
            return

        if step == "explain_mask":
            _show_tokens(masked=True)
            ax_seq.set_title("the key difference from GPT",
                             fontsize=11, color=MUTE, pad=6)
            sub.set_text("the model can see tokens on BOTH sides "
                         "of each [MASK] — bidirectional context")
            cap.set_text('"sat" and "on" are visible when predicting '
                         '[MASK] at position 1 ("cat")')
            return

        if step == "bidir_mask":
            _show_tokens(masked=True)
            _show_bidir_mask()
            ax_seq.set_title("masked input",
                             fontsize=11, color=MUTE, pad=6)
            ax_mask.set_title("bidirectional attention mask (all 1s)",
                              fontsize=11, color=MUTE, pad=6)
            sub.set_text("every token attends to every other — "
                         "no causal constraint")
            cap.set_text("compare with GPT's lower-triangular mask — "
                         "here the full matrix is green")
            return

        # ===== Predict each masked position =====
        for mi, midx in enumerate(MASK_IDX):
            if step == f"predict_{mi}":
                _show_tokens(masked=True)
                _show_bidir_mask(highlight_rows=[midx])
                # Show prediction.
                pred_texts[midx].set_text(f'→ "{SENTENCE[midx]}"')
                pred_texts[midx].set_alpha(0.9 if t > 0.3 else 0.0)
                # Also show previous predictions.
                for prev_mi in range(mi):
                    prev_idx = MASK_IDX[prev_mi]
                    pred_texts[prev_idx].set_text(
                        f'→ "{SENTENCE[prev_idx]}" ✓')
                    pred_texts[prev_idx].set_alpha(0.7)
                ax_seq.set_title(
                    f'predict position {midx}: '
                    f'[MASK] → "{SENTENCE[midx]}"',
                    fontsize=11, color=MUTE, pad=6)
                ax_mask.set_title(
                    f"row {midx}: attends to ALL positions "
                    f"(bidirectional)",
                    fontsize=11, color=MUTE, pad=6)
                visible = [SENTENCE[j] if j not in MASK_IDX
                           else "[MASK]" for j in range(N)]
                ctx = " ".join(visible)
                sub.set_text(f'input: "{ctx}"')
                cap.set_text(
                    f"loss at position {midx} = "
                    f'−log P("{SENTENCE[midx]}" | all visible tokens)')
                return

        if step == "all_predicted":
            _show_tokens(masked=True)
            _show_bidir_mask()
            for mi, midx in enumerate(MASK_IDX):
                pred_texts[midx].set_text(f'→ "{SENTENCE[midx]}" ✓')
                pred_texts[midx].set_alpha(0.9)
            ax_seq.set_title("all masked positions predicted",
                             fontsize=11, color=MUTE, pad=6)
            ax_mask.set_title("bidirectional attention (full matrix)",
                              fontsize=11, color=MUTE, pad=6)
            sub.set_text(f"loss = sum over {len(MASK_IDX)} masked positions only "
                         "(unmasked positions are ignored)")
            cap.set_text("all predictions computed in one forward pass "
                         "(parallel, like GPT)")
            return

        if step == "compare_gpt":
            _show_tokens(masked=True)
            _show_bidir_mask()
            for midx in MASK_IDX:
                pred_texts[midx].set_text(f'"{SENTENCE[midx]}" ✓')
                pred_texts[midx].set_alpha(0.7)
            ax_seq.set_title("masked modelling vs autoregressive",
                             fontsize=11, color=MUTE, pad=6)
            ax_mask.set_title("",
                              fontsize=11, color=MUTE, pad=6)
            sub.set_text("GPT: left-to-right, causal mask, predict ALL "
                         "next tokens\n"
                         "BERT: bidirectional, full mask, predict ONLY "
                         "masked tokens")
            cap.set_text("bidirectional context → better representations   "
                         "·   but can't generate text left-to-right")
            return

        # ===== Summary =====
        _show_tokens(masked=True)
        _show_bidir_mask()
        for midx in MASK_IDX:
            pred_texts[midx].set_text(f'"{SENTENCE[midx]}" ✓')
            pred_texts[midx].set_alpha(0.7)
        ax_seq.set_title("masked self-supervised training",
                         fontsize=11, color=MUTE, pad=6)
        ax_mask.set_title("bidirectional attention",
                          fontsize=11, color=MUTE, pad=6)
        sub.set_text("mask random tokens, predict them using "
                     "full bidirectional context")
        cap.set_text("BERT, RoBERTa, MAE, BEiT, ... — "
                     "foundation of encoder models")

    anim = animation.FuncAnimation(fig, update, frames=TOTAL,
                                   interval=1000/fps, blit=False)
    save_animation(anim, out_path, fps=fps)
    plt.close(fig)
    print(f"wrote {out_path}  ({TOTAL} frames, {fps} fps)")
