"""GLOW architecture walkthrough — built incrementally, phase by phase."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle, Arc
from .helpers import (save_animation, DEFAULT_FPS, FG, MUTE, GRID, BG,
                      TRACK_COLOR, TOPO_COLOR, QUERY_COLOR,
                      ENCODE_COLOR, DECODE_COLOR, OK_COLOR)

# ---------------------------------------------------------------------------
# Toy CLIC event data
# ---------------------------------------------------------------------------
# 4 tracks (charged particles curving through the tracker).
# Each track: (phi_start, curvature, pT, true_particle_id)
TRACKS = [
    {"phi": 0.3,  "curv": 0.15, "pT": 4.2, "eta": 0.1, "pid": 0},
    {"phi": 1.8,  "curv":-0.10, "pT": 6.1, "eta":-0.3, "pid": 1},
    {"phi": 3.5,  "curv": 0.08, "pT": 2.8, "eta": 0.5, "pid": 2},
    {"phi": 5.0,  "curv":-0.20, "pT": 8.3, "eta":-0.1, "pid": 3},
]

# 5 topoclusters (energy deposits in calorimeter).
# Each: (phi_center, E, layer, true_particle_ids, fractions)
TOPOS = [
    {"phi": 0.4,  "E": 3.8, "layer": 0, "pids": [0],    "fracs": [1.0]},
    {"phi": 1.7,  "E": 7.2, "layer": 0, "pids": [1],    "fracs": [1.0]},
    {"phi": 2.5,  "E": 5.0, "layer": 1, "pids": [1, 4], "fracs": [0.4, 0.6]},
    {"phi": 3.6,  "E": 2.5, "layer": 0, "pids": [2],    "fracs": [1.0]},
    {"phi": 5.1,  "E": 9.1, "layer": 1, "pids": [3],    "fracs": [1.0]},
]

N_TRACKS = len(TRACKS)
N_TOPOS  = len(TOPOS)
N_INPUTS = N_TRACKS + N_TOPOS   # 9 detector objects
N_QUERIES = 6                    # learned particle queries
N_PARTICLES = 5                  # true particles (one neutral: id=4)

# Detector geometry (radii for 2D cross-section).
R_BEAM   = 0.3
R_TRACK1 = 1.2
R_TRACK2 = 2.0
R_ECAL   = 3.0
R_HCAL   = 4.0
R_OUTER  = 4.5

# Ground-truth incidence matrix (N_INPUTS × N_PARTICLES).
# Rows sum to 1: each detector object's energy is fully accounted for.
GT_INCIDENCE = np.zeros((N_INPUTS, N_PARTICLES))
# Tracks → their particle (1-to-1, full energy).
for i, trk in enumerate(TRACKS):
    GT_INCIDENCE[i, trk["pid"]] = 1.0
# Topoclusters → possibly shared.
for i, topo in enumerate(TOPOS):
    for pid, frac in zip(topo["pids"], topo["fracs"]):
        GT_INCIDENCE[N_TRACKS + i, pid] = frac


def _track_xy(phi, curv, r):
    """Position of a track at radius r (simplified helix in 2D)."""
    angle = phi + curv * r
    return r * np.cos(angle), r * np.sin(angle)


def _draw_detector(ax):
    """Draw the simplified 2D CLIC cross-section."""
    for r, label, col in [
        (R_TRACK1, "", "#D6D2C4"),
        (R_TRACK2, "", "#D6D2C4"),
        (R_ECAL,   "ECAL", "#E8E4D8"),
        (R_HCAL,   "HCAL", "#DDD8CA"),
    ]:
        circ = Circle((0, 0), r, fill=False, edgecolor=col,
                       lw=1.2, ls="--", zorder=0)
        ax.add_patch(circ)
        if label:
            ax.text(r * 0.71 + 0.15, r * 0.71 + 0.15, label,
                    fontsize=7, color=MUTE, ha="left", zorder=1)
    # Beam spot.
    ax.plot(0, 0, "x", color=MUTE, ms=6, mew=1.5, zorder=2)


def _draw_tracks(ax, alpha=1.0):
    """Draw track curves from origin outward. Returns list of Line2D."""
    lines = []
    for trk in TRACKS:
        rs = np.linspace(0, R_ECAL * 0.95, 80)
        xs = [_track_xy(trk["phi"], trk["curv"], r)[0] for r in rs]
        ys = [_track_xy(trk["phi"], trk["curv"], r)[1] for r in rs]
        ln, = ax.plot(xs, ys, color=TRACK_COLOR, lw=2.0,
                      alpha=alpha, zorder=3)
        lines.append(ln)
    return lines


def _draw_topos(ax, alpha=1.0):
    """Draw topocluster blobs on the calorimeter. Returns scatter."""
    xs, ys, sizes = [], [], []
    for topo in TOPOS:
        r = R_ECAL + 0.3 if topo["layer"] == 0 else R_HCAL - 0.3
        x = r * np.cos(topo["phi"])
        y = r * np.sin(topo["phi"])
        xs.append(x); ys.append(y)
        sizes.append(topo["E"] * 40)
    sc = ax.scatter(xs, ys, s=sizes, c=TOPO_COLOR, alpha=alpha,
                    edgecolors="white", linewidths=1.5, zorder=4,
                    marker="s")
    return sc


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------
def render(out_path: Path, fps: int = DEFAULT_FPS):

    fig = plt.figure(figsize=(14.0, 7.8), dpi=100)
    # Two panels: detector left, architecture/matrix right.
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.3], wspace=0.08,
                          left=0.03, right=0.97, top=0.84, bottom=0.08)
    ax_det = fig.add_subplot(gs[0])
    ax_r   = fig.add_subplot(gs[1])

    # Detector panel.
    span = R_OUTER + 0.8
    ax_det.set_xlim(-span, span); ax_det.set_ylim(-span, span)
    ax_det.set_aspect("equal")
    ax_det.set_xticks([]); ax_det.set_yticks([])
    for s in ("left", "right", "top", "bottom"):
        ax_det.spines[s].set_visible(False)

    # Right panel — starts blank, used for tokens/matrix later.
    ax_r.set_xlim(0, 10); ax_r.set_ylim(-6, 10)
    ax_r.set_xticks([]); ax_r.set_yticks([])
    for s in ("left", "right", "top", "bottom"):
        ax_r.spines[s].set_visible(False)

    # Header texts.
    title = fig.text(0.5, 0.96, "GLOW: a unified particle flow transformer",
                     ha="center", va="top", fontsize=15, fontweight="bold")
    sub = fig.text(0.5, 0.92, "", ha="center", va="top",
                   fontsize=11, color=MUTE)
    cap = fig.text(0.5, 0.025, "", ha="center", va="bottom",
                   fontsize=11, color=FG)
    det_title = ax_det.set_title("", fontsize=11, color=MUTE, pad=6)
    r_title = ax_r.set_title("", fontsize=11, color=MUTE, pad=6)

    # Pre-draw detector geometry (always visible).
    _draw_detector(ax_det)

    # Track lines (initially invisible).
    track_lines = _draw_tracks(ax_det, alpha=0.0)

    # Topocluster scatter (initially invisible).
    topo_sc = _draw_topos(ax_det, alpha=0.0)

    # Track labels.
    track_labels = []
    for i, trk in enumerate(TRACKS):
        x, y = _track_xy(trk["phi"], trk["curv"], R_TRACK2 + 0.3)
        txt = ax_det.text(x, y, f"t{i}", fontsize=9, color=TRACK_COLOR,
                          fontweight="bold", ha="center", va="center",
                          alpha=0.0, zorder=5)
        track_labels.append(txt)

    # Topo labels.
    topo_labels = []
    for i, topo in enumerate(TOPOS):
        r = R_ECAL + 0.3 if topo["layer"] == 0 else R_HCAL - 0.3
        x = (r + 0.4) * np.cos(topo["phi"])
        y = (r + 0.4) * np.sin(topo["phi"])
        txt = ax_det.text(x, y, f"c{i}", fontsize=9, color=TOPO_COLOR,
                          fontweight="bold", ha="center", va="center",
                          alpha=0.0, zorder=5)
        topo_labels.append(txt)

    # ---- Right-panel objects (tokens) ----
    # Token boxes: N_INPUTS rows, each a small rectangle with label + features.
    token_rects = []
    token_texts = []
    token_feat_texts = []
    box_h = 0.55
    box_w = 7.0
    x0_tok = 1.5
    y0_tok = 9.0
    row_sp = box_h + 0.10
    for idx in range(N_INPUTS):
        y = y0_tok - idx * row_sp
        is_track = idx < N_TRACKS
        col = TRACK_COLOR if is_track else TOPO_COLOR
        rect = Rectangle((x0_tok, y), box_w, box_h,
                          facecolor=col, alpha=0.0,
                          edgecolor="white", lw=1.5, zorder=2)
        ax_r.add_patch(rect)
        token_rects.append(rect)
        if is_track:
            lbl = f"t{idx}"
            trk = TRACKS[idx]
            feat = f"pT={trk['pT']:.1f}  η={trk['eta']:.1f}  φ={trk['phi']:.1f}"
        else:
            lbl = f"c{idx - N_TRACKS}"
            topo = TOPOS[idx - N_TRACKS]
            feat = f"E={topo['E']:.1f}  φ={topo['phi']:.1f}  layer={topo['layer']}"
        txt = ax_r.text(x0_tok + 0.3, y + box_h / 2, lbl,
                        fontsize=9, fontweight="bold", color="white",
                        va="center", alpha=0.0, zorder=3)
        token_texts.append(txt)
        ftxt = ax_r.text(x0_tok + 1.4, y + box_h / 2, feat,
                         fontsize=8, color="white", va="center",
                         alpha=0.0, zorder=3)
        token_feat_texts.append(ftxt)

    # ---- Encoder visual: "self-attention" bracket around tokens ----
    enc_bracket = Rectangle((x0_tok - 0.3, y0_tok - (N_INPUTS - 1) * row_sp - 0.3),
                            box_w + 0.6,
                            N_INPUTS * row_sp + 0.15,
                            facecolor="none", edgecolor=ENCODE_COLOR,
                            lw=2.5, ls="-", zorder=1, alpha=0.0)
    ax_r.add_patch(enc_bracket)
    enc_label = ax_r.text(x0_tok + box_w / 2, y0_tok + 0.5, "",
                          ha="center", va="bottom", fontsize=10,
                          fontweight="bold", color=ENCODE_COLOR,
                          alpha=0.0, zorder=3)

    # Self-attention lines between token pairs (drawn in right panel).
    attn_lines = []
    for _ in range(20):
        ln, = ax_r.plot([], [], color=ENCODE_COLOR, lw=1.0,
                        alpha=0.0, zorder=1)
        attn_lines.append(ln)

    # ---- Query tokens (right panel, below encoder tokens) ----
    q_y0 = y0_tok - N_INPUTS * row_sp - 1.2
    query_rects = []
    query_texts = []
    for qi in range(N_QUERIES):
        y = q_y0 - qi * row_sp
        rect = Rectangle((x0_tok, y), box_w, box_h,
                          facecolor=QUERY_COLOR, alpha=0.0,
                          edgecolor="white", lw=1.5, zorder=2)
        ax_r.add_patch(rect)
        query_rects.append(rect)
        txt = ax_r.text(x0_tok + box_w / 2, y + box_h / 2,
                        f"query q{qi}  (learned)", ha="center", va="center",
                        fontsize=9, fontweight="bold", color="white",
                        alpha=0.0, zorder=3)
        query_texts.append(txt)

    # Cross-attention arrows from queries to tokens.
    xattn_lines = []
    for _ in range(N_QUERIES * N_INPUTS):
        ln, = ax_r.plot([], [], color=DECODE_COLOR, lw=1.2,
                        alpha=0.0, zorder=1)
        xattn_lines.append(ln)

    # Decoder bracket.
    dec_bracket = Rectangle(
        (x0_tok - 0.3, q_y0 - (N_QUERIES - 1) * row_sp - 0.3),
        box_w + 0.6,
        N_QUERIES * row_sp + 0.15,
        facecolor="none", edgecolor=DECODE_COLOR,
        lw=2.5, ls="-", zorder=1, alpha=0.0)
    ax_r.add_patch(dec_bracket)
    dec_label = ax_r.text(x0_tok + box_w / 2,
                          q_y0 - (N_QUERIES - 1) * row_sp - 0.6,
                          "", ha="center", va="top", fontsize=10,
                          fontweight="bold", color=DECODE_COLOR,
                          alpha=0.0, zorder=3)

    # ---- Incidence matrix panel (reuses ax_r, drawn over tokens/queries) ----
    # We'll create a second axes overlaid on ax_r for the matrix view.
    ax_mat = fig.add_axes([0.52, 0.12, 0.44, 0.65])
    ax_mat.set_visible(False)

    # Predicted incidence matrix (plausible output, not GT).
    PRED_INCIDENCE = np.array([
        # p0    p1    p2    p3    p4    (no-obj)
        [0.95, 0.03, 0.01, 0.01, 0.00, 0.00],  # t0
        [0.02, 0.94, 0.02, 0.01, 0.01, 0.00],  # t1
        [0.01, 0.01, 0.96, 0.01, 0.01, 0.00],  # t2
        [0.01, 0.01, 0.01, 0.96, 0.01, 0.00],  # t3
        [0.97, 0.01, 0.01, 0.00, 0.01, 0.00],  # c0
        [0.02, 0.95, 0.01, 0.01, 0.01, 0.00],  # c1
        [0.01, 0.38, 0.01, 0.01, 0.59, 0.00],  # c2 — shared!
        [0.01, 0.01, 0.96, 0.01, 0.01, 0.00],  # c3
        [0.01, 0.01, 0.01, 0.96, 0.01, 0.00],  # c4
    ])

    obj_labels = [f"t{i}" for i in range(N_TRACKS)] + \
                 [f"c{i}" for i in range(N_TOPOS)]
    query_labels = [f"q{i}" for i in range(N_QUERIES)]

    # Pre-build incidence cell text objects.
    inc_cell_texts = []
    inc_cell_bgs = []
    inc_row_sum_texts = []

    # Particle energies from incidence-weighted sum.
    obj_energies = np.array([t["pT"] for t in TRACKS] +
                            [t["E"] for t in TOPOS])
    pred_particle_E = PRED_INCIDENCE.T @ obj_energies

    # ---- Frame schedule ----
    HOLD = 70  # ~5s at 14fps

    steps = []
    steps.append(("detector_empty",   HOLD))
    steps.append(("tracks_appear",    HOLD))
    steps.append(("topos_appear",     HOLD))
    steps.append(("event_hold",       HOLD))
    steps.append(("tokenize",         int(HOLD * 1.2)))
    steps.append(("encoder_intro",    HOLD))       # explain encoder
    for layer in range(6):
        steps.append((f"encoder_L{layer}", 30))    # pulse each layer
    steps.append(("encoder_done",     HOLD))
    steps.append(("queries_appear",   HOLD))       # learned queries
    steps.append(("decoder_intro",    HOLD))       # explain cross-attn
    steps.append(("decoder_pulse",    int(HOLD * 1.0)))  # 4-layer decoder
    steps.append(("decoder_done",     HOLD))
    steps.append(("incidence_intro",  HOLD))       # introduce incidence matrix
    steps.append(("incidence_show",   int(HOLD * 1.5)))  # show numbers + row sums
    steps.append(("incidence_share",  HOLD))       # c2 shared energy
    steps.append(("weighted_sum",     int(HOLD * 1.2)))  # E_a = sum I*E
    steps.append(("three_heads",      HOLD))       # class + kinematics
    steps.append(("hungarian_loss",   int(HOLD * 1.2)))  # matching in training
    steps.append(("reconstruct",      HOLD))       # final overlay
    steps.append(("final_hold",       60))

    frame_to_step = []
    step_offsets = {}
    for sname, dur in steps:
        step_offsets[sname] = len(frame_to_step)
        for _ in range(dur):
            frame_to_step.append(sname)
    TOTAL = len(frame_to_step)

    def _step_t(f, step_name):
        """Progress 0→1 within the current step."""
        off = step_offsets[step_name]
        dur = dict(steps)[step_name]
        return min(1.0, (f - off + 1) / dur)

    def update(f):
        step = frame_to_step[min(f, TOTAL - 1)]
        t = _step_t(f, step)

        # ===== Phase 1a: empty detector =====
        if step == "detector_empty":
            ax_det.set_title("CLIC detector (simplified 2D cross-section)",
                             fontsize=11, color=MUTE, pad=6)
            sub.set_text("a particle collision at the centre sends "
                         "particles outward through detector layers")
            cap.set_text("inner layers = tracker (curved tracks)   "
                         "·   outer layers = calorimeter (energy deposits)")
            return

        # ===== Phase 1b: tracks appear =====
        if step == "tracks_appear":
            a = min(1.0, t * 2.5)
            for ln in track_lines:
                ln.set_alpha(a)
            for lbl in track_labels:
                lbl.set_alpha(a)
            ax_det.set_title("tracks (charged particles)",
                             fontsize=11, color=MUTE, pad=6)
            sub.set_text(f"{N_TRACKS} reconstructed tracks — "
                         "curved by the magnetic field")
            cap.set_text("each track has measured pT, η, φ")
            return

        # Always show tracks from here on.
        for ln in track_lines:
            ln.set_alpha(1.0)
        for lbl in track_labels:
            lbl.set_alpha(1.0)

        # ===== Phase 1c: topoclusters appear =====
        if step == "topos_appear":
            a = min(1.0, t * 2.5)
            topo_sc.set_alpha(a)
            for lbl in topo_labels:
                lbl.set_alpha(a)
            ax_det.set_title("topoclusters (calorimeter energy deposits)",
                             fontsize=11, color=MUTE, pad=6)
            sub.set_text(f"{N_TOPOS} topoclusters — "
                         "size ∝ deposited energy")
            cap.set_text("some clusters are shared: c2 has energy from "
                         "two particles (overlap)")
            return

        # Always show topos from here on.
        topo_sc.set_alpha(1.0)
        for lbl in topo_labels:
            lbl.set_alpha(1.0)

        # ===== Phase 1d: full event hold =====
        if step == "event_hold":
            ax_det.set_title("full event: 4 tracks + 5 topoclusters",
                             fontsize=11, color=MUTE, pad=6)
            sub.set_text(f"{N_INPUTS} detector objects total — "
                         "these are the inputs to GLOW")
            cap.set_text("goal: reconstruct the original particles "
                         "from these measurements")
            return

        # ===== Phase 2: tokenize =====
        if step == "tokenize":
            ax_det.set_title("input detector objects",
                             fontsize=11, color=MUTE, pad=6)
            ax_r.set_title("feature vectors (tokens)",
                           fontsize=11, color=MUTE, pad=6)
            n_show = max(1, int(t * N_INPUTS))
            for idx in range(N_INPUTS):
                if idx < n_show:
                    token_rects[idx].set_alpha(0.25)
                    token_texts[idx].set_alpha(1.0)
                    token_feat_texts[idx].set_alpha(1.0)
            sub.set_text("each detector object → one feature vector (token)")
            cap.set_text("tracks and topoclusters share the same "
                         "token format — heterogeneous inputs unified")
            return

        # --- from here on, all tokens visible ---
        def _show_all_tokens():
            for idx in range(N_INPUTS):
                token_rects[idx].set_alpha(0.25)
                token_texts[idx].set_alpha(1.0)
                token_feat_texts[idx].set_alpha(1.0)
        _show_all_tokens()

        # --- clear transient visuals each frame ---
        enc_bracket.set_alpha(0.0)
        enc_label.set_alpha(0.0)
        for ln in attn_lines:
            ln.set_alpha(0.0)
        dec_bracket.set_alpha(0.0)
        dec_label.set_alpha(0.0)
        for ln in xattn_lines:
            ln.set_alpha(0.0)
        for qr in query_rects:
            qr.set_alpha(0.0)
        for qt in query_texts:
            qt.set_alpha(0.0)

        # ===== Phase 3: encoder intro =====
        if step == "encoder_intro":
            enc_bracket.set_alpha(0.8)
            enc_label.set_text("transformer encoder (6 layers)")
            enc_label.set_alpha(1.0)
            ax_det.set_title("input detector objects",
                             fontsize=11, color=MUTE, pad=6)
            ax_r.set_title("self-attention encoder",
                           fontsize=11, color=MUTE, pad=6)
            sub.set_text("all 9 tokens enter a 6-layer self-attention "
                         "transformer encoder")
            cap.set_text("every token attends to every other token — "
                         "tracks learn about nearby clusters and vice versa")
            return

        # ===== Phase 3b: encoder layers (pulse) =====
        is_enc_layer = step.startswith("encoder_L")
        if is_enc_layer:
            layer_num = int(step[-1])
            enc_bracket.set_alpha(0.8)
            enc_label.set_text(f"encoder layer {layer_num + 1}/6")
            enc_label.set_alpha(1.0)
            # Draw some self-attention arcs.
            rng = np.random.default_rng(layer_num * 7)
            pairs = rng.choice(N_INPUTS, size=(min(len(attn_lines), 12), 2))
            pulse = 0.5 + 0.5 * np.sin(t * np.pi)
            for k, (i, j) in enumerate(pairs):
                if i == j:
                    continue
                yi = y0_tok - i * row_sp + box_h / 2
                yj = y0_tok - j * row_sp + box_h / 2
                attn_lines[k].set_data([x0_tok - 0.15, x0_tok - 0.15],
                                       [yi, yj])
                attn_lines[k].set_alpha(pulse * 0.5)
            ax_r.set_title(f"self-attention layer {layer_num + 1}",
                           fontsize=11, color=MUTE, pad=6)
            sub.set_text("tokens exchange information via self-attention")
            cap.set_text(f"layer {layer_num + 1}/6 — each token updates "
                         "based on all other tokens")
            return

        # ===== Phase 3c: encoder done =====
        if step == "encoder_done":
            enc_bracket.set_alpha(0.8)
            enc_label.set_text("encoded tokens ✓")
            enc_label.set_alpha(1.0)
            # Darken token rects to show they've been transformed.
            for idx in range(N_INPUTS):
                token_rects[idx].set_alpha(0.35)
            ax_r.set_title("encoded representations",
                           fontsize=11, color=MUTE, pad=6)
            sub.set_text("after 6 layers, each token encodes "
                         "context from the whole event")
            cap.set_text("a track token now knows about neighbouring "
                         "clusters — and vice versa")
            return

        # --- encoder always visible from here ---
        enc_bracket.set_alpha(0.5)
        enc_label.set_text("encoder")
        enc_label.set_alpha(0.6)
        for idx in range(N_INPUTS):
            token_rects[idx].set_alpha(0.35)

        # ===== Phase 4: queries appear =====
        if step == "queries_appear":
            a = min(1.0, t * 2.5)
            for qi in range(N_QUERIES):
                query_rects[qi].set_alpha(0.3 * a)
                query_texts[qi].set_alpha(a)
            ax_r.set_title("learnable particle queries",
                           fontsize=11, color=MUTE, pad=6)
            sub.set_text(f"{N_QUERIES} learned query embeddings — "
                         "each will try to predict one particle")
            cap.set_text("like DETR: queries are not from data, "
                         "they are learned parameters of the model")
            return

        # --- queries always visible from here ---
        for qi in range(N_QUERIES):
            query_rects[qi].set_alpha(0.3)
            query_texts[qi].set_alpha(1.0)

        # ===== Phase 5: decoder intro =====
        if step == "decoder_intro":
            dec_bracket.set_alpha(0.8)
            dec_label.set_text("masked cross-attention decoder (4 layers)")
            dec_label.set_alpha(1.0)
            ax_r.set_title("decoder: queries attend to encoded tokens",
                           fontsize=11, color=MUTE, pad=6)
            sub.set_text("each query attends to the encoded detector "
                         "objects via cross-attention")
            cap.set_text("attention masks are learned — each query "
                         "focuses on relevant detector objects")
            return

        # ===== Phase 5b: decoder pulse (4 layers) =====
        if step == "decoder_pulse":
            layer_idx = min(3, int(t * 4))
            dec_bracket.set_alpha(0.8)
            dec_label.set_text(f"decoder layer {layer_idx + 1}/4")
            dec_label.set_alpha(1.0)
            # Draw cross-attention lines from queries → tokens.
            rng = np.random.default_rng(layer_idx * 13 + 5)
            k = 0
            pulse = 0.5 + 0.5 * np.sin(t * 4 * np.pi)
            for qi in range(N_QUERIES):
                # Each query attends to ~3 random tokens.
                tgt = rng.choice(N_INPUTS, size=3, replace=False)
                qy = q_y0 - qi * row_sp + box_h / 2
                for ti in tgt:
                    ty = y0_tok - ti * row_sp + box_h / 2
                    if k < len(xattn_lines):
                        xattn_lines[k].set_data(
                            [x0_tok + box_w + 0.1, x0_tok + box_w + 0.1],
                            [qy, ty])
                        xattn_lines[k].set_alpha(pulse * 0.4)
                        k += 1
            ax_r.set_title(f"cross-attention layer {layer_idx + 1}/4",
                           fontsize=11, color=MUTE, pad=6)
            sub.set_text("queries pull information from the encoded "
                         "detector tokens")
            cap.set_text(f"layer {layer_idx + 1}/4 — masks dynamically "
                         "adjusted based on learned similarity")
            return

        # ===== Phase 5c: decoder done =====
        if step == "decoder_done":
            dec_bracket.set_alpha(0.8)
            dec_label.set_text("decoded queries ✓")
            dec_label.set_alpha(1.0)
            for qi in range(N_QUERIES):
                query_rects[qi].set_alpha(0.4)
            ax_r.set_title("decoded particle representations",
                           fontsize=11, color=MUTE, pad=6)
            sub.set_text("each query now encodes a candidate particle — "
                         "three prediction heads follow")
            cap.set_text("(i) incidence matrix   (ii) particle class   "
                         "(iii) kinematics")
            return

        # ===== From here on, hide the token/query panel and show matrix =====
        ax_r.set_visible(False)
        ax_mat.set_visible(True)
        ax_mat.clear()
        ax_mat.set_facecolor(BG)

        def _draw_incidence(highlight_row=-1, highlight_col=-1,
                            show_sums=True, n_rows=N_INPUTS,
                            n_cols=N_QUERIES):
            """Draw the incidence matrix on ax_mat."""
            ax_mat.set_xlim(-1.5, n_cols + 1.5)
            ax_mat.set_ylim(-1.5, n_rows + 0.5)
            ax_mat.set_xticks([]); ax_mat.set_yticks([])
            for s in ("left", "right", "top", "bottom"):
                ax_mat.spines[s].set_visible(False)
            vmax = max(PRED_INCIDENCE[:n_rows, :n_cols].max(), 0.01)
            for i in range(n_rows):
                # Row label.
                col = TRACK_COLOR if i < N_TRACKS else TOPO_COLOR
                ax_mat.text(-0.7, n_rows - 1 - i, obj_labels[i],
                            fontsize=9, fontweight="bold", color=col,
                            ha="right", va="center")
                for j in range(n_cols):
                    val = PRED_INCIDENCE[i, j]
                    intensity = val / vmax
                    r, g, b = (1 - 0.55*intensity, 1 - 0.25*intensity,
                               1 - 0.05*intensity)
                    ec = "white"
                    lw_rect = 1.5
                    if i == highlight_row:
                        ec = TOPO_COLOR if i >= N_TRACKS else TRACK_COLOR
                        lw_rect = 3.0
                    if j == highlight_col:
                        ec = QUERY_COLOR
                        lw_rect = 3.0
                    rect = Rectangle((j - 0.45, n_rows - 1 - i - 0.4),
                                     0.9, 0.8,
                                     facecolor=(r, g, b), edgecolor=ec,
                                     lw=lw_rect, zorder=1)
                    ax_mat.add_patch(rect)
                    txt_col = FG if val > 0.05 else MUTE
                    ax_mat.text(j, n_rows - 1 - i, f"{val:.2f}",
                                ha="center", va="center", fontsize=8,
                                fontweight="bold", color=txt_col, zorder=2)
                # Row sum.
                if show_sums:
                    rs = PRED_INCIDENCE[i, :n_cols].sum()
                    ax_mat.text(n_cols + 0.3, n_rows - 1 - i,
                                f"Σ={rs:.2f}",
                                fontsize=8, fontweight="bold",
                                color=OK_COLOR if abs(rs - 1.0) < 0.05
                                else FG,
                                ha="left", va="center")
            # Column labels.
            for j in range(n_cols):
                ax_mat.text(j, n_rows + 0.1, query_labels[j],
                            fontsize=9, fontweight="bold", color=QUERY_COLOR,
                            ha="center", va="bottom")

        # ===== Phase 6a: incidence intro =====
        if step == "incidence_intro":
            _draw_incidence(show_sums=False)
            ax_mat.set_title("incidence matrix  I[i, a]",
                             fontsize=12, color=MUTE, pad=8)
            sub.set_text("each query predicts a soft assignment over "
                         "all detector objects")
            cap.set_text("I[i,a] = fraction of object i's energy "
                         "attributed to particle a")
            return

        # ===== Phase 6b: show with row sums =====
        if step == "incidence_show":
            _draw_incidence(show_sums=True)
            ax_mat.set_title("incidence matrix — rows sum to 1.0",
                             fontsize=12, color=MUTE, pad=8)
            sub.set_text("every row sums to ~1.0 — energy conservation "
                         "by construction")
            cap.set_text("each detector object's energy is fully "
                         "distributed across particles (no energy lost)")
            return

        # ===== Phase 6c: highlight shared cluster c2 =====
        if step == "incidence_share":
            _draw_incidence(highlight_row=N_TRACKS + 2, show_sums=True)
            ax_mat.set_title("shared energy: topocluster c2",
                             fontsize=12, color=MUTE, pad=8)
            c2_row = PRED_INCIDENCE[N_TRACKS + 2]
            nz = [(j, c2_row[j]) for j in range(N_QUERIES)
                   if c2_row[j] > 0.05]
            parts = ", ".join(f"q{j}: {v:.0%}" for j, v in nz)
            sub.set_text(f"c2 splits its energy across particles: {parts}")
            cap.set_text("this is the key insight — one detector object "
                         "can contribute to multiple particles")
            return

        # ===== Phase 7: incidence-weighted sum =====
        if step == "weighted_sum":
            _draw_incidence(highlight_col=0, show_sums=True)
            # Annotate the weighted sum for particle 0.
            col_vals = PRED_INCIDENCE[:, 0]
            terms = []
            for i in range(N_INPUTS):
                if col_vals[i] > 0.05:
                    terms.append(f"{col_vals[i]:.2f}×{obj_energies[i]:.1f}")
            formula = " + ".join(terms[:3])
            if len(terms) > 3:
                formula += " + ..."
            ax_mat.set_title(
                f"particle q0 energy:  E = Σᵢ I[i,0]·Eᵢ = {pred_particle_E[0]:.1f}",
                fontsize=11, color=MUTE, pad=8)
            sub.set_text(f"E(q0) = {formula} = {pred_particle_E[0]:.1f}")
            cap.set_text("kinematics computed as incidence-weighted sums, "
                         "then refined by a regression head")
            return

        # ===== Phase 8: three prediction heads =====
        if step == "three_heads":
            _draw_incidence(show_sums=False)
            # Add annotations for the three heads.
            ax_mat.text(N_QUERIES / 2, -0.8,
                        "head (i): incidence matrix ↑",
                        ha="center", fontsize=10, color=DECODE_COLOR,
                        fontweight="bold")
            ax_mat.text(N_QUERIES / 2, -1.2,
                        "head (ii): class — photon | electron | hadron",
                        ha="center", fontsize=10, color=QUERY_COLOR,
                        fontweight="bold")
            ax_mat.text(N_QUERIES / 2, -1.6,
                        "head (iii): kinematics — E, pT, η, φ "
                        "(from weighted sum + regression)",
                        ha="center", fontsize=10, color=ENCODE_COLOR,
                        fontweight="bold")
            ax_mat.set_title("three prediction heads per query",
                             fontsize=12, color=MUTE, pad=8)
            sub.set_text("each query simultaneously predicts assignment, "
                         "particle type, and kinematics")
            cap.set_text("")
            return

        # ===== Phase 9: Hungarian matching in training =====
        if step == "hungarian_loss":
            _draw_incidence(show_sums=False)
            # Draw matching lines from predicted to GT.
            # Predicted q0→p0, q1→p1, ..., q4→p4, q5→∅
            match_colors = ["#2E8B57"] * N_PARTICLES + ["#C62828"]
            for j in range(N_QUERIES):
                y_q = -0.5
                if j < N_PARTICLES:
                    lbl = f"q{j} → particle {j}"
                    c = "#2E8B57"
                else:
                    lbl = f"q{j} → ∅ (no particle)"
                    c = "#C62828"
                ax_mat.text(j, -0.7, lbl, ha="center", fontsize=7,
                            fontweight="bold", color=c, rotation=45)
            ax_mat.set_title("training: Hungarian matching "
                             "(predictions → ground truth)",
                             fontsize=11, color=MUTE, pad=8)
            sub.set_text("permutation-invariant loss: Hungarian algorithm "
                         "finds the optimal prediction↔truth assignment")
            cap.set_text("same Hungarian algorithm from our earlier scenes "
                         "— but here it's in the loss, not the model")
            return

        # ===== Phase 10: reconstructed event =====
        if step == "reconstruct":
            ax_mat.set_visible(False)
            ax_r.set_visible(True)
            # Clear ax_r and draw reconstructed particles as summary.
            ax_r.clear()
            ax_r.set_xlim(0, 10); ax_r.set_ylim(-1, 10)
            ax_r.set_xticks([]); ax_r.set_yticks([])
            for s in ("left", "right", "top", "bottom"):
                ax_r.spines[s].set_visible(False)
            ax_r.set_facecolor(BG)
            ax_r.set_title("reconstructed particles",
                           fontsize=11, color=MUTE, pad=6)
            ptypes = ["hadron", "hadron", "hadron", "hadron",
                      "photon"]
            pcols = [TRACK_COLOR, ENCODE_COLOR, QUERY_COLOR,
                     DECODE_COLOR, TOPO_COLOR]
            for pi in range(N_PARTICLES):
                y = 8.0 - pi * 1.6
                ax_r.add_patch(Rectangle((1, y), 8, 1.2,
                               facecolor=pcols[pi], alpha=0.2,
                               edgecolor=pcols[pi], lw=2, zorder=1))
                ax_r.text(1.5, y + 0.6,
                          f"particle {pi}  ({ptypes[pi]})",
                          fontsize=10, fontweight="bold",
                          color=pcols[pi], va="center", zorder=2)
                ax_r.text(6.0, y + 0.6,
                          f"E = {pred_particle_E[pi]:.1f}",
                          fontsize=10, color=FG, va="center", zorder=2)
            ax_det.set_title("input event",
                             fontsize=11, color=MUTE, pad=6)
            sub.set_text(f"{N_PARTICLES} particles reconstructed from "
                         f"{N_INPUTS} detector objects")
            cap.set_text("GLOW: a single transformer handles "
                         "tracks + clusters → particles, end to end")
            return

        # ===== Phase 11: final hold =====
        ax_mat.set_visible(False)
        ax_r.set_visible(True)
        ax_r.clear()
        ax_r.set_xlim(0, 10); ax_r.set_ylim(-1, 10)
        ax_r.set_xticks([]); ax_r.set_yticks([])
        for s in ("left", "right", "top", "bottom"):
            ax_r.spines[s].set_visible(False)
        ax_r.set_facecolor(BG)
        ax_r.set_title("", fontsize=11, color=MUTE, pad=6)
        ax_r.text(5, 7, "GLOW", fontsize=28, fontweight="bold",
                  ha="center", color=FG)
        ax_r.text(5, 5.5, "encoder: 6-layer self-attention",
                  fontsize=11, ha="center", color=ENCODE_COLOR)
        ax_r.text(5, 4.5, "decoder: 4-layer masked cross-attention",
                  fontsize=11, ha="center", color=DECODE_COLOR)
        ax_r.text(5, 3.5, "incidence matrix: energy-conserving "
                  "soft assignment",
                  fontsize=11, ha="center", color=QUERY_COLOR)
        ax_r.text(5, 2.5, "Hungarian matching in the training loss",
                  fontsize=11, ha="center", color=FG)
        ax_r.text(5, 1.2, "Kobylianskii et al., 2025",
                  fontsize=10, ha="center", color=MUTE)
        ax_r.text(5, 0.5, "arxiv: 2508.20092",
                  fontsize=10, ha="center", color=MUTE)
        sub.set_text("a unified transformer for particle flow "
                     "reconstruction")
        cap.set_text("differentiable · scalable · "
                     "energy-conserving by construction")

    anim = animation.FuncAnimation(fig, update, frames=TOTAL,
                                   interval=1000/fps, blit=False)
    save_animation(anim, out_path, fps=fps)
    plt.close(fig)
    print(f"wrote {out_path}  ({TOTAL} frames, {fps} fps)")
