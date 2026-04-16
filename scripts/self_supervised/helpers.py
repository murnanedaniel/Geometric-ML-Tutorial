"""Shared helpers for self-supervised training visualisation scenes."""

from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from metric_learning import (  # noqa: E402
    PillowWriterNoLoop, save_animation, DEFAULT_FPS,
    BG, FG, MUTE, GRID, _ease_inout,
)

# -- Colours -----------------------------------------------------------------
TOKEN_COLORS = {
    "the":    "#3D7ABF",
    "cat":    "#E85D4A",
    "sat":    "#2A9D8F",
    "on":     "#8B6DB0",
    "a":      "#D4A843",
    "mat":    "#C77DBA",
    "and":    "#5B8C5A",
    "purred": "#E8913A",
}
DEFAULT_TOKEN_COLOR = "#6E7B8B"

MASK_COLOR   = "#4A4A4A"
PREDICT_COLOR = "#2E8B57"
CORRECT_COLOR = "#2E8B57"
WRONG_COLOR  = "#C62828"
CAUSAL_COLOR = "#C0C0C0"

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "assets" / "self_supervised"
OUT_DIR.mkdir(parents=True, exist_ok=True)
