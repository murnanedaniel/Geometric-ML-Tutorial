"""Shared helpers for the particle-flow visualisation scenes."""

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
TRACK_COLOR  = "#2A9D8F"   # teal — charged tracks
TOPO_COLOR   = "#E85D4A"   # ember — topoclusters
QUERY_COLOR  = "#6E63B5"   # violet — particle queries
ENCODE_COLOR = "#3D7ABF"   # blue — encoder
DECODE_COLOR = "#C77DBA"   # pink — decoder
OK_COLOR     = "#2E8B57"
GREY = np.array([0.54, 0.56, 0.60, 1.0])

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "assets" / "particle_flow"
OUT_DIR.mkdir(parents=True, exist_ok=True)
