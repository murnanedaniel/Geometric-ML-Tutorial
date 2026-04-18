#!/usr/bin/env python
"""CLI runner for set-matching scenes.

    python scripts/run_set_matching.py 02a 02b 02c ...
    python scripts/run_set_matching.py all
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from set_matching.helpers import OUT_DIR  # noqa: E402

SCENES = {
    "02a": ("02a_twins_dataset.gif",      "set_matching.scene_02a"),
    "02b": ("02b_chamfer_local.gif",      "set_matching.scene_02b"),
    "02c": ("02c_chamfer_full.gif",       "set_matching.scene_02c"),
    "02d": ("02d_chamfer_optim.gif",      "set_matching.scene_02d"),
    "02e_math": ("02e_math_hungarian.gif","set_matching.scene_02e_math"),
    "02e": ("02e_hungarian_concept.gif",  "set_matching.scene_02e"),
    "02f": ("02f_hungarian_full.gif",     "set_matching.scene_02f"),
    "02g": ("02g_hungarian_diff.gif",     "set_matching.scene_02g"),
    "02h": ("02h_hungarian_optim.gif",    "set_matching.scene_02h"),
    "02i_math": ("02i_math_sinkhorn.gif", "set_matching.scene_02i_math"),
    "02i": ("02i_sinkhorn_steps.gif",     "set_matching.scene_02i"),
    "02j": ("02j_sinkhorn_sweep.gif",     "set_matching.scene_02j"),
    "02k": ("02k_sinkhorn_optim.gif",     "set_matching.scene_02k"),
    "02l": ("02l_synthesis.gif",          "set_matching.scene_02l"),
}

def main():
    targets = sys.argv[1:] if len(sys.argv) > 1 else ["02a"]
    if targets == ["all"]:
        targets = list(SCENES)
    for t in targets:
        if t not in SCENES:
            raise SystemExit(f"unknown scene {t!r}; choose from {list(SCENES)}")
        fname, mod_name = SCENES[t]
        print(f"\n--- {t} ---")
        import importlib
        mod = importlib.import_module(mod_name)
        mod.render(OUT_DIR / fname)

if __name__ == "__main__":
    main()
