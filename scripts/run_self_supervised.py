#!/usr/bin/env python
"""CLI runner for self-supervised training scenes.

    python scripts/run_self_supervised.py autoregressive
    python scripts/run_self_supervised.py masked
    python scripts/run_self_supervised.py diffusion
    python scripts/run_self_supervised.py all
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from self_supervised.helpers import OUT_DIR  # noqa: E402

SCENES = {
    "autoregressive": ("04a_autoregressive.gif", "self_supervised.scene_autoregressive"),
    "masked":         ("04b_masked_modelling.gif", "self_supervised.scene_masked"),
    "diffusion":      ("04c_diffusion.gif",       "self_supervised.scene_diffusion"),
}

def main():
    targets = sys.argv[1:] if len(sys.argv) > 1 else list(SCENES)
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
