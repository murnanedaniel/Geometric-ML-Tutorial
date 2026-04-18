#!/usr/bin/env python
"""CLI runner for particle-flow scenes.

    python scripts/run_particle_flow.py glow
    python scripts/run_particle_flow.py all
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from particle_flow.helpers import OUT_DIR  # noqa: E402

SCENES = {
    "glow": ("03a_glow_architecture.gif", "particle_flow.scene_glow"),
}

def main():
    targets = sys.argv[1:] if len(sys.argv) > 1 else ["glow"]
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
