"""Manual Phase-1 debug script for one S1 shot."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from pprint import pprint

# Ensure `python MLAgent/scripts/debug_single_shot.py` works from repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MLAgent.envs.pooltool_core import PoolToolS1Core

# Manual knobs for one-shot debugging.
MANUAL_PRESET = "five_point_straight_in"
MANUAL_V0 = 1.8
MANUAL_PHI: float | None = None


def _direct_hit_phi_degrees(
    cue_xy: tuple[float, float], object_xy: tuple[float, float]
) -> float:
    dx = object_xy[0] - cue_xy[0]
    dy = object_xy[1] - cue_xy[1]
    return math.degrees(math.atan2(dy, dx)) % 360.0


def main() -> None:
    core = PoolToolS1Core()
    preview_layout = core.build_layout(MANUAL_PRESET)

    phi = (
        MANUAL_PHI
        if MANUAL_PHI is not None
        else _direct_hit_phi_degrees(
            preview_layout.cue_ball_pos, preview_layout.object_ball_pos
        )
    )

    result = core.run_shot(preset_name=MANUAL_PRESET, phi=phi, V0=MANUAL_V0)

    print("=== S1 Debug Single Shot ===")
    print(f"preset: {result['preset_name']}")
    print(f"layout_id: {result['layout_id']}")
    print(f"action: phi={result['phi']:.3f}, V0={result['V0']:.3f}")
    print(f"reward: {result['reward']:.3f}")
    print(f"success: {result['success']}")
    print(f"scratch: {result['cue_scratch']}")
    print(f"legal_first_hit: {result['legal_first_hit']}")
    print(f"obj_final_pos: {result['obj_final_pos']}")
    print(f"cue_final_pos: {result['cue_final_pos']}")
    print("events_summary:")
    pprint(result["events_summary"], sort_dicts=False)


if __name__ == "__main__":
    main()
