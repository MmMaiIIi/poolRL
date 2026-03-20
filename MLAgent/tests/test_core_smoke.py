"""Smoke tests for the minimal Phase-1 S1 core path."""

from __future__ import annotations

import math

from MLAgent.envs.pooltool_core import PoolToolS1Core


def _phi_to_object(cue_xy: tuple[float, float], object_xy: tuple[float, float]) -> float:
    dx = object_xy[0] - cue_xy[0]
    dy = object_xy[1] - cue_xy[1]
    return math.degrees(math.atan2(dy, dx)) % 360.0


def test_core_run_shot_smoke() -> None:
    core = PoolToolS1Core()
    layout = core.build_layout("five_point_straight_in")

    obs = core.encode_obs()
    assert obs.shape == (6,)

    phi = _phi_to_object(layout.cue_ball_pos, layout.object_ball_pos)
    result = core.run_shot("five_point_straight_in", phi=phi, V0=1.8)

    required_keys = {
        "preset_name",
        "layout_id",
        "phi",
        "V0",
        "success",
        "cue_scratch",
        "legal_first_hit",
        "termination_reason",
        "reward",
        "obj_final_pos",
        "cue_final_pos",
        "events_summary",
    }
    assert required_keys.issubset(result.keys())

