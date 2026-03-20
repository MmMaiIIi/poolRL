"""Smoke tests for optional render behavior in S1 eval path."""

from __future__ import annotations

import os

import pytest

from MLAgent.envs.s1_env import PoolToolS1Env
from MLAgent.train.eval_s1 import run_evaluation


def test_env_render_noop_without_shot_and_headless_mode() -> None:
    env = PoolToolS1Env(render_mode=None)
    # Should be a no-op and not crash before any simulation.
    env.render()
    env.close()


def test_headless_eval_path_runs_without_visualization() -> None:
    summary = run_evaluation(
        episodes=3,
        preset_name="five_point_straight_in",
        policy="heuristic",
        seed=0,
        render_human=False,
    )
    assert summary["episodes"] == 3
    assert summary["render_human_requested"] is False
    assert summary["rendered_episodes"] == 0


def test_human_render_smoke_skips_safely_when_gui_unavailable() -> None:
    # CI should remain headless by default.
    if os.environ.get("MLAGENT_ENABLE_GUI_SMOKE", "0") != "1":
        pytest.skip("GUI smoke test disabled. Set MLAGENT_ENABLE_GUI_SMOKE=1 to run.")

    try:
        summary = run_evaluation(
            episodes=1,
            preset_name="five_point_straight_in",
            policy="heuristic",
            seed=0,
            render_human=True,
            render_episodes=1,
        )
    except RuntimeError as exc:
        pytest.skip(f"GUI unavailable in this runtime: {exc}")

    assert summary["episodes"] == 1
    assert summary["render_human_requested"] is True

