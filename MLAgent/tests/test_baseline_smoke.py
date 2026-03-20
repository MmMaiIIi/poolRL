"""Phase-2 baseline smoke tests."""

from __future__ import annotations

from MLAgent.baselines.heuristic_baseline import run_heuristic_baseline
from MLAgent.baselines.random_baseline import run_random_baseline


def _assert_summary(summary: dict[str, object], expected_baseline: str) -> None:
    assert summary["baseline"] == expected_baseline
    assert summary["episodes"] == 3
    assert isinstance(summary["mean_reward"], float)
    assert isinstance(summary["success_rate"], float)
    assert isinstance(summary["cue_scratch_rate"], float)
    assert isinstance(summary["illegal_first_hit_rate"], float)

    assert 0.0 <= float(summary["success_rate"]) <= 1.0
    assert 0.0 <= float(summary["cue_scratch_rate"]) <= 1.0
    assert 0.0 <= float(summary["illegal_first_hit_rate"]) <= 1.0


def test_random_baseline_smoke() -> None:
    summary = run_random_baseline(episodes=3, preset_name="five_point_straight_in", seed=7)
    _assert_summary(summary, expected_baseline="random")


def test_heuristic_baseline_smoke() -> None:
    summary = run_heuristic_baseline(
        episodes=3,
        preset_name="five_point_straight_in",
        seed=7,
    )
    _assert_summary(summary, expected_baseline="heuristic")

