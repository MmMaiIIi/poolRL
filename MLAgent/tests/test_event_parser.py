"""Tests for S1 event parsing utilities."""

from __future__ import annotations

import pooltool as pt

from MLAgent.envs.event_parser import parse_shot_events


def test_parse_shot_events_smoke() -> None:
    system = pt.System.example()
    pt.simulate(system, inplace=True)

    parsed = parse_shot_events(
        system,
        target_pocket_id="rt",
        cue_ball_id="cue",
        object_ball_id="1",
    )

    assert isinstance(parsed.success, bool)
    assert isinstance(parsed.cue_scratch, bool)
    assert isinstance(parsed.legal_first_hit, bool)
    assert isinstance(parsed.simulation_ended_normally, bool)
    assert isinstance(parsed.events_summary, dict)
    assert parsed.events_summary["total_events"] > 0
    assert "event_type_counts" in parsed.events_summary
    assert "pocket_events" in parsed.events_summary


def test_parse_shot_events_marks_simulation_error() -> None:
    parsed = parse_shot_events(
        system=None,
        target_pocket_id="rt",
        cue_ball_id="cue",
        object_ball_id="1",
        simulation_error="RuntimeError: forced failure",
    )

    assert parsed.success is False
    assert parsed.simulation_ended_normally is False
    assert parsed.termination_reason == "simulation_error"
    assert parsed.events_summary["simulation_error"] is not None

