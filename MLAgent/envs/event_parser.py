"""Minimal event parsing for S1 single-shot outcome extraction."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import pooltool as pt
import pooltool.constants as const


@dataclass(frozen=True)
class ParsedS1Events:
    """Parsed S1 shot outcome with explicit best-effort signals."""

    success: bool
    cue_scratch: bool
    legal_first_hit: bool
    simulation_ended_normally: bool
    termination_reason: str
    events_summary: dict[str, Any]
    uncertainty_notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "cue_scratch": self.cue_scratch,
            "legal_first_hit": self.legal_first_hit,
            "simulation_ended_normally": self.simulation_ended_normally,
            "termination_reason": self.termination_reason,
            "events_summary": self.events_summary,
            "uncertainty_notes": list(self.uncertainty_notes),
        }


def _extract_ball_and_pocket_ids(event: pt.events.Event) -> tuple[str | None, str | None]:
    ball_id: str | None = None
    pocket_id: str | None = None

    for agent in event.agents:
        if agent.agent_type == pt.events.AgentType.BALL:
            ball_id = agent.id
        elif agent.agent_type == pt.events.AgentType.POCKET:
            pocket_id = agent.id

    return ball_id, pocket_id


def _first_cue_ball_contacted_ball_id(
    events: list[pt.events.Event],
    cue_ball_id: str,
) -> str | None:
    """Find the first ball hit by the cue ball.

    Best effort: this assumes first legal-hit logic can be inferred from the earliest
    `BALL_BALL` event involving the cue ball.
    """

    for event in events:
        if event.event_type != pt.EventType.BALL_BALL:
            continue

        if cue_ball_id not in event.ids:
            continue

        if len(event.ids) != 2:
            return None

        left_id, right_id = event.ids
        return right_id if left_id == cue_ball_id else left_id

    return None


def parse_shot_events(
    system: pt.System | None,
    *,
    target_pocket_id: str,
    cue_ball_id: str = "cue",
    object_ball_id: str = "1",
    simulation_error: str | None = None,
) -> ParsedS1Events:
    """Parse core S1 signals from a simulated shot."""

    if system is None:
        return ParsedS1Events(
            success=False,
            cue_scratch=False,
            legal_first_hit=False,
            simulation_ended_normally=False,
            termination_reason="simulation_error" if simulation_error else "missing_system",
            events_summary={
                "total_events": 0,
                "event_type_counts": {},
                "pocket_events": [],
                "target_pocket_events": [],
                "first_cue_ball_hit_ball_id": None,
                "last_event_type": None,
                "simulation_error": simulation_error,
                "uncertainty_notes": ["System object is missing; parser could not inspect events."],
            },
            uncertainty_notes=["System object is missing; parser could not inspect events."],
        )

    events = list(system.events)
    uncertainty_notes: list[str] = []
    pocket_events: list[dict[str, Any]] = []

    for event in events:
        if event.event_type != pt.EventType.BALL_POCKET:
            continue

        ball_id, pocket_id = _extract_ball_and_pocket_ids(event)
        if ball_id is None or pocket_id is None:
            uncertainty_notes.append(
                "Encountered BALL_POCKET event without explicit ball/pocket agent IDs."
            )
            continue

        pocket_events.append(
            {
                "ball_id": ball_id,
                "pocket_id": pocket_id,
                "time": float(event.time),
            }
        )

    target_pocket_events = [
        item
        for item in pocket_events
        if item["ball_id"] == object_ball_id and item["pocket_id"] == target_pocket_id
    ]

    success = bool(target_pocket_events)

    cue_scratch = any(item["ball_id"] == cue_ball_id for item in pocket_events)
    if not cue_scratch and cue_ball_id in system.balls:
        # Fallback: if event stream missed the pocket event, inspect final motion state.
        cue_scratch = system.balls[cue_ball_id].state.s == const.pocketed

    first_hit_ball_id = _first_cue_ball_contacted_ball_id(events=events, cue_ball_id=cue_ball_id)
    legal_first_hit = first_hit_ball_id == object_ball_id
    if first_hit_ball_id is None:
        uncertainty_notes.append(
            "No cue-ball BALL_BALL event found; legal_first_hit inferred as False."
        )

    last_event_type = events[-1].event_type.value if events else None
    last_event_is_none = bool(events) and events[-1].event_type == pt.EventType.NONE
    all_balls_stopped_or_pocketed = all(
        ball.state.s in {const.stationary, const.pocketed}
        for ball in system.balls.values()
    )
    simulation_ended_normally = (
        simulation_error is None
        and bool(events)
        and last_event_is_none
        and all_balls_stopped_or_pocketed
    )

    event_type_counts = Counter(event.event_type.value for event in events)

    if simulation_error:
        termination_reason = "simulation_error"
    elif not simulation_ended_normally:
        termination_reason = "abnormal_termination"
    elif cue_scratch:
        termination_reason = "cue_scratch"
    elif not legal_first_hit:
        termination_reason = "illegal_first_hit"
    elif success:
        termination_reason = "target_ball_pocketed"
    else:
        termination_reason = "shot_complete_no_score"

    events_summary: dict[str, Any] = {
        "total_events": len(events),
        "event_type_counts": dict(event_type_counts),
        "pocket_events": pocket_events,
        "target_pocket_events": target_pocket_events,
        "first_cue_ball_hit_ball_id": first_hit_ball_id,
        "last_event_type": last_event_type,
        "simulation_error": simulation_error,
        "uncertainty_notes": list(uncertainty_notes),
    }

    return ParsedS1Events(
        success=success,
        cue_scratch=cue_scratch,
        legal_first_hit=legal_first_hit,
        simulation_ended_normally=simulation_ended_normally,
        termination_reason=termination_reason,
        events_summary=events_summary,
        uncertainty_notes=uncertainty_notes,
    )

