"""Reward functions for S1 single-shot tasks."""

from __future__ import annotations

from typing import Any, Mapping

from MLAgent.envs.event_parser import ParsedS1Events


def _read_flag(parsed: Mapping[str, Any] | ParsedS1Events, key: str) -> bool:
    if isinstance(parsed, ParsedS1Events):
        return bool(getattr(parsed, key))
    return bool(parsed[key])


def compute_s1_reward(parsed: Mapping[str, Any] | ParsedS1Events) -> float:
    """Compute the phase-1 S1 reward.

    Rules:
    - target ball in target pocket: +1.0
    - cue scratch: -1.0
    - illegal first hit: -1.0
    - simulation failure / invalid outcome: -1.0
    - otherwise: 0.0
    """

    success = _read_flag(parsed, "success")
    cue_scratch = _read_flag(parsed, "cue_scratch")
    legal_first_hit = _read_flag(parsed, "legal_first_hit")
    simulation_ended_normally = _read_flag(parsed, "simulation_ended_normally")

    if not simulation_ended_normally:
        return -1.0
    if cue_scratch:
        return -1.0
    if not legal_first_hit:
        return -1.0
    if success:
        return 1.0
    return 0.0

