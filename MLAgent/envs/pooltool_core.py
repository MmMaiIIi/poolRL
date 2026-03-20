"""Minimal non-Gym bare core for S1 single-shot simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pooltool as pt
from pooltool.objects.ball.sets import BallSet

from MLAgent.envs.event_parser import ParsedS1Events, parse_shot_events
from MLAgent.envs.layout_generators import S1Layout, build_s1_layout
from MLAgent.envs.reward_fns import compute_s1_reward


@dataclass
class ShotAction:
    """Manual shot action in physical units."""

    phi: float
    V0: float


class PoolToolS1Core:
    """Bare S1 core: build layout, apply action, simulate once, parse outcome."""

    cue_ball_id: str = "cue"
    object_ball_id: str = "1"

    def __init__(self, *, max_events: int = 5000) -> None:
        self.max_events = int(max_events)
        self.table: pt.Table | None = None
        self.layout: S1Layout | None = None
        self.system: pt.System | None = None
        self.action: ShotAction | None = None
        self._simulation_error: str | None = None
        self._parsed: ParsedS1Events | None = None

    def build_layout(self, preset_name: str) -> S1Layout:
        """Create table + balls + cue for a fixed S1 preset."""

        self.table = pt.Table.default()
        self.layout = build_s1_layout(preset_name=preset_name, table=self.table)

        cue_ball = pt.Ball.create(self.cue_ball_id, xy=self.layout.cue_ball_pos)
        object_ball = pt.Ball.create(self.object_ball_id, xy=self.layout.object_ball_pos)
        cue = pt.Cue(cue_ball_id=self.cue_ball_id)

        self.system = pt.System(
            cue=cue,
            table=self.table,
            balls={
                self.cue_ball_id: cue_ball,
                self.object_ball_id: object_ball,
            },
        )
        # Associate a real pool ball set so GUI rendering shows the cue/object balls
        # with proper models rather than leaving manual balls under-specified.
        self.system.set_ballset(BallSet("pooltool_pocket"))
        self.action = None
        self._simulation_error = None
        self._parsed = None

        return self.layout

    def encode_obs(self) -> np.ndarray:
        """Encode minimal S1 observation: cue/object/pocket XY."""

        if self.layout is None or self.system is None:
            raise RuntimeError("Call build_layout(...) before encode_obs().")

        cue_xyz = self.system.balls[self.cue_ball_id].xyz
        obj_xyz = self.system.balls[self.object_ball_id].xyz
        pocket_x, pocket_y = self.layout.target_pocket_pos

        return np.array(
            [
                float(cue_xyz[0]),
                float(cue_xyz[1]),
                float(obj_xyz[0]),
                float(obj_xyz[1]),
                float(pocket_x),
                float(pocket_y),
            ],
            dtype=np.float64,
        )

    def apply_action(self, phi: float, V0: float) -> None:
        """Apply one manual action `(phi, V0)` to the cue."""

        if self.system is None:
            raise RuntimeError("Call build_layout(...) before apply_action(...).")

        # Phase-1 scope: no english and no elevation.
        action = ShotAction(phi=float(phi) % 360.0, V0=max(0.0, float(V0)))
        self.system.strike(phi=action.phi, V0=action.V0, a=0.0, b=0.0, theta=0.0)
        self.action = action

    def simulate_once(self) -> bool:
        """Run exactly one pooltool simulation on the current shot."""

        if self.system is None:
            raise RuntimeError("Call build_layout(...) before simulate_once().")

        self._simulation_error = None
        try:
            pt.simulate(self.system, inplace=True, max_events=self.max_events)
            return True
        except Exception as exc:  # pragma: no cover - defensive path
            self._simulation_error = f"{type(exc).__name__}: {exc}"
            return False

    def parse_events(self) -> ParsedS1Events:
        """Parse simulation events into S1 outcome signals."""

        if self.layout is None:
            raise RuntimeError("Call build_layout(...) before parse_events().")

        self._parsed = parse_shot_events(
            self.system,
            target_pocket_id=self.layout.target_pocket_id,
            cue_ball_id=self.cue_ball_id,
            object_ball_id=self.object_ball_id,
            simulation_error=self._simulation_error,
        )
        return self._parsed

    def compute_reward(self) -> float:
        """Compute S1 reward from the latest parsed outcome."""

        if self._parsed is None:
            raise RuntimeError("Call parse_events() before compute_reward().")
        return compute_s1_reward(self._parsed)

    def run_shot(self, preset_name: str, phi: float, V0: float) -> dict[str, Any]:
        """End-to-end one-shot run and structured result."""

        layout = self.build_layout(preset_name=preset_name)
        _ = self.encode_obs()
        self.apply_action(phi=phi, V0=V0)
        self.simulate_once()
        parsed = self.parse_events()
        reward = self.compute_reward()

        if self.system is None or self.action is None:
            raise RuntimeError("Shot run did not initialize system/action correctly.")

        obj_final_pos = (
            float(self.system.balls[self.object_ball_id].xyz[0]),
            float(self.system.balls[self.object_ball_id].xyz[1]),
        )
        cue_final_pos = (
            float(self.system.balls[self.cue_ball_id].xyz[0]),
            float(self.system.balls[self.cue_ball_id].xyz[1]),
        )

        return {
            "preset_name": layout.preset_name,
            "layout_id": layout.layout_id,
            "phi": float(self.action.phi),
            "V0": float(self.action.V0),
            "success": bool(parsed.success),
            "cue_scratch": bool(parsed.cue_scratch),
            "legal_first_hit": bool(parsed.legal_first_hit),
            "termination_reason": parsed.termination_reason,
            "reward": float(reward),
            "obj_final_pos": obj_final_pos,
            "cue_final_pos": cue_final_pos,
            "events_summary": parsed.events_summary,
        }
