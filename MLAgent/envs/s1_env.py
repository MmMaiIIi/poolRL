"""Gymnasium wrapper for the Phase-1 S1 bare core."""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
import pooltool as pt
from gymnasium import spaces

from MLAgent.envs.pooltool_core import PoolToolS1Core

DEFAULT_PRESET_NAME = "five_point_straight_in"
DEFAULT_PHI_HALF_WINDOW_DEG = 12.0
DEFAULT_V0_MIN = 1.2
DEFAULT_V0_MAX = 2.2
DEFAULT_HEURISTIC_V0 = 1.8


class PoolToolS1Env(gym.Env):
    """Single-shot S1 Gymnasium environment.

    Observation:
    - 6D float vector: [x_cue, y_cue, x_obj, y_obj, x_pocket, y_pocket]

    Action:
    - 2D normalized vector in [-1, 1]
    - action[0] maps to phi inside a narrow window centered on the ideal direction
    - action[1] maps to V0 inside a narrow fixed speed range
    """

    metadata = {"render_modes": [None, "human"]}

    def __init__(
        self,
        *,
        preset_name: str = DEFAULT_PRESET_NAME,
        render_mode: str | None = None,
        phi_half_window_deg: float = DEFAULT_PHI_HALF_WINDOW_DEG,
        v0_min: float = DEFAULT_V0_MIN,
        v0_max: float = DEFAULT_V0_MAX,
        heuristic_v0: float = DEFAULT_HEURISTIC_V0,
        max_events: int = 5000,
    ) -> None:
        super().__init__()

        if render_mode not in {None, "human"}:
            raise ValueError("render_mode must be one of: None, 'human'.")
        if phi_half_window_deg <= 0:
            raise ValueError("phi_half_window_deg must be positive.")
        if v0_max <= v0_min:
            raise ValueError("v0_max must be greater than v0_min.")

        self.preset_name = preset_name
        self.render_mode = render_mode
        self.phi_half_window_deg = float(phi_half_window_deg)
        self.v0_min = float(v0_min)
        self.v0_max = float(v0_max)
        self.heuristic_v0 = float(heuristic_v0)

        self.core = PoolToolS1Core(max_events=max_events)
        self.last_system: pt.System | None = None

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(6,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

        self._episode_done = False
        self._last_obs: np.ndarray | None = None

    @staticmethod
    def _wrap_angle_deg(phi_deg: float) -> float:
        return float(phi_deg) % 360.0

    @staticmethod
    def _angle_diff_deg(lhs_deg: float, rhs_deg: float) -> float:
        """Signed angular difference lhs-rhs in degrees, wrapped to [-180, 180)."""

        return ((lhs_deg - rhs_deg + 180.0) % 360.0) - 180.0

    @staticmethod
    def _ideal_phi_degrees_from_obs(obs: np.ndarray) -> float:
        """Ideal cue direction for first contact (cue -> object)."""

        dx = float(obs[2] - obs[0])
        dy = float(obs[3] - obs[1])
        return math.degrees(math.atan2(dy, dx)) % 360.0

    def map_action_to_physical(
        self,
        action: np.ndarray | list[float] | tuple[float, float],
        *,
        obs: np.ndarray | None = None,
    ) -> tuple[float, float]:
        """Map normalized action in [-1, 1]^2 to (phi, V0)."""

        norm_action = np.asarray(action, dtype=np.float64).reshape(-1)
        if norm_action.shape != (2,):
            raise ValueError(f"Action must have shape (2,), got {norm_action.shape}.")
        norm_action = np.clip(norm_action, -1.0, 1.0)

        reference_obs = self._last_obs if obs is None else np.asarray(obs, dtype=np.float64)
        if reference_obs is None:
            raise RuntimeError("No observation available. Call reset() first.")

        ideal_phi = self._ideal_phi_degrees_from_obs(reference_obs)
        phi = self._wrap_angle_deg(ideal_phi + norm_action[0] * self.phi_half_window_deg)
        v0 = self.v0_min + ((norm_action[1] + 1.0) * 0.5) * (self.v0_max - self.v0_min)
        return float(phi), float(v0)

    def normalized_action_from_physical(
        self,
        phi: float,
        V0: float,
        *,
        obs: np.ndarray | None = None,
    ) -> np.ndarray:
        """Inverse map from physical action to normalized action in [-1, 1]^2."""

        reference_obs = self._last_obs if obs is None else np.asarray(obs, dtype=np.float64)
        if reference_obs is None:
            raise RuntimeError("No observation available. Call reset() first.")

        ideal_phi = self._ideal_phi_degrees_from_obs(reference_obs)
        phi_offset = self._angle_diff_deg(float(phi), ideal_phi)
        norm_phi = np.clip(phi_offset / self.phi_half_window_deg, -1.0, 1.0)

        v0_span = self.v0_max - self.v0_min
        norm_v0 = np.clip((2.0 * (float(V0) - self.v0_min) / v0_span) - 1.0, -1.0, 1.0)

        return np.array([norm_phi, norm_v0], dtype=np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        selected_preset = self.preset_name
        if options is not None and "preset_name" in options:
            selected_preset = str(options["preset_name"])

        layout = self.core.build_layout(selected_preset)
        obs = self.core.encode_obs().astype(np.float32)
        self._last_obs = obs
        self._episode_done = False
        self.last_system = None

        info: dict[str, Any] = {
            "preset_name": layout.preset_name,
            "layout_id": layout.layout_id,
            "target_pocket_id": layout.target_pocket_id,
        }
        return obs, info

    def step(
        self,
        action: np.ndarray | list[float] | tuple[float, float],
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._episode_done:
            raise RuntimeError("Episode already terminated. Call reset() before step().")

        if self.core.system is None or self.core.layout is None:
            raise RuntimeError("Environment is not initialized. Call reset() first.")

        phi, V0 = self.map_action_to_physical(action)
        self.core.apply_action(phi=phi, V0=V0)
        self.core.simulate_once()
        parsed = self.core.parse_events()
        reward = float(self.core.compute_reward())
        obs = self.core.encode_obs().astype(np.float32)
        self._last_obs = obs
        self._episode_done = True

        system = self.core.system
        self.last_system = system.copy()
        layout = self.core.layout
        obj_final_pos = (
            float(system.balls[self.core.object_ball_id].xyz[0]),
            float(system.balls[self.core.object_ball_id].xyz[1]),
        )
        cue_final_pos = (
            float(system.balls[self.core.cue_ball_id].xyz[0]),
            float(system.balls[self.core.cue_ball_id].xyz[1]),
        )

        info: dict[str, Any] = {
            "preset_name": layout.preset_name,
            "layout_id": layout.layout_id,
            "success": bool(parsed.success),
            "cue_scratch": bool(parsed.cue_scratch),
            "legal_first_hit": bool(parsed.legal_first_hit),
            "phi": float(phi),
            "V0": float(V0),
            "obj_final_pos": obj_final_pos,
            "cue_final_pos": cue_final_pos,
            "termination_reason": parsed.termination_reason,
            "events_summary": parsed.events_summary,
        }

        terminated = True
        truncated = False
        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        """Render the latest simulated shot in pooltool GUI when enabled.

        Behavior:
        - `render_mode is None`: no-op
        - `render_mode == "human"` and no completed shot yet: no-op
        - `render_mode == "human"` and shot exists: call `pooltool.show(...)`
        """

        if self.render_mode is None:
            return

        if self.last_system is None:
            return

        try:
            pt.show(self.last_system)
        except Exception as exc:  # pragma: no cover - depends on GUI runtime
            raise RuntimeError(
                "Pooltool human rendering failed. "
                "Your runtime may not support GUI rendering."
            ) from exc

    def close(self) -> None:
        self.last_system = None
