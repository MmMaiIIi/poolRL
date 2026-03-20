"""Heuristic baseline for the S1 Gymnasium environment.

Usage:
    python MLAgent/baselines/heuristic_baseline.py --episodes 100
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Ensure `python MLAgent/baselines/heuristic_baseline.py` works from repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MLAgent.envs.s1_env import PoolToolS1Env


def _angle_diff_deg(lhs_deg: float, rhs_deg: float) -> float:
    return ((lhs_deg - rhs_deg + 180.0) % 360.0) - 180.0


def build_heuristic_action(env: PoolToolS1Env, obs: np.ndarray) -> np.ndarray:
    """Compute a simple geometric action and return normalized action."""

    cue_xy = (float(obs[0]), float(obs[1]))
    obj_xy = (float(obs[2]), float(obs[3]))
    pocket_xy = (float(obs[4]), float(obs[5]))

    # Direction of cue-ball approach to hit object ball.
    phi_cue_to_obj = math.degrees(
        math.atan2(obj_xy[1] - cue_xy[1], obj_xy[0] - cue_xy[0])
    ) % 360.0

    # Desired object-ball travel direction to pocket.
    phi_obj_to_pocket = math.degrees(
        math.atan2(pocket_xy[1] - obj_xy[1], pocket_xy[0] - obj_xy[0])
    ) % 360.0

    # Blend both geometric cues; for near-collinear presets this stays very close to
    # straight-in while still using cue->object->pocket information.
    phi_blend = (
        phi_cue_to_obj + 0.30 * _angle_diff_deg(phi_obj_to_pocket, phi_cue_to_obj)
    ) % 360.0
    v0_target = env.heuristic_v0

    return env.normalized_action_from_physical(phi=phi_blend, V0=v0_target, obs=obs)


def run_heuristic_baseline(
    *,
    episodes: int = 100,
    preset_name: str = "five_point_straight_in",
    seed: int | None = 0,
) -> dict[str, Any]:
    """Run heuristic policy episodes and return summary metrics."""

    if episodes <= 0:
        raise ValueError("episodes must be positive.")

    env = PoolToolS1Env(preset_name=preset_name)

    rewards: list[float] = []
    success_count = 0
    scratch_count = 0
    illegal_first_hit_count = 0

    for episode_idx in range(episodes):
        reset_seed = None if seed is None else int(seed) + episode_idx
        obs, _ = env.reset(seed=reset_seed)

        action = build_heuristic_action(env, obs)
        _, reward, terminated, truncated, info = env.step(action)
        assert terminated and not truncated

        rewards.append(float(reward))
        success_count += int(bool(info["success"]))
        scratch_count += int(bool(info["cue_scratch"]))
        illegal_first_hit_count += int(not bool(info["legal_first_hit"]))

    env.close()

    summary = {
        "baseline": "heuristic",
        "preset_name": preset_name,
        "episodes": episodes,
        "mean_reward": float(np.mean(rewards)),
        "success_rate": float(success_count / episodes),
        "cue_scratch_rate": float(scratch_count / episodes),
        "illegal_first_hit_rate": float(illegal_first_hit_count / episodes),
    }
    return summary


def _print_summary(summary: dict[str, Any]) -> None:
    print("=== S1 Heuristic Baseline ===")
    print(f"preset: {summary['preset_name']}")
    print(f"episodes: {summary['episodes']}")
    print(f"mean_reward: {summary['mean_reward']:.4f}")
    print(f"success_rate: {summary['success_rate']:.4f}")
    print(f"cue_scratch_rate: {summary['cue_scratch_rate']:.4f}")
    print(f"illegal_first_hit_rate: {summary['illegal_first_hit_rate']:.4f}")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run heuristic baseline on PoolToolS1Env.")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--preset-name", type=str, default="five_point_straight_in")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Optional path to save summary JSON.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    summary = run_heuristic_baseline(
        episodes=args.episodes,
        preset_name=args.preset_name,
        seed=args.seed,
    )
    _print_summary(summary)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"saved_summary_json: {output_path}")


if __name__ == "__main__":
    main()

