"""Random baseline for the S1 Gymnasium environment.

Usage:
    python MLAgent/baselines/random_baseline.py --episodes 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Ensure `python MLAgent/baselines/random_baseline.py` works from repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MLAgent.envs.s1_env import PoolToolS1Env


def run_random_baseline(
    *,
    episodes: int = 100,
    preset_name: str = "five_point_straight_in",
    seed: int | None = 0,
) -> dict[str, Any]:
    """Run random policy episodes and return summary metrics."""

    if episodes <= 0:
        raise ValueError("episodes must be positive.")

    env = PoolToolS1Env(preset_name=preset_name)
    if seed is not None:
        env.action_space.seed(seed)

    rewards: list[float] = []
    success_count = 0
    scratch_count = 0
    illegal_first_hit_count = 0

    for episode_idx in range(episodes):
        reset_seed = None if seed is None else int(seed) + episode_idx
        _, _ = env.reset(seed=reset_seed)

        action = env.action_space.sample()
        _, reward, terminated, truncated, info = env.step(action)
        assert terminated and not truncated

        rewards.append(float(reward))
        success_count += int(bool(info["success"]))
        scratch_count += int(bool(info["cue_scratch"]))
        illegal_first_hit_count += int(not bool(info["legal_first_hit"]))

    env.close()

    summary = {
        "baseline": "random",
        "preset_name": preset_name,
        "episodes": episodes,
        "mean_reward": float(np.mean(rewards)),
        "success_rate": float(success_count / episodes),
        "cue_scratch_rate": float(scratch_count / episodes),
        "illegal_first_hit_rate": float(illegal_first_hit_count / episodes),
    }
    return summary


def _print_summary(summary: dict[str, Any]) -> None:
    print("=== S1 Random Baseline ===")
    print(f"preset: {summary['preset_name']}")
    print(f"episodes: {summary['episodes']}")
    print(f"mean_reward: {summary['mean_reward']:.4f}")
    print(f"success_rate: {summary['success_rate']:.4f}")
    print(f"cue_scratch_rate: {summary['cue_scratch_rate']:.4f}")
    print(f"illegal_first_hit_rate: {summary['illegal_first_hit_rate']:.4f}")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run random baseline on PoolToolS1Env.")
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
    summary = run_random_baseline(
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

