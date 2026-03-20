"""Evaluation entrypoint for S1 checkpoints (headless by default)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np

# Ensure `python MLAgent/train/eval_s1.py` works from repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MLAgent.baselines.heuristic_baseline import build_heuristic_action
from MLAgent.envs.s1_env import PoolToolS1Env
from MLAgent.train.train_ppo_s1 import ActorCritic

try:
    import torch
except Exception:  # pragma: no cover - import smoke should still pass
    torch = None

PolicyName = Literal["checkpoint", "random", "heuristic"]


def _load_checkpoint_agent(
    checkpoint_path: str | Path,
    *,
    device: "torch.device",
) -> tuple[ActorCritic, dict[str, Any]]:
    if torch is None:
        raise RuntimeError(
            "Torch is required for checkpoint evaluation. Install with: "
            "`python -m pip install torch`."
        )

    ckpt = torch.load(checkpoint_path, map_location=device)
    ckpt_config = ckpt.get("config", {})
    hidden_sizes = tuple(int(v) for v in ckpt_config.get("hidden_sizes", [64, 64]))

    env = PoolToolS1Env(
        preset_name=str(ckpt_config.get("preset_name", "five_point_straight_in")),
        render_mode=None,
    )
    obs_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.shape[0])
    env.close()

    agent = ActorCritic(obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=hidden_sizes).to(
        device
    )
    agent.load_state_dict(ckpt["model_state_dict"])
    agent.eval()
    return agent, ckpt_config


def run_evaluation(
    *,
    episodes: int = 100,
    preset_name: str = "five_point_straight_in",
    policy: PolicyName = "checkpoint",
    checkpoint_path: str | Path | None = None,
    seed: int = 0,
    render_human: bool = False,
    render_episodes: int = 0,
) -> dict[str, Any]:
    """Run S1 evaluation and return summary metrics.

    - default mode (`policy=checkpoint`) loads a trained PPO checkpoint.
    - `policy=random` / `policy=heuristic` are retained as lightweight baselines.
    """

    if policy == "checkpoint" and checkpoint_path is None:
        raise ValueError("checkpoint_path is required when policy='checkpoint'.")

    if policy == "checkpoint" and torch is None:
        raise RuntimeError(
            "Torch is required for checkpoint evaluation. Install with: "
            "`python -m pip install torch`."
        )

    env = PoolToolS1Env(
        preset_name=preset_name,
        render_mode="human" if render_human else None,
    )
    env.action_space.seed(seed)

    agent = None
    if policy == "checkpoint":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        agent, ckpt_config = _load_checkpoint_agent(checkpoint_path, device=device)
        if "preset_name" in ckpt_config:
            preset_name = str(ckpt_config["preset_name"])

    rewards: list[float] = []
    success_count = 0
    scratch_count = 0
    illegal_first_hit_count = 0
    rendered_count = 0
    render_enabled = bool(render_human)

    for episode_idx in range(episodes):
        obs, _ = env.reset(seed=seed + episode_idx, options={"preset_name": preset_name})

        if policy == "random":
            action = env.action_space.sample()
        elif policy == "heuristic":
            action = build_heuristic_action(env, obs)
        elif policy == "checkpoint":
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                hidden = agent.backbone(obs_t)
                action_t = agent.actor_mean(hidden)
            action = np.clip(action_t.squeeze(0).cpu().numpy(), -1.0, 1.0)
        else:  # pragma: no cover
            raise ValueError(f"Unsupported policy: {policy!r}")

        _, reward, terminated, truncated, info = env.step(action)
        assert terminated and not truncated

        rewards.append(float(reward))
        success_count += int(bool(info["success"]))
        scratch_count += int(bool(info["cue_scratch"]))
        illegal_first_hit_count += int(not bool(info["legal_first_hit"]))

        should_render_this_episode = render_enabled and (
            render_episodes <= 0 or rendered_count < render_episodes
        )
        if should_render_this_episode:
            try:
                env.render()
                rendered_count += 1
            except RuntimeError as exc:
                print(
                    "[eval_s1] Human rendering disabled: "
                    f"{exc}",
                    file=sys.stderr,
                )
                render_enabled = False

    env.close()

    return {
        "episodes": int(episodes),
        "preset_name": preset_name,
        "policy": policy,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "success_rate": float(success_count / episodes) if episodes > 0 else 0.0,
        "cue_scratch_rate": float(scratch_count / episodes) if episodes > 0 else 0.0,
        "illegal_first_hit_rate": (
            float(illegal_first_hit_count / episodes) if episodes > 0 else 0.0
        ),
        "render_human_requested": bool(render_human),
        "rendered_episodes": int(rendered_count),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate S1 checkpoint or baseline policy.")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--preset-name", type=str, default="five_point_straight_in")
    parser.add_argument(
        "--policy",
        choices=["checkpoint", "random", "heuristic"],
        default="checkpoint",
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-json", type=str, default=None)
    parser.add_argument("--render-human", action="store_true")
    parser.add_argument("--render-episodes", type=int, default=0)
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    summary = run_evaluation(
        episodes=args.episodes,
        preset_name=args.preset_name,
        policy=args.policy,
        checkpoint_path=args.checkpoint,
        seed=args.seed,
        render_human=args.render_human,
        render_episodes=args.render_episodes,
    )

    print("=== S1 Evaluation ===")
    print(f"preset: {summary['preset_name']}")
    print(f"policy: {summary['policy']}")
    print(f"checkpoint: {summary['checkpoint_path']}")
    print(f"episodes: {summary['episodes']}")
    print(f"mean_reward: {summary['mean_reward']:.4f}")
    print(f"success_rate: {summary['success_rate']:.4f}")
    print(f"cue_scratch_rate: {summary['cue_scratch_rate']:.4f}")
    print(f"illegal_first_hit_rate: {summary['illegal_first_hit_rate']:.4f}")
    print(f"render_human_requested: {summary['render_human_requested']}")
    print(f"rendered_episodes: {summary['rendered_episodes']}")

    if args.save_json:
        output_path = Path(args.save_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"saved_summary_json: {output_path}")


if __name__ == "__main__":
    main()

