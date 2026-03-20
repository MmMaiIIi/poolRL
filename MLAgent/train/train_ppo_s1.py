"""CleanRL-style PPO training entrypoint for S1.

This script is tightly scoped to S1 fixed-preset training:
- preset: five_point_straight_in
- observation: existing 6D low-dimensional state
- action: existing normalized 2D action
- reward: unchanged Phase 1/2 reward
- episode: one shot
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# Ensure `python MLAgent/train/train_ppo_s1.py` works from repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception:  # pragma: no cover - import smoke should still pass without torch
    torch = None
    nn = None
    optim = None

from MLAgent.envs.s1_env import PoolToolS1Env


DEFAULT_CONFIG_PATH = REPO_ROOT / "MLAgent" / "configs" / "train_ppo_s1.yaml"


def load_train_config(path: str | Path) -> dict[str, Any]:
    """Load YAML config for S1 PPO training."""

    with Path(path).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("Training config must be a YAML mapping.")
    return config


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


if nn is not None:

    class ActorCritic(nn.Module):
        """Small MLP actor-critic with Gaussian policy for continuous actions."""

        def __init__(
            self,
            obs_dim: int,
            action_dim: int,
            hidden_sizes: tuple[int, int],
        ) -> None:
            super().__init__()

            h1, h2 = hidden_sizes
            self.backbone = nn.Sequential(
                nn.Linear(obs_dim, h1),
                nn.Tanh(),
                nn.Linear(h1, h2),
                nn.Tanh(),
            )
            self.actor_mean = nn.Linear(h2, action_dim)
            self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
            self.critic = nn.Linear(h2, 1)

            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                    nn.init.constant_(module.bias, 0.0)
            nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
            nn.init.orthogonal_(self.critic.weight, gain=1.0)

        def get_value(self, obs: torch.Tensor) -> torch.Tensor:
            return self.critic(self.backbone(obs)).squeeze(-1)

        def get_action_and_value(
            self,
            obs: torch.Tensor,
            action: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            hidden = self.backbone(obs)
            mean = self.actor_mean(hidden)
            logstd = self.actor_logstd.expand_as(mean)
            std = torch.exp(logstd)
            dist = torch.distributions.Normal(mean, std)

            if action is None:
                action = dist.sample()

            logprob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
            value = self.critic(hidden).squeeze(-1)
            return action, logprob, entropy, value

else:

    class ActorCritic:  # pragma: no cover
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("torch is required for PPO training.")


@dataclass
class RunPaths:
    run_dir: Path
    checkpoint_dir: Path
    log_dir: Path
    train_log_jsonl: Path
    eval_log_jsonl: Path
    latest_ckpt: Path
    best_ckpt: Path


def _build_run_paths(config: dict[str, Any], run_name: str | None) -> RunPaths:
    base_run_root = Path(config["run_root_dir"])
    resolved_run_name = run_name or time.strftime("%Y%m%d_%H%M%S")
    run_dir = base_run_root / f"run_{resolved_run_name}"
    checkpoint_dir = run_dir / str(config["checkpoint_dir"])
    log_dir = run_dir / str(config["log_dir"])

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    return RunPaths(
        run_dir=run_dir,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        train_log_jsonl=log_dir / "train_log.jsonl",
        eval_log_jsonl=log_dir / "eval_log.jsonl",
        latest_ckpt=checkpoint_dir / "latest.pt",
        best_ckpt=checkpoint_dir / "best.pt",
    )


def _evaluate_agent(
    agent: ActorCritic,
    *,
    device: "torch.device",
    preset_name: str,
    episodes: int,
    seed: int,
) -> dict[str, float]:
    """Headless deterministic evaluation for checkpoint selection."""

    env = PoolToolS1Env(preset_name=preset_name, render_mode=None)

    rewards: list[float] = []
    success_count = 0
    scratch_count = 0
    illegal_first_hit_count = 0

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            hidden = agent.backbone(obs_t)
            action = agent.actor_mean(hidden)
        action_np = action.squeeze(0).cpu().numpy()
        action_np = np.clip(action_np, -1.0, 1.0)

        _, reward, terminated, truncated, info = env.step(action_np)
        assert terminated and not truncated

        rewards.append(float(reward))
        success_count += int(bool(info["success"]))
        scratch_count += int(bool(info["cue_scratch"]))
        illegal_first_hit_count += int(not bool(info["legal_first_hit"]))

    env.close()

    return {
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "success_rate": float(success_count / episodes),
        "cue_scratch_rate": float(scratch_count / episodes),
        "illegal_first_hit_rate": float(illegal_first_hit_count / episodes),
    }


def _save_checkpoint(
    *,
    path: Path,
    agent: ActorCritic,
    optimizer: "optim.Optimizer",
    global_step: int,
    config: dict[str, Any],
    best_eval_success_rate: float,
) -> None:
    payload = {
        "global_step": int(global_step),
        "model_state_dict": agent.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "best_eval_success_rate": float(best_eval_success_rate),
    }
    torch.save(payload, path)


def run_ppo_training(config: dict[str, Any], *, run_name: str | None = None) -> dict[str, Any]:
    """Run minimal end-to-end PPO training on fixed S1 preset."""

    if torch is None or nn is None or optim is None:
        raise RuntimeError(
            "Torch is required for PPO training. Install with: "
            "`python -m pip install torch`."
        )

    preset_name = str(config["preset_name"])
    if preset_name != "five_point_straight_in":
        raise ValueError(
            "Phase 3 is fixed-preset only. preset_name must be 'five_point_straight_in'."
        )

    seed = int(config["seed"])
    total_timesteps = int(config["total_timesteps"])
    num_envs = int(config["num_envs"])
    num_steps = int(config["num_steps"])
    update_epochs = int(config["update_epochs"])
    num_minibatches = int(config["num_minibatches"])
    gamma = float(config["gamma"])
    gae_lambda = float(config["gae_lambda"])
    clip_coef = float(config["clip_coef"])
    vf_coef = float(config["vf_coef"])
    ent_coef = float(config["ent_coef"])
    max_grad_norm = float(config["max_grad_norm"])
    anneal_lr = bool(config["anneal_lr"])
    learning_rate = float(config["learning_rate"])
    eval_frequency = int(config["eval_frequency"])
    eval_episodes = int(config["eval_episodes"])

    batch_size = num_envs * num_steps
    if batch_size % num_minibatches != 0:
        raise ValueError("num_envs * num_steps must be divisible by num_minibatches.")
    minibatch_size = batch_size // num_minibatches

    _set_seed(seed)
    paths = _build_run_paths(config, run_name)

    config_copy_path = paths.run_dir / "config.yaml"
    config_copy_path.parent.mkdir(parents=True, exist_ok=True)
    with config_copy_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    device = torch.device("cuda" if torch.cuda.is_available() and bool(config["cuda"]) else "cpu")

    envs = [PoolToolS1Env(preset_name=preset_name, render_mode=None) for _ in range(num_envs)]
    obs_dim = int(envs[0].observation_space.shape[0])
    action_dim = int(envs[0].action_space.shape[0])
    hidden_sizes = tuple(int(v) for v in config["hidden_sizes"])

    agent = ActorCritic(obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=hidden_sizes).to(
        device
    )
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)

    next_obs = np.zeros((num_envs, obs_dim), dtype=np.float32)
    next_done = np.zeros(num_envs, dtype=np.float32)
    for i, env in enumerate(envs):
        obs, _ = env.reset(seed=seed + i)
        next_obs[i] = obs

    obs_buf = torch.zeros((num_steps, num_envs, obs_dim), dtype=torch.float32, device=device)
    actions_buf = torch.zeros(
        (num_steps, num_envs, action_dim),
        dtype=torch.float32,
        device=device,
    )
    logprobs_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
    rewards_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
    dones_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
    values_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)

    global_step = 0
    best_eval_success_rate = -1.0
    next_eval_step = eval_frequency
    num_updates = max(total_timesteps // batch_size, 1)

    for update in range(1, num_updates + 1):
        if anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            current_lr = frac * learning_rate
            optimizer.param_groups[0]["lr"] = current_lr
        else:
            current_lr = optimizer.param_groups[0]["lr"]

        rollout_success = 0
        rollout_scratch = 0
        rollout_illegal_first_hit = 0

        for step in range(num_steps):
            global_step += num_envs

            obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device)
            done_t = torch.tensor(next_done, dtype=torch.float32, device=device)

            obs_buf[step] = obs_t
            dones_buf[step] = done_t

            with torch.no_grad():
                action_t, logprob_t, _, value_t = agent.get_action_and_value(obs_t)

            action_t = torch.clamp(action_t, -1.0, 1.0)
            actions_buf[step] = action_t
            logprobs_buf[step] = logprob_t
            values_buf[step] = value_t

            action_np = action_t.cpu().numpy()
            next_obs_np = np.zeros_like(next_obs)
            next_done_np = np.zeros_like(next_done)

            for env_idx, env in enumerate(envs):
                _, reward, terminated, truncated, info = env.step(action_np[env_idx])
                done = bool(terminated or truncated)

                rewards_buf[step, env_idx] = float(reward)
                rollout_success += int(bool(info["success"]))
                rollout_scratch += int(bool(info["cue_scratch"]))
                rollout_illegal_first_hit += int(not bool(info["legal_first_hit"]))

                if done:
                    reset_obs, _ = env.reset(seed=seed + global_step + env_idx)
                    next_obs_np[env_idx] = reset_obs
                    next_done_np[env_idx] = 1.0
                else:  # one-shot env should always terminate
                    next_obs_np[env_idx] = next_obs[env_idx]
                    next_done_np[env_idx] = 0.0

            next_obs = next_obs_np
            next_done = next_done_np

        with torch.no_grad():
            next_obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device)
            next_value = agent.get_value(next_obs_t)

            advantages = torch.zeros_like(rewards_buf, device=device)
            lastgaelam = torch.zeros(num_envs, dtype=torch.float32, device=device)

            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - torch.tensor(
                        next_done,
                        dtype=torch.float32,
                        device=device,
                    )
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_buf[t + 1]
                    nextvalues = values_buf[t + 1]

                delta = rewards_buf[t] + gamma * nextvalues * nextnonterminal - values_buf[t]
                lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
                advantages[t] = lastgaelam
            returns = advantages + values_buf

        b_obs = obs_buf.reshape((-1, obs_dim))
        b_logprobs = logprobs_buf.reshape(-1)
        b_actions = actions_buf.reshape((-1, action_dim))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buf.reshape(-1)

        b_inds = np.arange(batch_size)
        clipfracs: list[float] = []
        policy_loss_v = 0.0
        value_loss_v = 0.0
        entropy_v = 0.0

        for _epoch in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean().item()
                    clipfracs.append(
                        float(((ratio - 1.0).abs() > clip_coef).float().mean().item())
                    )

                mb_adv = b_advantages[mb_inds]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds], -clip_coef, clip_coef
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

                policy_loss_v = float(pg_loss.item())
                value_loss_v = float(v_loss.item())
                entropy_v = float(entropy_loss.item())

                # keep variable alive for potential debug logging
                _ = approx_kl

        rollout_episodes = num_steps * num_envs
        train_metrics = {
            "global_step": int(global_step),
            "episodic_return": float(rewards_buf.mean().item()),
            "success_rate": float(rollout_success / rollout_episodes),
            "cue_scratch_rate": float(rollout_scratch / rollout_episodes),
            "illegal_first_hit_rate": float(rollout_illegal_first_hit / rollout_episodes),
            "policy_loss": policy_loss_v,
            "value_loss": value_loss_v,
            "entropy": entropy_v,
            "learning_rate": float(current_lr),
            "clipfrac": float(np.mean(clipfracs)) if clipfracs else 0.0,
        }
        _append_jsonl(paths.train_log_jsonl, train_metrics)

        print(
            "[train] "
            f"step={train_metrics['global_step']} "
            f"ret={train_metrics['episodic_return']:.4f} "
            f"succ={train_metrics['success_rate']:.4f} "
            f"pi_loss={train_metrics['policy_loss']:.4f} "
            f"v_loss={train_metrics['value_loss']:.4f}"
        )

        while global_step >= next_eval_step:
            eval_metrics = _evaluate_agent(
                agent,
                device=device,
                preset_name=preset_name,
                episodes=eval_episodes,
                seed=seed + next_eval_step,
            )
            eval_payload = {
                "global_step": int(global_step),
                **eval_metrics,
            }
            _append_jsonl(paths.eval_log_jsonl, eval_payload)

            _save_checkpoint(
                path=paths.latest_ckpt,
                agent=agent,
                optimizer=optimizer,
                global_step=global_step,
                config=config,
                best_eval_success_rate=best_eval_success_rate,
            )

            if eval_metrics["success_rate"] > best_eval_success_rate:
                best_eval_success_rate = eval_metrics["success_rate"]
                shutil.copy2(paths.latest_ckpt, paths.best_ckpt)

            print(
                "[eval] "
                f"step={global_step} "
                f"mean_reward={eval_metrics['mean_reward']:.4f} "
                f"success_rate={eval_metrics['success_rate']:.4f} "
                f"best_success={best_eval_success_rate:.4f}"
            )
            next_eval_step += eval_frequency

        if global_step >= total_timesteps:
            break

    _save_checkpoint(
        path=paths.latest_ckpt,
        agent=agent,
        optimizer=optimizer,
        global_step=global_step,
        config=config,
        best_eval_success_rate=best_eval_success_rate,
    )

    for env in envs:
        env.close()

    return {
        "run_dir": str(paths.run_dir),
        "latest_checkpoint": str(paths.latest_ckpt),
        "best_checkpoint": str(paths.best_ckpt),
        "train_log": str(paths.train_log_jsonl),
        "eval_log": str(paths.eval_log_jsonl),
        "global_step": int(global_step),
        "best_eval_success_rate": float(best_eval_success_rate),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train PPO on S1 fixed preset.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--run-name", type=str, default=None)

    # Optional overrides for key settings.
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--eval-frequency", type=int, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default=None)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    config = load_train_config(args.config)

    if args.total_timesteps is not None:
        config["total_timesteps"] = args.total_timesteps
    if args.seed is not None:
        config["seed"] = args.seed
    if args.learning_rate is not None:
        config["learning_rate"] = args.learning_rate
    if args.num_envs is not None:
        config["num_envs"] = args.num_envs
    if args.eval_frequency is not None:
        config["eval_frequency"] = args.eval_frequency
    if args.checkpoint_dir is not None:
        config["checkpoint_dir"] = args.checkpoint_dir
    if args.log_dir is not None:
        config["log_dir"] = args.log_dir

    summary = run_ppo_training(config, run_name=args.run_name)
    print("=== S1 PPO Training Done ===")
    print(f"run_dir: {summary['run_dir']}")
    print(f"global_step: {summary['global_step']}")
    print(f"best_eval_success_rate: {summary['best_eval_success_rate']:.4f}")
    print(f"latest_checkpoint: {summary['latest_checkpoint']}")
    print(f"best_checkpoint: {summary['best_checkpoint']}")


if __name__ == "__main__":
    main()

