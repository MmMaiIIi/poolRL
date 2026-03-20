"""Phase-3 config load smoke test."""

from __future__ import annotations

from pathlib import Path

from MLAgent.train.train_ppo_s1 import load_train_config


def test_train_config_load() -> None:
    config_path = Path("MLAgent/configs/train_ppo_s1.yaml")
    cfg = load_train_config(config_path)

    assert cfg["preset_name"] == "five_point_straight_in"
    assert "total_timesteps" in cfg
    assert "learning_rate" in cfg
    assert "gamma" in cfg
    assert "num_steps" in cfg
    assert "num_minibatches" in cfg
    assert "update_epochs" in cfg
    assert "hidden_sizes" in cfg
    assert "eval_frequency" in cfg
    assert "checkpoint_dir" in cfg
    assert "log_dir" in cfg

