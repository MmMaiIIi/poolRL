"""Phase-2 Gym API smoke tests."""

from __future__ import annotations

import numpy as np
import pytest

from MLAgent.envs.s1_env import PoolToolS1Env


def test_env_reset_and_step_api() -> None:
    env = PoolToolS1Env()

    obs, info = env.reset(seed=123)
    assert isinstance(obs, np.ndarray)
    assert isinstance(info, dict)
    assert obs.shape == env.observation_space.shape
    assert env.observation_space.contains(obs)

    action = env.action_space.sample()
    assert action.shape == env.action_space.shape

    next_obs, reward, terminated, truncated, step_info = env.step(action)
    assert isinstance(next_obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(step_info, dict)
    assert next_obs.shape == env.observation_space.shape
    assert env.observation_space.contains(next_obs)
    assert terminated is True
    assert truncated is False

    required_info_keys = {
        "preset_name",
        "layout_id",
        "success",
        "cue_scratch",
        "legal_first_hit",
        "phi",
        "V0",
        "obj_final_pos",
        "cue_final_pos",
        "termination_reason",
        "events_summary",
    }
    assert required_info_keys.issubset(step_info.keys())


def test_env_one_shot_termination_enforced() -> None:
    env = PoolToolS1Env()
    env.reset(seed=0)
    env.step(env.action_space.sample())

    with pytest.raises(RuntimeError):
        env.step(env.action_space.sample())

