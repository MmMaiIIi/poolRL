"""Phase-3 training import smoke tests."""

from __future__ import annotations

import importlib


def test_train_module_imports() -> None:
    module = importlib.import_module("MLAgent.train.train_ppo_s1")
    assert hasattr(module, "run_ppo_training")
    assert hasattr(module, "load_train_config")

