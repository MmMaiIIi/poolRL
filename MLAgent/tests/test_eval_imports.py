"""Phase-3 evaluation import smoke tests."""

from __future__ import annotations

import importlib


def test_eval_module_imports() -> None:
    module = importlib.import_module("MLAgent.train.eval_s1")
    assert hasattr(module, "run_evaluation")

