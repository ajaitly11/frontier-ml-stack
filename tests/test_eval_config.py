from __future__ import annotations

from frontier_ml_stack.eval.config import EvalConfig


def test_eval_config_constructs() -> None:
    cfg = EvalConfig(eval_name="x", model_path="m", eval_records="r")
    assert cfg.eval_name == "x"
