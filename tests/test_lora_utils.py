from __future__ import annotations

import torch

from frontier_ml_stack.training.lora import (
    guess_target_modules,
    parse_target_modules,
    trainable_params_summary,
)


class Dummy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = torch.nn.Linear(4, 4)
        self.k_proj = torch.nn.Linear(4, 4)
        self.v_proj = torch.nn.Linear(4, 4)
        self.other = torch.nn.Linear(4, 4)


def test_parse_target_modules() -> None:
    assert parse_target_modules("q_proj, k_proj,,") == ["q_proj", "k_proj"]


def test_guess_target_modules_finds_common_linear_names() -> None:
    m = Dummy()
    targets = guess_target_modules(m)
    assert "q_proj" in targets
    assert "k_proj" in targets
    assert "v_proj" in targets


def test_trainable_params_summary_keys() -> None:
    m = Dummy()
    s = trainable_params_summary(m)
    assert "total_params" in s
    assert "trainable_params" in s
    assert "trainable_pct" in s
