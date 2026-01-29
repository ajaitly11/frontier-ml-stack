from __future__ import annotations

import torch
from peft import LoraConfig, get_peft_model


def parse_target_modules(csv: str) -> list[str]:
    items = [x.strip() for x in csv.split(",")]
    return [x for x in items if x]


def guess_target_modules(model: torch.nn.Module) -> list[str]:
    """
    Heuristic: choose common projection names that exist as Linear submodules.
    Works across Llama/Qwen/Mistral-like and GPT2-like models.
    """
    common = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "c_attn",  # GPT2
        "c_proj",  # GPT2
    ]

    present: set[str] = set()
    for name, module in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf not in common:
            continue

        # Accept Linear
        if isinstance(module, torch.nn.Linear):
            present.add(leaf)
            continue

        # Accept HF Conv1D (GPT2)
        if module.__class__.__name__ == "Conv1D":
            present.add(leaf)
            continue

    return [x for x in common if x in present]


def trainable_params_summary(model: torch.nn.Module) -> dict:
    total = 0
    trainable = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    pct = 100.0 * (trainable / total) if total else 0.0
    return {"total_params": total, "trainable_params": trainable, "trainable_pct": pct}


def apply_lora(
    model: torch.nn.Module,
    *,
    r: int,
    alpha: int,
    dropout: float,
    target_modules: list[str],
    bias: str,
) -> torch.nn.Module:
    if not target_modules:
        raise ValueError(
            "target_modules must be non-empty (auto-detection failed)."
            "Try --lora-target-modules c_attn,c_proj for GPT2-like models."
        )

    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias=bias,
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, cfg)
