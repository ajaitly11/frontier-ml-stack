from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LossEvalConfig:
    max_eval_samples: int = 64
    max_seq_length: int = 256


@dataclass(frozen=True)
class BehaviorEvalConfig:
    max_prompts: int = 12
    max_new_tokens: int = 64
    temperature: float = 0.0  # deterministic generation


@dataclass(frozen=True)
class EvalConfig:
    eval_name: str
    model_path: str  # can be HF model name or local dir
    eval_records: str  # path to records.jsonl

    output_dir: str = "artifacts/reports"
    seed: int = 42

    loss: LossEvalConfig = LossEvalConfig()
    behavior: BehaviorEvalConfig = BehaviorEvalConfig()
