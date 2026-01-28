from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SFTConfig:
    run_name: str
    model_name: str
    train_records: str  # path to records.jsonl
    output_dir: str = "artifacts/runs"

    # LoRA
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: str = ""  # comma-separated override; empty => auto
    lora_bias: str = "none"  # "none" | "all" | "lora_only"
    save_merged: bool = True  # if LoRA enabled, save merged model too

    # Training knobs (CPU-friendly defaults)
    max_steps: int = 20
    per_device_train_batch_size: int = 1
    learning_rate: float = 5e-5
    max_seq_length: int = 256
    seed: int = 42

    # Save/logging
    save_steps: int = 0  # 0 => don't save checkpoints in tiny runs
    logging_steps: int = 1
