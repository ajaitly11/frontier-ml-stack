from __future__ import annotations

from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from frontier_ml_stack.training.config import SFTConfig
from frontier_ml_stack.training.data import load_records_as_dataset
from frontier_ml_stack.training.lora import (
    apply_lora,
    guess_target_modules,
    parse_target_modules,
    trainable_params_summary,
)
from frontier_ml_stack.training.run_artifacts import prepare_run_dir, write_config, write_json


def _tokenize_dataset(ds, tokenizer, max_seq_length: int):
    def _tok(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
        )

    return ds.map(_tok, batched=True, remove_columns=["text"])


def run_sft(cfg: SFTConfig) -> Path:
    records_path = Path(cfg.train_records)
    out_root = Path(cfg.output_dir)
    run_dir = prepare_run_dir(out_root, cfg.run_name)

    write_config(run_dir / "config.json", cfg)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
    lora_info = {"use_lora": cfg.use_lora}

    if cfg.use_lora:
        override = parse_target_modules(cfg.lora_target_modules)
        targets = override or guess_target_modules(model)
        lora_info["target_modules"] = targets
        lora_info["override"] = bool(override)

        model = apply_lora(
            model,
            r=cfg.lora_r,
            alpha=cfg.lora_alpha,
            dropout=cfg.lora_dropout,
            target_modules=targets,
            bias=cfg.lora_bias,
        )
        # Helpful: log trainable params
        lora_info["params"] = {
            "r": cfg.lora_r,
            "alpha": cfg.lora_alpha,
            "dropout": cfg.lora_dropout,
            "bias": cfg.lora_bias,
        }
        lora_info["trainable_summary"] = trainable_params_summary(model)
    model.train()

    ds = load_records_as_dataset(records_path)
    tokenized = _tokenize_dataset(ds, tokenizer, cfg.max_seq_length)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        max_steps=cfg.max_steps,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        learning_rate=cfg.learning_rate,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        report_to=[],
        seed=cfg.seed,
        fp16=False,
        bf16=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=collator,
    )

    train_result = trainer.train()

    metrics = train_result.metrics
    metrics["model_name"] = cfg.model_name
    metrics["torch_version"] = torch.__version__
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "lora.json", lora_info)

    # Save final model (small models only; in bigger runs you will save checkpoints)
    final_dir = run_dir / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    if cfg.use_lora:
        # Save adapter weights only
        adapter_dir = run_dir / "lora_adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(adapter_dir))

        # Optionally save merged model for downstream eval/inference
        if cfg.save_merged and hasattr(model, "merge_and_unload"):
            merged = model.merge_and_unload()
            merged_dir = run_dir / "final_model_merged"
            merged_dir.mkdir(parents=True, exist_ok=True)
            merged.save_pretrained(str(merged_dir))
            tokenizer.save_pretrained(str(merged_dir))
    else:
        # Save full fine-tuned model
        final_dir = run_dir / "final_model"
        final_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(final_dir))

    return run_dir
