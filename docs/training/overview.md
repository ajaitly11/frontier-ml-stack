# Training subsystem

# Training subsystem

This repo supports a minimal SFT pipeline (supervised fine-tuning) using Hugging Face Trainer.

## Tiny CPU run (MacBook-friendly)

1) Build a dataset (from prior days) so you have a canonical `records.jsonl`.
2) Run SFT:

```bash
python -m frontier_ml_stack.cli training sft \
  --run-name tiny_sft_01 \
  --model-name sshleifer/tiny-gpt2 \
  --train-records artifacts/datasets/<dataset>/<build_id>/records.jsonl \
  --max-steps 20
```

Outputs

Written under artifacts/runs/<run_name>/:

 - config.json — captured training config
 - metrics.json — training metrics from HF Trainer
 - final_model/ — saved model + tokenizer

## LoRA (PEFT) run

LoRA trains small adapter weights instead of full model weights.

```bash
python -m frontier_ml_stack.cli training sft \
  --run-name tiny_lora_01 \
  --model-name sshleifer/tiny-gpt2 \
  --train-records artifacts/datasets/<dataset>/<build_id>/records.jsonl \
  --max-steps 20 \
  --lora true
```

Outputs (LoRA)

Written under artifacts/runs/<run_name>/:

-	config.json — run config
-	lora.json — LoRA settings + trainable parameter summary
-	metrics.json — training metrics
-	lora_adapter/ — adapter weights (PEFT)
-	final_model_merged/ — merged model (if enabled)