# Evaluation subsystem

This repo provides a minimal evaluation harness that produces:
- `artifacts/reports/<eval_name>/metrics.json`
- `artifacts/reports/<eval_name>/report.md`

## Run evaluation

Evaluate a base HF model:

```bash
    python -m frontier_ml_stack.cli eval run \
    --eval-name eval_base_tiny \
    --model-path sshleifer/tiny-gpt2 \
    --eval-records artifacts/datasets/<folder>/<build_id>/records.jsonl
```

Evaluate a trained run (LoRA merged output):

```bash
    python -m frontier_ml_stack.cli eval run \
    --eval-name eval_lora_merged \
    --model-path artifacts/runs/tiny_lora_01/final_model_merged \
    --eval-records artifacts/datasets/<folder>/<build_id>/records.jsonl
```