# frontier-ml-stack

An end-to-end ML/LLM stack with four subsystems:

- `data/` — dataset ingestion, cleaning, deduplication, quality scoring, synthetic generation
- `training/` — fine-tuning (SFT/LoRA/QLoRA), preference optimization (DPO), distillation
- `eval/` — evaluation suites: performance, robustness, safety, bias + reporting
- `inference/` — serving + benchmarking (latency/throughput) + deployment scaffolding

## Quickstart

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
pre-commit install
pytest
```

Repository layout

-	src/frontier_ml_stack/ — shared Python package (utilities used across subsystems)
-	docs/ — subsystem documentation
-	configs/ — run configurations (YAML) for pipelines
-	artifacts/ — local build outputs (ignored by git)

Documentation

Start here: docs/index.md

Subsystem overviews:

-	docs/data/overview.md
-	docs/training/overview.md
-	docs/eval/overview.md
-	docs/inference/overview.md
    