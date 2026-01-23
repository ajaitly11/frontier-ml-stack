# Data subsystem

The data subsystem produces deterministic dataset builds:

- `records.jsonl` — canonical records (`id`, `text`, `source`)
- `manifest.json` — reproducibility metadata (inputs, hashes, params, counts, git commit)

## Quickstart

From repo root:

```bash
source .venv/bin/activate
python -m frontier_ml_stack.cli data ingest \
  --dataset-name toyset \
  --input examples/data/toy.jsonl \
  --source-name example
```

Outputs are written under:

- artifacts/datasets/<dataset_name>/<build_id>/records.jsonl
- artifacts/datasets/<dataset_name>/<build_id>/manifest.json

Design notes

- Build IDs are deterministic: derived from input file hashes + key parameters.
- Invalid input rows are skipped and counted in the manifest.
- The canonical schema lives in src/frontier_ml_stack/data/schema.py.