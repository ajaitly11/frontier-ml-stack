# Data subsystem

Goal: deterministic dataset builds from raw inputs.

Planned capabilities:
- ingestion (jsonl/parquet/HF datasets)
- cleaning + normalization
- deduplication (MinHash/SimHash)
- quality scoring
- synthetic augmentation
- dataset manifests + dataset cards

Entry points will live under `src/frontier_ml_stack/data/`.