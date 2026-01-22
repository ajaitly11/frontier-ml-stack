# Inference subsystem

Goal: scalable serving and measurable performance.

Planned capabilities:
- vLLM-based serving wrapper
- benchmarking (p50/p95, tokens/sec, concurrency sweeps)
- observability hooks
- deployment manifests (Docker/K8s)

Entry points will live under `src/frontier_ml_stack/inference/`.