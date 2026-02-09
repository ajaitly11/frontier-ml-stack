from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

from frontier_ml_stack.inference.client import generate
from frontier_ml_stack.inference.types import GenerateRequest
from frontier_ml_stack.training.run_artifacts import prepare_run_dir, write_json

DEFAULT_PROMPTS = [
    "Explain overfitting in one paragraph.",
    "Write a short checklist for a job interview.",
    "Return a JSON object with keys name and age.",
    "Summarize what a transformer model is in simple terms.",
]


@dataclass(frozen=True)
class BenchResult:
    n: int
    latency_ms_mean: float
    latency_ms_p50: float
    latency_ms_p95: float
    tokens_per_sec_mean: float


def run_benchmark(
    *,
    bench_name: str,
    base_url: str,
    output_dir: str = "artifacts/benchmarks",
    prompts: list[str] | None = None,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> Path:
    out_root = Path(output_dir)
    out_dir = prepare_run_dir(out_root, bench_name)

    prompts = prompts or DEFAULT_PROMPTS
    latencies: list[float] = []
    tps: list[float] = []

    for p in prompts:
        resp = generate(
            base_url,
            GenerateRequest(
                prompt=p, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p
            ),
        )
        latencies.append(resp.elapsed_ms)
        secs = resp.elapsed_ms / 1000.0
        tok = max(1, resp.completion_tokens)
        tps.append(tok / max(1e-6, secs))

    lat_sorted = sorted(latencies)
    p50 = lat_sorted[int(0.50 * (len(lat_sorted) - 1))]
    p95 = lat_sorted[int(0.95 * (len(lat_sorted) - 1))]

    result = BenchResult(
        n=len(prompts),
        latency_ms_mean=statistics.mean(latencies),
        latency_ms_p50=p50,
        latency_ms_p95=p95,
        tokens_per_sec_mean=statistics.mean(tps),
    )
    write_json(out_dir / "results.json", asdict(result))
    return out_dir
