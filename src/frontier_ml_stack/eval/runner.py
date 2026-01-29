from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from frontier_ml_stack.eval.config import EvalConfig
from frontier_ml_stack.eval.report import write_json, write_markdown
from frontier_ml_stack.eval.suites.behavior_eval import eval_behavior
from frontier_ml_stack.eval.suites.loss_eval import eval_loss


def run_eval(cfg: EvalConfig) -> Path:
    out_root = Path(cfg.output_dir)
    out_dir = out_root / cfg.eval_name
    out_dir.mkdir(parents=True, exist_ok=True)

    records_path = Path(cfg.eval_records)

    loss = eval_loss(
        model_path=cfg.model_path,
        records_path=records_path,
        max_eval_samples=cfg.loss.max_eval_samples,
        max_seq_length=cfg.loss.max_seq_length,
    )
    behavior = eval_behavior(
        model_path=cfg.model_path,
        max_prompts=cfg.behavior.max_prompts,
        max_new_tokens=cfg.behavior.max_new_tokens,
        temperature=cfg.behavior.temperature,
    )

    metrics = {
        "eval_name": cfg.eval_name,
        "model_path": cfg.model_path,
        "eval_records": cfg.eval_records,
        "loss": asdict(loss),
        "behavior": {
            "n_prompts": behavior.n_prompts,
            "refusal_rate": behavior.refusal_rate,
            "json_format_rate": behavior.json_format_rate,
        },
    }
    write_json(out_dir / "metrics.json", metrics)

    # Write a minimal markdown report
    md = []
    md.append(f"# Eval report: {cfg.eval_name}\n")
    md.append(f"- Model: `{cfg.model_path}`")
    md.append(f"- Records: `{cfg.eval_records}`\n")
    md.append("## Loss\n")
    md.append(f"- Avg loss: **{loss.avg_loss:.4f}**")
    md.append(f"- Perplexity: **{loss.perplexity:.2f}**")
    md.append(f"- Samples: {loss.n_samples}\n")
    md.append("## Behavioral checks\n")
    md.append(f"- Refusal rate (safety prompts): **{behavior.refusal_rate:.2f}**")
    md.append(f"- JSON format rate: **{behavior.json_format_rate:.2f}**\n")
    md.append("### Samples\n")
    for s in behavior.samples:
        flags = ", ".join(s.flags) if s.flags else "ok"
        md.append(f"**{s.id}** ({flags})")
        md.append(f"- Prompt: {s.prompt}")
        md.append(f"- Completion: {s.completion}\n")

    write_markdown(out_dir / "report.md", "\n".join(md))
    return out_dir
