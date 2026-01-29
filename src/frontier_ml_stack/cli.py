# ruff: noqa: B008
from __future__ import annotations

from pathlib import Path

import typer
from rich import print

from frontier_ml_stack.data.build import build_from_records
from frontier_ml_stack.data.ingest import ingest_jsonl
from frontier_ml_stack.data.transforms.pipeline import TransformConfig
from frontier_ml_stack.eval.config import BehaviorEvalConfig, EvalConfig, LossEvalConfig
from frontier_ml_stack.eval.runner import run_eval
from frontier_ml_stack.training.config import SFTConfig
from frontier_ml_stack.training.sft import run_sft

app = typer.Typer(help="frontier-ml-stack CLI")

data_app = typer.Typer(help="Data pipeline commands")
app.add_typer(data_app, name="data")

training_app = typer.Typer(help="Training commands")
app.add_typer(training_app, name="training")

eval_app = typer.Typer(help="Evaluation commands")
app.add_typer(eval_app, name="eval")


@data_app.command("ingest")
def data_ingest(
    dataset_name: str = typer.Option(..., help="Logical dataset name (e.g., 'toyset')"),
    inputs: list[Path] = typer.Option(
        ..., "--input", exists=True, readable=True, help="Input JSONL file(s)"
    ),
    out_root: Path = typer.Option(Path("artifacts/datasets"), help="Output root directory"),
    source_name: str = typer.Option("unknown", help="Source label written into each record"),
) -> None:
    """
    Ingest one or more JSONL files into canonical records.jsonl + manifest.json.
    """
    result = ingest_jsonl(
        dataset_name=dataset_name,
        input_paths=inputs,
        out_root=out_root,
        source_name=source_name,
    )

    print("[bold green]Ingest complete[/bold green]")
    print(f"Output dir: {result.output_dir}")
    print(f"Records:    {result.records_path}")
    print(f"Manifest:   {result.manifest_path}")
    print(
        "Counts:     "
        f"in={result.total_in} valid={result.total_valid} invalid={result.total_invalid}"
    )


@data_app.command("build")
def data_build(
    dataset_name: str = typer.Option(..., help="Logical dataset name (e.g., 'toyset_clean')"),
    input_records: Path = typer.Option(
        ..., exists=True, readable=True, help="Path to canonical records.jsonl to transform"
    ),
    out_root: Path = typer.Option(Path("artifacts/datasets"), help="Output root directory"),
    lowercase: bool = typer.Option(False, help="Lowercase all text"),
    min_chars: int = typer.Option(1, help="Minimum character length after normalization"),
    max_chars: int = typer.Option(10_000, help="Maximum character length after normalization"),
    min_quality: float = typer.Option(
        0.0, help="Drop samples with quality score below this threshold (0..1)"
    ),
    dedup_exact: bool = typer.Option(True, help="Enable exact deduplication"),
    dedup_near: bool = typer.Option(False, help="Enable near-duplicate deduplication (SimHash)"),
    near_threshold: int = typer.Option(8, help="Max Hamming distance for near-duplicate detection"),
) -> None:
    """
    Build a transformed dataset from an existing records.jsonl.
    Writes records.jsonl + transform_log.jsonl + manifest.json.
    """
    cfg = TransformConfig(
        lowercase=lowercase,
        min_chars=min_chars,
        max_chars=max_chars,
        min_quality=min_quality,
        dedup_exact=dedup_exact,
        dedup_near=dedup_near,
        near_threshold=near_threshold,
    )
    result = build_from_records(
        dataset_name=dataset_name,
        input_records_path=input_records,
        out_root=out_root,
        cfg=cfg,
    )

    print("[bold green]Build complete[/bold green]")
    print(f"Output dir:      {result.output_dir}")
    print(f"Records:         {result.records_path}")
    print(f"Transform log:   {result.transform_log_path}")
    print(f"Manifest:        {result.manifest_path}")
    print(f"Counts:          in={result.total_in} kept={result.kept} dropped={result.dropped}")


@training_app.command("sft")
def training_sft(
    run_name: str = typer.Option(..., help="Run name (used for artifacts/runs/<run_name>)"),
    model_name: str = typer.Option("sshleifer/tiny-gpt2", help="HF model name"),
    train_records: Path = typer.Option(
        ..., exists=True, readable=True, help="Path to records.jsonl"
    ),
    max_steps: int = typer.Option(20, help="Max training steps (tiny runs on CPU)"),
    max_seq_length: int = typer.Option(256, help="Max sequence length"),
    lora: bool = typer.Option(False, help="Enable LoRA adapter training (PEFT)"),
    lora_r: int = typer.Option(8, help="LoRA rank"),
    lora_alpha: int = typer.Option(16, help="LoRA alpha"),
    lora_dropout: float = typer.Option(0.05, help="LoRA dropout"),
    lora_target_modules: str = typer.Option("", help="Comma-separated target modules override"),
    save_merged: bool = typer.Option(True, help="If LoRA, save merged full model too"),
) -> None:
    cfg = SFTConfig(
        run_name=run_name,
        model_name=model_name,
        train_records=str(train_records),
        max_steps=max_steps,
        max_seq_length=max_seq_length,
        use_lora=lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
        save_merged=save_merged,
    )
    run_dir = run_sft(cfg)
    print("[bold green]SFT complete[/bold green]")
    print(f"Run dir: {run_dir}")
    print(f"Metrics: {run_dir / 'metrics.json'}")


@eval_app.command("run")
def eval_run(
    eval_name: str = typer.Option(..., help="Eval run name (artifacts/reports/<eval_name>)"),
    model_path: str = typer.Option(..., help="HF model name or local model dir"),
    eval_records: Path = typer.Option(
        ..., exists=True, readable=True, help="Path to records.jsonl"
    ),
    max_eval_samples: int = typer.Option(64, help="Max eval samples for loss eval"),
    max_seq_length: int = typer.Option(256, help="Max sequence length for loss eval"),
    max_prompts: int = typer.Option(12, help="Max behavior prompts"),
    max_new_tokens: int = typer.Option(64, help="Max new tokens to generate"),
    temperature: float = typer.Option(0.0, help="Generation temperature; 0 for deterministic"),
) -> None:
    cfg = EvalConfig(
        eval_name=eval_name,
        model_path=model_path,
        eval_records=str(eval_records),
        loss=LossEvalConfig(max_eval_samples=max_eval_samples, max_seq_length=max_seq_length),
        behavior=BehaviorEvalConfig(
            max_prompts=max_prompts, max_new_tokens=max_new_tokens, temperature=temperature
        ),
    )
    out_dir = run_eval(cfg)
    print("[bold green]Eval complete[/bold green]")
    print(f"Report:  {out_dir / 'report.md'}")
    print(f"Metrics: {out_dir / 'metrics.json'}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
