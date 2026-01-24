# ruff: noqa: B008
from __future__ import annotations

from pathlib import Path

import typer
from rich import print

from frontier_ml_stack.data.build import build_from_records
from frontier_ml_stack.data.ingest import ingest_jsonl
from frontier_ml_stack.data.transforms.pipeline import TransformConfig

app = typer.Typer(help="frontier-ml-stack CLI")
data_app = typer.Typer(help="Data pipeline commands")
app.add_typer(data_app, name="data")


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
) -> None:
    """
    Build a transformed dataset from an existing records.jsonl.
    Writes records.jsonl + transform_log.jsonl + manifest.json.
    """
    cfg = TransformConfig(lowercase=lowercase, min_chars=min_chars, max_chars=max_chars)
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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
