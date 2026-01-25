from __future__ import annotations

from pathlib import Path

from frontier_ml_stack.data.build import build_from_records
from frontier_ml_stack.data.transforms.pipeline import TransformConfig


def test_build_drops_exact_duplicates(tmp_path: Path) -> None:
    input_records = tmp_path / "records.jsonl"
    input_records.write_text(
        "\n".join(
            [
                '{"id":"1","text":"Hello   world","source":"x"}',
                '{"id":"2","text":"Hello world","source":"x"}',  # exact dup after normalization
                '{"id":"3","text":"Completely different","source":"x"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    cfg = TransformConfig(
        min_chars=1, max_chars=1000, min_quality=0.0, dedup_exact=True, dedup_near=False
    )
    out_root = tmp_path / "datasets"

    r = build_from_records(
        dataset_name="toy_clean", input_records_path=input_records, out_root=out_root, cfg=cfg
    )

    lines = r.records_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2  # one duplicate dropped


def test_build_filters_low_quality(tmp_path: Path) -> None:
    input_records = tmp_path / "records.jsonl"
    input_records.write_text(
        "\n".join(
            [
                '{"id":"1","text":"Normal sentence with variety.","source":"x"}',
                '{"id":"2","text":"hello hello hello hello hello hello","source":"x"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    cfg = TransformConfig(
        min_chars=1, max_chars=1000, min_quality=0.8, dedup_exact=False, dedup_near=False
    )
    out_root = tmp_path / "datasets"

    r = build_from_records(
        dataset_name="toy_quality", input_records_path=input_records, out_root=out_root, cfg=cfg
    )

    lines = r.records_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
