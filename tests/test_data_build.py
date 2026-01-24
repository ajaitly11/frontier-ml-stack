from __future__ import annotations

import json
from pathlib import Path

from frontier_ml_stack.data.build import build_from_records
from frontier_ml_stack.data.transforms.pipeline import TransformConfig


def test_build_from_records_writes_outputs(tmp_path: Path) -> None:
    # Use the example toy input to first ingest, then build.
    # We read directly from examples to keep test simple.
    input_records = tmp_path / "toy_canonical_records.jsonl"

    # Create a minimal canonical records file for test
    
    input_records.write_text(
        "\n".join(
            [
                '{"id":"1","text":"  Hello   world  ","source":"example"}',
                '{"id":"2","text":"\\u0000bad","source":"example"}',
                '{"id":"3","text":"x","source":"example"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    out_root = tmp_path / "datasets"
    cfg = TransformConfig(lowercase=False, min_chars=2, max_chars=100)

    result = build_from_records(
        dataset_name="toyset_clean",
        input_records_path=input_records,
        out_root=out_root,
        cfg=cfg,
    )

    assert result.records_path.exists()
    assert result.transform_log_path.exists()
    assert result.manifest_path.exists()

    # record 3 should be dropped due to min_chars=2
    lines = result.records_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["counts"]["kept"] == 2
    assert manifest["counts"]["dropped"] == 1
