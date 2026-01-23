from __future__ import annotations

import json
from pathlib import Path

from frontier_ml_stack.data.ingest import ingest_jsonl


def test_ingest_writes_records_and_manifest(tmp_path: Path) -> None:
    toy = Path("examples/data/toy.jsonl")
    out_root = tmp_path / "datasets"

    result = ingest_jsonl(
        dataset_name="toyset",
        input_paths=[toy],
        out_root=out_root,
        source_name="example",
    )

    assert result.output_dir.exists()
    assert result.records_path.exists()
    assert result.manifest_path.exists()

    # records.jsonl should contain only valid lines (3 valid: hello, custom, hello)
    lines = result.records_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3

    # manifest sanity
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["dataset_name"] == "toyset"
    assert manifest["schema_version"] == "v1"
    assert manifest["counts"]["total_in"] == 5
    assert manifest["counts"]["valid"] == 3
    assert manifest["counts"]["invalid"] == 2
