from __future__ import annotations

from pathlib import Path

import pytest

from frontier_ml_stack.training.data import load_records_as_dataset


def test_load_records_as_dataset(tmp_path: Path) -> None:
    p = tmp_path / "records.jsonl"
    p.write_text(
        "\n".join(
            [
                '{"id":"1","text":"Hello world","source":"x"}',
                '{"id":"2","text":"Another example","source":"x"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    ds = load_records_as_dataset(p)
    assert len(ds) == 2
    assert "text" in ds.column_names


def test_load_records_as_dataset_empty_raises(tmp_path: Path) -> None:
    p = tmp_path / "records.jsonl"
    p.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        load_records_as_dataset(p)
