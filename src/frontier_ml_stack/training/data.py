from __future__ import annotations

from pathlib import Path

from datasets import Dataset

from frontier_ml_stack.data.schema import TextRecord


def load_records_as_dataset(records_path: Path) -> Dataset:
    """
    Load canonical records.jsonl into a Hugging Face Dataset with a single 'text' column.
    """
    records: list[dict[str, str]] = []
    with records_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = TextRecord.model_validate_json(line)
            records.append({"text": r.text})

    if not records:
        raise ValueError(f"No records found in {records_path}")

    return Dataset.from_list(records)
