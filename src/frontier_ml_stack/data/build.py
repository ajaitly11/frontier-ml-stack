from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from frontier_ml_stack.data.hashing import sha256_file, sha256_text
from frontier_ml_stack.data.manifest import new_manifest
from frontier_ml_stack.data.schema import TextRecord
from frontier_ml_stack.data.transforms.pipeline import TransformConfig, transform_text


@dataclass(frozen=True)
class BuildResult:
    output_dir: Path
    records_path: Path
    manifest_path: Path
    transform_log_path: Path
    total_in: int
    kept: int
    dropped: int


def _iter_records_jsonl(path: Path) -> list[TextRecord]:
    records: list[TextRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(TextRecord.model_validate_json(line))
    return records


def build_from_records(
    *,
    dataset_name: str,
    input_records_path: Path,
    out_root: Path,
    cfg: TransformConfig,
    build_id: str | None = None,
) -> BuildResult:
    input_records_path = input_records_path.resolve()
    if not input_records_path.exists():
        raise FileNotFoundError(input_records_path)

    # Deterministic build id: based on input record file hash + transform params
    input_hash = sha256_file(input_records_path)
    cfg_fingerprint = json.dumps(cfg.__dict__, sort_keys=True)
    fingerprint = json.dumps(
        {"input_records_sha256": input_hash, "cfg": cfg_fingerprint, "dataset": dataset_name},
        sort_keys=True,
    )
    computed_build_id = sha256_text(fingerprint)[:12]
    build_id = build_id or computed_build_id

    output_dir = out_root / dataset_name / build_id
    output_dir.mkdir(parents=True, exist_ok=True)

    records_path = output_dir / "records.jsonl"
    transform_log_path = output_dir / "transform_log.jsonl"
    manifest_path = output_dir / "manifest.json"

    total_in = 0
    kept = 0
    dropped = 0

    with (
        records_path.open("w", encoding="utf-8") as out_f,
        transform_log_path.open("w", encoding="utf-8") as log_f,
    ):
        for r in _iter_records_jsonl(input_records_path):
            total_in += 1
            decision = transform_text(r.text, cfg)

            log_event: dict[str, Any] = {
                "id": r.id,
                "kept": decision.kept,
                "reason": decision.reason,
            }

            if decision.kept:
                kept += 1
                out_record = TextRecord(id=r.id, text=decision.text_after or "", source=r.source)
                out_f.write(out_record.model_dump_json() + "\n")
                log_event["text_after"] = decision.text_after
            else:
                dropped += 1

            log_f.write(json.dumps(log_event, ensure_ascii=False) + "\n")

    manifest = new_manifest(
        schema_version="v1",
        dataset_name=dataset_name,
        build_id=build_id,
        input_files=[{"path": str(input_records_path), "sha256": input_hash}],
        params={"transform_config": cfg.__dict__},
        counts={"total_in": total_in, "kept": kept, "dropped": dropped},
    )
    manifest.write(manifest_path)

    return BuildResult(
        output_dir=output_dir,
        records_path=records_path,
        manifest_path=manifest_path,
        transform_log_path=transform_log_path,
        total_in=total_in,
        kept=kept,
        dropped=dropped,
    )
