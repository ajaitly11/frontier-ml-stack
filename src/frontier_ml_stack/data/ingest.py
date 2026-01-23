from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from frontier_ml_stack.data.hashing import sha256_file, sha256_text
from frontier_ml_stack.data.manifest import new_manifest
from frontier_ml_stack.data.schema import TextRecord

SCHEMA_VERSION = "v1"


@dataclass(frozen=True)
class IngestResult:
    output_dir: Path
    manifest_path: Path
    records_path: Path
    total_in: int
    total_valid: int
    total_invalid: int


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def ingest_jsonl(
    *,
    dataset_name: str,
    input_paths: list[Path],
    out_root: Path,
    source_name: str = "unknown",
    build_id: str | None = None,
) -> IngestResult:
    """
    Deterministically ingests JSONL files into a canonical JSONL format + manifest.

    Input JSONL schema accepted:
      - {"text": "..."} or {"id": "...", "text": "..."}.
    """
    input_paths = [p.resolve() for p in input_paths]
    for p in input_paths:
        if not p.exists():
            raise FileNotFoundError(p)

    # Deterministic build id: hash of input file hashes + key params.
    file_hashes = [(p.name, sha256_file(p)) for p in input_paths]
    build_fingerprint = json.dumps(
        {
            "files": file_hashes,
            "schema": SCHEMA_VERSION,
            "dataset": dataset_name,
            "source": source_name,
        },
        sort_keys=True,
    )
    computed_build_id = sha256_text(build_fingerprint)[:12]
    build_id = build_id or computed_build_id

    output_dir = out_root / dataset_name / build_id
    output_dir.mkdir(parents=True, exist_ok=True)

    records_path = output_dir / "records.jsonl"
    manifest_path = output_dir / "manifest.json"

    total_in = 0
    total_valid = 0
    total_invalid = 0

    with records_path.open("w", encoding="utf-8") as out_f:
        for p in input_paths:
            for obj in _iter_jsonl(p):
                total_in += 1
                try:
                    text = obj.get("text")
                    if not isinstance(text, str) or not text.strip():
                        raise ValueError("missing/empty text")

                    rid = obj.get("id")
                    if not isinstance(rid, str) or not rid.strip():
                        # stable id derived from content
                        rid = sha256_text(text.strip())[:16]

                    record = TextRecord(id=rid, text=text.strip(), source=source_name)
                    out_f.write(record.model_dump_json() + "\n")
                    total_valid += 1
                except Exception:
                    total_invalid += 1

    manifest = new_manifest(
        schema_version=SCHEMA_VERSION,
        dataset_name=dataset_name,
        build_id=build_id,
        input_files=[
            {"path": str(p), "sha256": h}
            for (p, (_, h)) in zip(input_paths, file_hashes, strict=False)
        ],
        params={"source_name": source_name},
        counts={"total_in": total_in, "valid": total_valid, "invalid": total_invalid},
    )
    manifest.write(manifest_path)

    return IngestResult(
        output_dir=output_dir,
        manifest_path=manifest_path,
        records_path=records_path,
        total_in=total_in,
        total_valid=total_valid,
        total_invalid=total_invalid,
    )
