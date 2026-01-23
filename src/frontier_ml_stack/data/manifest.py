from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _git_commit_sha() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        return out
    except Exception:
        return "unknown"


@dataclass(frozen=True)
class DatasetManifest:
    schema_version: str
    dataset_name: str
    build_id: str
    created_utc: str
    git_commit: str
    input_files: list[dict[str, Any]]
    params: dict[str, Any]
    counts: dict[str, int]

    @staticmethod
    def now_utc_iso() -> str:
        return datetime.now(UTC).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def new_manifest(
    *,
    schema_version: str,
    dataset_name: str,
    build_id: str,
    input_files: list[dict[str, Any]],
    params: dict[str, Any],
    counts: dict[str, int],
) -> DatasetManifest:
    return DatasetManifest(
        schema_version=schema_version,
        dataset_name=dataset_name,
        build_id=build_id,
        created_utc=DatasetManifest.now_utc_iso(),
        git_commit=_git_commit_sha(),
        input_files=input_files,
        params=params,
        counts=counts,
    )
