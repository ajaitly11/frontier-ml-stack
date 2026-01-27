from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any


def prepare_run_dir(root: Path, run_name: str) -> Path:
    run_dir = root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def write_config(path: Path, cfg: Any) -> None:
    # works for dataclasses
    write_json(path, asdict(cfg))
