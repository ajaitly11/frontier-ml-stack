from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def write_markdown(path: Path, md: str) -> None:
    path.write_text(md, encoding="utf-8")
