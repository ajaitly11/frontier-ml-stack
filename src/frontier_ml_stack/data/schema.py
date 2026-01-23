from __future__ import annotations

from pydantic import BaseModel, Field


class TextRecord(BaseModel):
    """
    Canonical training record format used across the repo.
    Keep this minimal; expand later with metadata, safety labels, provenance, etc.
    """

    id: str = Field(..., description="Stable identifier for the record")
    text: str = Field(..., min_length=1, description="Primary text content")
    source: str = Field("unknown", description="Origin of the record")
