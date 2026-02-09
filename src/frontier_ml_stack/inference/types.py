from __future__ import annotations

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    max_new_tokens: int = Field(64, ge=1, le=512)
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    top_p: float = Field(1.0, ge=0.0, le=1.0)


class GenerateResponse(BaseModel):
    completion: str
    model_path: str
    elapsed_ms: float
    prompt_tokens: int
    completion_tokens: int
