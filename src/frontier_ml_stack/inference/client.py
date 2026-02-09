from __future__ import annotations

import httpx

from frontier_ml_stack.inference.types import GenerateRequest, GenerateResponse


def generate(base_url: str, req: GenerateRequest, timeout_s: float = 60.0) -> GenerateResponse:
    url = base_url.rstrip("/") + "/generate"
    with httpx.Client(timeout=timeout_s) as client:
        r = client.post(url, json=req.model_dump())
        r.raise_for_status()
        return GenerateResponse.model_validate(r.json())
