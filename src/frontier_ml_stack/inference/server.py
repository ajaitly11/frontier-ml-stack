from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer

from frontier_ml_stack.inference.types import GenerateRequest, GenerateResponse


@dataclass
class LoadedModel:
    model_path: str
    tokenizer: any
    model: any


def create_app(model_path: str) -> FastAPI:
    app = FastAPI(title="Frontier ML Stack - Inference")
    state: dict[str, LoadedModel] = {}

    @app.on_event("startup")
    def _load() -> None:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(model_path, use_safetensors=False)
        model.eval()

        # CPU mode (Mac-friendly)
        state["lm"] = LoadedModel(model_path=model_path, tokenizer=tokenizer, model=model)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model_path": model_path}

    @torch.no_grad()
    @app.post("/generate", response_model=GenerateResponse)
    def generate(req: GenerateRequest) -> GenerateResponse:
        lm = state["lm"]
        tok = lm.tokenizer
        model = lm.model

        t0 = time.time()

        enc = tok(req.prompt, return_tensors="pt")
        input_ids = enc["input_ids"]
        attention_mask = enc.get("attention_mask")

        do_sample = req.temperature > 0.0
        gen = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=req.max_new_tokens,
            do_sample=do_sample,
            temperature=req.temperature if do_sample else None,
            top_p=req.top_p if do_sample else None,
            pad_token_id=tok.eos_token_id,
        )

        decoded = tok.decode(gen[0], skip_special_tokens=True)
        completion = decoded[len(req.prompt) :].strip()

        elapsed_ms = (time.time() - t0) * 1000.0
        prompt_tokens = int(input_ids.shape[1])
        completion_tokens = int(gen.shape[1] - input_ids.shape[1])

        return GenerateResponse(
            completion=completion,
            model_path=lm.model_path,
            elapsed_ms=elapsed_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    return app
