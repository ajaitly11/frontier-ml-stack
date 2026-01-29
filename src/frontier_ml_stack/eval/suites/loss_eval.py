from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from frontier_ml_stack.training.data import load_records_as_dataset


@dataclass(frozen=True)
class LossEvalResult:
    avg_loss: float
    perplexity: float
    n_samples: int


def _tokenize(ds: Dataset, tokenizer, max_seq_length: int) -> Dataset:
    def _tok(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
        )

    tok = ds.map(_tok, batched=True, remove_columns=["text"])
    tok.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return tok


@torch.no_grad()
def eval_loss(
    *,
    model_path: str,
    records_path: Path,
    max_eval_samples: int,
    max_seq_length: int,
) -> LossEvalResult:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.eval()

    ds = load_records_as_dataset(records_path)
    if max_eval_samples > 0:
        ds = ds.select(range(min(len(ds), max_eval_samples)))

    tok = _tokenize(ds, tokenizer, max_seq_length)

    losses: list[float] = []
    for i in range(len(tok)):
        batch = tok[i]
        input_ids = batch["input_ids"].unsqueeze(0)
        attention_mask = batch["attention_mask"].unsqueeze(0)
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        losses.append(float(out.loss))

    avg_loss = sum(losses) / max(1, len(losses))
    ppl = float(math.exp(avg_loss)) if avg_loss < 20 else float("inf")
    return LossEvalResult(avg_loss=avg_loss, perplexity=ppl, n_samples=len(losses))
