from __future__ import annotations

from dataclasses import dataclass

from frontier_ml_stack.data.transforms.text import apply_basic_normalization


@dataclass(frozen=True)
class TransformConfig:
    lowercase: bool = False
    min_chars: int = 1
    max_chars: int = 10_000


@dataclass(frozen=True)
class TransformDecision:
    kept: bool
    reason: str
    text_before: str
    text_after: str | None


def transform_text(text: str, cfg: TransformConfig) -> TransformDecision:
    before = text
    after = apply_basic_normalization(before, lowercase=cfg.lowercase)

    if len(after) < cfg.min_chars:
        return TransformDecision(
            kept=False, reason="too_short", text_before=before, text_after=None
        )

    if len(after) > cfg.max_chars:
        return TransformDecision(kept=False, reason="too_long", text_before=before, text_after=None)

    if not after.strip():
        return TransformDecision(
            kept=False, reason="empty_after_norm", text_before=before, text_after=None
        )

    return TransformDecision(kept=True, reason="kept", text_before=before, text_after=after)
