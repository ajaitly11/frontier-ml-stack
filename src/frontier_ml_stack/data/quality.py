from __future__ import annotations

import re
from dataclasses import dataclass

_WHITESPACE = re.compile(r"\s+")
_ALPHA = re.compile(r"[A-Za-z]")
_DIGIT = re.compile(r"\d")


@dataclass(frozen=True)
class QualityResult:
    score: float  # 0..1
    flags: list[str]


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def quality_score(text: str) -> QualityResult:
    """
    Lightweight, deterministic quality scoring.
    Not "truth"—just cheap heuristics to remove obvious junk.

    Heuristics:
    - too many digits
    - too little alphabetic content
    - high repetition (same token repeated)
    - very low unique token ratio
    """
    flags: list[str] = []
    t = text.strip()
    if not t:
        return QualityResult(score=0.0, flags=["empty"])

    tokens = _WHITESPACE.split(t)
    n_tokens = len(tokens)
    unique_tokens = len(set(tokens))
    unique_ratio = unique_tokens / max(1, n_tokens)

    alpha_count = len(_ALPHA.findall(t))
    digit_count = len(_DIGIT.findall(t))
    total_chars = len(t)

    alpha_ratio = alpha_count / max(1, total_chars)
    digit_ratio = digit_count / max(1, total_chars)

    # repetition: max frequency of a single token
    freqs = {}
    for tok in tokens:
        freqs[tok] = freqs.get(tok, 0) + 1
    max_freq = max(freqs.values()) if freqs else 0
    max_freq_ratio = max_freq / max(1, n_tokens)

    # Start from 1 and subtract penalties
    score = 1.0

    if alpha_ratio < 0.5:
        flags.append("low_alpha_ratio")
        score -= (0.5 - alpha_ratio) * 1.2  # up to ~0.6 penalty

    if digit_ratio > 0.2:
        flags.append("high_digit_ratio")
        score -= (digit_ratio - 0.2) * 1.0

    if unique_ratio < 0.6:
        flags.append("low_unique_ratio")
        score -= (0.6 - unique_ratio) * 1.0

    if max_freq_ratio > 0.3:
        flags.append("high_repetition")
        score -= (max_freq_ratio - 0.3) * 1.2

    return QualityResult(score=_clamp01(score), flags=flags)
