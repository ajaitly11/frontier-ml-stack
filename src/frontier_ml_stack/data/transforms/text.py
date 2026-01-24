from __future__ import annotations

import re

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")


def strip_control_chars(text: str) -> str:
    """
    Remove ASCII control characters that can break tokenizers/logging.
    """
    return _CONTROL_CHARS.sub("", text)


def normalize_whitespace(text: str) -> str:
    """
    Collapse repeated whitespace and trim.
    """
    return " ".join(text.split()).strip()


def maybe_lower(text: str, *, enabled: bool) -> str:
    return text.lower() if enabled else text


def apply_basic_normalization(text: str, *, lowercase: bool) -> str:
    text = strip_control_chars(text)
    text = normalize_whitespace(text)
    text = maybe_lower(text, enabled=lowercase)
    return text
