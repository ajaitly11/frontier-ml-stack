from __future__ import annotations

import hashlib
import re

_TOKEN = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def simhash64(text: str) -> int:
    """
    Deterministic 64-bit SimHash for near-duplicate detection.
    """
    tokens = _tokenize(text)
    if not tokens:
        return 0

    v = [0] * 64
    for tok in tokens:
        h = hashlib.sha256(tok.encode("utf-8")).digest()
        # take first 8 bytes => 64 bits
        x = int.from_bytes(h[:8], "big", signed=False)
        for i in range(64):
            bit = (x >> i) & 1
            v[i] += 1 if bit else -1

    out = 0
    for i in range(64):
        if v[i] > 0:
            out |= 1 << i
    return out


def hamming_distance64(a: int, b: int) -> int:
    return (a ^ b).bit_count()
