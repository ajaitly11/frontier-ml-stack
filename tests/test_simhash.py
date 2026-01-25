from __future__ import annotations

from frontier_ml_stack.data.dedup.simhash import hamming_distance64, simhash64


def test_simhash_identical_text_distance_zero() -> None:
    a = simhash64("Hello world this is a test")
    b = simhash64("Hello world this is a test")
    assert hamming_distance64(a, b) == 0


def test_simhash_similar_text_small_distance() -> None:
    a = simhash64("Hello world this is a test of deduplication")
    b = simhash64("Hello world, this is a deduplication test")
    d = hamming_distance64(a, b)
    assert d < 20  # heuristic threshold; SimHash is approximate


def test_simhash_different_text_larger_distance() -> None:
    a = simhash64("Cats and dogs are common pets")
    b = simhash64("Quantum chromodynamics and gauge symmetries")
    d = hamming_distance64(a, b)
    assert d > 5
