from __future__ import annotations

from frontier_ml_stack.data.quality import quality_score


def test_quality_high_for_normal_sentence() -> None:
    r = quality_score("This is a normal sentence with varied tokens.")
    assert r.score > 0.7


def test_quality_low_for_repetition() -> None:
    r = quality_score("hello hello hello hello hello hello")
    assert r.score < 0.7
    assert "high_repetition" in r.flags or "low_unique_ratio" in r.flags


def test_quality_low_for_digits() -> None:
    r = quality_score("1234567890 1234567890 1234567890")
    assert r.score < 0.7
    assert "high_digit_ratio" in r.flags
