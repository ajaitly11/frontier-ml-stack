from __future__ import annotations

from frontier_ml_stack.data.transforms.pipeline import TransformConfig, transform_text


def test_transform_normalizes_whitespace() -> None:
    cfg = TransformConfig(lowercase=False, min_chars=1, max_chars=100)
    d = transform_text("  Hello   world \n\n", cfg)
    assert d.kept is True
    assert d.text_after == "Hello world"


def test_transform_strips_control_chars() -> None:
    cfg = TransformConfig(lowercase=False, min_chars=1, max_chars=100)
    d = transform_text("Hello\x00world", cfg)
    assert d.kept is True
    assert d.text_after == "Helloworld"


def test_transform_filters_short() -> None:
    cfg = TransformConfig(lowercase=False, min_chars=5, max_chars=100)
    d = transform_text("hey", cfg)
    assert d.kept is False
    assert d.reason == "too_short"


def test_transform_lowercase_toggle() -> None:
    cfg = TransformConfig(lowercase=True, min_chars=1, max_chars=100)
    d = transform_text("HeLLo", cfg)
    assert d.kept is True
    assert d.text_after == "hello"
