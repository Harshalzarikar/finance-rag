"""Tests for versioned prompt packs."""

from __future__ import annotations

from src.core.prompt_loader import get_prompt_pack, load_prompt_pack


def test_default_pack_has_version():
    pack = load_prompt_pack("default")
    assert pack.version == "1.0.0"
    assert "Context" in pack.grounding_rules or "context" in pack.grounding_rules.lower()
    assert "research" in pack.persona
    assert "concise" in pack.response_style


def test_unknown_pack_falls_back_to_default():
    get_prompt_pack.cache_clear()
    pack = get_prompt_pack("does-not-exist")
    assert pack.pack_id == "default"
