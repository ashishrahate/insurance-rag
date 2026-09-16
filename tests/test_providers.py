"""Unit tests for provider selection (Phase 5): EMBED_PROVIDER, LLM_PROVIDER,
and JUDGE_LLM_PROVIDER resolve independently. No live infra needed -- these
never call embed()/chat(), just check which class gets built.
"""
import pytest

import src.providers as providers_module
from src.providers import _build_provider, get_embed_provider, get_judge_provider, get_llm_provider
from src.providers.ollama_provider import OllamaProvider


def test_build_provider_ollama():
    assert isinstance(_build_provider("ollama"), OllamaProvider)


def test_build_provider_ollama_with_model_override():
    p = _build_provider("ollama", model="some-other-model")
    assert p.model == "some-other-model"


def test_build_provider_unknown_raises():
    with pytest.raises(ValueError, match="Unknown provider"):
        _build_provider("not-a-real-provider")


def test_build_provider_openai_without_key_raises_cleanly(monkeypatch):
    monkeypatch.setattr("src.providers.openai_provider.OPENAI_API_KEY", None)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        _build_provider("openai")


def test_provider_axes_resolve_independently(monkeypatch):
    """EMBED_PROVIDER/LLM_PROVIDER/JUDGE_LLM_PROVIDER are separate settings --
    changing one must not affect what the others resolve to, and the judge
    provider must be built with JUDGE_LLM_MODEL, not LLM_MODEL."""
    monkeypatch.setattr(providers_module, "EMBED_PROVIDER", "ollama")
    monkeypatch.setattr(providers_module, "LLM_PROVIDER", "ollama")
    monkeypatch.setattr(providers_module, "JUDGE_LLM_PROVIDER", "ollama")
    monkeypatch.setattr(providers_module, "JUDGE_LLM_MODEL", "judge-model")
    get_embed_provider.cache_clear()
    get_llm_provider.cache_clear()
    get_judge_provider.cache_clear()

    try:
        embed_p, llm_p, judge_p = get_embed_provider(), get_llm_provider(), get_judge_provider()

        assert isinstance(embed_p, OllamaProvider)
        assert isinstance(llm_p, OllamaProvider)
        assert isinstance(judge_p, OllamaProvider)
        assert judge_p.model == "judge-model"
    finally:
        # Don't leak monkeypatched settings into other tests via the cache.
        get_embed_provider.cache_clear()
        get_llm_provider.cache_clear()
        get_judge_provider.cache_clear()
