"""Provider factory: `LLM_PROVIDER` in config/settings.py selects the backend.

Cached with `lru_cache` -- same "expensive, deterministic, called repeatedly"
pattern as `get_client()`/`_reranker()` elsewhere in this project.
"""
from functools import lru_cache

from config.settings import EMBED_PROVIDER, JUDGE_LLM_MODEL, JUDGE_LLM_PROVIDER, LLM_PROVIDER
from src.providers.base import Provider

_PROVIDERS = {
    "ollama": "src.providers.ollama_provider.OllamaProvider",
    "openai": "src.providers.openai_provider.OpenAIProvider",
}


def _build_provider(provider_name: str, model: str | None = None) -> Provider:
    if provider_name not in _PROVIDERS:
        raise ValueError(
            f"Unknown provider={provider_name!r}; available: {sorted(_PROVIDERS)}"
        )
    module_path, class_name = _PROVIDERS[provider_name].rsplit(".", 1)
    import importlib

    cls = getattr(importlib.import_module(module_path), class_name)
    return cls(model) if model else cls()


@lru_cache(maxsize=1)
def get_embed_provider() -> Provider:
    """The embedding provider -- EMBED_PROVIDER, used by both ingestion
    (src/ingestion/embed.py) and query-time search, which must always match."""
    return _build_provider(EMBED_PROVIDER)


@lru_cache(maxsize=1)
def get_llm_provider() -> Provider:
    """The generation provider -- LLM_PROVIDER/LLM_MODEL. Independent of
    get_embed_provider(): swapping this alone changes nothing about
    retrieval, so it needs no re-ingestion to compare."""
    return _build_provider(LLM_PROVIDER)


@lru_cache(maxsize=1)
def get_judge_provider() -> Provider:
    """The judge-service provider -- JUDGE_LLM_PROVIDER/JUDGE_LLM_MODEL,
    independent of the generation provider above (see src/judge_service/)."""
    return _build_provider(JUDGE_LLM_PROVIDER, JUDGE_LLM_MODEL)
