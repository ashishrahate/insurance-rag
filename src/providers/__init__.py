"""Provider factory: `LLM_PROVIDER` in config/settings.py selects the backend.

Cached with `lru_cache` -- same "expensive, deterministic, called repeatedly"
pattern as `get_client()`/`_reranker()` elsewhere in this project.
"""
from functools import lru_cache

from config.settings import LLM_PROVIDER
from src.providers.base import Provider

_PROVIDERS = {
    "ollama": "src.providers.ollama_provider.OllamaProvider",
    # "openai": "src.providers.openai_provider.OpenAIProvider",  # Phase 5
}


@lru_cache(maxsize=1)
def get_provider() -> Provider:
    if LLM_PROVIDER not in _PROVIDERS:
        raise ValueError(
            f"Unknown LLM_PROVIDER={LLM_PROVIDER!r}; "
            f"available: {sorted(_PROVIDERS)}"
        )
    module_path, class_name = _PROVIDERS[LLM_PROVIDER].rsplit(".", 1)
    import importlib

    cls = getattr(importlib.import_module(module_path), class_name)
    return cls()
