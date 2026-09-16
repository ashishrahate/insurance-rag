"""Provider abstraction: swap the LLM/embedding backend behind one interface.

Phase 3 implemented only `OllamaProvider` (this project's dev backend, see
`ollama_provider.py`) and let `chat()` return Ollama's own native response
shape, deferring normalization until a second provider actually existed.
Phase 5 needs two more `chat()` callers -- the judge service and (later)
`OpenAIProvider` -- so that's now, per `chat()`'s docstring below.

`embed()` doesn't need this treatment: every caller already just wants
`list[list[float]]`, so there's nothing provider-specific leaking through.
"""
from abc import ABC, abstractmethod
from typing import TypedDict


class ChatResponse(TypedDict):
    """Common shape every Provider.chat() returns, regardless of backend.

    `raw` keeps the provider's native response alongside the normalized
    fields -- anything not yet promoted into this shape (e.g. Ollama's
    `load_duration`, used by `ollama_metrics.py`) is still reachable there
    without every caller needing to know which provider produced it.
    """
    content: str
    prompt_tokens: int | None
    completion_tokens: int | None
    total_duration_ms: float | None
    raw: object


class Provider(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input string, in order."""

    @abstractmethod
    def chat(self, messages: list[dict], *, json_mode: bool = False) -> ChatResponse:
        """Return a ChatResponse for the given messages."""
