"""Provider abstraction: swap the LLM/embedding backend behind one interface.

Phase 3 implements only `OllamaProvider` (this project's dev backend, see
`ollama_provider.py`). The Phase 5 OpenAI swap becomes a new `OpenAIProvider`
class plus a one-line config change (`LLM_PROVIDER` in `config/settings.py`),
not a rewrite of `embed.py`/`generate.py` -- both already call through
`get_provider()`, never `ollama.*` directly.

`chat()` currently returns Ollama's own response shape (a dict/pydantic model
with `message.content`, `prompt_eval_count`, `eval_duration`, etc.) so
`src/observability/ollama_metrics.py`'s existing extraction keeps working
unchanged. That's a deliberate scope call for Phase 3 (see `docs/next-session.md`
/ the Phase 3 plan) -- a real second provider would need `chat()`'s return
shape generalized at that point, not before it's needed.
"""
from abc import ABC, abstractmethod


class Provider(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input string, in order."""

    @abstractmethod
    def chat(self, messages: list[dict], *, json_mode: bool = False):
        """Return the raw chat-completion response for the given messages."""
