"""OpenAI-backed Provider (Phase 5 task 5) -- the second real implementation
behind `Provider`, proving the abstraction: `embed.py`/`generate.py` never
change, only `LLM_PROVIDER=openai` + `OPENAI_API_KEY`.

Not a like-for-like drop-in with OllamaProvider at the embedding layer --
`text-embedding-3-small` is 1536-dim, `nomic-embed-text` is 768 (see
OPENAI_EMBED_DIM in config/settings.py). Switching providers therefore needs
its own collection, not a same-collection swap.
"""
from openai import OpenAI

from config.settings import OPENAI_API_KEY, OPENAI_EMBED_MODEL, OPENAI_LLM_MODEL
from src.providers.base import ChatResponse, Provider


class OpenAIProvider(Provider):
    def __init__(self, model: str | None = None):
        if not OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is not set -- required to use LLM_PROVIDER=openai "
                "(or JUDGE_LLM_PROVIDER=openai)."
            )
        # Same constructor-override pattern as OllamaProvider: overrides the
        # *chat* model only (e.g. the judge service picking JUDGE_LLM_MODEL);
        # embed() always uses OPENAI_EMBED_MODEL, since nothing needs to
        # override the embedding model independently yet.
        self.model = model or OPENAI_LLM_MODEL
        self._client = OpenAI(api_key=OPENAI_API_KEY)

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=OPENAI_EMBED_MODEL, input=texts)
        return [item.embedding for item in response.data]

    def chat(self, messages: list[dict], *, json_mode: bool = False) -> ChatResponse:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={"type": "json_object"} if json_mode else None,
        )
        usage = response.usage
        return ChatResponse(
            content=response.choices[0].message.content,
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            # OpenAI's API doesn't report server-side generation duration the
            # way Ollama does -- ollama_metrics.py's fields stay empty for
            # this provider (it reads them from `raw`, which just won't have
            # Ollama's keys). Round-trip wall-clock is still captured by
            # generate.py's own Stopwatch, upstream of this call.
            total_duration_ms=None,
            raw=response.model_dump(),
        )
