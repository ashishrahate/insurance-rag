"""OpenRouter-backed Provider -- chat only, used for the independent
LLM-as-judge (JUDGE_LLM_PROVIDER=openrouter).

OpenRouter speaks the OpenAI chat-completions API, so this is just
OpenAIProvider pointed at a different base_url with a different key. One
key reaches many vendors' models (DeepSeek, Anthropic, Google, ...), which
makes swapping the judge model a JUDGE_LLM_MODEL change, nothing more.

embed() deliberately refuses: embeddings must match the collection's
768-dim nomic-embed-text vectors (EMBED_PROVIDER), and nothing here needs
remote embeddings.
"""
from config.settings import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_LLM_MODEL
from src.providers.openai_provider import OpenAIProvider


class OpenRouterProvider(OpenAIProvider):
    def __init__(self, model: str | None = None):
        if not OPENROUTER_API_KEY:
            raise ValueError(
                "OPENROUTER_API_KEY is not set -- required to use "
                "JUDGE_LLM_PROVIDER=openrouter (or LLM_PROVIDER=openrouter)."
            )
        super().__init__(
            model or OPENROUTER_LLM_MODEL,
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError(
            "OpenRouterProvider is chat-only; keep EMBED_PROVIDER=ollama."
        )
