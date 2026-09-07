"""Turn text into embedding vectors, via whichever backend `LLM_PROVIDER`
selects (see `src/providers/`)."""
from src.providers import get_provider


def embed_text(text: str) -> list[float]:
    """Return the embedding vector for a single string."""
    return get_provider().embed([text])[0]


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Return one embedding vector per input string, in the same order."""
    return get_provider().embed(texts)
