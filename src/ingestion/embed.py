"""Turn text into embedding vectors via the Ollama embedding model."""
import ollama

from config.settings import EMBED_MODEL


def embed_text(text: str) -> list[float]:
    """Return the embedding vector for a single string."""
    response = ollama.embed(model=EMBED_MODEL, input=text)
    return response["embeddings"][0]


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Return one embedding vector per input string, in the same order."""
    response = ollama.embed(model=EMBED_MODEL, input=texts)
    return response["embeddings"]
