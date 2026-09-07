"""Ollama-backed Provider -- wraps the local embed/chat calls this project
has used since Phase 0/1, now behind the `Provider` interface.
"""
import ollama

from config.settings import EMBED_MODEL, LLM_MODEL, LLM_NUM_PREDICT, OLLAMA_KEEP_ALIVE
from src.providers.base import Provider


class OllamaProvider(Provider):
    def embed(self, texts: list[str]) -> list[list[float]]:
        response = ollama.embed(
            model=EMBED_MODEL, input=texts, keep_alive=OLLAMA_KEEP_ALIVE
        )
        return response["embeddings"]

    def chat(self, messages: list[dict], *, json_mode: bool = False):
        return ollama.chat(
            model=LLM_MODEL,
            messages=messages,
            keep_alive=OLLAMA_KEEP_ALIVE,
            options={"num_predict": LLM_NUM_PREDICT},
            format="json" if json_mode else None,
        )
