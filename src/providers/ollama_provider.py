"""Ollama-backed Provider -- wraps the local embed/chat calls this project
has used since Phase 0/1, now behind the `Provider` interface.
"""
import ollama

from config.settings import EMBED_MODEL, LLM_MODEL, LLM_NUM_PREDICT, OLLAMA_KEEP_ALIVE
from src.providers.base import ChatResponse, Provider


class OllamaProvider(Provider):
    def __init__(self, model: str | None = None):
        # Defaults to the generation model (LLM_MODEL); the judge service
        # (Phase 5) passes JUDGE_LLM_MODEL instead so which model judges is
        # independently configurable from what answers questions.
        self.model = model or LLM_MODEL

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = ollama.embed(
            model=EMBED_MODEL, input=texts, keep_alive=OLLAMA_KEEP_ALIVE
        )
        return response["embeddings"]

    def chat(self, messages: list[dict], *, json_mode: bool = False) -> ChatResponse:
        resp = ollama.chat(
            model=self.model,
            messages=messages,
            keep_alive=OLLAMA_KEEP_ALIVE,
            options={"num_predict": LLM_NUM_PREDICT},
            format="json" if json_mode else None,
        )
        total_ns = resp.get("total_duration")
        return ChatResponse(
            content=resp["message"]["content"],
            prompt_tokens=resp.get("prompt_eval_count"),
            completion_tokens=resp.get("eval_count"),
            total_duration_ms=(total_ns / 1_000_000) if total_ns is not None else None,
            raw=resp,
        )
