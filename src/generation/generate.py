"""Naive RAG: retrieve chunks, then ask the local LLM to answer from them."""
import ollama

from config.settings import LLM_MODEL
from src.generation.prompt import build_messages
from src.retrieval.search import search


def _sources(hits) -> list[dict]:
    """One entry per source document, best score first, deduped by doc_id."""
    seen: dict[str, dict] = {}
    for h in hits:
        p = h.payload
        cur = seen.get(p["doc_id"])
        if cur is None or h.score > cur["score"]:
            seen[p["doc_id"]] = {
                "doc_id": p["doc_id"],
                "bulletin_number": p.get("bulletin_number"),
                "title": p.get("title"),
                "source_url": p.get("source_url"),
                "date_issued": p.get("date_issued"),
                "score": h.score,
            }
    return sorted(seen.values(), key=lambda s: s["score"], reverse=True)


def answer_question(question: str, state: str | None = "CA", k: int = 5) -> dict:
    hits = search(question, state=state, limit=k)
    if not hits:
        return {"answer": "No matching passages were retrieved.", "sources": [], "hits": []}

    messages = build_messages(question, hits)
    resp = ollama.chat(model=LLM_MODEL, messages=messages)
    return {
        "answer": resp["message"]["content"].strip(),
        "sources": _sources(hits),
        "hits": hits,
    }
