"""Naive RAG: retrieve chunks, then ask the local LLM to answer from them.

Instrumented for latency comparison across hardware. All measurement is
perf_counter deltas plus Ollama's own counters; the single log write happens
after the answer is assembled and cannot raise into the query path.
"""
import ollama

from config.settings import (
    EMBED_MODEL,
    LLM_MODEL,
    LLM_NUM_PREDICT,
    OLLAMA_KEEP_ALIVE,
    RETRIEVE_K,
)
from src.generation.prompt import build_messages
from src.observability.logger import log_run, new_correlation_id
from src.observability.ollama_metrics import extract_ollama_metrics
from src.observability.timing import Stopwatch
from src.retrieval.search import search

REFUSAL_PREFIX = "The provided bulletins do not cover"


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


def answer_question(
    question: str, state: str | None = "CA", k: int = RETRIEVE_K
) -> dict:
    sw = Stopwatch()
    cid = new_correlation_id()

    hits = search(question, state=state, limit=k, sw=sw)

    if not hits:
        result = {
            "answer": "No matching passages were retrieved.",
            "sources": [],
            "hits": [],
        }
        meta = {
            "correlation_id": cid,
            "status": "no_results",
            **sw.snapshot(),
        }
        result["meta"] = meta
        log_run({"op": "query", "question": question, "state": state, "k": k, **meta})
        return result

    with sw.stage("prompt_build"):
        messages = build_messages(question, hits)

    with sw.stage("llm"):
        resp = ollama.chat(
            model=LLM_MODEL,
            messages=messages,
            keep_alive=OLLAMA_KEEP_ALIVE,
            options={"num_predict": LLM_NUM_PREDICT},
        )

    answer = resp["message"]["content"].strip()
    sources = _sources(hits)

    meta = {
        "correlation_id": cid,
        "status": "refused" if answer.startswith(REFUSAL_PREFIX) else "answered",
        "llm_model": LLM_MODEL,
        "embed_model": EMBED_MODEL,
        "k": k,
        "n_hits": len(hits),
        "top_score": round(hits[0].score, 4),
        "answer_chars": len(answer),
        **extract_ollama_metrics(resp),
        **sw.snapshot(),
    }

    result = {"answer": answer, "sources": sources, "hits": hits, "meta": meta}

    log_run(
        {
            "op": "query",
            "question": question,
            "state": state,
            "doc_ids": [s["doc_id"] for s in sources],
            **meta,
        }
    )
    return result
