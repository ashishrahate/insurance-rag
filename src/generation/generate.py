"""Naive RAG: retrieve chunks, then ask the local LLM to answer from them.

Instrumented for latency comparison across hardware. All measurement is
perf_counter deltas plus Ollama's own counters; the single log write happens
after the answer is assembled and cannot raise into the query path.
"""
import json

from pydantic import BaseModel, ValidationError

from config.settings import EMBED_MODEL, LLM_MODEL, REFUSAL_SCORE_CUTOFF, RETRIEVE_K
from src.generation.prompt import build_messages
from src.observability.logger import log_run, new_correlation_id
from src.observability.ollama_metrics import extract_ollama_metrics
from src.observability.timing import Stopwatch
from src.providers import get_provider
from src.retrieval.hybrid import retrieve_chunks

REFUSAL_PREFIX = "The provided bulletins do not cover"
REFUSAL_MESSAGE = f"{REFUSAL_PREFIX} this."


class LLMAnswer(BaseModel):
    """Validates the LLM's JSON reply in `json_mode` (API path only -- the
    CLI's default free-text path never constructs this)."""
    answer: str


def _parse_json_answer(content: str) -> str:
    return LLMAnswer.model_validate(json.loads(content)).answer


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
    question: str,
    state: str | None = "CA",
    k: int = RETRIEVE_K,
    json_mode: bool = False,
) -> dict:
    sw = Stopwatch()
    cid = new_correlation_id()

    with sw.stage("retrieval"):
        hits = retrieve_chunks(question, state=state, k=k)

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

    top_score = hits[0].score
    if top_score < REFUSAL_SCORE_CUTOFF:
        # Score-threshold guardrail (Phase 3): skip the LLM call entirely --
        # cheaper than the old prompt-based refusal, which still paid for a
        # full generation only to have the model decline. Cutoff derivation:
        # config/settings.py's REFUSAL_SCORE_CUTOFF comment.
        result = {
            "answer": REFUSAL_MESSAGE,
            "sources": [],
            "hits": hits,
        }
        meta = {
            "correlation_id": cid,
            "status": "refused_guardrail",
            "k": k,
            "n_hits": len(hits),
            "top_score": round(top_score, 4),
            **sw.snapshot(),
        }
        result["meta"] = meta
        log_run({"op": "query", "question": question, "state": state, **meta})
        return result

    with sw.stage("prompt_build"):
        messages = build_messages(question, hits, json_mode=json_mode)

    with sw.stage("llm"):
        resp = get_provider().chat(messages, json_mode=json_mode)

    raw_content = resp["message"]["content"].strip()
    parse_error = None
    if json_mode:
        try:
            answer = _parse_json_answer(raw_content)
        except (json.JSONDecodeError, ValidationError):
            # One retry with a corrective follow-up, per roadmap task 6.
            retry_messages = messages + [
                {"role": "assistant", "content": raw_content},
                {"role": "user", "content": (
                    'That reply was not valid JSON of the shape {"answer": "..."}.'
                    " Reply again with ONLY that JSON object, nothing else."
                )},
            ]
            with sw.stage("llm_retry"):
                resp = get_provider().chat(retry_messages, json_mode=True)
            raw_content = resp["message"]["content"].strip()
            try:
                answer = _parse_json_answer(raw_content)
            except (json.JSONDecodeError, ValidationError) as e:
                parse_error = str(e)
                answer = raw_content  # best-effort fallback, flagged via status below
    else:
        answer = raw_content

    sources = _sources(hits)

    status = "answered"
    if parse_error:
        status = "malformed_llm_output"
    elif answer.startswith(REFUSAL_PREFIX):
        status = "refused"

    meta = {
        "correlation_id": cid,
        "status": status,
        "llm_model": LLM_MODEL,
        "embed_model": EMBED_MODEL,
        "k": k,
        "n_hits": len(hits),
        "top_score": round(hits[0].score, 4),
        "answer_chars": len(answer),
        **({"parse_error": parse_error} if parse_error else {}),
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
