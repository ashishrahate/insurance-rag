"""Thin HTTP client for the judge service (src/judge_service/), used by
src/evaluation/run_full_eval.py.

Deliberately minimal: one requests.post per call, one try/except, no
retries. A down judge service or a malformed reply logs a warning and
returns None for that question -- the orchestrator records it as a
judge_error rather than aborting the whole run. See src/judge_service/main.py
for why this is an acceptable amount of resilience for a batch eval tool.
"""
import requests

from config.settings import JUDGE_SERVICE_URL
from src.observability.logger import get_logger

_log = get_logger("judge_client")
_TIMEOUT_S = 30


def _call_judge(payload: dict) -> float | None:
    try:
        resp = requests.post(f"{JUDGE_SERVICE_URL}/judge", json=payload, timeout=_TIMEOUT_S)
        resp.raise_for_status()
        return resp.json()["score"]
    except Exception as e:
        _log.warning("judge call failed (criterion=%s): %s", payload.get("criterion"), e)
        return None


def score_faithfulness(question: str, answer: str, context_chunks: list[str]) -> float | None:
    return _call_judge({
        "question": question,
        "answer": answer,
        "context": context_chunks,
        "criterion": "faithfulness",
    })


def score_answer_relevance(question: str, answer: str) -> float | None:
    return _call_judge({
        "question": question,
        "answer": answer,
        "context": [],
        "criterion": "answer_relevance",
    })
