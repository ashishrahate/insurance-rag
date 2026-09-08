"""FastAPI service: /query, /ingest, /healthcheck, /feedback (Phase 3),
/admin/feedback_stats, /admin/recent_queries (Phase 4).

Run: uvicorn src.api.main:app --reload
(requires scripts/start.sh's Qdrant + Ollama up first)

Startup-time singletons: the BM25 index and cross-encoder reranker
(`src/retrieval/hybrid.py`'s `bm25_corpus()`/`reranker()`) are warmed once in
`lifespan`, not on the first request -- otherwise the first `/query` would
eat their ~1-2s load cost (see docs/scaling-notes.md #1/#3 for why these are
process-lifetime singletons at this scale, and what replaces them at
production scale). `lifespan` also runs `feedback_db.init_db()`.

Route handlers are sync `def`, not `async def`, on purpose: the Ollama client
and Qdrant client used underneath are synchronous. FastAPI runs sync `def`
routes in its threadpool automatically, so this avoids blocking the event
loop without needing an async rewrite of embed.py/generate.py.

Phase 4's UI (`ui/app.py`) talks to this service over HTTP only -- it never
imports `generate.py`/`retrieval/` directly. See docs/pipeline.md and the
Phase 4 plan for why (single API surface, one writer for feedback/cache state).
"""
import json
from contextlib import asynccontextmanager

import ollama
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from qdrant_client.http.exceptions import ResponseHandlingException

from config.settings import RUN_LOG_FILE
from src.api.schemas import (
    Citation,
    FeedbackRequest,
    FeedbackResponse,
    FeedbackStats,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    RecentQuery,
    RetrievedChunk,
)
from src.generation.generate import answer_question
from src.ingestion.chunk_and_index import reindex
from src.observability.logger import log_run, new_correlation_id
from src.retrieval.hybrid import bm25_corpus, reranker
from src.retrieval.search import get_client
from src.storage import feedback_db

# Exact-query cache: normalized (question, state, k) -> QueryResponse. In-memory
# only, no TTL/eviction -- fine for Phase 3's single-process dev scope, cleared
# wholesale on /ingest so a re-index can't serve a stale answer. See
# docs/scaling-notes.md #3 for the Redis+TTL version of this at real scale.
_QUERY_CACHE: dict[tuple, QueryResponse] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    bm25_corpus()
    reranker()
    feedback_db.init_db()
    yield


app = FastAPI(title="insurance-rag API", lifespan=lifespan)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Last-resort structured error -- never a bare 500 traceback (task 8)."""
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "detail": str(exc)},
    )


def _cache_key(req: QueryRequest) -> tuple:
    return (req.question.strip().lower(), req.state, req.document_type, req.k)


@app.get("/healthcheck", response_model=HealthResponse)
def healthcheck():
    qdrant_ok = True
    try:
        get_client().get_collections()
    except Exception:
        qdrant_ok = False

    ollama_ok = True
    try:
        ollama.list()
    except Exception:
        ollama_ok = False

    return HealthResponse(
        status="ok" if (qdrant_ok and ollama_ok) else "degraded",
        qdrant=qdrant_ok,
        ollama=ollama_ok,
    )


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    cache_key = _cache_key(req)
    cached = _QUERY_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        result = answer_question(
            req.question,
            state=req.state,
            k=req.k,
            json_mode=True,
            document_type=req.document_type,
        )
    except (ConnectionError, ResponseHandlingException) as e:
        # ConnectionError: Ollama unreachable (its client raises this directly).
        # ResponseHandlingException: Qdrant unreachable (its own exception type,
        # NOT a builtin ConnectionError -- verified empirically, don't assume).
        raise HTTPException(status_code=503, detail=f"Qdrant/Ollama unreachable: {e}")
    except TimeoutError as e:
        raise HTTPException(status_code=504, detail=f"LLM call timed out: {e}")

    meta = result["meta"]
    refused = meta["status"] in ("refused", "refused_guardrail", "no_results")
    response = QueryResponse(
        answer=result["answer"],
        citations=[Citation(**s) for s in result["sources"]],
        retrieved_chunks=[
            RetrievedChunk(
                chunk_id=h.payload["chunk_id"],
                doc_id=h.payload["doc_id"],
                title=h.payload.get("title"),
                content=h.payload["content"],
                score=h.score,
            )
            for h in result["hits"]
        ],
        confidence=meta.get("top_score"),
        state=req.state,
        retrieval_ms=meta.get("retrieval_ms"),
        refused=refused,
        correlation_id=meta["correlation_id"],
    )

    if not refused and meta["status"] != "malformed_llm_output":
        _QUERY_CACHE[cache_key] = response
    return response


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest | None = None):
    only = req.only if req else None
    try:
        summary = reindex(only=only)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"nothing to ingest: {e}")

    _QUERY_CACHE.clear()  # a re-index can invalidate any cached answer
    return IngestResponse(
        status="ok",
        n_docs=summary["n_docs"],
        n_chunks=summary["n_chunks"],
        points_count=summary.get("points_count"),
    )


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(req: FeedbackRequest):
    cid = req.correlation_id or new_correlation_id()

    # Durable, queryable store (Phase 4) -- the actual answer, for later review.
    feedback_db.insert_feedback(
        correlation_id=cid,
        question=req.question,
        answer=req.answer,
        rating=req.rating,
        comment=req.comment,
    )
    # Existing observability trail (Phase 3) -- unchanged, kept for continuity
    # with every other op this run log already records.
    log_run({
        "op": "feedback",
        "correlation_id": cid,
        "question": req.question,
        "rating": req.rating,
        "comment": req.comment,
    })
    return FeedbackResponse(status="ok")


@app.get("/admin/feedback_stats", response_model=FeedbackStats)
def admin_feedback_stats():
    return FeedbackStats(**feedback_db.feedback_stats())


@app.get("/admin/recent_queries", response_model=list[RecentQuery])
def admin_recent_queries(limit: int = 50):
    """Newest-first tail of `logs/runs.jsonl`, `op == "query"` rows only.

    Reads the file directly rather than via a database -- `logs/runs.jsonl`
    is already the append-only source of truth for every query (Phase 1's
    observability layer); this just filters and serves it, no new storage.
    """
    if not RUN_LOG_FILE.exists():
        return []

    rows: list[RecentQuery] = []
    with open(RUN_LOG_FILE, encoding="utf-8") as fh:
        for line in fh:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("op") != "query":
                continue
            rows.append(RecentQuery(
                ts=record.get("ts"),
                correlation_id=record.get("correlation_id"),
                question=record.get("question"),
                status=record.get("status"),
                top_score=record.get("top_score"),
            ))
    return list(reversed(rows))[:limit]
