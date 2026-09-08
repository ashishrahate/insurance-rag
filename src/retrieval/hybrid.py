"""Shared hybrid (BM25 + dense) retrieval with cross-encoder reranking.

This is the chunk-level retrieval core used by BOTH the Phase 2 eval harness
(`src/evaluation/retrievers.py`, which further dedupes to distinct doc_ids
for Hit@K/Recall@K/MRR) and the production query path
(`src/generation/generate.py`, which needs the actual chunk text intact to
build the LLM prompt). One implementation, two thin consumers -- see
`docs/pipeline.md` for why this split exists.

`retrieve_chunks()` is the production entry point: dense + BM25 candidates,
RRF-fused, reranked by a cross-encoder, returns the top-k chunks AS-IS (no
doc-level dedup -- `generate.py::_sources()` already handles collapsing to
distinct documents for citation display, same as it does for Phase 1's
dense-only hits).
"""
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

from rank_bm25 import BM25Okapi

from config.settings import CA_COLLECTION, RERANK_MODEL, RRF_K
from src.retrieval.search import get_client, search

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

# Over-fetch chunks so k *distinct* documents can still be recovered after
# dedup (eval side) and so the reranker has a real pool to reorder (both sides).
CANDIDATE_POOL = 20


@dataclass
class ScoredChunk:
    """Duck-types against Qdrant's `ScoredPoint`: `.payload` + `.score`.

    `prompt.py::build_messages()` and `generate.py::_sources()` were written
    against real `ScoredPoint` objects from dense-only search and read only
    these two attributes -- so hybrid/reranked results can flow through them
    unchanged.
    """
    payload: dict
    score: float


@lru_cache(maxsize=1)
def bm25_corpus() -> tuple[BM25Okapi, list[dict]]:
    """Build a BM25 index over every chunk in the CA collection, once.

    Unlike dense search, BM25 has no per-query notion of "the collection" --
    it scores a query against a fixed, pre-tokenized corpus it was built from.
    So instead of a per-query vector search, we pull every chunk's payload
    once via `scroll` (no vector needed) and index it in memory. Fine at this
    corpus's size; see `docs/scaling-notes.md` #1 for what replaces this past
    a few thousand chunks.

    Returns (index, chunks) where chunks[i] is the payload that produced row i
    of the index -- BM25Okapi only returns scores by position, so this list is
    what maps a score back to a chunk_id/doc_id.

    Cached for the process lifetime -- see `docs/scaling-notes.md` #3 for why
    that's a real limitation for a long-lived server (stale after a re-index)
    and what the FastAPI `lifespan` hook needs to do about it.
    """
    points, _ = get_client().scroll(
        collection_name=CA_COLLECTION, limit=1000, with_payload=True, with_vectors=False
    )
    chunks = [p.payload for p in points]
    tokenized = [c["content"].lower().split() for c in chunks]
    return BM25Okapi(tokenized), chunks


def bm25_ranked_chunk_ids(query: str, limit: int) -> list[str]:
    """Top `limit` chunk_ids by BM25 score, best first."""
    index, chunks = bm25_corpus()
    scores = index.get_scores(query.lower().split())
    ranked = sorted(zip(chunks, scores), key=lambda pair: pair[1], reverse=True)
    return [c["chunk_id"] for c, _ in ranked[:limit]]


def reciprocal_rank_fusion(*ranked_id_lists: list[str]) -> dict[str, float]:
    """Merge any number of ranked-id lists into one fused score per id.

    RRF ignores each list's raw scores entirely -- only rank position matters:
    score(id) = sum over lists of 1 / (RRF_K + rank), rank starting at 1. This
    is what lets us combine dense cosine similarity and BM25 scores, which
    live on unrelated, uncalibrated scales, without any normalization step.
    An id near the top of *both* lists outscores one near the top of only one.
    """
    scores: dict[str, float] = {}
    for ranked_ids in ranked_id_lists:
        for rank, item_id in enumerate(ranked_ids, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (RRF_K + rank)
    return scores


def hybrid_fused_chunks(
    query: str,
    limit: int,
    state: str | None = "CA",
    document_type: str | None = None,
) -> list[tuple[dict, float]]:
    """Top `limit` chunk payloads from dense+BM25, RRF-fused, best first.

    `state`/`document_type` filter BOTH branches, not just the dense side's
    Qdrant pre-filter. BM25 (`bm25_ranked_chunk_ids`) has no per-query notion
    of a filtered collection -- it scores the WHOLE corpus every time (see
    `bm25_corpus()`) -- so without this post-fusion filter, a chunk of the
    wrong state/document_type could still win a slot via the BM25 branch even
    though the dense branch correctly excluded it. Caught while adding
    `document_type` filtering (Phase 4); pre-existing for `state` too, just
    invisible until now since this corpus is CA-only.
    """
    dense_points = search(query, state=state, document_type=document_type, limit=limit)
    dense_chunk_ids = [p.payload["chunk_id"] for p in dense_points]
    bm25_chunk_ids = bm25_ranked_chunk_ids(query, limit)

    fused = reciprocal_rank_fusion(dense_chunk_ids, bm25_chunk_ids)

    _, all_chunks = bm25_corpus()  # already-cached; free after the first call
    by_id = {c["chunk_id"]: c for c in all_chunks}

    def _matches(chunk: dict) -> bool:
        if state is not None and chunk.get("state") != state:
            return False
        if document_type is not None and chunk.get("document_type") != document_type:
            return False
        return True

    ranked_ids = [
        cid for cid in sorted(fused, key=fused.get, reverse=True)
        if cid in by_id and _matches(by_id[cid])
    ][:limit]
    return [(by_id[cid], fused[cid]) for cid in ranked_ids]


@lru_cache(maxsize=1)
def reranker() -> "CrossEncoder":
    """Load bge-reranker-base once per process (a real model load, ~1-2s + a
    one-time ~2GB download). Lazy-imported so modules that never rerank
    (e.g. the `dense`-only eval strategy) don't pay for importing torch.
    """
    from sentence_transformers import CrossEncoder

    return CrossEncoder(RERANK_MODEL)


def retrieve_chunks(
    query: str,
    state: str | None = "CA",
    k: int = 3,
    document_type: str | None = None,
) -> list[ScoredChunk]:
    """Hybrid candidates, reranked, top-k chunks -- the production entry point.

    No doc-level dedup: this mirrors Phase 1's dense-only `search()`, which
    also returns raw top-k chunks and lets `generate.py::_sources()` collapse
    to distinct documents for citation display. Multiple chunks from the same
    document in the top-k is fine -- more context for the LLM, not a bug.
    """
    fused_chunks = [
        c for c, _ in hybrid_fused_chunks(
            query, CANDIDATE_POOL, state=state, document_type=document_type
        )
    ]
    pairs = [(query, c["content"]) for c in fused_chunks]
    scores = reranker().predict(pairs)

    ranked = sorted(zip(fused_chunks, scores), key=lambda pair: pair[1], reverse=True)
    return [ScoredChunk(payload=chunk, score=float(score)) for chunk, score in ranked[:k]]
