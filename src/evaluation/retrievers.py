"""Named retrieval strategies for the Phase 2 eval harness.

A retriever is a callable ``(query: str, k: int) -> list[tuple[str, float]]``
returning up to ``k`` *distinct* ``doc_id``s in rank order, each paired with the
score of its best-matching chunk. The eval harness (``retrieval_eval.py``) only
ever sees ``doc_id``s and the top-1 score, so a new strategy plugs in here
without the harness changing.

Strategies land one per Phase 2 change:
  - ``dense``          : vector-only (the naive baseline)          [Change 0]
  - ``hybrid``         : BM25 + dense, fused with RRF               [Change 2]
  - ``hybrid_rerank``  : hybrid candidates -> bge-reranker-base    [Change 3]
"""
from collections.abc import Callable
from functools import lru_cache
from typing import TYPE_CHECKING

from qdrant_client.models import ScoredPoint
from rank_bm25 import BM25Okapi

from config.settings import CA_COLLECTION, RERANK_MODEL, RRF_K
from src.retrieval.search import get_client, search

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

# A retriever returns [(doc_id, best_chunk_score), ...], length <= k, rank order.
Retriever = Callable[[str, int], list[tuple[str, float]]]

# Over-fetch chunks so that k *distinct* documents can be recovered even when
# several top chunks belong to the same bulletin (common in this corpus).
CANDIDATE_POOL = 20


@lru_cache(maxsize=1)
def _bm25_corpus() -> tuple[BM25Okapi, list[dict]]:
    """Build a BM25 index over every chunk in the CA collection, once.

    Unlike dense search, BM25 has no per-query notion of "the collection" --
    it scores a query against a fixed, pre-tokenized corpus it was built from.
    So instead of a per-query vector search, we pull every chunk's payload
    once via `scroll` (no vector needed) and index it in memory. 53 chunks is
    trivial for this; it would not scale past a few thousand without a real
    BM25 backend (Elasticsearch, Qdrant's own sparse vectors, etc).

    Returns (index, chunks) where chunks[i] is the payload that produced row i
    of the index -- BM25Okapi only returns scores by position, so this list is
    what maps a score back to a chunk_id/doc_id.
    """
    points, _ = get_client().scroll(
        collection_name=CA_COLLECTION, limit=1000, with_payload=True, with_vectors=False
    )
    chunks = [p.payload for p in points]
    tokenized = [c["content"].lower().split() for c in chunks]
    return BM25Okapi(tokenized), chunks


def _dedup_by_doc(points: list[ScoredPoint], k: int) -> list[tuple[str, float]]:
    """First k distinct doc_ids in rank order, each with its best chunk score."""
    best: dict[str, float] = {}
    for p in points:
        doc_id = p.payload["doc_id"]
        if doc_id not in best:
            best[doc_id] = p.score
            if len(best) >= k:
                break
    return list(best.items())


def dense_retriever(query: str, k: int) -> list[tuple[str, float]]:
    """Vector-only search over the CA collection, state-prefiltered. Baseline."""
    points = search(query, state="CA", limit=CANDIDATE_POOL)
    return _dedup_by_doc(points, k)


def _bm25_ranked_chunk_ids(query: str, limit: int) -> list[str]:
    """Top `limit` chunk_ids by BM25 score, best first."""
    index, chunks = _bm25_corpus()
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


def _hybrid_fused_chunks(query: str, limit: int) -> list[tuple[dict, float]]:
    """Top `limit` chunk payloads from dense+BM25, RRF-fused, best first.

    Returns full chunk payloads (not just ids) paired with their fused score,
    because the reranker (unlike the plain hybrid retriever) needs the actual
    `content` text of these chunks, not just which doc they belong to.
    """
    dense_points = search(query, state="CA", limit=limit)
    dense_chunk_ids = [p.payload["chunk_id"] for p in dense_points]
    bm25_chunk_ids = _bm25_ranked_chunk_ids(query, limit)

    fused = reciprocal_rank_fusion(dense_chunk_ids, bm25_chunk_ids)

    _, all_chunks = _bm25_corpus()  # already-cached; free after the first call
    by_id = {c["chunk_id"]: c for c in all_chunks}

    ranked_ids = sorted(fused, key=fused.get, reverse=True)[:limit]
    return [(by_id[cid], fused[cid]) for cid in ranked_ids]


def hybrid_retriever(query: str, k: int) -> list[tuple[str, float]]:
    """BM25 + dense, fused with RRF at chunk level, then deduped to k docs.

    Fusing at the chunk level (before collapsing to doc_ids) is what lets a
    document win on a chunk BM25 ranked highly even if that same document's
    best *dense* chunk ranked poorly, or vice versa -- exactly the rescue
    dense-only search misses on near-duplicate bulletins that share prose but
    differ in the specific terms/numbers BM25 is good at matching.
    """
    fused_chunks = _hybrid_fused_chunks(query, CANDIDATE_POOL)
    best_by_doc: dict[str, float] = {}
    for chunk, score in fused_chunks:
        doc_id = chunk["doc_id"]
        if doc_id not in best_by_doc:
            best_by_doc[doc_id] = score
            if len(best_by_doc) >= k:
                break
    return list(best_by_doc.items())


@lru_cache(maxsize=1)
def _reranker() -> "CrossEncoder":
    """Load bge-reranker-base once per process (a real model load, ~1-2s + the
    one-time download). Same lazy-import-inside-the-function trick as
    `sentence_transformers`/`torch` being a heavy, optional dependency only
    this retriever needs -- the eval harness's other strategies never pay for
    importing torch at all.
    """
    from sentence_transformers import CrossEncoder

    return CrossEncoder(RERANK_MODEL)


def hybrid_rerank_retriever(query: str, k: int) -> list[tuple[str, float]]:
    """Hybrid's top-20 fused chunks, reranked by a cross-encoder, then deduped.

    A cross-encoder jointly encodes (query, chunk) as one input and outputs a
    single relevance score -- much more accurate than comparing two
    independently-computed embeddings, but too expensive to run over the
    whole corpus. So it only ever reorders a small candidate pool it cannot
    expand: whatever didn't make hybrid's top 20 has no chance to be rescued
    here (this is why hybrid had to be measured and fixed first, see Change 2).
    """
    fused_chunks = [chunk for chunk, _ in _hybrid_fused_chunks(query, CANDIDATE_POOL)]
    pairs = [(query, chunk["content"]) for chunk in fused_chunks]
    scores = _reranker().predict(pairs)

    best_by_doc: dict[str, float] = {}
    ranked = sorted(zip(fused_chunks, scores), key=lambda pair: pair[1], reverse=True)
    for chunk, score in ranked:
        doc_id = chunk["doc_id"]
        if doc_id not in best_by_doc:
            best_by_doc[doc_id] = float(score)
            if len(best_by_doc) >= k:
                break
    return list(best_by_doc.items())


RETRIEVERS: dict[str, Retriever] = {
    "dense": dense_retriever,
    "hybrid": hybrid_retriever,
    "hybrid_rerank": hybrid_rerank_retriever,
}
