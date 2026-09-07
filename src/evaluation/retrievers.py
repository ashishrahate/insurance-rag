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

from qdrant_client.models import ScoredPoint

from src.retrieval.search import search

# A retriever returns [(doc_id, best_chunk_score), ...], length <= k, rank order.
Retriever = Callable[[str, int], list[tuple[str, float]]]

# Over-fetch chunks so that k *distinct* documents can be recovered even when
# several top chunks belong to the same bulletin (common in this corpus).
CANDIDATE_POOL = 20


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


RETRIEVERS: dict[str, Retriever] = {
    "dense": dense_retriever,
}
