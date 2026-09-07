"""Named retrieval strategies for the Phase 2 eval harness.

A retriever is a callable ``(query: str, k: int) -> list[tuple[str, float]]``
returning up to ``k`` *distinct* ``doc_id``s in rank order, each paired with the
score of its best-matching chunk. The eval harness (``retrieval_eval.py``) only
ever sees ``doc_id``s and the top-1 score, so a new strategy plugs in here
without the harness changing.

The actual dense+BM25+RRF+rerank logic lives in `src/retrieval/hybrid.py`,
shared with the production query path (`src/generation/generate.py`), which
needs chunk-level results (content intact), not the doc-deduped shape this
module produces. Everything here is a thin "dedupe to k docs" wrapper over
that shared core.

Strategies land one per Phase 2 change:
  - ``dense``          : vector-only (the naive baseline)          [Change 0]
  - ``hybrid``         : BM25 + dense, fused with RRF               [Change 2]
  - ``hybrid_rerank``  : hybrid candidates -> bge-reranker-base    [Change 3]
"""
from collections.abc import Callable

from qdrant_client.models import ScoredPoint

from src.retrieval.hybrid import CANDIDATE_POOL, hybrid_fused_chunks, reranker
from src.retrieval.search import search

# A retriever returns [(doc_id, best_chunk_score), ...], length <= k, rank order.
Retriever = Callable[[str, int], list[tuple[str, float]]]


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


def hybrid_retriever(query: str, k: int) -> list[tuple[str, float]]:
    """BM25 + dense, fused with RRF at chunk level, then deduped to k docs.

    Fusing at the chunk level (before collapsing to doc_ids) is what lets a
    document win on a chunk BM25 ranked highly even if that same document's
    best *dense* chunk ranked poorly, or vice versa -- exactly the rescue
    dense-only search misses on near-duplicate bulletins that share prose but
    differ in the specific terms/numbers BM25 is good at matching.
    """
    fused_chunks = hybrid_fused_chunks(query, CANDIDATE_POOL, state="CA")
    best_by_doc: dict[str, float] = {}
    for chunk, score in fused_chunks:
        doc_id = chunk["doc_id"]
        if doc_id not in best_by_doc:
            best_by_doc[doc_id] = score
            if len(best_by_doc) >= k:
                break
    return list(best_by_doc.items())


def hybrid_rerank_retriever(query: str, k: int) -> list[tuple[str, float]]:
    """Hybrid's top-20 fused chunks, reranked by a cross-encoder, then deduped.

    A cross-encoder jointly encodes (query, chunk) as one input and outputs a
    single relevance score -- much more accurate than comparing two
    independently-computed embeddings, but too expensive to run over the
    whole corpus. So it only ever reorders a small candidate pool it cannot
    expand: whatever didn't make hybrid's top 20 has no chance to be rescued
    here (this is why hybrid had to be measured and fixed first, see Change 2).
    """
    fused_chunks = [chunk for chunk, _ in hybrid_fused_chunks(query, CANDIDATE_POOL, state="CA")]
    pairs = [(query, chunk["content"]) for chunk in fused_chunks]
    scores = reranker().predict(pairs)

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
