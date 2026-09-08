"""Vector search against the California collection, with optional
metadata pre-filtering (e.g. restrict to one state).
"""
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, ScoredPoint

from config.settings import CA_COLLECTION, QDRANT_HOST, QDRANT_PORT
from src.ingestion.embed import embed_text
from src.observability.timing import Stopwatch, stage


@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    """Shared Qdrant client.

    Cached because constructing one opens an HTTP session and, by default,
    makes a server version-check round trip (~200 ms). That is noise beside
    CPU LLM inference but not beside GPU inference, and it multiplies across
    the per-question loop in the Phase 2 eval harness and the Phase 3 API.

    `check_compatibility=False` drops the version-check request: the server
    image is pinned in docker-compose.yml, so the check costs latency without
    telling us anything we don't already know.

    Call `get_client.cache_clear()` to force a fresh client.
    """
    return QdrantClient(
        host=QDRANT_HOST, port=QDRANT_PORT, check_compatibility=False
    )


def _build_filter(state: str | None, document_type: str | None = None) -> Filter | None:
    conditions = []
    if state is not None:
        conditions.append(FieldCondition(key="state", match=MatchValue(value=state)))
    if document_type is not None:
        conditions.append(
            FieldCondition(key="document_type", match=MatchValue(value=document_type))
        )
    return Filter(must=conditions) if conditions else None


def search(
    query: str,
    state: str | None = None,
    document_type: str | None = None,
    limit: int = 5,
    sw: Stopwatch | None = None,
) -> list[ScoredPoint]:
    """Embed `query` and run vector search, optionally restricted to one
    state and/or document_type.

    Returns Qdrant scored points (each has `.score` and `.payload`).

    `sw` is optional instrumentation: pass a Stopwatch to record the
    client / embed / qdrant stage timings. Omitting it changes nothing.
    """
    with stage(sw, "qdrant_client"):
        client = get_client()
    with stage(sw, "embed"):
        vector = embed_text(query)
    with stage(sw, "qdrant_search"):
        result = client.query_points(
            collection_name=CA_COLLECTION,
            query=vector,
            query_filter=_build_filter(state, document_type),
            limit=limit,
            with_payload=True,
        )
    return result.points
