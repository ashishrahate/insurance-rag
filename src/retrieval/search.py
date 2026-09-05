"""Vector search against the California collection, with optional
metadata pre-filtering (e.g. restrict to one state).
"""
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, ScoredPoint

from config.settings import CA_COLLECTION, QDRANT_HOST, QDRANT_PORT
from src.ingestion.embed import embed_text


def get_client() -> QdrantClient:
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def _state_filter(state: str | None) -> Filter | None:
    if state is None:
        return None
    return Filter(
        must=[FieldCondition(key="state", match=MatchValue(value=state))]
    )


def search(
    query: str,
    state: str | None = None,
    limit: int = 5,
) -> list[ScoredPoint]:
    """Embed `query` and run vector search, optionally restricted to one state.

    Returns Qdrant scored points (each has `.score` and `.payload`).
    """
    client = get_client()
    vector = embed_text(query)
    result = client.query_points(
        collection_name=CA_COLLECTION,
        query=vector,
        query_filter=_state_filter(state),
        limit=limit,
        with_payload=True,
    )
    return result.points
