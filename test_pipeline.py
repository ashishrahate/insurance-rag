"""Phase 0 smoke test: prove the embed -> store -> filtered-retrieve loop
works end to end against the real California collection.

Run from the repo root:
    python test_pipeline.py
"""
import uuid

from qdrant_client.models import PointStruct

from config.settings import CA_COLLECTION
from src.ingestion.embed import embed_text
from src.retrieval.search import get_client, search

SAMPLE_CHUNK = {
    "doc_id": "CA_SMOKE_TEST",
    "state": "CA",
    "document_type": "bulletin",
    "title": "Smoke Test Bulletin",
    "date_issued": "2024-03-15",
    "date_effective": "2024-04-01T00:00:00Z",
    "source_url": "https://www.insurance.ca.gov/smoke-test",
    "parent_headers": ["Title 10", "Chapter 5", "Section 2695.18"],
    "chunk_id": "CA_SMOKE_TEST_c0",
    "content": (
        "California requires all automobile insurers to offer a "
        "good driver discount to qualifying policyholders."
    ),
}


def point_id(chunk_id: str) -> str:
    """Deterministic UUID from chunk_id so re-runs overwrite, not duplicate."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def upsert_sample() -> None:
    client = get_client()
    vector = embed_text(SAMPLE_CHUNK["content"])
    client.upsert(
        collection_name=CA_COLLECTION,
        points=[
            PointStruct(
                id=point_id(SAMPLE_CHUNK["chunk_id"]),
                vector=vector,
                payload=SAMPLE_CHUNK,
            )
        ],
    )
    print(f"Upserted 1 point into '{CA_COLLECTION}'")


def main() -> None:
    upsert_sample()

    hits = search(
        "What discounts must California auto insurers provide?",
        state="CA",
        limit=3,
    )

    print(f"\nTop {len(hits)} hits (state=CA):")
    for h in hits:
        print(f"  {h.score:.4f}  {h.payload['doc_id']}  {h.payload['title']}")

    assert hits, "no results returned"
    assert hits[0].payload["doc_id"] == "CA_SMOKE_TEST", "sample not retrieved"
    print("\nPhase 0 end-to-end loop: OK")


if __name__ == "__main__":
    main()
