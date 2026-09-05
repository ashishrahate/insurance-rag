"""Create the California Qdrant collection and its payload indexes.

Idempotent: safe to run repeatedly. Existing data is left untouched.
Pass --recreate to drop and rebuild the collection (destructive; dev only).

Run from the repo root:
    python -m src.ingestion.bootstrap_collection
"""
import argparse

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from config.settings import (
    CA_COLLECTION,
    EMBED_DIM,
    PAYLOAD_INDEXES,
    QDRANT_HOST,
    QDRANT_PORT,
)


def get_client() -> QdrantClient:
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def ensure_collection(client: QdrantClient, recreate: bool = False) -> None:
    exists = client.collection_exists(CA_COLLECTION)

    if exists and recreate:
        print(f"Dropping existing collection '{CA_COLLECTION}'")
        client.delete_collection(CA_COLLECTION)
        exists = False

    if exists:
        print(f"Collection '{CA_COLLECTION}' already exists - leaving it as is")
        return

    client.create_collection(
        collection_name=CA_COLLECTION,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
    )
    print(f"Created collection '{CA_COLLECTION}' ({EMBED_DIM}-dim, cosine)")


def ensure_payload_indexes(client: QdrantClient) -> None:
    for field, schema in PAYLOAD_INDEXES.items():
        client.create_payload_index(
            collection_name=CA_COLLECTION,
            field_name=field,
            field_schema=schema,
        )
        print(f"  payload index ready: {field} ({schema})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="drop the collection first (destroys all stored vectors)",
    )
    args = parser.parse_args()

    client = get_client()
    ensure_collection(client, recreate=args.recreate)
    ensure_payload_indexes(client)

    info = client.get_collection(CA_COLLECTION)
    print(
        f"\nDone. '{CA_COLLECTION}' status={info.status}, "
        f"points={info.points_count}"
    )


if __name__ == "__main__":
    main()
