"""Chunk the cleaned CA bulletins, embed each chunk, and upsert into Qdrant.

Naive fixed-size chunking: CHUNK_SIZE_WORDS sliding window with
CHUNK_OVERLAP_WORDS overlap (config.settings). No header awareness -- that is
a Phase 2 experiment.

Idempotent: for each doc, existing points with the same doc_id are deleted
before its fresh chunks are inserted, so re-running after a chunk-parameter
change leaves no stale points behind.

Run from the repo root:
    python -m src.ingestion.chunk_and_index
    python -m src.ingestion.chunk_and_index --only CA_BULLETIN_2025_7
    python -m src.ingestion.chunk_and_index --dry-run
"""
import argparse
import json
import uuid

from qdrant_client.models import (
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
)

from config.settings import (
    CA_COLLECTION,
    CHUNK_OVERLAP_WORDS,
    CHUNK_SIZE_WORDS,
    EMBED_MODEL,
    PROCESSED_CA_DIR,
)
from src.ingestion.embed import embed_batch
from src.observability.logger import log_run, new_correlation_id
from src.observability.timing import Stopwatch
from src.retrieval.search import get_client

SMOKE_TEST_DOC_ID = "CA_SMOKE_TEST"


def load_docs(only: str | None) -> list[dict]:
    manifest = json.loads(
        (PROCESSED_CA_DIR / "manifest.json").read_text(encoding="utf-8")
    )
    ids = [d["doc_id"] for d in manifest["documents"] if d["status"] == "ok"]
    if only:
        ids = [i for i in ids if i == only]
        if not ids:
            raise SystemExit(f"{only} not in processed manifest (or status != ok)")
    return [
        json.loads((PROCESSED_CA_DIR / f"{doc_id}.json").read_text(encoding="utf-8"))
        for doc_id in ids
    ]


def chunk_words(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    step = size - overlap
    chunks: list[str] = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start : start + size]))
        if start + size >= len(words):
            break
    return chunks


def build_points(doc: dict, chunks: list[str], sw: Stopwatch | None = None) -> list[PointStruct]:
    if sw is not None:
        with sw.stage("embed"):
            vectors = embed_batch(chunks)
    else:
        vectors = embed_batch(chunks)
    points = []
    for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
        chunk_id = f"{doc['doc_id']}_c{i}"
        points.append(
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id)),
                vector=vec,
                payload={
                    "doc_id": doc["doc_id"],
                    "state": doc["state"],
                    "document_type": doc["document_type"],
                    "title": doc["title"],
                    "date_issued": doc.get("date_issued"),
                    "date_effective": doc.get("date_effective"),
                    "source_url": doc["source_url"],
                    "parent_headers": [],
                    "chunk_id": chunk_id,
                    "content": chunk,
                },
            )
        )
    return points


def delete_doc_points(client, doc_id: str) -> None:
    client.delete(
        collection_name=CA_COLLECTION,
        points_selector=FilterSelector(
            filter=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            )
        ),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", help="index just this doc_id")
    ap.add_argument("--dry-run", action="store_true", help="show chunk counts, write nothing")
    args = ap.parse_args()

    sw = Stopwatch()
    docs = load_docs(args.only)
    client = get_client()

    if not args.dry_run and not args.only:
        delete_doc_points(client, SMOKE_TEST_DOC_ID)  # clear the Phase 0 leftover

    total_chunks = 0
    for doc in docs:
        chunks = chunk_words(doc["text"], CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS)
        total_chunks += len(chunks)
        n_words = len(doc["text"].split())
        print(f"  {doc['doc_id']:<26} {n_words:>4}w -> {len(chunks)} chunk(s)")
        if args.dry_run or not chunks:
            continue
        points = build_points(doc, chunks, sw=sw)
        with sw.stage("upsert"):
            delete_doc_points(client, doc["doc_id"])
            client.upsert(collection_name=CA_COLLECTION, points=points)

    tail = "(dry run)" if args.dry_run else f"upserted into '{CA_COLLECTION}'"
    print(f"\n{len(docs)} docs -> {total_chunks} chunks {tail}")
    if not args.dry_run:
        info = client.get_collection(CA_COLLECTION)
        print(f"'{CA_COLLECTION}' now holds {info.points_count} points")

        snap = sw.snapshot()
        embed_ms = snap.get("embed_ms", 0.0)
        chunks_per_s = round(total_chunks / (embed_ms / 1000), 2) if embed_ms else None
        print(f"  embed {embed_ms:,.0f}ms  upsert {snap.get('upsert_ms', 0):,.0f}ms"
              f"  ({chunks_per_s} chunks/s)")
        log_run({
            "op": "index",
            "correlation_id": new_correlation_id(),
            "status": "ok",
            "embed_model": EMBED_MODEL,
            "n_docs": len(docs),
            "n_chunks": total_chunks,
            "chunk_size_words": CHUNK_SIZE_WORDS,
            "chunk_overlap_words": CHUNK_OVERLAP_WORDS,
            "chunks_per_s": chunks_per_s,
            **snap,
        })


if __name__ == "__main__":
    main()
