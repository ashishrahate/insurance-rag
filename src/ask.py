"""Ask a question against the California bulletin index.

Run from the repo root:
    python -m src.ask "What discounts must California auto insurers offer?"
    python -m src.ask "..." --k 8 --state CA
    python -m src.ask "..." --show-chunks
"""
import argparse

from config.settings import RETRIEVE_K, RUN_ENV
from src.generation.generate import answer_question


def _timing_line(meta: dict) -> str:
    """Compact one-line latency summary."""
    if not meta:
        return ""
    total = meta.get("total_ms", 0) / 1000.0
    parts = []
    for label, key in (("embed", "embed_ms"), ("qdrant", "qdrant_search_ms"),
                       ("llm", "llm_ms")):
        v = meta.get(key)
        if isinstance(v, (int, float)):
            parts.append(f"{label} {v:,.0f}ms")
    tok = meta.get("gen_tok_s")
    if tok:
        parts.append(f"{tok:,.1f} tok/s")
    if meta.get("ollama_load_ms"):
        parts.append(f"load {meta['ollama_load_ms']:,.0f}ms")
    return (f"[{RUN_ENV}] total {total:,.1f}s  ({' | '.join(parts)})  "
            f"cid={meta.get('correlation_id', '?')}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("question")
    ap.add_argument("--state", default="CA", help="state filter, or 'all'")
    ap.add_argument("--k", type=int, default=RETRIEVE_K, help="chunks to retrieve")
    ap.add_argument("--show-chunks", action="store_true", help="print retrieved passages")
    ap.add_argument("--quiet-timing", action="store_true", help="hide the latency line")
    args = ap.parse_args()

    state = None if args.state.lower() == "all" else args.state
    result = answer_question(args.question, state=state, k=args.k)

    print("\n" + result["answer"] + "\n")

    refused = result["answer"].startswith("The provided bulletins do not cover")
    label = "Retrieved context (not used - answer was a refusal):" if refused \
        else "Sources:"
    print(label)
    for i, s in enumerate(result["sources"], 1):
        num = s["bulletin_number"] or s["doc_id"]
        date = s["date_issued"] or "n/a"
        print(f"  [{i}] Bulletin {num} ({date})  score={s['score']:.3f}")
        print(f"      {s['title']}")
        print(f"      {s['source_url']}")

    if args.show_chunks:
        print("\nRetrieved passages:")
        for i, h in enumerate(result["hits"], 1):
            print(f"\n  [{i}] {h.payload['chunk_id']}  score={h.score:.3f}")
            print("      " + h.payload["content"][:500].replace("\n", " ") + " ...")

    if not args.quiet_timing:
        print("\n" + _timing_line(result.get("meta", {})))


if __name__ == "__main__":
    main()
