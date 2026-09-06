"""Ask a question against the California bulletin index.

Run from the repo root:
    python -m src.ask "What discounts must California auto insurers offer?"
    python -m src.ask "..." --k 8 --state CA
    python -m src.ask "..." --show-chunks
"""
import argparse

from config.settings import RETRIEVE_K
from src.generation.generate import answer_question


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("question")
    ap.add_argument("--state", default="CA", help="state filter, or 'all'")
    ap.add_argument("--k", type=int, default=RETRIEVE_K, help="chunks to retrieve")
    ap.add_argument("--show-chunks", action="store_true", help="print retrieved passages")
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


if __name__ == "__main__":
    main()
