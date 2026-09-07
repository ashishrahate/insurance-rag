"""Phase 2 Change 4: chunk size / overlap sweep.

Chunk size and overlap are baked into the Qdrant collection at INDEX time
(chunk_and_index.py), unlike retriever choice which is a QUERY-time switch.
So each config in the sweep needs a real re-chunk + re-embed + re-upsert
before it can be measured -- this script just drives that loop; it adds no
new chunking or eval logic of its own.

Runs {200, 300, 500} x {0, 25, 50} on the best pipeline so far
(hybrid_rerank, per the roadmap: "on the best pipeline so far"). Each config
becomes its own row in data/eval/results.md via the existing eval harness.

At the end, the collection is re-indexed back to the CURRENT default
(CHUNK_SIZE_WORDS/CHUNK_OVERLAP_WORDS in config/settings.py) so nothing is
left mid-sweep -- picking the winner and re-indexing to it is a deliberate
follow-up step, not something this script does automatically.

Usage:
    python -m src.evaluation.chunk_sweep
    python -m src.evaluation.chunk_sweep --retriever hybrid   # override
"""
import argparse
import os
import subprocess
import sys

from config.settings import CHUNK_OVERLAP_WORDS, CHUNK_SIZE_WORDS

SIZES = [200, 300, 500]
OVERLAPS = [0, 25, 50]


def run(cmd: list[str], env: dict) -> None:
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)


def reindex_and_measure(size: int, overlap: int, retriever: str, label: str) -> None:
    env = {**os.environ, "CHUNK_SIZE_WORDS": str(size), "CHUNK_OVERLAP_WORDS": str(overlap)}
    run([sys.executable, "-m", "src.ingestion.chunk_and_index"], env)
    run(
        [sys.executable, "-m", "src.evaluation.retrieval_eval",
         "--retriever", retriever, "--label", label],
        env,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--retriever", default="hybrid_rerank",
                    help="strategy to measure each config with (default: best pipeline so far)")
    args = ap.parse_args()

    configs = [(size, overlap) for size in SIZES for overlap in OVERLAPS
               if overlap < size]

    print(f"Sweeping {len(configs)} configs on retriever={args.retriever}: {configs}\n")

    for size, overlap in configs:
        label = f"Change 4 sweep: {size}/{overlap}, {args.retriever}"
        reindex_and_measure(size, overlap, args.retriever, label)

    print(f"\nSweep done. Restoring collection to the current default "
          f"({CHUNK_SIZE_WORDS}/{CHUNK_OVERLAP_WORDS})...")
    reindex_and_measure(
        CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS, args.retriever,
        f"post-sweep restore: back to default {CHUNK_SIZE_WORDS}/{CHUNK_OVERLAP_WORDS}, {args.retriever}",
    )

    print(
        "\nAll 9 configs measured and logged to data/eval/results.md, "
        "collection restored to the pre-sweep default.\n"
        "Next: read the results table, pick the MRR winner, and if it beats "
        f"the current default ({CHUNK_SIZE_WORDS}/{CHUNK_OVERLAP_WORDS}), update "
        "CHUNK_SIZE_WORDS/CHUNK_OVERLAP_WORDS in config/settings.py and re-index once more."
    )


if __name__ == "__main__":
    main()
