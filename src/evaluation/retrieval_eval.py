"""Phase 2 retrieval evaluation harness.

Scores a named retrieval strategy (see ``retrievers.py``) against
``data/eval/ca_eval_set.json`` and appends one row to ``data/eval/results.md``,
so every Phase 2 change (header-aware chunking, BM25+RRF, cross-encoder rerank,
size/overlap sweep) leaves a visible, individually attributable delta.

Metrics, computed at k = EVAL_K over the top-k *distinct* doc_ids per question,
averaged over the in-scope questions:

  Hit@k     fraction of questions with >=1 relevant doc in the top k
  Recall@k  mean over questions of |retrieved & relevant| / |relevant|
  MRR       mean of 1 / (rank of the first relevant doc), 0 if none in top k

Out-of-scope questions (empty relevant_doc_ids) are not scored on the above.
Instead their top-1 retrieval score is summarised next to the in-scope top-1
scores -- the separation between the two is what the Phase 3 refusal threshold
is tuned on.

Usage:
    python -m src.evaluation.retrieval_eval --label "naive baseline: dense, 300/50"
    python -m src.evaluation.retrieval_eval --retriever dense --k 3 --show-questions
    python -m src.evaluation.retrieval_eval --label "..." --dry-run   # no results.md write
"""
import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

from config.settings import (
    CHUNK_OVERLAP_WORDS,
    CHUNK_SIZE_WORDS,
    EVAL_K,
    EVAL_RESULTS_FILE,
    EVAL_SET_FILE,
)
from src.evaluation.retrievers import RETRIEVERS
from src.observability.logger import log_run, new_correlation_id

OUT_OF_SCOPE = "out_of_scope"


def load_eval_set() -> dict:
    return json.loads(EVAL_SET_FILE.read_text(encoding="utf-8"))


def score_question(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> dict:
    """Hit@k / Recall@k / reciprocal-rank for one in-scope question."""
    topk = retrieved_ids[:k]
    relevant = set(relevant_ids)
    found = relevant.intersection(topk)

    rr = 0.0
    for rank, doc_id in enumerate(topk, start=1):
        if doc_id in relevant:
            rr = 1.0 / rank
            break

    return {
        "hit": 1.0 if found else 0.0,
        "recall": len(found) / len(relevant) if relevant else 0.0,
        "rr": rr,
    }


def evaluate(retriever_name: str, k: int) -> dict:
    retriever = RETRIEVERS[retriever_name]
    eval_set = load_eval_set()
    questions = eval_set["questions"]

    per_question: list[dict] = []
    for q in questions:
        results = retriever(q["question"], k)          # [(doc_id, score), ...]
        retrieved_ids = [doc_id for doc_id, _ in results]
        top_score = results[0][1] if results else None
        row = {
            "id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "relevant": q["relevant_doc_ids"],
            "retrieved": retrieved_ids,
            "top_score": top_score,
        }
        if q["category"] != OUT_OF_SCOPE:
            row.update(score_question(retrieved_ids, q["relevant_doc_ids"], k))
        per_question.append(row)

    in_scope = [r for r in per_question if r["category"] != OUT_OF_SCOPE]
    oos = [r for r in per_question if r["category"] == OUT_OF_SCOPE]

    aggregate = {
        "hit_at_k": statistics.mean(r["hit"] for r in in_scope),
        "recall_at_k": statistics.mean(r["recall"] for r in in_scope),
        "mrr": statistics.mean(r["rr"] for r in in_scope),
        "n_in_scope": len(in_scope),
        "n_out_of_scope": len(oos),
    }

    by_category: dict[str, dict] = {}
    for cat in sorted({r["category"] for r in in_scope}):
        rows = [r for r in in_scope if r["category"] == cat]
        by_category[cat] = {
            "n": len(rows),
            "hit_at_k": statistics.mean(r["hit"] for r in rows),
            "recall_at_k": statistics.mean(r["recall"] for r in rows),
            "mrr": statistics.mean(r["rr"] for r in rows),
        }

    def _score_summary(rows: list[dict]) -> dict | None:
        scores = [r["top_score"] for r in rows if r["top_score"] is not None]
        if not scores:
            return None
        return {
            "min": min(scores),
            "mean": statistics.mean(scores),
            "max": max(scores),
        }

    return {
        "retriever": retriever_name,
        "k": k,
        "aggregate": aggregate,
        "by_category": by_category,
        "in_scope_top_score": _score_summary(in_scope),
        "out_of_scope_top_score": _score_summary(oos),
        "per_question": per_question,
    }


def print_report(report: dict, show_questions: bool) -> None:
    agg = report["aggregate"]
    print(f"\nRetriever: {report['retriever']}   k={report['k']}   "
          f"({agg['n_in_scope']} in-scope, {agg['n_out_of_scope']} out-of-scope)\n")
    print(f"  Hit@{report['k']}     {agg['hit_at_k']:.3f}")
    print(f"  Recall@{report['k']}  {agg['recall_at_k']:.3f}")
    print(f"  MRR        {agg['mrr']:.3f}")

    print("\n  By category:")
    print(f"    {'category':<24} {'n':>3}  {'hit':>6} {'recall':>7} {'mrr':>6}")
    for cat, m in report["by_category"].items():
        print(f"    {cat:<24} {m['n']:>3}  {m['hit_at_k']:>6.3f} "
              f"{m['recall_at_k']:>7.3f} {m['mrr']:>6.3f}")

    ins, oos = report["in_scope_top_score"], report["out_of_scope_top_score"]
    if ins and oos:
        print("\n  Top-1 score separation (for Phase 3 threshold tuning):")
        print(f"    in-scope      min {ins['min']:.3f}  mean {ins['mean']:.3f}  max {ins['max']:.3f}")
        print(f"    out-of-scope  min {oos['min']:.3f}  mean {oos['mean']:.3f}  max {oos['max']:.3f}")

    if show_questions:
        print("\n  Per question:")
        for r in report["per_question"]:
            marker = "  " if r["category"] == OUT_OF_SCOPE else (
                "OK" if r.get("hit") else "XX")
            score = f"{r['top_score']:.3f}" if r["top_score"] is not None else "  -  "
            print(f"    {marker} {r['id']}  top={score}  "
                  f"got={r['retrieved']}  want={r['relevant'] or '(none)'}")
    print()


def append_results_row(report: dict, label: str) -> None:
    agg = report["aggregate"]
    ins = report["in_scope_top_score"] or {}
    oos = report["out_of_scope_top_score"] or {}
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")

    header = (
        "| date (UTC) | label | retriever | k | chunk | Hit@k | Recall@k | MRR "
        "| in-scope top1 (mean) | OOS top1 (mean) |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n"
    )
    row = (
        f"| {ts} | {label} | {report['retriever']} | {report['k']} "
        f"| {CHUNK_SIZE_WORDS}/{CHUNK_OVERLAP_WORDS} "
        f"| {agg['hit_at_k']:.3f} | {agg['recall_at_k']:.3f} | {agg['mrr']:.3f} "
        f"| {ins.get('mean', float('nan')):.3f} | {oos.get('mean', float('nan')):.3f} |\n"
    )

    if not EVAL_RESULTS_FILE.exists():
        EVAL_RESULTS_FILE.write_text(
            "# Phase 2 retrieval eval results\n\n"
            "One row per configuration. Baseline first; each later row is a "
            "single measured change against it.\n\n" + header + row,
            encoding="utf-8",
        )
    else:
        with open(EVAL_RESULTS_FILE, "a", encoding="utf-8") as fh:
            fh.write(row)
    print(f"appended results row to {EVAL_RESULTS_FILE}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--retriever", default="dense", choices=sorted(RETRIEVERS),
                    help="named strategy from retrievers.RETRIEVERS (default: dense)")
    ap.add_argument("--k", type=int, default=EVAL_K, help=f"score depth (default: {EVAL_K})")
    ap.add_argument("--label", default=None,
                    help="short description written to the results table")
    ap.add_argument("--show-questions", action="store_true",
                    help="print the per-question hit/miss breakdown")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the report but do not touch results.md")
    ap.add_argument("--dump-json", metavar="PATH",
                    help="write the full report (incl. per-question top_score) "
                         "to PATH, e.g. for guardrail threshold analysis (Phase 3)")
    args = ap.parse_args()

    report = evaluate(args.retriever, args.k)
    print_report(report, args.show_questions)

    if args.dump_json:
        Path(args.dump_json).write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8"
        )
        print(f"dumped full report to {args.dump_json}")

    label = args.label or f"{args.retriever} (unlabelled)"
    if not args.dry_run:
        append_results_row(report, label)

    log_run({
        "op": "eval",
        "correlation_id": new_correlation_id(),
        "status": "ok",
        "retriever": args.retriever,
        "k": args.k,
        "label": label,
        "chunk_size_words": CHUNK_SIZE_WORDS,
        "chunk_overlap_words": CHUNK_OVERLAP_WORDS,
        **report["aggregate"],
    })


if __name__ == "__main__":
    main()
