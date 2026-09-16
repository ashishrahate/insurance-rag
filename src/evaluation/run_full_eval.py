"""Phase 5 full-pipeline evaluation harness.

Runs the eval set through the *whole* pipeline (retrieve -> generate ->
judge), not just retrieval -- retrieval_eval.py's harness stays as-is for
fast retriever-only comparisons (Phase 2). This one is the heavier,
end-to-end number: Hit@k/Recall@k/MRR (same logic, reused from
retrieval_eval.py) plus Faithfulness/Answer Relevance from the judge service
(src/judge_service/, via judge_client.py).

Requires: Qdrant + Ollama up (scripts/start.sh) AND the judge service up
(`uvicorn src.judge_service.main:app --port 8100`).

Usage:
    python -m src.evaluation.run_full_eval --label "phase 5 baseline"
    python -m src.evaluation.run_full_eval --show-questions --dry-run
"""
import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

from config.settings import (
    ANSWER_EVAL_RESULTS_FILE,
    EVAL_K,
    FAITHFULNESS_GATE,
    HIT_AT_K_GATE,
)
from src.evaluation.judge_client import score_answer_relevance, score_faithfulness
from src.evaluation.retrieval_eval import OUT_OF_SCOPE, load_eval_set, score_question
from src.generation.generate import answer_question
from src.observability.logger import log_run, new_correlation_id


def evaluate(k: int) -> dict:
    eval_set = load_eval_set()
    questions = eval_set["questions"]

    per_question: list[dict] = []
    for q in questions:
        result = answer_question(q["question"], k=k)
        answer = result["answer"]
        hits = result["hits"]
        retrieved_ids = list(dict.fromkeys(h.payload["doc_id"] for h in hits))
        context_chunks = [h.payload["content"] for h in hits]

        row = {
            "id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "relevant": q["relevant_doc_ids"],
            "retrieved": retrieved_ids,
            "answer": answer,
            "status": result["meta"]["status"],
        }
        if q["category"] != OUT_OF_SCOPE:
            row.update(score_question(retrieved_ids, q["relevant_doc_ids"], k))

        # Only judge substantive answers. A guardrail refusal makes no
        # content claims to check for faithfulness, and correctly declining
        # an out-of-scope question isn't a relevance failure -- judging
        # refusals the same way as answers pulled the aggregate down as an
        # artifact of the eval set's OOS mix, not actual answer quality
        # (caught running the first real baseline: q24-28, all refused, all
        # scored 0/0). Refusal correctness is what REFUSAL_SCORE_CUTOFF /
        # the out-of-scope top-1 score separation already measures
        # (retrieval_eval.py), not this harness's job.
        if row["status"] == "answered":
            row["faithfulness"] = score_faithfulness(q["question"], answer, context_chunks)
            row["answer_relevance"] = score_answer_relevance(q["question"], answer)
        else:
            row["faithfulness"] = None
            row["answer_relevance"] = None
        per_question.append(row)

    in_scope = [r for r in per_question if r["category"] != OUT_OF_SCOPE]
    oos = [r for r in per_question if r["category"] == OUT_OF_SCOPE]

    def _mean_scored(rows: list[dict], field: str) -> float | None:
        vals = [r[field] for r in rows if r.get(field) is not None]
        return statistics.mean(vals) if vals else None

    aggregate = {
        "hit_at_k": statistics.mean(r["hit"] for r in in_scope),
        "recall_at_k": statistics.mean(r["recall"] for r in in_scope),
        "mrr": statistics.mean(r["rr"] for r in in_scope),
        "faithfulness": _mean_scored(per_question, "faithfulness"),
        "answer_relevance": _mean_scored(per_question, "answer_relevance"),
        "n_in_scope": len(in_scope),
        "n_out_of_scope": len(oos),
        # Only "answered" rows are judged at all (see the skip note above);
        # a None score on one of those is a real judge_client failure, not
        # an intentionally-skipped refusal.
        "n_judge_errors": sum(
            1 for r in per_question
            if r["status"] == "answered"
            and (r["faithfulness"] is None or r["answer_relevance"] is None)
        ),
    }

    gates = {
        "hit_at_k_pass": aggregate["hit_at_k"] >= HIT_AT_K_GATE,
        "faithfulness_pass": (
            aggregate["faithfulness"] is not None
            and aggregate["faithfulness"] >= FAITHFULNESS_GATE
        ),
    }
    gates["passed"] = gates["hit_at_k_pass"] and gates["faithfulness_pass"]

    return {"k": k, "aggregate": aggregate, "gates": gates, "per_question": per_question}


def print_report(report: dict, show_questions: bool) -> None:
    agg, gates = report["aggregate"], report["gates"]
    print(f"\nFull-pipeline eval   k={report['k']}   "
          f"({agg['n_in_scope']} in-scope, {agg['n_out_of_scope']} out-of-scope, "
          f"{agg['n_judge_errors']} judge errors)\n")
    print(f"  Hit@{report['k']}          {agg['hit_at_k']:.3f}"
          f"   (gate {HIT_AT_K_GATE:.2f}: {'PASS' if gates['hit_at_k_pass'] else 'FAIL'})")
    print(f"  Recall@{report['k']}       {agg['recall_at_k']:.3f}")
    print(f"  MRR             {agg['mrr']:.3f}")
    faith = agg["faithfulness"]
    print(f"  Faithfulness    {faith:.3f}" if faith is not None else "  Faithfulness    n/a",
          f"  (gate {FAITHFULNESS_GATE:.2f}: {'PASS' if gates['faithfulness_pass'] else 'FAIL'})")
    rel = agg["answer_relevance"]
    print(f"  Answer Relev.   {rel:.3f}" if rel is not None else "  Answer Relev.   n/a")
    print(f"\n  Overall gate: {'PASS' if gates['passed'] else 'FAIL'}")

    if show_questions:
        print("\n  Per question:")
        for r in report["per_question"]:
            not_judged = r["status"] != "answered"

            def _fmt_score(v):
                if v is not None:
                    return f"{v:.2f}"
                return " -- " if not_judged else "err "

            f, a = _fmt_score(r["faithfulness"]), _fmt_score(r["answer_relevance"])
            print(f"    {r['id']}  status={r['status']:<20} faith={f} rel={a}  {r['question'][:60]}")
    print()


def append_results_row(report: dict, label: str) -> None:
    agg, gates = report["aggregate"], report["gates"]
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")

    def _fmt(v):
        return f"{v:.3f}" if v is not None else "n/a"

    header = (
        "| date (UTC) | label | k | Hit@k | Recall@k | MRR | Faithfulness "
        "| Answer Relevance | judge errors | gate |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n"
    )
    row = (
        f"| {ts} | {label} | {report['k']} | {agg['hit_at_k']:.3f} "
        f"| {agg['recall_at_k']:.3f} | {agg['mrr']:.3f} "
        f"| {_fmt(agg['faithfulness'])} | {_fmt(agg['answer_relevance'])} "
        f"| {agg['n_judge_errors']} | {'PASS' if gates['passed'] else 'FAIL'} |\n"
    )

    if not ANSWER_EVAL_RESULTS_FILE.exists():
        ANSWER_EVAL_RESULTS_FILE.write_text(
            "# Phase 5 full-pipeline eval results\n\n"
            "One row per run: retrieval metrics + LLM-as-judge scores, "
            "gated against config/settings.py's HIT_AT_K_GATE/FAITHFULNESS_GATE.\n\n"
            + header + row,
            encoding="utf-8",
        )
    else:
        with open(ANSWER_EVAL_RESULTS_FILE, "a", encoding="utf-8") as fh:
            fh.write(row)
    print(f"appended results row to {ANSWER_EVAL_RESULTS_FILE}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=EVAL_K, help=f"score depth (default: {EVAL_K})")
    ap.add_argument("--label", default=None,
                    help="short description written to the results table")
    ap.add_argument("--show-questions", action="store_true",
                    help="print the per-question status/scores breakdown")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the report but do not touch answer_eval_results.md")
    ap.add_argument("--dump-json", metavar="PATH",
                    help="write the full report (per-question detail incl. rationale-free "
                         "scores) to PATH")
    args = ap.parse_args()

    report = evaluate(args.k)
    print_report(report, args.show_questions)

    if args.dump_json:
        Path(args.dump_json).write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8"
        )
        print(f"dumped full report to {args.dump_json}")

    label = args.label or "full-pipeline eval (unlabelled)"
    if not args.dry_run:
        append_results_row(report, label)

    log_run({
        "op": "answer_eval",
        "correlation_id": new_correlation_id(),
        "status": "ok",
        "k": args.k,
        "label": label,
        **report["aggregate"],
        **report["gates"],
    })


if __name__ == "__main__":
    main()
