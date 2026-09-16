"""Integration test for the Phase 5 full-pipeline harness
(src/evaluation/run_full_eval.py). Needs the REAL judge service process up
(unlike test_judge_service.py's in-process TestClient) since run_full_eval
talks to it over HTTP via judge_client.py -- see conftest.py's
`require_judge_service` fixture, which skips cleanly if it isn't running.

Slow: retrieval + a real LLM generation + two real judge calls, per question.
Sliced to 3 questions (not the full eval set) to keep this a smoke test, not
a second full baseline run -- `python -m src.evaluation.run_full_eval` is
the tool for that.

Run: pytest tests/  (requires start.sh's infra AND the judge service up)
"""
from src.evaluation import run_full_eval


def test_full_eval_smoke(require_judge_service, monkeypatch):
    orig_load = run_full_eval.load_eval_set

    def _sliced_load():
        d = orig_load()
        d["questions"] = d["questions"][:3]
        return d

    monkeypatch.setattr(run_full_eval, "load_eval_set", _sliced_load)

    report = run_full_eval.evaluate(k=3)

    agg = report["aggregate"]
    assert agg["n_in_scope"] + agg["n_out_of_scope"] == 3
    assert agg["n_judge_errors"] == 0
    assert 0.0 <= agg["hit_at_k"] <= 1.0

    for row in report["per_question"]:
        if row["status"] == "answered":
            assert row["faithfulness"] is not None
            assert row["answer_relevance"] is not None
            assert 0.0 <= row["faithfulness"] <= 1.0
        else:
            # Refusals/no-results are deliberately not judged (Challenges
            # and Learnings.md #6) -- not scored, not counted as errors.
            assert row["faithfulness"] is None
            assert row["answer_relevance"] is None

    assert "gates" in report
    assert set(report["gates"]) >= {"hit_at_k_pass", "faithfulness_pass", "passed"}
