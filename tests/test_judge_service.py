"""Integration tests for the judge service (Phase 5), against the real local
Ollama (see conftest.py's `judge_app_client` fixture -- an in-process
TestClient, so these don't need the separate uvicorn process up, only
Ollama). Slow -- each case pays for a real LLM call, same tradeoff as
tests/test_query.py.

Run: pytest tests/  (requires `bash scripts/start.sh` first)

Deliberately asymmetric assertions below: a "should score low" case is
asserted on score (empirically stable, <=0.3, across repeated runs). A
"should score high" case is asserted on *shape* only (valid 0-1 float,
non-empty rationale), not a threshold -- the same clearly-grounded input
scored 0.5 on one run and 0.0 on the next with llama3.2:3b (the default
local judge model), pure sampling non-determinism from a small,
unconfident model. That's exactly the known judge-quality gap
JUDGE_LLM_PROVIDER exists to let you fix independently (see
rag-project-status memory / Faithfulness gate failure) -- asserting a
score threshold here would test this specific weak model's mood on a
given sample, not our code's wiring, and would make the suite flaky.
"""
QUESTION = "What is the minimum days to submit proof of loss during a state of emergency?"
GROUNDED_CONTEXT = [
    "During a declared state of emergency, insurers must allow at least 100 "
    "days from the date of loss for proof of loss submission, per CIC "
    "2051.5(b)(3)(A)."
]
CONTRADICTING_CONTEXT = [
    "This bulletin concerns wildfire smoke damage claims handling and does "
    "not address proof-of-loss deadlines."
]


def _assert_valid_score_shape(body: dict):
    assert 0.0 <= body["score"] <= 1.0
    assert body["rationale"]


def test_healthcheck(judge_app_client):
    resp = judge_app_client.get("/healthcheck")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_faithfulness_returns_valid_score_shape(judge_app_client):
    resp = judge_app_client.post("/judge", json={
        "question": QUESTION,
        "answer": "Insurers must allow at least 100 days from the date of loss.",
        "context": GROUNDED_CONTEXT,
        "criterion": "faithfulness",
    })
    assert resp.status_code == 200
    _assert_valid_score_shape(resp.json())


def test_faithfulness_scores_unsupported_answer_low(judge_app_client):
    resp = judge_app_client.post("/judge", json={
        "question": QUESTION,
        "answer": "Insurers must allow at least 100 days from the date of loss.",
        "context": CONTRADICTING_CONTEXT,
        "criterion": "faithfulness",
    })
    assert resp.status_code == 200
    body = resp.json()
    _assert_valid_score_shape(body)
    assert body["score"] <= 0.3


def test_answer_relevance_returns_valid_score_shape(judge_app_client):
    resp = judge_app_client.post("/judge", json={
        "question": QUESTION,
        "answer": "Insurers must allow at least 100 days from the date of loss.",
        "context": [],
        "criterion": "answer_relevance",
    })
    assert resp.status_code == 200
    _assert_valid_score_shape(resp.json())


def test_answer_relevance_scores_off_topic_answer_low(judge_app_client):
    resp = judge_app_client.post("/judge", json={
        "question": QUESTION,
        "answer": "Sacramento is the capital of California.",
        "context": [],
        "criterion": "answer_relevance",
    })
    assert resp.status_code == 200
    body = resp.json()
    _assert_valid_score_shape(body)
    assert body["score"] <= 0.3
