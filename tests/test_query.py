"""Integration tests for the FastAPI service, against real local Qdrant +
Ollama (see conftest.py). Slow -- each answerable-question case pays for a
real ~30-45s CPU LLM call. That's the deliberate tradeoff of testing against
real infra instead of mocks (Phase 3 plan decision).

Run: pytest tests/  (requires `bash scripts/start.sh` first)
"""
import time

from qdrant_client.http.exceptions import ResponseHandlingException

ANSWERABLE_QUESTION = "smoke damage claims"
OUT_OF_SCOPE_QUESTION = (
    "What are New York's requirements for personal auto insurance rate filings?"
)


def test_healthcheck(client):
    resp = client.get("/healthcheck")
    assert resp.status_code == 200
    body = resp.json()
    assert body["qdrant"] is True
    assert body["ollama"] is True
    assert body["status"] == "ok"


def test_answerable_question(client):
    resp = client.post("/query", json={"question": ANSWERABLE_QUESTION})
    assert resp.status_code == 200
    body = resp.json()
    assert body["refused"] is False
    assert body["answer"]
    assert len(body["citations"]) > 0
    assert body["confidence"] >= 0.5  # above REFUSAL_SCORE_CUTOFF
    assert body["correlation_id"]


def test_out_of_scope_question_refuses(client):
    resp = client.post("/query", json={"question": OUT_OF_SCOPE_QUESTION})
    assert resp.status_code == 200
    body = resp.json()
    assert body["refused"] is True
    assert body["citations"] == []
    assert body["confidence"] < 0.5  # below REFUSAL_SCORE_CUTOFF -- guardrail, not the LLM


def test_cache_hit_is_fast(client):
    client.post("/query", json={"question": ANSWERABLE_QUESTION})  # ensure it's cached

    start = time.perf_counter()
    resp = client.post("/query", json={"question": ANSWERABLE_QUESTION})
    elapsed = time.perf_counter() - start

    assert resp.status_code == 200
    assert elapsed < 1.0  # a real LLM call is 30s+; a cache hit is milliseconds


def test_ingest_single_doc(client):
    resp = client.post("/ingest", json={"only": "CA_BULLETIN_2025_7"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["n_docs"] == 1
    assert body["n_chunks"] > 0


def test_feedback_accepts(client):
    resp = client.post("/feedback", json={"rating": "up", "question": "test"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_error_path_returns_structured_503(client, monkeypatch):
    """Simulates the Qdrant-unreachable failure mode (verified empirically:
    qdrant_client raises its own ResponseHandlingException, NOT a builtin
    ConnectionError -- see src/api/main.py's comment on that except clause).
    Simulated via monkeypatch rather than actually stopping the shared local
    Qdrant container, which would be destructive to a dev instance other
    tests in this same run still depend on.
    """
    import src.api.main as main_module

    def _boom(*args, **kwargs):
        raise ResponseHandlingException("simulated Qdrant outage")

    monkeypatch.setattr(main_module, "answer_question", _boom)
    resp = client.post("/query", json={"question": "anything, doesn't matter here"})
    assert resp.status_code == 503
    assert "unreachable" in resp.json()["detail"]
