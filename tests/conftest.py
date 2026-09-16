"""Shared pytest fixtures for API integration tests.

These tests run against the REAL local Qdrant + Ollama (Phase 3 plan
decision: integration tests over mocks, matching how this project has been
built and verified throughout). Requires `scripts/start.sh` to have been run
first -- if the services aren't up, tests skip cleanly instead of hard-failing
so `pytest` doesn't look broken to someone who just forgot to start infra.

Phase 5 extends the same pattern to the judge service (`src/judge_service/`,
its own process on port 8100) -- tests that need it skip cleanly if it isn't
already running, exactly like Qdrant/Ollama below.
"""
import ollama
import pytest
import requests
from fastapi.testclient import TestClient

from config.settings import JUDGE_SERVICE_URL
from src.api.main import app
from src.judge_service.main import app as judge_app
from src.retrieval.search import get_client


def _services_up() -> bool:
    try:
        get_client().get_collections()
        ollama.list()
    except Exception:
        return False
    return True


def _judge_service_up() -> bool:
    try:
        resp = requests.get(f"{JUDGE_SERVICE_URL}/healthcheck", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


@pytest.fixture(scope="session", autouse=True)
def _require_local_services():
    if not _services_up():
        pytest.skip("Local Qdrant/Ollama not reachable -- run `bash scripts/start.sh` first.")


@pytest.fixture(scope="session")
def client():
    # `with` triggers the app's `lifespan` -- warms the BM25 index and
    # reranker once for the whole test session, same as a real server start.
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def require_judge_service():
    """For tests that need the REAL judge service process up (unlike
    `judge_client` above, an in-process TestClient) -- run_full_eval.py's
    judge_client.py makes real HTTP calls to JUDGE_SERVICE_URL."""
    if not _judge_service_up():
        pytest.skip(
            "Judge service not reachable -- run "
            "`uvicorn src.judge_service.main:app --port 8100` first."
        )


@pytest.fixture(scope="session")
def judge_app_client():
    """In-process TestClient against the judge service's FastAPI app --
    exercises its route logic without needing the real uvicorn process up.
    Named distinctly from src/evaluation/judge_client.py (run_full_eval.py's
    real HTTP client, which DOES need the separate process -- see
    `require_judge_service` above / test_run_full_eval.py)."""
    with TestClient(judge_app) as c:
        yield c
