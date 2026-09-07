"""Shared pytest fixtures for API integration tests.

These tests run against the REAL local Qdrant + Ollama (Phase 3 plan
decision: integration tests over mocks, matching how this project has been
built and verified throughout). Requires `scripts/start.sh` to have been run
first -- if the services aren't up, tests skip cleanly instead of hard-failing
so `pytest` doesn't look broken to someone who just forgot to start infra.
"""
import ollama
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.retrieval.search import get_client


def _services_up() -> bool:
    try:
        get_client().get_collections()
        ollama.list()
    except Exception:
        return False
    return True


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
