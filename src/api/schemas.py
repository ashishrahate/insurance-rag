"""Pydantic request/response models for the FastAPI service."""
from pydantic import BaseModel

from config.settings import RETRIEVE_K


class QueryRequest(BaseModel):
    question: str
    state: str | None = "CA"
    k: int = RETRIEVE_K


class Citation(BaseModel):
    doc_id: str
    bulletin_number: str | None = None
    title: str | None = None
    source_url: str | None = None
    date_issued: str | None = None
    score: float


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation] = []
    confidence: float | None = None
    state: str | None
    retrieval_ms: float | None = None
    refused: bool
    correlation_id: str


class HealthResponse(BaseModel):
    status: str  # "ok" | "degraded"
    qdrant: bool
    ollama: bool


class IngestRequest(BaseModel):
    only: str | None = None  # re-index just this doc_id, or all if omitted


class IngestResponse(BaseModel):
    status: str
    n_docs: int
    n_chunks: int
    points_count: int | None = None


class FeedbackRequest(BaseModel):
    correlation_id: str | None = None
    question: str | None = None
    rating: str  # "up" | "down"
    comment: str | None = None


class FeedbackResponse(BaseModel):
    status: str
