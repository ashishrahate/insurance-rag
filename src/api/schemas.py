"""Pydantic request/response models for the FastAPI service."""
from pydantic import BaseModel

from config.settings import RETRIEVE_K


class QueryRequest(BaseModel):
    question: str
    state: str | None = "CA"
    document_type: str | None = None  # Phase 4: e.g. "bulletin"; None = no filter
    k: int = RETRIEVE_K


class Citation(BaseModel):
    doc_id: str
    bulletin_number: str | None = None
    title: str | None = None
    source_url: str | None = None
    date_issued: str | None = None
    parent_headers: list[str] = []
    score: float


class RetrievedChunk(BaseModel):
    """One retrieved-and-reranked chunk, for the UI's sidebar (Phase 4 task 3)."""
    chunk_id: str
    doc_id: str
    title: str | None = None
    content: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation] = []
    retrieved_chunks: list[RetrievedChunk] = []
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
    answer: str | None = None
    citations: list[Citation] | None = None
    rating: str  # "up" | "down"
    comment: str | None = None


class FeedbackResponse(BaseModel):
    status: str


class FeedbackStats(BaseModel):
    up: int
    down: int
    total: int


class RecentQuery(BaseModel):
    ts: str | None = None
    correlation_id: str | None = None
    question: str | None = None
    status: str | None = None
    top_score: float | None = None
