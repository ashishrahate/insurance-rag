# State Insurance Regulations Knowledge Assistant

A production-pattern RAG system that answers questions about US state insurance
regulations, with metadata filtering, hybrid retrieval, cross-encoder reranking,
refusal guardrails, automated evaluation, and Blue/Green index deployment.

Scope is built out **one state at a time, California first**. 

## Prerequisites

- Python 3.11+ (developed on 3.13)
- Docker Desktop
- [Ollama](https://ollama.com) (native install, not Docker)

## Setup

```bash
# 1. Python environment
python -m venv venv
venv\Scripts\activate            # Windows
pip install -r requirements.txt

# 2. Ollama models
ollama pull llama3.2:3b
ollama pull nomic-embed-text

# 3. Vector database
docker compose up -d             # Qdrant on localhost:6333

# 4. Create the California collection + payload indexes
python -m src.ingestion.bootstrap_collection
```

## Daily workflow

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start.ps1   # Docker + Qdrant + Ollama + models, warmed
venv\Scripts\activate
# ... work ...
powershell -ExecutionPolicy Bypass -File scripts\stop.ps1    # append docs/session-log.md, stop deps
```

`start.ps1` is idempotent and self-heals a stray container on port 6333.
`stop.ps1 -Full` also quits the Docker Desktop and Ollama apps;
`stop.ps1 -Note "..."` adds a one-line summary to the session log.

## Verify

```bash
python test_pipeline.py
```

Embeds a sample chunk, upserts it into `insurance_ca_v1`, runs a
`state="CA"` filtered search, and asserts the chunk is retrieved.

## Layout

| Path | Purpose |
|---|---|
| `config/settings.py` | Central config: Qdrant, collection names, model IDs, payload index map |
| `src/ingestion/` | Scraping, parsing, chunking, embedding, collection bootstrap |
| `src/retrieval/` | Vector / BM25 / hybrid search, reranking |
| `src/generation/` | Prompt building, LLM calls, response validation |
| `src/api/` | FastAPI service |
| `src/evaluation/` | Deterministic + LLM-judge eval harnesses |
| `ui/` | Streamlit app |
| `data/{raw,processed,eval}/` | Scraped docs, cleaned chunks, eval set |
