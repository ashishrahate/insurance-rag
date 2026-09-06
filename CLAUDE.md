# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project state

This is a **greenfield learning project**, not an existing codebase. As of this writing `src/`, `ui/`, `config/`, `tests/`, and `data/` are empty scaffolding. The only Python is `test_pipeline.py` at the repo root — a partially-wired Phase 0 script whose embed-and-upsert steps are still commented out.

**`RAG_Roadmap_Final.md` is the source of truth for what to build and in what order.** It supersedes `RAG_Project_Roadmap.md` and `RAG Roadmap update.md` (kept for history only). Read the final roadmap before starting work; follow its phase order and "done when" criteria.

## How the user wants to work

The user is learning by building. Default behavior:
- Explain concepts and approach in the conversation.
- Do write implementation code into project files but before that explain the approach and reasoning in concise and to the point manner.


Reviewing, debugging, and discussing their code is welcome.

## Environment / running infrastructure

Windows. Python virtualenv lives in `venv/` (activate: `venv\Scripts\activate`). Dependencies are in `requirements.txt` (UTF-16 encoded — `pip install -r requirements.txt` still works).

External services the pipeline depends on:
- **Qdrant** in Docker: `docker run -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant` (client connects to `localhost:6333`).
- **Ollama** native Windows install (not Docker). Models: `ollama pull llama3.1:8b`, `ollama pull nomic-embed-text`.

Run the Phase 0 script: `python test_pipeline.py` (requires Qdrant + Ollama up).

No build step, linter, or test runner is configured yet. `tests/` is empty. When a test framework is added it will be pytest (per roadmap Phase 3 mention of integration tests).

## Fixed conventions (from the roadmap — do not drift)

These were decided deliberately; changing them has downstream cost.

- **Scope one US state at a time, California first.** Phases 0–5 are CA-only. Do not add NY/TX data or logic before Phase 6.
- **Chunk payload field names are locked:** `doc_id`, `state`, `document_type` (not `doc_type`), `title`, `date_issued`, `date_effective`, `source_url`, `parent_headers`, `chunk_id`, `content`.
- **Qdrant payload indexes:** `state`, `document_type`, `date_effective` — must be created before any ingestion.
- **Collections:** `insurance_ca_v1` (and `_v2` for Blue/Green staging), live alias `insurance_ca_live`. Vector size 768, cosine distance (`nomic-embed-text`).
- **Baseline before enhancement.** Phase 1 builds naive fixed-size chunking + vector-only retrieval. Header-aware chunking, BM25+RRF hybrid, and cross-encoder reranking each land in Phase 2 as *individually measured* changes against `data/eval/ca_eval_set.json`.
- **The refusal-guardrail score cutoff is tuned on the eval set, never hardcoded** — `bge-reranker-base` emits uncalibrated logits.
- **Exact query cache only** for now; semantic caching is a deferred stretch goal.
- PDF parsing uses `pdfplumber` (MIT), not PyMuPDF (AGPL).
- LLM/embedding access goes through a provider abstraction (`OllamaProvider` / `OpenAIProvider`) so the Phase 5 OpenAI swap is a config change.

## Target architecture (once built)

Layered pipeline under `src/`: `ingestion/` (scrape CDOI bulletins → clean → chunk → embed → upsert) → `retrieval/` (dense + BM25 sparse, merged via Reciprocal Rank Fusion, then `bge-reranker-base` reranking to top-3) → `generation/` (prompt build, JSON-mode LLM call, Pydantic validation) → `api/` (FastAPI: `/query`, `/ingest`, `/healthcheck`, `/feedback`; exact cache; UUID correlation IDs; refusal guardrail) → `ui/` (Streamlit). `evaluation/` holds the deterministic harness (Hit@K, Recall@K, MRR) and the LLM-as-judge harness (Faithfulness, Answer Relevance) that gate the Blue/Green collection alias swap.
