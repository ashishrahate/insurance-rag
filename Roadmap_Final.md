# Production RAG System — Final Roadmap

**Project:** State Insurance Regulations Knowledge Assistant
**Goal:** A production-grade RAG system answering questions on US state insurance regulations, with metadata filtering, hybrid retrieval, cross-encoder reranking, refusal guardrails, automated evaluation (deterministic + LLM-judge), and Blue/Green index deployment.
**OS:** Windows
**LLM strategy:** Local Ollama (`llama3.1:8b`, `nomic-embed-text`) for dev → OpenAI (`gpt-4o-mini`, `text-embedding-3-small`) for evaluation and production backend.

> This document supersedes `RAG_Project_Roadmap.md` and `RAG Roadmap update.md`. It adopts the enhanced production architecture, but **structured as a learning loop** (build baseline → measure → change one thing → re-measure) and **scoped one state at a time**.

---

## Guiding principles

1. **One state at a time.** Everything through Phase 5 is **California only** (CA DOI bulletins). A second state is added in Phase 6, once the pipeline and eval harness are proven. This keeps the dataset small, the eval set hand-checkable, and every metric interpretable.
2. **Baseline before enhancement.** No optimization lands without a before/after number from the eval set. If a change can't be shown to help, it doesn't stay.
3. **One change at a time.** In Phase 2 especially, each retrieval improvement is measured in isolation so its individual contribution is known.
4. **No magic numbers.** Thresholds (refusal cutoff, quality gates) are *tuned against the eval set*, not hardcoded from a guide.
5. **Timeline is not a constraint.** Phases are milestones, not days. Move to the next phase when the current "done" criteria are met.

---

## Technical stack

| Layer | Component | Implementation | Why |
|---|---|---|---|
| Frontend | Streamlit | Search, citations sidebar, feedback buttons | Fast POC; swap to React later if needed |
| API | FastAPI | REST, exact query cache, Pydantic validation, correlation IDs | Low latency, strict contracts, traceable requests |
| LLM inference | Ollama → OpenAI | Provider abstraction (`OllamaProvider` / `OpenAIProvider`) | Zero-cost local dev, production quality on swap |
| Embeddings | Ollama `nomic-embed-text` → OpenAI `text-embedding-3-small` | Same abstraction as LLM | Consistent swap strategy |
| Vector DB | Qdrant (Docker) | Explicit payload indexes on `state`, `document_type`, `date_effective` | Fast metadata filtering at scale |
| Hybrid search | Sparse + Dense | `rank-bm25` + Qdrant vector search, merged via Reciprocal Rank Fusion | Catches exact statute/bulletin numbers *and* semantic matches |
| Reranking | Cross-encoder | `bge-reranker-base` via `sentence-transformers` | Precision scoring to pick the top 3 context chunks |
| Parsing | Python | `BeautifulSoup` + `LangChain HTMLHeaderTextSplitter` (HTML); `pdfplumber` (PDF) | Preserves legal hierarchy and tables; pdfplumber is MIT-licensed |
| Guardrails | Relevance thresholding | Rerank score below a **tuned** cutoff → hard refusal | Prevents hallucination when CA has no coverage of the question |
| Evaluation | Deterministic + LLM-judge | Hit@K, Recall@K, MRR + Faithfulness / Answer Relevance | Retrieval quality *and* answer grounding, as CI gates |
| Deployment | Docker Compose | Multi-container; Blue/Green collection alias swap | Zero-downtime index updates |

---

## Document metadata schema

Every chunk stored in Qdrant carries this payload. **Field names are fixed here** — payload index names are hard to change after data is loaded.

```json
{
  "doc_id": "CA_BULLETIN_2024_03",
  "state": "CA",
  "document_type": "bulletin",
  "title": "Wildfire Risk Mitigation Regulations",
  "date_issued": "2024-03-15",
  "date_effective": "2024-04-01",
  "source_url": "https://www.insurance.ca.gov/...",
  "parent_headers": ["Title 10", "Chapter 5", "Section 2695.18"],
  "chunk_id": "CA_BULLETIN_2024_03_c12",
  "content": "..."
}
```

Payload indexes to create in Phase 0: `state` (keyword), `document_type` (keyword), `date_effective` (datetime).

---

## Phase 0 — Environment & Qdrant schema

**Goal:** Everything installed, Qdrant running with the CA collection and payload indexes created, one end-to-end vector round-trip proven.

**Tasks:**
1. Install Python 3.11+, Docker Desktop, Ollama for Windows (native installer, not Docker).
2. Pull models: `ollama pull llama3.1:8b`, `ollama pull nomic-embed-text`.
3. Start Qdrant: `docker run -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant`.
4. Create collection `insurance_ca_v1` (vector size matching `nomic-embed-text` = 768, cosine distance).
5. Create explicit payload indexes on `state`, `document_type`, `date_effective`.
6. Finish `test_pipeline.py`: embed a test sentence → upsert with a full metadata payload → run a **filtered** search (`state = "CA"`) → confirm the point comes back. *(The embed/upsert half is currently commented out — this is the task that closes Phase 0.)*
7. Move `test_pipeline.py` logic into `src/` modules (`src/ingestion/embed.py`, `src/retrieval/search.py`) once it works.
8. Add `docker-compose.yml` (Qdrant service + volume), a `README.md`, and initialize a git repo.

**Concepts to be able to explain:** what a vector embedding is (conceptually); vector DB vs. Postgres; what a payload index does and why it must exist before ingestion; what Docker is providing.

**Done when:** `python test_pipeline.py` embeds, stores, and retrieves a payload-filtered result against `insurance_ca_v1`, and the code lives in `src/`.

---

## Phase 1 — Naive RAG, end-to-end (California)

**Goal:** Ask a question about CA insurance regulation, get an answer citing its source bulletin. Quality does not matter yet — a working pipeline does. **This is the baseline everything in Phase 2 is measured against.**

**Tasks:**
1. Scrape 15–20 HTML bulletins from the California Department of Insurance (CDOI) bulletins page. Save raw HTML to `data/raw/ca/`.
2. Parse HTML → clean text (strip nav, header, footer, boilerplate). Save to `data/processed/ca/`.
3. **Naive chunking:** fixed 300 words, 50-word overlap. No header awareness yet — that is a Phase 2 experiment.
4. Generate embeddings with Ollama `nomic-embed-text`.
5. Upsert chunks + full metadata payload into `insurance_ca_v1`.
6. Query function: embed question → Qdrant vector search → top 5 chunks.
7. Generation function: stuff top-5 into a RAG prompt → call Ollama → return answer.
8. Print the answer with source `doc_id` / `title` / `source_url` for each cited chunk.

**Concepts:** why "retrieve then generate" beats asking the LLM directly; why chunk + overlap at all; cosine similarity; prompt structure for grounded answers.

**Done when:** run a script, type a CA insurance question, get a reasonable (if imperfect) answer citing which bulletin it came from.

---

## Phase 2 — Retrieval quality (California)

**Goal:** Improve retrieval measurably. Each change is applied and evaluated **one at a time**, with results recorded in a table.

**Tasks:**
1. Build `data/eval/ca_eval_set.json`: 25 CA questions, each mapped to the target `doc_id`(s) that should be retrieved.
2. Write `src/evaluation/retrieval_eval.py`: computes **Hit@5, Recall@5, MRR** over the eval set for a given retrieval function.
3. **Measure the Phase 1 naive baseline.** Record it.
4. Change 1 — **Header-aware chunking:** re-chunk using `HTMLHeaderTextSplitter` (`<h1>`–`<h3>`) so child chunks inherit `parent_headers`. Re-embed, re-upsert, re-measure.
5. Change 2 — **BM25 sparse search:** index chunk text with `rank-bm25`. Measure BM25 alone, then **RRF fusion** of top-20 dense + top-20 sparse. Re-measure.
6. Change 3 — **Cross-encoder rerank:** pass the top-20 fused candidates through `bge-reranker-base`, take top 3. Re-measure. (Note CPU latency per query.)
7. Change 4 — **Chunk size / overlap sweep:** try {200, 300, 500} words × {0, 25, 50} overlap on the best pipeline so far. Pick the winner on MRR.
8. Keep a results table: each row = one configuration, columns = Hit@5 / Recall@5 / MRR / notes.

**Concepts:** why semantic-only retrieval misses (exact bulletin numbers, vocabulary mismatch); how BM25's TF-IDF weighting works; how RRF merges ranked lists; pre-filter vs. post-filter metadata filtering; precision@k vs. recall@k vs. MRR; why cross-encoders are more accurate but too slow to run on the whole corpus.

**Done when:** there is a measured accuracy number, it has improved from the naive baseline through hybrid + rerank, and each change's individual contribution can be explained.

---

## Phase 3 — Production API, guardrails & caching (California)

**Goal:** Wrap the pipeline in a FastAPI service with structured responses, tracing, a tuned refusal guardrail, exact caching, and provider abstraction.

**Retrieval → answer flow:**

```
Query
  └─► Exact cache ──hit──► return cached JSON
        │ miss
        ▼
   Hybrid candidate gen (BM25 + Qdrant dense)  → top 20
        ▼
   Cross-encoder rerank (bge-reranker-base)
        ├─ top score <  cutoff ──► return refusal JSON ("Insufficient CA context")
        └─ top score >= cutoff ──► top-3 contexts → LLM (JSON mode) → validated response
```

**Tasks:**
1. FastAPI endpoints: `/query`, `/ingest`, `/healthcheck`, `/feedback`.
2. Pydantic response schema: `answer`, `citations[]`, `confidence`, `state`, `retrieval_ms`, `refused` (bool).
3. **Exact query caching:** in-memory dict keyed by normalized query string. (Semantic caching is a stretch goal, not now.)
4. **Refusal guardrail:** compute the refusal cutoff by running the eval set through the reranker and picking the threshold that best separates answerable from unanswerable questions — add ~5 deliberately out-of-scope questions to the eval set for this. Document the chosen value and how it was derived.
5. Correlation IDs: generate a UUID per request, log it at every pipeline step (retrieve → rerank → generate).
6. JSON-mode LLM output + Pydantic validation with one retry on malformed output.
7. Provider abstraction: `OllamaProvider` / `OpenAIProvider` behind one interface, selected by config.
8. Error handling: LLM timeout, empty retrieval, malformed LLM response, Qdrant unreachable — each degrades gracefully with a structured error.
9. Integration tests for `/query` (answerable, refused, cache hit, error paths).

**Concepts:** why structured LLM output matters; correlation IDs for tracing; designing for provider-swappability; how cross-encoder scores are *not* calibrated probabilities, so the cutoff must be tuned empirically.

**Done when:** `curl` the API, get structured JSON with citations; out-of-scope questions return a refusal; the full request lifecycle is visible in logs by correlation ID.

---

## Phase 4 — UI, citations & feedback (California)

**Goal:** A browser interface with answers, linked citations, metadata display, and feedback capture.

**Tasks:**
1. Streamlit UI: search box, answer display, `document_type` filter (bulletin / circular / notice / all).
2. Render citations as clickable links to `source_url`, with `parent_headers` shown as a breadcrumb.
3. Sidebar: retrieved context chunks with their rerank confidence scores.
4. Thumbs up/down feedback → `feedback.db` (SQLite), linked to the query + response + correlation ID.
5. Admin tab: feedback counts, recent low-confidence and refused queries.

**Concepts:** why citation tracking matters for trust and debugging; how feedback signals feed system improvement; why AI output shown in a browser should be sanitized.

**Done when:** open browser → ask a CA question → see an answer with working citations → click through to the source → leave feedback that lands in SQLite.

---

## Phase 5 — Automated evaluation & Blue/Green (California)

**Goal:** An automated quality harness (deterministic + LLM-judge) gating a Blue/Green index swap, plus the OpenAI provider comparison.

**Tasks:**
1. `src/evaluation/answer_eval.py` — **LLM-as-a-Judge:** score each generated answer for **Faithfulness** (is every claim supported by the retrieved context?) and **Answer Relevance** (does it address the question?).
2. Full harness: run the eval set through the whole pipeline, report Hit@5, Recall@5, MRR, Faithfulness, Answer Relevance.
3. **Quality gates** (tune against observed baseline, start at): Hit@5 ≥ 0.80 **and** Faithfulness ≥ 0.85.
4. **Blue/Green collection swap:**
   - Ingest into `insurance_ca_v2` (staging).
   - Run the harness against staging.
   - If gates pass: point alias `insurance_ca_live` → `insurance_ca_v2`.
   - If gates fail: keep the current live collection, log the failing metrics.
5. Collection management endpoints: `/collections`, `/promote`.
6. Swap the backend to OpenAI (`gpt-4o-mini` + `text-embedding-3-small`), re-run the harness, and write a short doc on quality vs. cost vs. latency vs. Ollama.
7. Write the architecture doc (diagram + README) and a "lessons learned" note.

**Concepts:** Blue/Green for *data*, not just code; why automated eval is non-negotiable for production RAG; the cost/quality/latency tradeoff across providers.

**Done when:** re-ingesting CA documents runs the harness automatically and promotes the new index only if quality passes; every component can be explained to someone else.

---

## Phase 6 — Add the second state (New York or Texas)

**Goal:** Prove the pipeline and eval harness generalize beyond California.

**Tasks:**
1. Generalize the scraper for NY DFS circulars *or* TX DOI bulletins (pick one). Add 15–20 documents.
2. Parse 5 PDF bulletins with `pdfplumber` — confirm legal tables survive extraction.
3. Extend the eval set with 15–20 questions for the new state, including **cross-state** questions ("does this apply in NY?") that must filter correctly.
4. **ADR:** one collection with a `state` filter vs. one collection per state. Decide and record rationale (payload-filtered single collection is the likely answer given the payload indexes already exist).
5. Re-run the full harness **per state** and combined. Confirm state filtering keeps CA and NY/TX results from bleeding into each other.
6. Update the UI state filter (CA / NY / TX / All).

**Done when:** the system answers questions for two states, keeps them correctly separated by metadata filter, and the eval harness reports per-state metrics.

---

## Stretch goals (not scheduled)

- Semantic query caching (embedding + similarity-threshold cache hits).
- More states (FL OIR, NAIC model laws).
- React UI replacing Streamlit.
- Postgres feedback store replacing SQLite.
- Cloud VM deployment.
- Query rewriting / HyDE for recall.
- **Document storage abstraction + object-store ingestion.** Replace direct
  local-disk reads with a `Storage` interface (`exists` / `get_bytes` /
  `put_bytes`): `LocalStorage` for dev, `S3Storage` (or GCS / Azure Blob) for
  production. The manifest / catalog stores object **keys**, not absolute paths.
  Once in place, raw documents can be dropped into a bucket by any external
  process and the Phase 3 `/ingest` endpoint (or a bucket event / queue)
  triggers parse -> chunk -> embed. Raw bytes are kept (not just the vectors)
  for reprocessing, provenance / audit, citation serving, and parser debugging.
- **Derived SQLite catalog.** Promote the per-folder `manifest.json` files to
  `data/catalog.db` (one row per document, a column + timestamp per pipeline
  stage: scraped / parsed / chunked / embedded, plus `content_hash`). Built by
  scanning the manifests; the manifests stay as immutable per-scrape records.
  Do this once a second state or multi-stage per-doc status tracking is needed.

---

## Architectural Decision Records

| Date | Decision | Options considered | Rationale |
|---|---|---|---|
| — | Qdrant over ChromaDB / Weaviate | Qdrant, ChromaDB, Weaviate, Milvus | Native payload filtering, runs in Docker, production-ready; ChromaDB is in-memory-first |
| — | Ollama first, OpenAI later | Ollama, OpenAI, local HF | Zero cost in dev; the swap tests the provider abstraction |
| — | One state at a time, CA first | All states at once | Small hand-checkable eval set; every metric interpretable; scope stays bounded |
| — | Naive chunking in Phase 1, header-aware as a Phase 2 experiment | Commit to header-aware from the start | Need a baseline to prove header-aware chunking actually helps |
| — | Refusal cutoff tuned on eval set | Hardcoded score < 0.35 | `bge-reranker-base` outputs uncalibrated logits; a fixed cutoff refuses everything or nothing |
| — | `pdfplumber` over PyMuPDF | PyMuPDF, pdfplumber, PyPDF | pdfplumber is MIT-licensed (PyMuPDF is AGPL) and handles tables well |
| — | Exact cache now, semantic cache deferred | Both up front | Exact cache is trivial and safe; semantic cache risks serving subtly-wrong answers |

---

## Daily log template

```
## Session [N] — [Date]
### What I built:
### What I learned:
### What broke and how I fixed it:
### Metrics this session (if Phase 2+):
### Questions for next session:
```

---

## Key references

- Lewis et al. 2020 — original RAG paper
- Liu et al. 2023 — "Lost in the Middle" (long-context retrieval behavior)
- Kamradt 2024 — chunking strategy evaluation
- RAGAS docs — https://docs.ragas.io/ (Faithfulness / Answer Relevance definitions)
- Qdrant docs — https://qdrant.tech/documentation/ (payload indexes, aliases, Blue/Green)
- Eugene Yan 2023 — "Patterns for Building LLM-based Systems"
