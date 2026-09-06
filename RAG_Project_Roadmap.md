# Production RAG System — Project Roadmap

## Project: State Insurance Regulations Knowledge Assistant

**Goal:** Build a production-grade RAG system that answers questions about US state insurance regulations, with state-based filtering, hybrid retrieval, evaluation framework, and a usable UI.

**Timeline:** 2 weeks (14 days)
**LLM Strategy:** Start with Ollama (local, free) → swap to OpenAI API
**OS:** Windows

---

## Tech Stack (Simplified Production-Pattern Stack)

| Layer | Original (PI Knowledge Assistant) | Your Build | Why |
|---|---|---|---|
| Frontend | React 19, TypeScript, Vite | Streamlit → React (later) | Fast iteration for POC, swap when ready |
| API Layer | Node.js Lambda + Flask SAPI | FastAPI (single service) | Handles both routing and RAG logic cleanly |
| LLM | Azure OpenAI (GPT-4o) | Ollama (local) → OpenAI API | Zero cost start, production quality later |
| Embeddings | Azure OpenAI (text-embedding-3-small) | Ollama (nomic-embed-text) → OpenAI | Same swap strategy as LLM |
| Vector DB | MongoDB Atlas Vector Search | Qdrant (Docker) | Open source, purpose-built, runs locally |
| Keyword Search | MongoDB Atlas Text Search (BM25) | Qdrant payload filtering + rank-bm25 Python lib | Hybrid retrieval without extra infrastructure |
| Document Processing | Databricks + Step Functions | Python scripts (BeautifulSoup, chunking logic) | Same logic, no platform cost |
| Data Source | LivePublish DFS file shares | Web-scraped state insurance bulletins (HTML/PDF) | Publicly available, same content challenges |
| Auth | Okta OAuth2 PKCE | None (POC) → API key (v2) | Auth is important but not a RAG learning goal |
| Feedback | SQS + Lambda + S3 | SQLite / JSON file → PostgreSQL (later) | Lightweight, same pattern |
| Evaluation | Anemometer (custom) | Custom eval script | Core learning — you'll build this yourself |
| Deployment | EKS + Lambda + CloudFront | Docker Compose (local) → Cloud VM (later) | Learn containers without Kubernetes complexity |

---

## Dataset: US State Insurance Regulations

### Sources (all publicly available)

**Tier 1 — Start here (well-structured, easy to scrape):**
- California DOI Bulletins & Notices: https://www.insurance.ca.gov/0250-insurers/0300-insurers/0200-bulletins/
- New York DFS Circulars: https://www.dfs.ny.gov/industry-guidance/circular-letters
- Texas DOI Bulletins: https://www.tdi.texas.gov/bulletins/index.html

**Tier 2 — Add after POC works:**
- Florida OIR Orders: https://www.floir.com/
- NAIC Model Laws (publicly available summaries)
- State-specific insurance code sections

### Why this dataset exercises production challenges:
- **State-based filtering** → directly mirrors the original system's location filtering
- **Mixed content types** → bulletins have prose, tables, lists, legal citations
- **Temporal metadata** → regulations have dates, effective periods, superseded versions
- **Hierarchical structure** → title → chapter → section → subsection
- **Real domain complexity** → you can't fake understanding; retrieval quality matters

### Metadata schema for each document:
- `state`: CA, NY, TX, FL, etc.
- `document_type`: bulletin, circular, regulation, notice
- `date_issued`: when published
- `date_effective`: when it takes effect
- `title`: document title
- `source_url`: original URL
- `topics`: list of topics covered (auto-extracted or manual)

---

## Phase Plan

### Phase 0 — Environment Setup (Day 1)
**Goal:** Everything installed, a "hello world" from Ollama, Qdrant running, project structure ready.

**Tasks:**
1. Install Python 3.11+ (via official installer or pyenv-win)
2. Install Docker Desktop for Windows
3. Install Ollama for Windows (native installer, NOT Docker)
4. Pull models: `ollama pull llama3.1:8b` and `ollama pull nomic-embed-text`
5. Start Qdrant in Docker: `docker run -p 6333:6333 qdrant/qdrant`
6. Create project structure (see below)
7. Create virtual environment and install base dependencies
8. Write a test script: embed a sentence with Ollama, store in Qdrant, retrieve it

**Key concepts to understand:**
- What is a vector embedding? (conceptual, not mathematical)
- What is a vector database and how is it different from PostgreSQL?
- What is Docker doing for you? (isolated environment, not magic)

**Project structure:**
```
insurance-rag/
├── data/
│   ├── raw/              # Original scraped HTML/PDF files
│   ├── processed/        # Cleaned, chunked documents
│   └── eval/             # Test questions and expected answers
├── src/
│   ├── ingestion/        # Scraping, parsing, chunking, embedding
│   ├── retrieval/        # Search logic (semantic, BM25, hybrid)
│   ├── generation/       # Prompt building, LLM calls, response parsing
│   ├── evaluation/       # Accuracy testing framework
│   └── api/              # FastAPI endpoints
├── ui/                   # Streamlit app (later React)
├── config/               # Settings, model configs, prompts
├── tests/                # Unit and integration tests
├── docker-compose.yml    # Qdrant + any other services
├── requirements.txt
└── README.md
```

---

### Phase 1 — Naive RAG, End-to-End (Days 2–3)
**Goal:** Ask a question about insurance regulations, get an answer with source references. Quality doesn't matter yet — the pipeline working end-to-end matters.

**Tasks:**
1. Scrape 10–20 bulletins from California DOI (BeautifulSoup)
2. Parse HTML → clean text (strip navigation, headers, footers)
3. Chunk text (fixed-size, 300 words, 50-word overlap — same as original)
4. Generate embeddings with Ollama (nomic-embed-text)
5. Store chunks + metadata in Qdrant
6. Build a query function: embed question → search Qdrant → get top 5 chunks
7. Build a generation function: stuff chunks into prompt → call Ollama → get answer
8. Print the answer with source document references

**Key concepts to understand:**
- Chunking strategies: why 300 words? Why overlap?
- Embedding models: what do dimensions mean? What is cosine similarity?
- The "retrieve then generate" pattern: why not just ask the LLM directly?
- Prompt engineering for RAG: how to instruct the LLM to use context

**What "done" looks like:** You can run a script, type a question about CA insurance, and get a reasonable (if imperfect) answer citing which bulletin it came from.

---

### Phase 2 — Retrieval Quality (Days 4–6)
**Goal:** Significantly improve answer quality through better retrieval. This is where most production RAG effort goes.

**Tasks:**
1. Build your evaluation set: 20–30 questions with expected source documents
2. Measure baseline retrieval accuracy (what % of expected docs appear in top-5?)
3. Implement BM25 keyword search alongside semantic search
4. Implement Reciprocal Rank Fusion (RRF) to combine results
5. Add state-based metadata filtering (query only CA docs for CA questions)
6. Experiment with chunk sizes: try 200, 300, 500 words; measure impact
7. Experiment with overlap: try 0, 25, 50, 100 words; measure impact
8. Add 10–20 documents from NY and TX to test cross-state filtering
9. Re-measure accuracy after each change

**Key concepts to understand:**
- Hybrid search: why semantic alone isn't enough (keyword misses, semantic drift)
- BM25 algorithm: how term frequency and inverse document frequency work
- Reciprocal Rank Fusion: how to merge two ranked lists
- Metadata filtering: pre-filter vs post-filter and why it matters for relevance
- Evaluation metrics: precision@k, recall@k, MRR (Mean Reciprocal Rank)
- The original system's 80% accuracy threshold — what does that mean in practice?

**What "done" looks like:** You have a measurable accuracy number, you've improved it through hybrid search and filtering, and you can explain why each change helped or didn't.

---

### Phase 3 — Production API (Days 7–9)
**Goal:** Wrap the RAG pipeline in a proper API with structured responses, logging, and error handling.

**Tasks:**
1. Create FastAPI app with `/query`, `/ingest`, `/healthcheck` endpoints
2. Define response schema (answer, citations, confidence, metadata)
3. Add JSON schema validation for LLM responses (like the original)
4. Implement correlation ID logging (generate UUID per request, log at every step)
5. Add request validation and input sanitization
6. Add a `/feedback` endpoint (store to SQLite)
7. Implement the LLM abstraction layer: swap between Ollama and OpenAI with config
8. Add proper error handling (LLM timeout, empty retrieval, malformed response)
9. Write integration tests for the query endpoint

**Key concepts to understand:**
- Why structured LLM output matters (JSON mode, schema validation, retry logic)
- Correlation IDs: how to trace a request across retrieve → augment → generate
- LLM abstraction: designing for provider-swappability
- Error handling in LLM systems: what fails and how to degrade gracefully
- The original system's xAPI/SAPI separation — why two API layers?

**What "done" looks like:** You can `curl` your API, get a structured JSON response with citations, and see the full request lifecycle in your logs.

---

### Phase 4 — UI and Feedback Loop (Days 10–11)
**Goal:** Users can interact with the system through a browser and provide feedback.

**Tasks:**
1. Build Streamlit UI: search box, answer display, citation links
2. Render citations as clickable references to source documents
3. Show "related documents" sidebar
4. Add thumbs up/down feedback buttons
5. Display metadata (state filter applied, retrieval time, sources used)
6. Store feedback and link it to the query + response for analysis
7. Add a simple admin view: see feedback stats, recent queries

**Key concepts to understand:**
- Why citation tracking matters (trust, verifiability, debugging)
- Feedback loops: how user signals improve the system over time
- The original system's DOMPurify sanitization — why sanitize AI output?

**What "done" looks like:** Open browser, ask a question, see an answer with citations, click a citation to see the source, give feedback.

---

### Phase 5 — Evaluation Framework + Blue/Green Collections (Days 12–14)
**Goal:** Automated quality assurance and safe collection updates. This is what separates a demo from a production system.

**Tasks:**
1. Build automated evaluation script (run all test questions, report metrics)
2. Implement accuracy threshold check (pass/fail at 80%)
3. Implement Blue/Green collection pattern:
   - Ingest new documents into a "staging" collection
   - Run evaluation against staging
   - If pass: promote staging to live
   - If fail: keep current live, alert
4. Add collection management endpoints (`/collections`, `/promote`)
5. Swap LLM backend to OpenAI API, re-run evaluation, compare
6. Document your architecture (draw the diagram, write the README)
7. Write a "lessons learned" document

**Key concepts to understand:**
- Blue/Green deployment for data (not just code)
- Why automated evaluation is non-negotiable in production RAG
- The cost/quality tradeoff when switching LLM providers
- What the original Anemometer framework does and how yours compares

**What "done" looks like:** Re-ingest all documents, system auto-evaluates, promotes if quality passes. You can explain every component to someone else.

---

## Key References

### RAG Fundamentals
- "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al., 2020) — the original RAG paper
- "From Local to Global: A Graph RAG Approach" (Microsoft, 2024) — graph-based RAG evolution
- Anthropic's "Building effective RAG systems" guide

### Chunking & Retrieval
- "Evaluating Chunking Strategies for Retrieval" (Kamradt, 2024)
- "Lost in the Middle: How Language Models Use Long Contexts" (Liu et al., 2023)
- Qdrant documentation: https://qdrant.tech/documentation/

### Evaluation
- RAGAS framework documentation: https://docs.ragas.io/
- "Benchmarking Large Language Models in Retrieval-Augmented Generation" (Chen et al., 2024)

### Production Patterns
- "Patterns for Building LLM-based Systems" (Eugene Yan, 2023)
- "Building RAG-based LLM Applications for Production" (Anyscale, 2023)
- Haystack documentation (the framework the original system uses): https://docs.haystack.deepset.ai/

### Vector Databases
- Qdrant tutorials: https://qdrant.tech/documentation/tutorials/
- "Vector Database Comparison" — various benchmarks on ANN-benchmarks.com

---

## Architecture Decisions Log

Track every significant decision here as you build. Example format:

| Date | Decision | Options Considered | Rationale |
|---|---|---|---|
| Day 1 | Use Qdrant over ChromaDB | Qdrant, ChromaDB, Weaviate, Milvus | Qdrant has native filtering, runs in Docker, production-ready. ChromaDB simpler but in-memory only. |
| Day 1 | Ollama first, OpenAI later | Ollama, OpenAI, local HuggingFace | Zero cost during development, swap tests production readiness of abstraction layer |
| Day 3 | 300-word chunks, 50-word overlap | 200/300/500 words, 0/25/50/100 overlap | Matches original system; will benchmark alternatives in Phase 2 |

---

## Daily Log Template

```
## Day [N] — [Date]
### What I built:
### What I learned:
### What broke and how I fixed it:
### Questions for next session:
### Time spent:
```
