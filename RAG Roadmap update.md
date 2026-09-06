# Production RAG System — Project Roadmap (Enhanced Edition)

**Project:** State Insurance Regulations Knowledge Assistant  
**Goal:** Build a production-grade RAG system answering queries on US state insurance regulations featuring state metadata filtering, hybrid search, cross-encoder reranking, strict refusal guardrails, query caching, automated evaluation, and a Blue/Green deployment strategy.  
**Timeline:** 2 Weeks (14 Days)  
**LLM Strategy:** Local (Ollama `llama3.1:8b`) for dev $\rightarrow$ OpenAI API (`gpt-4o-mini`) for evaluations & production backend  
**OS:** Windows  

---

## Technical Stack Architecture

| Layer | Component | Production Pattern / Implementation | Why |
|---|---|---|---|
| **Frontend** | Streamlit | UI with search, citations sidebar, and user feedback | Fast POC development, easy swap to React later |
| **API Layer** | FastAPI | REST API, exact/semantic query caching, Pydantic validation | Low latency, strict request/response contracts |
| **LLM Inference** | Ollama $\rightarrow$ OpenAI | Abstraction layer switching between Ollama & OpenAI | Zero-cost local dev, high-throughput production |
| **Embeddings** | Ollama (`nomic-embed-text`) $\rightarrow$ OpenAI | `text-embedding-3-small` | High-dimensional semantic alignment |
| **Vector DB** | Qdrant (Docker) | Explicit payload indexing (`state`, `doc_type`, `date`) | Fast metadata filtering at scale |
| **Hybrid Search** | Sparse + Dense | BM25 (`rank-bm25`) + Qdrant Vector Search merged via RRF | Prevents vocabulary miss and semantic drift |
| **Reranking** | Cross-Encoder | `bge-reranker-base` (`sentence-transformers`) | Precision scoring to select top 3 context chunks |
| **Parsing** | Python | `BeautifulSoup` (HTML) + `PyMuPDF`/`pdfplumber` (PDFs) | Preserves tabular and structural regulatory layout |
| **Guardrails** | Relevance Thresholding | Score cutoff ($Score < 0.35$) $\rightarrow$ Hard refusal | Prevents hallucinations on missing coverage |
| **Evaluation** | Deterministic + LLM-Judge | $Hit@K$, $Recall@K$, $MRR$ + Faithfulness/Relevance scoring | Continuous integration quality gates |
| **Deployment** | Docker Compose | Multi-container setup with Blue/Green collection swaps | Zero-downtime index updates |

---

## Document Metadata Schema

Each processed document chunk stores the following payload structure in Qdrant:

```json
{
  "doc_id": "CA_BULLETIN_2024_03",
  "state": "CA",
  "document_type": "bulletin",
  "title": "Wildfire Risk Mitigation Regulations",
  "date_issued": "2024-03-15",
  "date_effective": "2024-04-01",
  "source_url": "[https://www.insurance.ca.gov/](https://www.insurance.ca.gov/)...",
  "parent_headers": ["Title 10", "Chapter 5", "Section 2695.18"],
  "chunk_id": "CA_BULLETIN_2024_03_c12",
  "content": "..."
}

Phase Execution PlanPhase 0 — Environment Setup & Payload Indexing (Day 1)Goal: Environment validation, Qdrant setup with payload schema indexing, and baseline end-to-end vector test.Tasks:Install Python 3.11+, Docker Desktop, and Ollama for Windows.Pull Ollama models: ollama pull llama3.1:8b and ollama pull nomic-embed-text.Start Qdrant in Docker: docker run -p 6333:6333 qdrant/qdrant.Initialize Qdrant collection and create explicit Payload Indexes for state, document_type, and date_effective.Write test_pipeline.py: Embed a test sentence, insert with metadata payload, run a filtered search query.Phase 1 — Layout-Aware Scraping & Structural Ingestion (Days 2–3)Goal: Scrape real regulatory bulletins, parse nested legal structures, and ingest header-aware chunks into Qdrant.Tasks:Scrape 15–20 HTML bulletins from California Department of Insurance (CDOI).Implement Header-Aware Chunking: Parse HTML using BeautifulSoup / LangChain HTMLHeaderTextSplitter (<h1>-<h3>) so child chunks inherit parent context headings.Parse 5 PDF bulletins from Texas TDI or NY DFS using PyMuPDF or pdfplumber to extract legal tables without layout corruption.Generate embeddings via Ollama nomic-embed-text and upsert vector points with payload into Qdrant.Phase 2 — Two-Stage Hybrid Retrieval & Evaluation (Days 4–6)Goal: Implement hybrid search (Dense + BM25), Cross-Encoder reranking, and establish deterministic retrieval benchmarking.Tasks:Build a target test dataset (eval_set.json) containing 25 queries mapped to target doc_ids.Implement Sparse Search: Index chunk text using rank-bm25.Implement Reciprocal Rank Fusion (RRF): Combine top 20 dense and sparse candidate lists.Implement Two-Stage Reranking: Pass top 20 candidates into bge-reranker-base (sentence-transformers) to select the top 3 context chunks.Build retrieval evaluation script calculating $Hit@5$, $Recall@5$, and Mean Reciprocal Rank ($MRR$). Compare Naive Vector vs. Hybrid + Rerank.Phase 3 — Production API, Guardrails & Query Caching (Days 7–9)Goal: Build a FastAPI application featuring correlation ID tracing, score-based refusal guardrails, exact query caching, and provider abstraction.Query ──► [ Exact Cache ] ──(Hit)──► Return Cached JSON Response
                │ (Miss)
                ▼
      [ Hybrid Candidate Gen ] (Sparse BM25 + Qdrant Dense)
                │ Top 20
                ▼
      [ Cross-Encoder Rerank ] (bge-reranker-base)
                │
                ├── Rerank Score < 0.35 ──► Return Refusal JSON ("Insufficient State Context")
                └── Rerank Score >= 0.35 ──► Pass Top-3 Contexts to LLM (JSON Mode)
Tasks:Create FastAPI endpoints: /query, /ingest, /healthcheck, /feedback.Add Exact & Semantic Query Caching using an in-memory layer to return instant cached responses for identical queries.Implement Relevance Guardrail: If top reranker score is $< 0.35$, bypass LLM and return a structured refusal.Add structured JSON output validation via Pydantic and attach UUID correlation IDs to request logs.Create provider abstraction switch (OllamaProvider vs. OpenAIProvider).Phase 4 — UI, Citation Engine & Feedback Tracking (Days 10–11)Goal: Browser interface displaying answers, linked regulatory citations, metadata filters, and user feedback capture.Tasks:Build Streamlit UI with state filter dropdown (CA, NY, TX, All).Display response output alongside clickable source document links and header breadcrumbs.Add a sidebar showing retrieved context chunks and confidence scores.Implement thumbs up/down feedback mechanisms logged directly to SQLite (feedback.db).Build a lightweight admin tab showing feedback metrics and low-confidence queries.Phase 5 — Automated Evaluation, Blue/Green Deployments & Swap Test (Days 12–14)Goal: Continuous evaluation harness using LLM-as-a-Judge, automated index promotion, and provider comparison.Tasks:Build LLM-as-a-Judge Evaluation Script: Evaluate generated answers for Faithfulness (hallucination check) and Answer Relevance.Implement Blue/Green Collection Strategy:Ingest new documents into insurance_v2 (staging collection).Run automated test harness ($Hit@5 \ge 0.80$ and Faithfulness $\ge 0.85$).If passed, swap active alias insurance_live to insurance_v2.Swap backend LLM to OpenAI gpt-4o-mini, re-evaluate dataset, and document quality vs. cost vs. latency trade-offs.Architectural Decision Records (ADR)DecisionSelected OptionAlternatives ConsideredRationaleChunkingHeader-Aware Structural SplittingFixed 300-word chunkingPrevents breaking sub-clauses and losing legal parent context.RetrievalHybrid (BM25 + Dense) + Cross-EncoderVector Search onlyBM25 captures exact statute/bulletin numbers; Reranker filters noise.Metadata IndexingQdrant Payload IndexesIn-memory filteringPrevents performance degradation when executing state filters over large collections.SafetyReranker Score Thresholding ($< 0.35$)Direct LLM promptingStops LLMs from hallucinating answers when state data is absent.