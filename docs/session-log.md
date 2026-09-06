# Session Log

Appended by `scripts/stop.ps1` at the end of each working session.

## 2026-09-05 22:01
- **Note:** Phase 1 complete; added scripts/start.ps1 + stop.ps1; dev LLM -> llama3.2:3b (keep_alive 30m, num_predict 300, k=3); Qdrant migrated to docker compose stack; added docs/pipeline.md + Challenges and Learnings.md
- Branch `main`, HEAD `dad91f0 started using smaller 3b  model for faster inference`
- Qdrant `insurance_ca_v1`: (qdrant not reachable) points
- Commits this session (6):
  - `dad91f0 started using smaller 3b  model for faster inference`
  - `8c64c1c Phase 1 Complete : Naive RAG Implemented`
  - `560eb9c Phase 1: Task 3 Chunking->Indexing->storing indexes in CA collection`
  - `8ff55af Phase 1 Parser  built for CA PDFs`
  - `96d0619 Phase 1 Scraper built for CA PDFs after 2020`
  - `8f8a4af Phase 0: environment, Qdrant schema, end-to-end vector test`
- Uncommitted (2):
  - `?? scripts/start.ps1`
  - `?? scripts/stop.ps1`

## 2026-09-05 22:11
- **Note:** Bash test run: added scripts/start.sh + stop.sh
- Branch `main`, HEAD `24fd183 added scripts for starting and stopping environment dependencies`
- Qdrant `insurance_ca_v1`: 49 points
- Commits this session (1):
  - `24fd183 added scripts for starting and stopping environment dependencies`
- Uncommitted (2):
  - `?? scripts/start.sh`
  - `?? scripts/stop.sh`

## 2026-09-05 22:13
- Branch `main`, HEAD `24fd183 added scripts for starting and stopping environment dependencies`
- Qdrant `insurance_ca_v1`: 49 points
- Commits this session: none
- Uncommitted (4):
  - `M README.md`
  - `M docs/session-log.md`
  - `?? scripts/start.sh`
  - `?? scripts/stop.sh`

## 2026-09-06 00:37
- Branch `main`, HEAD `24fd183 added scripts for starting and stopping environment dependencies`
- Qdrant `insurance_ca_v1`: 49 points
- Commits this session: none
- Uncommitted (17):
  - `M .gitignore`
  - `M "Challenges and Learnings.md"`
  - `M README.md`
  - `M config/settings.py`
  - `M docs/session-log.md`
  - `M src/ask.py`
  - `M src/generation/generate.py`
  - `M src/ingestion/chunk_and_index.py`
  - `M src/retrieval/search.py`
  - `?? docs/next-session.md`
  - `?? scripts/start.sh`
  - `?? scripts/stop.sh`
  - `?? src/observability/__init__.py`
  - `?? src/observability/logger.py`
  - `?? src/observability/ollama_metrics.py`
  - `?? src/observability/report.py`
  - `?? src/observability/timing.py`
