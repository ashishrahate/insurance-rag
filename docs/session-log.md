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

## 2026-09-07 20:42
- **Note:** testing new stop_by_cmdline logic
- Branch `main`, HEAD `556557d phase 4: explicit system prompt, delimiter forgery, max question limit`
- Qdrant `insurance_ca_v1`: 53 points
- Commits this session (7):
  - `556557d phase 4: explicit system prompt, delimiter forgery, max question limit`
  - `1813b00 phase 4 : streamlit UI, APi entension`
  - `b50d2e7 phase 3 retrieval refactor, provider abstraction, Refusal Guardrail, FastAPI service`
  - `0e49c33 phase 2 complete: ran the 9 round sweep eval`
  - `aef6cc3 phase 2 task 2: Hybrid BM25 + RRF implemented and tested`
  - `34dafc1 phase 2 change 1, measured and settled, back from header aware chunking enbedding`
  - `8ed7875 Observability for CPU timing`
- Uncommitted (6):
  - `M docs/next-session.md`
  - `M docs/pipeline.md`
  - `M scripts/start.ps1`
  - `M scripts/start.sh`
  - `M scripts/stop.ps1`
  - `M scripts/stop.sh`

## 2026-09-07 20:42
- **Note:** testing idempotency - nothing should be running
- Branch `main`, HEAD `556557d phase 4: explicit system prompt, delimiter forgery, max question limit`
- Qdrant `insurance_ca_v1`: (qdrant not reachable) points
- Commits this session: none
- Uncommitted (7):
  - `M docs/next-session.md`
  - `M docs/pipeline.md`
  - `M docs/session-log.md`
  - `M scripts/start.ps1`
  - `M scripts/start.sh`
  - `M scripts/stop.ps1`
  - `M scripts/stop.sh`

## 2026-09-07 20:43
- **Note:** third run - confirming true idempotent no-op
- Branch `main`, HEAD `556557d phase 4: explicit system prompt, delimiter forgery, max question limit`
- Qdrant `insurance_ca_v1`: (qdrant not reachable) points
- Commits this session: none
- Uncommitted (7):
  - `M docs/next-session.md`
  - `M docs/pipeline.md`
  - `M docs/session-log.md`
  - `M scripts/start.ps1`
  - `M scripts/start.sh`
  - `M scripts/stop.ps1`
  - `M scripts/stop.sh`

## 2026-09-07 20:44
- **Note:** verifying self-match fix, run 1
- Branch `main`, HEAD `556557d phase 4: explicit system prompt, delimiter forgery, max question limit`
- Qdrant `insurance_ca_v1`: (qdrant not reachable) points
- Commits this session: none
- Uncommitted (7):
  - `M docs/next-session.md`
  - `M docs/pipeline.md`
  - `M docs/session-log.md`
  - `M scripts/start.ps1`
  - `M scripts/start.sh`
  - `M scripts/stop.ps1`
  - `M scripts/stop.sh`

## 2026-09-07 20:44
- **Note:** run 2
- Branch `main`, HEAD `556557d phase 4: explicit system prompt, delimiter forgery, max question limit`
- Qdrant `insurance_ca_v1`: (qdrant not reachable) points
- Commits this session: none
- Uncommitted (7):
  - `M docs/next-session.md`
  - `M docs/pipeline.md`
  - `M docs/session-log.md`
  - `M scripts/start.ps1`
  - `M scripts/start.sh`
  - `M scripts/stop.ps1`
  - `M scripts/stop.sh`

## 2026-09-07 20:44
- **Note:** run 3
- Branch `main`, HEAD `556557d phase 4: explicit system prompt, delimiter forgery, max question limit`
- Qdrant `insurance_ca_v1`: (qdrant not reachable) points
- Commits this session: none
- Uncommitted (7):
  - `M docs/next-session.md`
  - `M docs/pipeline.md`
  - `M docs/session-log.md`
  - `M scripts/start.ps1`
  - `M scripts/start.sh`
  - `M scripts/stop.ps1`
  - `M scripts/stop.sh`

## 2026-09-07 20:45
- **Note:** final verification - real single API instance
- Branch `main`, HEAD `556557d phase 4: explicit system prompt, delimiter forgery, max question limit`
- Qdrant `insurance_ca_v1`: 53 points
- Commits this session: none
- Uncommitted (7):
  - `M docs/next-session.md`
  - `M docs/pipeline.md`
  - `M docs/session-log.md`
  - `M scripts/start.ps1`
  - `M scripts/start.sh`
  - `M scripts/stop.ps1`
  - `M scripts/stop.sh`

## 2026-09-07 20:46
- **Note:** final settle check
- Branch `main`, HEAD `556557d phase 4: explicit system prompt, delimiter forgery, max question limit`
- Qdrant `insurance_ca_v1`: (qdrant not reachable) points
- Commits this session: none
- Uncommitted (7):
  - `M docs/next-session.md`
  - `M docs/pipeline.md`
  - `M docs/session-log.md`
  - `M scripts/start.ps1`
  - `M scripts/start.sh`
  - `M scripts/stop.ps1`
  - `M scripts/stop.sh`

## 2026-09-07 20:48
- **Note:** confirming reverted behavior
- Branch `main`, HEAD `556557d phase 4: explicit system prompt, delimiter forgery, max question limit`
- Qdrant `insurance_ca_v1`: (qdrant not reachable) points
- Commits this session: none
- Uncommitted (7):
  - `M docs/next-session.md`
  - `M docs/pipeline.md`
  - `M docs/session-log.md`
  - `M scripts/start.ps1`
  - `M scripts/start.sh`
  - `M scripts/stop.ps1`
  - `M scripts/stop.sh`

## 2026-09-07 20:48
- **Note:** confirming reverted behavior (ps1)
- Branch `main`, HEAD `556557d phase 4: explicit system prompt, delimiter forgery, max question limit`
- Qdrant `insurance_ca_v1`: (qdrant not reachable) points
- Commits this session (10):
  - `556557d phase 4: explicit system prompt, delimiter forgery, max question limit`
  - `1813b00 phase 4 : streamlit UI, APi entension`
  - `b50d2e7 phase 3 retrieval refactor, provider abstraction, Refusal Guardrail, FastAPI service`
  - `0e49c33 phase 2 complete: ran the 9 round sweep eval`
  - `aef6cc3 phase 2 task 2: Hybrid BM25 + RRF implemented and tested`
  - `34dafc1 phase 2 change 1, measured and settled, back from header aware chunking enbedding`
  - `8ed7875 Observability for CPU timing`
  - `24fd183 added scripts for starting and stopping environment dependencies`
  - `dad91f0 started using smaller 3b  model for faster inference`
  - `8c64c1c Phase 1 Complete : Naive RAG Implemented`
- Uncommitted (7):
  - `M docs/next-session.md`
  - `M docs/pipeline.md`
  - `M docs/session-log.md`
  - `M scripts/start.ps1`
  - `M scripts/start.sh`
  - `M scripts/stop.ps1`
  - `M scripts/stop.sh`

## 2026-09-07 20:49
- **Note:** Phase 3 (FastAPI: /query,/ingest,/healthcheck,/feedback, guardrail, provider abstraction) + Phase 4 (Streamlit UI, feedback.db, admin endpoints) + prompt-injection defense increment 1 + start/stop.sh docs
- Branch `main`, HEAD `556557d phase 4: explicit system prompt, delimiter forgery, max question limit`
- Qdrant `insurance_ca_v1`: (qdrant not reachable) points
- Commits this session: none
- Uncommitted (7):
  - `M docs/next-session.md`
  - `M docs/pipeline.md`
  - `M docs/session-log.md`
  - `M scripts/start.ps1`
  - `M scripts/start.sh`
  - `M scripts/stop.ps1`
  - `M scripts/stop.sh`

## 2026-09-15 23:44
- **Note:** Phase 5 part 1 (judge service, full harness, provider decoupling) + pre-Phase-6 hardening plan complete: tests added, CA corpus grown 19->37 docs / 28->61 eval Qs, both harnesses re-run, Colab GPU plan documented
- Branch `main`, HEAD `556557d phase 4: explicit system prompt, delimiter forgery, max question limit`
- Qdrant `insurance_ca_v1`: 129 points
- Commits this session: none
- Uncommitted (33):
  - `M "Challenges and Learnings.md"`
  - `M config/settings.py`
  - `M data/eval/ca_eval_set.json`
  - `M data/eval/results.md`
  - `M docs/next-session.md`
  - `M docs/pipeline.md`
  - `M docs/session-log.md`
  - `M requirements.txt`
  - `M scripts/start.ps1`
  - `M scripts/start.sh`
  - `M scripts/stop.ps1`
  - `M scripts/stop.sh`
  - `M src/generation/generate.py`
  - `M src/ingestion/embed.py`
  - `M src/observability/__init__.py`
  - `M src/observability/ollama_metrics.py`
  - `M src/providers/__init__.py`
  - `M src/providers/base.py`
  - `M src/providers/ollama_provider.py`
  - `M tests/conftest.py`
  - `?? .env.example`
  - `?? data/eval/answer_eval_results.md`
  - `?? docs/colab-gpu-plan.md`
  - `?? src/evaluation/judge_client.py`
  - `?? src/evaluation/run_full_eval.py`
  - `?? src/judge_service/__init__.py`
  - `?? src/judge_service/main.py`
  - `?? src/judge_service/prompts.py`
  - `?? src/judge_service/schemas.py`
  - `?? src/providers/openai_provider.py`
  - `?? tests/test_judge_service.py`
  - `?? tests/test_providers.py`
  - `?? tests/test_run_full_eval.py`

## 2026-09-15 23:48
- **Note:** final full shutdown - closing Ollama app and Docker Desktop
- Branch `main`, HEAD `556557d phase 4: explicit system prompt, delimiter forgery, max question limit`
- Qdrant `insurance_ca_v1`: (qdrant not reachable) points
- Commits this session: none
- Uncommitted (33):
  - `M "Challenges and Learnings.md"`
  - `M config/settings.py`
  - `M data/eval/ca_eval_set.json`
  - `M data/eval/results.md`
  - `M docs/next-session.md`
  - `M docs/pipeline.md`
  - `M docs/session-log.md`
  - `M requirements.txt`
  - `M scripts/start.ps1`
  - `M scripts/start.sh`
  - `M scripts/stop.ps1`
  - `M scripts/stop.sh`
  - `M src/generation/generate.py`
  - `M src/ingestion/embed.py`
  - `M src/observability/__init__.py`
  - `M src/observability/ollama_metrics.py`
  - `M src/providers/__init__.py`
  - `M src/providers/base.py`
  - `M src/providers/ollama_provider.py`
  - `M tests/conftest.py`
  - `?? .env.example`
  - `?? data/eval/answer_eval_results.md`
  - `?? docs/colab-gpu-plan.md`
  - `?? src/evaluation/judge_client.py`
  - `?? src/evaluation/run_full_eval.py`
  - `?? src/judge_service/__init__.py`
  - `?? src/judge_service/main.py`
  - `?? src/judge_service/prompts.py`
  - `?? src/judge_service/schemas.py`
  - `?? src/providers/openai_provider.py`
  - `?? tests/test_judge_service.py`
  - `?? tests/test_providers.py`
  - `?? tests/test_run_full_eval.py`
