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

Start deps (Docker + Qdrant + Ollama + models, warmed), work, then stop deps
and log the session to `docs/session-log.md`.

**PowerShell**

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start.ps1
venv\Scripts\activate
# ... work ...
powershell -ExecutionPolicy Bypass -File scripts\stop.ps1
```

**Git Bash**

```bash
bash scripts/start.sh
source venv/Scripts/activate
# ... work ...
bash scripts/stop.sh
```

Notes:
- `start` is idempotent and self-heals a stray container on port 6333.
  Add `--no-warm` / `-NoWarm` to skip warming the models.
- `stop --full` / `-Full` also quits the Docker Desktop and Ollama apps.
- `stop --note "..."` / `-Note "..."` adds a one-line summary to the log.
- Both stop scripts append to the same `docs/session-log.md`.

## Verify

```bash
python test_pipeline.py
```

Embeds a sample chunk, upserts it into `insurance_ca_v1`, runs a
`state="CA"` filtered search, and asserts the chunk is retrieved.

## Latency & run logging

Every query and index run appends one JSON line to `logs/runs.jsonl` with
per-stage timings plus Ollama's own counters (prefill/generation tok/s, model
load time). Instrumentation is `perf_counter` deltas and a single post-hoc file
write — it does not touch the inference path.

```bash
python -m src.ask "..."                         # prints a latency line
python -m src.observability.report              # percentiles grouped by env
python -m src.observability.report --group llm_model --last 50
python -m src.observability.report --op index   # embedding throughput
```

**Comparing hardware.** Tag runs with `RUN_ENV`, then report over both logs:

```bash
RUN_ENV=local-cpu python -m src.ask "..."       # this machine
RUN_ENV=colab-t4  python -m src.ask "..."       # on the GPU box
python -m src.observability.report --file merged.jsonl   # cat both logs together
```

Env vars: `RUN_ENV` (tag, default `local-cpu`), `OBS_ENABLED=0` (disable the
log write), `LOG_LEVEL`.

### Baseline — local CPU (Intel Ultra 9 185H, no GPU), `llama3.2:3b`

| Metric | p50 |
|---|---|
| `total_ms` (query) | 41,126 |
| `llm_ms` | 40,880 (~99% of total) |
| `qdrant_client_ms` | 177 (once per process; was 290 × N calls) |
| `embed_ms` | 122 |
| `qdrant_search_ms` | 31 |
| generation | 12.0 tok/s |
| prefill | 62.3 tok/s |
| indexing | 2.59 chunks/s (49 chunks in 18.9 s) |

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
