# Pick up here — next session

Snapshot written 2026-09-07. Overwrite this file at the end of each session;
`docs/session-log.md` keeps the append-only history.

---

## Where the project stands

**Phase 0 — done.** Environment, Qdrant collection `insurance_ca_v1`
(768-dim cosine + payload indexes), end-to-end vector round-trip.

**Phase 1 — done.** Full naive-RAG loop over 18 California DOI bulletins,
dense-only retrieval. `python -m src.ask "..."`.

**Observability — done** (not a roadmap phase). `src/observability/`.

**Phase 2 — done.** Retrieval quality, measured one change at a time against
`data/eval/ca_eval_set.json` (28 questions: 23 in-scope, 5 out-of-scope).
Every strategy lives in `src/evaluation/retrievers.py`'s `RETRIEVERS`
registry and is run via `python -m src.evaluation.retrieval_eval --retriever <name>`.

| Change | MRR | Kept? |
|---|---|---|
| Baseline: dense-only, 300/50 | 0.833 (after header-aware chunking landed as the corpus's structure — see below) | — |
| Change 1: header-aware chunking, heading prepended to embedding | 0.804 | **reverted** (heading text is boilerplate across near-duplicates, dilutes signal) |
| Change 1 (kept form): sections + `parent_headers` payload, raw-text embed | 0.833 | **kept** — `EMBED_WITH_HEADERS=0` default |
| Change 2: BM25 + dense, RRF-fused at chunk level | 0.855 | **kept** — `hybrid` retriever |
| Change 3: hybrid top-20 → `bge-reranker-base` → top-3 | **0.884** | **kept** — `hybrid_rerank` retriever |
| Change 4: chunk size/overlap sweep (9 configs) | 0.884 (tied) | **no change** — 300/50 confirmed already-optimal |

Full numbers: `data/eval/results.md`. Full narrative + the "why" behind each
call: `docs/pipeline.md` §6 and §10, `docs/scaling-notes.md` (production-scale
version of each shortcut taken here).

**Important nuance:** `hybrid`/`hybrid_rerank` exist **only in the eval
harness**. `src/ask.py` (the live app) still calls dense-only `search()` —
that's correct, per the roadmap: Phase 2 measures, Phase 3 wires the winner
into production.

**Phase 3 — not started.** This is next.

---

## Phase 3 task list (from `Roadmap_Final.md`)

Goal: wrap the pipeline in a FastAPI service — structured responses, tracing,
a tuned refusal guardrail, exact caching, provider abstraction.

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

1. **Wire `hybrid_rerank` into the live query path** — this isn't an explicit
   numbered roadmap task but is the prerequisite for the diagram above;
   `src/ask.py`/`generate.py` currently call dense-only `search()`.
2. FastAPI endpoints: `/query`, `/ingest`, `/healthcheck`, `/feedback`.
3. Pydantic response schema: `answer`, `citations[]`, `confidence`, `state`,
   `retrieval_ms`, `refused` (bool).
4. **Exact query caching:** in-memory dict keyed by normalized query string.
5. **Refusal guardrail:** derive the cutoff from the reranker score
   separation already measured — `data/eval/results.md`'s `hybrid_rerank`
   row shows in-scope top1 mean ~0.965, out-of-scope mean ~0.100, a wide,
   usable gap. Document the chosen threshold and how it was derived.
6. Correlation IDs: UUID per request, logged at every stage
   (retrieve → rerank → generate) — extends the `correlation_id` pattern
   `src/observability/logger.py` already has for the CLI.
7. JSON-mode LLM output + Pydantic validation, one retry on malformed output.
8. Provider abstraction: `OllamaProvider` / `OpenAIProvider` behind one
   interface, selected by config.
9. Error handling: LLM timeout, empty retrieval, malformed LLM response,
   Qdrant unreachable — each degrades gracefully with a structured error.
10. Integration tests for `/query` (answerable, refused, cache hit, error paths).

**Done when:** `curl` the API, get structured JSON with citations; out-of-scope
questions return a refusal; the full request lifecycle is visible in logs by
correlation ID.

**New dependency:** `fastapi`/`uvicorn` are already in `requirements.txt`
(pulled in earlier, unused until now).

---

## Resume commands

```bash
bash scripts/start.sh                 # Docker + Qdrant + Ollama + models, warmed
source venv/Scripts/activate
python -m src.ask "smoke damage claims" --show-chunks   # sanity check, still dense-only
python -m src.evaluation.retrieval_eval --retriever hybrid_rerank --show-questions  # current best
```

At the end of the session:

```bash
bash scripts/stop.sh --note "what happened"
```

---

## Carried-forward issues

See `Challenges and Learnings.md` and `docs/pipeline.md` §10 for full write-ups.

1. **Prompt-based refusal is unreliable** (#1) — real fix is Phase 3's
   score-threshold guardrail (task 5 above), now unblocked by measured data.
2. **CPU-only inference** (#2) — mitigated (`llama3.2:3b`, keep_alive 30m,
   num_predict 300, k=3). Real fix is the OpenAI provider (Phase 3 task 8) or
   GPU (Phase 5).
3. `date_effective` is null on every chunk — still open.
4. Footnote superscripts inline as digits (cosmetic).
5. Every eval number so far assumes **exact** vector search (collection is
   below Qdrant's 10,000-vector HNSW threshold) — see `docs/scaling-notes.md` §5.
6. `requirements.txt` now pins `rank-bm25==0.2.2`, `sentence-transformers==6.0.1`,
   `torch==2.14.0` (Phase 2 deps, previously unpinned).
