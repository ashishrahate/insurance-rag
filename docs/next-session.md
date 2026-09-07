# Pick up here — next session

Snapshot written 2026-09-07. Overwrite this file at the end of each session;
`docs/session-log.md` keeps the append-only history.

---

## Where the project stands

**Phase 0, 1 — done.**

**Phase 2 — done.** Retrieval quality, measured one change at a time.
Best pipeline: `hybrid_rerank` (BM25+dense, RRF-fused, `bge-reranker-base`
reranked), MRR 0.833 → 0.884. Full numbers: `data/eval/results.md`.

**Phase 3 — done.** Production API, guardrails & caching.

- **Retrieval refactor:** `src/retrieval/hybrid.py::retrieve_chunks()` is now
  the shared chunk-level retrieval core — used by `src/ask.py` (CLI, unchanged
  interface) AND `src/api/main.py` (new). `src/evaluation/retrievers.py` was
  refactored to call into it too (re-verified: identical eval numbers,
  MRR 0.884, post-refactor).
- **Provider abstraction:** `src/providers/` — `Provider` interface,
  `OllamaProvider` implemented, `LLM_PROVIDER` config switch.
  `embed.py`/`generate.py` route through `get_provider()`, never `ollama.*`
  directly. `OpenAIProvider` deferred to Phase 5 (no API key needed yet).
- **Refusal guardrail:** `REFUSAL_SCORE_CUTOFF = 0.5` (config/settings.py),
  derived from `data/eval/hybrid_rerank_scores.json` — in-scope top1 scores
  [0.693, 1.000], out-of-scope [0.003, 0.286], clean non-overlapping gap.
  Below cutoff, `generate.py` skips the LLM call entirely (verified: refused
  queries return in ~4-12s vs. ~40-47s for an actual generation).
- **JSON-mode output:** `answer_question(..., json_mode=True)` — Ollama
  `format="json"`, Pydantic `LLMAnswer` validation, one retry with a
  corrective follow-up message on `JSONDecodeError`/`ValidationError`.
- **FastAPI service:** `src/api/main.py` — `/query`, `/ingest`,
  `/healthcheck`, `/feedback`. Startup `lifespan` warms the BM25 index +
  reranker once (not per-request). Sync `def` routes (not `async def`) so
  FastAPI's threadpool handles Ollama's blocking calls without an async
  rewrite. Exact-query cache (in-memory dict, cleared on `/ingest`).
  Correlation IDs thread through `logs/runs.jsonl`.
- **Error handling — actually tested, not assumed:** stopped the real Qdrant
  container mid-session, confirmed `/query` returns a structured `503`
  (`qdrant_client.http.exceptions.ResponseHandlingException`, NOT Python's
  builtin `ConnectionError` — verified empirically, an early version of the
  except clause had this wrong), then restarted Qdrant and confirmed recovery.
  A catch-all `Exception` handler backstops anything else — never a bare 500.
- **Tests:** `tests/` — pytest integration tests against real local
  Qdrant+Ollama (`pytest tests/`, all 7 passing). One exception:
  the Qdrant-outage test uses `monkeypatch` (not a live container stop —
  that was done manually once, separately, to derive the exception type and
  verify the real behavior).

Run it: `bash scripts/start.sh`, then
`uvicorn src.api.main:app --reload` (or `python -m src.ask "..."` for the CLI).

**Phase 4 — not started.** This is next.

---

## Phase 4 task list (from `Roadmap_Final.md`)

Goal: browser interface with answers, linked citations, metadata display,
feedback capture.

1. Streamlit UI: search box, answer display, `document_type` filter.
2. Render citations as clickable links to `source_url`, `parent_headers` as
   a breadcrumb.
3. Sidebar: retrieved context chunks with their rerank confidence scores.
4. Thumbs up/down feedback → SQLite (`/feedback` already exists and logs to
   `logs/runs.jsonl` — Phase 4 adds real `feedback.db` storage, linked to
   query + response + correlation ID).
5. Admin tab: feedback counts, recent low-confidence and refused queries.

**Done when:** open browser → ask a CA question → see an answer with working
citations → click through to the source → leave feedback that lands in SQLite.

---

## Resume commands

```bash
bash scripts/start.sh                 # Docker + Qdrant + Ollama + models, warmed
source venv/Scripts/activate
python -m src.ask "smoke damage claims" --show-chunks   # CLI, hybrid+rerank now
uvicorn src.api.main:app --reload     # API service
pytest tests/                          # integration tests (needs infra up)
```

At the end of the session:

```bash
bash scripts/stop.sh --note "what happened"
```

---

## Carried-forward issues

See `Challenges and Learnings.md` and `docs/pipeline.md` §10 for full write-ups.

1. `date_effective` is null on every chunk — still open.
2. Footnote superscripts inline as digits (cosmetic).
3. Every retrieval eval number so far assumes **exact** vector search — see
   `docs/scaling-notes.md` §5.
4. **`docs/pipeline.md` §3–7 (query-path detail) is now stale** — still
   describes Phase 1's dense-only flow. Needs a rewrite for Phase 3's
   hybrid+rerank+guardrail+API flow; not done this session (flagged in the
   doc itself, not silently wrong).
5. Cache/BM25/reranker singletons don't invalidate on re-index within a
   running server process — known, accepted limitation at this scale, see
   `docs/scaling-notes.md` §3/§8 (Redis+TTL, Blue/Green are the real fixes,
   both deferred).
6. `requirements.txt` now also pins `pytest==9.1.1` (+ `pluggy`, `iniconfig`).
