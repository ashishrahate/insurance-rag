# Pick up here — next session

Snapshot written 2026-09-07. Overwrite this file at the end of each session;
`docs/session-log.md` keeps the append-only history.

---

## Where the project stands

**Phase 0, 1, 2 — done.** See `docs/pipeline.md` / `data/eval/results.md`.

**Phase 3 — done.** Production API (`src/api/`): `/query`, `/ingest`,
`/healthcheck`, `/feedback`, hybrid+rerank wired into production
(`src/retrieval/hybrid.py`), score-threshold refusal guardrail
(`REFUSAL_SCORE_CUTOFF=0.5`), provider abstraction (`src/providers/`, Ollama
only), `pytest tests/` (7 integration tests against real local infra).

**Phase 4 — done.** UI, citations & feedback.

- **`ui/app.py`** — Streamlit, two tabs (Ask / Admin), pure HTTP client of
  the API (never imports pipeline internals directly). Verified working by
  the user in a real browser at `http://localhost:8501`.
- **API extensions** (all additive, `pytest tests/` still 7/7 passing):
  `Citation.parent_headers`, `QueryResponse.retrieved_chunks`,
  `QueryRequest.document_type`, `GET /admin/feedback_stats`,
  `GET /admin/recent_queries`.
- **`feedback.db`** (SQLite, `src/storage/feedback_db.py`) — `/feedback` now
  writes durable rows (question, answer, rating, comment), not just a
  `log_run()` line. One writer (the API); the UI never touches SQLite
  directly.
- **A real bug caught and fixed while adding `document_type` filtering:**
  `hybrid_fused_chunks()`'s BM25 branch never honored `state`/`document_type`
  at all — only the dense branch applied Qdrant's pre-filter. Invisible with
  a single-state, single-doc-type corpus; fixed by filtering the *fused*
  candidate list (both branches) before reranking. See `docs/pipeline.md`
  §10 finding 13.
- **A real gotcha caught running the UI:** `streamlit run ui/app.py` failed
  with `ModuleNotFoundError: No module named 'config'` even from the repo
  root — Streamlit only puts the script's own directory on `sys.path`,
  unlike `python -m src.ask`. Fixed with an explicit `sys.path.insert(0, ...)`
  at the top of `ui/app.py`. See `docs/pipeline.md` §10 finding 14.

Run it: `bash scripts/start.sh` → `uvicorn src.api.main:app --reload` →
`streamlit run ui/app.py` → `http://localhost:8501`.

**Phase 5 — not started.** This is next.

---

## Phase 5 task list (from `Roadmap_Final.md`)

Goal: automated evaluation (deterministic + LLM-judge) gating a Blue/Green
index swap, plus the OpenAI provider comparison.

1. `src/evaluation/answer_eval.py` — LLM-as-a-Judge: score each generated
   answer for **Faithfulness** (every claim supported by retrieved context?)
   and **Answer Relevance** (addresses the question?).
2. Full harness: eval set through the whole pipeline → Hit@5, Recall@5, MRR,
   Faithfulness, Answer Relevance.
3. Quality gates (tune from baseline, start at): Hit@5 ≥ 0.80 **and**
   Faithfulness ≥ 0.85.
4. **Blue/Green collection swap:** `insurance_ca_v2` (staging) → eval gate →
   flip `insurance_ca_live` alias. (`docs/scaling-notes.md` §8 already has
   the "senior engineering decision" write-up for this — a gated pipeline
   step, not a manual command.)
5. `OpenAIProvider` — real second implementation behind the Phase 3
   abstraction (`src/providers/`), config-flip via `LLM_PROVIDER`.

**GPU trigger, per the Phase 2 decision already on record:** LLM-as-judge is
~25 questions × 2 judgements, each a generation — genuinely slow on CPU
(~15-25 min/run, re-run on every change). This is the phase to actually move
off CPU-only inference, either via the OpenAI provider (task 5, cheapest to
set up) or a GPU environment.

**Done when:** an automated run reports Hit@5/Recall@5/MRR + Faithfulness +
Answer Relevance, gates pass before `insurance_ca_live` flips, and the
OpenAI/Ollama comparison is a documented, measured result (not assumed).

---

## Resume commands

```bash
bash scripts/start.sh                 # Docker + Qdrant + Ollama + models, warmed
source venv/Scripts/activate
python -m src.ask "smoke damage claims" --show-chunks   # CLI
uvicorn src.api.main:app --reload     # API
streamlit run ui/app.py                # UI (needs API up)
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
4. Cache/BM25/reranker singletons don't invalidate on re-index within a
   running server process — known, accepted limitation, see
   `docs/scaling-notes.md` §3/§8 (Redis+TTL, Blue/Green are the real fixes;
   Blue/Green is now next up, Phase 5 task 4).
5. `requirements.txt` now also pins `streamlit==1.63.0`, `pandas==3.0.5`.
6. No auth on `/admin/*` — local dev tool, out of roadmap scope, noted not built.
7. **Not yet investigated: prompt/query injection defense.** Next
   conversation topic per the user — retrieved-document content and/or user
   questions could contain text crafted to manipulate the LLM's instructions
   (e.g. a bulletin PDF containing "ignore previous instructions..."). Not
   addressed by anything built so far (`SYSTEM_PROMPT`/`JSON_SYSTEM_PROMPT`
   have no defense beyond normal instruction framing). Worth deciding whether
   this belongs in Phase 5's hardening or stands alone.
