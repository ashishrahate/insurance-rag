# Pick up here — next session

Snapshot written 2026-09-16. Overwrite this file at the end of each session;
`docs/session-log.md` keeps the append-only history.

---

## Where the project stands

**Phase 0-4 — done.** See `docs/pipeline.md` / `data/eval/results.md`.

**Phase 5 part 1 — done.** Judge service (`src/judge_service/`, port 8100,
`POST /judge`), full-pipeline harness (`src/evaluation/run_full_eval.py`),
quality gates (`HIT_AT_K_GATE=0.80`, `FAITHFULNESS_GATE=0.85`). Providers
fully decoupled: `EMBED_PROVIDER` / `LLM_PROVIDER` / `JUDGE_LLM_PROVIDER`
each independently configurable (`src/providers/`). `OpenAIProvider` built
and wired into the factory but never called against the real API yet —
needs `OPENAI_API_KEY` in a local `.env` (see `.env.example`; `python-dotenv`
loads it automatically, `config/settings.py`).

**Pre-Phase-6 hardening plan — done, all 5 steps** (same session,
2026-09-15/16):
1. Test coverage for Phase 5 (`tests/test_judge_service.py`,
   `tests/test_run_full_eval.py`, `tests/test_providers.py`) — 18/18
   passing.
2. CA corpus grown 19→37 docs (2024-2026), 53→129 chunks, in place in
   `insurance_ca_v1`.
3. Eval set grown 28→61 questions (`data/eval/ca_eval_set.json`) — new
   near-duplicate clusters covered (wildfire moratoriums, PBR fiscal years,
   Export List history, FAIR Plan recoupment).
4. Both harnesses re-run and recorded. Current numbers (61 Qs, full
   pipeline): Hit@3 **0.929** (gate PASS), **Faithfulness 0.706** (gate
   **FAIL** vs 0.85, though up from the smaller-corpus 0.518 — see caveat
   below), Answer Relevance 0.835.
5. `docs/colab-gpu-plan.md` written (planning only, not executed) — how to
   run `llama3.1:8b` on a free Colab T4 against this pipeline.

**Two real, unresolved findings from this session** (full write-ups in
`Challenges and Learnings.md` #6 and #7):
- LLM-as-judge scoring is **noisy run-to-run** with `llama3.2:3b` — the
  same input scored 0.5 then 0.0 across two identical test runs. This means
  the Faithfulness delta (0.518→0.706) between the small-corpus and
  grown-corpus baselines **cannot be trusted as real improvement** — corpus,
  BM25 stats, and generation sampling all changed at once, and the judge
  alone is known to swing 0.5+ on identical input. A same-corpus
  repeated-run comparison is needed before drawing any conclusion.
- 2 of the 33 new questions (q38, q44 — both in the wildfire-moratorium
  near-duplicate cluster) got refused rather than answered in the full
  pipeline run. Guardrail correctly fired; signals those specific
  near-duplicate pairs are right at the edge of what retrieval reliably
  separates. Not investigated further; noted as future headroom.

---

## What's next (user chose "wrap up" this session; pick one next time)

1. **Re-run with an independent judge** — set `JUDGE_LLM_PROVIDER=openai` in
   `.env` (needs `OPENAI_API_KEY`), restart the judge service, re-run
   `run_full_eval.py` on the current 61-question set. This is the real fix
   for the same-model self-evaluation bias flagged in Challenge #6, and
   would give a trustworthy Faithfulness number for the first time.
   Generation stays local/free (`LLM_PROVIDER` untouched) — only the judge
   costs anything, and it's a fraction of a cent for 61 questions.
2. **Investigate q38/q44** — the two near-duplicate moratorium questions
   that got refused. Understand whether it's a retrieval-ranking issue
   (BM25/rerank not surfacing the distinguishing chunk) or a genuinely
   ambiguous case.
3. **Execute the Colab GPU plan** (`docs/colab-gpu-plan.md`) — run
   `llama3.1:8b` on a free T4, including the one small `get_client()` code
   change the doc calls for (`QDRANT_LOCAL_PATH` embedded-mode branch, not
   yet made). Gives a real second data point (bigger local model) before
   deciding whether OpenAI is even necessary for generation.
4. Only after the above give real confidence: **Phase 5 remainder**
   (Blue/Green swap, `OpenAIProvider` real-API generation comparison) →
   **Phase 6** (second state, per `Roadmap_Final.md` — explicitly not
   started yet; CA-only scope was deliberately reconfirmed this session).

---

## Resume commands

```bash
bash scripts/start.sh                 # Docker + Qdrant + Ollama + models, warmed
source venv/Scripts/activate
python -m src.ask "smoke damage claims" --show-chunks   # CLI
uvicorn src.api.main:app --reload     # API
uvicorn src.judge_service.main:app --port 8100   # judge service (own terminal)
streamlit run ui/app.py                # UI (needs API up)
pytest tests/                          # integration tests (needs infra + judge service up)
python -m src.evaluation.retrieval_eval --retriever hybrid_rerank --label "..."
python -m src.evaluation.run_full_eval --label "..."   # needs judge service up too
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
   running server process — known, accepted limitation; Blue/Green
   (Phase 5, not yet built) is the real fix.
5. No auth on `/admin/*` — local dev tool, out of roadmap scope, noted not built.
6. **Prompt injection: increment 1 done**, not a finished defense — see
   `docs/scaling-notes.md` §11.
7. **LLM-as-judge scoring is noisy run-to-run** (this session's finding,
   Challenge #7) — don't trust a single-run score delta as a real signal
   until either an independent/stronger judge or repeated-run averaging is
   in place.
8. **`CA_BULLETIN_2024_4` has no parseable `date_issued`** (new this
   session, cosmetic like #2 above — same class of issue, different doc).
