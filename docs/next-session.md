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

1. **Two-phase `run_full_eval.py` (generate-then-judge), plus finish the
   independent-judge comparison.** A same-GPU Colab attempt this session
   (`llama3.2:3b` gen + `gemma3:12b` judge, both via one local Ollama
   instance) hit **25 judge errors out of ~54 judged questions (~46%
   failure)** — Ollama very likely swap-thrashing the two models' weights
   in and out of the T4's VRAM on every question. The resulting
   Faithfulness 0.929 / Answer Relevance 0.980 in
   `data/eval/answer_eval_results.md` is marked **NOT TRUSTED** — see the
   caveat there and `Challenges and Learnings.md` #8. Plan (agreed this
   session, not yet built):
   - Split `run_full_eval.py`'s single generate+judge loop into two passes:
     **Pass 1** runs `answer_question()` for all questions and writes each
     row (question, answer, hits/context, status) to an intermediate file
     (e.g. `data/eval/_pending_full_eval.json`) — only the generation model
     ever needs to be loaded. **Pass 2** reads that file and runs
     `score_faithfulness()`/`score_answer_relevance()` on every row,
     producing the same aggregate report as today — only the judge model
     needs to be loaded, for the whole pass.
   - Expose as CLI flags (`--generate-only`, `--judge-only <file>`) with
     today's single-pass behavior kept as the default — this only matters
     when generation and judge are different models sharing one GPU, not
     for the local `llama3.2:3b`-does-both setup.
   - **Let Ollama handle model eviction itself for now** (no explicit
     `ollama stop` between passes) — but before assuming that's sufficient,
     check whether there's a cheap way to know *in advance* whether a given
     pass will actually force an eviction (e.g. query Ollama's API for
     currently-loaded models + their VRAM footprint vs. what the next call
     needs, `ollama ps` equivalent) rather than only inferring it after the
     fact from call latency/errors like this session did. Open question,
     not yet investigated.
   - `src/evaluation/judge_client.py`'s `_TIMEOUT_S` is now configurable via
     `JUDGE_TIMEOUT_S` (done this session) — helps regardless of the
     two-phase fix, since 30s was always too tight for a 12B judge model.
   - Once the two-phase restructure exists, re-run the Colab comparison (or
     redo it with `JUDGE_LLM_PROVIDER=openai` instead of a second local
     model, avoiding the swap-thrashing problem entirely by not sharing a
     GPU between two Ollama models at all — needs `OPENAI_API_KEY` in
     `.env`, see `.env.example`). Either path is the real fix for the
     same-model self-evaluation bias flagged in Challenge #6.
2. **Investigate q38/q44** — the two near-duplicate moratorium questions
   that got refused. Understand whether it's a retrieval-ranking issue
   (BM25/rerank not surfacing the distinguishing chunk) or a genuinely
   ambiguous case.
3. **Colab GPU plan — executed this session, partially.** `get_client()` now
   supports both `QDRANT_LOCAL_PATH` (embedded mode) and `QDRANT_URL`/
   `QDRANT_API_KEY` (Qdrant Cloud, used this run) — see `src/retrieval/search.py`
   and the same fix ported to `src/ingestion/bootstrap_collection.py` (it had
   its own duplicate host/port-only client that bypassed both). Retrieval
   numbers on Colab matched the local baseline exactly (Hit@3/Recall@3/MRR
   0.929/0.929/0.878, 129 points, 37-doc corpus) — good sanity check that the
   environment replicated correctly. Full-pipeline numbers did **not**
   complete cleanly — see item 1 above. Reusable artifacts for next time:
   `requirements-colab.txt` (Colab-safe deps — no `pywin32`, no forced
   `torch`/`torchvision` pins, `accelerate`/`jedi` added), `colab/colab_gpu_eval.ipynb`
   (every working command in order, including the torch/torchvision ABI
   mismatch recovery path), `docs/colab-shutdown.md` (end-of-session
   checklist). `llama3.1:8b` generation-model comparison itself never ran —
   the session stopped at the judge-swap-thrashing problem before getting to
   the second (`llama3.1:8b`) run planned in the notebook.
4. **Full-pipeline chunking comparison (naive vs. header-aware)** — header-aware
   chunking (`chunk_and_index.py` without `--naive`) was already measured once,
   in Phase 2, but only against the retrieval-only harness (Hit@K/Recall/MRR),
   where it was a net loss and reverted (`Challenges and Learnings.md` #5: MRR
   0.862 -> 0.804/0.833). It has never been run through the full-pipeline
   LLM-as-judge harness (Faithfulness/Answer Relevance), which didn't exist
   yet at the time. Open question: does the `parent_headers` context help the
   LLM's *answer* quality even though it didn't help retrieval *ranking*?
   Run as its own single-variable comparison — same corpus, same judge model,
   same generation model as whatever the current baseline run is, chunking
   mode as the only thing that changes — not mixed into the judge/generation
   model comparisons in items 1/3, to keep each change individually
   attributable per this project's own convention (CLAUDE.md "Baseline before
   enhancement").
5. Only after the above give real confidence: **Phase 5 remainder**
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
