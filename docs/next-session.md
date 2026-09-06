# Pick up here — next session

Snapshot written 2026-09-06. Overwrite this file at the end of each session;
`docs/session-log.md` keeps the append-only history.

---

## Where the project stands

**Phase 0 — done.** Environment, Qdrant collection `insurance_ca_v1`
(768-dim cosine + payload indexes on `state` / `document_type` /
`date_effective`), end-to-end vector round-trip.

**Phase 1 — done.** Full naive-RAG loop over 18 California DOI bulletins:

```
scrape (PDF) -> parse (pdfplumber + boilerplate strip + date_issued)
             -> chunk (300w / 50 overlap) -> embed (nomic-embed-text)
             -> Qdrant (49 points) -> dense search + state pre-filter
             -> llama3.2:3b -> answer + citations
```

Ask a question with `python -m src.ask "..."`.

**Observability — done (not a roadmap phase).** `src/observability/`:
per-stage `Stopwatch`, JSON-Lines run log at `logs/runs.jsonl`, Ollama's own
prefill/generation token counters, and a percentile report
(`python -m src.observability.report`). Every run is tagged with `RUN_ENV`.

**Phase 2 — not started.** This is tomorrow's work.

---

## Decision made: stay on CPU through Phase 2 (and Phase 3)

**Rationale:** Phase 2's metrics — Hit@5, Recall@5, MRR — are *retrieval*
metrics. They never call the LLM, so the 41 s `llm_ms` bottleneck is simply not
in the loop. Everything Phase 2 needs is cheap on CPU:

| Work | CPU cost |
|---|---|
| Re-index 49 chunks after a chunking change | ~19 s |
| Full chunk-size × overlap sweep (9 configs) | ~3 min |
| Query embedding, 25 eval questions | ~3 s per run |
| BM25 + RRF | milliseconds (pure Python) |
| Cross-encoder rerank, 25 Q × 20 candidates | ~15–50 s per run |

Moving to Colab now would cost real friction (uploading `data/`, standing
Qdrant up inside the VM or paying for Qdrant Cloud, session timeouts, syncing
results back) for no measurable gain.

**GPU trigger is Phase 5 — LLM-as-a-Judge.** That scores every generated answer
for Faithfulness and Answer Relevance (~25 questions × 2 judgements, each a
generation). On CPU that is ~15–25 min per eval run, re-run on every change —
genuinely unworkable. Phase 3 is a secondary candidate if realistic API latency
numbers are wanted.

Sequencing: **Phase 2 (CPU) → Phase 3 (CPU) → move to GPU for Phase 5.**
The observability framework is already in place, so the CPU baseline below is
the control for that comparison.

### CPU baseline (control for the future GPU comparison)

Intel Ultra 9 185H, no Ollama-usable GPU, `llama3.2:3b`, k=3:

| Metric | p50 |
|---|---|
| `total_ms` (query) | 41,126 |
| `llm_ms` | 40,880 (~99% of total) |
| `qdrant_client_ms` | 177 (once per process, after the lru_cache fix) |
| `embed_ms` | 122 |
| `qdrant_search_ms` | 31 |
| generation | 12.0 tok/s |
| prefill | 62.3 tok/s |
| indexing | 2.59 chunks/s (49 chunks in 18.9 s) |

---

## Open decisions — answer these first tomorrow

| # | Decision | Options |
|---|---|---|
| **A** | **Cross-encoder dependency.** `bge-reranker-base` needs `sentence-transformers` + `torch` (~2 GB local install, ~1 s/query on CPU) | (a) install it and accept the cost, (b) lighter model `ms-marco-MiniLM-L6-v2` (~4× faster, slightly worse), (c) hybrid-only first, add rerank as a separate measured step |
| **B** | **"Header-aware chunking" for PDFs.** The roadmap assumed HTML + `HTMLHeaderTextSplitter`; our source is PDF | (a) derive structure from the PDF text (the `RE:` line + numbered section headings) into `parent_headers`, (b) skip it and rely on the size sweep + hybrid + rerank |
| **C** | **Eval `k`.** The app retrieves `k=3`; eval is independent | measure at k=5 (roadmap default) and k=3, report both |

---

## Phase 2 task list

1. **`data/eval/ca_eval_set.json`** — ~25 questions → target `doc_id`(s).
   Requires reading the 18 bulletins. Must span:
   - single-doc factual (e.g. "aggregate assessment for FY 2025-26 life PBR")
   - **comparison across near-duplicates** — Life vs LTC PBR (2025-8/2026-3 vs
     2025-9/2026-4), moratorium bulletin A vs B. This is the failure class from
     Challenges & Learnings #1.
   - temporal disambiguation ("the *2024-25* life assessment" → 2025-8, not 2026-3)
   - ~5 deliberately out-of-scope (seeds the Phase 3 guardrail threshold tuning)
2. **`src/evaluation/retrieval_eval.py`** — takes a retrieval function + the eval
   set, computes Hit@5 / Recall@5 / MRR, appends a row to a results table.
3. **Measure the naive baseline** (dense-only, 300/50). Record it — everything
   else is compared against this number.
4. **Change 1 — header-aware chunking** (pending decision B). Re-chunk,
   re-embed, re-index, re-measure.
5. **Change 2 — BM25 + RRF.** Add `rank-bm25` sparse index over chunk text,
   Reciprocal Rank Fusion of top-20 dense + top-20 sparse. Measure BM25 alone,
   then fused.
6. **Change 3 — cross-encoder rerank** (pending decision A). Top-20 fused →
   reranker → top-N. Re-measure, note per-query latency.
7. **Change 4 — chunk size / overlap sweep.** {200, 300, 500} × {0, 25, 50} on
   the best pipeline so far. Pick the winner on MRR.
8. **Results table** — one row per configuration so each change's individual
   contribution stays visible.

**New dependencies:** `rank-bm25`; plus `sentence-transformers` + `torch` if
decision A is (a) or (b). Add to `requirements.txt`.

**Done when:** there is a measured accuracy number, it has improved from the
naive baseline through hybrid + rerank, and each change's individual effect can
be explained from the results table.

---

## Resume commands

```bash
bash scripts/start.sh                 # Docker + Qdrant + Ollama + models, warmed
source venv/Scripts/activate
python -m src.ask "smoke damage claims" --show-chunks   # sanity check
```

At the end of the session:

```bash
bash scripts/stop.sh --note "what happened"
```

---

## Carried-forward issues

See `Challenges and Learnings.md` for the full write-ups.

1. **Prompt-based refusal is unreliable** (#1) — softened prompt is an interim
   fix; the real fix is Phase 3's score-threshold guardrail. Phase 2 seeds the
   out-of-scope questions needed to tune that threshold.
2. **CPU-only inference** (#2) — mitigated (`llama3.2:3b`, keep_alive 30m,
   num_predict 300, k=3). Real fix is the OpenAI provider (Phase 3) or GPU
   (Phase 5).
3. **Fixed client-construction cost** (#3) — fixed with `lru_cache` +
   `check_compatibility=False`. Phase 3 should promote it to a FastAPI lifespan
   singleton.
4. `date_effective` is null on every chunk — populate in Phase 2 or accept.
5. Footnote superscripts inline as digits in parsed text (cosmetic).
