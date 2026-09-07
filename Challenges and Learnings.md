# Challenges & Learnings

A running log of non-obvious problems hit while building this system, their
root causes, and the best fix (not just the one applied on the day).

Each entry: **Symptom → Root cause → Learning → Best solution → Applied so far.**

## Index

1. An in-corpus question returned a refusal *and still printed citations* — Phase 1
2. Local LLM answers take 1–2 min per question — Phase 1
3. A fixed ~290 ms client-construction cost hid behind the LLM bottleneck — Phase 1 / 2
4. The naive retrieval baseline scored high — what that does and doesn't tell us — Phase 2
5. Header-aware chunking made retrieval *worse* on this corpus — Phase 2 (Change 1)

---

## 1. An in-corpus question returned a refusal *and still printed citations*

**Phase:** 1 (naive RAG)
**Query:** *"What is the difference between the Life and Long-Term Care annual
aggregate assessments under principle-based reserving?"*

### Symptom

The answer was the refusal sentence `"The provided bulletins do not cover
this."` — even though the relevant bulletins are in the corpus. Underneath the
refusal, `src/ask.py` still printed a `Sources:` list with three real bulletin
citations (2025-9, 2026-4, 2025-8) at similarity scores 0.66–0.69.

Two separate defects, one trigger.

### Root cause A — the model over-refused (generation)

Retrieval was **correct**: the top hits were the Life bulletins (2025-8, 2026-3)
and the Long-Term Care bulletins (2025-9, 2026-4), scores 0.66–0.69. Every fact
needed to answer was in context:

- different authorising code sections — CIC `10489.992(b)(1)` (Life) vs
  `1067.11(4)(j)` (LTC)
- different premium basis (life premium vs long-term care premium)
- different aggregate amounts (`$3,078,000` vs `$1,329,000` for FY 2025-26)
- different assessment tier tables

The original `SYSTEM_PROMPT` said: *"using ONLY the numbered context passages"*
and *"If the context does not contain the answer, say: 'The provided bulletins
do not cover this.'"*

But **no single passage states the *difference*** — each bulletin describes only
its own assessment. A "what is the difference" question requires the model to
*synthesise* a contrast across two passages. `llama3.1:8b` is small and literal:
it searched for a passage containing the answer, found no verbatim statement of
the comparison, and took the refusal branch.

The deeper flaw: **refusal was delegated to the LLM's judgement through prompt
text**, and the prompt never distinguished

- "the topic is absent from the retrieved context" (should refuse) from
- "the answer requires combining/comparing several retrieved passages" (should
  answer).

A small instruction-follower collapses that ambiguity toward refusal.

### Root cause B — answer and citations are decoupled code paths

`generate.answer_question()` builds `sources` directly from `hits` (the
retrieved chunks). It is purely mechanical and never inspects `answer`.
`src/ask.py` then prints that list unconditionally. There was no branch linking
"did the model refuse?" to "should sources be shown, and how labelled?"

Retrieval always returns top-k chunks, so `sources` is always populated — hence
citations appear even under a refusal.

### Learning

1. **Prompt-based refusal is unreliable**, especially with small models. "Only
   answer from context" + "refuse if the answer isn't in context" makes the
   model a literal string-matcher: it refuses whenever the answer is not stated
   verbatim, including valid synthesis, comparison, and multi-hop questions.
2. **Refusal should be driven by a measurable retrieval signal, not the
   generator's opinion.** If retrieval is confident, the system should attempt
   an answer; the LLM's job is then narrowly "answer using this context," not
   "decide whether an answer is possible."
3. **A RAG response is not just a string.** Answer text, refusal state,
   retrieved passages, and *actually-cited* passages are distinct fields.
   Collapsing them into "print the answer, then print whatever was retrieved"
   produces misleading output (citations under a refusal).
4. **Near-duplicate documents are a distinct failure mode.** Parallel bulletins
   (Life vs LTC, one moratorium per fire declaration) invite comparison
   questions that naive retrieval + strict prompting handle badly. They belong
   in the eval set on purpose.

### Best solution

**A. Score-threshold refusal guardrail (the real fix — Phase 3).**
Decide refusal from the retrieval/rerank score, before the LLM call:

- After reranking, take the top passage's score `s`.
- If `s < cutoff` → return a structured refusal, skip the LLM entirely.
- If `s >= cutoff` → call the LLM with a prompt that says *"answer using the
  context below"* and has **no self-refusal clause at all**.
- Tune `cutoff` on the eval set: include ~5 deliberately out-of-scope questions,
  pick the threshold that best separates answerable from unanswerable. Never a
  hardcoded magic number — cross-encoder scores are uncalibrated logits.
- For the example query, top score ≈ 0.68 would be well above any reasonable
  cutoff, so the system would proceed to answer.

**B. Prompt redesign (interim, applies until the guardrail exists).**
Split the two conditions the old prompt conflated:

- Explicitly permit combining, comparing, and summarising across passages.
- Refuse **only** when *"the passages do not address the topic at all"* — not
  when the answer merely isn't stated in one sentence.
- Keep: no outside knowledge, no guessed figures/dates/code sections, cite
  bulletin numbers.

**C. Structured response contract (fixes the citation-decoupling).**
`answer_question()` should return an explicit status, and the presenter should
branch on it:

```
{
  "status": "answered" | "refused_low_relevance" | "no_results",
  "answer": "...",
  "retrieved": [ {doc_id, chunk_id, score, title, source_url}, ... ],
  "cited":     [ doc_id, ... ]      # subset of retrieved that the answer used
}
```

- `no_results` → retrieval returned nothing.
- `refused_low_relevance` → guardrail tripped; show *"retrieved but below
  confidence"* context, clearly not as "sources."
- `answered` → show `cited` as sources; optionally show the rest as "also
  retrieved."
- Populate `cited` for real by having the LLM emit the passage numbers it used
  (JSON mode, Phase 3), so `cited ⊆ retrieved` is genuine rather than "all
  retrieved."

**D. Eval-set implication (Phase 2).**
Add comparison / synthesis questions across the near-duplicate clusters
(Life vs LTC assessment; moratorium bulletin A vs B) with the expected
`doc_id`s, so this class of query is measured, not just spot-checked.

### Applied so far

- ✅ `SYSTEM_PROMPT` softened (solution B) — synthesis allowed, refusal narrowed
  to "topic not addressed at all."
- ✅ `src/ask.py` relabels the header to *"Retrieved context (not used - answer
  was a refusal)"* when the answer starts with the refusal sentence (partial C —
  the list still prints).
- ⏳ Score-threshold guardrail (A) — Phase 3.
- ⏳ Structured response contract with real `cited` (C) — Phase 3.
- ⏳ Comparison questions in the eval set (D) — Phase 2.

---

## 2. Local LLM answers take 1–2 min per question

**Phase:** 1

### Symptom

Every `python -m src.ask "..."` call takes ~60–120 s.

### Root cause

**No GPU that Ollama can use.** The machine has an Intel Arc *integrated* GPU,
which Ollama does not support (it needs NVIDIA CUDA or AMD ROCm; Intel requires a
separate IPEX-LLM build). `ollama ps` confirms `llama3.1:8b … 100% CPU`.

So each answer is an 8B model (4.9 GB, Q4) running entirely on CPU:

- **prompt prefill** — `k=5` chunks × ~300 words + system prompt ≈ 2,000–2,500
  tokens processed on CPU → ~20–60 s
- **generation** — 150–400 tokens at ~5–10 tok/s → ~20–80 s
- **model reload** — Ollama's default `keep_alive` is 5 min; spaced-out
  questions reload 4.9 GB from disk → +5–20 s

Embedding (`nomic-embed-text`) and Qdrant search are <1 s combined — not
implicated.

### Learning

The dev-loop bottleneck is **local CPU inference of an 8B model, not the RAG
code**. Match the local model to the job: Phase 1 only needs the pipeline to
*work* — real answer quality is the Phase 3/5 OpenAI swap. Keep the model
resident and bound the work per call.

### Best solution

- **Small dev model** — `llama3.2:3b` instead of `llama3.1:8b`. ~2–3× faster on
  CPU; adequate for "pipeline works".
- **Keep models warm** — `keep_alive="30m"` on every Ollama call, so
  back-to-back questions skip the reload.
- **Cap output** — `num_predict=300`; bounds worst-case generation time.
- **Lower default `k`** — 5 → 3; ~40% less prefill, and this corpus rarely needs
  5 chunks.
- **OpenAI `gpt-4o-mini` for generation** — near-instant, ~$0.0001/query;
  arrives with the Phase 3 provider switch.
- **Not worth it now:** Intel Arc acceleration via IPEX-LLM — a real speedup but
  a separate Ollama build and another moving part mid-Phase-1.

### Applied so far

- ✅ `LLM_MODEL` → `llama3.2:3b` (`config/settings.py`)
- ✅ `OLLAMA_KEEP_ALIVE = "30m"` — used by `embed.py` and `generate.py`
- ✅ `LLM_NUM_PREDICT = 300` — passed as `options={"num_predict": ...}` in `generate.py`
- ✅ `RETRIEVE_K = 3` — default `k` in `answer_question()` and `src/ask.py`
- ⏳ OpenAI generation provider — Phase 3
- ✗ IPEX-LLM / Arc acceleration — declined

---

## 3. A fixed ~290 ms client-construction cost hid behind the LLM bottleneck

**Phase:** 1 → 2 (found immediately after adding latency instrumentation)

### Symptom

The new run log showed `qdrant_client_ms` at ~209 ms p50 on every query — a
stage nobody had thought about, because it is 0.5% of a 41 s CPU query and had
never been measured.

### Root cause

`search()` called `get_client()` on every invocation, and `get_client()`
constructed a brand-new `QdrantClient` each time. Construction is not free: it
builds an HTTP session and, with the default `check_compatibility=True`, makes
a server version-check round trip.

Measured (median of 8 constructions):

| | median |
|---|---|
| `QdrantClient(..., check_compatibility=True)` | 290 ms |
| `QdrantClient(..., check_compatibility=False)` | 177 ms |

### Learning

1. **Fixed per-call overhead hides behind whatever currently dominates.** At
   41 s of CPU LLM latency, 209 ms is 0.5% and reads as nothing. Move the LLM to
   a GPU and `llm_ms` drops toward ~2 s — the *same* 209 ms becomes ~10%.
   Percentages move; absolute costs don't. Profile and fix absolute overheads
   **before** a hardware change, or they contaminate the comparison you were
   trying to make.
2. **It compounds in loops.** One CLI query constructs one client. The Phase 2
   eval harness runs ~25 questions in a single process → 25 constructions
   ≈ 7.2 s of pure setup, which would have been silently charged to
   "retrieval" in the results table.
3. **Benchmark with repeats.** A single-shot measurement made
   `check_compatibility=False` look *slower* (303 ms vs 219 ms). Median-of-8
   showed it 39% faster. Construction jitter was larger than the effect size —
   one sample would have led to the wrong conclusion.
4. **You cannot optimise what you do not measure.** This cost existed through
   all of Phase 1 and was invisible until the observability framework landed.

### Best solution

- **Cache the client** — `@lru_cache(maxsize=1)` on `get_client()`, so a process
  constructs exactly one. `get_client.cache_clear()` forces a fresh one in tests.
- **Skip the version check** — `check_compatibility=False`. The server image is
  pinned in `docker-compose.yml`, so the check spends a round trip confirming
  something already known.
- **Phase 3:** promote the cached client to a proper FastAPI lifespan singleton
  (qdrant-client is safe to share across requests) rather than relying on an
  `lru_cache` side effect.
- **General rule adopted:** instrument first, fix fixed-cost overheads, *then*
  change hardware — otherwise setup cost masquerades as workload.

### Applied so far

- ✅ `@lru_cache(maxsize=1)` on `get_client()` (`src/retrieval/search.py`)
- ✅ `check_compatibility=False`
- ✅ Verified: 5 searches in one process → **1,450 ms → 208 ms** of client setup;
  single CLI query → 290 ms → 177 ms
- ⏳ FastAPI lifespan singleton — Phase 3

---

## 4. The naive retrieval baseline scored high — what that does and doesn't tell us

**Phase:** 2 (retrieval quality)

### Symptom

The first `retrieval_eval.py` run — dense-only vector search, naive 300/50
chunking, no header awareness, no hybrid, no rerank — scored:

| Hit@3 | Recall@3 | MRR | in-scope top-1 (mean) | out-of-scope top-1 (mean) |
|---|---|---|---|---|
| 0.957 | 0.957 | 0.862 | 0.750 | 0.705 |

23 in-scope + 5 out-of-scope questions, `data/eval/ca_eval_set.json`. Only one
outright miss. A first reaction of "retrieval is basically solved, Phase 2 has
no headroom" would be the wrong reading.

### Root cause / why the number is inflated

1. **The corpus is tiny (18 docs, 49 chunks)** and most questions are
   lexically distinct from the other 17 bulletins ("bail fugitive recovery
   agents", "export list", "proof of loss"). Nearest-neighbour on 768-dim
   embeddings barely has to discriminate. Hit@**3** over 18 docs is a low bar —
   there is almost no room for the metric to fall.
2. **Hit@K and Recall@K are pass/fail at depth K.** They reward "the right doc
   is somewhere in the top 3" and say nothing about rank-1 correctness or about
   how close the wrong answers came. They look saturated here because the easy
   majority of the set drags the mean up.

### Learning — three things the baseline actually tells us

1. **Hit@3 is near-ceilinged (0.957); the signal is in MRR and in the score
   gap.** On this corpus, track **MRR** (0.862 — rank-1 correctness, real
   headroom) and the **in-scope vs out-of-scope top-1 score separation**, not
   Hit@3. Report Hit@3 for continuity but do not optimise against it.
2. **The one hard miss is the near-duplicate failure mode, on cue.** Q13
   ("which bulletin extends the moratorium to *commercial* property") returned
   three wildfire-ZIP moratorium bulletins instead of `CA_BULLETIN_2026_6`.
   `2026_6` shares ~90% of its text (the SB 824 moratorium boilerplate) with
   five siblings; the distinguishing content — "commercial property", "SB 547",
   "§ 675.55" — is a thin slice, so the whole-doc embedding sits in the same
   neighbourhood as its near-duplicates. This is exactly the class
   header-aware chunking, BM25 (exact "675.55" / "commercial property"), and
   cross-encoder rerank are supposed to fix — Q13 is the canary for whether each
   change works.
3. **The score-only refusal guardrail (Challenge #1 / Phase 3) cannot be built
   on these scores.** In-scope top-1 spans 0.684–0.829; out-of-scope top-1 spans
   0.676–0.762 — the ranges **overlap**. Out-of-scope Q27 scores 0.762, above
   half the real questions. No single cosine cutoff separates answerable from
   unanswerable today. The reranker in Change 3 is not just an accuracy tweak —
   it is what makes a usable threshold *possible*, because cross-encoder logits
   should separate genuine matches from topical-but-absent far better than
   bi-encoder cosine. Re-check this separation after every Phase 2 change.

### Best solution

- Treat the baseline as a **control, not a target**. Optimise MRR + score
  separation; keep Q13 and the out-of-scope set as the questions that must
  improve.
- Keep every configuration's row in `data/eval/results.md` so a change that
  moves Hit@3 by noise but MRR by a real margin is still visible.
- When the corpus grows (Phase 6, second state), revisit K — Hit@5 / Recall@5
  regain meaning once "the right doc in the top 3 of 18" is no longer trivial.

### Applied so far

- ✅ Baseline measured and recorded (`data/eval/results.md`, row 1).
- ✅ Eval set includes the near-duplicate comparison cluster and 5 out-of-scope
  questions (Challenge #1, solution D).
- ✅ Change 1 header-aware chunking measured — not a win, reverted (Challenge #5).
- ⏳ Changes 2–3 (BM25+RRF, rerank) — re-measure MRR + score separation each time.
- ⏳ Score-threshold guardrail tuned on the out-of-scope set — Phase 3.

---

## 5. Header-aware chunking made retrieval *worse* on this corpus

**Phase:** 2, Change 1

### Symptom

The roadmap's first planned retrieval improvement — split each bulletin into
heading-keyed sections (`RE:` topic line, `I.`/`II.` roman sections, `A.`/`B.`
subsections), chunk within each section, stamp every chunk with its heading path
in `parent_headers`, and prepend that path to the embedded text — regressed
every headline metric against the naive baseline:

| config | Hit@3 | Recall@3 | MRR |
|---|---|---|---|
| naive baseline (dense, 300/50) | 0.957 | 0.957 | **0.862** |
| header-aware, heading prefix embedded | 0.957 | 0.913 | 0.804 |
| header-aware chunks, raw-text embed (prefix off) | 0.957 | 0.913 | 0.833 |

Reverted to naive for the pipeline; the module and the `--naive` / header-aware
switch stay for a future corpus that actually has structure.

### Root cause

Two independent effects, both negative here:

1. **The embedded heading prefix is identical boilerplate across the
   near-duplicate pairs.** Life PBR 2025-8 and 2026-3 share the exact `RE:`
   line ("Principle-Based Reserving - Life Annual Aggregate Assessment"); LTC
   2025-9 / 2026-4 likewise; 5 of the 6 wildfire-moratorium bulletins share a
   generic `RE:` line with no fire name. Prepending identical text to both
   members of a pair *adds a common component to their vectors* and drowns the
   body signal that was doing the disambiguation — the fiscal year and the
   dollar figure. Result: PBR questions started returning the wrong year at
   rank 1 (q01, q14, q15, q17, q19, q22).
2. **Section splitting fragments and shifts chunk boundaries even with the
   prefix off.** 15 of 18 bulletins have no roman/letter structure, so they gain
   nothing; the 3 that do (mainly the AB 144 bulletin) get shattered into many
   sub-100-word chunks, and the sliding window now stops at every section edge,
   so chunk contents differ corpus-wide. 49 → 53 chunks, MRR still down to
   0.833.

### Learning

1. **Header-aware chunking is a technique for structured long documents, not a
   universal upgrade.** It pays off when sections are long, semantically
   distinct, and headed by *discriminative* titles. This corpus is the
   opposite: short bulletins (median ~1 chunk), flat structure, and where a
   heading exists it is shared boilerplate. Matching the technique to the
   document shape matters more than adopting it because the roadmap listed it.
2. **Prepending a heading to the embedded text is only safe when the heading
   discriminates.** A heading that is constant across the documents you most
   need to tell apart is pure common-mode noise in the vector.
3. **This is why Phase 2 measures one change at a time against a recorded
   baseline.** The roadmap *assumed* header-aware chunking would help (it was
   written for an HTML corpus + `HTMLHeaderTextSplitter`). Measurement, not the
   plan, decided it. The baseline row made the regression obvious and the
   revert safe.
4. **The near-duplicate disambiguation still needs a fix — just not this one.**
   Q13 still misses; the PBR-year and moratorium-clone separation is still thin.
   That work moves to BM25 (exact "FY 2024-25", "$3,188,000", "§ 675.55",
   "commercial property") and the cross-encoder reranker, which score the query
   against the *chunk body*, not a boilerplate header.

### Best solution

- **Keep naive fixed-size chunking as the Phase 2 pipeline base.** Carry the
  0.862 MRR baseline into Change 2.
- **Keep `src/ingestion/headers.py`, the `--naive` flag, and
  `EMBED_WITH_HEADERS` (default off).** Header-aware chunking is likely correct
  for a second state whose source documents are long regulations with a real
  table of contents; it should be re-measured there, not assumed then either.
- If `parent_headers` is wanted for Phase 4 citations, populate it as payload
  metadata without changing the embedded text or the chunk boundaries.

### Applied so far

- ✅ `src/ingestion/headers.py` — section splitter + `format_for_embedding`.
- ✅ `chunk_and_index.py` — `chunk_document(...)`, `--naive` flag, `header_aware`
  logged in the run record.
- ✅ `EMBED_WITH_HEADERS` config switch, default off, with the rationale inline.
- ✅ Measured both variants, recorded all rows in `data/eval/results.md`,
  re-indexed naive — baseline reproduced exactly (MRR 0.862).
- ✅ Pipeline reverted to naive chunking; module retained for a structured corpus.
