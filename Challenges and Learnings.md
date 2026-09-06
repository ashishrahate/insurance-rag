# Challenges & Learnings

A running log of non-obvious problems hit while building this system, their
root causes, and the best fix (not just the one applied on the day).

Each entry: **Symptom → Root cause → Learning → Best solution → Applied so far.**

## Index

1. An in-corpus question returned a refusal *and still printed citations* — Phase 1
2. Local LLM answers take 1–2 min per question — Phase 1

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
