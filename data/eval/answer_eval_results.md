# Phase 5 full-pipeline eval results

One row per run: retrieval metrics + LLM-as-judge scores, gated against config/settings.py's HIT_AT_K_GATE/FAITHFULNESS_GATE.

**Caveat on the colab-t4/gemma3:12b row:** 25 of ~54 judged (answered)
questions had at least one failed judge call (timeout or 500) --
`Challenges and Learnings.md` #8. Faithfulness/Answer Relevance are means
over only the ~46% (or fewer) of questions that happened to succeed, not
the full set -- a non-random survivor sample, not a real measurement. The
PASS is not meaningful until re-run with the swap-thrashing / timeout issue
actually fixed (not just worked around by a longer client timeout).

| date (UTC) | label | k | Hit@k | Recall@k | MRR | Faithfulness | Answer Relevance | judge errors | gate |
|---|---|---|---|---|---|---|---|---|---|
| 2026-09-16 00:52 | phase 5 baseline (refusal-aware) | 3 | 0.957 | 0.913 | 0.884 | 0.518 | 0.918 | 0 | FAIL |
| 2026-09-16 03:41 | grown CA corpus (37 docs, 61 Qs) | 3 | 0.929 | 0.929 | 0.878 | 0.706 | 0.835 | 0 | FAIL |
| 2026-09-16 ~19:30 | colab-t4, llama3.2:3b gen, gemma3:12b judge -- **NOT TRUSTED, see caveat** | 3 | 0.929 | 0.929 | 0.878 | 0.929 | 0.980 | 25 | PASS* |
