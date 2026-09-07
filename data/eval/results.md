# Phase 2 retrieval eval results

One row per configuration. Baseline first; each later row is a single measured change against it.

| date (UTC) | label | retriever | k | chunk | Hit@k | Recall@k | MRR | in-scope top1 (mean) | OOS top1 (mean) |
|---|---|---|---|---|---|---|---|---|---|
| 2026-09-06 17:29 | naive baseline: dense-only, 300/50, header-unaware | dense | 3 | 300/50 | 0.957 | 0.957 | 0.862 | 0.750 | 0.705 |
| 2026-09-06 17:47 | Change 1: header-aware chunking, heading path prepended to embedded text | dense | 3 | 300/50 | 0.957 | 0.913 | 0.804 | 0.773 | 0.674 |
| 2026-09-06 17:48 | Change 1: sections + parent_headers payload, raw-text embed (EMBED_WITH_HEADERS=0) | dense | 3 | 300/50 | 0.957 | 0.913 | 0.833 | 0.746 | 0.697 |
| 2026-09-06 17:49 | sanity check: `--naive` flag, flat chunking, no sections (confirms sections change doesn't break the baseline path) | dense | 3 | 300/50 | 0.957 | 0.957 | 0.862 | 0.750 | 0.705 |
| 2026-09-07 21:03 | **current default** (header_aware=True, EMBED_WITH_HEADERS=0) re-verified after re-index — matches row 3 | dense | 3 | 300/50 | 0.957 | 0.913 | 0.833 | 0.746 | 0.697 |
| 2026-09-07 21:15 | Change 2: BM25 + dense, RRF-fused at chunk level (RRF_K=60). NOTE: top1 columns are now RRF scores, not cosine -- not comparable to dense rows above | hybrid | 3 | 300/50 | 0.957 | 0.913 | 0.855 | 0.033 | 0.032 |
