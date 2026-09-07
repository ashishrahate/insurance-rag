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
| 2026-09-07 22:01 | Change 3: hybrid top-20 -> bge-reranker-base -> top-3 | hybrid_rerank | 3 | 300/50 | 0.957 | 0.913 | 0.884 | 0.965 | 0.100 |
| 2026-09-07 22:15 | Change 4 sweep: 200/0, hybrid_rerank | hybrid_rerank | 3 | 200/0 | 0.957 | 0.913 | 0.884 | 0.940 | 0.222 |
| 2026-09-07 22:17 | Change 4 sweep: 200/25, hybrid_rerank | hybrid_rerank | 3 | 200/25 | 0.957 | 0.913 | 0.884 | 0.964 | 0.138 |
| 2026-09-07 22:19 | Change 4 sweep: 200/50, hybrid_rerank | hybrid_rerank | 3 | 200/50 | 0.957 | 0.913 | 0.862 | 0.955 | 0.104 |
| 2026-09-07 22:22 | Change 4 sweep: 300/0, hybrid_rerank | hybrid_rerank | 3 | 300/0 | 0.957 | 0.913 | 0.884 | 0.952 | 0.069 |
| 2026-09-07 22:25 | Change 4 sweep: 300/25, hybrid_rerank | hybrid_rerank | 3 | 300/25 | 0.957 | 0.913 | 0.884 | 0.952 | 0.072 |
| 2026-09-07 22:28 | Change 4 sweep: 300/50, hybrid_rerank | hybrid_rerank | 3 | 300/50 | 0.957 | 0.913 | 0.884 | 0.965 | 0.100 |
| 2026-09-07 22:31 | Change 4 sweep: 500/0, hybrid_rerank | hybrid_rerank | 3 | 500/0 | 0.913 | 0.870 | 0.790 | 0.930 | 0.043 |
| 2026-09-07 22:33 | Change 4 sweep: 500/25, hybrid_rerank | hybrid_rerank | 3 | 500/25 | 1.000 | 0.957 | 0.848 | 0.945 | 0.043 |
| 2026-09-07 22:36 | Change 4 sweep: 500/50, hybrid_rerank | hybrid_rerank | 3 | 500/50 | 1.000 | 0.957 | 0.877 | 0.963 | 0.042 |
| 2026-09-07 22:39 | post-sweep restore: back to default 300/50, hybrid_rerank | hybrid_rerank | 3 | 300/50 | 0.957 | 0.913 | 0.884 | 0.965 | 0.100 |
