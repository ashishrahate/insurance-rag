# Scaling & architecture notes

This project is built at toy scale on purpose (18 CA bulletins, 53 chunks) so
the RAG *logic* (chunking, fusion, reranking, guardrails) can be learned and
measured cheaply. Several implementation choices are toy-scale shortcuts that
would not survive contact with a real corpus (thousands of documents,
millions of chunks, concurrent users).

This file is where that gap gets written down: what we actually built here,
what changes at production scale, and why. **Update it whenever a scale
shortcut is taken knowingly** — not as a TODO list to action now, but as a
record of the decision and its future replacement, so it doesn't have to be
re-derived later.

Format per entry: what we did · why it's fine at this scale · what breaks ·
what replaces it · **the senior system engineering decision** — not just the
menu of options, but the specific call a senior engineer would make, the
concrete parameters/tools, and the reasoning that resolves the tradeoff one
way rather than leaving it open.

---

## 1. BM25 as an in-memory `rank_bm25.BM25Okapi` index

**What we did:** `src/evaluation/retrievers.py::_bm25_corpus()` pulls every
chunk from Qdrant via `scroll`, tokenizes with `.lower().split()`, and builds
a `BM25Okapi` index that lives entirely in process memory. Cached with
`@lru_cache(maxsize=1)` so it's built once per process.

**Fine at this scale because:** 53 chunks fits trivially in RAM; `get_scores()`
linearly scanning 53 documents per query costs microseconds.

**Breaks at scale because:** `BM25Okapi` has no inverted index — every query
scans the *entire* corpus, and the whole tokenized corpus must fit in one
process's memory. At millions of chunks this is both too slow (linear scan
per query) and too large (RAM).

**What replaces it:**
- **Qdrant sparse vectors** — least new infrastructure, since Qdrant is
  already the vector store here. Same client, same collection.
- **Elasticsearch / OpenSearch** — BM25 is their native default scorer, built
  for this scale with real inverted indexes and sharding.
- Either way, the corpus becomes a persistent, queryable *service*, not
  something rebuilt into RAM per process start.

**What stays the same:** the `reciprocal_rank_fusion()` function and the
`Retriever` interface. RRF only ever sees two ranked ID lists — it is
indifferent to how each list was produced. Swapping the backend means editing
`_bm25_ranked_chunk_ids()`, not the fusion or eval logic.

**Senior system engineering decision:** ship Qdrant sparse vectors, not a
second Elasticsearch cluster. The deciding factor isn't feature richness —
Elasticsearch's analyzer/highlighting/faceting are genuinely better — it's
**operational surface area**: a second stateful search system means a second
thing to deploy, monitor, back up, and keep consistent with the dense index
(a doc updated in one but not the other is a correctness bug, not just an
inconvenience). Sparse vectors in the same Qdrant point mean one write path,
one consistency model, one system to page on at 3am. Only revisit this if a
real requirement shows up that Qdrant's sparse vectors can't serve — rich
full-text UX (highlighting, typo-tolerant fuzzy match) — not preemptively.

---

## 2. Tokenization: `.lower().split()`

**What we did:** whitespace split + lowercase, no stemming, no stopword
removal, no punctuation handling.

**Fine at this scale because:** the eval set is hand-written English prose
over a small, homogeneous corpus (CA DOI bulletins) — crude tokenization
still produces reasonable term overlap.

**Breaks at scale because:** no stemming means "assessment" and "assessments"
are different tokens; no stopword handling wastes index space and biases
scores toward documents that happen to repeat common words; punctuation
artifacts (footnote digits, page numbers) pollute the vocabulary.

**What replaces it:** a real analyzer pipeline — tokenize → lowercase →
stopword removal → stemming/lemmatization. Elasticsearch ships this built-in
per-field; Qdrant sparse vectors typically pair with a proper analyzer or a
learned sparse model (e.g. SPLADE) instead of raw term matching.

**Senior system engineering decision:** don't hand-roll an analyzer, and
don't reach for a heavy NLP stack (spaCy pipelines, custom stemmers) either —
use whatever the chosen search backend ships as its default English analyzer
(Elasticsearch's `standard` + `snowball` filter, or a SPLADE-style learned
sparse model if going the Qdrant route). The reasoning: tokenization quality
here is a solved, commoditized problem — time is better spent on the things
that are actually specific to this corpus (near-duplicate disambiguation,
the refusal threshold) than reimplementing stemming. Only build something
custom if the domain vocabulary demands it (e.g. insurance-specific
abbreviations a generic analyzer mangles) — measure that before investing.

---

## 3. Caching the corpus vs. caching queries

**What we did:** `lru_cache(maxsize=1)` caches the *entire tokenized corpus*
in the process, for the process's lifetime. Same pattern as `get_client()`
in `src/retrieval/search.py`.

**Fine at this scale because:** the corpus is small and the eval harness is a
short-lived, single-purpose process — caching the whole thing is strictly
cheaper than refetching it 28 times per eval run.

**Breaks at scale because:** you can't hold a multi-million-chunk corpus in
one process's memory, and a long-lived server process (the future FastAPI
`/query` endpoint) would serve a *stale* corpus after any re-index, since
`lru_cache` never invalidates itself.

**What replaces it:** the index moves out of the caching layer entirely — it
becomes an external, persistent, independently-updatable service (§1). What
*is* still worth caching at scale is query-level, not corpus-level: an
**exact-query cache** (already on the roadmap for Phase 3) — cache the
*answer* to a previously-seen question, not the whole searchable corpus.

**Senior system engineering decision:** use Redis (or an equivalent managed
cache) for the exact-query cache, keyed on a hash of the normalized question
+ state filter, with a TTL rather than manual invalidation as the primary
defense against staleness — and additionally flush/namespace-bump the cache
on every Blue/Green alias swap (§8) so a re-index can never serve an answer
built from the old index. TTL-first, event-driven-second: relying solely on
"remember to invalidate on deploy" is exactly the kind of manual step that
gets missed under deadline pressure; a TTL bounds the blast radius even if
the invalidation hook is ever skipped.

---

## 4. Retriever registry pattern (`RETRIEVERS` dict)

**What we did:** `RETRIEVERS: dict[str, Retriever]` maps a string name to a
callable of a fixed signature (`(query, k) -> [(doc_id, score), ...]`), so
`retrieval_eval.py` looks up a strategy by name instead of importing it
directly.

**Holds at scale, unchanged.** This is a decoupling pattern, not a toy-scale
shortcut — swapping `dense` for a production ANN index, or `bm25` for
Elasticsearch, only ever means changing what a registered function does
internally, never the eval harness or the registry itself. Kept in this file
as an explicit note of "this one doesn't need revisiting."

**Senior system engineering decision:** keep the pattern, but graduate the
*selection* mechanism when this becomes a live service — a hardcoded
`RETRIEVERS[name]` lookup in code is fine for an eval CLI, but a production
`/query` endpoint should select the retrieval strategy via a **config value
or feature flag** (env var, remote config service), not a code change. That's
what makes an A/B test between `hybrid` and `hybrid_rerank` on live traffic a
config flip instead of a deploy. This is the standard Strategy pattern; the
only thing that changes at scale is who/what chooses the key at runtime.

---

## 5. Vector search: exact brute force, not ANN

**What we did:** nothing deliberate — it falls out of scale. `search.py`
does a single `query_points` cosine search over the whole collection.

**Fine at this scale because:** Qdrant only builds an HNSW (approximate
nearest-neighbor) index once a collection passes `indexing_threshold`
(default 10,000 vectors). At 53 points, `indexed_vectors_count = 0` — every
search is exact brute-force scoring, not approximate.

**Breaks at scale because:** past the threshold, Qdrant switches to HNSW
automatically, and HNSW is *approximate* — it trades a small, tunable amount
of recall for speed (controlled by `ef`/`m` params). This is a real behavior
change, not just a performance one.

**What replaces it:** nothing to *build* — Qdrant does this itself. The thing
to remember is that **every Hit@K/Recall@K/MRR number measured on this eval
set assumes exact search.** At production scale those numbers would need
re-validating against the approximate index, and `ef_search` becomes a real
tuning knob (recall vs. latency) that doesn't exist yet in this project.

**Senior system engineering decision:** treat recall as an explicit SLA, not
a side effect of default settings. Concretely: pick a target (e.g. "Recall@3
must stay within 1pt of exact search," derived from the eval harness already
built here), tune `ef_search` up from Qdrant's default until the eval set
meets that bar, and wire the eval harness into whatever re-index pipeline
exists (§8) so an HNSW parameter change or a corpus-size crossing the
indexing threshold gets caught by CI, not discovered as a silent quality
regression in production. The eval harness this project already has is the
reusable asset here — the decision is to keep running it after every
infra-level change, not just after retrieval-logic changes.

---

## 6. Embedding calls: one synchronous `ollama.embed()` per batch

**What we did:** `embed_text()` / `embed_batch()` in `src/ingestion/embed.py`
make a single blocking call to local Ollama, no retry, no rate limiting, no
size cap on the batch.

**Fine at this scale because:** 53 chunks embed in one call in ~20s locally;
there's nothing to retry against (no network, no third-party rate limits).

**Breaks at scale because:** a hosted embedding API (OpenAI, etc. — the
Phase 5 provider swap) has request size limits, rate limits, and transient
failures a single unguarded call doesn't handle. Millions of chunks also
can't be embedded in one call regardless of provider.

**What replaces it:** batched requests capped at the provider's limit,
retry-with-backoff on transient errors, and parallel workers with a
concurrency cap (bounded by the provider's rate limit) instead of one
sequential loop over all documents. This is exactly the seam the
`OllamaProvider`/`OpenAIProvider` abstraction (already planned, see
`CLAUDE.md`) is meant to hide behind a stable interface.

**Senior system engineering decision:** put retry/backoff and batching
*inside* the provider abstraction, not at every call site — `tenacity` (or
equivalent) wrapping the provider's HTTP call, exponential backoff capped at
a few retries, batch size derived from the provider's documented max (not
guessed), and a bounded worker pool sized just under the provider's
requests-per-minute limit. One more call worth making explicitly: **dedupe
by content hash before embedding** — if the same chunk text is ever
re-submitted (a re-index that didn't actually change that chunk), skip the
paid API call entirely. At scale, embedding cost is real recurring spend;
this is a near-free optimization once a content-hash → vector cache exists.

---

## 7. Run log: local append-only JSONL file

**What we did:** `src/observability/logger.py::log_run()` appends one JSON
line per operation to a single local file, `logs/runs.jsonl`.

**Fine at this scale because:** one developer, one process at a time — no
concurrent writers, and `python -m src.observability.report` can just read
the whole file into memory.

**Breaks at scale because:** concurrent processes/requests (the Phase 3
FastAPI service, under real traffic) appending to the same local file risk
interleaved/corrupted writes, and "grep a local file" stops being a viable
way to query observability data once there's meaningful volume.

**What replaces it:** structured logs shipped to a real observability
backend (e.g. OpenTelemetry → a log/trace store), not a bigger local file.
The `correlation_id` already threaded through every stage (`new_correlation_id()`)
is exactly the piece that carries forward unchanged — request tracing across
distributed services depends on it existing already.

**Senior system engineering decision:** the moment the FastAPI service (Phase
3) exists, logs go to **stdout as structured JSON**, collected by whatever
the deploy platform provides (a sidecar, a log-shipping agent) into a real
backend — never back to a self-managed local file. Retrofit OpenTelemetry at
that point rather than earlier: instrumenting a CLI script with a full
tracing SDK before there's a service topology to trace across is effort
spent before it pays off. Rename `correlation_id` to `trace_id` at that point
too, so it lines up with whatever tracing vocabulary the chosen backend uses
— a naming migration is free now and mildly annoying to do retroactively
across every log call site later.

---

## 8. Re-indexing: full delete-then-upsert, single sequential process

**What we did:** `chunk_and_index.py` loops over all docs in one process,
and for each doc deletes all its existing points before upserting the new
ones (`delete_doc_points` → `upsert`).

**Fine at this scale because:** 18 docs / 53 chunks re-index in ~20s; a
delete-then-upsert per doc is simple and correct, and there's no live
traffic to disrupt while it runs.

**Breaks at scale because:** re-indexing millions of chunks sequentially in
one process is slow, and doing it against the *live* collection means
queries during re-index can see a half-updated index (some docs deleted,
not yet re-upserted).

**What replaces it:** nothing new to invent — the roadmap already has the
answer: **Blue/Green collection staging** (`insurance_ca_v2` built fully
offline, then the `insurance_ca_live` alias flipped atomically once eval
gates pass). Re-indexing becomes "build a new collection, swap an alias,"
never in-place mutation of what's serving traffic. Parallelizing the
embed/upsert work itself (batched, concurrent) is the secondary win once
that's in place.

**Senior system engineering decision:** make the Blue/Green swap a **gated
pipeline step, not a manual command a person runs**. Concretely: build
`insurance_ca_v2` fully offline → automatically run the Phase 2 eval harness
against it → require it to meet or beat the current live collection's
numbers (Hit@K/Recall@K/MRR, not just "it ran without errors") → only then
flip the `insurance_ca_live` alias, and auto-rollback the alias if a
post-swap health check regresses. The eval harness stops being a
developer-run script at that point and becomes a **release gate** — this is
the same shift as unit tests going from "run before I commit" to "CI blocks
the merge." It's the single highest-leverage change in this whole document
because it converts every other scale decision (ANN tuning, embedding
provider swap, chunking changes) into something that can't ship a silent
quality regression.

---

## 9. Local flat files as the source of truth (`data/raw/`, `data/processed/`)

**What we did:** scraped PDFs and parsed JSON/text live as plain files on
local disk, tracked via hand-rolled `manifest.json` envelopes.

**Fine at this scale because:** 18 files, one machine, one developer — a
JSON manifest is trivially readable and diffable.

**Breaks at scale because:** local disk isn't shared across ingestion
workers, isn't durable the way a managed store is, and a hand-rolled
manifest doesn't scale as a query surface (e.g. "which docs are stuck in
`empty_or_scanned` status" becomes a real query, not a `grep`).

**What replaces it:** an object store (S3/GCS) for the raw/processed
artifacts, and a real metadata store (even just a Postgres table) instead of
a JSON envelope, once ingestion needs to run from more than one place.

**Senior system engineering decision:** move the metadata (the manifest) to
Postgres before worrying about the object store — that's the higher-value,
lower-effort half of this change. A `documents` table with a `status` column
turns "which docs are stuck in `empty_or_scanned`" from a `grep`/manual JSON
read into `SELECT * WHERE status = 'empty_or_scanned'`, and makes each
ingestion stage an idempotent, resumable job keyed on status transitions
(`scraped → parsed → chunked → indexed`) instead of a script that has to
re-derive "what's already done" by re-reading manifests. The object store
swap (S3/GCS) is comparatively mechanical — file writes become client calls
— so it's the lower-priority half of this decision, done when local disk
actually becomes the bottleneck, not preemptively.

---

## 10. Scraping: sequential, one source, full re-crawl every run

**What we did:** `scrape_ca_bulletins.py` fetches one listing page, downloads
PDFs sequentially with a fixed `REQUEST_DELAY_SEC` sleep between them, and
re-evaluates the whole listing on every run (cache hits skip the download,
but the listing is always re-fetched and re-parsed in full).

**Fine at this scale because:** one source (CA DOI), 18 documents — a
sequential polite crawl finishes in well under a minute.

**Breaks at scale because:** more states/sources (Phase 6) means many hosts
to crawl, and a sequential single-threaded loop doesn't parallelize across
independent sources; re-evaluating the full listing every run also doesn't
scale once a source's catalog is large.

**What replaces it:** concurrent crawling with a **per-host** rate limit
(not a single global sleep), and incremental/delta scraping (only fetch what
changed since the last run, e.g. via listing timestamps or ETags) instead of
re-walking the entire catalog each time.

**Senior system engineering decision:** structure ingestion as one worker
*type* per source (`CADOIWorker`, `NYDFSWorker`, ...) implementing a shared
interface (`list_documents() -> [...]`, `fetch(doc) -> bytes`), each owning
its own per-host rate limit and its own "last successful crawl" checkpoint —
orchestrated by a scheduler (cron is enough at this scale; Airflow/Dagster
once there are enough sources that failure handling and backfills need real
tooling) rather than one script per state. This mirrors the `RETRIEVERS`
registry decision (§4) on purpose: sources are pluggable strategies behind a
stable interface, so Phase 6's second state is a new worker class, not a
rewrite of `scrape_ca_bulletins.py`'s internals.

---

## 11. Prompt injection defense

**What we did:** `src/generation/prompt.py` frames retrieved passages and
the user's question as explicitly-delimited, explicitly-untrusted data —
each wrapped in structural boundary tokens (`<<<PASSAGE_DATA_START/END>>>`,
`<<<USER_QUESTION_START/END>>>`), with the system prompt instructing the
model to treat any instruction-like text found inside those boundaries as
ordinary quoted content, never a command. Those same tokens are stripped
(replaced with `[filtered]`) out of both the question and retrieved chunk
content before interpolation, so neither a user nor a compromised source
document can forge a fake boundary and smuggle content past the framing.
`MAX_QUESTION_CHARS=500` bounds worst-case prompt size, enforced twice
(`QueryRequest`'s Pydantic `max_length` for a clean API 422; a truncate in
`generate.py::answer_question()` for the CLI, which has no schema layer).

**Fine at this scale because:** the corpus is 18 official CA DOI PDFs from
one hardcoded, `%PDF`-magic-byte-validated domain — indirect injection via a
compromised source document is a low-*probability* risk today, even though
the architecture doesn't structurally prevent it. No tool-use/execution
capability exists anywhere in this app, so the worst case of a successful
injection is a wrong *displayed* answer, not a system compromise.

**Breaks at scale because:** more sources (Phase 6, multi-state) means more
domains to trust, and any user-facing deployment (vs. a single learning
project) means direct injection attempts from real adversarial users, not
just a hypothetical. Delimiter/framing defenses like this one are also not
airtight — a sufficiently capable adversarial prompt can still sometimes get
a model to ignore framing instructions, especially a small local model like
`llama3.2:3b`, which has weaker instruction-following robustness than a
frontier model. This is a genuinely unsolved problem industry-wide, not
something any single technique fully closes.

**What replaces it:**
- A dedicated **input/output guardrail layer** — not just prompt framing,
  but a real classifier or moderation pass on both the user's question
  (before retrieval) and the model's answer (before it's returned), able to
  flag or block attempts, not just hope the framing holds.
- **Provider-level safety features** where available — e.g. a hosted
  provider's built-in prompt-injection/jailbreak detection (a real
  motivation for the Phase 5 `OpenAIProvider`, beyond just answer quality).
- **Least-privilege by design, kept true as the app grows:** the current
  "no execution capability" mitigation is only a mitigation as long as it
  stays true — the moment this app gains any tool-use/agentic capability
  (e.g. an `/ingest` triggered by natural language, not just an explicit
  endpoint), injection stops being "worst case: wrong answer" and starts
  being a real action-execution risk, and defenses need to escalate with it.

**Senior system engineering decision:** treat this as the *first increment*
of a dedicated **guardrails feature**, not the finished defense — the
delimiter/framing approach here is cheap, real, and worth having, but a
mature system needs guardrails as a first-class, testable layer: an
eval-set-style corpus of known injection *attempts* (mirroring how
`data/eval/ca_eval_set.json` already tests retrieval quality) that gets run
through the pipeline on every change, asserting the framing holds and the
refusal/citation behavior isn't hijacked — the same "measure the change,
don't just ship it" discipline this project has applied to every retrieval
change since Phase 2. Scope that as its own unit of work when it's time
(a natural fit alongside the Phase 5 LLM-as-judge harness, which already
needs an automated "score this generated answer" pass), not folded silently
into whichever phase happens to be active.

---

*(Add new entries above this line as they come up — single Ollama instance
vs. a hosted provider under concurrent load, FastAPI service concurrency,
etc.)*
