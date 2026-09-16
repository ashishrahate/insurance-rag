# Colab / GPU testing plan

**Status: planning only, not yet executed.** This doc exists so a future
session can actually run this without re-deriving the approach. Written for
someone with zero prior Colab experience — every step is spelled out, and
every open decision is flagged as open rather than silently assumed.

## Why

The recorded Phase 5 baseline (`data/eval/answer_eval_results.md`) uses
`llama3.2:3b` on CPU — deliberately small because this machine has no
Ollama-usable GPU (Intel Arc iGPU, unsupported; see
`Challenges and Learnings.md` #2). Faithfulness fails its gate (0.518-0.52x
vs. 0.85). Before spending on the OpenAI API to see if a stronger model
fixes that, this plan tests a bigger **open** model on a **free** GPU —
`llama3.1:8b` (the roadmap's original Phase 1 model, before the CPU-driven
downgrade) on Colab's T4. `RUN_ENV` tagging and `src/observability/report.py`
already exist specifically to compare runs like this (see `README.md`'s
"Comparing hardware" section) — this plan uses that machinery, not new
tooling.

## 1. Runtime

Open a Colab notebook, then **Runtime → Change runtime type → T4 GPU**
before running anything. Free tier gives one T4 (16GB) — enough headroom for
`llama3.1:8b` (≈4.9GB at Q4) with room to spare.

## 2. Getting the code there — open decision

Two options, not resolved here — pick when actually running this:

- **(a) Push to a GitHub remote, `git clone` in a Colab cell.** Cleanest if
  this comparison gets re-run more than once. This repo currently has no
  remote configured (commits are local-only, per this project's own
  convention of manual user commits) — would need `git remote add` +
  `git push` first, a deliberate step to take consciously, not a side effect
  of this plan.
- **(b) Zip and upload, or mount Google Drive.** Simpler for a genuine
  one-off run; no GitHub involvement.

## 3. Ollama + the bigger model

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull llama3.1:8b
ollama pull nomic-embed-text   # only needed if EMBED_PROVIDER stays ollama (it should — see below)
```

Ollama auto-detects the CUDA GPU; no extra flags needed. Verify with
`ollama ps` after a first call — it should show GPU%, not `100% CPU` like
the recorded CPU baseline.

## 4. Qdrant — the one real snag

Colab notebooks can't run Docker, so `docker-compose.yml` doesn't apply
here. Three options, recommending the first for a first attempt:

### Option 1 (recommended first attempt): embedded/local Qdrant, no server

`qdrant-client` supports an embedded mode (`QdrantClient(path=...)`,
RocksDB-backed, in-process, no server to run at all). No signup, no
tunnel, no dependency on the local machine staying up during the Colab
session — fully self-contained inside the Colab VM.

**One small code change needed**, not yet made (deliberately deferred to
the execution session, per this plan's "planning only" scope):
`get_client()` in `src/retrieval/search.py` currently only builds a
host/port client:

```python
@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, check_compatibility=False)
```

Add an optional local-path branch, gated on a new setting so today's
behavior is unchanged by default:

```python
# config/settings.py
QDRANT_LOCAL_PATH = os.getenv("QDRANT_LOCAL_PATH")  # e.g. "/content/qdrant_data" in Colab

# search.py
@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    if QDRANT_LOCAL_PATH:
        return QdrantClient(path=QDRANT_LOCAL_PATH)
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, check_compatibility=False)
```

Then in Colab: run the full ingestion pipeline fresh
(`bootstrap_collection.py` → `scrape_ca_bulletins.py` → `parse_ca_bulletins.py`
→ `chunk_and_index.py`, same commands as any fresh local setup) against this
local-path collection. Cheap: the recorded baseline shows ~2.6-3.4 chunks/s,
so even the grown 129-chunk corpus re-embeds in well under a minute.
Downside: nothing persists across Colab sessions unless you mount Drive and
point `QDRANT_LOCAL_PATH` at a Drive path.

### Option 2: tunnel the existing local Docker Qdrant

ngrok or Cloudflare Tunnel exposes `localhost:6333` publicly; point Colab's
`QDRANT_HOST`/`QDRANT_PORT` at the tunnel's address. Zero code changes
(still host/port under the hood). Downside: the local machine must stay up
and network-reachable for the whole Colab session — more moving parts than
option 1 for a first attempt.

### Option 3: Qdrant Cloud free tier

Fully hosted, no tunnel, no local-machine dependency. Downside: needs a
signup, and `get_client()` would need `url=`/`api_key=`/HTTPS support added
(not just `host`/`port`) — a slightly bigger code change than option 1.

## 5. Judge service in Colab

Same as running it locally — no Colab-specific change, since everything it
depends on (Ollama) is already local to that VM:

```bash
uvicorn src.judge_service.main:app --port 8100 &
```

If judging with the bigger model too (not required — you can keep judging
on `llama3.2:3b` for continuity with the existing baseline, or switch to
`llama3.1:8b` to also test whether a bigger *judge* changes anything), set
`JUDGE_LLM_MODEL=llama3.1:8b` before starting it.

## 6. Config to set

All already env-var-driven in `config/settings.py` — no code changes beyond
the `get_client()` branch above:

```bash
export RUN_ENV=colab-t4
export LLM_MODEL=llama3.1:8b
export QDRANT_LOCAL_PATH=/content/qdrant_data   # if option 1
# EMBED_PROVIDER left at its default ("ollama") -- only LLM_MODEL changes,
# so this is a pure generation-quality comparison, consistent with the
# EMBED_PROVIDER/LLM_PROVIDER decoupling already in place (see
# rag-project-status memory). No re-embedding-cost concern here since
# nothing leaves the Colab VM.
```

## 7. Run it

```bash
python -m src.ingestion.bootstrap_collection   # creates the collection + payload indexes fresh
python -m src.ingestion.scrape_ca_bulletins    # or copy data/raw + data/processed over instead
python -m src.ingestion.parse_ca_bulletins
python -m src.ingestion.chunk_and_index --naive
python -m src.evaluation.retrieval_eval --retriever hybrid_rerank --label "colab-t4, llama3.1:8b"
python -m src.evaluation.run_full_eval --label "colab-t4, llama3.1:8b"
```

## 8. Getting results back for comparison

`logs/runs.jsonl` and `data/eval/*.md` live in the ephemeral Colab VM —
download them (or write straight to a mounted Drive path) before the
session ends. Then, back on the local machine:

```bash
python -m src.observability.report --file colab_runs.jsonl --group env
```

exactly the workflow `README.md`'s existing "Comparing hardware" section
already documents. Compare the new `answer_eval_results.md` row against the
local `llama3.2:3b` baseline row directly — same file format, same columns.

## Open decisions, explicitly not resolved by this doc

- Code-transfer method (§2) — GitHub remote vs. zip/Drive upload.
- Whether to also change `JUDGE_LLM_MODEL` for this run, or keep the judge
  on `llama3.2:3b` for a cleaner single-variable comparison (generation
  model only) — recommend the latter unless there's a specific reason to
  change both at once.
- Whether to persist the Colab Qdrant data via Drive, or treat every Colab
  run as fully ephemeral (simplest, and fine for a one-off comparison).
